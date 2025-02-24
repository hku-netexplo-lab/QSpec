import copy
from collections import defaultdict
from functools import cached_property
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import torch
import torch.nn as nn

from vllm.config import ParallelConfig, SpeculativeConfig, VllmConfig
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.logger import init_logger
from vllm.model_executor.layers.rejection_sampler import RejectionSampler
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.layers.spec_decode_base_sampler import (
    SpecDecodeBaseSampler, SpecDecodeStochasticBaseSampler)
from vllm.model_executor.layers.typical_acceptance_sampler import (
    TypicalAcceptanceSampler)
from vllm.platforms import current_platform
from vllm.sequence import (VLLM_INVALID_TOKEN_ID,
                           CompletionSequenceGroupOutput, ExecuteModelRequest,
                           HiddenStates, SequenceGroupMetadata,
                           get_all_seq_ids_and_request_ids)
from vllm.spec_decode.batch_expansion import BatchExpansionTop1Scorer

if current_platform.is_cuda_alike():
    from vllm.spec_decode.draft_model_runner import TP1DraftModelRunner

from vllm.spec_decode.interfaces import (SpeculativeProposals,
                                         SpeculativeScorer, SpeculativeScores)
from vllm.spec_decode.medusa_worker import MedusaWorker
from vllm.spec_decode.metrics import AsyncMetricsCollector
from vllm.spec_decode.mlp_speculator_worker import MLPSpeculatorWorker
from vllm.spec_decode.mqa_scorer import MQAScorer
from vllm.spec_decode.multi_step_worker import MultiStepWorker
from vllm.spec_decode.ngram_worker import NGramWorker
from vllm.spec_decode.proposer_worker_base import ProposerWorkerBase
from vllm.spec_decode.smaller_tp_proposer_worker import SmallerTpProposerWorker
from vllm.spec_decode.target_model_runner import TargetModelRunner
from vllm.spec_decode.util import (Timer, create_logprobs_output,
                                   create_sequence_group_output,
                                   get_all_num_logprobs,
                                   get_sampled_token_logprobs, nvtx_range,
                                   split_batch_by_proposal_len)
from vllm.utils import resolve_obj_by_qualname
from vllm.worker.worker_base import LoraNotSupportedWorkerBase, WorkerBase
from spec_decode_worker import SpecDecodeWorker


logger = init_logger(__name__)


def create_qspec_worker(*args, **kwargs) -> "QSpecDecodeWorker":
    """Helper method that is the entrypoint for Executors which use
    WorkerWrapper. It constructs a SpecDecodeWorker from the speculative config.
    """
    vllm_config: VllmConfig = kwargs.get("vllm_config")
    speculative_config: SpeculativeConfig = vllm_config.speculative_config
    assert speculative_config is not None

    if vllm_config.parallel_config.pipeline_parallel_size > 1:
        raise NotImplementedError("Speculative decoding is currently "
                                  "incompatible with pipeline parallelism")

    draft_worker_kwargs = kwargs.copy()

    kwargs["model_runner_cls"] = TargetModelRunner
    target_worker_config = copy.deepcopy(vllm_config)
    target_worker_config.parallel_config.worker_cls =\
        target_worker_config.parallel_config.sd_worker_cls
    cls = resolve_obj_by_qualname(
        target_worker_config.parallel_config.worker_cls)
    target_worker = cls(*args, **kwargs)
    # Set the disable_logprobs variable in the TargetModelRunner instance
    # as per its value specified in the SpeculativeConfig.
    target_worker.model_runner.disable_logprobs =\
         speculative_config.disable_logprobs

    draft_worker_config = copy.deepcopy(vllm_config)
    draft_worker_config.model_config = speculative_config.draft_model_config
    draft_worker_config.quant_config = VllmConfig._get_quantization_config(
        draft_worker_config.model_config,
        vllm_config.load_config,
    )
    speculative_config.draft_parallel_config.worker_cls =\
        draft_worker_config.parallel_config.sd_worker_cls
    draft_worker_config.parallel_config = speculative_config.draft_parallel_config  # noqa
    # TODO allow draft-model specific load config.

    # Override draft-model specific worker args.
    draft_worker_kwargs.update(
        vllm_config=draft_worker_config,
        ngram_prompt_lookup_max=speculative_config.ngram_prompt_lookup_max,
        ngram_prompt_lookup_min=speculative_config.ngram_prompt_lookup_min,
    )

    qspec_decode_worker = QSpecDecodeWorker.create_worker(
        scorer_worker=target_worker,
        draft_worker_kwargs=draft_worker_kwargs,
        disable_mqa_scorer=speculative_config.speculative_disable_mqa_scorer,
        disable_by_batch_size=speculative_config.
        speculative_disable_by_batch_size,
        draft_token_acceptance_method=speculative_config.
        draft_token_acceptance_method,
        typical_acceptance_sampler_posterior_threshold=speculative_config.
        typical_acceptance_sampler_posterior_threshold,
        typical_acceptance_sampler_posterior_alpha=speculative_config.
        typical_acceptance_sampler_posterior_alpha,
        disable_logprobs=speculative_config.disable_logprobs,
        disable_log_stats=speculative_config.disable_log_stats,
    )

    return qspec_decode_worker



class QSpecDecodeWorker(SpecDecodeWorker):
    
    
    @classmethod
    def create_worker(
        cls,
        scorer_worker: WorkerBase,
        draft_worker_kwargs: Dict[str, Any],
        disable_mqa_scorer: bool,
        disable_by_batch_size: Optional[int],
        draft_token_acceptance_method: str,
        typical_acceptance_sampler_posterior_threshold: float,
        typical_acceptance_sampler_posterior_alpha: float,
        disable_logprobs: bool,
        disable_log_stats: bool,
    ) -> "QSpecDecodeWorker":

        allow_zero_draft_token_step = True
        ngram_prompt_lookup_max = (
            draft_worker_kwargs.pop("ngram_prompt_lookup_max"))
        ngram_prompt_lookup_min = (
            draft_worker_kwargs.pop("ngram_prompt_lookup_min"))
        draft_model_config = draft_worker_kwargs["vllm_config"].model_config
        draft_parallel_config: ParallelConfig = draft_worker_kwargs[
            'vllm_config'].parallel_config
        if ngram_prompt_lookup_max > 0:
            draft_worker_kwargs[
                "device_type"] = scorer_worker.device_config.device.type
            proposer_worker = NGramWorker(**draft_worker_kwargs)
            proposer_worker.set_ngram_window_size(ngram_prompt_lookup_min,
                                                  ngram_prompt_lookup_max)
        else:
            draft_tp = draft_parallel_config.tensor_parallel_size
            target_tp = scorer_worker.parallel_config.tensor_parallel_size

            if draft_model_config.hf_config.model_type == "mlp_speculator":
                proposer_worker = MLPSpeculatorWorker(**draft_worker_kwargs)
            elif draft_model_config.hf_config.model_type == "medusa":
                proposer_worker = MedusaWorker(**draft_worker_kwargs)
            else:
                if draft_tp == 1:
                    if current_platform.is_cuda_alike():
                        draft_worker_kwargs[
                            "model_runner_cls"] = TP1DraftModelRunner
                else:
                    if draft_model_config.hf_config.model_type == "eagle":
                        raise NotImplementedError(
                            "EAGLE does not support TP > 1 yet")

                    allow_zero_draft_token_step = False
                proposer_worker = MultiStepWorker(**draft_worker_kwargs)

            proposer_worker = SmallerTpProposerWorker.maybe_wrap_worker(
                proposer_worker, draft_tp, target_tp)

        logger.info("Configuring SpecDecodeWorker with proposer=%s",
                    type(proposer_worker))

        spec_decode_sampler: SpecDecodeBaseSampler = None
        if draft_token_acceptance_method == "rejection_sampler":
            spec_decode_sampler = RejectionSampler()
        elif draft_token_acceptance_method == "typical_acceptance_sampler":
            spec_decode_sampler = TypicalAcceptanceSampler(
                posterior_threshold=\
                    typical_acceptance_sampler_posterior_threshold,
                posterior_alpha=typical_acceptance_sampler_posterior_alpha,
            )
        logger.info(
            "[Speculative Decoding] Configuring"
            " SpecDecodeWorker with sampler=%s", type(spec_decode_sampler))

        if not disable_mqa_scorer:
            if scorer_worker.model_runner.attn_backend.get_name(
            ) != "FLASH_ATTN":
                disable_mqa_scorer = True
                logger.info(
                    "[Speculative Decoding] Disabling MQA scorer as the "
                    "MQA is only available with flash attn backend.")

            if draft_model_config and \
                draft_model_config.max_model_len < \
                    scorer_worker.model_config.max_model_len:
                disable_mqa_scorer = True
                logger.info(
                    "[Speculative Decoding] Disabling MQA scorer as the "
                    "draft model max_model_len is smaller than the target "
                    "model max_model_len.")

            if not scorer_worker.model_runner.model_config.enforce_eager:
                disable_mqa_scorer = True
                logger.info(
                    "[Speculative Decoding] Disabling MQA scorer as the "
                    "target model is not running in eager mode.")

        return QSpecDecodeWorker(
            proposer_worker,
            scorer_worker,
            disable_mqa_scorer=disable_mqa_scorer,
            disable_logprobs=disable_logprobs,
            disable_log_stats=disable_log_stats,
            disable_by_batch_size=disable_by_batch_size,
            spec_decode_sampler=spec_decode_sampler,
            allow_zero_draft_token_step=allow_zero_draft_token_step)
    