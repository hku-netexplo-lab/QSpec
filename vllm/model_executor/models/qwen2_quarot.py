"""Inference-only Qwen2 model compatible with HuggingFace weights."""
import math
from typing import Iterable, List, Optional, Set, Tuple, Union

import quarot
import quarot.transformers

import torch
from torch import nn
from transformers import Qwen2Config
import fast_hadamard_transform
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS


from vllm.model_executor.layers import quarot_nn


from vllm.attention import Attention, AttentionMetadata, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput

from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (AutoWeightsLoader, PPMissingLayer, WeightsMapper,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

# from vllm_custom.model_executor.layers.quantization.utils.fake_quant_utils import ActivationQuantizer
from quarot.functional import hadamard as hadamard_utils

logger = init_logger(__name__)

ALL_LAYERNORM_LAYERS.append(quarot_nn.RMSNorm)

class QuarotQwenConfig(Qwen2Config):
    model_type = "qwen2_quarot"
    

class Qwen2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        fake_quant_config: dict,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.quantizer = quarot_nn.Quantizer()
        # self.gate_up_proj = MergedColumnParallelLinear(
        #     hidden_size,
        #     [intermediate_size] * 2,
        #     bias=False,
        #     quant_config=quant_config,
        #     prefix=f"{prefix}.gate_up_proj",
        # )
        self.gate_up_proj = quarot_nn.Linear4bit(
            in_features=hidden_size,
            out_features=intermediate_size * 2,
            bias=False,
            **kwargs,
        )
        # self.down_proj = RowParallelLinear(
        #     intermediate_size,
        #     hidden_size,
        #     bias=False,
        #     quant_config=quant_config,
        #     prefix=f"{prefix}.down_proj",
        # )
        self.online_hadamard = quarot_nn.OnlineHadamard(self.intermediate_size)
        self.down_proj = quarot_nn.Linear4bit(
                in_features=intermediate_size,
                out_features=hidden_size,
                bias=False,
                **kwargs,
            )
        
            
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

        # fake quantization for activations
        # self.gate_up_quant = ActivationQuantizer(bits=fake_quant_config["a_bits"], sym=not fake_quant_config["a_asym"],
        #                                         lac=False, groupsize=-1, clip_ratio=fake_quant_config["a_clip_ratio"])
        # self.down_quant = ActivationQuantizer(bits=fake_quant_config["a_bits"], sym=not fake_quant_config["a_asym"],
        #                                     lac=False, groupsize=-1, clip_ratio=fake_quant_config["a_clip_ratio"])
        
        # hadamard transformation
        self.down_hadK, self.down_K = hadamard_utils.get_hadK(intermediate_size // 1)
        self.down_had_scale = 1.0 / math.sqrt(intermediate_size)

    def forward(self, x, **kwargs):
        # x = self.gate_up_quant(x)
        # breakpoint()
        gate_up = self.gate_up_proj(x, **kwargs)
        x = self.act_fn(gate_up)
        x = hadamard_utils.matmul_hadU_cuda(x, self.down_hadK, self.down_K, self.down_had_scale)
        # x = self.down_quant(x)
        x = self.quantizer(x, **kwargs)
        x = self.down_proj(x, **kwargs)
        return x


class Qwen2Attention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 max_position: int = 4096 * 32,
                 rope_theta: float = 10000,
                 fake_quant_config: dict = None,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 rope_scaling: Optional[Tuple] = None,
                 prefix: str = "",
                 attn_type: str = AttentionType.DECODER,
                 **kwargs) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        # self.qkv_proj = QKVParallelLinear(
        #     hidden_size,
        #     self.head_dim,
        #     self.total_num_heads,
        #     self.total_num_kv_heads,
        #     bias=True,
        #     quant_config=quant_config,
        #     prefix=f"{prefix}.qkv_proj",
        # )
        
        self.qkv_proj = quarot_nn.Linear4bit(
            in_features=hidden_size,
            out_features=self.q_size + 2 * self.kv_size,
            bias=True,
            **kwargs
        )
        
        # self.o_proj = RowParallelLinear(
        #     self.total_num_heads * self.head_dim,
        #     hidden_size,
        #     bias=False,
        #     quant_config=quant_config,
        #     prefix=f"{prefix}.o_proj",
        # )
        
        self.o_proj = quarot_nn.Linear4bit(
            in_features=self.total_num_heads * self.head_dim,
            out_features=hidden_size,
            bias=False,
            **kwargs,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
        )
 
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn",
                              attn_type=attn_type)

        # fake quantization for activations
        # self.qkv_quant = ActivationQuantizer(bits=fake_quant_config["a_bits"], sym=not fake_quant_config["a_asym"],
        #                                      lac=False, groupsize=-1, clip_ratio=fake_quant_config["a_clip_ratio"])
        # self.o_quant = ActivationQuantizer(bits=fake_quant_config["a_bits"], sym=not fake_quant_config["a_asym"],
        #                                     lac=False, groupsize=-1, clip_ratio=fake_quant_config["a_clip_ratio"])
        # self.k_cache_quant = ActivationQuantizer(bits=fake_quant_config["k_bits"], sym=not fake_quant_config["k_asym"],
        #                                         lac=False, groupsize=fake_quant_config["k_groupsize"], clip_ratio=fake_quant_config["k_clip_ratio"])
        # self.v_cache_quant = ActivationQuantizer(bits=fake_quant_config["v_bits"], sym=not fake_quant_config["v_asym"],
        #                                         lac=False, groupsize=fake_quant_config["v_groupsize"], clip_ratio=fake_quant_config["v_clip_ratio"])
        
        # # hadamard transformation
        self.o_hadK, self.o_K = hadamard_utils.get_hadK(self.num_heads)
        self.o_had_dim = hidden_size // self.num_heads
        self.o_hadk = self.o_hadK.to("cuda")
        self.o_had_scale = 1.0 / math.sqrt(hidden_size // self.o_had_dim)

        self.head_had_scale = 1.0 / math.sqrt(self.head_dim)
        self.o_reshape_dims = (-1, hidden_size // self.o_had_dim, self.o_had_dim)


        # self.o_proj_hadamard = quarot_nn.OnlineHadamard(self.num_heads)
        self.quantizer = quarot_nn.Quantizer()
        
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Optional[dict],
    ) -> torch.Tensor:
        # hidden_states = self.qkv_quant(hidden_states)
        qkv = self.qkv_proj(hidden_states, **kwargs)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        q, k = self.qk_had_trans(q, k)
        # k = self.k_cache_quant(k)
        # v = self.v_cache_quant(v)
        
        # if q.dtype != k.dtype:
        #     breakpoint()
        #     k = k.to(q.dtype)
        # if v.dtype != k.dtype:
        #     breakpoint()
        #     v = v.to(k.dtype)
        # breakpoint()
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        attn_output = self.o_had_trans(attn_output)
        attn_output = self.quantizer(attn_output,**kwargs)
        output = self.o_proj(attn_output, **kwargs)
        return output
    
    def o_had_trans(self, x):
        init_shape = x.shape
        # breakpoint()
        if self.o_K == 1:
            x = fast_hadamard_transform.hadamard_transform(x.reshape(*self.o_reshape_dims).transpose(1, 2),
                                                            scale=self.o_had_scale).transpose(1, 2)
        else:
            # had_k to o_hadK
            x = (self.o_hadK.to(x.device, x.dtype) @ x.reshape(*self.o_reshape_dims)) * self.o_had_scale
        x = x.reshape(init_shape)
        return x
    
    def qk_had_trans(self, q, k):
        init_q_shape, init_k_shape = q.shape, k.shape
        dtype = q.dtype
        q = q.reshape(-1, self.head_dim)
        k = k.reshape(-1, self.head_dim)
        q = fast_hadamard_transform.hadamard_transform( q.float(), scale=self.head_had_scale ).to(dtype)
        k = fast_hadamard_transform.hadamard_transform( k.float(), scale=self.head_had_scale ).to(dtype)
        q = q.reshape(*init_q_shape)
        k = k.reshape(*init_k_shape)
        return q, k


class Qwen2DecoderLayer(nn.Module):

    def __init__(
        self,
        config ,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)

        # By default, Qwen2 uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. Alibaba-NLP/gte-Qwen2-7B-instruct)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            cache_config=cache_config,
            fake_quant_config=config.fake_quant_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            **kwargs,
        )
        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fake_quant_config=config.fake_quant_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            **kwargs,
        )
        # self.input_layernorm = RMSNorm(config.hidden_size,
        #                                eps=config.rms_norm_eps)
        # self.post_attention_layernorm = RMSNorm(config.hidden_size,
        #                                         eps=config.rms_norm_eps)
        self.input_layernorm = quarot_nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = quarot_nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        # residual: Optional[torch.Tensor],
        **kwargs: Optional[dict],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        # breakpoint()
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states, **kwargs)
       
        hidden_states = self.self_attn(positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            **kwargs,
        )
        
        hidden_states = residual + hidden_states.view_as(residual)
        
        #-------
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states,**kwargs)
        hidden_states = self.mlp(hidden_states,**kwargs).view_as(residual)
        hidden_states = residual + hidden_states
        outputs = hidden_states
        
        return outputs


# @support_torch_compile(
#     dynamic_arg_dims={
#         "input_ids": 0,
#         # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
#         # otherwise (seq_len, ).
#         "positions": -1,
#         "intermediate_tensors": 0,
#         "inputs_embeds": 0,
#     })
class Qwen2Model(nn.Module):

    def __init__(self, *, qwen_config, vllm_config, prefix: str = "", **kwargs):
        super().__init__()
        
        self.config = qwen_config
        
        self.vllm_config = vllm_config
        
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        
        # TODO (@robertgshaw2): see if this can be moved out
        # if (cache_config.sliding_window is not None
        #         and hasattr(config, "max_window_layers")):
        #     raise ValueError("Sliding window for some but all layers is not "
        #                      "supported. This model uses sliding window "
        #                      "but `max_window_layers` = {} is less than "
        #                      "`num_hidden_layers` = {}. Please open an issue "
        #                      "to discuss this feature.".format(
        #                          config.max_window_layers,
        #                          config.num_hidden_layers,
        #                      ))

        
        self.quant_config = quant_config
        self.padding_idx = self.config.pad_token_id
        self.vocab_size = self.config.vocab_size

        if get_pp_group().is_first_rank or (self.config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # self.start_layer, self.end_layer, self.layers = make_layers(
        #     config.num_hidden_layers,
        #     lambda prefix: Qwen2DecoderLayer(config=config,
        #                                      cache_config=cache_config,
        #                                      quant_config=quant_config,
        #                                      prefix=prefix),
        #     prefix=f"{prefix}.layers",
        # )
        
        self.layers = nn.ModuleList()
        # Create layers with the specified prefix
        for i in range(self.config.num_hidden_layers):
            layer = Qwen2DecoderLayer(
                config=self.config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.layers.{i}",
                **kwargs,
            )
            self.layers.append(layer)
        self.start_layer = 0
        self.end_layer = self.config.num_hidden_layers

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], self.config.hidden_size))
        if get_pp_group().is_last_rank:
            self.norm = quarot_nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps,fuse=False)
        else:
            self.norm = PPMissingLayer()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Optional[dict],
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i - self.start_layer],
                attn_metadata,
                **kwargs
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen2QuaRotForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(self, *, qwen_config, vllm_config, prefix: str = "", **kwargs):
        super().__init__()
        
        self.config = qwen_config
        self.vllm_config = vllm_config
        # config = vllm_config.model_config.hf_config
        config = qwen_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config
        self.model = Qwen2Model(qwen_config=qwen_config, vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"), **kwargs)

        if get_pp_group().is_last_rank:
            if self.config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(self.config.vocab_size,
                                              self.config.hidden_size,
                                              quant_config=self.quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "lm_head"))
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = get_sampler()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Optional[dict],
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors,
                                   inputs_embeds, **kwargs)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)


class Qwen2EmbeddingModel(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"model.": ""})

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        pooler_config = vllm_config.model_config.pooler_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = Qwen2Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        # TODO: Replace this model class with as_embedding_model(
        # Qwen2ForCausalLM) after changing the default pooling method
        if pooler_config.pooling_type is None:
            logger.warning(
                "This embedding model will default to last-token pooling in "
                "an upcoming version. To avoid breaking changes, you should "
                "pass `--override-pooler-config '{\"pooling_type\": \"MEAN\"}'`"
                " explicitly.")

        self._pooler = Pooler.from_config_with_defaults(
            pooler_config,
            pooling_type=PoolingType.MEAN,
            normalize=True,
            softmax=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, kv_caches, attn_metadata,
                          intermediate_tensors)

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        weights = self.hf_to_vllm_mapper.apply(weights)
        weights = ((name, data) for name, data in weights
                   if not name.startswith("lm_head."))
        self.model.load_weights(weights)
