import functools
import quarot
import quarot.transformers
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, \
 LlamaForCausalLM, apply_rotary_pos_emb, LlamaMLP, LlamaDecoderLayer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from typing import Optional, Tuple, Type
from transformers import Cache,DynamicCache
from torch import nn

from vllm.model_executor.layers import quarot_nn

from vllm.config import VllmConfig,CacheConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.attention import Attention, AttentionMetadata
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from .utils import (AutoWeightsLoader, PPMissingLayer, extract_layer_index,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope

from typing import Dict, List, Optional,Union


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None,:].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)

ALL_LAYERNORM_LAYERS.append(quarot_nn.RMSNorm)

class QuarotLlamaConfig(LlamaConfig):
    model_type = "llama_quarot"

class QuarotLlamaAttention(LlamaAttention):
    def __init__(self,
                 config,
                layer_idx,
                num_heads: int,
                num_kv_heads: int,
                rope_theta: int,
                rope_scaling: Optional[Dict[str, int]],
                max_position_embeddings: int,
                quant_config: Optional[QuantizationConfig],
                bias: bool,
                bias_o_proj: bool,
                cache_config: Optional[CacheConfig],
                prefix: str,
            *args, **kwargs):

        super().__init__(config, layer_idx)
        # self.quantizer = quarot_nn.Quantizer()
        self.q_proj = quarot_nn.Linear4bit.from_float(self.q_proj)
        self.k_proj = quarot_nn.Linear4bit.from_float(self.k_proj)
        self.v_proj = quarot_nn.Linear4bit.from_float(self.v_proj)
        self.qkv_proj = None
        self.o_proj_hadamard = quarot_nn.OnlineHadamard(num_heads)
        self.o_proj = torch.nn.Sequential(
            quarot_nn.Quantizer(),
            quarot_nn.Linear4bit.from_float(self.o_proj)
        )
        self.num_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = config.hidden_size // num_heads
        self.total_num_heads = num_heads 
        self.scaling = self.head_dim ** -0.5
        # self.config = getattr(self, "config", None)
        self.config = config
        self.qkv_fused= False
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.rope_scaling = rope_scaling
        self.hidden_size = config.hidden_size
        self.is_neox_style = True
        self.bsz = 1
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=self.rope_scaling,
            is_neox_style=self.is_neox_style,
        )
        if hasattr(self.config, "interleaved_sliding_window"):
            interleaved_sliding_window = self.config.interleaved_sliding_window
            if isinstance(interleaved_sliding_window, int):
                sliding_window = interleaved_sliding_window
            elif isinstance(interleaved_sliding_window, list):
                sw_idx = self.layer_idx % len(interleaved_sliding_window)
                sliding_window = interleaved_sliding_window[sw_idx]
            else:
                raise ValueError(
                    f"{type(interleaved_sliding_window)} is not supported.")
        else:
            sliding_window = None
            
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )
            
        
        
        
    def fuse_qkv(self):
        if self.qkv_fused:
            return self
        self.qkv_fused = True
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_key_value_heads * self.head_dim
        self.qkv_proj = quarot_nn.Linear4bit(self.q_proj.in_features,q_size+ 2*kv_size,bias=False)
        self.qkv_proj.weight.data[:q_size] = self.q_proj.weight.data
        self.qkv_proj.weight.data[q_size:q_size+kv_size] = self.k_proj.weight.data
        self.qkv_proj.weight.data[q_size+kv_size:] = self.v_proj.weight.data
        self.qkv_proj.weight.data = self.qkv_proj.weight.data.to(torch.int8).cuda()
        self.qkv_proj.weight_scales = self.qkv_proj.weight_scales.view(-1)
        self.qkv_proj.weight_scales.data[:q_size] = self.q_proj.weight_scales.data.view(-1)
        self.qkv_proj.weight_scales.data[q_size:q_size+kv_size] = self.k_proj.weight_scales.data.view(-1)
        self.qkv_proj.weight_scales.data[q_size+kv_size:] = self.v_proj.weight_scales.data.view(-1)
        self.qkv_proj.weight_scales = self.qkv_proj.weight_scales.cuda()
        # self.qkv_proj.output_buffer = torch.empty(self.bsz, self.q_proj.out_features * 3, dtype=torch.float16, device="cuda")
        del self.q_proj
        del self.k_proj
        del self.v_proj
        
        

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        # hidden_states = self.quantizer(hidden_states)
        if self.qkv_fused:
            # TODO need to modify the code to adapt Llama3+
            qkv_states = self.qkv_proj(hidden_states)
            q_len  = qkv_states.shape[0] // self.bsz
            bsz = self.bsz
            q_size = self.num_heads * self.head_dim
            kv_size = self.num_key_value_heads * self.head_dim
            query_states, key_states, value_states = qkv_states.split([q_size,kv_size,kv_size],dim=-1)
             
        else:    
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            
            q_len  = query_states.shape[0] // self.bsz
            bsz = self.bsz
            
            # Flash attention requires the input to have the shape
            # batch_size x seq_length x head_dim x hidden_dim
            # therefore we just need to keep the original shape
        # query_states = query_states.reshape(q_len, self.num_heads, self.head_dim).contiguous() ##.transpose(1, 2)
        # key_states = key_states.reshape(q_len, self.num_key_value_heads, self.head_dim).contiguous()  ##.transpose(1, 2)
        # value_states = value_states.reshape(q_len, self.num_key_value_heads, self.head_dim).contiguous()  ##.transpose(1, 2)

        # breakpoint()

        query_states, key_states = self.rotary_emb(positions, query_states, key_states)
        
        # for spec-decode as draft model
        query_states = query_states.to(torch.bfloat16)
        key_states = key_states.to(torch.bfloat16)
        value_states = value_states.to(torch.bfloat16)
        # end———————————————————————————— 
        attn_output = self.attn(query_states, key_states, value_states, kv_cache, attn_metadata).view(-1, self.num_heads, self.head_dim)

        # for spec-decode as draft model
        attn_output = attn_output.to(torch.float16)
        # end————————————————————————————
        
        # breakpoint()
        attn_output = self.o_proj_hadamard(attn_output.transpose(-1, -2)).transpose(-1, -2)
        # breakpoint()
        attn_output = attn_output.reshape(bsz * q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output



class QuarotLlamaMLP(LlamaMLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = quarot_nn.Quantizer()
        self.up_proj = quarot_nn.Linear4bit.from_float(self.up_proj)
        self.gate_proj = quarot_nn.Linear4bit.from_float(self.gate_proj)
        self.down_proj = torch.nn.Sequential(
            quarot_nn.OnlineHadamard(self.intermediate_size),
            quarot_nn.Quantizer(),
            quarot_nn.Linear4bit.from_float(self.down_proj)
        )

    def forward(self, x):
        # bsz, seq_len, _ = x.size()
        # x = self.quantizer(x)
        return super().forward(x) #.view(bsz, seq_len, -1)


class QuarotDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx,
                cache_config: Optional[CacheConfig] = None,
                quant_config: Optional[QuantizationConfig] = None,
                prefix: str = "",
        ):
        super().__init__(config,layer_idx)
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
            
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
    
        self.layer_idx = layer_idx
        
        self.self_attn = QuarotLlamaAttention(config=config, layer_idx=layer_idx,
                                            num_heads=config.num_attention_heads,
                                            num_kv_heads=getattr(config, "num_key_value_heads",config.num_attention_heads),
                                            rope_theta=rope_theta,
                                            rope_scaling=rope_scaling,
                                            max_position_embeddings=max_position_embeddings,
                                            quant_config=quant_config,
                                            bias=attention_bias,
                                            bias_o_proj=False,
                                            cache_config=cache_config,
                                            prefix=f"{prefix}.self_attn")
        
        self.mlp = QuarotLlamaMLP(config=config)
        
        self.input_layernorm = quarot_nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = quarot_nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        
        
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:


        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)    
            
        hidden_states = self.self_attn(positions=positions,
                                hidden_states=hidden_states,
                                kv_cache=kv_cache,
                                attn_metadata=attn_metadata)
        
        hidden_states = residual + hidden_states.view_as(residual)
        # Fully Connected
        residual = hidden_states
        
        hidden_states = self.post_attention_layernorm(hidden_states)
        # breakpoint()
        
        hidden_states = self.mlp(hidden_states).view_as(residual)
        hidden_states = residual + hidden_states

        outputs = hidden_states

        return outputs
    
    


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: Type[QuarotDecoderLayer] = QuarotDecoderLayer):
        
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            vocab_size,
            config.hidden_size,
        )
        
        # config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config


        self.quant_config = quant_config
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        
        self.layers = nn.ModuleList(
            [layer_type(config, i, cache_config, quant_config, prefix=f"{prefix}.layers.{i}")
             for i in range(config.num_hidden_layers)]
        )
        self.start_layer = 0
        self.end_layer = config.num_hidden_layers
        
            
        self.norm = quarot_nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            if inputs_embeds is None:    # For VLM models
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = inputs_embeds
                
            for i in range(self.start_layer, self.end_layer):
                layer = self.layers[i]
                #hidden_states, residual 
                hidden_states= layer(positions, hidden_states,
                                                kv_caches[i - self.start_layer],
                                                attn_metadata)
            # need to connect hidden_states and residual
            hidden_states = self.norm(hidden_states)
                
        return hidden_states





class QuarotFP16LlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(config)
        self.model = LlamaModel(config,vllm_config= vllm_config, prefix=maybe_prefix(prefix, "model"))
        # assert config._attn_implementation == "flash_attention_2"
        # for layer_idx, layer in enumerate(self.model.layers):
        #     layer.self_attn = QuarotFP16LlamaAttention(config=config, layer_idx=layer_idx)
        self.cache_dtype = "float16"
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.lora_config = lora_config
        self._expected_max_length = None
        self.model.norm =quarot_nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps,fuse=False)
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.unpadded_vocab_size = config.vocab_size
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                        config.vocab_size,
                                        logit_scale)
        self.sampler = get_sampler()

        self.make_empty_intermediate_tensors = (self.model.make_empty_intermediate_tensors)
        
    # def build_cache(self, batch_size, page_size, max_length):
    #     device = torch.device('cuda') #self.model.layers[0].self_attn.o_proj.weight.device
    #     dtype = self.cache_dtype #or self.model.layers[0].self_attn.o_proj.weight.dtype
        
    #     num_heads = self.config.num_attention_heads
    #     model_dim = self.config.hidden_size
    #     head_dim = model_dim // num_heads
    #     disable_quant = self.cache_dtype == "float16" 
    #     disable_quant = True
    #     return quarot.transformers.MultiLayerPagedKVCache4Bit(
    #         batch_size=batch_size,
    #         page_size=page_size, 
    #         max_seq_len=max_length, 
    #         device=device, 
    #         n_layers=len(self.model.layers),
    #         num_heads=num_heads,
    #         head_dim=head_dim,
    #         disable_quant=disable_quant,
    #         hadamard_dtype=None if disable_quant else torch.float16
    #     )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # breakpoint()
        model_output = self.model(input_ids, positions, kv_caches,
                                  attn_metadata, intermediate_tensors,
                                  inputs_embeds)
        return model_output
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(self, logits: torch.Tensor,
               sampling_metadata: SamplingMetadata) -> Optional[SamplerOutput]:
        
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens


class QuarotLlamaForCausalLM(QuarotFP16LlamaForCausalLM):
    def __init__(self, config, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(config,vllm_config= vllm_config, prefix=prefix)
        # assert config._attn_implementation == "flash_attention_2"
        # used to have a rmsn
        # for layer_idx, layer in enumerate(self.model.layers):
        #     self.model.layers[layer_idx] = QuarotDecoderLayer(config=config,layer_idx=layer_idx)
            # layer.self_attn = QuarotLlamaAttention(config=config, layer_idx=layer_idx)
            # layer.input_layernorm = quarot_nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            # layer.post_attention_layernorm = quarot_nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            # layer.mlp = QuarotLlamaMLP(config=config)
        self.cache_dtype = "float16"
        
    def fuse_qkv(self):
        for layer in self.model.layers:
            layer.self_attn.fuse_qkv()
        return self
