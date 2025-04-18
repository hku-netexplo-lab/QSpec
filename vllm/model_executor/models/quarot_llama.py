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
from vllm.model_executor.layers.rotary_embedding import ERotaryEmbedding
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
from vllm import _custom_ops as ops

from concurrent.futures import ThreadPoolExecutor
import os
def run_forward_attn(layer, positions, hidden_states, kv_cache, attn_metadata, stream, kwargs):
    with torch.cuda.stream(stream):
        return layer.forward_attn(positions, hidden_states, kv_cache, attn_metadata, **kwargs)


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
    
class QuaRotSequential(nn.Sequential):
    def forward(self, input,**kwargs):
        for module in self:
            input = module(input,**kwargs)
        return input

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
        self.q_proj = quarot_nn.Linear4bit.from_float(self.q_proj,**kwargs)
        self.k_proj = quarot_nn.Linear4bit.from_float(self.k_proj,**kwargs)
        self.v_proj = quarot_nn.Linear4bit.from_float(self.v_proj,**kwargs)
        self.qkv_proj = None
        self.o_proj_hadamard = quarot_nn.OnlineHadamard(num_heads)
        self.quantizer = quarot_nn.Quantizer(**kwargs)
        self.o_proj = quarot_nn.Linear4bit.from_float(self.o_proj,**kwargs)
        # self.o_proj = QuaRotSequential(
        #     quarot_nn.Quantizer(**kwargs),
        #     quarot_nn.Linear4bit.from_float(self.o_proj,**kwargs)
        # )
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
        self.head_size = self.head_dim
        self.rotary_emb = ERotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
            is_neox_style=self.is_neox_style,
            dtype = torch.get_default_dtype()
        )
        self.cos_sin_cache = self.rotary_emb.cos_sin_cache.cuda().to(torch.float16)
        
        # get_rope(
        #     self.head_dim,
        #     rotary_dim=self.head_dim,
        #     max_position=self.max_position_embeddings,
        #     base=self.rope_theta,
        #     rope_scaling=self.rope_scaling,
        #     is_neox_style=self.is_neox_style,
        # )
       
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
            
        
        
        
    def fuse_qkv(self,matmul_a16):
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
        self.qkv_proj.a16_matmul = matmul_a16
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

        if self.qkv_fused:
            act_buffer_qkv = kwargs.get("act_buffer_qkv",None)
            qkv_states = self.qkv_proj(hidden_states,act_buffer_qkv,**kwargs)
            q_len  = qkv_states.shape[0] // self.bsz
            bsz = self.bsz
            q_size = self.num_heads * self.head_dim
            kv_size = self.num_key_value_heads * self.head_dim
            query_states, key_states, value_states = qkv_states.split([q_size,kv_size,kv_size],dim=-1)
             
        else:    
            query_states = self.q_proj(hidden_states,**kwargs)
            key_states = self.k_proj(hidden_states,**kwargs)
            value_states = self.v_proj(hidden_states,**kwargs)
            q_len  = query_states.shape[0] // self.bsz
            bsz = self.bsz
            
            # Flash attention requires the input to have the shape
            # batch_size x seq_length x head_dim x hidden_dim
            # therefore we just need to keep the original shape
            
        if kwargs.get("w4a4", False):
            ops.rotary_embedding(positions, query_states, key_states, self.head_size,
                                 self.cos_sin_cache, self.is_neox_style)
        else:
            query_states, key_states = self.rotary_emb.forward_cuda(positions, query_states, key_states)
        

        # for spec-decode as draft model ------------------
        # query_states = query_states.to(torch.bfloat16)
        # key_states = key_states.to(torch.bfloat16)
        # value_states = value_states.to(torch.bfloat16)
        # end———————————————————————————— 

        act_buffer_attn = kwargs.get("act_buffer_attn",None)
        attn_output = self.attn(query_states, key_states, value_states, kv_cache, attn_metadata,act_buffer_attn).view(-1, self.num_heads, self.head_dim)


        # for spec-decode as draft model ------------------
        # attn_output = attn_output.to(torch.float16)
        # end————————————————————————————
        
                # hadamard

        act_buffer_had = kwargs.get("act_buffer_had",None)
        attn_output = self.o_proj_hadamard(attn_output.transpose(-1, -2).reshape(-1,self.num_heads) , act_buffer_had,**kwargs)
        attn_output = attn_output.view(-1, self.head_dim, self.num_heads).transpose(-1, -2)
            
        attn_output = attn_output.reshape(bsz * q_len, self.hidden_size).contiguous()
        
        quantized_buffer_qkv = kwargs.get("quantized_buffer_qkv",None)
        scale_buffer = kwargs.get("scale_buffer",None)
        quantized_attn_output = self.quantizer(attn_output, scale_buffer,quantized_buffer_qkv,**kwargs)
        
        act_buffer_output = kwargs.get("act_buffer_output",None)
        attn_output = self.o_proj(quantized_attn_output, act_buffer_output,**kwargs)

        return attn_output



class QuarotLlamaMLP(LlamaMLP):
    def __init__(self, config, layer_idx, **kwargs):
        super().__init__(config=config)
        self.quantizer = quarot_nn.Quantizer()
        self.up_proj = quarot_nn.Linear4bit.from_float(self.up_proj,**kwargs)
        self.gate_proj = quarot_nn.Linear4bit.from_float(self.gate_proj,**kwargs)
        self.online_hadamard = quarot_nn.OnlineHadamard(self.intermediate_size)
        self.down_proj = quarot_nn.Linear4bit.from_float(self.down_proj,**kwargs)
        self.fused = False
        self.gate_up = None
        self.layer_idx = layer_idx
        
        # self.down_proj = QuaRotSequential(
        #     quarot_nn.OnlineHadamard(self.intermediate_size),
        #     quarot_nn.Quantizer(**kwargs),
        #     quarot_nn.Linear4bit.from_float(self.down_proj,**kwargs)
        # )


    def forward(self, x,**kwargs):
        
        if not self.fused:
            # gate_proj
            act_buffer_gate = kwargs.get("act_buffer_gate",None)
            gate = self.gate_proj(x,act_buffer_gate,**kwargs)
            
            # up_proj
            act_buffer_up = kwargs.get("act_buffer_up",None)
            up_proj = self.up_proj(x,act_buffer_up,**kwargs)
        else:
            act_buffer_gate_up = kwargs.get("act_buffer_gate_up",None)
            gate_up = self.gate_up(x,act_buffer_gate_up,**kwargs)
            up_proj, gate = gate_up[:,:14336], gate_up[:,14336:]
            
            
        
        # Silu activation
        gate_up = self.act_fn(gate) * up_proj
        
        # hadamard

        act_buffer_had_mlp = kwargs.get("act_buffer_had_mlp",None)
        gate_up = self.online_hadamard(gate_up,act_buffer_had_mlp,**kwargs).view(-1,self.intermediate_size)
        
        # quantisation
        quantized_buffer_mlp = kwargs.get("quantized_buffer_mlp",None)
        scale_buffer = kwargs.get("scale_buffer",None)
        quantized_gate_up = self.quantizer(gate_up,scale_buffer,quantized_buffer_mlp,**kwargs)
        
        # down_proj
        act_buffer_output = kwargs.get("act_buffer_output",None)
        down_proj = self.down_proj(quantized_gate_up,act_buffer_output,**kwargs)
        return down_proj
    
    def fuse_gate_up(self,matmul_a16):
        self.fused = True
        self.gate_up = quarot_nn.Linear4bit(self.up_proj.in_features,self.up_proj.out_features*2,bias=False)
        self.gate_up.weight.data[:self.up_proj.out_features] = self.up_proj.weight.data
        self.gate_up.weight.data[self.up_proj.out_features:] = self.gate_proj.weight.data
        self.gate_up.weight.data = self.gate_up.weight.data.to(torch.int8).cuda()
        
        self.gate_up.weight_scales = self.gate_up.weight_scales.view(-1)
        self.gate_up.weight_scales.data[:self.up_proj.out_features] = self.up_proj.weight_scales.data.view(-1)
        self.gate_up.weight_scales.data[self.up_proj.out_features:] = self.gate_proj.weight_scales.data.view(-1)
        self.gate_up.weight_scales = self.gate_up.weight_scales.cuda()
        self.gate_up.a16_matmul = matmul_a16
        del self.up_proj
        del self.gate_proj
    
    


class QuarotDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx,
                cache_config: Optional[CacheConfig] = None,
                quant_config: Optional[QuantizationConfig] = None,
                prefix: str = "",
                **kwargs
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
                                            prefix=f"{prefix}.self_attn",
                                            **kwargs)
        
        self.mlp = QuarotLlamaMLP(config=config,layer_idx=layer_idx,**kwargs)
        
        self.input_layernorm = quarot_nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps,fuse = kwargs.get("w4a4", False))
        self.post_attention_layernorm = quarot_nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, fuse = kwargs.get("w4a4", False))
        
        
        
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:


        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states,**kwargs)    
        hidden_states = self.self_attn(positions=positions,
                                hidden_states=hidden_states,
                                kv_cache=kv_cache,
                                attn_metadata=attn_metadata,
                                **kwargs)
        hidden_states = residual + hidden_states.view_as(residual)
        
        # Attention
        # -----
        # MLP
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states,**kwargs)
        hidden_states = self.mlp(hidden_states,**kwargs).view_as(residual)
        hidden_states = residual + hidden_states
        outputs = hidden_states

        return outputs
    
    
    def forward_attn(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states,**kwargs)    
        hidden_states = self.self_attn(positions=positions,
                                hidden_states=hidden_states,
                                kv_cache=kv_cache,
                                attn_metadata=attn_metadata,
                                **kwargs)
        
        return hidden_states.view_as(residual)
    

    def forward_mlp(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # MLP
        # hidden_states = residual(original input) + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states,**kwargs)
        hidden_states = self.mlp(hidden_states,**kwargs).view_as(residual)
        hidden_states = residual + hidden_states
        outputs = hidden_states

        return outputs
    
    


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: Type[QuarotDecoderLayer] = QuarotDecoderLayer,
        **kwargs):
        
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
            [layer_type(config, i, cache_config, quant_config, prefix=f"{prefix}.layers.{i}",**kwargs)
             for i in range(config.num_hidden_layers)]
        )
        self.start_layer = 0
        self.end_layer = config.num_hidden_layers
        
            
        self.norm = quarot_nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        
        # self.streams = [torch.cuda.Stream() for _ in range(8)]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        with torch.no_grad():
            if inputs_embeds is None:    # For VLM models
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = inputs_embeds
                
            for i in range(self.start_layer, self.end_layer):
                # skip_set = [8,9,10,11, 
                #             16,17,18,19,
                #             24,25,26,27]
                
                # if (i in skip_set ) and kwargs.get("w4a4", False):
                #     stride = 4
                #     layer = self.layers[i]
                #     layer_list = [self.layers[j] for j in range(i, i + stride)]
                #     hidden_states_a1 = layer.forward_attn(positions, hidden_states,
                #                                     kv_caches[i - self.start_layer],
                #                                     attn_metadata,**kwargs)
                #     hidden_states_list = [hidden_states_a1.clone() for _ in range(stride)]
                #     for j in range(stride):
                #         hidden_states_list[j] = layer_list[j].forward_mlp(
                #             positions, hidden_states_list[j] + (hidden_states if j == 0 else hidden_states_list[j - 1]),
                #             None,
                #             attn_metadata, **kwargs
                #         )
                #     hidden_states = hidden_states_list[-1]
                #     i += (stride-1)
                # else:
                #     #
                #     if (i in skip_set) and kwargs.get("w4a4", False):
                #         continue
                    #
                layer = self.layers[i]
                hidden_states = layer(positions, hidden_states,
                                                kv_caches[i - self.start_layer],
                                                attn_metadata,**kwargs)

                
            # end of for loop
            hidden_states = self.norm(hidden_states)  
                
        return hidden_states





class QuarotFP16LlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, vllm_config: VllmConfig, prefix: str = "", **kwargs):
        super().__init__(config)
        self.model = LlamaModel(config,vllm_config= vllm_config, prefix=maybe_prefix(prefix, "model"),**kwargs)
        # assert config._attn_implementation == "flash_attention_2"
        # for layer_idx, layer in enumerate(self.model.layers):
        #     layer.self_attn = QuarotFP16LlamaAttention(config=config, layer_idx=layer_idx)
        self.cache_dtype = "float16"
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.lora_config = lora_config
        self._expected_max_length = None
        self.w4a4 = kwargs.get("w4a4", False)
        self.model.norm =quarot_nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.unpadded_vocab_size = config.vocab_size
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                        config.vocab_size,
                                        logit_scale)
        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (self.model.make_empty_intermediate_tensors)
        


    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # breakpoint()
        model_output = self.model(input_ids, positions, kv_caches,
                                  attn_metadata, intermediate_tensors,
                                  inputs_embeds,**kwargs)
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
    def __init__(self, config, vllm_config: VllmConfig, prefix: str = "", **kwargs):
        print(f"W4A4{kwargs.get('w4a4', False)}")
        super().__init__(config,vllm_config= vllm_config, prefix=prefix, **kwargs)
        self.cache_dtype = "float16"
        
    def fuse_qkv(self):
        from vllm.model_executor.layers.quarot_nn.linear import get_matmul_op_w4a16
        in_feature = 4096
        out_feature = 4096 + 1024 * 2
        matmul_a16 = get_matmul_op_w4a16(in_features=in_feature, out_features=out_feature)
        for layer in self.model.layers:
            layer.self_attn.fuse_qkv(matmul_a16)
        return self
    
    
    def fuse_gate_up(self):
        from vllm.model_executor.layers.quarot_nn.linear import get_matmul_op_w4a16
        in_feature = 4096
        out_feature = 14336 * 2
        matmul_a16 = get_matmul_op_w4a16(in_features=in_feature, out_features=out_feature)
        for layer in self.model.layers:
            layer.mlp.fuse_gate_up(matmul_a16)
        return self