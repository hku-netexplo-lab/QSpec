import functools
import quarot
import quarot.transformers
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, \
LlamaForCausalLM, apply_rotary_pos_emb, LlamaMLP, LlamaDecoderLayer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from typing import Optional, Tuple
from transformers import Cache,DynamicCache


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

ALL_LAYERNORM_LAYERS.append(quarot.nn.RMSNorm)

class QuarotLlamaConfig(LlamaConfig):
    model_type = "llama_quarot"

class QuarotFP16LlamaAttention(LlamaAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = getattr(self, "config", None)
        self.quantizer = torch.nn.Identity()
        self.o_proj_hadamard = torch.nn.Identity()
        self.qkv_fused= False
        self.bsz = 1
        
    def fuse_qkv(self):
        if self.qkv_fused:
            return self
        self.qkv_fused = True
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_key_value_heads * self.head_dim
        self.qkv_proj = quarot.nn.Linear4bit(self.q_proj.in_features,q_size+ 2*kv_size,bias=False)
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
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
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
        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim).contiguous() ##.transpose(1, 2)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).contiguous()  ##.transpose(1, 2)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).contiguous()  ##.transpose(1, 2)

        # breakpoint()
        
        kv_seq_len = key_states.shape[1] ##[-2]
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # breakpoint()
        try:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids, unsqueeze_dim=2) #, unsqueeze_dim=2
        except:
            breakpoint()
        past_key_value = getattr(self, "past_key_value", past_key_value)
        assert past_key_value is not None
        # sin and cos are specific to RoPE models; position_ids needed for the static cache
        
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, "attention_mask": attention_mask}
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        
        
        cache_out = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        

        dropout_rate = self.attention_dropout if self.training else 0.0

        assert self.is_causal

        if isinstance(cache_out, tuple):
            key_states, value_states = cache_out
            
            # query_states = query_states.transpose(1, 2)
            # key_states = key_states.transpose(1, 2)
            # value_states = value_states.transpose(1, 2)
            
            attn_output = self._flash_attention_forward(
                query_states, 
                key_states, 
                value_states, 
                query_length=q_len, 
                attention_mask=attention_mask
            )

        else:
            attn_output = cache_out(query_states)

        breakpoint()
        attn_output = self.o_proj_hadamard(attn_output.transpose(-1, -2)).transpose(-1, -2)
        # breakpoint()
        attn_output = attn_output.reshape(bsz * q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class QuarotLlamaAttention(QuarotFP16LlamaAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = quarot.nn.Quantizer()
        self.q_proj = quarot.nn.Linear4bit.from_float(self.q_proj)
        self.k_proj = quarot.nn.Linear4bit.from_float(self.k_proj)
        self.v_proj = quarot.nn.Linear4bit.from_float(self.v_proj)
        self.qkv_proj = None
        self.num_heads = self.config.num_attention_heads
        self.o_proj_hadamard = quarot.nn.OnlineHadamard(self.num_heads)
        self.o_proj = torch.nn.Sequential(
            quarot.nn.Quantizer(),
            quarot.nn.Linear4bit.from_float(self.o_proj)
        )

class QuarotLlamaMLP(LlamaMLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = quarot.nn.Quantizer()
        self.up_proj = quarot.nn.Linear4bit.from_float(self.up_proj)
        self.gate_proj = quarot.nn.Linear4bit.from_float(self.gate_proj)
        self.down_proj = torch.nn.Sequential(
            quarot.nn.OnlineHadamard(self.intermediate_size),
            quarot.nn.Quantizer(),
            quarot.nn.Linear4bit.from_float(self.down_proj)
        )

    def forward(self, x):
        # bsz, seq_len, _ = x.size()
        # x = self.quantizer(x)
        return super().forward(x) #.view(bsz, seq_len, -1)


class QuarotDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config,layer_idx):
        super().__init__(config,layer_idx)
        self.layer_idx = layer_idx
        self.self_attn = QuarotLlamaAttention(config=config, layer_idx=layer_idx)
        self.input_layernorm = quarot.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = quarot.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = QuarotLlamaMLP(config=config)
        
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        # if "padding_mask" in kwargs:
        #     warnings.warn(
        #         "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        #     )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # breakpoint()
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        
        hidden_states = residual + hidden_states.view_as(residual)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # breakpoint()
        hidden_states = self.mlp(hidden_states).view_as(residual)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class QuarotFP16LlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        assert config._attn_implementation == "flash_attention_2"
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = QuarotFP16LlamaAttention(config=config, layer_idx=layer_idx)
        self.cache_dtype = "float16"
        self._expected_max_length = None
        self.model.norm =quarot.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps,fuse=False)


        
    def build_cache(self, batch_size, page_size, max_length):
        device = torch.device('cuda') #self.model.layers[0].self_attn.o_proj.weight.device
        dtype = self.cache_dtype #or self.model.layers[0].self_attn.o_proj.weight.dtype
        
        num_heads = self.config.num_attention_heads
        model_dim = self.config.hidden_size
        head_dim = model_dim // num_heads
        disable_quant = self.cache_dtype == "float16" 
        disable_quant = True
        return quarot.transformers.MultiLayerPagedKVCache4Bit(
            batch_size=batch_size,
            page_size=page_size, 
            max_seq_len=max_length, 
            device=device, 
            n_layers=len(self.model.layers),
            num_heads=num_heads,
            head_dim=head_dim,
            disable_quant=disable_quant,
            hadamard_dtype=None if disable_quant else torch.float16
        )

    def _get_logits_processor(self, generation_config, *args, **kwargs):
        # This is a hack to get the max length from generation_config.
        # Doing it here because max_length might not be set before this 
        # method is called.
        self._expected_max_length = generation_config.max_length # This value will be reset at the next forward call
        return super()._get_logits_processor(generation_config, *args, **kwargs)


    def forward(self, input_ids, *args, past_key_values=None, **kwargs):
        if past_key_values is None:
            max_length = self._expected_max_length or input_ids.shape[1]
            self._expected_max_length = None # Reset this value.
            past_key_values = self.build_cache(
                input_ids.shape[0], 
                page_size=max_length,  # For now working with single page per batch.
                max_length=max_length)
            # past_key_values = DynamicCache()
        out = super().forward(input_ids, *args, past_key_values=past_key_values, **kwargs)
        return out
    


class QuarotLlamaForCausalLM(QuarotFP16LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        assert config._attn_implementation == "flash_attention_2"
        # used to have a rmsn
        for layer_idx, layer in enumerate(self.model.layers):
            self.model.layers[layer_idx] = QuarotDecoderLayer(config=config,layer_idx=layer_idx)
            # layer.self_attn = QuarotLlamaAttention(config=config, layer_idx=layer_idx)
            # layer.input_layernorm = quarot.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            # layer.post_attention_layernorm = quarot.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            # layer.mlp = QuarotLlamaMLP(config=config)
        self.cache_dtype = "float16"
        
    def fuse_qkv(self):
        for layer in self.model.layers:
            layer.self_attn.fuse_qkv()
        return self
