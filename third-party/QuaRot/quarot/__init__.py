import torch
from . import nn
from . import functional
from . import triton
from torchao.ops import rowwise_scaled_linear_cutlass_s4s4_unified, rowwise_scaled_linear_cutlass_s4s4

import quarot._CUDA



__all__ = [ 
           "matmul", #int-4 matmul
           "sym_quant", "sym_dequant", "PackedQuantizedTensor", # Quantization
           "sym_quant_i8", "sym_dequant_i8", # INT8 Quantization
           "fuse_sym_quant_with_buffer", "fuse_sym_quant", "fuse_sym_quant_i8", # Fused Quantization
]

class ShapeHandler:
    def __init__(self, x: torch.Tensor):
        self.size_excl_last = x.numel()//x.shape[-1]
        self.shape_excl_last = tuple(x.shape[:-1])

    # Keep the last dim unchanged, flatten all previous dims
    def flatten(self, x: torch.Tensor):
        return x.view(self.size_excl_last, -1)

    # Recover back to the original shape.
    def unflatten(self, x: torch.Tensor):
        return x.view(self.shape_excl_last + (-1,))

    def unflatten_scale(self, x: torch.Tensor):
        return x.view(self.shape_excl_last)


def flatten_last_dim_and_return_shape(x: torch.Tensor):
    shape_excl_last = x.shape[:-1]
    x = x.view(-1, x.shape[-1])
    return x, shape_excl_last


def matmul(A, B):
    assert A.shape[-1] % 32 == 0, "A.shape[-1]: {} must be multiplication of 32".format(A.shape[-1])
    A, A_shape_excl_last = flatten_last_dim_and_return_shape(A)
    B, B_shape_excl_last = flatten_last_dim_and_return_shape(B)
    return quarot._CUDA.matmul(A, B).view(*A_shape_excl_last, *B_shape_excl_last)

def opt_matmul(A, B):
    assert A.shape[-1] % 32 == 0, "A.shape[-1]: {} must be multiplication of 32".format(A.shape[-1])
    A, A_shape_excl_last = flatten_last_dim_and_return_shape(A)
    # B, B_shape_excl_last = flatten_last_dim_and_return_shape(B)
    B_shape_excl_last = B.shape[:-1]
    return quarot._CUDA.matmul(A, B).view(*A_shape_excl_last, *B_shape_excl_last)

def fuse_matmul(A,A_Scales,B,B_Scales,bias=None, C=None):
    assert A.shape[-1] % 32 == 0, "A.shape[-1]: {} must be multiplication of 32".format(A.shape[-1])
    # if A.dtype != torch.int8:
    #     A = A.to(torch.int8)
    # if A_Scales.dim() >= 2:
    #     A_Scales = A_Scales.view(-1)
    #A, A_shape_excl_last = flatten_last_dim_and_return_shape(A)
    # from quarot.functional import unpack_i4
    # unpacked_A = unpack_i4(A.to(torch.uint8)).to(torch.float16)
    # unpacked_B = unpack_i4(B.to(torch.uint8)).to(torch.float16)
    # x_d = unpacked_A * A_Scales.view(-1, 1)
    # w_d = B_Scales.view(-1, 1) * unpacked_B  
    # ref1 = x_d @ w_d.T
    # ref2 = (
    #     (unpacked_A @ unpacked_B.T)
    #     * A_Scales.view(-1, 1)
    #     * B_Scales.view(1, -1)
    # ).to(torch.float16)
    # breakpoint()
    # breakpoint()
    # C = torch.empty(A.shape[0], B.shape[0] , dtype=torch.float16, device=A.device)

    return rowwise_scaled_linear_cutlass_s4s4_unified(A, A_Scales, B, B_Scales, bias,C) #.view(*A_shape_excl_last, -1)
    # return rowwise_scaled_linear_cutlass_s4s4(A, A_Scales, B, B_Scales, bias)
    
    
def fuse_asym_quantize_and_pack_i4(x_k, x_v):
    assert x_k.dtype == x_v.dtype == torch.float16
    assert x_k.shape == x_v.shape
    bsz, q_len, num_key_value_heads, head_dim = x_k.shape
    # just creat once then split
    q_all = torch.empty(2,bsz, q_len, num_key_value_heads, head_dim//2, dtype=torch.uint8, device=x_k.device)
    q_k = q_all[0]
    q_v = q_all[1]
    # q_k = torch.zeros(bsz, q_len, num_key_value_heads, head_dim//2, dtype=torch.uint8, device=x_k.device)
    # q_v = torch.zeros(bsz, q_len, num_key_value_heads, head_dim//2, dtype=torch.uint8, device=x_k.device)
    sz_all = torch.empty(4,bsz, q_len, num_key_value_heads,1, dtype=torch.float16, device=x_k.device)
    scale_k = sz_all[0]
    zero_k = sz_all[1]
    scale_v = sz_all[2]
    zero_v = sz_all[3]
    # scale_k = torch.zeros(bsz, q_len, num_key_value_heads,1, dtype=torch.float16, device=x_k.device)
    # zero_k = torch.zeros(bsz, q_len, num_key_value_heads,1, dtype=torch.float16, device=x_k.device)
    # scale_v = torch.zeros(bsz, q_len, num_key_value_heads,1, dtype=torch.float16, device=x_k.device)
    # zero_v = torch.zeros(bsz, q_len, num_key_value_heads,1, dtype=torch.float16, device=x_k.device)
    quarot._CUDA.fuse_asym_quantize_and_pack_i4(x_k, x_v, q_k, q_v, scale_k, zero_k, scale_v, zero_v,bsz, q_len, num_key_value_heads, head_dim)
    return q_k, q_v, scale_k, zero_k, scale_v, zero_v
    

def sym_quant(x, scale):
    assert x.dtype == scale.dtype == torch.float16
    x, x_shape_excl_last = flatten_last_dim_and_return_shape(x)
    return quarot._CUDA.sym_quant(x, scale.view(-1)).view(*x_shape_excl_last, -1)

def sym_quant_i8(x, scale):
    assert x.dtype == scale.dtype == torch.float16
    x, x_shape_excl_last = flatten_last_dim_and_return_shape(x)
    return quarot._CUDA.sym_quant_i8(x, scale.view(-1)).view(*x_shape_excl_last, -1)

def sym_dequant_i8(q, scale):
    assert q.dtype == torch.int8
    assert scale.dtype == torch.float16
    q, q_shape_excl_last = flatten_last_dim_and_return_shape(q)
    return quarot._CUDA.sym_dequant_i8(q, scale.view(-1)).view(*q_shape_excl_last, -1)

def fuse_sym_quant(x, clip_ratio = 1.0):
    batched_seq_len, hidden_size = x.shape
    scale = torch.empty(batched_seq_len, dtype=torch.float16, device=x.device)
    out = torch.empty(batched_seq_len, hidden_size//2, dtype=torch.int8, device=x.device)
    quarot._CUDA.fuse_sym_quant(x, scale, out, clip_ratio)
    #return out.view(*x_shape_excl_last, -1), scale
    return out, scale

def fuse_sym_quant_i8(x, clip_ratio = 1.0):
    batched_seq_len, hidden_size = x.shape
    scale = torch.empty(batched_seq_len, dtype=torch.float16, device=x.device)
    out = torch.empty(batched_seq_len, hidden_size, dtype=torch.int8, device=x.device)
    quarot._CUDA.fuse_sym_quant_i8(x, scale, out, clip_ratio)
    return out, scale

def fuse_sym_quant_i8_with_buffer(x, scale, out, clip_ratio = 1.0):
    quarot._CUDA.fuse_sym_quant_i8(x, scale, out, clip_ratio)
    return out, scale

def fuse_sym_quant_with_buffer(x, scale, out, clip_ratio = 1.0):
    quarot._CUDA.fuse_sym_quant(x, scale, out, clip_ratio)
    return out, scale

def fuse_sym_quant_with_buffer(x,scales,out,clip_ratio = 1.0):
    quarot._CUDA.fuse_sym_quant(x, scales, out, clip_ratio)
    return out, scales


def sym_dequant(q, scale_row, scale_col, bits=32):
    assert q.dtype == torch.int32
    assert scale_row.dtype == scale_col.dtype == torch.float16
    q, q_shape_excl_last = flatten_last_dim_and_return_shape(q)
    return quarot._CUDA.sym_dequant(q, scale_row.view(-1), scale_col, bits).view(*q_shape_excl_last, -1)


class PackedQuantizedTensor:
    def __init__(self, 
                 quantized_x: torch.Tensor, 
                 scales_x: torch.Tensor):
        self.quantized_x = quantized_x
        self.scales_x = scales_x

    def size(self):
        return self.quantized_x.size()
    
    @property
    def device(self):
        return self.quantized_x.device
    
    @property
    def dtype(self):
        return self.quantized_x.dtype
