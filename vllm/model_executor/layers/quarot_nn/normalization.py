import torch
from qserve_backend import layernorm_ops
import quarot
class RMSNorm(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int, eps=1e-5,fuse=False):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        self.fuse=fuse
        self.bsz=1


        
        
    def forward(self, x,  **kwargs):
        if kwargs.get("w4a4",False):
            out_q = kwargs.get("quantized_buffer_qkv",None)
            scales = kwargs.get("scale_buffer",None)
            input_sum = kwargs.get("input_sum_buffer",None)
            return self.fuse_forward(x, out_q, scales, input_sum)
        else:
            return self.unfuse_forward(x)
        

    def fuse_forward(self, x: torch.Tensor,out_q = None, scaling_factor=None, input_sum=None) -> torch.Tensor:
        # input_dtype = x.dtype
        # if x.dtype == torch.float16:
        #     x = x.to(torch.float32)
        # variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        # x = x * torch.rsqrt(variance + self.eps)
        # return x.to(input_dtype)

        # breakpoint()

        if x.dim() == 2:
            batched_length,hidden_size = x.size()
        else:
            bsz, length, hidden_size = x.size()
            batched_length = bsz * length
            
        if out_q is None or input_sum is None or scaling_factor is None:
            out_q = torch.zeros(batched_length, hidden_size //2 , dtype=torch.int8, device="cuda")
            scaling_factor_and_input_sum = torch.zeros(2,batched_length, dtype=torch.float16, device="cuda")
            input_sum = scaling_factor_and_input_sum[0]
            scaling_factor = scaling_factor_and_input_sum[1]
                
        layernorm_ops.rms_norm_general_fuse_sum_i4(
            out_q,
            x,
            input_sum,
            scaling_factor,
            self.eps,
            True,
        )
        # breakpoint()
        return quarot.PackedQuantizedTensor(out_q, scaling_factor)

    def unfuse_forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)