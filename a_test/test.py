import vllm._custom_ops as ops
import torch 

# 
M, N, K = 1024, 4096, 4096
input_1 = torch.randint(-8, 8, (M, N), dtype=torch.float16,device='cuda')
output_1 = torch.empty_like(input_1, device='cuda')
weight_1 = torch.ones(N, dtype=torch.float16, device='cuda')

ops.rms_norm(output_1, input_1, weight_1, 1e-5)

breakpoint()

output_2 = torch.empty_like(input_1, device='cuda')
ops.qspec_rms_norm_w4a16(output_2, input_1, 1e-5)


breakpoint()