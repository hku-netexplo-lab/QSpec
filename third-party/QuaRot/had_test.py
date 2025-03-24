import torch, math
import fast_hadamard_transform
# Adapted from https://github.com/Cornell-RelaxML/quip-sharp/blob/main/lib/utils/matmul_had.py
import fast_hadamard_transform_cuda


# test

seq_len = 1024
num_head = 32
hidden_size = 4096
intermediate_size = 14336

X_attn = torch.randn(seq_len, hidden_size//num_head, num_head, dtype=torch.float16, device='cuda')
X_mlp = torch.randn(seq_len, 28, intermediate_size//28, dtype=torch.float16, device='cuda')
scale_32 = 1.0/torch.tensor(32).sqrt()
scale_14336 = 1.0/torch.tensor(14336).sqrt()



input = X_attn.view(-1, num_head)
out = torch.empty_like(input)
out = fast_hadamard_transform_cuda.faster_fast_hadamard_transform(input, scale_32, out )
# out = fast_hadamard_transform_cuda.faster_fast_hadamard_transform(X, scale, out )
ref = fast_hadamard_transform.hadamard_transform(X_attn, 1.0/torch.tensor(32).sqrt()) 
breakpoint()
out = out.view_as(X_attn)
all_close = torch.allclose(out, ref)
print(all_close)