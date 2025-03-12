import torch
import triton
import triton.language as tl


# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_warps=8),
#         # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_warps=8),
#         # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128}, num_warps=8),
#         # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_warps=8),
#     ],
#     key=['M', 'N', 'K'],  # Autotune based on problem size
# )
@triton.jit
def awq_gemm_kernel_sym_nozp_perchannel(
    a_ptr,          # FP16, [M, K]
    b_ptr,          # UINT8, qweight，形状: [N, K//2]
    c_ptr,          # FP16, 输出矩阵 [M, N]
    scales_ptr,     # FP16, 每个输出通道的缩放因子，形状: [N]
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr = 64,
    BLOCK_SIZE_N: tl.constexpr = 64,
    BLOCK_SIZE_K: tl.constexpr = 32,
    SPLIT_K: tl.constexpr = None,
):
    # Get program IDs for M and N dimensions
    pid_m = tl.program_id(0) // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = tl.program_id(0) % tl.cdiv(N, BLOCK_SIZE_N)
    
    # Compute block offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize pointers
    a_ptrs = a_ptr + (offs_m[:, None] * K + offs_k[None, :])
    # For qweight: each UINT8 contains 2 INT4 values, so K//2 elements per row
    b_ptrs = b_ptr + (offs_n[:, None] * (K // 2) + offs_k[None, :] // 2)
    
    # Load scales for this N block
    scale_offs = offs_n
    scales = tl.load(scales_ptr + scale_offs, mask=offs_n < N, other=0.0)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Main K-loop
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Compute current K offset
        k_off = k * BLOCK_SIZE_K
        
        # Load A block (FP16)
        a_mask = (offs_m[:, None] < M) & (k_off + offs_k[None, :] < K)
        a = tl.load(a_ptrs + k_off, mask=a_mask, other=0.0)
        
        # Load B block (UINT8) and unpack
        b_mask = (offs_n[:, None] < N) & ((k_off + offs_k[None, :]) // 2 < K // 2)
        b_packed = tl.load(b_ptrs + (k_off // 2), mask=b_mask, other=0)
        
        # Unpack UINT8 to two signed INT4 values
        # Lower 4 bits (first value)
        b_low = (b_packed & 0x0F)
        b_low = tl.where(b_low >= 8, b_low - 16, b_low)
        
        # Upper 4 bits (second value)
        b_high = ((b_packed >> 4) & 0x0F)
        b_high = tl.where(b_high >= 8, b_high - 16, b_high)
        
        
        # Select between low and high based on even/odd K index
        b_unpacked = tl.where(offs_k[None, :] % 2 == 0, b_low, b_high)
        # Apply scaling (convert to float32 first)
        b_scaled =  scales[:, None] * b_unpacked 
        b_scaled_t = tl.trans(b_scaled)

        # Accumulate results
        acc += tl.dot(a, b_scaled_t, allow_tf32=False)
    
    # Write results
    c_ptrs = c_ptr + (offs_m[:, None] * N + offs_n[None, :])
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)

    

def awq_gemm_triton_sym_nozp_perchannel(
    input: torch.Tensor,      # [M, K], FP16
    qweight: torch.Tensor,    # [N, K//2], INT8 (2 INT4 per byte)，已转置
    scales: torch.Tensor,     # [N], FP16, per-channel (针对输出维度)
    split_k_iters: int,
    block_size_m: int = 32,
    block_size_n: int = 32,
    block_size_k: int = 32,
    result: torch.Tensor = None,
) -> torch.Tensor:
    M, K = input.shape
    N = qweight.shape[0]  # 注意：这里 N 对应 qweight 的第一维

    # 检查断言
    assert N > 0 and K > 0 and M > 0
    assert qweight.shape[1] * 2 == K or qweight.shape[1] == (K // 2), "qweight shape mismatch"
    assert scales.shape[0] == N  # per-channel scales
    # 这里只允许 split_k_iters==1 的情况，若需要支持更高版本，可扩展循环融合
    assert split_k_iters == 1

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    
    if result is None:
        result = torch.zeros((M, N), dtype=torch.float16, device=input.device)

    with torch.cuda.device(input.device):
        awq_gemm_kernel_sym_nozp_perchannel[grid](
            input,
            qweight,
            result,
            scales,
            M,
            N,
            K,
            # BLOCK_SIZE_M=block_size_m,
            # BLOCK_SIZE_N=block_size_n,
            # BLOCK_SIZE_K=block_size_k,
            SPLIT_K=split_k_iters,
        )

    return result