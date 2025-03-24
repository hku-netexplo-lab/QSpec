# import triton
# import triton.language as tl
# import torch
# from triton.language.extra import libdevice
# import pdb


# @triton.jit
# def sym_quantize_f16_i4_kernel(
#     x_ptr, q_ptr, row_max_ptr,
#     rows, colsSrc, colsDst, input_clip_ratio,
#     BLOCK_SIZE: tl.constexpr
# ):
#     pdb.set_trace()
    
#     row_idx = tl.program_id(0)  # 处理的行索引
#     col_idx = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 处理的列索引
#     row_start = row_idx * colsSrc  # 该行的起始位置

#     # **计算每个 block 负责的 `max(abs(x))`**
#     valid = col_idx < colsSrc  # 避免越界
#     x = tl.load(x_ptr + row_start + col_idx, mask=valid, other=0.0)
#     local_max = tl.max(tl.abs(x), axis=0)

#     local_max_fp32 = local_max.to(tl.float32)
#     # **多个 blocks 计算同一行时，使用 `atomic_max` 归约**
#     tl.atomic_max(row_max_ptr + row_idx, local_max_fp32)

#     # **同步，确保所有 `blockIdx.x` 计算完 `row_max[row]`**
#     tl.debug_barrier()
#     row_max = tl.load(row_max_ptr + row_idx)

#     # **计算 scale[row]**
#     scale = (row_max / 7.0) * input_clip_ratio

#     # **执行量化**
#     x_quant = tl.clamp(libdevice.round(x / scale), -8, 7).to(tl.int8)

#     # **打包量化值，两个 `int4` 合并成一个 `int8`**
#     # 假设每两个量化值（`x_quant`）打包为一个 `int8`
#     col_idx = tl.arange(0, BLOCK_SIZE)
#     idx_in_group = col_idx % 2  # 判断当前列属于第1个还是第2个`int4`
    
#     # **打包**
    
#     # 从两个 `int4` 中创建一个 `int8`
#     x_quant_0 = tl.where(idx_in_group == 0, x_quant, 0)
#     x_quant_1 = tl.where(idx_in_group == 1, x_quant, 0)
#     packed = tl.zeros([BLOCK_SIZE // 2], dtype=tl.int8)  # 创建一个新的容器

#     packed = tl.atomic_or(packed, x_quant_0 << 4)
#     packed = tl.atomic_or(packed, x_quant_1)
    
    
#     # **存储打包后的结果**
#     valid_mask = col_idx < colsDst * 2  # 避免越界
#     tl.store(q_ptr + row_idx * colsDst + col_idx // 2, packed, mask=valid_mask)

# def fuse_sym_quant_torch(x: torch.Tensor, input_clip_ratio: float):
#     rows, colsSrc = x.shape
#     colsDst = colsSrc // 2  # 量化后形状，两个 `int4` 合并为一个 `int8`

#     q = torch.empty((rows, colsDst), dtype=torch.int8, device=x.device)
#     row_max = torch.zeros((rows,), dtype=torch.float32, device=x.device)

#     BLOCK_SIZE = 256  # 每个 block 计算的列数
#     num_blocks = (colsSrc + BLOCK_SIZE - 1) // BLOCK_SIZE  # 让多个 blocks 并行计算同一行

#     grid = (rows, num_blocks)

#     sym_quantize_f16_i4_kernel[grid](
#         x, q, row_max,
#         rows, colsSrc, colsDst, input_clip_ratio,
#         BLOCK_SIZE=BLOCK_SIZE
#     )

#     return q, row_max
