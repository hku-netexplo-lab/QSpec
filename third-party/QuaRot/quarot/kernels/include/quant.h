#pragma once

#include <common.h>


void sym_quant_host(
        const half *x,
        const half *scale,
        uint32_t rows,
        uint32_t colsSrc,
        uint32_t colsDst,
        Int4Storage *q
);


void sym_dequant_host(
        const int32_t *q,
        const half *scale_row,
        const half *scale_col,
        uint32_t rows,
        uint32_t cols,
        half *x
);

void rowAbsMaxQuantize(
        at::Tensor x,
        at::Tensor scale,
        at::Tensor q,
        float clip_ratio,
        int block_size
);

void asym_quantize_and_pack_i4(
    at::Tensor x_k, at::Tensor x_v, 
    at::Tensor q_k, at::Tensor q_v, 
    at::Tensor scale_k, at::Tensor zero_k, 
    at::Tensor scale_v, at::Tensor zero_v,
    int bsz, int q_len, int num_key_value_heads, int head_dim
);