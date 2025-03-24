#include <quant.h>


template<typename T>
__device__ __half int_to_half(T value)
{
    return __int2half_rn(static_cast<int>(value));
}


__global__
void sym_quantize_f16_i4_kernel(
        const half *__restrict__ x,
        const half *__restrict__ scale,
        uint32_t rows,
        uint32_t colsSrc,
        uint32_t colsDst,
        Int4Storage *__restrict__ q
)
{
    uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t colDst = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= rows || colDst * kElementsPerVector >= colsSrc)
    {
        return;
    }
    Int4Storage storage;
    memset(&storage, 0, sizeof(storage));
    uint32_t id = colDst * kElementsPerVector + row * colsSrc;
#pragma unroll
    for (int i = 0; i < kElementsPerVector; ++i)
    {
        bool safe = (colDst * kElementsPerVector + i) < colsSrc;
        if (safe)
        {
            half data = __hdiv(x[id + i], scale[row]);

            int qval = clamp(__half2int_rn(data), qmin, qmax);
            Int4Subbyte{reinterpret_cast<cutlass::int4b_t *>(&storage), i}.set(
                    qval);
        }
    }

    q[colDst + row * colsDst] = storage;
}


void sym_quant_host(
        const half *x,
        const half *scale,
        uint32_t rows,
        uint32_t colsSrc,
        uint32_t colsDst,
        Int4Storage *q
)
{

    dim3 block{std::min<uint32_t>(colsDst, 32), std::min<uint32_t>(rows, 16)};
    dim3 grid{cdiv(colsDst, block.x), cdiv(rows, block.y)};
    sym_quantize_f16_i4_kernel<<<grid, block>>>(x, scale, rows, colsSrc, colsDst, q);
}


__global__ void sym_dequantize_i32_f16_kernel(
        const int32_t *__restrict__ q,
        const half *__restrict__ scale_row,
        const half *__restrict__ scale_col,
        uint32_t rows, uint32_t cols,
        half *__restrict__ x)
{
    uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t col = threadIdx.x + blockIdx.x * blockDim.x;

    if (col >= cols || row >= rows)
    {
        return;
    }

    half xElement = int_to_half(q[col + row * cols]);
    x[col + row * cols] = scale_row[row] * scale_col[col] * xElement;
}

void sym_dequant_host(const int32_t *q,
                                 const half *scale_row,
                                 const half *scale_col,
                                 uint32_t rows,
                                 uint32_t cols,
                                 half *x
)
{
    dim3 block{std::min<uint32_t>(cols, 16), std::min<uint32_t>(rows, 16)};
    dim3 grid{cdiv(cols, block.x), cdiv(rows, block.y)};
    sym_dequantize_i32_f16_kernel<<<grid, block>>>(
            q,
            scale_row, scale_col,
            rows, cols, x);
}


__global__ 
void rowAbsMaxQuantizeKernel(const at::Half* d_mat, at::Half* d_scale, int8_t* d_out, int m, int n, float clip_ratio, int block_size) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int col_start = tid;

    extern __shared__ at::Half sharedMax[];
    sharedMax[tid] = __float2half(0.0f);

    // 计算当前行的最大绝对值
    for (int col = col_start; col < n; col += block_size) {
        sharedMax[tid] = __hmax(sharedMax[tid], __habs(d_mat[row * n + col]));
    }
    __syncthreads();

    // 归约求最大值
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedMax[tid] = __hmax(sharedMax[tid], sharedMax[tid + stride]);
        }
        __syncthreads();
    }

    at::Half max_abs = sharedMax[0];
    if (tid == 0) {
        d_scale[row] = __hdiv(max_abs, __float2half(7.0f)) * __float2half(clip_ratio);
    }

    __syncthreads();
    at::Half scale = d_scale[row];

    // // 量化为 INT4 范围
    // for (int col = col_start; col < n; col += block_size) {
    //     half val = d_mat[row * n + col];
    //     int quantized = __float2int_rn(__half2float(val) / __half2float(scale));
    //     quantized = max(-8, min(7, quantized));

    //     int out_index = (row * n + col) / 2;
    //     int8_t is_high = (col % 2 == 0) ? 0 : 1;

    //     atomicAnd(&d_out[out_index], is_high ? 0x0F : 0xF0);
    //     atomicOr(&d_out[out_index], (quantized & 0x0F) << (is_high ? 4 : 0));
    // }

    // 量化为 INT4 范围
    for (int col = col_start; col < n; col += block_size) {
        at::Half val = d_mat[row * n + col];
        //int quantized = __float2int_rn(__half2float(val) / __half2float(scale));
        at::Half quantized = __hdiv(val, scale);
    
        int8_t quantized_int =  max(-8, min(7, __half2int_rn(quantized)));

        if (col % 2 == 0) {
            int out_index = (row * n + col) / 2;
            int8_t quantized_next = 0;

            if (col + 1 < n) {
                at::Half val_next = d_mat[row * n + col + 1];
                at::Half quantized = __hdiv(val_next, scale);
                quantized_next = max(-8, min(7, __half2int_rn(quantized)));
            }

            d_out[out_index] = (quantized_int & 0x0F) | ((quantized_next & 0x0F) << 4);
        }
    }
}

void rowAbsMaxQuantize(
    at::Tensor d_mat, at::Tensor d_scale, at::Tensor d_out, float clip_ratio, int block_size) {

    const at::Half* d_mat_ptr = d_mat.data_ptr<at::Half>();
    at::Half* d_scale_ptr = d_scale.data_ptr<at::Half>();
    int8_t* d_out_ptr = d_out.data_ptr<int8_t>();

    int m = d_mat.size(0);
    int n = d_mat.size(1);

    dim3 grid(m, 1);
    dim3 block(block_size, 1);

    rowAbsMaxQuantizeKernel<<<grid, block, block_size * sizeof(half)>>>(
        d_mat_ptr, d_scale_ptr, d_out_ptr, m, n, clip_ratio, block_size);
}

__global__ void asym_quantize_and_pack_i4_kernel(
    const at::Half *x_k, const at::Half* x_v, 
    uint8_t *q_k, uint8_t *q_v, 
    at::Half* scale_k, at::Half* zero_k, 
    at::Half* scale_v, at::Half* zero_v,
    int bsz, int q_len, int num_key_value_heads, int head_dim) {
    
    int batch_idx = blockIdx.x;
    int q_idx = blockIdx.y;
    int head_idx = blockIdx.z;
    int dim_idx = threadIdx.x;

    if (batch_idx >= bsz || q_idx >= q_len || head_idx >= num_key_value_heads || dim_idx >= head_dim) {
        return; // Out of bounds
    }

    int idx_k = batch_idx * q_len * num_key_value_heads * head_dim + q_idx * num_key_value_heads * head_dim + head_idx * head_dim + dim_idx;
    int idx_v = idx_k; // Same index for value as key

    // Step 1: Calculate the min and max values for quantization for both key and value states
    at::Half xmax_k = -FLT_MAX;
    at::Half xmin_k = FLT_MAX;
    at::Half xmax_v = -FLT_MAX;
    at::Half xmin_v = FLT_MAX;

    // Compute the max and min values for each (batch, q_idx, head_idx)
    for (int i = 0; i < head_dim; ++i) {
        int idx_k_temp = batch_idx * q_len * num_key_value_heads * head_dim + q_idx * num_key_value_heads * head_dim + head_idx * head_dim + i;
        int idx_v_temp = batch_idx * q_len * num_key_value_heads * head_dim + q_idx * num_key_value_heads * head_dim + head_idx * head_dim + i;
        
        at::Half val_k = x_k[idx_k_temp];
        at::Half val_v = x_v[idx_v_temp];
        
        xmax_k = __hmax(xmax_k, val_k);
        xmin_k = __hmin(xmin_k, val_k);
        xmax_v = __hmax(xmax_v, val_v);
        xmin_v = __hmin(xmin_v, val_v);
    }

    // Step 2: Calculate the scale and zero-point for both key and value states
    at::Half maxq = 15;  // 4-bit quantization
    at::Half scale_value_k = __hmax((xmax_k - xmin_k) / maxq, __float2half(1e-5f));
    at::Half zero_value_k = -xmin_k;
    at::Half scale_value_v = __hmax((xmax_v - xmin_v) / maxq, __float2half(1e-5f));
    at::Half zero_value_v = -xmin_v;

    // Step 3: Quantize and pack the values for both key and value states
    uint8_t packed_val_k = 0;
    uint8_t packed_val_v = 0;

    for (int i = 0; i < head_dim; i += 2) {
        int idx_k1 = batch_idx * q_len * num_key_value_heads * head_dim + q_idx * num_key_value_heads * head_dim + head_idx * head_dim + i;
        int idx_k2 = batch_idx * q_len * num_key_value_heads * head_dim + q_idx * num_key_value_heads * head_dim + head_idx * head_dim + i + 1;
        int idx_v1 = batch_idx * q_len * num_key_value_heads * head_dim + q_idx * num_key_value_heads * head_dim + head_idx * head_dim + i;
        int idx_v2 = batch_idx * q_len * num_key_value_heads * head_dim + q_idx * num_key_value_heads * head_dim + head_idx * head_dim + i + 1;

        // Quantize key values
        int q_k1 = roundf((x_k[idx_k1] + zero_value_k) / scale_value_k);
        int q_k2 = roundf((x_k[idx_k2] + zero_value_k) / scale_value_k);

        // Quantize value values
        int q_v1 = roundf((x_v[idx_v1] + zero_value_v) / scale_value_v);
        int q_v2 = roundf((x_v[idx_v2] + zero_value_v) / scale_value_v);

        // Clamp to the valid quantization range [0, maxq]
        q_k1 = min(max(q_k1, 0), (int)maxq);
        q_k2 = min(max(q_k2, 0), (int)maxq);
        q_v1 = min(max(q_v1, 0), (int)maxq);
        q_v2 = min(max(q_v2, 0), (int)maxq);

        // Pack into the uint8_t
        packed_val_k = (q_k2 << 4) | q_k1;
        packed_val_v = (q_v2 << 4) | q_v1;

        // Store the packed values
        int q_idx_packed_k = batch_idx * q_len * num_key_value_heads * (head_dim / 2) + q_idx * num_key_value_heads * (head_dim / 2) + head_idx * (head_dim / 2) + i / 2;
        int q_idx_packed_v = batch_idx * q_len * num_key_value_heads * (head_dim / 2) + q_idx * num_key_value_heads * (head_dim / 2) + head_idx * (head_dim / 2) + i / 2;

        q_k[q_idx_packed_k] = packed_val_k;
        q_v[q_idx_packed_v] = packed_val_v;
    }

    // Store scale and zero for both key and value
    scale_k[batch_idx * q_len * num_key_value_heads + q_idx * num_key_value_heads + head_idx] = scale_value_k;
    zero_k[batch_idx * q_len * num_key_value_heads + q_idx * num_key_value_heads + head_idx] = zero_value_k;
    scale_v[batch_idx * q_len * num_key_value_heads + q_idx * num_key_value_heads + head_idx] = scale_value_v;
    zero_v[batch_idx * q_len * num_key_value_heads + q_idx * num_key_value_heads + head_idx] = zero_value_v;
}

void asym_quantize_and_pack_i4(
    at::Tensor x_k, at::Tensor x_v, 
    at::Tensor q_k, at::Tensor q_v, 
    at::Tensor scale_k, at::Tensor zero_k, 
    at::Tensor scale_v, at::Tensor zero_v,
    int bsz, int q_len, int num_key_value_heads, int head_dim) {

    const at::Half* x_k_ptr = x_k.data_ptr<at::Half>();
    const at::Half* x_v_ptr = x_v.data_ptr<at::Half>();
    uint8_t* q_k_ptr = q_k.data_ptr<uint8_t>();
    uint8_t* q_v_ptr = q_v.data_ptr<uint8_t>();
    at::Half* scale_k_ptr = scale_k.data_ptr<at::Half>();
    at::Half* zero_k_ptr = zero_k.data_ptr<at::Half>();
    at::Half* scale_v_ptr = scale_v.data_ptr<at::Half>();
    at::Half* zero_v_ptr = zero_v.data_ptr<at::Half>();

    
    dim3 block_size(head_dim);  // One thread per dimension of head_dim
    dim3 grid_size(bsz, q_len, num_key_value_heads);

    asym_quantize_and_pack_i4_kernel<<<grid_size, block_size>>>(
        x_k_ptr, x_v_ptr, 
        q_k_ptr, q_v_ptr, 
        scale_k_ptr, zero_k_ptr, 
        scale_v_ptr, zero_v_ptr, 
        bsz, q_len, num_key_value_heads, head_dim);
    
    //cudaDeviceSynchronize(); // Ensure kernel has finished execution
}


