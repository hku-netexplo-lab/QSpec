#include <torch/library.h>
#include "core/registration.h"
#include "rowwise_scaled_linear_cutlass_unified.cuh"





at::Tensor
rowwise_scaled_linear_cutlass_s4s4_unified(
    const at::Tensor& xq, const at::Tensor& x_scale, const at::Tensor& wq,
    const at::Tensor& w_scale, const at::Tensor& bias, at::Tensor& output) {
  // Validate input datatypes.
  TORCH_CHECK(xq.dtype() == at::kChar && wq.dtype() == at::kChar,
              __func__, " : The input datatypes combination ", xq.dtype(),
              " for xq and ", wq.dtype(), " for wq is not supported");

  // Dispatch to appropriate kernel template.
  using ElementA = cutlass::int4b_t;
  using ElementB = cutlass::int4b_t;
  return vllm::rowwise_scaled_linear_cutlass_unified<ElementA, ElementB>(
      xq, x_scale, wq, w_scale, bias, output);

}



// TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
//   m.impl("rowwise_scaled_linear_cutlass_s4s4_unified",&rowwise_scaled_linear_cutlass_s4s4_unified);
// }
