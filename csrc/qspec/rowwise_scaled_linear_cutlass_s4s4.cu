#include <torch/library.h>

#include "rowwise_scaled_linear_cutlass.cuh"

#include "rowwise_scaled_linear_cutlass_unified.cuh"

namespace vllm {

at::Tensor
rowwise_scaled_linear_cutlass_s4s4(
    const at::Tensor& xq, const at::Tensor& x_scale, const at::Tensor& wq,
    const at::Tensor& w_scale, const at::Tensor& bias) {
  // Validate input datatypes.
  TORCH_CHECK(xq.dtype() == at::kChar && wq.dtype() == at::kChar,
              __func__, " : The input datatypes combination ", xq.dtype(),
              " for xq and ", wq.dtype(), " for wq is not supported");

  // Dispatch to appropriate kernel template.
  using ElementA = cutlass::int4b_t;
  using ElementB = cutlass::int4b_t;
  return rowwise_scaled_linear_cutlass<ElementA, ElementB>(
      xq, x_scale, wq, w_scale, bias);
}



// at::Tensor
// rowwise_scaled_linear_cutlass_awq_unified(
//     const at::Tensor& xq, const at::Tensor& x_scale, const at::Tensor& wq,
//     const at::Tensor& w_scale, const at::Tensor& bias, at::Tensor& output) {
//   // Validate input datatypes.

//   // Dispatch to appropriate kernel template.
//   using ElementA = cutlass::half_t;
//   using ElementB = cutlass::int4b_t;
//   return rowwise_scaled_linear_cutlass_unified<ElementA, ElementB>(
//       xq, x_scale, wq, w_scale, bias, output);
// }


//rowwise_scaled_linear_cutlass_unified




}  // namespace torchao
