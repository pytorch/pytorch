#pragma once
#include <ATen/jit_macros.h>

#if AT_USE_JITERATOR()

#include <c10/macros/Export.h>
#include <ATen/core/Tensor.h>

#include <string>
#include <vector>

namespace at {
namespace cuda {

TORCH_CUDA_CPP_API at::Tensor CompileAndLaunchKernel(
  const std::string& code_string,
  const std::string& kernel_name,
  const std::vector<at::Tensor>& tensors,
  const std::vector<at::Scalar>& extra_args);

}} // namespace at::cuda

#else

namespace at { namespace cuda {
TORCH_CUDA_CPP_API at::Tensor CompileAndLaunchKernel(
  const std::string& code_string,
  const std::string& kernel_name,
  const std::vector<at::Tensor>& tensors,
  const std::vector<at::Scalar>& extra_args) {
    TORCH_CHECK(false, "Jiterator is not supported on ROCm");
  }
}} // namespace at::cuda

#endif // AT_USE_JITERATOR()
