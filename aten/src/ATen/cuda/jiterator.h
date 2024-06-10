#pragma once
#include <ATen/jit_macros.h>

#if AT_USE_JITERATOR()

#include <c10/macros/Export.h>
#include <c10/util/SmallVector.h>
#include <ATen/core/Tensor.h>

#include <string>
#include <vector>

namespace at::cuda {

TORCH_CUDA_CPP_API c10::SmallVector<at::Tensor> CompileAndLaunchKernel(
  const std::string& code_string,
  const std::string& kernel_name,
  const int num_outputs,
  const c10::SmallVector<at::Tensor>& tensors,
  const c10::SmallVector<at::Scalar>& extra_args,
  bool return_by_ref);

} // namespace at::cuda

#else

namespace at::cuda {

TORCH_CUDA_CPP_API c10::SmallVector<at::Tensor> CompileAndLaunchKernel(
  const std::string& code_string,
  const std::string& kernel_name,
  const int num_outputs,
  const c10::SmallVector<at::Tensor>& tensors,
  const c10::SmallVector<at::Scalar>& extra_args,
  bool return_by_ref) {
    TORCH_CHECK(false, "Jiterator is not supported");
  }
} // namespace at::cuda

#endif // AT_USE_JITERATOR()
