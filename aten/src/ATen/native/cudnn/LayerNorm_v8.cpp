#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_CUDNN_ENABLED

#if AT_CUDNN_ENABLED()

#include <ATen/cudnn/cudnn-wrapper.h>

#include <c10/macros/Macros.h>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wsuggest-override")
#include <cudnn_frontend.h>
C10_DIAGNOSTIC_POP()

#include <cudnn_frontend_find_plan.h>
#include <cudnn_frontend_get_plan.h>
#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/native/cudnn/LayerNorm_v8.h>
#include <ATen/native/utils/ParamsHash.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/TensorUtils.h>

#include <c10/util/env.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <unordered_map>
#include <list>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

#ifdef __linux__
#include <dlfcn.h>
#endif

namespace at { namespace native {

void raw_cudnn_layernorm_forward_out(const Tensor& X, const Tensor& scale, const Tensor& bias, float epsilon) {
  TORCH_WARN("called");
}


}} // at::native

#endif  // AT_CUDNN_ENABLED
