#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// ${generated_comment}

#include <ATen/InferSize.h>
#include <ATen/Tensor.h>
#include <ATen/native/Resize.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Operators.h>
#else
#include <ATen/ops/clone.h>
$ops_headers
#endif

namespace at {
namespace native {

// This file contains a number of kernels for aten functions that are fully code-generated.
// TODO: rename this file to something more generic.

at::Tensor clone_arg(const at::Tensor& t) {
    return t.clone();
}

std::vector<at::Tensor> clone_arg(const at::TensorList& t_list) {
    std::vector<at::Tensor> out(t_list.size());
    for (const auto& i : c10::irange(t_list.size())) {
        out[i] = t_list[i].clone();
    }
    return out;
}

void copy_arg(const at::Tensor& dst, const at::Tensor& src) {
    dst.copy_(src);
}

void copy_arg(const at::TensorList& dst, const at::TensorList& src) {
    TORCH_INTERNAL_ASSERT(dst.size() == src.size());
    for (const auto& i : c10::irange(dst.size())) {
        dst[i].copy_(src[i]);
    }
}

void resize_out_helper(const at::Tensor& dst, const at::Tensor& src) {
    at::native::resize_output(dst, src.sizes());
}

void resize_out_helper(const at::TensorList& dst, const at::TensorList& src) {
    TORCH_INTERNAL_ASSERT(dst.size() == src.size());
    for (const auto& i : c10::irange(dst.size())) {
        at::native::resize_output(dst[i], src[i].sizes());
    }
}


${CompositeViewCopyKernel_Definitions}

${GeneratedCompositeFunctional_Definitions}

${GeneratedCompositeOut_Definitions}

} // namespace native
} // namespace at
