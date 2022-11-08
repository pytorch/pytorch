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

// duped with gen_resize_out_helper from structured kernels
void copy_arg(const at::Tensor& dst, const at::Tensor& src) {
    TORCH_CHECK(src.dtype() == dst.dtype(),
        "Expected out tensor to have dtype ", src.dtype(), ", but got ", dst.dtype(), " instead");
    TORCH_CHECK(src.device() == dst.device(),
        "Expected out tensor to have device ", src.device(), ", but got ", dst.device(), " instead");
    dst.copy_(src);
}

void copy_arg(const at::TensorList& dst, const at::TensorList& src) {
    TORCH_INTERNAL_ASSERT(dst.size() == src.size());
    for (const auto& i : c10::irange(dst.size())) {
        copy_arg(dst[i], src[i]);
    }
}

// TODO: this doesn't handle restriding empty tensors correctly; see
// gen_resize_out_helper for the correct algorithm

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
