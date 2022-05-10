#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// ${generated_comment}

#include <ATen/Tensor.h>

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


${CompositeViewCopyKernel_Definitions}

${GeneratedCompositeFunctional_Definitions}

} // namespace native
} // namespace at
