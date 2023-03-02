#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#endif

namespace at {
namespace native {

// Dummy impl required by codegen infra, not used
at::Tensor all_reduce(at::Tensor const& self, const c10::string_view reduceOp, const c10::string_view tag, c10::ArrayRef<int64_t> ranks, int64_t group_size) {
    // This should never get called
    // Defer to python impls in torch/distributed/_functional_collectives.py and _meta_registrations.py
    TORCH_INTERNAL_ASSERT(false);
}

at::Tensor wait_tensor(at::Tensor const& self) {
    // This should never get called
    // Defer to python impls in torch/distributed/_functional_collectives.py and _meta_registrations.py
    TORCH_INTERNAL_ASSERT(false);
}

} // namespace native
} // namespace at
