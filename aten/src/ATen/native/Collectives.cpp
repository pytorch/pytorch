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

// Dummy impls required by codegen infra, not used
// These should never get called
// Defer to python impls in torch/distributed/_functional_collectives.py and _meta_registrations.py

at::Tensor all_reduce(at::Tensor const& self, const c10::string_view reduceOp, const c10::string_view tag, c10::ArrayRef<int64_t> ranks, int64_t group_size) {
    TORCH_INTERNAL_ASSERT(false);
}

at::Tensor all_gather_into_tensor(at::Tensor const& shard, const c10::string_view tag, c10::ArrayRef<int64_t> ranks, int64_t group_size) {
    TORCH_INTERNAL_ASSERT(false);
}

at::Tensor reduce_scatter_tensor(at::Tensor const& input, const c10::string_view reduceOp, int64_t scatter_dim, const c10::string_view tag, c10::ArrayRef<int64_t> ranks, int64_t group_size) {
    TORCH_INTERNAL_ASSERT(false);
}

at::Tensor wait_tensor(at::Tensor const& self) {
    TORCH_INTERNAL_ASSERT(false);
}

} // namespace native
} // namespace at
