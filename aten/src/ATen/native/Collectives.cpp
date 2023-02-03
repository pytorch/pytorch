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

// Can we skip having a c++ impl? i will try doing a python_dispatcher impl
at::Tensor all_reduce(at::Tensor const& self, const c10::string_view reducePp, const c10::string_view tag, c10::ArrayRef<int64_t> ranks, int64_t stride) {
    // This should never get called
    // Defer to python impls in traceable_collectives.py and _meta_registrations.py
    TORCH_INTERNAL_ASSERT(false);
}

} // namespace native
} // namespace at