#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// #include <ATen/native/layer_norm.h>

#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <ATen/native/cpu/mixed_data_type.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
// #include <ATen/ops/empty.h>
// #include <ATen/ops/empty_like.h>
// #include <ATen/ops/empty_like_native.h>

// #include <ATen/ops/zeros_like_native.h>
#endif

#include <array>
#include <tuple>
#include <vector>

namespace at {
namespace native {

// Can we skip having a c++ impl? i will try doing a python_dispatcher impl
at::Tensor all_reduce(at::Tensor const& self, int64_t group_id, const c10::string_view reduce_op) {
    return self;
}

} // namespace native
} // namespace at
