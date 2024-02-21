#include <ATen/native/vulkan/impl/Arithmetic.h>
#include <ATen/native/vulkan/impl/Common.h>

#include <ATen/native/vulkan/graph/Functions.h>

#include <ATen/native/vulkan/graph/ops/Arithmetic.h>
#include <ATen/native/vulkan/graph/ops/Staging.h>

namespace at {
namespace native {
namespace vulkan {

#define DEFINE_ARITHMETIC_FN(function, op_type)                               \
  ValueRef function(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    return add_arithmetic_node(                                               \
        graph,                                                                \
        args[0],                                                              \
        args[1],                                                              \
        args[2],                                                              \
        arithmetic::OpType::op_type,                                          \
        args[3]);                                                             \
  }

DEFINE_ARITHMETIC_FN(add, ADD);
DEFINE_ARITHMETIC_FN(sub, SUB);
DEFINE_ARITHMETIC_FN(mul, MUL);
DEFINE_ARITHMETIC_FN(div, DIV);
DEFINE_ARITHMETIC_FN(floor_div, FLOOR_DIV);
DEFINE_ARITHMETIC_FN(pow, POW);

} // namespace vulkan
} // namespace native
} // namespace at
