#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/impl/Arithmetic.h>

#include <ATen/native/vulkan/graph/Graph.h>

namespace at {
namespace native {
namespace vulkan {

void add_arithmetic_node(
    ComputeGraph& graph,
    const ValueRef t1,
    const ValueRef t2,
    const ValueRef out,
    const float alpha,
    const arithmetic::OpType optype);

ValueRef add_arithmetic_node(
    ComputeGraph& graph,
    const ValueRef t1,
    const ValueRef t2,
    const float alpha,
    const arithmetic::OpType optype);

class ArithmeticPrepack : public virtual OpNode {
 public:
  explicit ArithmeticPrepack(const ValueRef tref, const ValueRef packed);

  void encode_prepack(ComputeGraph* graph) const override;
};

class ArithmeticNode : public virtual OpNode {
 public:
  explicit ArithmeticNode(
      const ValueRef t1,
      const ValueRef t2,
      const ValueRef out,
      const float alpha,
      const arithmetic::OpType optype);

  void encode_execute(ComputeGraph* graph) const override;

 private:
  float alpha_;
  arithmetic::OpType optype_;
};

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
