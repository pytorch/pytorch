#include <gtest/gtest.h>

#include <c10/util/Exception.h>
#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/ir_metadata.h>
#include <torch/csrc/lazy/core/ir_util.h>

namespace torch {
namespace lazy {

class IrUtilNode : public Node {
 public:
  explicit IrUtilNode()
      : Node(OpKind(), /* num_outputs */ 1, /* hash_func */ [&](bool /*bakeInSizes*/) -> hash_t { return Hash(0); }) {}
  ~IrUtilNode() override = default;

  void AddOperand(Value v) {
    if (!v.node) {
      return;
    }
    operands_as_outputs_.emplace_back(v.node.get(), v.index);
    operands_.push_back(std::move(v.node));
  }

  const std::vector<Output>& operands() const override {
    return operands_as_outputs_;
  }

  const Output& operand(size_t i) const override {
    return operands_as_outputs_.at(i);
  }

  const Shape& shape(size_t i) const override { return shape_; }
  c10::ArrayRef<Shape> shapes() const override { return {shape_}; }

 private:
  std::vector<NodePtr> operands_;
  std::vector<Output> operands_as_outputs_;
  Shape shape_;
};

/*  a
 * / \
 *b   c
 * \ /
 *  d
 * Post-order: d c b a
 */
TEST(IrUtilTest, BasicTest) {
  NodePtr a = MakeNode<IrUtilNode>();
  NodePtr b = MakeNode<IrUtilNode>();
  NodePtr c = MakeNode<IrUtilNode>();
  NodePtr d = MakeNode<IrUtilNode>();

  dynamic_cast<IrUtilNode*>(a.get())->AddOperand(Value(b, 0));
  dynamic_cast<IrUtilNode*>(a.get())->AddOperand(Value(c, 1));
  dynamic_cast<IrUtilNode*>(b.get())->AddOperand(Value(d, 0));
  dynamic_cast<IrUtilNode*>(c.get())->AddOperand(Value(d, 0));

  std::vector<Node*> postorder = Util::ComputePostOrder({a.get()});
  EXPECT_EQ(postorder.size(), 4);
  EXPECT_EQ(postorder.at(0), d.get());
  EXPECT_EQ(postorder.at(1), c.get());
  EXPECT_EQ(postorder.at(2), b.get());
  EXPECT_EQ(postorder.at(3), a.get());
}

/*  a
 * / \
 *b---c
 * Post-order: not valid
 */
TEST(IrUtilTest, TestCircle) {
  NodePtr a = MakeNode<IrUtilNode>();
  NodePtr b = MakeNode<IrUtilNode>();
  NodePtr c = MakeNode<IrUtilNode>();

  dynamic_cast<IrUtilNode*>(a.get())->AddOperand(Value(b, 0));
  dynamic_cast<IrUtilNode*>(b.get())->AddOperand(Value(c, 0));
  dynamic_cast<IrUtilNode*>(c.get())->AddOperand(Value(a, 0));

  EXPECT_THROW(Util::ComputePostOrder({a.get()}), c10::Error);
}

} // namespace lazy
} // namespace torch
