#include <gtest/gtest.h>

#include <c10/util/Exception.h>
#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/ir_metadata.h>
#include <torch/csrc/lazy/core/ir_util.h>

namespace torch {
namespace lazy {

class TestNode : public Node {
 public:
  explicit TestNode()
      : Node(OpKind(), /* num_outputs */ 1, /* hash_seed */ Hash("")) {}
  ~TestNode() override {}

  void AddOperand(Value v) {
    if (!v.node) {
      return;
    }
    operands_as_outputs_.push_back(Output(v.node.get(), v.index));
    operands_.push_back(std::move(v.node));
  }

  const std::vector<Output>& operands() const override {
    return operands_as_outputs_;
  }

  const Output& operand(size_t i) const override {
    return operands_as_outputs_.at(i);
  }

 private:
  std::vector<NodePtr> operands_;
  std::vector<Output> operands_as_outputs_;
};

//   a
//  / \
// b   c
//  \ /
//   d
// Post-order: d c b a
TEST(IrUtilTest, BasicTest) {
  NodePtr a = MakeNode<TestNode>();
  NodePtr b = MakeNode<TestNode>();
  NodePtr c = MakeNode<TestNode>();
  NodePtr d = MakeNode<TestNode>();

  dynamic_cast<TestNode*>(a.get())->AddOperand(Value(b, 0));
  dynamic_cast<TestNode*>(a.get())->AddOperand(Value(c, 1));
  dynamic_cast<TestNode*>(b.get())->AddOperand(Value(d, 0));
  dynamic_cast<TestNode*>(c.get())->AddOperand(Value(d, 0));

  std::vector<Node*> postorder = Util::ComputePostOrder({a.get()});
  EXPECT_EQ(postorder.size(), 4);
  EXPECT_EQ(postorder.at(0), d.get());
  EXPECT_EQ(postorder.at(1), c.get());
  EXPECT_EQ(postorder.at(2), b.get());
  EXPECT_EQ(postorder.at(3), a.get());
}

//   a
//  / \
// b---c
TEST(IrUtilTest, TestCircle) {
  NodePtr a = MakeNode<TestNode>();
  NodePtr b = MakeNode<TestNode>();
  NodePtr c = MakeNode<TestNode>();

  dynamic_cast<TestNode*>(a.get())->AddOperand(Value(b, 0));
  dynamic_cast<TestNode*>(b.get())->AddOperand(Value(c, 0));
  dynamic_cast<TestNode*>(c.get())->AddOperand(Value(a, 0));

  EXPECT_DEBUG_DEATH(Util::ComputePostOrder({a.get()}), "Graph loop found at");
}

} // namespace lazy
} // namespace torch
