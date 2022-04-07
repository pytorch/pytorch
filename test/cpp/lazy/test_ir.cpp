#include <gtest/gtest.h>

#include <torch/csrc/lazy/generated/LazyIr.h>
#include <c10/util/Exception.h>
#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/ir_metadata.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

class TestLeafNode : public Node {
 public:
  explicit TestLeafNode(size_t param)
      : Node(OpKind(), /* num_outputs */ 1, /* hash_func */[&](bool /*bakeInSizes*/) -> hash_t { return Hash(param); }),
        param_(param) {}
  ~TestLeafNode() override = default;

  const std::vector<Output>& operands() const override {
    TORCH_INTERNAL_ASSERT(false, "Can't access operands of leaf node");
  }

  const Output& operand(size_t i) const override {
    TORCH_INTERNAL_ASSERT(false, "Can't access operand[i] of leaf node");
  }
  const Shape& shape(size_t i) const override { return shape_; }
  c10::ArrayRef<Shape> shapes() const override { return {shape_}; }
 private:
  size_t param_;
  Shape shape_;
};

TEST(IrTest, BasicTest) {
  NodePtr node1 = MakeNode<TestLeafNode>(1);
  NodePtr node2 = MakeNode<TestLeafNode>(2);
  EXPECT_NE(node1->hash(), node2->hash());

  EXPECT_EQ(node1->num_outputs(), 1);

  const TestLeafNode* leafptr = NodeCast<TestLeafNode>(node1.get(), OpKind());
  EXPECT_TRUE(leafptr != nullptr);
}

TEST(IrTest, MetaDataTest) {
  bool restore_FLAGS_torch_lazy_ir_debug = FLAGS_torch_lazy_ir_debug;
  FLAGS_torch_lazy_ir_debug = false;
  NodePtr node = MakeNode<TestLeafNode>(1);
  auto metaWithoutDebug = node->metadata();
  EXPECT_EQ(metaWithoutDebug.scope.size(), 0);
  EXPECT_EQ(metaWithoutDebug.frame_info.size(), 0);

  FLAGS_torch_lazy_ir_debug = true;
  node = MakeNode<TestLeafNode>(1);
  auto metaWithEmptyDebug = node->metadata();
  EXPECT_EQ(metaWithEmptyDebug.scope.size(), 0);
  EXPECT_EQ(metaWithEmptyDebug.frame_info.size(), 0);

  {
    ScopePusher scope("TestScope");
    node = MakeNode<TestLeafNode>(1);
    auto metaWithScope = node->metadata();
    EXPECT_EQ(metaWithScope.scope, "TestScope.1");
    EXPECT_EQ(metaWithScope.frame_info.size(), 0);
  }

  SourceLocation dummySourceLocation;
  dummySourceLocation.file = "file";
  dummySourceLocation.function = "function";
  dummySourceLocation.line = 10;
  RegisterGetFrameInfo(
      [&]() -> std::vector<SourceLocation> { return {dummySourceLocation}; });
  node = MakeNode<TestLeafNode>(1);
  auto metaWithSourceLoc = node->metadata();
  EXPECT_EQ(metaWithSourceLoc.scope.size(), 0);
  EXPECT_EQ(metaWithSourceLoc.frame_info.size(), 1);
  EXPECT_EQ(metaWithSourceLoc.frame_info[0].file, "file");
  EXPECT_EQ(metaWithSourceLoc.frame_info[0].function, "function");
  EXPECT_EQ(metaWithSourceLoc.frame_info[0].line, 10);
  FLAGS_torch_lazy_ir_debug = restore_FLAGS_torch_lazy_ir_debug;
}

TEST(IrTest, TsNode) {
  NodePtr node1 = MakeNode<TsNode>(
      OpKind(at::aten::view),
      Shape(),
      /*num_outputs*/ 1,
      /*hash_seed*/ kHashSeed);
  NodePtr node2 = MakeNode<TsNode>(
      OpKind(at::aten::view),
      Shape(),
      /*num_outputs*/ 1,
      /*hash_seed*/ kHashSeed);
  EXPECT_EQ(node1->hash(), node2->hash());

  EXPECT_EQ(node1->num_outputs(), 1);

  const TsNode* leafptr = NodeCast<TsNode>(node1.get(), OpKind(at::aten::view));
  EXPECT_TRUE(leafptr != nullptr);
}

} // namespace lazy
} // namespace torch
