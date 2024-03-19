#include <gtest/gtest.h>

#include <c10/util/Exception.h>
#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/ir_builder.h>
#include <torch/csrc/lazy/core/ir_metadata.h>
#include <torch/csrc/lazy/core/ir_util.h>
#include <memory>

namespace torch {
namespace lazy {

class TrieCacheNode : public Node {
 public:
  static OpKind ClassOpKind() {
    return OpKind();
  }

  explicit TrieCacheNode(size_t id)
      : Node(ClassOpKind(), /* num_outputs */ 1), id_(id), hash_(Hash(id_)) {}
  ~TrieCacheNode() override = default;

  bool CanBeReused(size_t id) const {
    return (id_ == id);
  }

  void AddOperand(Value v) {
    if (!v.node) {
      return;
    }
    operands_as_outputs_.emplace_back(v.node.get(), v.index);
    operands_.push_back(std::move(v.node));
  }

  hash_t hash() const override {
    return hash_;
  }
  hash_t shapeHash() const override {
    return hash_;
  }

 private:
  size_t id_;
  hash_t hash_;
};

TEST(TrieCacheTest, TestSinglePath) {
  FLAGS_torch_lazy_reuse_ir = true;
  TrieCache::Get()->Clear();

  NodePtr a = ReuseOrMakeNode<TrieCacheNode>(0);
  NodePtr b = ReuseOrMakeNode<TrieCacheNode>(1);
  NodePtr c = ReuseOrMakeNode<TrieCacheNode>(2);
  TrieCache::Get()->ResetCurrent(); // MarkStep

  EXPECT_EQ(ReuseOrMakeNode<TrieCacheNode>(0).get(), a.get());
  EXPECT_EQ(ReuseOrMakeNode<TrieCacheNode>(1).get(), b.get());
  EXPECT_EQ(ReuseOrMakeNode<TrieCacheNode>(2).get(), c.get());
  TrieCache::Get()->ResetCurrent(); // MarkStep
}

/*
 *    0
 *    |
 *    1
 *   / \
 *  2   3
 */
TEST(TrieCacheTest, TestTwoPaths) {
  FLAGS_torch_lazy_reuse_ir = true;
  TrieCache::Get()->Clear();

  NodePtr a = ReuseOrMakeNode<TrieCacheNode>(0);
  NodePtr b = ReuseOrMakeNode<TrieCacheNode>(1);
  NodePtr c = ReuseOrMakeNode<TrieCacheNode>(2);
  TrieCache::Get()->ResetCurrent(); // MarkStep

  EXPECT_EQ(ReuseOrMakeNode<TrieCacheNode>(0).get(), a.get());
  EXPECT_EQ(ReuseOrMakeNode<TrieCacheNode>(1).get(), b.get());
  NodePtr d = ReuseOrMakeNode<TrieCacheNode>(3);
  EXPECT_NE(d.get(), c.get());
  TrieCache::Get()->ResetCurrent(); // MarkStep

  EXPECT_EQ(ReuseOrMakeNode<TrieCacheNode>(0).get(), a.get());
  EXPECT_EQ(ReuseOrMakeNode<TrieCacheNode>(1).get(), b.get());
  EXPECT_EQ(ReuseOrMakeNode<TrieCacheNode>(3).get(), d.get());
  TrieCache::Get()->ResetCurrent(); // MarkStep

  EXPECT_EQ(ReuseOrMakeNode<TrieCacheNode>(0).get(), a.get());
  EXPECT_EQ(ReuseOrMakeNode<TrieCacheNode>(1).get(), b.get());
  EXPECT_EQ(ReuseOrMakeNode<TrieCacheNode>(2).get(), c.get());
  TrieCache::Get()->ResetCurrent(); // MarkStep
}

} // namespace lazy
} // namespace torch
