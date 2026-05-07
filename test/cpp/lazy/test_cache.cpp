#include <gtest/gtest.h>

#include <c10/util/Exception.h>
#include <torch/csrc/lazy/core/cache.h>
#include <torch/csrc/lazy/core/hash.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/shape.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

class CacheNode : public Node {
 public:
  explicit CacheNode(const std::string& str)
      : Node(OpKind(), /* num_outputs */ 1), hash_(Hash(str)), str_(str) {}
  ~CacheNode() override = default;

  const std::vector<Output>& operands() const override {
    TORCH_INTERNAL_ASSERT(false, "Can't access operands of test node");
  }

  const Output& operand(size_t i) const override {
    TORCH_INTERNAL_ASSERT(false, "Can't access operand[i] of test node");
  }

  hash_t hash() const override {
    return hash_;
  }
  hash_t shapeHash() const override {
    return hash_;
  }

 private:
  hash_t hash_;
  std::string str_;
};

TEST(CacheTest, BasicTest) {
  std::shared_ptr<CacheNode> a = std::make_shared<CacheNode>("a");
  std::shared_ptr<CacheNode> b = std::make_shared<CacheNode>("b");
  std::shared_ptr<CacheNode> c = std::make_shared<CacheNode>("c");
  Cache<hash_t, CacheNode, HashReducer> cache(2);

  cache.Add(a->hash(), a);
  EXPECT_EQ(cache.Get(a->hash()), a);
  EXPECT_EQ(cache.Get(b->hash()), nullptr);
  EXPECT_EQ(cache.Get(c->hash()), nullptr);

  cache.Add(b->hash(), b);
  EXPECT_EQ(cache.Get(a->hash()), a);
  EXPECT_EQ(cache.Get(b->hash()), b);
  EXPECT_EQ(cache.Get(c->hash()), nullptr);

  cache.Add(c->hash(), c);
  EXPECT_EQ(cache.Get(a->hash()), nullptr); // a has been evicted
  EXPECT_EQ(cache.Get(b->hash()), b);
  EXPECT_EQ(cache.Get(c->hash()), c);

  cache.Erase(c->hash());
  EXPECT_EQ(cache.Get(a->hash()), nullptr);
  EXPECT_EQ(cache.Get(b->hash()), b);
  EXPECT_EQ(cache.Get(c->hash()), nullptr); // c has been removed

  cache.Clear();
  EXPECT_EQ(cache.Get(a->hash()), nullptr);
  EXPECT_EQ(cache.Get(b->hash()), nullptr);
  EXPECT_EQ(cache.Get(c->hash()), nullptr);
}

class CacheNodeWithShape : public TsNode {
 public:
  explicit CacheNodeWithShape(const Shape& shape)
      : TsNode(OpKind(), shape, /* num_outputs */ 1, /* seed */ 0) {}
};

TEST(CacheTest, ShapeCacheTestForDynamicShape) {
  // enable dynamic shape
  FLAGS_ltc_enable_dynamic_shapes = true;

  CacheNodeWithShape nodes[] = {
      CacheNodeWithShape(Shape(c10::kFloat, {2, 4})),
      CacheNodeWithShape(Shape(c10::kFloat, {4, 2}))};

  /*
   * Make sure the cached shape for node (2, 4) is not used for node (4, 2)
   */
  for (auto& node : nodes) {
    EXPECT_EQ(node.shape(), node.computeShape([&]() { return node.shape(); }));
  }

  // reset the flag
  FLAGS_ltc_enable_dynamic_shapes = false;
}

} // namespace lazy
} // namespace torch
