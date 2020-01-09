#include <gtest/gtest.h>

// TODO: Move the include into `ATen/ATen.h`, once C++ tensor indexing
// is ready to ship.
#include <ATen/native/TensorIndexing.h>
#include <torch/torch.h>

#include <test/cpp/api/support.h>

using namespace torch::indexing;
using namespace torch::test;

TEST(TensorIndexingTest, Slice) {
  Slice slice(1, 2, 3);
  ASSERT_EQ(slice.start(), 1);
  ASSERT_EQ(slice.stop(), 2);
  ASSERT_EQ(slice.step(), 3);

  ASSERT_EQ(c10::str(slice), "1:2:3");
}

TEST(TensorIndexingTest, TensorIndex) {
  {
    std::vector<TensorIndex> indices = {None, "...", Ellipsis, 0, true, {1, None, 2}, torch::tensor({1, 2})};
    ASSERT_TRUE(indices[0].is_none());
    ASSERT_TRUE(indices[1].is_ellipsis());
    ASSERT_TRUE(indices[2].is_ellipsis());
    ASSERT_TRUE(indices[3].is_integer());
    ASSERT_TRUE(indices[3].integer() == 0);
    ASSERT_TRUE(indices[4].is_boolean());
    ASSERT_TRUE(indices[4].boolean() == true);
    ASSERT_TRUE(indices[5].is_slice());
    ASSERT_TRUE(indices[5].slice().start() == 1);
    ASSERT_TRUE(indices[5].slice().stop() == INDEX_MAX);
    ASSERT_TRUE(indices[5].slice().step() == 2);
    ASSERT_TRUE(indices[6].is_tensor());
    ASSERT_TRUE(torch::equal(indices[6].tensor(), torch::tensor({1, 2})));
  }

  ASSERT_THROWS_WITH(
    TensorIndex(".."),
    "Expected \"...\" to represent an ellipsis index, but got \"..\"");

  // NOTE: Some compilers such as Clang 5 and MSVC always treat `TensorIndex({1})` the same as
  // `TensorIndex(1)`. Therefore, we should only run this test on compilers that treat
  // `TensorIndex({1})` as `TensorIndex(std::initializer_list<c10::optional<int64_t>>({1}))`.
#if (!defined(__clang__) || (defined(__clang__) && __clang_major__ != 5)) && !defined(_MSC_VER)
  ASSERT_THROWS_WITH(
    TensorIndex({1}),
    "Expected 0 / 2 / 3 elements in the braced-init-list to represent a slice index, but got 1 element(s)");
#endif

  ASSERT_THROWS_WITH(
    TensorIndex({1, 2, 3, 4}),
    "Expected 0 / 2 / 3 elements in the braced-init-list to represent a slice index, but got 4 element(s)");

  {
    std::vector<TensorIndex> indices = {None, "...", Ellipsis, 0, true, {1, None, 2}};
    ASSERT_EQ(c10::str(indices), c10::str("(None, ..., ..., 0, true, 1:", INDEX_MAX, ":2)"));
    ASSERT_EQ(c10::str(indices[0]), "None");
    ASSERT_EQ(c10::str(indices[1]), "...");
    ASSERT_EQ(c10::str(indices[2]), "...");
    ASSERT_EQ(c10::str(indices[3]), "0");
    ASSERT_EQ(c10::str(indices[4]), "true");
    ASSERT_EQ(c10::str(indices[5]), c10::str("1:", INDEX_MAX, ":2"));
  }

  ASSERT_EQ(c10::str(std::vector<TensorIndex>({{}})), c10::str("(0:", INDEX_MAX, ":1)"));
  ASSERT_EQ(c10::str(std::vector<TensorIndex>({{None, None}})), c10::str("(0:", INDEX_MAX, ":1)"));
  ASSERT_EQ(c10::str(std::vector<TensorIndex>({{None, None, None}})), c10::str("(0:", INDEX_MAX, ":1)"));

  ASSERT_EQ(c10::str(std::vector<TensorIndex>({{1, None}})), c10::str("(1:", INDEX_MAX, ":1)"));
  ASSERT_EQ(c10::str(std::vector<TensorIndex>({{1, None, None}})), c10::str("(1:", INDEX_MAX, ":1)"));
  ASSERT_EQ(c10::str(std::vector<TensorIndex>({{None, 3}})), c10::str("(0:3:1)"));
  ASSERT_EQ(c10::str(std::vector<TensorIndex>({{None, 3, None}})), c10::str("(0:3:1)"));
  ASSERT_EQ(c10::str(std::vector<TensorIndex>({{None, None, 2}})), c10::str("(0:", INDEX_MAX, ":2)"));
  ASSERT_EQ(c10::str(std::vector<TensorIndex>({{None, None, -1}})), c10::str("(", INDEX_MAX, ":", INDEX_MIN, ":-1)"));

  ASSERT_EQ(c10::str(std::vector<TensorIndex>({{1, 3}})), c10::str("(1:3:1)"));
  ASSERT_EQ(c10::str(std::vector<TensorIndex>({{1, None, 2}})), c10::str("(1:", INDEX_MAX, ":2)"));
  ASSERT_EQ(c10::str(std::vector<TensorIndex>({{1, None, -1}})), c10::str("(1:", INDEX_MIN, ":-1)"));
  ASSERT_EQ(c10::str(std::vector<TensorIndex>({{None, 3, 2}})), c10::str("(0:3:2)"));
  ASSERT_EQ(c10::str(std::vector<TensorIndex>({{None, 3, -1}})), c10::str("(", INDEX_MAX, ":3:-1)"));

  ASSERT_EQ(c10::str(std::vector<TensorIndex>({{1, 3, 2}})), c10::str("(1:3:2)"));
}

// TODO: I will remove the Python tests in the comments once the PR is approved.

/*
class TestIndexing(TestCase):
    def test_single_int(self):
        v = torch.randn(5, 7, 3)
        self.assertEqual(v[4].shape, (7, 3))
*/
TEST(TensorIndexingTest, TestSingleInt) {
  auto v = torch::randn({5, 7, 3});
  ASSERT_EQ(v.idx({4}).sizes(), torch::IntArrayRef({7, 3}));
}

/*
    def test_multiple_int(self):
        v = torch.randn(5, 7, 3)
        self.assertEqual(v[4].shape, (7, 3))
        self.assertEqual(v[4, :, 1].shape, (7,))
*/
TEST(TensorIndexingTest, TestMultipleInt) {
  auto v = torch::randn({5, 7, 3});
  ASSERT_EQ(v.idx({4}).sizes(), torch::IntArrayRef({7, 3}));
  ASSERT_EQ(v.idx({4, {}, 1}).sizes(), torch::IntArrayRef({7}));

  // To show that `.idx_put_` works
  v.idx_put_({4, 3, 1}, 0);
  ASSERT_EQ(v.idx({4, 3, 1}).item<double>(), 0);
}

/*
    def test_none(self):
        v = torch.randn(5, 7, 3)
        self.assertEqual(v[None].shape, (1, 5, 7, 3))
        self.assertEqual(v[:, None].shape, (5, 1, 7, 3))
        self.assertEqual(v[:, None, None].shape, (5, 1, 1, 7, 3))
        self.assertEqual(v[..., None].shape, (5, 7, 3, 1))
*/
TEST(TensorIndexingTest, TestNone) {
  auto v = torch::randn({5, 7, 3});
  ASSERT_EQ(v.idx({None}).sizes(), torch::IntArrayRef({1, 5, 7, 3}));
  ASSERT_EQ(v.idx({{}, None}).sizes(), torch::IntArrayRef({5, 1, 7, 3}));
  ASSERT_EQ(v.idx({{}, None, None}).sizes(), torch::IntArrayRef({5, 1, 1, 7, 3}));
  ASSERT_EQ(v.idx({"...", None}).sizes(), torch::IntArrayRef({5, 7, 3, 1}));
}

/*
    def test_step(self):
        v = torch.arange(10)
        self.assertEqual(v[::1], v)
        self.assertEqual(v[::2].tolist(), [0, 2, 4, 6, 8])
        self.assertEqual(v[::3].tolist(), [0, 3, 6, 9])
        self.assertEqual(v[::11].tolist(), [0])
        self.assertEqual(v[1:6:2].tolist(), [1, 3, 5])
*/
TEST(TensorIndexingTest, TestStep) {
  auto v = torch::arange(10);
  assert_tensor_equal(v.idx({{None, None, 1}}), v);
  assert_tensor_equal(v.idx({{None, None, 2}}), torch::tensor({0, 2, 4, 6, 8}));
  assert_tensor_equal(v.idx({{None, None, 3}}), torch::tensor({0, 3, 6, 9}));
  assert_tensor_equal(v.idx({{None, None, 11}}), torch::tensor({0}));
  assert_tensor_equal(v.idx({{1, 6, 2}}), torch::tensor({1, 3, 5}));
}

/*
    def test_step_assignment(self):
        v = torch.zeros(4, 4)
        v[0, 1::2] = torch.tensor([3., 4.])
        self.assertEqual(v[0].tolist(), [0, 3, 0, 4])
        self.assertEqual(v[1:].sum(), 0)
*/
TEST(TensorIndexingTest, TestStepAssignment) {
  auto v = torch::zeros({4, 4});
  v.idx_put_({0, {1, None, 2}}, torch::tensor({3., 4.}));
  assert_tensor_equal(v.idx({0}), torch::tensor({0., 3., 0., 4.}));
  assert_tensor_equal(v.idx({{1, None}}).sum(), 0);
}

/*
    def test_bool_indices(self):
        v = torch.randn(5, 7, 3)
        boolIndices = torch.tensor([True, False, True, True, False], dtype=torch.bool)
        self.assertEqual(v[boolIndices].shape, (3, 7, 3))
        self.assertEqual(v[boolIndices], torch.stack([v[0], v[2], v[3]]))

        v = torch.tensor([True, False, True], dtype=torch.bool)
        boolIndices = torch.tensor([True, False, False], dtype=torch.bool)
        uint8Indices = torch.tensor([1, 0, 0], dtype=torch.uint8)
        with warnings.catch_warnings(record=True) as w:
            self.assertEqual(v[boolIndices].shape, v[uint8Indices].shape)
            self.assertEqual(v[boolIndices], v[uint8Indices])
            self.assertEqual(v[boolIndices], tensor([True], dtype=torch.bool))
            self.assertEquals(len(w), 2)
*/
TEST(TensorIndexingTest, TestBoolIndices) {
  {
    auto v = torch::randn({5, 7, 3});
    auto boolIndices = torch::tensor({true, false, true, true, false}, torch::kBool);
    ASSERT_EQ(v.idx({boolIndices}).sizes(), torch::IntArrayRef({3, 7, 3}));
    assert_tensor_equal(v.idx({boolIndices}), torch::stack({v.idx({0}), v.idx({2}), v.idx({3})}));
  }
  {
    auto v = torch::tensor({true, false, true}, torch::kBool);
    auto boolIndices = torch::tensor({true, false, false}, torch::kBool);
    auto uint8Indices = torch::tensor({1, 0, 0}, torch::kUInt8);

    {
      std::stringstream buffer;
      CerrRedirect cerr_redirect(buffer.rdbuf());

      ASSERT_EQ(v.idx({boolIndices}).sizes(), v.idx({uint8Indices}).sizes());
      assert_tensor_equal(v.idx({boolIndices}), v.idx({uint8Indices}));
      assert_tensor_equal(v.idx({boolIndices}), torch::tensor({true}, torch::kBool));

      ASSERT_EQ(count_substr_occurrences(buffer.str(), "indexing with dtype torch.uint8 is now deprecated"), 2);
    }
  }
}
