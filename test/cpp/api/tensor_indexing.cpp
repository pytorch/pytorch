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
  // `TensorIndex(1)`. This is in violation of the C++ standard
  // (`https://en.cppreference.com/w/cpp/language/list_initialization`), which says:
  // ```
  // copy-list-initialization:
  //
  // U( { arg1, arg2, ... } )
  //
  // functional cast expression or other constructor invocations, where braced-init-list is used
  // in place of a constructor argument. Copy-list-initialization initializes the constructor's parameter
  // (note; the type U in this example is not the type that's being list-initialized; U's constructor's parameter is)
  // ```
  // When we call `TensorIndex({1})`, `TensorIndex`'s constructor's parameter is being list-initialized with {1}.
  // And since we have the `TensorIndex(std::initializer_list<c10::optional<int64_t>>)` constructor, the following
  // rule in the standard applies:
  // ```
  // The effects of list initialization of an object of type T are:
  //
  // if T is a specialization of std::initializer_list, the T object is direct-initialized or copy-initialized,
  // depending on context, from a prvalue of the same type initialized from the braced-init-list.
  // ```
  // Therefore, if the compiler strictly follows the standard, it should treat `TensorIndex({1})` as
  // `TensorIndex(std::initializer_list<c10::optional<int64_t>>({1}))`. However, this is not the case for
  // compilers such as Clang 5 and MSVC, and hence we skip this test for those compilers.
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
  ASSERT_EQ(v.index({4}).sizes(), torch::IntArrayRef({7, 3}));
}

/*
    def test_multiple_int(self):
        v = torch.randn(5, 7, 3)
        self.assertEqual(v[4].shape, (7, 3))
        self.assertEqual(v[4, :, 1].shape, (7,))
*/
TEST(TensorIndexingTest, TestMultipleInt) {
  auto v = torch::randn({5, 7, 3});
  ASSERT_EQ(v.index({4}).sizes(), torch::IntArrayRef({7, 3}));
  ASSERT_EQ(v.index({4, {}, 1}).sizes(), torch::IntArrayRef({7}));

  // To show that `.index_put_` works
  v.index_put_({4, 3, 1}, 0);
  ASSERT_EQ(v.index({4, 3, 1}).item<double>(), 0);
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
  ASSERT_EQ(v.index({None}).sizes(), torch::IntArrayRef({1, 5, 7, 3}));
  ASSERT_EQ(v.index({{}, None}).sizes(), torch::IntArrayRef({5, 1, 7, 3}));
  ASSERT_EQ(v.index({{}, None, None}).sizes(), torch::IntArrayRef({5, 1, 1, 7, 3}));
  ASSERT_EQ(v.index({"...", None}).sizes(), torch::IntArrayRef({5, 7, 3, 1}));
}
