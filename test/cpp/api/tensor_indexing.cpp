#include <gtest/gtest.h>

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
    std::vector<TensorIndex> indices = {
        None,
        "...",
        Ellipsis,
        0,
        true,
        Slice(1, None, 2),
        torch::tensor({1, 2})};
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

  {
    std::vector<TensorIndex> indices = {
        None, "...", Ellipsis, 0, true, Slice(1, None, 2)};
    ASSERT_EQ(
        c10::str(indices),
        c10::str("(None, ..., ..., 0, true, 1:", INDEX_MAX, ":2)"));
    ASSERT_EQ(c10::str(indices[0]), "None");
    ASSERT_EQ(c10::str(indices[1]), "...");
    ASSERT_EQ(c10::str(indices[2]), "...");
    ASSERT_EQ(c10::str(indices[3]), "0");
    ASSERT_EQ(c10::str(indices[4]), "true");
    ASSERT_EQ(c10::str(indices[5]), c10::str("1:", INDEX_MAX, ":2"));
  }

  ASSERT_EQ(
      c10::str(std::vector<TensorIndex>({Slice()})),
      c10::str("(0:", INDEX_MAX, ":1)"));
  ASSERT_EQ(
      c10::str(std::vector<TensorIndex>({Slice(None, None)})),
      c10::str("(0:", INDEX_MAX, ":1)"));
  ASSERT_EQ(
      c10::str(std::vector<TensorIndex>({Slice(None, None, None)})),
      c10::str("(0:", INDEX_MAX, ":1)"));

  ASSERT_EQ(
      c10::str(std::vector<TensorIndex>({Slice(1, None)})),
      c10::str("(1:", INDEX_MAX, ":1)"));
  ASSERT_EQ(
      c10::str(std::vector<TensorIndex>({Slice(1, None, None)})),
      c10::str("(1:", INDEX_MAX, ":1)"));
  ASSERT_EQ(
      c10::str(std::vector<TensorIndex>({Slice(None, 3)})),
      c10::str("(0:3:1)"));
  ASSERT_EQ(
      c10::str(std::vector<TensorIndex>({Slice(None, 3, None)})),
      c10::str("(0:3:1)"));
  ASSERT_EQ(
      c10::str(std::vector<TensorIndex>({Slice(None, None, 2)})),
      c10::str("(0:", INDEX_MAX, ":2)"));
  ASSERT_EQ(
      c10::str(std::vector<TensorIndex>({Slice(None, None, -1)})),
      c10::str("(", INDEX_MAX, ":", INDEX_MIN, ":-1)"));

  ASSERT_EQ(
      c10::str(std::vector<TensorIndex>({Slice(1, 3)})), c10::str("(1:3:1)"));
  ASSERT_EQ(
      c10::str(std::vector<TensorIndex>({Slice(1, None, 2)})),
      c10::str("(1:", INDEX_MAX, ":2)"));
  ASSERT_EQ(
      c10::str(std::vector<TensorIndex>({Slice(1, None, -1)})),
      c10::str("(1:", INDEX_MIN, ":-1)"));
  ASSERT_EQ(
      c10::str(std::vector<TensorIndex>({Slice(None, 3, 2)})),
      c10::str("(0:3:2)"));
  ASSERT_EQ(
      c10::str(std::vector<TensorIndex>({Slice(None, 3, -1)})),
      c10::str("(", INDEX_MAX, ":3:-1)"));

  ASSERT_EQ(
      c10::str(std::vector<TensorIndex>({Slice(1, 3, 2)})),
      c10::str("(1:3:2)"));
}

TEST(TensorIndexingTest, TestNoIndices) {
  torch::Tensor tensor = torch::randn({20, 20});
  torch::Tensor value = torch::randn({20, 20});
  std::vector<TensorIndex> indices;

  ASSERT_THROWS_WITH(
      tensor.index({}),
      "Passing an empty index list to Tensor::index() is not valid syntax");
  ASSERT_THROWS_WITH(
      tensor.index_put_({}, 1),
      "Passing an empty index list to Tensor::index_put_() is not valid syntax");
  ASSERT_THROWS_WITH(
      tensor.index_put_({}, value),
      "Passing an empty index list to Tensor::index_put_() is not valid syntax");

  ASSERT_THROWS_WITH(
      tensor.index(indices),
      "Passing an empty index list to Tensor::index() is not valid syntax");
  ASSERT_THROWS_WITH(
      tensor.index_put_(indices, 1),
      "Passing an empty index list to Tensor::index_put_() is not valid syntax");
  ASSERT_THROWS_WITH(
      tensor.index_put_(indices, value),
      "Passing an empty index list to Tensor::index_put_() is not valid syntax");
}

TEST(TensorIndexingTest, TestAdvancedIndexingWithListOfTensor) {
  {
    torch::Tensor tensor = torch::randn({20, 20});
    torch::Tensor index = torch::arange(10, torch::kLong).cpu();
    torch::Tensor result = at::index(tensor, {index});
    torch::Tensor result_with_init_list = tensor.index({index});
    ASSERT_TRUE(result.equal(result_with_init_list));
  }
  {
    torch::Tensor tensor = torch::randn({20, 20});
    torch::Tensor index = torch::arange(10, torch::kLong).cpu();
    torch::Tensor result = at::index_put_(tensor, {index}, torch::ones({20}));
    torch::Tensor result_with_init_list =
        tensor.index_put_({index}, torch::ones({20}));
    ASSERT_TRUE(result.equal(result_with_init_list));
  }
  {
    torch::Tensor tensor = torch::randn({20, 20});
    torch::Tensor index = torch::arange(10, torch::kLong).cpu();
    torch::Tensor result =
        at::index_put_(tensor, {index}, torch::ones({1, 20}));
    torch::Tensor result_with_init_list =
        tensor.index_put_({index}, torch::ones({1, 20}));
    ASSERT_TRUE(result.equal(result_with_init_list));
  }
}

TEST(TensorIndexingTest, TestSingleInt) {
  auto v = torch::randn({5, 7, 3});
  ASSERT_EQ(v.index({4}).sizes(), torch::IntArrayRef({7, 3}));
}

TEST(TensorIndexingTest, TestMultipleInt) {
  auto v = torch::randn({5, 7, 3});
  ASSERT_EQ(v.index({4}).sizes(), torch::IntArrayRef({7, 3}));
  ASSERT_EQ(v.index({4, Slice(), 1}).sizes(), torch::IntArrayRef({7}));

  // To show that `.index_put_` works
  v.index_put_({4, 3, 1}, 0);
  ASSERT_EQ(v.index({4, 3, 1}).item<double>(), 0);
}

TEST(TensorIndexingTest, TestNone) {
  auto v = torch::randn({5, 7, 3});
  ASSERT_EQ(v.index({None}).sizes(), torch::IntArrayRef({1, 5, 7, 3}));
  ASSERT_EQ(v.index({Slice(), None}).sizes(), torch::IntArrayRef({5, 1, 7, 3}));
  ASSERT_EQ(
      v.index({Slice(), None, None}).sizes(),
      torch::IntArrayRef({5, 1, 1, 7, 3}));
  ASSERT_EQ(v.index({"...", None}).sizes(), torch::IntArrayRef({5, 7, 3, 1}));
}

TEST(TensorIndexingTest, TestStep) {
  auto v = torch::arange(10);
  assert_tensor_equal(v.index({Slice(None, None, 1)}), v);
  assert_tensor_equal(
      v.index({Slice(None, None, 2)}), torch::tensor({0, 2, 4, 6, 8}));
  assert_tensor_equal(
      v.index({Slice(None, None, 3)}), torch::tensor({0, 3, 6, 9}));
  assert_tensor_equal(v.index({Slice(None, None, 11)}), torch::tensor({0}));
  assert_tensor_equal(v.index({Slice(1, 6, 2)}), torch::tensor({1, 3, 5}));
}

TEST(TensorIndexingTest, TestStepAssignment) {
  auto v = torch::zeros({4, 4});
  v.index_put_({0, Slice(1, None, 2)}, torch::tensor({3., 4.}));
  assert_tensor_equal(v.index({0}), torch::tensor({0., 3., 0., 4.}));
  assert_tensor_equal(v.index({Slice(1, None)}).sum(), torch::tensor(0));
}

TEST(TensorIndexingTest, TestBoolIndices) {
  {
    auto v = torch::randn({5, 7, 3});
    auto boolIndices =
        torch::tensor({true, false, true, true, false}, torch::kBool);
    ASSERT_EQ(v.index({boolIndices}).sizes(), torch::IntArrayRef({3, 7, 3}));
    assert_tensor_equal(
        v.index({boolIndices}),
        torch::stack({v.index({0}), v.index({2}), v.index({3})}));
  }
  {
    auto v = torch::tensor({true, false, true}, torch::kBool);
    auto boolIndices = torch::tensor({true, false, false}, torch::kBool);
    auto uint8Indices = torch::tensor({1, 0, 0}, torch::kUInt8);

    {
      WarningCapture warnings;

      ASSERT_EQ(
          v.index({boolIndices}).sizes(), v.index({uint8Indices}).sizes());
      assert_tensor_equal(v.index({boolIndices}), v.index({uint8Indices}));
      assert_tensor_equal(
          v.index({boolIndices}), torch::tensor({true}, torch::kBool));

      ASSERT_EQ(
          count_substr_occurrences(
              warnings.str(),
              "indexing with dtype torch.uint8 is now deprecated"),
          2);
    }
  }
}

TEST(TensorIndexingTest, TestBoolIndicesAccumulate) {
  auto mask = torch::zeros({10}, torch::kBool);
  auto y = torch::ones({10, 10});
  y.index_put_({mask}, {y.index({mask})}, /*accumulate=*/true);
  assert_tensor_equal(y, torch::ones({10, 10}));
}

TEST(TensorIndexingTest, TestMultipleBoolIndices) {
  auto v = torch::randn({5, 7, 3});
  // note: these broadcast together and are transposed to the first dim
  auto mask1 = torch::tensor({1, 0, 1, 1, 0}, torch::kBool);
  auto mask2 = torch::tensor({1, 1, 1}, torch::kBool);
  ASSERT_EQ(
      v.index({mask1, Slice(), mask2}).sizes(), torch::IntArrayRef({3, 7}));
}

TEST(TensorIndexingTest, TestByteMask) {
  {
    auto v = torch::randn({5, 7, 3});
    auto mask = torch::tensor({1, 0, 1, 1, 0}, torch::kByte);
    {
      WarningCapture warnings;

      ASSERT_EQ(v.index({mask}).sizes(), torch::IntArrayRef({3, 7, 3}));
      assert_tensor_equal(v.index({mask}), torch::stack({v[0], v[2], v[3]}));

      ASSERT_EQ(
          count_substr_occurrences(
              warnings.str(),
              "indexing with dtype torch.uint8 is now deprecated"),
          2);
    }
  }
  {
    auto v = torch::tensor({1.});
    assert_tensor_equal(v.index({v == 0}), torch::randn({0}));
  }
}

TEST(TensorIndexingTest, TestByteMaskAccumulate) {
  auto mask = torch::zeros({10}, torch::kUInt8);
  auto y = torch::ones({10, 10});
  {
    WarningCapture warnings;

    y.index_put_({mask}, y.index({mask}), /*accumulate=*/true);
    assert_tensor_equal(y, torch::ones({10, 10}));

    ASSERT_EQ(
        count_substr_occurrences(
            warnings.str(),
            "indexing with dtype torch.uint8 is now deprecated"),
        2);
  }
}

TEST(TensorIndexingTest, TestMultipleByteMask) {
  auto v = torch::randn({5, 7, 3});
  // note: these broadcast together and are transposed to the first dim
  auto mask1 = torch::tensor({1, 0, 1, 1, 0}, torch::kByte);
  auto mask2 = torch::tensor({1, 1, 1}, torch::kByte);
  {
    WarningCapture warnings;

    ASSERT_EQ(
        v.index({mask1, Slice(), mask2}).sizes(), torch::IntArrayRef({3, 7}));

    ASSERT_EQ(
        count_substr_occurrences(
            warnings.str(),
            "indexing with dtype torch.uint8 is now deprecated"),
        2);
  }
}

TEST(TensorIndexingTest, TestByteMask2d) {
  auto v = torch::randn({5, 7, 3});
  auto c = torch::randn({5, 7});
  int64_t num_ones = (c > 0).sum().item().to<int64_t>();
  auto r = v.index({c > 0});
  ASSERT_EQ(r.sizes(), torch::IntArrayRef({num_ones, 3}));
}

TEST(TensorIndexingTest, TestIntIndices) {
  auto v = torch::randn({5, 7, 3});
  ASSERT_EQ(
      v.index({torch::tensor({0, 4, 2})}).sizes(),
      torch::IntArrayRef({3, 7, 3}));
  ASSERT_EQ(
      v.index({Slice(), torch::tensor({0, 4, 2})}).sizes(),
      torch::IntArrayRef({5, 3, 3}));
  ASSERT_EQ(
      v.index({Slice(), torch::tensor({{0, 1}, {4, 3}})}).sizes(),
      torch::IntArrayRef({5, 2, 2, 3}));
}

TEST(TensorIndexingTest, TestIntIndices2d) {
  // From the NumPy indexing example
  auto x = torch::arange(0, 12, torch::kLong).view({4, 3});
  auto rows = torch::tensor({{0, 0}, {3, 3}});
  auto columns = torch::tensor({{0, 2}, {0, 2}});
  assert_tensor_equal(
      x.index({rows, columns}), torch::tensor({{0, 2}, {9, 11}}));
}

TEST(TensorIndexingTest, TestIntIndicesBroadcast) {
  // From the NumPy indexing example
  auto x = torch::arange(0, 12, torch::kLong).view({4, 3});
  auto rows = torch::tensor({0, 3});
  auto columns = torch::tensor({0, 2});
  auto result = x.index({rows.index({Slice(), None}), columns});
  assert_tensor_equal(result, torch::tensor({{0, 2}, {9, 11}}));
}

TEST(TensorIndexingTest, TestEmptyIndex) {
  auto x = torch::arange(0, 12).view({4, 3});
  auto idx = torch::tensor({}, torch::kLong);
  ASSERT_EQ(x.index({idx}).numel(), 0);

  // empty assignment should have no effect but not throw an exception
  auto y = x.clone();
  y.index_put_({idx}, -1);
  assert_tensor_equal(x, y);

  auto mask = torch::zeros({4, 3}, torch::kBool);
  y.index_put_({mask}, -1);
  assert_tensor_equal(x, y);
}

TEST(TensorIndexingTest, TestEmptyNdimIndex) {
  torch::Device device(torch::kCPU);
  {
    auto x = torch::randn({5}, device);
    assert_tensor_equal(
        torch::empty({0, 2}, device),
        x.index({torch::empty(
            {0, 2}, torch::TensorOptions(torch::kInt64).device(device))}));
  }
  {
    auto x = torch::randn({2, 3, 4, 5}, device);
    assert_tensor_equal(
        torch::empty({2, 0, 6, 4, 5}, device),
        x.index(
            {Slice(),
             torch::empty(
                 {0, 6}, torch::TensorOptions(torch::kInt64).device(device))}));
  }
  {
    auto x = torch::empty({10, 0});
    ASSERT_EQ(
        x.index({torch::tensor({1, 2})}).sizes(), torch::IntArrayRef({2, 0}));
    ASSERT_EQ(
        x.index(
             {torch::tensor({}, torch::kLong), torch::tensor({}, torch::kLong)})
            .sizes(),
        torch::IntArrayRef({0}));
    ASSERT_THROWS_WITH(
        x.index({Slice(), torch::tensor({0, 1})}), "for dimension with size 0");
  }
}

TEST(TensorIndexingTest, TestEmptyNdimIndex_CUDA) {
  torch::Device device(torch::kCUDA);
  {
    auto x = torch::randn({5}, device);
    assert_tensor_equal(
        torch::empty({0, 2}, device),
        x.index({torch::empty(
            {0, 2}, torch::TensorOptions(torch::kInt64).device(device))}));
  }
  {
    auto x = torch::randn({2, 3, 4, 5}, device);
    assert_tensor_equal(
        torch::empty({2, 0, 6, 4, 5}, device),
        x.index(
            {Slice(),
             torch::empty(
                 {0, 6}, torch::TensorOptions(torch::kInt64).device(device))}));
  }
}

TEST(TensorIndexingTest, TestEmptyNdimIndexBool) {
  torch::Device device(torch::kCPU);
  auto x = torch::randn({5}, device);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(
      x.index({torch::empty(
          {0, 2}, torch::TensorOptions(torch::kUInt8).device(device))}),
      c10::Error);
}

TEST(TensorIndexingTest, TestEmptyNdimIndexBool_CUDA) {
  torch::Device device(torch::kCUDA);
  auto x = torch::randn({5}, device);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(
      x.index({torch::empty(
          {0, 2}, torch::TensorOptions(torch::kUInt8).device(device))}),
      c10::Error);
}

TEST(TensorIndexingTest, TestEmptySlice) {
  torch::Device device(torch::kCPU);
  auto x = torch::randn({2, 3, 4, 5}, device);
  auto y = x.index({Slice(), Slice(), Slice(), 1});
  auto z = y.index({Slice(), Slice(1, 1), Slice()});
  ASSERT_EQ(z.sizes(), torch::IntArrayRef({2, 0, 4}));
  // this isn't technically necessary, but matches NumPy stride calculations.
  ASSERT_EQ(z.strides(), torch::IntArrayRef({60, 20, 5}));
  ASSERT_TRUE(z.is_contiguous());
}

TEST(TensorIndexingTest, TestEmptySlice_CUDA) {
  torch::Device device(torch::kCUDA);
  auto x = torch::randn({2, 3, 4, 5}, device);
  auto y = x.index({Slice(), Slice(), Slice(), 1});
  auto z = y.index({Slice(), Slice(1, 1), Slice()});
  ASSERT_EQ(z.sizes(), torch::IntArrayRef({2, 0, 4}));
  // this isn't technically necessary, but matches NumPy stride calculations.
  ASSERT_EQ(z.strides(), torch::IntArrayRef({60, 20, 5}));
  ASSERT_TRUE(z.is_contiguous());
}

TEST(TensorIndexingTest, TestIndexGetitemCopyBoolsSlices) {
  auto true_tensor = torch::tensor(1, torch::kUInt8);
  auto false_tensor = torch::tensor(0, torch::kUInt8);

  std::vector<torch::Tensor> tensors = {torch::randn({2, 3}), torch::tensor(3)};

  for (auto& a : tensors) {
    ASSERT_NE(a.data_ptr(), a.index({true}).data_ptr());
    {
      std::vector<int64_t> sizes = {0};
      sizes.insert(sizes.end(), a.sizes().begin(), a.sizes().end());
      assert_tensor_equal(torch::empty(sizes), a.index({false}));
    }
    ASSERT_NE(a.data_ptr(), a.index({true_tensor}).data_ptr());
    {
      std::vector<int64_t> sizes = {0};
      sizes.insert(sizes.end(), a.sizes().begin(), a.sizes().end());
      assert_tensor_equal(torch::empty(sizes), a.index({false_tensor}));
    }
    ASSERT_EQ(a.data_ptr(), a.index({None}).data_ptr());
    ASSERT_EQ(a.data_ptr(), a.index({"..."}).data_ptr());
  }
}

TEST(TensorIndexingTest, TestIndexSetitemBoolsSlices) {
  auto true_tensor = torch::tensor(1, torch::kUInt8);
  auto false_tensor = torch::tensor(0, torch::kUInt8);

  std::vector<torch::Tensor> tensors = {torch::randn({2, 3}), torch::tensor(3)};

  for (auto& a : tensors) {
    // prefix with a 1,1, to ensure we are compatible with numpy which cuts off
    // prefix 1s (some of these ops already prefix a 1 to the size)
    auto neg_ones = torch::ones_like(a) * -1;
    auto neg_ones_expanded = neg_ones.unsqueeze(0).unsqueeze(0);
    a.index_put_({true}, neg_ones_expanded);
    assert_tensor_equal(a, neg_ones);
    a.index_put_({false}, 5);
    assert_tensor_equal(a, neg_ones);
    a.index_put_({true_tensor}, neg_ones_expanded * 2);
    assert_tensor_equal(a, neg_ones * 2);
    a.index_put_({false_tensor}, 5);
    assert_tensor_equal(a, neg_ones * 2);
    a.index_put_({None}, neg_ones_expanded * 3);
    assert_tensor_equal(a, neg_ones * 3);
    a.index_put_({"..."}, neg_ones_expanded * 4);
    assert_tensor_equal(a, neg_ones * 4);
    if (a.dim() == 0) {
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
      ASSERT_THROW(a.index_put_({Slice()}, neg_ones_expanded * 5), c10::Error);
    }
  }
}

TEST(TensorIndexingTest, TestIndexScalarWithBoolMask) {
  torch::Device device(torch::kCPU);

  auto a = torch::tensor(1, device);
  auto uintMask =
      torch::tensor(true, torch::TensorOptions(torch::kUInt8).device(device));
  auto boolMask =
      torch::tensor(true, torch::TensorOptions(torch::kBool).device(device));
  assert_tensor_equal(a.index({uintMask}), a.index({boolMask}));
  ASSERT_EQ(a.index({uintMask}).dtype(), a.index({boolMask}).dtype());

  a = torch::tensor(true, torch::TensorOptions(torch::kBool).device(device));
  assert_tensor_equal(a.index({uintMask}), a.index({boolMask}));
  ASSERT_EQ(a.index({uintMask}).dtype(), a.index({boolMask}).dtype());
}

TEST(TensorIndexingTest, TestIndexScalarWithBoolMask_CUDA) {
  torch::Device device(torch::kCUDA);

  auto a = torch::tensor(1, device);
  auto uintMask =
      torch::tensor(true, torch::TensorOptions(torch::kUInt8).device(device));
  auto boolMask =
      torch::tensor(true, torch::TensorOptions(torch::kBool).device(device));
  assert_tensor_equal(a.index({uintMask}), a.index({boolMask}));
  ASSERT_EQ(a.index({uintMask}).dtype(), a.index({boolMask}).dtype());

  a = torch::tensor(true, torch::TensorOptions(torch::kBool).device(device));
  assert_tensor_equal(a.index({uintMask}), a.index({boolMask}));
  ASSERT_EQ(a.index({uintMask}).dtype(), a.index({boolMask}).dtype());
}

TEST(TensorIndexingTest, TestSetitemExpansionError) {
  auto true_tensor = torch::tensor(true);
  auto a = torch::randn({2, 3});
  // check prefix with  non-1s doesn't work
  std::vector<int64_t> tensor_sizes{5, 1};
  tensor_sizes.insert(tensor_sizes.end(), a.sizes().begin(), a.sizes().end());
  auto a_expanded = a.expand(tensor_sizes);
  // NumPy: ValueError
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(a.index_put_({true}, a_expanded), c10::Error);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(a.index_put_({true_tensor}, a_expanded), c10::Error);
}

TEST(TensorIndexingTest, TestGetitemScalars) {
  auto zero = torch::tensor(0, torch::kInt64);
  auto one = torch::tensor(1, torch::kInt64);

  // non-scalar indexed with scalars
  auto a = torch::randn({2, 3});
  assert_tensor_equal(a.index({0}), a.index({zero}));
  assert_tensor_equal(a.index({0}).index({1}), a.index({zero}).index({one}));
  assert_tensor_equal(a.index({0, 1}), a.index({zero, one}));
  assert_tensor_equal(a.index({0, one}), a.index({zero, 1}));

  // indexing by a scalar should slice (not copy)
  ASSERT_EQ(a.index({0, 1}).data_ptr(), a.index({zero, one}).data_ptr());
  ASSERT_EQ(a.index({1}).data_ptr(), a.index({one.to(torch::kInt)}).data_ptr());
  ASSERT_EQ(
      a.index({1}).data_ptr(), a.index({one.to(torch::kShort)}).data_ptr());

  // scalar indexed with scalar
  auto r = torch::randn({});
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(r.index({Slice()}), c10::Error);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(r.index({zero}), c10::Error);
  assert_tensor_equal(r, r.index({"..."}));
}

TEST(TensorIndexingTest, TestSetitemScalars) {
  auto zero = torch::tensor(0, torch::kInt64);

  // non-scalar indexed with scalars
  auto a = torch::randn({2, 3});
  auto a_set_with_number = a.clone();
  auto a_set_with_scalar = a.clone();
  auto b = torch::randn({3});

  a_set_with_number.index_put_({0}, b);
  a_set_with_scalar.index_put_({zero}, b);
  assert_tensor_equal(a_set_with_number, a_set_with_scalar);
  a.index_put_({1, zero}, 7.7);
  ASSERT_TRUE(a.index({1, 0}).allclose(torch::tensor(7.7)));

  // scalar indexed with scalars
  auto r = torch::randn({});
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(r.index_put_({Slice()}, 8.8), c10::Error);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(r.index_put_({zero}, 8.8), c10::Error);
  r.index_put_({"..."}, 9.9);
  ASSERT_TRUE(r.allclose(torch::tensor(9.9)));
}

TEST(TensorIndexingTest, TestBasicAdvancedCombined) {
  // From the NumPy indexing example
  auto x = torch::arange(0, 12).to(torch::kLong).view({4, 3});
  assert_tensor_equal(
      x.index({Slice(1, 2), Slice(1, 3)}),
      x.index({Slice(1, 2), torch::tensor({1, 2})}));
  assert_tensor_equal(
      x.index({Slice(1, 2), Slice(1, 3)}), torch::tensor({{4, 5}}));

  // Check that it is a copy
  {
    auto unmodified = x.clone();
    x.index({Slice(1, 2), torch::tensor({1, 2})}).zero_();
    assert_tensor_equal(x, unmodified);
  }

  // But assignment should modify the original
  {
    auto unmodified = x.clone();
    x.index_put_({Slice(1, 2), torch::tensor({1, 2})}, 0);
    assert_tensor_not_equal(x, unmodified);
  }
}

TEST(TensorIndexingTest, TestIntAssignment) {
  {
    auto x = torch::arange(0, 4).to(torch::kLong).view({2, 2});
    x.index_put_({1}, 5);
    assert_tensor_equal(x, torch::tensor({{0, 1}, {5, 5}}));
  }

  {
    auto x = torch::arange(0, 4).to(torch::kLong).view({2, 2});
    x.index_put_({1}, torch::arange(5, 7).to(torch::kLong));
    assert_tensor_equal(x, torch::tensor({{0, 1}, {5, 6}}));
  }
}

TEST(TensorIndexingTest, TestByteTensorAssignment) {
  auto x = torch::arange(0., 16).to(torch::kFloat).view({4, 4});
  auto b = torch::tensor({true, false, true, false}, torch::kByte);
  auto value = torch::tensor({3., 4., 5., 6.});

  {
    WarningCapture warnings;

    x.index_put_({b}, value);

    ASSERT_EQ(
        count_substr_occurrences(
            warnings.str(),
            "indexing with dtype torch.uint8 is now deprecated"),
        1);
  }

  assert_tensor_equal(x.index({0}), value);
  assert_tensor_equal(x.index({1}), torch::arange(4, 8).to(torch::kLong));
  assert_tensor_equal(x.index({2}), value);
  assert_tensor_equal(x.index({3}), torch::arange(12, 16).to(torch::kLong));
}

TEST(TensorIndexingTest, TestVariableSlicing) {
  auto x = torch::arange(0, 16).view({4, 4});
  auto indices = torch::tensor({0, 1}, torch::kInt);
  int i = indices[0].item<int>();
  int j = indices[1].item<int>();
  assert_tensor_equal(x.index({Slice(i, j)}), x.index({Slice(0, 1)}));
}

TEST(TensorIndexingTest, TestEllipsisTensor) {
  auto x = torch::arange(0, 9).to(torch::kLong).view({3, 3});
  auto idx = torch::tensor({0, 2});
  assert_tensor_equal(
      x.index({"...", idx}), torch::tensor({{0, 2}, {3, 5}, {6, 8}}));
  assert_tensor_equal(
      x.index({idx, "..."}), torch::tensor({{0, 1, 2}, {6, 7, 8}}));
}

TEST(TensorIndexingTest, TestOutOfBoundIndex) {
  auto x = torch::arange(0, 100).view({2, 5, 10});
  ASSERT_THROWS_WITH(
      x.index({0, 5}), "index 5 is out of bounds for dimension 1 with size 5");
  ASSERT_THROWS_WITH(
      x.index({4, 5}), "index 4 is out of bounds for dimension 0 with size 2");
  ASSERT_THROWS_WITH(
      x.index({0, 1, 15}),
      "index 15 is out of bounds for dimension 2 with size 10");
  ASSERT_THROWS_WITH(
      x.index({Slice(), Slice(), 12}),
      "index 12 is out of bounds for dimension 2 with size 10");
}

TEST(TensorIndexingTest, TestZeroDimIndex) {
  auto x = torch::tensor(10);

  auto runner = [&]() -> torch::Tensor {
    std::cout << x.index({0}) << std::endl;
    return x.index({0});
  };

  ASSERT_THROWS_WITH(runner(), "invalid index");
}

// The tests below are from NumPy test_indexing.py with some modifications to
// make them compatible with libtorch. It's licensed under the BDS license
// below:
//
// Copyright (c) 2005-2017, NumPy Developers.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//        copyright notice, this list of conditions and the following
//        disclaimer in the documentation and/or other materials provided
//        with the distribution.
//
//     * Neither the name of the NumPy Developers nor the names of any
//        contributors may be used to endorse or promote products derived
//        from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

TEST(NumpyTests, TestNoneIndex) {
  // `None` index adds newaxis
  auto a = torch::tensor({1, 2, 3});
  ASSERT_EQ(a.index({None}).dim(), a.dim() + 1);
}

TEST(NumpyTests, TestEmptyFancyIndex) {
  // Empty list index creates an empty array
  auto a = torch::tensor({1, 2, 3});
  assert_tensor_equal(
      a.index({torch::tensor({}, torch::kLong)}), torch::tensor({}));

  auto b = torch::tensor({}).to(torch::kLong);
  assert_tensor_equal(
      a.index({torch::tensor({}, torch::kLong)}),
      torch::tensor({}, torch::kLong));

  b = torch::tensor({}).to(torch::kFloat);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(a.index({b}), c10::Error);
}

TEST(NumpyTests, TestEllipsisIndex) {
  auto a = torch::tensor({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  ASSERT_FALSE(a.index({"..."}).is_same(a));
  assert_tensor_equal(a.index({"..."}), a);
  // `a[...]` was `a` in numpy <1.9.
  ASSERT_EQ(a.index({"..."}).data_ptr(), a.data_ptr());

  // Slicing with ellipsis can skip an
  // arbitrary number of dimensions
  assert_tensor_equal(a.index({0, "..."}), a.index({0}));
  assert_tensor_equal(a.index({0, "..."}), a.index({0, Slice()}));
  assert_tensor_equal(a.index({"...", 0}), a.index({Slice(), 0}));

  // In NumPy, slicing with ellipsis results in a 0-dim array. In PyTorch
  // we don't have separate 0-dim arrays and scalars.
  assert_tensor_equal(a.index({0, "...", 1}), torch::tensor(2));

  // Assignment with `Ellipsis` on 0-d arrays
  auto b = torch::tensor(1);
  b.index_put_({Ellipsis}, 2);
  ASSERT_EQ(b.item<int64_t>(), 2);
}

TEST(NumpyTests, TestSingleIntIndex) {
  // Single integer index selects one row
  auto a = torch::tensor({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});

  assert_tensor_equal(a.index({0}), torch::tensor({1, 2, 3}));
  assert_tensor_equal(a.index({-1}), torch::tensor({7, 8, 9}));

  // Index out of bounds produces IndexError
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(a.index({1 << 30}), c10::Error);
  // NOTE: According to the standard
  // (http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0543r0.html), for
  // signed integers, if during the evaluation of an expression, the result is
  // not mathematically defined or not in the range of representable values for
  // its type, the behavior is undefined. Therefore, there is no way to check
  // for index overflow case because it might not throw exception.
  // ASSERT_THROW(a(1 << 64), c10::Error);
}

TEST(NumpyTests, TestSingleBoolIndex) {
  // Single boolean index
  auto a = torch::tensor({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});

  assert_tensor_equal(a.index({true}), a.index({None}));
  assert_tensor_equal(a.index({false}), a.index({None}).index({Slice(0, 0)}));
}

TEST(NumpyTests, TestBooleanShapeMismatch) {
  auto arr = torch::ones({5, 4, 3});

  auto index = torch::tensor({true});
  ASSERT_THROWS_WITH(arr.index({index}), "mask");

  index = torch::tensor({false, false, false, false, false, false});
  ASSERT_THROWS_WITH(arr.index({index}), "mask");

  {
    WarningCapture warnings;

    index = torch::empty({4, 4}, torch::kByte).zero_();
    ASSERT_THROWS_WITH(arr.index({index}), "mask");
    ASSERT_THROWS_WITH(arr.index({Slice(), index}), "mask");

    ASSERT_EQ(
        count_substr_occurrences(
            warnings.str(),
            "indexing with dtype torch.uint8 is now deprecated"),
        2);
  }
}

TEST(NumpyTests, TestBooleanIndexingOnedim) {
  // Indexing a 2-dimensional array with
  // boolean array of length one
  auto a = torch::tensor({{0., 0., 0.}});
  auto b = torch::tensor({true});
  assert_tensor_equal(a.index({b}), a);
  // boolean assignment
  a.index_put_({b}, 1.);
  assert_tensor_equal(a, torch::tensor({{1., 1., 1.}}));
}

TEST(NumpyTests, TestBooleanAssignmentValueMismatch) {
  // A boolean assignment should fail when the shape of the values
  // cannot be broadcast to the subscription. (see also gh-3458)
  auto a = torch::arange(0, 4);

  auto f = [](torch::Tensor a, std::vector<int64_t> v) -> void {
    a.index_put_({a > -1}, torch::tensor(v));
  };

  ASSERT_THROWS_WITH(f(a, {}), "shape mismatch");
  ASSERT_THROWS_WITH(f(a, {1, 2, 3}), "shape mismatch");
  ASSERT_THROWS_WITH(f(a.index({Slice(None, 1)}), {1, 2, 3}), "shape mismatch");
}

TEST(NumpyTests, TestBooleanIndexingTwodim) {
  // Indexing a 2-dimensional array with
  // 2-dimensional boolean array
  auto a = torch::tensor({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  auto b = torch::tensor(
      {{true, false, true}, {false, true, false}, {true, false, true}});
  assert_tensor_equal(a.index({b}), torch::tensor({1, 3, 5, 7, 9}));
  assert_tensor_equal(a.index({b.index({1})}), torch::tensor({{4, 5, 6}}));
  assert_tensor_equal(a.index({b.index({0})}), a.index({b.index({2})}));

  // boolean assignment
  a.index_put_({b}, 0);
  assert_tensor_equal(a, torch::tensor({{0, 2, 0}, {4, 0, 6}, {0, 8, 0}}));
}

TEST(NumpyTests, TestBooleanIndexingWeirdness) {
  // Weird boolean indexing things
  auto a = torch::ones({2, 3, 4});
  ASSERT_EQ(
      a.index({false, true, "..."}).sizes(), torch::IntArrayRef({0, 2, 3, 4}));
  assert_tensor_equal(
      torch::ones({1, 2}),
      a.index(
          {true,
           torch::tensor({0, 1}),
           true,
           true,
           torch::tensor({1}),
           torch::tensor({{2}})}));
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(a.index({false, torch::tensor({0, 1}), "..."}), c10::Error);
}

TEST(NumpyTests, TestBooleanIndexingWeirdnessTensors) {
  // Weird boolean indexing things
  auto false_tensor = torch::tensor(false);
  auto true_tensor = torch::tensor(true);
  auto a = torch::ones({2, 3, 4});
  ASSERT_EQ(
      a.index({false, true, "..."}).sizes(), torch::IntArrayRef({0, 2, 3, 4}));
  assert_tensor_equal(
      torch::ones({1, 2}),
      a.index(
          {true_tensor,
           torch::tensor({0, 1}),
           true_tensor,
           true_tensor,
           torch::tensor({1}),
           torch::tensor({{2}})}));
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(
      a.index({false_tensor, torch::tensor({0, 1}), "..."}), c10::Error);
}

TEST(NumpyTests, TestBooleanIndexingAlldims) {
  auto true_tensor = torch::tensor(true);
  auto a = torch::ones({2, 3});
  ASSERT_EQ(a.index({true, true}).sizes(), torch::IntArrayRef({1, 2, 3}));
  ASSERT_EQ(
      a.index({true_tensor, true_tensor}).sizes(),
      torch::IntArrayRef({1, 2, 3}));
}

TEST(NumpyTests, TestBooleanListIndexing) {
  // Indexing a 2-dimensional array with
  // boolean lists
  auto a = torch::tensor({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  auto b = torch::tensor({true, false, false});
  auto c = torch::tensor({true, true, false});
  assert_tensor_equal(a.index({b}), torch::tensor({{1, 2, 3}}));
  assert_tensor_equal(a.index({b, b}), torch::tensor({1}));
  assert_tensor_equal(a.index({c}), torch::tensor({{1, 2, 3}, {4, 5, 6}}));
  assert_tensor_equal(a.index({c, c}), torch::tensor({1, 5}));
}

TEST(NumpyTests, TestEverythingReturnsViews) {
  // Before `...` would return a itself.
  auto a = torch::tensor({5});

  ASSERT_FALSE(a.is_same(a.index({"..."})));
  ASSERT_FALSE(a.is_same(a.index({Slice()})));
}

TEST(NumpyTests, TestBroaderrorsIndexing) {
  auto a = torch::zeros({5, 5});
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(
      a.index({torch::tensor({0, 1}), torch::tensor({0, 1, 2})}), c10::Error);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(
      a.index_put_({torch::tensor({0, 1}), torch::tensor({0, 1, 2})}, 0),
      c10::Error);
}

TEST(NumpyTests, TestTrivialFancyOutOfBounds) {
  auto a = torch::zeros({5});
  auto ind = torch::ones({20}, torch::kInt64);
  ind.index_put_({-1}, 10);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(a.index({ind}), c10::Error);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(a.index_put_({ind}, 0), c10::Error);
  ind = torch::ones({20}, torch::kInt64);
  ind.index_put_({0}, 11);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(a.index({ind}), c10::Error);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(a.index_put_({ind}, 0), c10::Error);
}

TEST(NumpyTests, TestIndexIsLarger) {
  // Simple case of fancy index broadcasting of the index.
  auto a = torch::zeros({5, 5});
  a.index_put_(
      {torch::tensor({{0}, {1}, {2}}), torch::tensor({0, 1, 2})},
      torch::tensor({2., 3., 4.}));

  ASSERT_TRUE(
      (a.index({Slice(None, 3), Slice(None, 3)}) == torch::tensor({2., 3., 4.}))
          .all()
          .item<bool>());
}

TEST(NumpyTests, TestBroadcastSubspace) {
  auto a = torch::zeros({100, 100});
  auto v = torch::arange(0., 100).index({Slice(), None});
  auto b = torch::arange(99, -1, -1).to(torch::kLong);
  a.index_put_({b}, v);
  auto expected = b.to(torch::kDouble).unsqueeze(1).expand({100, 100});
  assert_tensor_equal(a, expected);
}
