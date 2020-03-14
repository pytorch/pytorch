#include <gtest/gtest.h>

#include <ATen/ATen.h>

using namespace at;

#define ASSERT_EQUAL(t1, t2) ASSERT_TRUE(t1.equal(t2));

#define ASSERT_ALLCLOSE(t1, t2)     \
  ASSERT_TRUE(t1.is_same_size(t2)); \
  ASSERT_TRUE(t1.allclose(t2));

#define ASSERT_ALLCLOSE_TOLERANCES(t1, t2, atol, rtol) \
  ASSERT_TRUE(t1.is_same_size(t2));                    \
  ASSERT_TRUE(t1.allclose(t2, atol, rtol));

void requireEqualTensorList(TensorList t1, TensorList t2) {
  ASSERT_EQ(t1.size(), t2.size());
  for (size_t i = 0; i < t1.size(); ++i) {
    ASSERT_EQUAL(t1[i], t2[i]);
  }
}

// split: test method, namespace give same result
void TestSplit(TensorOptions T, Tensor& t) {
  auto splitMethod = t.split(1, 0);
  auto splitNs = at::split(t, 1, 0);
  requireEqualTensorList(splitMethod, splitNs);

  // test rebuilding with cat
  ASSERT_EQUAL(at::cat(splitMethod, 0), t);
}

// chunk: test method, namespace give same result
void TestChunk(TensorOptions T, Tensor& t) {
  // test method, type, namespace give same result
  auto chunkMethod = t.chunk(3, 0);
  auto chunkNs = at::chunk(t, 3, 0);
  requireEqualTensorList(chunkMethod, chunkNs);

  // test rebuilding with cat
  ASSERT_EQUAL(at::cat(chunkMethod, 0), t);
}

void TestStack(TensorOptions T, Tensor& t) {
  auto x = rand({2, 3, 4});
  auto y = rand({2, 3, 4});
  auto z = rand({2, 3, 4});
  for (int64_t dim = 0; dim < 4; ++dim) {
    auto res = at::stack({x, y, z}, dim);
    auto res_neg = at::stack({x, y, z}, dim - 4);
    std::vector<int64_t> expected_size;
    expected_size.insert(
        expected_size.end(), x.sizes().begin(), x.sizes().begin() + dim);
    expected_size.insert(expected_size.end(), 3);
    expected_size.insert(
        expected_size.end(), x.sizes().begin() + dim, x.sizes().end());

    ASSERT_EQUAL(res, res_neg);
    ASSERT_TRUE(res.sizes().equals(expected_size));
    ASSERT_EQUAL(res.select(dim, 0), x);
    ASSERT_EQUAL(res.select(dim, 1), y);
    ASSERT_EQUAL(res.select(dim, 2), z);
  }
}

// size / stride
void TestSize(TensorOptions T, Tensor& t) {
  auto scalar = randn({}, T);
  // Throw StartsWith("dimension specified as 0 but tensor has no dimensions")
  ASSERT_ANY_THROW(scalar.size(0));
  // Throw StartsWith("dimension specified as -1 but tensor has no dimensions")
  ASSERT_ANY_THROW(scalar.size(-1));
  // Throw StartsWith("dimension specified as 0 but tensor has no dimensions")
  ASSERT_ANY_THROW(scalar.stride(0));
  // Throw StartsWith("dimension specified as -1 but tensor has no dimensions")
  ASSERT_ANY_THROW(scalar.stride(-1));

  auto empty = randn({0}, T);
  ASSERT_EQ(empty.size(0), 0);
  ASSERT_EQ(empty.size(-1), 0);
  ASSERT_EQ(empty.stride(0), 1);
  ASSERT_EQ(empty.stride(-1), 1);
}

void TestMatmul(TensorOptions T, Tensor& t, TensorOptions AccT) {
  auto scalar = randn({}, T);
  auto d1 = randn({3}, T);
  auto d2 = randn({2, 3}, T);

  // 0-d
  // Throw StartsWith("both arguments to matmul need to be at least 1D")
  ASSERT_ANY_THROW(scalar.matmul(d2));
  // Throw StartsWith("both arguments to matmul need to be at least 1D")
  ASSERT_ANY_THROW(d2.matmul(scalar));

  // 1-d
  ASSERT_ALLCLOSE(d1.matmul(d1), d1.dot(d1));
  ASSERT_ALLCLOSE(d2.matmul(d1), d2.mv(d1));
  auto d1o = randn({2}, T);
  ASSERT_ALLCLOSE(d1o.matmul(d2), d1o.unsqueeze(0).mm(d2).squeeze(0));

  // 2-d
  auto d2o = randn({3, 5}, T);
  ASSERT_ALLCLOSE(d2.matmul(d2o), d2.mm(d2o));

  // > 2-d, 1-d
  auto d3 = randn({5, 2, 3}, T);
  ASSERT_ALLCLOSE(
      d3.matmul(d1), d3.bmm(d1.view({1, 3, 1}).expand({5, 3, 1})).view({5, 2}));
  ASSERT_ALLCLOSE(d1o.matmul(d3), d1o.expand({5, 1, 2}).bmm(d3).view({5, 3}));

  auto d5 = randn({3, 2, 4, 2, 3}, T);
  ASSERT_ALLCLOSE(
      d5.matmul(d1),
      d5.view({24, 2, 3})
          .bmm(d1.view({1, 3, 1}).expand({24, 3, 1}))
          .view({3, 2, 4, 2}));
  ASSERT_ALLCLOSE(
      d1o.matmul(d5),
      d1o.expand({24, 1, 2}).bmm(d5.view({24, 2, 3})).view({3, 2, 4, 3}));

  // > 2-d, 2-d
  // we use a "folding" algorithm in this case of matmul, so the direct
  // comparison to bmm doesn't work; instead, compare to the higher precision
  // computation (technically, we should always do this). Tolerances are
  // selected empirically.
  double atol = 1e-04;
  double rtol = 1e-06;
  d2 = randn({3, 4}, T);
  d2o = randn({4, 2}, T);
  auto result = d5.matmul(d2).to(AccT);

  auto d5Acc = d5.to(AccT);
  auto d2Acc = d2.to(AccT);
  auto acc_result = d5Acc.view({24, 2, 3})
                        .bmm(d2Acc.expand({24, 3, 4}))
                        .view({3, 2, 4, 2, 4});
  ASSERT_ALLCLOSE_TOLERANCES(result, acc_result, atol, rtol);
  ASSERT_ALLCLOSE(
      d2o.matmul(d5),
      d2o.expand({24, 4, 2}).bmm(d5.view({24, 2, 3})).view({3, 2, 4, 4, 3}));

  // > 2-d, > 2-d
  auto d5o = randn({2, 1, 2, 4, 3, 2}, T);
  auto d5_bmm_view =
      d5.expand({2, 3, 2, 4, 2, 3}).contiguous().view({48, 2, 3});
  auto d5o_bmm_view =
      d5o.expand({2, 3, 2, 4, 3, 2}).contiguous().view({48, 3, 2});
  ASSERT_ALLCLOSE(
      d5.matmul(d5o), d5_bmm_view.bmm(d5o_bmm_view).view({2, 3, 2, 4, 2, 2}));

  // non-expandable case
  auto d5wrong = randn({2, 4, 2, 4, 3, 2}, T);
  // Throw Contains("must match the size")
  ASSERT_ANY_THROW(d5.matmul(d5wrong));
}

void TestStandardGammaGrad(TensorOptions T, Tensor& t) {
  // check empty
  auto empty = ones({0}, T);
  ASSERT_EQUAL(empty, at::_standard_gamma_grad(empty, empty));

  // check scalar equals one element
  auto one_scalar = ones({}, T).mul(5);
  auto one_with_dim = ones({1}, T).mul(5);
  ASSERT_ALLCLOSE(
      at::_standard_gamma_grad(one_scalar, one_scalar),
      at::_standard_gamma_grad(one_with_dim, one_with_dim).sum());

  // check mixing types
  auto t1 = randn({3, 4}, T);
  auto t2 = randn({3, 4}, T).toType(kDouble);
  // Throw StartsWith("expected scalar type")
  ASSERT_ANY_THROW(at::_standard_gamma_grad(t1, t2));
}

void TestWhere(TensorOptions T, Tensor& t) {
  // empty
  auto empty = ones({0}, T);
  auto bT = T.dtype(kByte);
  auto empty_byte = ones({0}, bT);
  ASSERT_EQUAL(empty, at::where(empty_byte, empty, empty));

  // check scalar equals one element
  auto x_scalar = ones({}, T).mul(5);
  auto y_scalar = ones({}, T).mul(7);
  auto cond_scalar = zeros({}, bT);
  auto x_1d = x_scalar.unsqueeze(0);
  auto y_1d = y_scalar.unsqueeze(0);
  auto cond_1d = cond_scalar.unsqueeze(0);
  ASSERT_ALLCLOSE(
      at::where(cond_scalar, x_scalar, y_scalar).unsqueeze(0),
      at::where(cond_1d, x_1d, y_1d));
}

void test(TensorOptions T, TensorOptions AccT) {
  auto t = randn({3, 3}, T);
  TestSplit(T, t);
  TestChunk(T, t);
  TestStack(T, t);
  TestSize(T, t);
  TestMatmul(T, t, AccT);
  TestStandardGammaGrad(T, t);
  TestWhere(T, t);
}

TEST(TestNative, NativeTestCPU) {
  manual_seed(123);

  test(at::device(kCPU).dtype(kFloat),
       at::device(kCPU).dtype(kDouble));
}

TEST(TestNative, NativeTestGPU) {
  manual_seed(123);

  if (at::hasCUDA()) {
    test(at::device(kCUDA).dtype(kFloat),
         at::device(kCUDA).dtype(kDouble));
  }
}
