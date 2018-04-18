#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "ATen/ATen.h"
#include "test_seed.h"

using namespace at;

#define REQUIRE_EQUAL(t1, t2) \
  REQUIRE(t1.equal(t2));

#define REQUIRE_ALLCLOSE(t1, t2)   \
  REQUIRE(t1.is_same_size(t2));    \
  REQUIRE(t1.allclose(t2));

#define REQUIRE_ALLCLOSE_TOLERANCES(t1, t2, atol, rtol)   \
  REQUIRE(t1.is_same_size(t2));    \
  REQUIRE(t1.allclose(t2, atol, rtol));

void requireEqualTensorList(TensorList t1, TensorList t2) {
  REQUIRE(t1.size() == t2.size());
  for (size_t i = 0; i < t1.size(); ++i) {
    REQUIRE_EQUAL(t1[ i ], t2[ i ]);
  }
}

void test(Type & T, Type & AccT) {
  auto t = randn(T, {3, 3});

  SECTION( "split: test method, type, namespace give same result" ) {
    auto splitMethod = t.split(1, 0);
    auto splitType = T.split(t, 1, 0);
    auto splitNs = at::split(t, 1, 0);
    requireEqualTensorList(splitMethod, splitType);
    requireEqualTensorList(splitMethod, splitNs);

    // test rebuilding with cat
    REQUIRE_EQUAL(at::cat(splitMethod, 0), t);
  }

  SECTION( "chunk: test method, type, namespace give same result" ) {
    // test method, type, namespace give same result
    auto chunkMethod = t.chunk(3, 0);
    auto chunkType = T.chunk(t, 3, 0);
    auto chunkNs = at::chunk(t, 3, 0);
    requireEqualTensorList(chunkMethod, chunkType);
    requireEqualTensorList(chunkMethod, chunkNs);

    // test rebuilding with cat
    REQUIRE_EQUAL(at::cat(chunkMethod, 0), t);
  }

  // stack
  SECTION( "stack" ) {
    auto x = rand(T, {2, 3, 4});
    auto y = rand(T, {2, 3, 4});
    auto z = rand(T, {2, 3, 4});
    for (int64_t dim = 0; dim < 4; ++dim) {
      auto res = at::stack({x, y, z}, dim);
      auto res_neg = at::stack({x, y, z}, dim - 4);
      std::vector<int64_t> expected_size;
      expected_size.insert(expected_size.end(), x.sizes().begin(), x.sizes().begin() + dim);
      expected_size.insert(expected_size.end(), 3);
      expected_size.insert(expected_size.end(), x.sizes().begin() + dim, x.sizes().end());

      REQUIRE_EQUAL(res, res_neg);
      REQUIRE(res.sizes().equals(expected_size));
      REQUIRE_EQUAL(res.select(dim, 0), x);
      REQUIRE_EQUAL(res.select(dim, 1), y);
      REQUIRE_EQUAL(res.select(dim, 2), z);
    }
  }

  SECTION( "size / stride" ) {
    auto scalar = randn(T, {});
		REQUIRE_THROWS_WITH(scalar.size(0), "dimension specified as 0 but tensor has no dimensions");
    REQUIRE_THROWS_WITH(scalar.size(-1), "dimension specified as -1 but tensor has no dimensions");
    REQUIRE_THROWS_WITH(scalar.stride(0), "dimension specified as 0 but tensor has no dimensions");
    REQUIRE_THROWS_WITH(scalar.stride(-1), "dimension specified as -1 but tensor has no dimensions");

    auto empty = randn(T, {0});
    REQUIRE(empty.size(0) == 0);
    REQUIRE(empty.size(-1) == 0);
    REQUIRE(empty.stride(0) == 1);
    REQUIRE(empty.stride(-1) == 1);
  }

  // matmul
  SECTION( "matmul" ) {
    auto scalar = randn(T, {});
    auto d1 = randn(T, {3});
    auto d2 = randn(T, {2, 3});

    // 0-d
    REQUIRE_THROWS_WITH(scalar.matmul(d2), Catch::StartsWith("both arguments to matmul need to be at least 1D"));
    REQUIRE_THROWS_WITH(d2.matmul(scalar), Catch::StartsWith("both arguments to matmul need to be at least 1D"));

    // 1-d
    REQUIRE_ALLCLOSE(d1.matmul(d1), d1.dot(d1));
    REQUIRE_ALLCLOSE(d2.matmul(d1), d2.mv(d1));
    auto d1o = randn(T, {2});
    REQUIRE_ALLCLOSE(d1o.matmul(d2), d1o.unsqueeze(0).mm(d2).squeeze(0));

    // 2-d
    auto d2o = randn(T, {3, 5});
    REQUIRE_ALLCLOSE(d2.matmul(d2o), d2.mm(d2o));

    // > 2-d, 1-d
    auto d3 = randn(T, {5, 2, 3});
    REQUIRE_ALLCLOSE(d3.matmul(d1), d3.bmm(d1.view({1, 3, 1}).expand({5, 3, 1})).view({5, 2}));
    REQUIRE_ALLCLOSE(d1o.matmul(d3), d1o.expand({5, 1, 2}).bmm(d3).view({5, 3}));

    auto d5 = randn(T, {3, 2, 4, 2, 3});
    REQUIRE_ALLCLOSE(d5.matmul(d1), d5.view({24, 2, 3}).bmm(d1.view({1, 3, 1}).expand({24, 3, 1})).view({3, 2, 4, 2}));
    REQUIRE_ALLCLOSE(d1o.matmul(d5), d1o.expand({24, 1, 2}).bmm(d5.view({24, 2, 3})).view({3, 2, 4, 3}));

    // > 2-d, 2-d
    // we use a "folding" algorithm in this case of matmul, so the direct comparison to bmm doesn't work;
    // instead, compare to the higher precision computation (technically, we should always do this).
    // Tolerances are selected empirically.
    double atol = 1e-04;
    double rtol = 1e-06;
    d2 = randn(T, {3, 4});
    d2o = randn(T, {4, 2});
    auto result = d5.matmul(d2).toType(AccT);

    auto d5Acc = d5.toType(AccT);
    auto d2Acc = d2.toType(AccT);
    auto acc_result = d5Acc.view({24, 2, 3}).bmm(d2Acc.expand({24, 3, 4})).view({3, 2, 4, 2, 4});
    REQUIRE_ALLCLOSE_TOLERANCES(result, acc_result, atol, rtol);
    REQUIRE_ALLCLOSE(d2o.matmul(d5), d2o.expand({24, 4, 2}).bmm(d5.view({24, 2, 3})).view({3, 2, 4, 4, 3}));

    // > 2-d, > 2-d
    auto d5o = randn(T, {2, 1, 2, 4, 3, 2});
    auto d5_bmm_view = d5.expand({2, 3, 2, 4, 2, 3}).contiguous().view({48, 2, 3});
    auto d5o_bmm_view = d5o.expand({2, 3, 2, 4, 3, 2}).contiguous().view({48, 3, 2});
    REQUIRE_ALLCLOSE(d5.matmul(d5o), d5_bmm_view.bmm(d5o_bmm_view).view({2, 3, 2, 4, 2, 2}));

    // non-expandable case
    auto d5wrong = randn(T, {2, 4, 2, 4, 3, 2});
    REQUIRE_THROWS_WITH(d5.matmul(d5wrong), Catch::Contains("must match the size"));
  }

  // _standard_gamma_grad
  SECTION( "_standard_gamma_grad" ) {
    if (!T.is_cuda()) {
      // check empty
      auto empty = ones(T, {0});
      REQUIRE_EQUAL(empty, empty._standard_gamma_grad(empty));

      // check scalar equals one element
      auto one_scalar = ones(T, {}).mul(5);
      auto one_with_dim = ones(T, {1}).mul(5);
      REQUIRE_ALLCLOSE(one_scalar._standard_gamma_grad(one_scalar),
                      one_with_dim._standard_gamma_grad(one_with_dim).sum());

      // check mixing types
      Type & DT = CPU(kDouble);
      auto t1 = randn(T, {3, 4});
      auto t2 = randn(DT, {3, 4});
      REQUIRE_THROWS_WITH(t1._standard_gamma_grad(t2), Catch::StartsWith("expected scalar type"));
    } else {
      auto ct1 = randn(T, {3, 4});
      auto ct2 = randn(T, {3, 4});
      auto t1 = randn(T.toBackend(Backend::CPU), {3, 4});
      REQUIRE_THROWS_WITH(ct1._standard_gamma_grad(ct2), Catch::Contains("not implemented"));
      REQUIRE_THROWS_WITH(ct1._standard_gamma_grad(t1), Catch::Contains("not implemented"));
      REQUIRE_THROWS_WITH(t1._standard_gamma_grad(ct2), Catch::Contains("CUDA Backend"));
    }
	}

  SECTION( "where" ) {
    // empty
    auto empty = ones(T, {0});
    auto &bT = T.toScalarType(ScalarType::Byte);
    auto empty_byte = ones(bT, {0});
    REQUIRE_EQUAL(empty, at::where(empty_byte, empty, empty));

    // check scalar equals one element
    auto x_scalar = ones(T, {}).mul(5);
    auto y_scalar = ones(T, {}).mul(7);
    auto cond_scalar = zeros(bT, {});
    auto x_1d = x_scalar.unsqueeze(0);
    auto y_1d = y_scalar.unsqueeze(0);
    auto cond_1d = cond_scalar.unsqueeze(0);
    REQUIRE_ALLCLOSE(at::where(cond_scalar, x_scalar, y_scalar).unsqueeze(0),
                     at::where(cond_1d, x_1d, y_1d));
  }
}

TEST_CASE( "native test CPU", "[cpu]" ) {
  manual_seed(123, at::Backend::CPU);

  test(CPU(kFloat), CPU(kDouble));
}

TEST_CASE( "native test CUDA", "[cuda]" ) {
  manual_seed(123, at::Backend::CUDA);

  if (at::hasCUDA()) {
    test(CUDA(kFloat), CUDA(kDouble));
  }
}
