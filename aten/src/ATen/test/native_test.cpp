#include "ATen/ATen.h"
#include "test_assert.h"

using namespace at;

void assertEqualTensorList(TensorList t1, TensorList t2) {
  ASSERT(t1.size() == t2.size());
  for (size_t i = 0; i < t1.size(); ++i) {
    ASSERT_EQUAL(t1[ i ], t2[ i ]);
  }
}

int main() {
  Type & T = CPU(kFloat);

  auto t = T.randn({3, 3});
  // split
  {
    // test method, type, namespace give same result
    auto splitMethod = t.split(1, 0);
    auto splitType = T.split(t, 1, 0);
    auto splitNs = at::split(t, 1, 0);
    assertEqualTensorList(splitMethod, splitType);
    assertEqualTensorList(splitMethod, splitNs);

    // test rebuilding with cat
    ASSERT_EQUAL(at::cat(splitMethod, 0), t);
  }

  {
    // test method, type, namespace give same result
    auto chunkMethod = t.chunk(3, 0);
    auto chunkType = T.chunk(t, 3, 0);
    auto chunkNs = at::chunk(t, 3, 0);
    assertEqualTensorList(chunkMethod, chunkType);
    assertEqualTensorList(chunkMethod, chunkNs);

    // test rebuilding with cat
    ASSERT_EQUAL(at::cat(chunkMethod, 0), t);
  }

  // stack
  {
    auto x = T.rand({2, 3, 4});
    auto y = T.rand({2, 3, 4});
    auto z = T.rand({2, 3, 4});
    for (int64_t dim = 0; dim < 4; ++dim) {
      auto res = at::stack({x, y, z}, dim);
      auto res_neg = at::stack({x, y, z}, dim - 4);
      std::vector<int64_t> expected_size;
      expected_size.insert(expected_size.end(), x.sizes().begin(), x.sizes().begin() + dim);
      expected_size.insert(expected_size.end(), 3);
      expected_size.insert(expected_size.end(), x.sizes().begin() + dim, x.sizes().end());

      ASSERT_EQUAL(res, res_neg);
      ASSERT(res.sizes().equals(expected_size));
      ASSERT_EQUAL(res.select(dim, 0), x);
      ASSERT_EQUAL(res.select(dim, 1), y);
      ASSERT_EQUAL(res.select(dim, 2), z);
    }
  }

  // size / stride
  {
    auto scalar = T.randn({});
    ASSERT_THROWS(scalar.size(0), "dimension specified as 0 but tensor has no dimensions");
    ASSERT_THROWS(scalar.size(-1), "dimension specified as -1 but tensor has no dimensions");
    ASSERT_THROWS(scalar.stride(0), "dimension specified as 0 but tensor has no dimensions");
    ASSERT_THROWS(scalar.stride(-1), "dimension specified as -1 but tensor has no dimensions");

    auto empty = T.randn({0});
    ASSERT(empty.size(0) == 0);
    ASSERT(empty.size(-1) == 0);
    ASSERT(empty.stride(0) == 1);
    ASSERT(empty.stride(-1) == 1);
  }

  // matmul
  {
    auto scalar = T.randn({});
    auto d1 = T.randn({3});
    auto d2 = T.randn({2, 3});

    // 0-d
    ASSERT_THROWS(scalar.matmul(d2), "both arguments to matmul need to be at least 1D");
    ASSERT_THROWS(d2.matmul(scalar), "both arguments to matmul need to be at least 1D");

    // 1-d
    ASSERT_ALLCLOSE(d1.matmul(d1), d1.dot(d1));
    ASSERT_ALLCLOSE(d2.matmul(d1), d2.mv(d1));
    auto d1o = T.randn({2});
    ASSERT_ALLCLOSE(d1o.matmul(d2), d1o.unsqueeze(0).mm(d2).squeeze(0));

    // 2-d
    auto d2o = T.randn({3, 5});
    ASSERT_ALLCLOSE(d2.matmul(d2o), d2.mm(d2o));

    // > 2-d, 1-d
    auto d3 = T.randn({5, 2, 3});
    ASSERT_ALLCLOSE(d3.matmul(d1), d3.bmm(d1.view({1, 3, 1}).expand({5, 3, 1})).view({5, 2}));
    ASSERT_ALLCLOSE(d1o.matmul(d3), d1o.expand({5, 1, 2}).bmm(d3).view({5, 3}));

    auto d5 = T.randn({3, 2, 4, 2, 3});
    ASSERT_ALLCLOSE(d5.matmul(d1), d5.view({24, 2, 3}).bmm(d1.view({1, 3, 1}).expand({24, 3, 1})).view({3, 2, 4, 2}));
    ASSERT_ALLCLOSE(d1o.matmul(d5), d1o.expand({24, 1, 2}).bmm(d5.view({24, 2, 3})).view({3, 2, 4, 3}));

    // > 2-d, 2-d
    d2 = T.randn({3, 4});
    d2o = T.randn({4, 2});
    ASSERT_ALLCLOSE(d3.matmul(d2), d3.bmm(d2.expand({5, 3, 4})));
    ASSERT_ALLCLOSE(d2o.matmul(d3), d2o.expand({5, 4, 2}).bmm(d3));

    ASSERT_ALLCLOSE(d5.matmul(d2), d5.view({24, 2, 3}).bmm(d2.expand({24, 3, 4})).view({3, 2, 4, 2, 4}));
    ASSERT_ALLCLOSE(d2o.matmul(d5), d2o.expand({24, 4, 2}).bmm(d5.view({24, 2, 3})).view({3, 2, 4, 4, 3}));

    // > 2-d, > 2-d
    auto d5o = T.randn({2, 1, 2, 4, 3, 2});
    auto d5_bmm_view = d5.expand({2, 3, 2, 4, 2, 3}).contiguous().view({48, 2, 3});
    auto d5o_bmm_view = d5o.expand({2, 3, 2, 4, 3, 2}).contiguous().view({48, 3, 2});
    ASSERT_ALLCLOSE(d5.matmul(d5o), d5_bmm_view.bmm(d5o_bmm_view).view({2, 3, 2, 4, 2, 2}));

    // non-expandable case
    auto d5wrong = T.randn({2, 4, 2, 4, 3, 2});
    ASSERT_THROWS(d5.matmul(d5wrong), "must match the size");
  }

  return 0;
}
