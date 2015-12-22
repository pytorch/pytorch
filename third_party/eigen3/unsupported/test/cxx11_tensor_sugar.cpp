#include "main.h"

#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;
using Eigen::RowMajor;

static void test_comparison_sugar() {
  // we already trust comparisons between tensors, we're simply checking that
  // the sugared versions are doing the same thing
  Tensor<int, 3> t(6, 7, 5);

  t.setRandom();
  // make sure we have at least one value == 0
  t(0,0,0) = 0;

  Tensor<bool,0> b;

#define TEST_TENSOR_EQUAL(e1, e2) \
  b = ((e1) == (e2)).all();       \
  VERIFY(b())

#define TEST_OP(op) TEST_TENSOR_EQUAL(t op 0, t op t.constant(0))

  TEST_OP(==);
  TEST_OP(!=);
  TEST_OP(<=);
  TEST_OP(>=);
  TEST_OP(<);
  TEST_OP(>);
#undef TEST_OP
#undef TEST_TENSOR_EQUAL
}

void test_cxx11_tensor_sugar()
{
  CALL_SUBTEST(test_comparison_sugar());
}
