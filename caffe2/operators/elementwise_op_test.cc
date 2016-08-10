#include "caffe2/operators/elementwise_op_test.h"

#include "caffe2/core/flags.h"

CAFFE2_DECLARE_string(caffe_test_root);

template <>
void CopyVector<caffe2::CPUContext>(const int N, const bool* x, bool* y) {
  memcpy(y, x, N * sizeof(bool));
}

TEST(ElementwiseCPUTest, And) {
  elementwiseAnd<caffe2::CPUContext>();
}

TEST(ElementwiseTest, Or) {
  elementwiseOr<caffe2::CPUContext>();
}

TEST(ElementwiseTest, Xor) {
  elementwiseXor<caffe2::CPUContext>();
}

TEST(ElementwiseTest, Not) {
  elementwiseNot<caffe2::CPUContext>();
}
