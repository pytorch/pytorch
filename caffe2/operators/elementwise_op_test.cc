#include "caffe2/operators/elementwise_op_test.h"

#include "caffe2/core/flags.h"

C10_DECLARE_string(caffe_test_root);

template <>
void CopyVector<caffe2::CPUContext, bool>(const int N, const bool* x, bool* y) {
  memcpy(y, x, N * sizeof(bool));
}

template <>
void CopyVector<caffe2::CPUContext, int32_t>(
    const int N,
    const int32_t* x,
    int32_t* y) {
  memcpy(y, x, N * sizeof(int32_t));
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

TEST(ElementwiseTest, EQ) {
  elementwiseEQ<caffe2::CPUContext>();
}
