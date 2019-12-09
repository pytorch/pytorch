#include "caffe2/operators/elementwise_op_test.h"

#include "caffe2/core/context_gpu.h"
#include "caffe2/core/flags.h"

C10_DECLARE_string(caffe_test_root);

template <>
void CopyVector<caffe2::CUDAContext>(const int N, const bool* x, bool* y) {
  CUDA_CHECK(cudaMemcpy(y, x, N * sizeof(bool), cudaMemcpyHostToDevice));
}

template <>
caffe2::OperatorDef CreateOperatorDef<caffe2::CUDAContext>() {
  caffe2::OperatorDef def;
  def.mutable_device_option()->set_device_type(caffe2::PROTO_CUDA);
  return def;
}

TEST(ElementwiseGPUTest, And) {
  if (!caffe2::HasCudaGPU())
    return;
  elementwiseAnd<caffe2::CUDAContext>();
}

TEST(ElementwiseGPUTest, Or) {
  if (!caffe2::HasCudaGPU())
    return;
  elementwiseOr<caffe2::CUDAContext>();
}

TEST(ElementwiseGPUTest, Xor) {
  if (!caffe2::HasCudaGPU())
    return;
  elementwiseXor<caffe2::CUDAContext>();
}

TEST(ElementwiseGPUTest, Not) {
  if (!caffe2::HasCudaGPU())
    return;
  elementwiseNot<caffe2::CUDAContext>();
}
