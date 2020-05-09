#include <fbgemm/FbgemmConvert.h>
#include "caffe2/contrib/fakelowp/sum_fp16_fake_op.h"
#include "caffe2/operators/elementwise_add_op.h"
#include "caffe2/operators/elementwise_div_op.h"
#include "caffe2/operators/elementwise_mul_op.h"
#include "caffe2/operators/elementwise_sub_op.h"
#include "caffe2/operators/utility_ops.h"

C10_DECLARE_bool(caffe2_fbgemm_fake_fp16_clamp);

namespace caffe2 {

namespace {

int getSizeFromDims(const std::vector<int>& dims) {
  int tot = 1;
  for (auto i = 0; i < dims.size(); i++) {
    tot *= dims[i];
  }
  return tot;
}

template <class OP>
struct FP16PairWiseCPUFunctor : public OP {
  bool Forward(
      const std::vector<int>& A_dims,
      const std::vector<int>& B_dims,
      const float* A,
      const float* B,
      float* C,
      CPUContext* context) const {
    auto A_sz = getSizeFromDims(A_dims);
    auto B_sz = getSizeFromDims(B_dims);

    std::vector<float> A_fp16(A_sz);
    std::vector<float> B_fp16(B_sz);

    fbgemm::RoundToFloat16(
        A, A_fp16.data(), A_sz, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
    fbgemm::RoundToFloat16(
        B, B_fp16.data(), B_sz, FLAGS_caffe2_fbgemm_fake_fp16_clamp);

    OP::Forward(A_dims, B_dims, A_fp16.data(), B_fp16.data(), C, context);
    fbgemm::RoundToFloat16(C, C, A_sz, FLAGS_caffe2_fbgemm_fake_fp16_clamp);

    return true;
  }
};

} // namespace

REGISTER_CPU_OPERATOR(SumFakeFp16, SumFP16FP16AccOp<CPUContext>);

REGISTER_CPU_OPERATOR(
    AddFakeFp16,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        FP16PairWiseCPUFunctor<AddFunctor<CPUContext>>>);

REGISTER_CPU_OPERATOR(
    DivFakeFp16,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        FP16PairWiseCPUFunctor<DivFunctor<CPUContext>>>);

REGISTER_CPU_OPERATOR(
    MulFakeFp16,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        FP16PairWiseCPUFunctor<MulFunctor<CPUContext>>>);

REGISTER_CPU_OPERATOR(
    SubFakeFp16,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        FP16PairWiseCPUFunctor<SubFunctor<CPUContext>>>);

} // namespace caffe2
