#include "caffe2/operators/utility_ops.h"
#include "caffe2/core/operator.h"
#include "caffe2/mkl/mkl_utils.h"
#ifdef CAFFE2_HAS_MKL_DNN

namespace caffe2 {
namespace mkl {

class CopyCPUToMKLOp final : public MKLOperator<float> {
 public:
  using MKLOperator<float>::MKLOperator;
  bool RunOnDevice() override {
    const auto& X = OperatorBase::Input<TensorCPU>(0);
    auto* Y = OperatorBase::OutputBlob(0);
    if (!Y->template IsType<MKLMemory<float>>() ||
        Y->Get<MKLMemory<float>>().dims() != X.dims()) {
      Y->Reset(new MKLMemory<float>(X.dims()));
    }
    Y->GetMutable<MKLMemory<float>>()->CopyFrom(X);
    return true;
  }
};

class CopyMKLToCPUOp final : public MKLOperator<float> {
 public:
  using MKLOperator<float>::MKLOperator;

  bool RunOnDevice() override {
    const auto& X = OperatorBase::Input<MKLMemory<float>>(0);
    auto* Y = OperatorBase::Output<TensorCPU>(0);
    X.CopyTo(Y);
    return true;
  }
};

} // namespace mkl

REGISTER_MKL_OPERATOR(CopyCPUToMKL, mkl::CopyCPUToMKLOp);
REGISTER_MKL_OPERATOR(CopyMKLToCPU, mkl::CopyMKLToCPUOp);

OPERATOR_SCHEMA(CopyCPUToMKL)
    .NumInputs(1)
    .NumOutputs(1)
    .Input(0, "cpu_blob", "The input TensorCPU to copy")
    .Output(0, "mkl_blob", "The output MKLMemory to copy to");
OPERATOR_SCHEMA(CopyMKLToCPU)
    .NumInputs(1)
    .NumOutputs(1)
    .Input(0, "mkl_blob", "The input MKLMemory to copy")
    .Output(0, "cpu_blob", "The output TensorCPU to copy to");

} // namespace caffe2

#endif // CAFFE2_HAS_MKL_DNN
