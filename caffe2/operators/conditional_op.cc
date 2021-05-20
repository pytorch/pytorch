#include "caffe2/operators/conditional_op.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

template <>
bool ConditionalOp<CPUContext>::RunOnDevice() {
  auto& condition = Input(0);
  auto& dataT = Input(1);
  auto& dataF = Input(2);

  // verify the inputs shape
  CAFFE_ENFORCE_EQ(condition.dim(), 1);
  CAFFE_ENFORCE(dataT.dim() >= 1);
  CAFFE_ENFORCE(dataT.sizes()[0] == condition.sizes()[0]);
  CAFFE_ENFORCE_EQ(dataT.dim(), dataF.dim());
  for (size_t i = 0; i < dataT.sizes().size(); i++) {
    CAFFE_ENFORCE(dataT.sizes().at(i) == dataF.sizes().at(i));
  }
  const auto innerSize = dataT.size_from_dim(1);
  const auto innerSizeBytes = innerSize * dataT.dtype().itemsize();
  CAFFE_ENFORCE(innerSize * dataF.dtype().itemsize() == innerSizeBytes);

  // initialize output shape
  auto* dataOut = Output(0);
  const auto* condPtr = condition.template data<bool>();
  dataOut->ResizeLike(dataT);
  auto* outPtr = (char*)dataOut->raw_mutable_data(dataT.dtype());

  // perform conditional op along first dimension
  const auto* ptrT = (char*)dataT.raw_data();
  const auto* ptrF = (char*)dataF.raw_data();
  for (int64_t i = 0; i < condition.numel(); i++) {
    auto* dst = outPtr + i * innerSizeBytes;
    if (condPtr[i]) {
      context_.CopyItemsSameDevice(
          dataT.dtype(), innerSize, ptrT + i * innerSizeBytes, dst);
    } else {
      context_.CopyItemsSameDevice(
          dataF.dtype(), innerSize, ptrF + i * innerSizeBytes, dst);
    }
  }
  return true;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Conditional, ConditionalOp<CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Conditional)
    .NumInputs(3)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Given a 1-D tensor of boolean values, apply conditional operator along the first
dimension of DataT and DataF and return DataO. Note, DataT and DataF must
have the exact same shape and type.
)DOC")
    .Input(0, "Condition", "Boolean tensor to select DataT or DataF")
    .Input(1, "DataT", "Data to use when True")
    .Input(2, "DataF", "Data to use when False")
    .Output(0, "DataO", "Output data after applying ConditionalOp")
    .IdenticalTypeAndShapeOfInput(1);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
NO_GRADIENT(Conditional);

} // caffe2
