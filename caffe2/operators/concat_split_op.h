#ifndef CAFFE2_OPERATORS_CONCAT_SPLIT_OP_H_
#define CAFFE2_OPERATORS_CONCAT_SPLIT_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {
inline int GetDimFromOrderString(const string& str) {
  auto order = StringToStorageOrder(str);
  switch(order) {
  case StorageOrder::NHWC:
    return 3;
  case StorageOrder::NCHW:
    return 1;
  default:
    LOG(FATAL) << "Unsupported storage order: " << str;
    return -1;
  }
}
}  // namespace

template <class Context>
class SplitOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SplitOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        split_(OperatorBase::GetRepeatedArgument<int>("split")) {
    CHECK(OperatorBase::HasArgument("axis") ^
                OperatorBase::HasArgument("order"))
        << "You should either specify the dim to split, or the order "
           "in the case of 4-D images.";
    if (OperatorBase::HasArgument("axis")) {
      axis_ = OperatorBase::GetSingleArgument<int>("axis", -1);
    } else {
      axis_ = GetDimFromOrderString(
          OperatorBase::GetSingleArgument<string>("order", ""));
   }
    CHECK_GE(axis_, 0);
  }

  bool RunOnDevice() override;

 protected:
  int axis_;
  vector<int> split_;
  // Input: X, optionally split
  // The split tensor is stored in CPU.
};

template <class Context>
class ConcatOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ConcatOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    CHECK(OperatorBase::HasArgument("axis") ^
                OperatorBase::HasArgument("order"))
        << "You should either specify the dim to split, or the order "
           "in the case of 4-D images.";
    if (OperatorBase::HasArgument("axis")) {
      axis_ = OperatorBase::GetSingleArgument<int>("axis", -1);
    } else {
      axis_ = GetDimFromOrderString(
          OperatorBase::GetSingleArgument<string>("order", ""));
    }
    CHECK_GE(axis_, 0);
  }


  bool RunOnDevice() override;

 protected:
  int axis_;
  // Input: a number of tensors. Output: Y, split
  // The split are stored in CPU.
};


// Implementations
template <class Context>
bool SplitOp<Context>::RunOnDevice() {
  auto& input = Input(0);
  const int* axis_data;
  if (InputSize() == 2) {
    // We obtain split from the input tensor.
    CHECK_EQ(split_.size(), 0)
        << "If you set split with an input blob, do not pass in "
        << "split in the argument.";
    auto& split_tensor = OperatorBase::Input<TensorCPU>(1);
    CHECK_EQ(split_tensor.size(), OutputSize());
    axis_data = split_tensor.template data<int>();
  } else {
    // We obtain split from the parameters.
    CHECK_EQ(split_.size(), OutputSize());
    axis_data = split_.data();
  }
  CHECK_LT(axis_, input.ndim());
  const int input_channels = input.dim32(axis_);
  CHECK_EQ(std::accumulate(axis_data, axis_data + OutputSize(), 0),
           input_channels)
      << "Sum of split dimensions do not match: should be " << input_channels;
  int input_offset = 0;
  vector<TIndex> output_dims(input.dims());
  int before = 1, after = 1;
  for (int i = 0; i < axis_; ++i) {
    before *= input.dim32(i);
  }
  for (int i = axis_ + 1; i < input.ndim(); ++i) {
    after *= input.dim32(i);
  }
  for (int i = 0; i < OutputSize(); ++i) {
    auto* output = Output(i);
    output_dims[axis_] = axis_data[i];
    output->Resize(output_dims);
    math::CopyMatrix<Context>(
        input.itemsize(), before, axis_data[i] * after,
        static_cast<const char*>(input.raw_data()) + input_offset,
        input.dim32(axis_) * after, output->raw_mutable_data(input.meta()),
        axis_data[i] * after, &context_);
    input_offset += axis_data[i] * after * input.itemsize();
  }
  return true;
}

template <class Context>
bool ConcatOp<Context>::RunOnDevice() {
  auto* output = Output(0);
  TensorCPU* split = OperatorBase::Output<TensorCPU>(1);
  split->Resize(vector<TIndex>(1, InputSize()));
  int* axis_data = split->template mutable_data<int>();
  int output_channels = 0;
  for (int i = 0; i < InputSize(); ++i) {
    axis_data[i] = Input(i).dim32(axis_);
    output_channels += axis_data[i];
  }
  auto& input_zero = Input(0);
  vector<TIndex> output_dims(input_zero.dims());
  output_dims[axis_] = output_channels;
  output->Resize(output_dims);
  int output_offset = 0;
  int before = 1, after = 1;
  for (int i = 0; i < axis_; ++i) {
    before *= input_zero.dim32(i);
  }
  for (int i = axis_ + 1; i < input_zero.ndim(); ++i) {
    after *= input_zero.dim32(i);
  }
  for (int i = 0; i < InputSize(); ++i) {
    auto& input = Input(i);
    // TODO: check if the input dims are compatible.
    math::CopyMatrix<Context>(
        input.itemsize(), before, input.dim32(axis_) * after, input.raw_data(),
        input.dim32(axis_) * after,
        static_cast<char*>(output->raw_mutable_data(input.meta()))
            + output_offset,
        output_channels * after, &context_);
    output_offset += input.dim32(axis_) * after * input.itemsize();
  }
  return true;
}

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_CONCAT_SPLIT_OP_H_
