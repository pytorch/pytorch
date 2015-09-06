#ifndef CAFFE2_OPERATORS_DEPTH_SPLIT_OP_H_
#define CAFFE2_OPERATORS_DEPTH_SPLIT_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename dtype, class DeviceContext>
class DepthSplitOp final : public Operator<dtype, DeviceContext> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  DepthSplitOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<dtype, DeviceContext>(operator_def, ws),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NHWC"))),
        dimensions_(
            OperatorBase::GetRepeatedArgument<int>("dimensions")) {}
  bool RunOnDevice() override;

 protected:
  StorageOrder order_;
  vector<int> dimensions_;
  // Input: X, optionally dimensions
  // The dimensions are stored in CPU.
  INPUT_OUTPUT_STATS(1, 2, 1, INT_MAX);
  DISABLE_COPY_AND_ASSIGN(DepthSplitOp);
};

template <typename dtype, class DeviceContext>
class DepthConcatOp final : public Operator<dtype, DeviceContext> {
 public:
  DepthConcatOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<dtype, DeviceContext>(operator_def, ws),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NHWC"))) {}
  USE_OPERATOR_BASE_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  StorageOrder order_;
  // Input: a number of tensors. Output: Y, dimensions
  // The dimensions are stored in CPU.
  INPUT_OUTPUT_STATS(1, INT_MAX, 2, 2);
  DISABLE_COPY_AND_ASSIGN(DepthConcatOp);
};


// Implementations
template <typename dtype, class DeviceContext>
bool DepthSplitOp<dtype, DeviceContext>::RunOnDevice() {
  auto& input = Input(0);
  const int* dim_data;
  if (InputSize() == 2) {
    // We obtain dimensions from the input tensor.
    CHECK_EQ(dimensions_.size(), 0)
        << "If you set dimensions with an input blob, do not pass in "
        << "dimensions in the argument.";
    auto& dimensions_tensor =
        OperatorBase::Input<Tensor<int, CPUContext> >(1);
    CHECK_EQ(dimensions_tensor.size(), OutputSize());
    dim_data = dimensions_tensor.data();
  } else {
    // We obtain dimensions from the parameters.
    CHECK_EQ(dimensions_.size(), OutputSize());
    dim_data = dimensions_.data();
  }
  const int input_channels =
      (order_ == StorageOrder::NCHW ? input.dim(1) : input.dim(3));
  CHECK_EQ(std::accumulate(dim_data, dim_data + OutputSize(), 0),
           input_channels)
      << "Dimensions do not match: should be " << input_channels;
  int input_offset = 0;
  for (int i = 0; i < OutputSize(); ++i) {
    auto* output = Output(i);
    int M, N, lda;
    switch (order_) {
      case StorageOrder::NCHW:
        output->Reshape(vector<int>{
            input.dim(0), dim_data[i], input.dim(2), input.dim(3)});
        M = input.dim(0);
        N = dim_data[i] * input.dim(2) * input.dim(3);
        lda = input.size() / input.dim(0);
        break;
      case StorageOrder::NHWC:
        output->Reshape(vector<int>{
            input.dim(0), input.dim(1), input.dim(2), dim_data[i]});
        M = input.dim(0) * input.dim(1) * input.dim(2);
        N = dim_data[i];
        lda = input.dim(3);
        break;
      default:
        LOG(FATAL) << "Unsupported storage order: " << order_;
    }
    math::CopyMatrix<dtype, DeviceContext>(
        M, N, input.data() + input_offset, lda, output->mutable_data(), N,
        &device_context_);
    input_offset += N;
  }
  return true;
}

template <typename dtype, class DeviceContext>
bool DepthConcatOp<dtype, DeviceContext>::RunOnDevice() {
  auto* output = Output(0);
  Tensor<int, CPUContext>* dimensions =
      OperatorBase::Output<Tensor<int, CPUContext> >(1);
  dimensions->Reshape(vector<int>(1, InputSize()));
  int* dim_data = dimensions->mutable_data();
  int output_channels = 0;
  for (int i = 0; i < InputSize(); ++i) {
    dim_data[i] =
        (order_ == StorageOrder::NCHW ? Input(i).dim(1) : Input(i).dim(3));
    output_channels += dim_data[i];
  }
  auto& input_zero = Input(0);
  output->Reshape(vector<int>{
      input_zero.dim(0),
      order_ == StorageOrder::NCHW ? output_channels : input_zero.dim(1),
      order_ == StorageOrder::NCHW ? input_zero.dim(2) : input_zero.dim(2),
      order_ == StorageOrder::NCHW ? input_zero.dim(3) : output_channels});
  int output_offset = 0;
  for (int i = 0; i < InputSize(); ++i) {
    auto& input = Input(i);
    int M, N, ldb;
    switch (order_) {
      case StorageOrder::NCHW:
        CHECK_EQ(input.dim(0), output->dim(0));
        CHECK_EQ(input.dim(2), output->dim(2));
        CHECK_EQ(input.dim(3), output->dim(3));
        M = input.dim(0);
        N = input.size() / M;
        ldb = output->size() / output->dim(0);
        break;
      case StorageOrder::NHWC:
        CHECK_EQ(input.dim(0), output->dim(0));
        CHECK_EQ(input.dim(1), output->dim(1));
        CHECK_EQ(input.dim(2), output->dim(2));
        M = input.dim(0) * input.dim(1) * input.dim(2);
        N = input.dim(3);
        ldb = output->dim(3);
        break;
      default:
        LOG(FATAL) << "Unsupported storage order: " << order_;
    }
    math::CopyMatrix<dtype, DeviceContext>(
        M, N, input.data(), N, output->mutable_data() + output_offset, ldb,
        &device_context_);
    output_offset += N;
  }
  return true;
}

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_DEPTH_SPLIT_OP_H_
