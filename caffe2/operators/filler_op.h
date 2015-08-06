#ifndef CAFFE2_OPERATORS_FILLER_OP_H_
#define CAFFE2_OPERATORS_FILLER_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "glog/logging.h"

namespace caffe2 {

template <typename dtype, class DeviceContext>
class FillerOp : public Operator<dtype, DeviceContext> {
 public:
  FillerOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<dtype, DeviceContext>(operator_def, ws),
        shape_(OperatorBase::GetRepeatedArgument<int>("shape")),
        run_once_(OperatorBase::GetSingleArgument<int>("run_once", true)),
        already_run_(false) {}
  virtual ~FillerOp() {}
  USE_OPERATOR_BASE_FUNCTIONS;

  bool RunOnDevice() override {
    if (run_once_ && already_run_) {
      return true;
    } else {
      already_run_ = true;
      auto* output = Operator<dtype, DeviceContext>::Output(0);
      if (InputSize()) {
        if (shape_.size() != 0) {
          LOG(ERROR) << "Cannot set the shape argument and pass in an input at "
                        "the same time.";
          return false;
        }
        output->ReshapeLike(Input(0));
      } else {
        output->Reshape(shape_);
      }
      return Fill(output);
    }
  }

  virtual bool Fill(Tensor<dtype, DeviceContext>* output) = 0;

 protected:
  vector<int> shape_;
  bool run_once_;
  bool already_run_;
  // FillerOp takes in either zero or one input. If the number of input is
  // 1, the shape will be identical to that of the input at run time, and
  // in that case the "shape" parameter should not be set.
  INPUT_OUTPUT_STATS(0, 1, 1, 1);
  DISABLE_COPY_AND_ASSIGN(FillerOp);
};

template <typename dtype, class DeviceContext>
class UniformFillOp final : public FillerOp<dtype, DeviceContext> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  UniformFillOp(const OperatorDef& operator_def, Workspace* ws)
      : FillerOp<dtype, DeviceContext>(operator_def, ws),
        min_(OperatorBase::template GetSingleArgument<float>("min", 0)),
        max_(OperatorBase::template GetSingleArgument<float>("max", 1)) {
    DCHECK_LT(min_, max_) << "Max value should be bigger than min value.";
  }

  bool Fill(Tensor<dtype, DeviceContext>* output) override {
    math::RandUniform<dtype, DeviceContext>(
        output->size(), min_, max_,
        output->mutable_data(), &device_context_);
    return true;
  }

 private:
  dtype min_;
  dtype max_;
  DISABLE_COPY_AND_ASSIGN(UniformFillOp);
};

template <typename dtype, class DeviceContext>
class ConstantFillOp final : public FillerOp<dtype, DeviceContext> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  ConstantFillOp(const OperatorDef& operator_def, Workspace* ws)
      : FillerOp<dtype, DeviceContext>(operator_def, ws),
        value_(OperatorBase::template GetSingleArgument<float>("value", 0)) {}

  bool Fill(Tensor<dtype, DeviceContext>* output) override {
    math::Set<dtype, DeviceContext>(
        output->size(), value_, output->mutable_data(), &device_context_);
    return true;
  }

 private:
  dtype value_;
  DISABLE_COPY_AND_ASSIGN(ConstantFillOp);
};

template <typename dtype, class DeviceContext>
class GivenTensorFillOp final : public FillerOp<dtype, DeviceContext> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  GivenTensorFillOp(const OperatorDef& operator_def, Workspace* ws)
      : FillerOp<dtype, DeviceContext>(operator_def, ws) {
    auto source_values = OperatorBase::template GetRepeatedArgument<float>(
        "values");
    for (float& f : source_values) {
      values_.push_back(static_cast<dtype>(f));
    }
  }

  bool Fill(Tensor<dtype, DeviceContext>* output) override {
    DCHECK_EQ(output->size(), values_.size())
        << "output size: " << output->size() << " given size: "
        << values_.size();
    device_context_.template Copy<dtype, CPUContext, DeviceContext>(
        output->size(), values_.data(), output->mutable_data());
    return true;
  }

 private:
  vector<dtype> values_;
  DISABLE_COPY_AND_ASSIGN(GivenTensorFillOp);
};

template <typename dtype, class DeviceContext>
class GaussianFillOp final : public FillerOp<dtype, DeviceContext> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  GaussianFillOp(const OperatorDef& operator_def, Workspace* ws)
      : FillerOp<dtype, DeviceContext>(operator_def, ws),
        mean_(OperatorBase::template GetSingleArgument<float>("mean", 0)),
        std_(OperatorBase::template GetSingleArgument<float>("std", 1)) {
    DCHECK_GT(std_, 0)
        << "Standard deviation should be nonnegative.";
  }

  bool Fill(Tensor<dtype, DeviceContext>* output) override {
    math::RandGaussian<dtype, DeviceContext>(
        output->size(), mean_, std_, output->mutable_data(),
        &device_context_);
    return true;
  }

 private:
  dtype mean_;
  dtype std_;
  DISABLE_COPY_AND_ASSIGN(GaussianFillOp);
};

template <typename dtype, class DeviceContext>
class XavierFillOp final : public FillerOp<dtype, DeviceContext> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  XavierFillOp(const OperatorDef& operator_def, Workspace* ws)
      : FillerOp<dtype, DeviceContext>(operator_def, ws) {}

  bool Fill(Tensor<dtype, DeviceContext>* output) override {
    const int fan_in = output->size() / output->dim(0);
    dtype scale = sqrt(dtype(3) / fan_in);
    math::RandUniform<dtype, DeviceContext>(
        output->size(), -scale, scale,
        output->mutable_data(), &device_context_);
    return true;
  }

  DISABLE_COPY_AND_ASSIGN(XavierFillOp);
};

// This is mostly used just as a debugging purpose stuff: it fills a tensor
// sequentially with values 0, 1, 2..., which can then be used to check e.g.
// reshape operations by allowing one to read the indices more easily.
template <typename dtype, class DeviceContext>
class RangeFillOp final : public FillerOp<dtype, DeviceContext> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  RangeFillOp(const OperatorDef& operator_def, Workspace* ws)
      : FillerOp<dtype, DeviceContext>(operator_def, ws) {}

  bool Fill(Tensor<dtype, DeviceContext>* output) override;
  DISABLE_COPY_AND_ASSIGN(RangeFillOp);
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_FILLER_OP_H_
