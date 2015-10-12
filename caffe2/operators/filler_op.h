#ifndef CAFFE2_OPERATORS_FILLER_OP_H_
#define CAFFE2_OPERATORS_FILLER_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/logging.h"

namespace caffe2 {

template <class Context>
class FillerOp : public Operator<Context> {
 public:
  FillerOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
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
      auto* output = Operator<Context>::Output(0);
      if (InputSize()) {
        if (shape_.size() != 0) {
          CAFFE_LOG_ERROR << "Cannot set the shape argument and pass in an input at "
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

  virtual bool Fill(Tensor<Context>* output) = 0;

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

template <typename T, class Context>
class UniformFillOp final : public FillerOp<Context> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  UniformFillOp(const OperatorDef& operator_def, Workspace* ws)
      : FillerOp<Context>(operator_def, ws),
        min_(OperatorBase::template GetSingleArgument<float>("min", 0)),
        max_(OperatorBase::template GetSingleArgument<float>("max", 1)) {
    CAFFE_DCHECK_LT(min_, max_) << "Max value should be bigger than min value.";
  }

  bool Fill(Tensor<Context>* output) override {
    math::RandUniform<T, Context>(
        output->size(), min_, max_,
        output->template mutable_data<T>(), &device_context_);
    return true;
  }

 private:
  T min_;
  T max_;
  DISABLE_COPY_AND_ASSIGN(UniformFillOp);
};

template <typename T, class Context>
class ConstantFillOp final : public FillerOp<Context> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  ConstantFillOp(const OperatorDef& operator_def, Workspace* ws)
      : FillerOp<Context>(operator_def, ws),
        value_(OperatorBase::template GetSingleArgument<float>("value", 0)) {}

  bool Fill(Tensor<Context>* output) override {
    math::Set<T, Context>(
        output->size(), value_, output->template mutable_data<T>(), &device_context_);
    return true;
  }

 private:
  T value_;
  DISABLE_COPY_AND_ASSIGN(ConstantFillOp);
};

template <typename T, class Context>
class GivenTensorFillOp final : public FillerOp<Context> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  GivenTensorFillOp(const OperatorDef& operator_def, Workspace* ws)
      : FillerOp<Context>(operator_def, ws) {
    auto source_values = OperatorBase::template GetRepeatedArgument<float>(
        "values");
    for (float& f : source_values) {
      values_.push_back(static_cast<T>(f));
    }
  }

  bool Fill(Tensor<Context>* output) override {
    CAFFE_DCHECK_EQ(output->size(), values_.size())
        << "output size: " << output->size() << " given size: "
        << values_.size();
    device_context_.template Copy<T, CPUContext, Context>(
        output->size(), values_.data(), output->template mutable_data<T>());
    return true;
  }

 private:
  vector<T> values_;
  DISABLE_COPY_AND_ASSIGN(GivenTensorFillOp);
};

template <typename T, class Context>
class GaussianFillOp final : public FillerOp<Context> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  GaussianFillOp(const OperatorDef& operator_def, Workspace* ws)
      : FillerOp<Context>(operator_def, ws),
        mean_(OperatorBase::template GetSingleArgument<float>("mean", 0)),
        std_(OperatorBase::template GetSingleArgument<float>("std", 1)) {
    CAFFE_DCHECK_GT(std_, 0)
        << "Standard deviation should be nonnegative.";
  }

  bool Fill(Tensor<Context>* output) override {
    math::RandGaussian<T, Context>(
        output->size(), mean_, std_, output->template mutable_data<T>(),
        &device_context_);
    return true;
  }

 private:
  T mean_;
  T std_;
  DISABLE_COPY_AND_ASSIGN(GaussianFillOp);
};

template <typename T, class Context>
class XavierFillOp final : public FillerOp<Context> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  XavierFillOp(const OperatorDef& operator_def, Workspace* ws)
      : FillerOp<Context>(operator_def, ws) {}

  bool Fill(Tensor<Context>* output) override {
    const int fan_in = output->size() / output->dim(0);
    T scale = sqrt(T(3) / fan_in);
    math::RandUniform<T, Context>(
        output->size(), -scale, scale,
        output->template mutable_data<T>(), &device_context_);
    return true;
  }

  DISABLE_COPY_AND_ASSIGN(XavierFillOp);
};

// This is mostly used just as a debugging purpose stuff: it fills a tensor
// sequentially with values 0, 1, 2..., which can then be used to check e.g.
// reshape operations by allowing one to read the indices more easily.
template <typename T, class Context>
class RangeFillOp final : public FillerOp<Context> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  RangeFillOp(const OperatorDef& operator_def, Workspace* ws)
      : FillerOp<Context>(operator_def, ws) {}

  bool Fill(Tensor<Context>* output) override;
  DISABLE_COPY_AND_ASSIGN(RangeFillOp);
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_FILLER_OP_H_
