#ifndef CAFFE2_OPERATORS_FILLER_OP_H_
#define CAFFE2_OPERATORS_FILLER_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// FillerOp takes in either zero or one input.
//
// If the number of input is 1, the shape will be identical to that of the input
// at run time with optional additional dimensions appended at the end as
// specified by "extra_shape" argument. In that case the "shape" parameter
// should not be set.
//
// If the number of inputs is 0, the full shape must be provided via "shape"
// argument
template <class Context>
class FillerOp : public Operator<Context> {
 public:
  FillerOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        shape_(ToVectorTIndex(OperatorBase::GetRepeatedArgument<int>("shape"))),
        extra_shape_(ToVectorTIndex(
            OperatorBase::GetRepeatedArgument<int>("extra_shape"))) {
    if (InputSize()) {
      if (shape_.size() != 0) {
        CAFFE_THROW(
            "Cannot set the shape argument and pass in an input at "
            "the same time");
      }
    } else if (!extra_shape_.empty()) {
      CAFFE_THROW("Cannot set extra_shape when there is no input");
    }
  }

  virtual ~FillerOp() {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    auto* output = Operator<Context>::Output(0);
    if (InputSize()) {
      auto shape = Input(0).dims();
      shape.insert(shape.end(), extra_shape_.begin(), extra_shape_.end());
      output->Resize(shape);
    } else {
      output->Resize(shape_);
    }
    if (output->size()) {
      return Fill(output);
    } else {
      VLOG(1) << "Output has zero size, skipping filling.";
      return true;
    }
  }

  virtual bool Fill(Tensor<Context>* output) = 0;

 protected:
  vector<TIndex> shape_;
  vector<TIndex> extra_shape_;
};

template <typename T, class Context>
class UniformFillOp final : public FillerOp<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  UniformFillOp(const OperatorDef& operator_def, Workspace* ws)
      : FillerOp<Context>(operator_def, ws),
        min_(OperatorBase::template GetSingleArgument<T>("min", 0)),
        max_(OperatorBase::template GetSingleArgument<T>("max", 1)) {
    DCHECK_LT(min_, max_) << "Max value should be bigger than min value.";
  }

  bool Fill(Tensor<Context>* output) override {
    math::RandUniform<T, Context>(
        output->size(),
        min_,
        max_,
        output->template mutable_data<T>(),
        &context_);
    return true;
  }

 private:
  T min_;
  T max_;
};

template <class Context>
class ConstantFillOp final : public FillerOp<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ConstantFillOp(const OperatorDef& operator_def, Workspace* ws)
      : FillerOp<Context>(operator_def, ws) {
    TensorProto_DataType dtype =
        static_cast<TensorProto_DataType>(OperatorBase::GetSingleArgument<int>(
            "dtype", TensorProto_DataType_FLOAT));

    if (!OperatorBase::HasArgument("dtype") &&
        OperatorBase::HasArgument("value")) {
      // If 'dtype' is not provided, infer type based on the type of 'value'
      // Currently, single argument contains either float, int64 or bytes
      if (OperatorBase::HasSingleArgumentOfType<float>("value")) {
        dtype = TensorProto_DataType_FLOAT;
      } else if (OperatorBase::HasSingleArgumentOfType<int64_t>("value")) {
        dtype = TensorProto_DataType_INT64;
      } else {
        CAFFE_THROW("Argument 'value' is of unexpected type");
      }
      VLOG(1) << "Argument 'dtype' is not provided. Assume the data type is "
              << "the same as that of argument 'value': " << dtype;
    }

    switch (dtype) {
      case TensorProto_DataType_FLOAT:
        body_ = &ConstantFillOp::FillWithType<float>;
        break;
      case TensorProto_DataType_INT32:
        body_ = &ConstantFillOp::FillWithType<int>;
        break;
      case TensorProto_DataType_BOOL:
        body_ = &ConstantFillOp::FillWithType<bool>;
        break;
      case TensorProto_DataType_INT64:
        body_ = &ConstantFillOp::FillWithType<int64_t>;
        break;
      case TensorProto_DataType_UNDEFINED:
        CAFFE_THROW("ConstantFill op cannot have undefined 'dtype' argument");
      // break;
      default:
        CAFFE_THROW("Unexpected 'dtype' argument value: ", dtype);
    }
  }

  bool Fill(Tensor<Context>* output) override {
    return (this->*body_)(output);
  }

  template <typename T>
  bool FillWithType(Tensor<Context>* output) {
    T value = OperatorBase::GetSingleArgument<T>("value", 0);
    math::Set<T, Context>(
        output->size(), value, output->template mutable_data<T>(), &context_);
    return true;
  }

 private:
  bool (ConstantFillOp::*body_)(Tensor<Context>* output);
};

template <typename T, class Context>
class GivenTensorFillOp final : public FillerOp<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  GivenTensorFillOp(const OperatorDef& operator_def, Workspace* ws)
      : FillerOp<Context>(operator_def, ws) {
    auto source_values =
        OperatorBase::template GetRepeatedArgument<T>("values");
    for (T f : source_values) {
      values_.push_back(static_cast<T>(f));
    }
  }

  bool Fill(Tensor<Context>* output) override {
    DCHECK_EQ(output->size(), values_.size())
        << "output size: " << output->size()
        << " given size: " << values_.size();
    context_.template Copy<T, CPUContext, Context>(
        output->size(), values_.data(), output->template mutable_data<T>());
    return true;
  }

 private:
  vector<T> values_;
};

template <typename T, class Context>
class GaussianFillOp final : public FillerOp<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  GaussianFillOp(const OperatorDef& operator_def, Workspace* ws)
      : FillerOp<Context>(operator_def, ws),
        mean_(OperatorBase::template GetSingleArgument<float>("mean", 0)),
        std_(OperatorBase::template GetSingleArgument<float>("std", 1)) {
    DCHECK_GT(std_, 0) << "Standard deviation should be nonnegative.";
  }

  bool Fill(Tensor<Context>* output) override {
    math::RandGaussian<T, Context>(
        output->size(),
        mean_,
        std_,
        output->template mutable_data<T>(),
        &context_);
    return true;
  }

 private:
  T mean_;
  T std_;
};

template <typename T, class Context>
class XavierFillOp final : public FillerOp<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  XavierFillOp(const OperatorDef& operator_def, Workspace* ws)
      : FillerOp<Context>(operator_def, ws) {}

  bool Fill(Tensor<Context>* output) override {
    const int fan_in = output->size() / output->dim32(0);
    T scale = std::sqrt(T(3) / fan_in);
    math::RandUniform<T, Context>(
        output->size(),
        -scale,
        scale,
        output->template mutable_data<T>(),
        &context_);
    return true;
  }
};

template <typename T, class Context>
class MSRAFillOp final : public FillerOp<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  MSRAFillOp(const OperatorDef& operator_def, Workspace* ws)
      : FillerOp<Context>(operator_def, ws) {}

  bool Fill(Tensor<Context>* output) override {
    const int fan_in = output->size() / output->dim32(0);
    T scale = std::sqrt(T(2) / fan_in);
    math::RandUniform<T, Context>(
        output->size(),
        -scale,
        scale,
        output->template mutable_data<T>(),
        &context_);
    return true;
  }
};

// This is mostly used just as a debugging purpose stuff: it fills a tensor
// sequentially with values 0, 1, 2..., which can then be used to check e.g.
// reshape operations by allowing one to read the indices more easily.
template <typename T, class Context>
class RangeFillOp final : public FillerOp<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  RangeFillOp(const OperatorDef& operator_def, Workspace* ws)
      : FillerOp<Context>(operator_def, ws) {}

  bool Fill(Tensor<Context>* output) override;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_FILLER_OP_H_
