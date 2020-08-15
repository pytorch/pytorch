#ifndef CAFFE2_OPERATORS_HALF_FLOAT_OPS_H_
#define CAFFE2_OPERATORS_HALF_FLOAT_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class FloatToHalfOp : public Operator<Context> {
 public:
  explicit FloatToHalfOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        clip_(this->template GetSingleArgument<bool>("clip", false)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 private:
  bool clip_;
};

template <class Context>
class HalfToFloatOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(HalfToFloatOp);

  bool RunOnDevice() override;
};

class Float16ConstantFillOp : public Operator<CPUContext> {
 public:
  template <class... Args>
  explicit Float16ConstantFillOp(Args&&... args)
      : Operator<CPUContext>(std::forward<Args>(args)...),
        shape_(this->template GetRepeatedArgument<int64_t>("shape")) {}

  USE_OPERATOR_FUNCTIONS(CPUContext);
  virtual ~Float16ConstantFillOp() {}

  bool RunOnDevice() override;

 private:
  vector<int64_t> shape_;
};

class Float16UniformFillOp : public Operator<CPUContext> {
 public:
  template <class... Args>
  explicit Float16UniformFillOp(Args&&... args)
      : Operator<CPUContext>(std::forward<Args>(args)...),
        shape_(this->template GetRepeatedArgument<int64_t>("shape")),
        min_(this->template GetSingleArgument<float>("min", 0)),
        max_(this->template GetSingleArgument<float>("max", 1)) {
    if (InputSize() == 3) {
      CAFFE_ENFORCE(
          !this->template HasSingleArgumentOfType<float>("min"),
          "Cannot set both min arg and min input blob");
      CAFFE_ENFORCE(
          !this->template HasSingleArgumentOfType<float>("max"),
          "Cannot set both max arg and max input blob");
    } else {
      CAFFE_ENFORCE_LT(
          min_, max_, "Max value should be bigger than min value.");
    }
  }

  USE_OPERATOR_FUNCTIONS(CPUContext);
  virtual ~Float16UniformFillOp() {}

  bool RunOnDevice() override;

 private:
  vector<int64_t> shape_;
  float min_;
  float max_;
};

inline std::vector<TensorShape> Float16FillerTensorInference(
    const OperatorDef& def,
    const vector<TensorShape>& in) {
  vector<TensorShape> out(1);
  ArgumentHelper helper(def);
  out[0].set_data_type(static_cast<TensorProto_DataType>(
      helper.GetSingleArgument<int>("dtype", TensorProto_DataType_FLOAT16)));
  auto shape = helper.GetRepeatedArgument<int>("shape");
  for (int d : shape) {
    out[0].add_dims(d);
  }
  return out;
}

} // namespace caffe2

#endif // CAFFE2_OPERATORS_HALF_FLOAT_OPS_H_
