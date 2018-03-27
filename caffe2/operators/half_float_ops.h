#ifndef CAFFE2_OPERATORS_HALF_FLOAT_OPS_H_
#define CAFFE2_OPERATORS_HALF_FLOAT_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class FloatToHalfOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(FloatToHalfOp);

  bool RunOnDevice() override;
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
  Float16ConstantFillOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        shape_(
            ToVectorTIndex(OperatorBase::GetRepeatedArgument<int>("shape"))) {}

  USE_OPERATOR_FUNCTIONS(CPUContext);
  virtual ~Float16ConstantFillOp() {}

  bool RunOnDevice() override;

 private:
  vector<TIndex> shape_;
};

inline std::vector<TensorShape> Float16FillerTensorInference(
    const OperatorDef& def,
    const vector<TensorShape>& in) {
  vector<TensorShape> out(1);
  ArgumentHelper helper(def);
  out[0].set_data_type(static_cast<TensorProto_DataType>(
      helper.GetSingleArgument<int>("dtype", TensorProto_DataType_FLOAT)));
  auto shape = helper.GetRepeatedArgument<int>("shape");
  for (int d : shape) {
    out[0].add_dims(d);
  }
  return out;
}

} // namespace caffe2

#endif // CAFFE2_OPERATORS_HALF_FLOAT_OPS_H_
