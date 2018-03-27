#include "caffe2/operators/shape_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Shape, ShapeOp<CPUContext>);

OPERATOR_SCHEMA(Shape)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /*def*/,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      out[0].add_dims(in[0].dims().size());
      out[0].set_data_type(TensorProto::INT32);
      return out;
    })
    .SetDoc("Produce a 1D int64 tensor with the shape of the input tensor.");

SHOULD_NOT_DO_GRADIENT(Shape);

} // namespace caffe2
