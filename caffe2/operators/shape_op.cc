#include "caffe2/operators/shape_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Shape, ShapeOp<CPUContext>);

OPERATOR_SCHEMA(Shape)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg(
        "axes",
        "(int[]) array of interested axes."
        "If given, this operators only returns the dimension of given axes."
        "Otherwise, the operator returns full dimension.")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper args(def);
      const vector<int>& axes = args.GetRepeatedArgument<int>("axes");
      vector<TensorShape> out(1);
      if (axes.empty()) {
        out[0].add_dims(in[0].dims().size());
      } else {
        out[0].add_dims(axes.size());
      }
      out[0].set_data_type(TensorProto::INT32);
      return out;
    })
    .SetDoc(R"DOC(
        Produce a 1D int64 tensor with the shape of the input tensor.
        If called with an optional argument \"axes\", the result will only
        contain the dimension of specified axes in particular order.)DOC");

SHOULD_NOT_DO_GRADIENT(Shape);

} // namespace caffe2
