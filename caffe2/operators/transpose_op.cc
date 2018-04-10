#include "caffe2/operators/transpose_op.h"

#ifdef CAFFE2_USE_MKL
#include "caffe2/mkl/operators/operator_fallback_mkl.h"
#endif // CAFFE2_USE_MKL

namespace caffe2 {

REGISTER_CPU_OPERATOR(Transpose, TransposeOp<CPUContext>);

#ifdef CAFFE2_HAS_MKL_DNN
// Registering in operator_fallback_mkl.cc results in a linker error in
// in opt build related to DoRunWithType().
REGISTER_MKL_OPERATOR(Transpose, mkl::MKLFallbackOp<TransposeOp<CPUContext>>);
#endif // CAFFE2_HAS_MKL_DNN

OPERATOR_SCHEMA(Transpose)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](
        const OperatorDef& def,
        const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      vector<int> axes = helper.GetRepeatedArgument<int>("axes");
      vector<TensorShape> out(1);
      out[0].set_data_type(in[0].data_type());

      if (axes.empty()) {
        for (auto axis = in [0].dims().rbegin(); axis != in[0].dims().rend();
             ++axis) {
          out[0].add_dims(*axis);
        }
      } else {
        auto tensor_size = in[0].dims().size();
        auto valid_axes =
            std::all_of(axes.begin(), axes.end(), [&tensor_size](int& axis) {
              return axis >= 0 && axis < tensor_size;
            });

        CAFFE_ENFORCE(valid_axes, "Axes argument passed in had invalid values");
        CAFFE_ENFORCE(
            axes.size() == tensor_size,
            "Axes argument passed in had the incorrect size");

        for (auto axis = axes.begin(); axis != axes.end(); ++axis) {
          out[0].add_dims(in[0].dims().Get(*axis));
        }
      }

      return out;
    })
    .SetDoc(R"DOC(
Transpose the input tensor similar to numpy.transpose. For example, when
axes=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape
will be (2, 1, 3).
)DOC")
    .Arg(
        "axes",
        "A list of integers. By default, reverse the dimensions, "
        "otherwise permute the axes according to the values given.")
    .Input(0, "data", "An input tensor.")
    .Output(0, "transposed", "Transposed output.")
    .InheritOnnxSchema("Transpose");

class GetTransposeGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  // We will create our own arguments.
  bool CopyArguments() const override {
    return false;
  }
  vector<OperatorDef> GetGradientDefs() override {
    auto ops = SingleGradientDef(
        "Transpose", "", vector<string>{GO(0)}, vector<string>{GI(0)});
    ops[0].mutable_arg()->CopyFrom(Def().arg());
    if (ArgumentHelper::HasArgument(Def(), "axes")) {
      // If axes is specified, we will need to figure out the inverse index.
      const Argument& old_axes = GetArgument(Def(), "axes");
      const int axes_size = old_axes.ints_size();
      Argument* new_arg = GetMutableArgument("axes", false, &ops[0]);
      for (int i = 0; i < axes_size; ++i) {
        new_arg->set_ints(old_axes.ints(i), i);
      }
    }
    return ops;
  }
};

REGISTER_GRADIENT(Transpose, GetTransposeGradient);

} // namespace caffe2
