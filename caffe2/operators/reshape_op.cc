#include "caffe2/operators/reshape_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Reshape, ReshapeOp<float, CPUContext>);

OPERATOR_SCHEMA(Reshape)
    .NumInputs(1, 2)
    .NumOutputs(2)
    .TensorInferenceFunction(
        [](const OperatorDef& def, const vector<TensorShape>& in) {
          vector<TensorShape> out(2);

          // Do shape inference for old_shape
          out[1].set_data_type(TensorProto::INT64);
          out[1].add_dims(in[0].dims_size());

          ArgumentHelper helper(def);
          if (!helper.HasArgument("shape")) {
            // Cannot do shape inference for reshaped tensor from runtime data.
            CAFFE_ENFORCE_EQ(
                in.size(),
                2,
                "New shape must be specified by either the input blob or the "
                "argument `shape`.");
            out[0].set_unknown_shape(true);
            return out;
          }
          CAFFE_ENFORCE_EQ(
              in.size(),
              1,
              "New shape must not be specified by the input blob and the "
              "argument `shape` at the same time.");

          // Infer the actual new shape
          auto actualNewShape = helper.GetRepeatedArgument<int64_t>("shape");

          // Copy over the dimensions for those that are specified zero
          // and check the eligibility of input
          for (int i = 0; i < actualNewShape.size(); ++i) {
            CAFFE_ENFORCE_GE(
                actualNewShape[i],
                -1,
                "The dimensions in argument `shape` "
                "must not be a negative number.");

            if (actualNewShape[i] == 0) {
              CAFFE_ENFORCE_LT(
                  i,
                  in[0].dims_size(),
                  "Argument `shape` has a dimension set to zero that exceeds "
                  "the original dimension size.");
              actualNewShape[i] = in[0].dims(i);
            }
          }

          // Check if the new shape is valid and fills in the missing dimension
          // specified by -1.
          int64_t totalSize = 1;
          for (const auto d : in[0].dims()) {
            totalSize *= d;
          }
          int64_t size = 1;
          int unknownIdx = -1;
          for (int i = 0; i < actualNewShape.size(); ++i) {
            const auto dim = actualNewShape[i];
            if (dim == -1) {
              CAFFE_ENFORCE(
                  unknownIdx == -1,
                  "Argument `shape` has more than one missing dimension.");
              unknownIdx = i;
            } else {
              size *= dim;
            }
          }

          if (unknownIdx != -1) {
            CAFFE_ENFORCE(
                totalSize % size == 0,
                "Argument `shape` does not agree with the input data.",
                " (",
                totalSize,
                " vs ",
                size,
                ")");
            actualNewShape[unknownIdx] = totalSize / size;
          } else {
            CAFFE_ENFORCE_EQ(
                totalSize,
                size,
                "Argument `shape` does not agree with the input data.",
                " (",
                totalSize,
                " != ",
                size,
                ")");
          }

          out[0].set_data_type(in[0].data_type());
          for (const auto d : actualNewShape) {
            out[0].add_dims(d);
          }
          return out;
        })
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Reshape the input tensor similar to numpy.reshape.

It takes a tensor as input and an optional tensor specifying the new shape.
When the second input is absent, an extra argument `shape` must be specified.
It outputs the reshaped tensor as well as the original shape.

At most one dimension of the new shape can be -1. In this case, the value is
inferred from the size of the tensor and the remaining dimensions. A dimension
could also be 0, in which case the actual dimension value is going to be copied
from the input tensor.
)DOC")
    .Arg("shape", "New shape")
    .Input(0, "data", "An input tensor.")
    .Input(1, "new_shape", "New shape.")
    .Output(0, "reshaped", "Reshaped data.")
    .Output(1, "old_shape", "Original shape.")
    .InheritOnnxSchema("Reshape");

class GetReshapeGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "Reshape",
        "",
        vector<string>{GO(0), O(1)},
        vector<string>{GI(0), "_" + GI(0) + "_dims"});
  }

  // Argument `shape` is no longer needed in backprop.
  bool CopyArguments() const override {
    return false;
  }
};

REGISTER_GRADIENT(Reshape, GetReshapeGradient);

} // namespace caffe2
