#include "caffe2/image/image_input_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(ImageInput, ImageInputOp<CPUContext>);

OPERATOR_SCHEMA(ImageInput)
    .NumInputs(0, 1)
    .NumOutputs(2)
    .TensorInferenceFunction(
        [](const OperatorDef& def, const vector<TensorShape>& /* unused */ in) {
          vector<TensorShape> out(2);
          ArgumentHelper helper(def);
          int batch_size = helper.GetSingleArgument<int>("batch_size", 0);
          int crop = helper.GetSingleArgument<int>("crop", -1);
          int color = helper.GetSingleArgument<int>("color", 1);
          CHECK_GT(crop, 0);
          out[0] = CreateTensorShape(
              vector<int>{batch_size, crop, crop, color ? 3 : 1},
              TensorProto::FLOAT);
          out[1] =
              CreateTensorShape(vector<int>{1, batch_size}, TensorProto::INT32);
          return out;
        });

NO_GRADIENT(ImageInput);

}  // namespace caffe2
