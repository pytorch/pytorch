#include "caffe2/image/image_input_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(ImageInput, ImageInputOp<CPUContext>);

OPERATOR_SCHEMA(ImageInput)
    .NumInputs(0, 1).NumOutputs(2);

NO_GRADIENT(ImageInput);

}  // namespace caffe2
