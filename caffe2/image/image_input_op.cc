#include "caffe2/image/image_input_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(ImageInput, ImageInputOp<CPUContext>);

NO_GRADIENT(ImageInput);

}  // namespace caffe2
