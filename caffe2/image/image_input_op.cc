#include "caffe2/image/image_input_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(ImageInput, ImageInputOp<CPUContext>);

}  // namespace caffe2
