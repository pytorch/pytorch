#pragma once

#include <caffe2/core/workspace.h>
#include <caffe2/proto/caffe2_pb.h>

namespace caffe2 {
/// We have a variant of 2-input Int8Quantize and 4-input Int8FC where the last
/// input points to a blob which contains the y_scale and y_zero_point. It's
/// orginated from online snapshot update but is creating complications for
/// onnxifi flow. Hence this pass is just to absorb the quantization params into
/// the op itself and remove the last input.
void freezeQuantizationParams(NetDef* net, Workspace* ws);
} // namespace caffe2
