#pragma once

#include "caffe2/core/common.h"
#include "caffe2/proto/caffe2_pb.h"
#include "nomnigraph/Representations/NeuralNet.h"

namespace caffe2 {
namespace opt {

// \brief This fuses Cast -> BatchOneHot -> Cast
// into a single call.
void fuseCastBatchOneHot(nom::repr::NNModule* nn);

} // namespace opt
} // namespace caffe2
