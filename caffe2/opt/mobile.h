#ifndef CAFFE2_OPT_MOBILE_H_
#define CAFFE2_OPT_MOBILE_H_

#include "caffe2/core/common.h"
#include "nomnigraph/Representations/NeuralNet.h"

namespace caffe2 {
namespace opt {

TORCH_API void addNNPACK(nom::repr::NNModule* nn, bool low_memory = false);
TORCH_API void fuseNNPACKConvRelu(nom::repr::NNModule* nn);

} // namespace opt
} // namespace caffe2

#endif // CAFFE2_OPT_MOBILE_H_
