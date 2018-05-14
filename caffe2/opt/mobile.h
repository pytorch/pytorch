#ifndef CAFFE2_OPT_MOBILE_H_
#define CAFFE2_OPT_MOBILE_H_

#include "nomnigraph/Representations/NeuralNet.h"

namespace caffe2 {
namespace opt {

void addNNPACK(nom::repr::NNModule* nn, bool low_memory = false);
void fuseNNPACKConvRelu(nom::repr::NNModule* nn);

} // namespace opt
} // namespace caffe2

#endif // CAFFE2_OPT_MOBILE_H_
