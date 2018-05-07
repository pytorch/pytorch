#pragma once

#include "caffe2/core/common.h"
#include "caffe2/proto/caffe2.pb.h"
#include "nomnigraph/Representations/NeuralNet.h"

#include <functional>

namespace caffe2 {
namespace opt {

// Fuse convuluation with relu if applicable
// \param nn Neural network module to be modified in place
// \param should_fuse Given a conv op, check whether we want to fuse it with
// subsequent relu or not 
// \param postprocess Functor to postprocess the conv node,
// attaching additional attributes if necessary
void fuseConvRelu(
    nom::repr::NNModule* nn,
    std::function<bool(const nom::repr::Conv& conv)> should_fuse,
    std::function<void(nom::repr::NNGraph::NodeRef conv_node)> postprocess);
} // namespace opt
} // namespace caffe2
