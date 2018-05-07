#pragma once

#include "caffe2/core/common.h"
#include "caffe2/proto/caffe2.pb.h"
#include "nomnigraph/Representations/NeuralNet.h"

#include <functional>

namespace caffe2 { namespace opt {

// Fuse convuluation with relu if applicable
void FuseConvRelu(nom::repr::NNModule* nn, std::function<bool(const repr::Conv& conv)> should_fuse);

}}
