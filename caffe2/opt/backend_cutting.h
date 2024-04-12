#pragma once

#include "caffe2/core/common.h"
#include "caffe2/proto/caffe2_pb.h"
#include "nomnigraph/Representations/NeuralNet.h"

#include <functional>

namespace caffe2 {
namespace opt {
struct CutResult {
  caffe2::NetDef net;
  int numberOfSubnets{0};
};

TORCH_API void DumpGraph(nom::repr::NNGraph* g, const std::string& fname);
TORCH_API CutResult OptimizeForBackend(
    caffe2::NetDef& net,
    std::function<bool(const caffe2::OperatorDef&)> supports,
    std::function<caffe2::NetDef(const caffe2::NetDef&)> transform_func,
    bool debug = false);
}
} // namespace caffe2
