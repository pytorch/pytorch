#pragma once

#include "caffe2/core/net.h"
#include "caffe2/core/workspace.h"
#include "caffe2/predictor/InferenceGraph.h"

namespace caffe2 {


// Make all operators of a given type inplace when possible
void InPlaceOps(const InferenceGraph& graph_, const std::string& op_type);

void RemoveOpsByType(const InferenceGraph& graph_, const std::string& op_type);

} // namespace caffe2
