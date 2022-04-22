#pragma once

#include "caffe2/core/net.h"
#include "caffe2/core/workspace.h"
#include "caffe2/predictor/InferenceGraph.h"

namespace caffe2 {

void RemoveOpsByType(InferenceGraph& graph_, const std::string& op_type);

} // namespace caffe2
