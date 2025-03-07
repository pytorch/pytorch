#pragma once

#include "torch/csrc/runtime/graph/Graph.h"

namespace torch::runtime {

void selectScalarOverload(Graph* graph);

std::string selectScalarOverloadName(const Node& node);

} // namespace torch::runtime
