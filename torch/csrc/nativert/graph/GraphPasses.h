#pragma once

#include "torch/csrc/nativert/graph/Graph.h"

namespace torch::nativert {

void selectScalarOverload(Graph* graph);

std::string selectScalarOverloadName(const Node& node);

} // namespace torch::nativert
