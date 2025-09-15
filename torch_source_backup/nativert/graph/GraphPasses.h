#pragma once

#include <torch/nativert/graph/Graph.h>

namespace torch::nativert {

void selectScalarOverload(torch::nativert::Graph* graph);

std::string selectScalarOverloadName(const torch::nativert::Node& node);

} // namespace torch::nativert
