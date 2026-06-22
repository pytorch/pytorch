#pragma once

#include <torch/nativert/graph/passes/pass_manager/GraphPassRegistry.h>

namespace torch::nativert {

using GraphPassIdentifier = std::string;

class GraphPassPipeline : public std::vector<GraphPassIdentifier> {
 public:
  using std::vector<GraphPassIdentifier>::vector;

  void push_front(GraphPassIdentifier pass) {
    std::vector<GraphPassIdentifier>::insert(begin(), std::move(pass));
  }

  // concats the passed pipeline to the end of the current
  // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
  void concat(GraphPassPipeline&& other) {
    std::move(other.begin(), other.end(), std::back_inserter(*this));
  }
};

} // namespace torch::nativert
