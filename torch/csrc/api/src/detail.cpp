#include <ATen/Config.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

#include "torch/detail.h"

namespace torch {
namespace detail {
tag::Engine engine;
}

void backward(Variable loss, bool keep_graph) {
  tag::edge_list edgelst;
  tag::variable_list varlst;
  edgelst.emplace_back(loss.grad_fn(), loss.output_nr());
  varlst.emplace_back(Var(at::ones_like(loss.data()), false));
  // create_graph should be set to true when we want to support double bwd
  detail::engine.execute(edgelst, varlst, keep_graph, false);
}

void backward(Tensor loss, bool keep_graph) {
  Variable tmp(loss);
  backward(tmp, keep_graph);
}

void setSeed(uint64_t seed) {
  // TODO: Move this to at::Context
  at::globalContext().defaultGenerator(at::Backend::CPU).manualSeed(seed);
  if (at::globalContext().hasCUDA()) {
    at::globalContext().defaultGenerator(at::Backend::CUDA).manualSeedAll(seed);
  }
}

int getNumGPUs() {
  return at::globalContext().getNumGPUs();
}

bool hasCuda() {
  // NB: the semantics of this are different from at::globalContext().hasCUDA();
  // ATen's function tells you if you have a working driver and CUDA build,
  // whereas this function also tells you if you actually have any GPUs.
  return getNumGPUs() > 0;
}

bool hasCudnn() {
  return hasCuda() && at::globalContext().hasCuDNN();
}

} // namespace torch
