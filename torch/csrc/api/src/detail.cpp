#include <ATen/Config.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

#if AT_CUDA_ENABLED()
#include <THC/THCTensorRandom.h>
#include <cuda.h>
#include <cuda_runtime.h>
#endif

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
  at::globalContext().defaultGenerator(at::Backend::CPU).manualSeed(seed);
#if AT_CUDA_ENABLED()
  if (getNumGPUs() > 0) {
    THCRandom_manualSeedAll(at::globalContext().lazyInitCUDA(), seed);
  }
#endif
}

int getNumGPUs() {
#if AT_CUDA_ENABLED()
  int count;
  auto err = cudaGetDeviceCount(&count);
  if (err == cudaErrorNoDevice) {
    return 0;
  } else if (err != cudaSuccess) {
    std::string msg = "CUDA error (";
    msg += std::to_string(static_cast<int>(err));
    msg += "): ";
    msg += cudaGetErrorString(err);
    throw std::runtime_error(msg);
  }
  return count;
#else
  return 0;
#endif
}

bool hasCuda() {
  return getNumGPUs() > 0;
}

bool hasCudnn() {
  return hasCuda() && AT_CUDNN_ENABLED();
}

} // namespace torch
