#pragma once

#include <TH/TH.h>
#include <ATen/Context.h>
#include <torch/types.h>

namespace torch {

/// Sets the global random seed for all newly created CPU and CUDA tensors.
using at::manual_seed;

// yf225 TODO: implement torch::get_rng_state() and torch::set_rng_state()
// yf225 TODO: we need to add torch::get_rng_state and torch::set_rng_state to C++ API,
// following torch/random.py, torch/cuda/random.py and torch/csrc/Generator.cpp
// yf225 TODO: Question: how to implement for CUDA?

Tensor get_rng_state(torch::Device device = torch::Device(torch::kCPU)) {
  at::Generator* default_generator = torch::globalContext().defaultGenerator(device);

  auto tensor = torch::empty({0}, torch::device(torch::kCPU).dtype(torch::kByte));
  if (default_generator->device().type() == torch::kCPU) {
    THByteTensor_getRNGState(default_generator, (THByteTensor*)(tensor.unsafeGetTensorImpl()));
  } else {
#ifdef USE_CUDA
    TORCH_INTERNAL_ASSERT(default_generator->device().type() == torch::kCUDA);
    THCRandom_getRNGState(default_generator, (THByteTensor*)(tensor.unsafeGetTensorImpl()));
#else
    TORCH_INTERNAL_ASSERT(false, "libtorch not compiled with CUDA");
#endif 
  }
  return std::move(tensor);
}

void set_rng_state(Tensor new_state, torch::Device device = torch::Device(torch::kCPU)) {
  auto tensor = new_state.clone();
  at::Generator* default_generator = torch::globalContext().defaultGenerator(device);

  if (tensor.layout() != kStrided || tensor.device().type() != kCPU || tensor.scalar_type() != kByte) {
    auto type_name = torch::utils::type_to_string(tensor.type());
    throw TypeError("expected a torch::ByteTensor, but got %s", type_name.c_str());
  }
  if (default_generator->device().type() == torch::kCPU) {
    THByteTensor_setRNGState(default_generator, (THByteTensor*)tensor.unsafeGetTensorImpl());
  } else {
#ifdef USE_CUDA
    TORCH_INTERNAL_ASSERT(default_generator->device().type() == torch::kCUDA);
    THCRandom_setRNGState(default_generator, (THByteTensor*)tensor.unsafeGetTensorImpl());
#else
    TORCH_INTERNAL_ASSERT(false, "libtorch not compiled with CUDA");
#endif
  }
}

namespace cuda {

Tensor get_rng_state(torch::Device device = torch::Device(torch::kCUDA)) {
  return torch::get_rng_state(device);
}

void set_rng_state(Tensor new_state, torch::Device device = torch::Device(torch::kCUDA)) {
  torch::set_rng_state(new_state, device);
}

}

}  // namespace torch

