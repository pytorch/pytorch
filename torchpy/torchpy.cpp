#include <torchpy.h>
#include <assert.h>
#include <pybind11/embed.h>
#include <stdio.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include "interpreter.h"

namespace torchpy {

std::vector<std::shared_ptr<Interpreter>> interpreters;
std::mutex interpreters_mtx;
size_t num_interpreters = 4;

void init() {
  for (size_t i = 0; i < num_interpreters; i++) {
    interpreters.push_back(std::make_shared<Interpreter>());
  }
}

void finalize() {
  interpreters.clear();
}

size_t load(const char* filename) {
  // for now just load all models into all interpreters
  // eventually, we'll share/dedup tensor data
  size_t model_id;
  for (auto interp : interpreters) {
    model_id = interp->load_model(filename);
  }
  return model_id;
}

at::Tensor forward(size_t model_id, at::Tensor input) {
  interpreters_mtx.lock();
  auto interp = interpreters.back();
  interpreters.pop_back();
  interpreters_mtx.unlock();

  at::Tensor output = interp->forward_model(model_id, input);

  interpreters_mtx.lock();
  interpreters.push_back(interp);
  interpreters_mtx.unlock();

  return output;
}
} // namespace torchpy