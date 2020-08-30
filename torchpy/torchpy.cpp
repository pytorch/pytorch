#include <torchpy.h>
#include <assert.h>
#include <pybind11/embed.h>
#include <stdio.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <iostream>
#include <mutex>
#include <thread>
#include "interpreter.h"

namespace torchpy {

std::vector<Interpreter> interpreters;

void init() {
  // interpreters.push_back(Interpreter());

  // // Make loader importable
  // for (auto interp : interpreters) {
  //   interp.run_some_python("import sys; sys.path.append('torchpy')");
  // }
}

void finalize() {
  interpreters.clear();
}

bool load(const char* filename) {
  // for now just load all models into all interpreters
  // eventually, we'll share/dedup tensor data
  // for (auto interp : interpreters) {
  // interp.load_model(filename);
  // }
  return true;
}

at::Tensor forward(at::Tensor input) {
  // at::Tensor output = interpreters[0].forward(input);
  // return output;
  return input;
}
} // namespace torchpy