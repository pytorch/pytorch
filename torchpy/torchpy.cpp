#include <torchpy.h>
#include <Python.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>

void torchpy::init() {
  Py_Initialize();
  PyRun_SimpleString(
      "from time import time,ctime\n"
      "print('Today is',ctime(time()))\n");
  Py_Finalize();
}

Module torchpy::load(char* filename) {
  return torch::jit::load(filename);
}

std::vector<torch::jit::IValue> torchpy::inputs() {
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({1, 3, 224, 224}));
  return inputs;
}
