#pragma once
#include <ATen/ATen.h>
#include <iostream>
#include <vector>
namespace torchpy {

class PyModule {
 public:
  PyModule(PyObject* globals, PyObject* module);
  ~PyModule();

  at::Tensor forward(std::vector<at::Tensor> inputs);

 private:
  PyObject* _globals;
  PyObject* _module;
};

void init();
void finalize();
void test_get_load();
std::string hello();
PyModule load(const std::string& filename);
}
