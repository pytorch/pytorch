#pragma once
#include <ATen/ATen.h>
#include <iostream>
#include <vector>
namespace torchpy {

class PyModule {
 public:
  PyModule() {}
  ~PyModule() {}

  at::Tensor forward(std::vector<at::Tensor> inputs);

 private:
};

void init();
std::string hello();
PyModule load(const std::string& filename);
std::vector<at::Tensor> inputs(std::vector<int64_t> shape);
}
