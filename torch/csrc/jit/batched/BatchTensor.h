#pragma once
#include "ATen/Tensor.h"
#include "torch/csrc/jit/pybind.h"
#include "ATen/ATen.h"
#include <iostream>
#include <vector>

namespace torch { namespace jit {
struct BatchTensor {
public:
  BatchTensor(at::Tensor data, at::Tensor mask, std::vector<bool> dims);
  BatchTensor(std::vector<at::Tensor>, std::vector<bool> dims);
  ~BatchTensor(){};
  const char * toString() const {
    return "BatchTensor";
  }
  at::IntList sizes() const {
    return data.sizes();
  }
  int64_t dim() const {
    return data.dim();
  }
  std::vector<at::Tensor> examples();

public:
  at::Tensor data;
  at::Tensor mask;
  std::vector<bool> dims;
};

void initBatchTensorBindings(PyObject* module);
}} // namespace torch::jit
