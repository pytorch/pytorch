#pragma once
#include "ATen/Tensor.h"
#include "torch/csrc/jit/pybind.h"
#include "ATen/ATen.h"
#include <iostream>
#include <vector>

namespace torch { namespace jit {
struct BatchTensor {
public:
  BatchTensor(at::Tensor data, at::Tensor mask, at::Tensor dims);
  BatchTensor(const std::vector<at::Tensor> datalist, at::Tensor dims);
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
  at::Tensor getData(){
    return data;
  }
  at::Tensor getMask(){
    return mask;
  }
  at::Tensor getDims(){
    return dims;
  }

public:
  at::Tensor data;
  at::Tensor mask;
  at::Tensor dims;
};

void initBatchTensorBindings(PyObject* module);
}} // namespace torch::jit
