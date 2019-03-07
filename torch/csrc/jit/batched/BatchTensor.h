#pragma once
#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <torch/csrc/jit/pybind.h>
#include <iostream>
#include <vector>

namespace torch {
namespace jit {
struct BatchTensor {
 public:
  BatchTensor(at::Tensor data, at::Tensor mask, at::Tensor dims);
  // expand a tensor to a batchtensor given batch_size
  BatchTensor(const at::Tensor& data, int64_t batch_size);
  BatchTensor(const std::vector<at::Tensor>& datalist, at::Tensor dims);
  const char* toString() const {
    return "BatchTensor";
  }
  at::IntArrayRef sizes() const {
    return data.sizes();
  }
  int64_t dim() const {
    return data.dim();
  }
  std::vector<at::Tensor> examples();
  at::Tensor get_data() {
    return data;
  }
  at::Tensor get_mask() {
    return mask;
  }
  at::Tensor get_dims() {
    return dims;
  }

 public:
  // data is a Tensor whose size is the batch size in the batch dimension,
  // the size of all examples in static dimensions,
  // and at least as large as the largest example in the batch in dynamic
  // dimensions.
  at::Tensor data;
  // mask is a Tensor whose size is the batch size in the batch dimension,
  // one in static dimensions,
  // and at least as large as the largest example in the batch in dynamic
  // dimensions. Each entry in the mask corresponds to one or more entries in
  // the data array (singleton, i.e., static, dimensions are broadcasted), with
  // a one in the mask denoting that the corresponding data entries represent
  // valid, meaningful data and a zero denoting that they do not.
  at::Tensor mask;
  // dims is a 1-dimensional tensor with a bool for each non-batch dimension,
  // representing whether that dimension is static (False) or dynamic (True).
  at::Tensor dims;
};

void initBatchTensorBindings(PyObject* module);
} // namespace jit
} // namespace torch
