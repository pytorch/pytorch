#pragma once

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/DynamicTypes.h"

#include <c2isl/core/tvmir_op.h>
#include <c2isl/core/common.h>
#include <dlpack/dlpack.h>
#include <ATen/ATen.h>

namespace torch { namespace autograd {

// The basic model for c2isl is you give it information about the sizes,
// types and strides of the input tensor, and then it generates JIT
// compiled kernel which is specialized to that particular size.


// DLTensor is a plain C object, so the onus is on us to correctly allocate
// and free members.  An added complication is c2isl also uses the DLTensor
// data type to represent size/stride information, WITHOUT an actual tensor.
//
// Here's our strategy:
//  - We new-allocate shape and strides for both DLTensor and DLMetadata,
//    and free them in our deleters.
//  - The deleter for DLTensor also holds a reference to at::Tensor, to keep
//    it live as long as we are passing around a smart pointer to DLTensor.

using DLMetadata = DLTensor;

struct DLMetadataDeleter {
  void operator()(DLTensor *t) {
    delete[] t->shape;
    delete[] t->strides;
    delete t;
  }
};

struct DLTensorDeleter {
  at::Tensor object;
  DLTensorDeleter(at::Tensor object) : object(object) {}
  void operator()(DLTensor *t) {
    delete[] t->shape;
    delete[] t->strides;
    delete t;
  }
};

using DLTensorUPtr = std::unique_ptr<DLTensor, DLTensorDeleter>;
using DLTensorSPtr = std::shared_ptr<DLTensor>;

using DLMetadataUPtr = std::unique_ptr<DLTensor, DLMetadataDeleter>;
using DLMetadataSPtr = std::shared_ptr<DLTensor>;

struct IslParams {
  // The TVM data.  We use these names to keep copy-pasting easier.
  // NB: the inputs here bake in the type in question, which means
  // that every IslFunction is monomorphic.
  std::string kernelName;
  std::vector<::tvm::Tensor> outputs;
  std::vector<::tvm::Tensor> inputs;
  std::vector<::tvm::Var> vars;
  std::vector<::tvm::Tensor> ops;
  // TODO: ISLKernelOptions
};

struct IslFunction : public Function, public IslParams {
  std::unique_ptr<c2isl::ISLTVMIROp> pImpl_;

  c2isl::ISLKernelOptions islKernelOptions;
  // NB: Ugh, but this is what c2isl's API returns, so...
  std::vector<::tvm::DLTensorUPtr> outputDLMetas_;

  IslFunction(IslParams params) : IslParams(std::move(params)) {}

  virtual variable_list apply(const variable_list& inputs) override;
};

/*
struct IslMatMul : public IslFunction {
  IslMatMul() {}
};
*/

}} // namespace torch::autograd
