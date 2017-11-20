#pragma once

#include <Python.h>
#include <memory>
#include "ATen/Type.h"

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/utils/tensor_geometry.h"

namespace torch { namespace autograd {

struct Identity : public TraceableFunction {
  using TraceableFunction::TraceableFunction;

  virtual variable_list apply(const variable_list& inputs) override;
};

struct CopyBackwards : public Function {
  virtual variable_list apply(const variable_list& inputs) override;

  at::Type *src_type;
};

struct Clone : public ForwardFunction<> {
  Clone() {}

  virtual variable_list apply(const variable_list& inputs) override;
};

struct Contiguous : public ForwardFunction<> {
  Contiguous() {}

  virtual variable_list apply(const variable_list& inputs) override;
};

struct Transpose : public ForwardFunction<> {
  Transpose(int64_t dim1, int64_t dim2)
    : dim1(dim1)
    , dim2(dim2) {}

  virtual variable_list apply(const variable_list& inputs) override;

  int64_t dim1;
  int64_t dim2;
};

struct View : public ForwardFunction<> {
  View(std::vector<int64_t> size)
    : size(size) {}

  virtual variable_list apply(const variable_list& inputs) override;

  std::vector<int64_t> size;
};

struct Expand : public ForwardFunction<> {
  Expand(std::vector<int64_t> size)
    : size(size) {}

  virtual variable_list apply(const variable_list& inputs) override;

  std::vector<int64_t> size;
};

struct Narrow : public ForwardFunction<> {
  Narrow(int64_t dim, int64_t start, int64_t size)
    : dim(dim)
    , start(start)
    , size(size) {}

  virtual variable_list apply(const variable_list& inputs) override;

  int64_t dim;
  int64_t start;
  int64_t size;
};

struct Cat : public ForwardFunction<> {
  Cat(int64_t dim)
    : dim(dim) {}

  virtual variable_list apply(const variable_list& inputs) override;

  int64_t dim;
};

struct Chunk : public Function {
  Chunk(int64_t chunks, int64_t dim)
    : chunks(chunks), dim(dim) {}

  virtual variable_list apply(const variable_list& inputs) override;

private:
  int64_t chunks;
  int64_t dim;
};

// Performs grad[idx] = fn(grad[idx]), but out-of-place. The slicing operation
// grad[idx] is defined by the relative sizes, strides, and offset of base and
// view.
struct CopySlices : public Function {
  CopySlices(const Variable& base, TensorGeometry view, std::shared_ptr<Function> fn);

  virtual variable_list apply(const variable_list& grads) override;
  virtual void releaseVariables() override;

  TensorGeometry base;
  TensorGeometry view;
  std::shared_ptr<Function> fn;
};

}}
