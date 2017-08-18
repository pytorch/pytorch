#pragma once

#include <Python.h>
#include <memory>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"

namespace torch { namespace autograd {

struct Identity : public TraceableFunction {
  Identity(FunctionFlags&& f)
    : TraceableFunction(std::move(f)) {};

  virtual variable_list apply(const variable_list& inputs) override;
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

}}
