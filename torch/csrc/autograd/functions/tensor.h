#pragma once

#include <Python.h>
#include <memory>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"

namespace torch { namespace autograd {

struct Identity : public Function {
  Identity(FunctionFlags&& f)
    : Function(std::move(f)) {};

  virtual variable_list apply(const variable_list& inputs) override;
};

struct Clone : public Function {
  Clone() {}

  virtual variable_list apply(const variable_list& inputs) override;
};

struct Contiguous : public Function {
  Contiguous() {}

  virtual variable_list apply(const variable_list& inputs) override;
};

struct Transpose : public Function {
  Transpose(int64_t dim1, int64_t dim2)
    : dim1(dim1)
    , dim2(dim2) {}

  virtual variable_list apply(const variable_list& inputs) override;

  int64_t dim1;
  int64_t dim2;
};

struct View : public Function {
  View(std::vector<int64_t> size)
    : size(size) {}

  virtual variable_list apply(const variable_list& inputs) override;

  std::vector<int64_t> size;
};

struct Expand : public Function {
  Expand(std::vector<int64_t> size)
    : size(size) {}

  virtual variable_list apply(const variable_list& inputs) override;

  std::vector<int64_t> size;
};

struct Narrow : public Function {
  Narrow(int64_t dim, int64_t start, int64_t size)
    : dim(dim)
    , start(start)
    , size(size) {}

  virtual variable_list apply(const variable_list& inputs) override;

  int64_t dim;
  int64_t start;
  int64_t size;
};

struct Cat : public Function {
  Cat(int64_t dim)
    : dim(dim) {}

  virtual variable_list apply(const variable_list& inputs) override;

  int64_t dim;
};

}}
