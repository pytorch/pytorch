#pragma once

#include <Python.h>
#include <THPP/THPP.h>
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
  Transpose(long dim1, long dim2)
    : dim1(dim1)
    , dim2(dim2) {}

  virtual variable_list apply(const variable_list& inputs) override;

  long dim1;
  long dim2;
};

struct View : public Function {
  View(std::vector<long> size)
    : size(size) {}

  virtual variable_list apply(const variable_list& inputs) override;

  std::vector<long> size;
};

struct Expand : public Function {
  Expand(std::vector<long> size)
    : size(size) {}

  virtual variable_list apply(const variable_list& inputs) override;

  std::vector<long> size;
};

}}


