#pragma once

#include <Python.h>
#include <mutex>
#include <memory>
#include <functional>
#include <ATen/ATen.h>

#include "torch/csrc/jit/tracer_state.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/variable_version.h"
#include "torch/csrc/Types.h"

namespace torch { namespace autograd {

struct Function;

extern const char* ERR_BACKWARD_TWICE;

struct SavedVariable {
  SavedVariable()
    : data()
    , has_grad_fn(false)
    , version()
    , requires_grad(false)
    , is_volatile(false)
    , expected_version(-1) {}

  SavedVariable(const Variable& variable, Function* saved_for);


  at::Tensor data;
  // The gradient function associated with this node. If has_grad_fn
  // is false, then this is a leaf node. Note that the grad_fn is not saved if
  // it would create a circular reference. In that case, the grad_fn must be
  // passed in to the unpack function when reconstructing the Variable.
  bool has_grad_fn;
  std::shared_ptr<Function> grad_fn;
  std::weak_ptr<Function> grad_accumulator;
  SavedVersion version;
  bool requires_grad;
  bool is_volatile;
  int expected_version;
  int output_nr;
  std::unique_ptr<jit::tracer::ValueTracingState> tracing_state;

  Variable unpack(std::shared_ptr<Function> saved_for=nullptr) const;
  at::Tensor unpack_data(std::shared_ptr<Function> saved_for=nullptr) const;
};

}} // namespace torch::autograd
