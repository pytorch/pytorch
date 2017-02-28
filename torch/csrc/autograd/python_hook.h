#pragma once

#include "torch/csrc/autograd/grad_hook.h"
#include <Python.h>

namespace torch { namespace autograd {

struct PyGradHook : public GradHook {
  PyGradHook(PyObject* dict);
  ~PyGradHook();
  std::shared_ptr<Variable> operator()(const std::shared_ptr<Variable>& _grad) override;
  PyObject* dict;
};

}} // namespace torch::autograd
