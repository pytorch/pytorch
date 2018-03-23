#pragma once

#include <Python.h>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/engine.h"

bool THPEngine_initModule(PyObject *module);

namespace torch { namespace autograd { namespace python {

struct PythonEngine : public Engine {
  virtual void thread_init(int device) override;
  virtual void thread_on_exception(FunctionTask& task, std::exception& e) override;
  virtual variable_list execute(
      const edge_list& roots,
      const variable_list& inputs,
      bool keep_graph,
      bool create_graph,
      const edge_list& outputs = {}) override;
};

}}} // namespace torch::autograd::python
