#pragma once

#include <Python.h>
#include "torch/csrc/autograd/engine.h"

bool THPEngine_initModule(PyObject *module);

namespace torch { namespace autograd { namespace python {

struct PythonEngine : public Engine {
  virtual void thread_main(std::shared_ptr<ReadyQueue> queue, int device) override;
  virtual void thread_on_exception(FunctionTask& task, std::exception& e) override;

  static PythonEngine& getDefaultEngine();
};

}}} // namespace torch::autograd::python
