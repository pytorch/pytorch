#pragma once

#include <torch/csrc/python_headers.h>

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/python_anomaly_mode.h>

bool THPEngine_initModule(PyObject *module);

namespace torch { namespace autograd { namespace python {

struct PythonEngine : public Engine {
  void thread_init(int device) override;
  void thread_on_exception(FunctionTask& task, std::exception& e) override;
  variable_list execute(
      const edge_list& roots,
      const variable_list& inputs,
      bool keep_graph,
      bool create_graph,
      const edge_list& outputs = {}) override;
  std::unique_ptr<AnomalyMetadata> make_anomaly_metadata() override;
};

}}} // namespace torch::autograd::python
