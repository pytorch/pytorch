#pragma once

#include <torch/csrc/python_headers.h>

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/engine.h>

bool THPEngine_initModule(PyObject *module);

namespace torch { namespace autograd { namespace python {

struct PythonEngine : public Engine {
  void thread_init(int device) override;
  void thread_on_exception(
      std::shared_ptr<GraphTask>& graph_task,
      const std::shared_ptr<Node>& fn,
      std::exception& e) override;
  variable_list execute(
      const edge_list& roots,
      const variable_list& inputs,
      bool keep_graph,
      bool create_graph,
      const edge_list& outputs = {}) override;

  variable_list execute_with_graph_task(
      std::shared_ptr<GraphTask> graph_task,
      std::shared_ptr<Node> graph_root) override;
  std::unique_ptr<AnomalyMetadata> make_anomaly_metadata() override;
};

}}} // namespace torch::autograd::python
