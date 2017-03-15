#pragma once

// Engine implements backpropagation from output variables and their gradients
// to "root" variables (variables created by the user with requires_grad=True).

#include <deque>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/input_buffer.h"

namespace torch { namespace autograd {

struct ReadyQueue;
struct FunctionTask;
struct GraphTask;

struct Engine {
  Engine();
  virtual ~Engine();

  using ready_queue_type = std::deque<std::pair<std::shared_ptr<Function>, InputBuffer>>;
  using function_queue = std::vector<Function*>;
  using dependencies_type = std::unordered_map<Function*, int>;

  // Given a list of (Function, int) pairs computes the value of the graph
  // by following next_function references.
  void execute(
      const function_list& roots,
      variable_list& inputs,
      bool keep_graph);

protected:
  function_queue find_roots(
      const function_list& roots,
      variable_list& inputs,
      GraphTask& task);
  void find_stochastic_functions(function_queue& queue, GraphTask& task);
  void compute_dependencies(function_queue queue, GraphTask& task);
  void evaluate_function(FunctionTask& task);
  ReadyQueue& ready_queue(int device);
  void start_threads();
  virtual void thread_main(ReadyQueue& queue);
  virtual void thread_on_exception(FunctionTask& task, std::exception& e);

  std::vector<std::unique_ptr<ReadyQueue>> ready_queues;
};

}} // namespace torch::autograd
