#pragma once

// Engine implements backpropagation from output variables and their gradients
// to "root" variables (variables created by the user with requires_grad=True).

#include <Python.h>
#include <deque>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>
#include <functional>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/input_buffer.h"

namespace torch { namespace autograd {

struct ReadyQueue;
struct FunctionTask;
struct GraphTask;

// A single instance of this struct should be created through the whole process lifetime.
// The worker thread creation logic and Engine's destructor rely on this.
struct Engine {
  Engine();
  virtual ~Engine();

  using ready_queue_type = std::deque<std::pair<std::shared_ptr<Function>, InputBuffer>>;
  using function_queue = std::vector<Function*>;
  using dependencies_type = std::unordered_map<Function*, int>;
  using callback_type = std::function<bool (Function*, variable_list&)>;
  using callback_map = std::unordered_map<Function*, callback_type>;

  // Given a list of (Function, input number) pairs computes the value of the graph
  // by following next_function references.
  void execute(
      const function_list& roots,
      const variable_list& inputs,
      bool keep_graph,
      const callback_map& callbacks = callback_map());

  void queue_callback(std::function<void()> callback);

protected:
  function_queue find_roots(
      const function_list& roots,
      variable_list& inputs,
      GraphTask& task);
  void find_stochastic_functions(function_queue& queue, Function* graph_root, GraphTask& task);
  void compute_dependencies(function_queue queue, GraphTask& task);
  void evaluate_function(FunctionTask& task);
  ReadyQueue& ready_queue(int device);
  void start_threads();
  virtual void thread_main(std::shared_ptr<ReadyQueue> queue, int device);
  virtual void thread_on_exception(FunctionTask& task, std::exception& e);

  std::once_flag start_threads_flag;
  std::vector<std::shared_ptr<ReadyQueue>> ready_queues;
  std::vector<std::function<void()>> post_callbacks;
  std::mutex post_callbacks_lock;
};

}} // namespace torch::autograd
