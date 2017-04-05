#include "torch/csrc/autograd/engine.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <set>
#include <string>
#include <THPP/THPP.h>
#include <thread>
#include <unordered_set>
#include <typeinfo>
#include <sstream>

#ifdef WITH_CUDA
#include <cuda.h>
#include <THC/THC.h>
#endif

using thpp::Tensor;

namespace torch { namespace autograd {

struct FunctionTask {
  BackwardTask* base;
  std::shared_ptr<Function> fn;
  GradBuffer grad;

  FunctionTask(BackwardTask* base, std::shared_ptr<Function> fn, GradBuffer grad)
    : base(base)
    , fn(fn)
    , grad(std::move(grad)) {}
};

struct ReadyQueue {
  std::deque<FunctionTask> queue;
  std::condition_variable not_empty;
  std::mutex mutex;

  void push_front(FunctionTask item);
  FunctionTask pop_back();
};

struct BackwardTask {
  std::exception_ptr exception;
  std::atomic_bool has_error;
  std::atomic<uint64_t> outstanding_tasks;
  bool retain_variables;
  bool node_requires_grad;

  std::mutex mutex;
  std::condition_variable not_done;
  std::unordered_map<Function*, GradBuffer> not_ready;
  std::unordered_map<Function*, int> dependencies;

  BackwardTask(bool retain_variables)
    : exception()
    , has_error(false)
    , outstanding_tasks(0)
    , retain_variables(retain_variables)
    , node_requires_grad(false)
    , mutex()
    , not_done()
    , not_ready()
    , dependencies() {}
};

auto ReadyQueue::push_front(FunctionTask item) -> void {
  {
    std::lock_guard<std::mutex> lock(mutex);
    ++item.base->outstanding_tasks;
    queue.push_front(std::move(item));
  }
  not_empty.notify_one();
}

auto ReadyQueue::pop_back() -> FunctionTask {
  std::unique_lock<std::mutex> lock(mutex);
  not_empty.wait(lock, [this]{ return !queue.empty(); });
  auto task = std::move(queue.back()); queue.pop_back();
  return task;
}

Engine::Engine() : ready_queues() {
}

Engine::~Engine() = default;

auto Engine::thread_main(ReadyQueue& queue) -> void {
  while (1) {
    FunctionTask task = queue.pop_back();
    if (!task.base->has_error.load()) {
      try {
        evaluate_function(task);
      } catch (std::exception& e) {
        thread_on_exception(task, e);
      }
    }
    if (--task.base->outstanding_tasks == 0) {
      std::lock_guard<std::mutex> lock(task.base->mutex);
      task.base->not_done.notify_all();
    }
  }
}

auto Engine::thread_on_exception(FunctionTask& task, std::exception& e) -> void {
  std::lock_guard<std::mutex> lock(task.base->mutex);
  if (!task.base->has_error.load()) {
    task.base->exception = std::current_exception();
    task.base->has_error = true;
  }
}

static variable_list call_pre_hooks(Function& fn, variable_list grad_output) {
  for (auto& hook : fn.pre_hooks) {
    grad_output = (*hook)(grad_output);
  }
  return grad_output;
}

static variable_list call_post_hooks(Function& fn, variable_list grad_input, variable_list grad_output) {
  for (auto& hook : fn.post_hooks) {
    grad_input = (*hook)(grad_input, grad_output);
  }
  return grad_input;
}

static variable_list call_function(FunctionTask& task) {
  auto grad_output = call_pre_hooks(*task.fn, GradBuffer::variables(std::move(task.grad)));
  auto grad_input = task.fn->apply(grad_output);
  return call_post_hooks(*task.fn, std::move(grad_input), std::move(grad_output));
}

auto Engine::evaluate_function(FunctionTask& task) -> void {
  auto grad_inputs = call_function(task);

  auto& fn = *task.fn;
  if (!task.base->retain_variables) {
    fn.releaseVariables();
  }

  if (grad_inputs.size() != fn.previous_functions.size()) {
    std::stringstream ss;
    ss << "Function '" << fn.name() << "' returned an invalid number of gradients - expected ";
    ss << fn.previous_functions.size() << ", but got " << grad_inputs.size();
    throw std::runtime_error(ss.str());
  }

  int size = grad_inputs.size();
  for (int i = 0; i < size; ++i) {
    auto& grad_input = grad_inputs[i];
    auto& prev_fn = fn.previous_functions[i].first;
    int output_nr = fn.previous_functions[i].second;

    // null inputs have no previous_function and we skip them here
    if (!prev_fn) {
      continue;
    }

    // Stochastic functions are placed in the ready queue by
    // compute_dependencies, so we can skip them here.
    if (prev_fn->is_stochastic || !prev_fn->requires_grad) {
      continue;
    }

    std::lock_guard<std::mutex> lock(task.base->mutex);
    if (auto var = dynamic_cast<Variable*>(prev_fn.get())) {
      if (!grad_input) {
        // NOTE: grad_input can be NULL if the function returns None for a
        // non_differentiable input. We may need to track additional information
        // at the function level to determine if a NULL grad_input is an error.
        std::stringstream ss;
        ss << "Function '" << fn.name() << "' missing gradient at " << i;
        throw std::runtime_error(ss.str());
      }
      var->backward(grad_input);
      continue;
    }

    // Check if the function is ready for backward
    bool is_ready = false;
    auto& dependencies = task.base->dependencies;
    auto it = dependencies.find(prev_fn.get());
    if (it == dependencies.end()) {
      auto name = prev_fn->name();
      throw std::runtime_error(std::string("dependency not found for ") + name);
    } else if (--it->second == 0) {
      dependencies.erase(it);
      is_ready = true;
    }

    auto& not_ready = task.base->not_ready;
    auto not_ready_it = not_ready.find(prev_fn.get());
    if (not_ready_it == not_ready.end()) {
      // No buffers have been allocated for the function
      GradBuffer prev_buffer(prev_fn->num_outputs);
      prev_buffer.addGrad(output_nr, std::move(grad_input));
      if (is_ready) {
        auto& queue = ready_queue(prev_buffer.device());
        queue.push_front(FunctionTask(task.base, prev_fn, std::move(prev_buffer)));
      } else {
        not_ready.emplace(prev_fn.get(), std::move(prev_buffer));
      }
    } else {
      // The function already has a buffer
      auto &prev_buffer = not_ready_it->second;
      prev_buffer.addGrad(output_nr, std::move(grad_input));
      if (is_ready) {
        auto& queue = ready_queue(prev_buffer.device());
        queue.push_front(FunctionTask(task.base, prev_fn, std::move(prev_buffer)));
        not_ready.erase(not_ready_it);
      }
    }
  }
}

/** Finds all stochastic functions and appends them to the queue */
auto Engine::find_stochastic_functions(function_queue& queue, BackwardTask& task) -> void {
  std::unordered_set<Function*> seen;
  function_queue search_queue(queue);
  while (search_queue.size() > 0) {
    auto fn = search_queue.back(); search_queue.pop_back();
    for (auto& prev_fn_pair : fn->previous_functions) {
      auto& prev_fn = prev_fn_pair.first;
      Function* prev_ptr = prev_fn.get();
      if (!prev_ptr) continue;
      if (prev_ptr->is_stochastic && prev_ptr->requires_grad && seen.count(prev_ptr) == 0) {
        ready_queue(-1).push_front(FunctionTask(&task, prev_fn, GradBuffer(0)));
        queue.push_back(prev_ptr);
        task.node_requires_grad = true;
      }
      if (seen.count(prev_ptr) == 0) {
        seen.insert(prev_ptr);
        search_queue.push_back(prev_ptr);
      }
    }
  }
}

/** Computes the number of dependencies for each function which requires grad */
auto Engine::compute_dependencies(function_queue queue, BackwardTask& task) -> void {
  // Just to make sure that they will never be added to the queue again
  std::unordered_set<Function*> seen(queue.begin(), queue.end());

  // Queue contains all nodes that will start propagating gradients.
  // We no longer have to expand functions that don't require grad.
  auto& dependencies = task.dependencies;
  while (queue.size() > 0) {
    auto fn = std::move(queue.back()); queue.pop_back();
    // This is needed only to filter out backward roots that don't require grad
    if (!fn->requires_grad) continue;
    for (auto& prev_fn_pair : fn->previous_functions) {
      Function* prev_ptr = prev_fn_pair.first.get();
      if (!prev_ptr) continue;
      if (dynamic_cast<Variable*>(prev_ptr)) continue;
      if (!prev_ptr->requires_grad) continue;
      if (prev_ptr->is_stochastic) continue; // Stochastic nodes were in the queue already
      dependencies[prev_ptr] += 1;
      if (seen.count(prev_ptr) == 0) {
        seen.insert(prev_ptr);
        queue.push_back(prev_ptr);
      }
    }
  }
}

auto Engine::find_creators(const variable_list& variables,
                           tensor_list& grad_variables,
                           BackwardTask& task) -> function_queue {
  function_queue creators;
  std::unordered_map<std::shared_ptr<Function>, std::unique_ptr<GradBuffer>> creator_grad;
  int size = variables.size();
  for (int i = 0; i < size; ++i) {
    auto& var = variables[i];
    auto& grad = grad_variables[i];
    if (!var->creator) {
      // If someone calls .backward() on a leaf, it's simple...
      if (var->requires_grad) {
        var->backward(std::make_shared<Variable>(std::move(grad), false, true));
        task.node_requires_grad = true;
      }
    } else {
      auto& creator = var->creator;
      auto& buf = creator_grad[creator];
      if (creator->requires_grad) {
        if (!buf) buf.reset(new GradBuffer(creator->num_outputs));
        buf->addGrad(var->output_nr, Variable::of(std::move(grad)));
      }
    }
  }

  for (auto& entry: creator_grad) {
    const auto& creator = entry.first;
    creators.push_back(creator.get());
    if (creator->requires_grad) {
      // NOTE: buf is null if creator doesn't require gradient
      auto& buf = entry.second;
      auto& queue = ready_queue(buf->device());
      queue.push_front(FunctionTask(&task, creator, std::move(*buf)));
      task.node_requires_grad = true;
    }
  }

  return creators;
}

auto Engine::backward(const variable_list& variables,
                      tensor_list& grad_variables,
                      bool retain_variables) -> void {
  static std::once_flag once_flag;
  std::call_once(once_flag, &Engine::start_threads, this);

  BackwardTask backward_task(retain_variables);
  std::unique_lock<std::mutex> lock(backward_task.mutex);

  // Find the unique creators and backprop into variables which don't have creators.
  auto creators = find_creators(variables, grad_variables, backward_task);

  // Search the graph and find all stochastic functions. Append them to the queue.
  find_stochastic_functions(creators, backward_task);

  if (!backward_task.node_requires_grad) {
    throw std::runtime_error(
      "there are no graph nodes that require computing gradients");
  }

  // Now compute the dependencies for each function which requires grad
  compute_dependencies(std::move(creators), backward_task);

  // wait for all tasks to complete
  backward_task.not_done.wait(lock, [&backward_task]{
    return backward_task.outstanding_tasks.load() == 0;
  });

  // check for an exception while running backwards
  if (backward_task.has_error.load()) {
    std::rethrow_exception(backward_task.exception);
  }

  if (!backward_task.not_ready.empty()) {
    throw std::runtime_error("could not compute gradients for some functions");
  }
}

auto Engine::ready_queue(int device) -> ReadyQueue& {
  return *ready_queues.at(device + 1);
}

auto Engine::start_threads() -> void {
  int num_devices = 0;
#ifdef WITH_CUDA
  cudaError_t err = cudaGetDeviceCount(&num_devices);

  // check for case of compiled with CUDA but no NVIDIA driver available
  if (err == cudaErrorInsufficientDriver) {
    num_devices = 0;
  } else {
    THCudaCheck(err);
  }
#endif
  ready_queues = std::vector<std::unique_ptr<ReadyQueue>>(num_devices + 1);
  for (auto& queue : ready_queues) {
    queue.reset(new ReadyQueue());
    std::thread t(&Engine::thread_main, this, std::ref(*queue));
    t.detach();
  }
}

}} // namespace torch::autograd
