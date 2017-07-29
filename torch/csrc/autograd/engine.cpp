#include "torch/csrc/autograd/engine.h"
#include "torch/csrc/autograd/functions/basic_ops.h"
#include "torch/csrc/utils/auto_gpu.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <iostream>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_set>
#include <typeinfo>
#include <sstream>
#include <TH/TH.h>

#ifdef WITH_CUDA
#include <cuda.h>
#include <THC/THC.h>
#endif

namespace torch { namespace autograd {

// XXX: Changes to the way multithreading works in execute should be done with
// great care. Right now the implementation guarantees that a single function's
// apply will never be entered concurrently (even if multiple graphs are
// executed at the same time). Adding multiple threads per-device or removing
// engine thread affinity to the device can break this invariant, and we depend
// on it in a few places (e.g. AccumulateGrad function).

struct FunctionTask {
  GraphTask* base;
  std::shared_ptr<Function> fn;
  // This buffer serves as an implicit "addition" node for all of the
  // gradients flowing here.  Once all the dependencies are finished, we
  // use the contents of this buffer to run the function.
  InputBuffer inputs;

  FunctionTask(GraphTask* base, std::shared_ptr<Function> fn, InputBuffer inputs)
    : base(base)
    , fn(fn)
    , inputs(std::move(inputs)) {}
};

struct ReadyQueue {
  std::deque<FunctionTask> queue;
  std::condition_variable not_empty;
  std::mutex mutex;

  void push_front(FunctionTask item);
  FunctionTask pop_back();
};

struct GraphTask {
  std::exception_ptr exception;
  // Indicates if an error occurred while executing any task.  When this is
  // true, it signals all threads to stop executing.
  std::atomic_bool has_error;
  std::atomic<uint64_t> outstanding_tasks;
  bool keep_graph;
  bool has_any_work;

  std::mutex mutex;
  // Notified when a task finishes executing.  Check outstanding_tasks to see
  // if all tasks are done.
  std::condition_variable not_done;
  const Engine::callback_map& function_callbacks;
  std::unordered_map<Function*, InputBuffer> not_ready;
  std::unordered_map<Function*, int> dependencies;

  GraphTask(bool keep_graph, const Engine::callback_map& function_callbacks)
    : exception()
    , has_error(false)
    , outstanding_tasks(0)
    , keep_graph(keep_graph)
    , has_any_work(false)
    , mutex()
    , not_done()
    , function_callbacks(function_callbacks)
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

// This Engine's ReadyQueues and their corresponding threads are leaked here
Engine::~Engine() = default;

auto Engine::thread_main(std::shared_ptr<ReadyQueue> queue, int device) -> void {
  THInferNumThreads();
  AutoGPU guard(device);
  while (1) {
    FunctionTask task = queue->pop_back();
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

static variable_list call_pre_hooks(Function& fn, variable_list inputs) {
  for (auto& hook : fn.pre_hooks) {
    inputs = (*hook)(inputs);
  }
  return inputs;
}

static variable_list call_post_hooks(Function& fn, variable_list outputs, variable_list inputs) {
  for (auto& hook : fn.post_hooks) {
    outputs = (*hook)(outputs, inputs);
  }
  return outputs;
}

static variable_list call_function(FunctionTask& task) {
  auto& fn = *task.fn;
  auto inputs = call_pre_hooks(fn, InputBuffer::variables(std::move(task.inputs)));

  auto& function_callbacks = task.base->function_callbacks;
  auto callback_it = function_callbacks.find(&fn);
  if (callback_it != function_callbacks.end()) {
    auto& callback = callback_it->second;
    if (!callback(&fn, inputs)) return variable_list(fn.next_functions.size());
  }

  auto fn_outputs = fn.apply(inputs);
  return call_post_hooks(fn, std::move(fn_outputs), std::move(inputs));
}

auto Engine::evaluate_function(FunctionTask& task) -> void {
  auto outputs = call_function(task);

  auto& fn = *task.fn;
  if (!task.base->keep_graph) {
    fn.releaseVariables();
  }

  if (outputs.size() != fn.next_functions.size()) {
    std::stringstream ss;
    ss << "Function '" << fn.name() << "' returned an invalid number of outputs - expected ";
    ss << fn.next_functions.size() << ", but got " << outputs.size();
    throw std::runtime_error(ss.str());
  }

  int num_outputs = outputs.size();
  for (int i = 0; i < num_outputs; ++i) {
    auto& output = outputs[i];
    auto& next_fn = fn.next_functions[i].first;
    int input_nr = fn.next_functions[i].second;

    if (!next_fn) {
      continue;
    }

    // Stochastic functions are placed in the ready queue by
    // compute_dependencies, so we have to skip them here.
    if (next_fn->is_stochastic || !next_fn->is_executable) {
      continue;
    }

    std::lock_guard<std::mutex> lock(task.base->mutex);
    // Check if the next function is ready to be computed
    bool is_ready = false;
    auto& dependencies = task.base->dependencies;
    auto it = dependencies.find(next_fn.get());
    if (it == dependencies.end()) {
      auto name = next_fn->name();
      throw std::runtime_error(std::string("dependency not found for ") + name);
    } else if (--it->second == 0) {
      dependencies.erase(it);
      is_ready = true;
    }

    auto& not_ready = task.base->not_ready;
    auto not_ready_it = not_ready.find(next_fn.get());
    if (not_ready_it == not_ready.end()) {
      // No buffers have been allocated for the function
      InputBuffer input_buffer(next_fn->num_inputs);
      input_buffer.add(input_nr, std::move(output));
      if (is_ready) {
        auto& queue = ready_queue(input_buffer.device());
        queue.push_front(FunctionTask(task.base, next_fn, std::move(input_buffer)));
      } else {
        not_ready.emplace(next_fn.get(), std::move(input_buffer));
      }
    } else {
      // The function already has a buffer
      auto &input_buffer = not_ready_it->second;
      input_buffer.add(input_nr, std::move(output));
      if (is_ready) {
        auto& queue = ready_queue(input_buffer.device());
        queue.push_front(FunctionTask(task.base, next_fn, std::move(input_buffer)));
        not_ready.erase(not_ready_it);
      }
    }
  }
}

/** Finds all stochastic functions and appends them to the queue */
auto Engine::find_stochastic_functions(function_queue& queue, Function* graph_root, GraphTask& task) -> void {
  std::unordered_set<Function*> seen {graph_root};
  function_queue search_queue {graph_root};
  while (search_queue.size() > 0) {
    auto fn = search_queue.back(); search_queue.pop_back();
    for (auto& next_fn_pair : fn->next_functions) {
      auto& next_fn = next_fn_pair.first;
      Function* next_ptr = next_fn.get();
      if (!next_ptr) continue;
      if (next_ptr->is_stochastic && next_ptr->is_executable && seen.count(next_ptr) == 0) {
        ready_queue(-1).push_front(FunctionTask(&task, next_fn, InputBuffer(0)));
        queue.push_back(next_ptr);
        task.has_any_work = true;
      }
      if (seen.count(next_ptr) == 0) {
        seen.insert(next_ptr);
        search_queue.push_back(next_ptr);
      }
    }
  }
}

/** Computes the number of dependencies for each function which requires grad */
auto Engine::compute_dependencies(function_queue queue, GraphTask& task) -> void {
  // Just to make sure that they will never be added to the queue again
  std::unordered_set<Function*> seen(queue.begin(), queue.end());

  // Queue contains all nodes that will start propagating gradients.
  // We no longer have to expand functions that don't require grad.
  auto& dependencies = task.dependencies;
  while (queue.size() > 0) {
    auto fn = std::move(queue.back()); queue.pop_back();
    for (auto& next_fn_pair : fn->next_functions) {
      Function* next_ptr = next_fn_pair.first.get();
      if (!next_ptr) continue;
      if (!next_ptr->is_executable) continue;
      if (next_ptr->is_stochastic) continue; // Stochastic nodes were in the queue already
      dependencies[next_ptr] += 1;
      if (seen.count(next_ptr) == 0) {
        seen.insert(next_ptr);
        queue.push_back(next_ptr);
      }
    }
  }
}

struct ClearCallbacks {
  ClearCallbacks(std::vector<std::function<void()>>& callbacks,
                 std::mutex &callbacks_lock)
    : callbacks(callbacks)
    , callbacks_lock(callbacks_lock) { clear(); }
  ~ClearCallbacks() { clear(); }

  void clear() {
    std::lock_guard<std::mutex> lock(callbacks_lock);
    callbacks.clear();
  }

  std::vector<std::function<void()>>& callbacks;
  std::mutex& callbacks_lock;
};

auto Engine::execute(const function_list& input_roots,
                     variable_list& inputs,
                     bool keep_graph,
                     const callback_map& callbacks) -> void {
  std::call_once(start_threads_flag, &Engine::start_threads, this);
  // Callbacks are only valid for the duration of this run and should always be cleared
  ClearCallbacks _cb_guard(post_callbacks, post_callbacks_lock);

  GraphTask graph_task(keep_graph, callbacks);
  std::unique_lock<std::mutex> lock(graph_task.mutex);

  auto graph_root = std::make_shared<GraphRoot>(input_roots, inputs);
  function_queue roots;
  for (auto entry : input_roots) {
    if (entry.first->is_executable) {
      graph_task.has_any_work = true;
      roots.push_back(graph_root.get());
      ready_queue(-1).push_front(FunctionTask(&graph_task, graph_root, InputBuffer(0)));
      break;
    }
  }

  // Search the graph and find all stochastic functions. Append them to the queue.
  find_stochastic_functions(roots, graph_root.get(), graph_task);

  if (!graph_task.has_any_work) {
    throw std::runtime_error(
      "there are no graph nodes that require computing gradients");
  }

  // Now compute the dependencies for all executable functions
  compute_dependencies(std::move(roots), graph_task);

  // Wait for all tasks to complete
  graph_task.not_done.wait(lock, [&graph_task]{
    return graph_task.outstanding_tasks.load() == 0;
  });

  // Check for an exception while running backwards
  if (graph_task.has_error.load()) {
    std::rethrow_exception(graph_task.exception);
  }

  if (!graph_task.not_ready.empty()) {
    throw std::runtime_error("could not compute gradients for some functions");
  }

  // Unlocking is necessary, because the callback can register
  // more callbacks (or they can be registered from other threads
  // while it's waiting.
  std::unique_lock<std::mutex> cb_lock(post_callbacks_lock);
  for (std::size_t i = 0; i < post_callbacks.size(); ++i) {
    cb_lock.unlock();
    post_callbacks[i]();
    cb_lock.lock();
  }
}

void Engine::queue_callback(std::function<void()> callback) {
  std::lock_guard<std::mutex> lock(post_callbacks_lock);
  post_callbacks.emplace_back(std::move(callback));
}

auto Engine::ready_queue(int device) -> ReadyQueue& {
  return *ready_queues.at(device + 1);
}

auto Engine::start_threads() -> void {
  int num_devices = 0;
#ifdef WITH_CUDA
  // check for case of compiled with CUDA but no available devices
  if (cudaGetDeviceCount(&num_devices) != cudaSuccess) {
    cudaGetLastError();
    num_devices = 0;
  }
#endif
  int num_threads = num_devices + 1;
  ready_queues = std::vector<std::shared_ptr<ReadyQueue>>(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    auto& queue = ready_queues[i];
    queue.reset(new ReadyQueue());
    std::thread t(&Engine::thread_main, this, queue, i - 1);
    t.detach();
  }
}

}} // namespace torch::autograd
