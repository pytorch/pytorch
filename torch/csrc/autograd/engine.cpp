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

// NB: -1 indicates the CPU worker!
static constexpr int NO_DEVICE = -2;
static thread_local int worker_device = NO_DEVICE;

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
  const Engine::pre_callback_map& pre_callbacks;
  const Engine::post_callback_map& post_callbacks;
  std::unordered_map<Function*, InputBuffer> not_ready;
  std::unordered_map<Function*, int> dependencies;

  int owner;

  GraphTask(bool keep_graph, const Engine::pre_callback_map& pre_callbacks, const Engine::post_callback_map& post_callbacks)
    : exception()
    , has_error(false)
    , outstanding_tasks(0)
    , keep_graph(keep_graph)
    , has_any_work(false)
    , mutex()
    , not_done()
    , pre_callbacks(pre_callbacks)
    , post_callbacks(post_callbacks)
    , not_ready()
    , dependencies()
    , owner(NO_DEVICE) {}
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

auto Engine::thread_init(int device) -> void {
  THInferNumThreads();
  AutoGPU guard(device);
  worker_device = device;
  thread_main(nullptr);
}

// NOTE: graph_tasks do not necessarily form a stack. Imagine this
// case:
//
//    +----> Eval1
//  Root
//    +----> Eval2
//
// Once Root is executed, both Eval1 and Eval2 are added to the ready queue.
// Next, Eval1 is run and this causes the worker to enter thread_main again.
// Then, it pops the next task from the queue, but at this point it is Eval2.
// It enters thread_main once again, but now with graph_task of Eval2, which is
// completely unrelated to that of Eval1 (it's not a recursive call).
// It's all ok and is handled right now, but it should be accounted for
// in case this code is to be changed.
auto Engine::thread_main(GraphTask *graph_task) -> void {
  auto queue = ready_queues[worker_device + 1];
  while (!graph_task || graph_task->outstanding_tasks > 0) {
    FunctionTask task = queue->pop_back();
    if (task.fn && !task.base->has_error.load()) {
      try {
        evaluate_function(task);
      } catch (std::exception& e) {
        thread_on_exception(task, e);
      }
    }
    auto base_owner = task.base->owner;
    // Task from a non-worker thread. Easy case.
    if (base_owner == NO_DEVICE) {
      if (--task.base->outstanding_tasks == 0) {
        std::lock_guard<std::mutex> lock(task.base->mutex);
        task.base->not_done.notify_all();
      }
    } else {
      // If it's a task initiated from this thread, decrease the counter, but
      // don't do anything - loop condition will do all checks for us next.
      if (base_owner == worker_device) {
        --task.base->outstanding_tasks;
      // Otherwise send a dummy function task to the owning thread just to
      // ensure that it's not sleeping. If it has work, it might see that
      // graph_task->outstanding_tasks == 0 before it gets to the task, but
      // it's a no-op anyway.
      } else if (base_owner != worker_device) {
        if (--task.base->outstanding_tasks == 0) {
          // Synchronize outstanding_tasks with queue mutex
          std::atomic_thread_fence(std::memory_order_release);
          ready_queue(base_owner).push_front(FunctionTask(task.base, nullptr, InputBuffer(0)));
        }
      }
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

  auto& pre_callbacks = task.base->pre_callbacks;
  for (auto it_p = pre_callbacks.equal_range(&fn); it_p.first != it_p.second; ++it_p.first) {
    auto& callback = it_p.first->second;
    if (!callback(&fn, inputs)) return variable_list(fn.next_functions.size());
  }

  auto outputs = fn(inputs);

  auto& post_callbacks = task.base->post_callbacks;
  for (auto it_p = post_callbacks.equal_range(&fn); it_p.first != it_p.second; ++it_p.first) {
    auto& callback = it_p.first->second;
    if (!callback(&fn, inputs, outputs)) return variable_list(fn.next_functions.size());
  }

  return call_post_hooks(fn, std::move(outputs), std::move(inputs));
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
                     const variable_list& inputs,
                     bool keep_graph,
                     const pre_callback_map& pre_callbacks,
                     const post_callback_map& post_callbacks) -> void {
  std::call_once(start_threads_flag, &Engine::start_threads, this);
  // Callbacks are only valid for the duration of this run and should always be cleared
  ClearCallbacks _cb_guard(final_callbacks, post_callbacks_lock);

  GraphTask graph_task(keep_graph, pre_callbacks, post_callbacks);

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

  // Not a worker
  if (worker_device == NO_DEVICE) {
    // Wait for all tasks to complete
    graph_task.not_done.wait(lock, [&graph_task]{
      return graph_task.outstanding_tasks.load() == 0;
    });
  } else {
    graph_task.owner = worker_device;
    lock.unlock();
    thread_main(&graph_task);
  }

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
  for (std::size_t i = 0; i < final_callbacks.size(); ++i) {
    cb_lock.unlock();
    final_callbacks[i]();
    cb_lock.lock();
  }
}

void Engine::queue_callback(std::function<void()> callback) {
  std::lock_guard<std::mutex> lock(post_callbacks_lock);
  final_callbacks.emplace_back(std::move(callback));
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
  // One for CPU, plus one for every GPU device
  int num_threads = num_devices + 1;
  ready_queues = std::vector<std::shared_ptr<ReadyQueue>>(num_threads);
  for (auto& queue : ready_queues)
    queue.reset(new ReadyQueue());
  for (int i = 0; i < num_threads; ++i) {
    std::thread t(&Engine::thread_init, this, i - 1);
    t.detach();
  }
}

}} // namespace torch::autograd
