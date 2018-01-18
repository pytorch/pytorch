#include "torch/csrc/autograd/engine.h"

#include "torch/csrc/autograd/grad_mode.h"
#include "torch/csrc/autograd/functions/basic_ops.h"
#include "torch/csrc/utils/auto_gpu.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
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
  bool grad_mode;

  std::mutex mutex;
  // Notified when a task finishes executing.  Check outstanding_tasks to see
  // if all tasks are done.
  std::condition_variable not_done;
  std::unordered_map<Function*, InputBuffer> not_ready;
  std::unordered_map<Function*, int> dependencies;

  struct ExecInfo {
    struct Capture {
      Capture(int input_idx, int output_idx) : input_idx(input_idx), output_idx(output_idx) {}
      int input_idx; // within Function inputs
      int output_idx; // within the output vector of a GraphTask
    };

    bool should_execute() const {
      return needed || captures;
    }

    bool needed = false;
    std::unique_ptr<std::vector<Capture>> captures;
  };
  // Exec info has a bit complicated semantics. If it's empty, it means the task is
  // run in a "default" mode, which means that all next_functions we encounter should
  // get executed. If it's not empty, only functions that have an entry and this entry
  // has needed == True should be executed.
  std::unordered_map<Function*, ExecInfo> exec_info;
  std::vector<Variable> captured_vars;

  void init_to_execute(Function& graph_root, const function_list& captures);

  int owner;

  GraphTask(bool keep_graph, bool grad_mode)
    : exception()
    , has_error(false)
    , outstanding_tasks(0)
    , keep_graph(keep_graph)
    , grad_mode(grad_mode)
    , mutex()
    , not_done()
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
      GradMode::set_enabled(task.base->grad_mode);
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

  if(!task.base->keep_graph) {
    fn.willReleaseVariables();
  }
  auto outputs = fn(inputs);

  return call_post_hooks(fn, std::move(outputs), std::move(inputs));
}

auto Engine::evaluate_function(FunctionTask& task) -> void {
  // If exec_info is not empty, we have to instrument the execution
  auto & exec_info = task.base->exec_info;
  if (!exec_info.empty()) {
    auto & fn_info = exec_info.at(task.fn.get());
    if (auto *capture_vec = fn_info.captures.get()) {
      std::lock_guard<std::mutex> lock(task.base->mutex);
      for (auto capture : *capture_vec) {
        task.base->captured_vars[capture.output_idx] = task.inputs[capture.input_idx];
      }
    }
    if (!fn_info.needed) return;
  }

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
  if (num_outputs == 0) return; // Don't even acquire the mutex
  std::lock_guard<std::mutex> lock(task.base->mutex);
  for (int i = 0; i < num_outputs; ++i) {
    auto& output = outputs[i];
    auto& next_fn = fn.next_functions[i].first;
    int input_nr = fn.next_functions[i].second;

    if (!next_fn) {
      continue;
    }

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
      // Skip functions that aren't supposed to be executed
      if (!exec_info.empty()) {
        auto it = exec_info.find(next_fn.get());
        if (it == exec_info.end() || !it->second.should_execute()) {
          continue;
        }
      }
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

/* Computes the number of dependencies for each function which requires grad */
auto Engine::compute_dependencies(Function* root, GraphTask& task) -> void {
  // Just to make sure that they will never be added to the queue again
  std::unordered_set<Function*> seen;
  std::vector<Function*> queue { root };

  // Queue contains all nodes that will start propagating gradients.
  // We no longer have to expand functions that don't require grad.
  auto& dependencies = task.dependencies;
  while (queue.size() > 0) {
    auto fn = queue.back(); queue.pop_back();
    for (auto& edge : fn->next_functions) {
      Function* next_ptr = edge.first.get();
      if (!next_ptr) continue;
      dependencies[next_ptr] += 1;
      bool inserted;
      std::tie(std::ignore, inserted) = seen.insert(next_ptr);
      if (inserted) queue.push_back(next_ptr);
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
                     bool create_graph,
                     const function_list& outputs) -> variable_list {
  std::call_once(start_threads_flag, &Engine::start_threads, this);
  // Callbacks are only valid for the duration of this run and should always be cleared
  ClearCallbacks _cb_guard(final_callbacks, post_callbacks_lock);

  GraphTask graph_task(keep_graph, create_graph);
  std::unique_lock<std::mutex> lock(graph_task.mutex);

  // Now compute the dependencies for all executable functions and queue the root
  auto graph_root = std::make_shared<GraphRoot>(input_roots, inputs);
  compute_dependencies(graph_root.get(), graph_task);
  if (!outputs.empty()) {
    graph_task.init_to_execute(*graph_root, outputs);
  }
  ready_queue(-1).push_front(FunctionTask(&graph_task, std::move(graph_root), InputBuffer(0)));

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

  return graph_task.captured_vars;
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

void GraphTask::init_to_execute(Function& graph_root, const function_list& outputs) {
  exec_info[&graph_root].needed = true;

  int output_idx = 0;
  for (auto & output_edge : outputs) {
    Function *output = output_edge.first.get();
    auto & info = exec_info[output];
    if (!info.captures)
      info.captures.reset(new std::vector<ExecInfo::Capture>());
    info.captures->emplace_back(output_edge.second, output_idx++);
  }
  captured_vars.resize(output_idx);

  // NB: this is an uglier version (recursion replaced with iteration) of the following code:
  // is_needed = {}
  // def compute_is_needed(fn):
  //   if fn not in is_needed:
  //     is_needed[fn] = any(compute_is_needed(next_fn)
  //                         for next_fn in fn.next_functions)
  //   return is_needed[fn]
  struct Frame {
    Frame (Function *fn) : fn(fn), next_next_fn(0) {}
    Function *fn;
    std::size_t next_next_fn;

    Function* get_next_fn() {
      auto & next = fn->next_functions;
      auto num_next = next.size();
      while (next_next_fn < num_next) {
        auto fn = next[next_next_fn++].first.get();
        if (fn) return fn;
      }
      return nullptr;
    }
  };
  std::vector<Frame> stack;
  std::unordered_set<Function*> seen;
  for (const auto & input : graph_root.next_functions) {
    if (seen.count(input.first.get()) > 0) continue;
    stack.emplace_back(input.first.get());
    while (!stack.empty()) {
      auto &frame = stack.back();
      if (Function *next_fn = frame.get_next_fn()) {
        if (/* bool unseen = */ seen.emplace(next_fn).second) {
          stack.emplace_back(next_fn);
          continue; // recurse
        }
      } else {
        // NB: if we were using real recursion we could have saved some lookups
        // using a return value from recursive call. It would make this manually unrolled
        // version a lot more complicated, so I skipped that.
        auto & next_fns = frame.fn->next_functions;
        bool needed = std::any_of(next_fns.begin(), next_fns.end(),
                                  [&](const edge_type& e) -> bool {
                                    auto it = exec_info.find(e.first.get());
                                    return it != exec_info.end() && it->second.should_execute();
                                  });
        exec_info[frame.fn].needed = needed;
        stack.pop_back();
      }
    }
  }
}


}} // namespace torch::autograd
