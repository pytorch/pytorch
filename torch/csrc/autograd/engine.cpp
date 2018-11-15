#include "torch/csrc/autograd/engine.h"

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/functions/basic_ops.h"
#include "torch/csrc/autograd/grad_mode.h"
#include "torch/csrc/autograd/anomaly_mode.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/utils/memory.h"

#include <ATen/DeviceGuard.h>
#include <ATen/ExpandUtils.h>
#include <c10/util/Exception.h>

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
#include <queue>
#include <TH/TH.h>

#ifdef USE_CUDA
#include <cuda.h>
#include <THC/THC.h>
#include <ATen/cuda/CUDAGuard.h>
#endif

namespace torch { namespace autograd {

// NB: -1 indicates the CPU worker!
static constexpr int NO_DEVICE = -2;

// Threads spawned by the engine are assigned a constant 'worker_device'
// specifying what device they process work for.  This variable is initialized
// at thread creation time and is constant afterwards.  This is used when
// handling reentrant backwards calls; see Note [Reentrant backwards]
static thread_local int worker_device = NO_DEVICE;

// This variable is true if ALL invocations in the stack of re-entrant engine
// invocations are imperative backwards. This special variable is needed for the
// gradient checkpointing feature only.
static thread_local bool checkpoint_valid = true;

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
    , fn(std::move(fn))
    , inputs(std::move(inputs)) {}
};

// Returns true when t2 should be (weakly) BEFORE t1 in the queue.
struct CompareFunctionTaskTime {
  bool operator()(FunctionTask const & t1, FunctionTask const & t2) {
    return t1.fn->sequence_nr() < t2.fn->sequence_nr();
  }
};

struct ReadyQueue {
  std::priority_queue<FunctionTask, std::vector<FunctionTask>, CompareFunctionTaskTime> heap;
  std::condition_variable not_empty;
  std::mutex mutex;

  void push(FunctionTask item);
  FunctionTask pop();
};

// Note [Reentrant backwards]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// To understand the reentrant backwards problem, we have to notice two
// aspects of how the autograd engine is implemented today:
//
//  1. When you call Engine::execute(), you want to block until
//  differentiation finishes so that you can get the final result variables
//  of the backwards pass.
//
//  2. The engine operates by having a single worker thread per work queue,
//  and every work queue is pinned to a specific device where the
//  operation is executed.
//
// The problem is, suppose that you call backward() inside of a worker
// thread.  By property (1), we're supposed to block until the nested task
// finishes.  However, by property (2), this worker thread is on the
// hook for processing the tasks assigned to it; we better not block,
// because then all of our backward executions (including the one we
// just started) will deadlock!
//
// Here's our cunning idea: instead of blocking, just get back to work
// on whatever task queue you should have been working on previously
// (this is saved via the thread local variable worker_device)!  There are
// "simply" two things you have to arrange for:
//
//  - We have to promptly kick ourselves out of the thread_main() loop
//    when our graph_task complete, because we need to unblock the
//    parent function tasks that started the reentrant execution in
//    the first place.  This is why thread_main() takes an optional
//    graph_task as input.
//
//  - When we finish a GraphTask, we have to make sure we wake up the worker
//    thread so that it actually has a chance to exit the thread_main()
//    loop.  Thus the faffing about in thread_main() after
//    evaluate_function() completes.


// GraphTask holds metadata needed for a single execution of backward()
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
  // run in a "default" mode, which means that all next_edges we encounter should
  // get executed. If it's not empty, only functions that have an entry and this entry
  // has needed == True should be executed.
  // exec_info.empty() means it's .backward(), otherwise it's .grad().
  std::unordered_map<Function*, ExecInfo> exec_info;
  std::vector<Variable> captured_vars;

  void init_to_execute(Function& graph_root, const edge_list& outputs);

  // The value of worker_device in the thread that created this task.
  // See Note [Reentrant backwards]
  int owner;

  bool can_checkpoint() {
    return exec_info.empty();
  }

  GraphTask(bool keep_graph, bool grad_mode)
    : has_error(false)
    , outstanding_tasks(0)
    , keep_graph(keep_graph)
    , grad_mode(grad_mode)
    , owner(NO_DEVICE) {}
};

auto ReadyQueue::push(FunctionTask item) -> void {
  {
    std::lock_guard<std::mutex> lock(mutex);
    ++item.base->outstanding_tasks;
    heap.push(std::move(item));
  }
  not_empty.notify_one();
}

auto ReadyQueue::pop() -> FunctionTask {
  std::unique_lock<std::mutex> lock(mutex);
  not_empty.wait(lock, [this]{ return !heap.empty(); });
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  auto task = std::move(const_cast<FunctionTask&>(heap.top())); heap.pop();
  return task;
}

Engine::Engine() = default;

// This Engine's ReadyQueues and their corresponding threads are leaked here
Engine::~Engine() = default;

// TODO: Engine is not written in a way that it can deal with anything that's
// not CUDA.
auto Engine::thread_init(int device) -> void {
  THInferNumThreads();
#ifdef USE_CUDA
  // NB: We MUST NOT construct the guard for device -1,
  // as in some settings we compile with USE_CUDA, but
  // have lazy stubs for CUDA functionality (so actually
  // attempting to setup a guard(-1) will cause an
  // error, because it will still query cudaGetDevice).
  at::cuda::OptionalCUDAGuard guard;
  if (device != -1) {
    guard.set_index(device);
  }
#endif
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
  // Why the test on graph_task->outstanding_tasks?  See
  // Note [Reentrant backwards]
  while (!graph_task || graph_task->outstanding_tasks > 0) {
    FunctionTask task = queue->pop();
    if (task.fn && !task.base->has_error.load()) {
      GradMode::set_enabled(task.base->grad_mode);
      try {
        evaluate_function(task);
      } catch (std::exception& e) {
        thread_on_exception(task, e);
      }
    }
    // Notify downstream about the completion of tasks depending
    // on both where the task was executed, and who owned the overall
    // graph (in case of reentrant execution.)  See Note [Reentrant backwards].
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
          ready_queue(base_owner).push(FunctionTask(task.base, nullptr, InputBuffer(0)));
        }
      }
    }
  }
}

auto Engine::thread_on_exception(FunctionTask& task, std::exception& e) -> void {
  std::lock_guard<std::mutex> lock(task.base->mutex);
  if (!task.base->has_error.load()) {
    if (AnomalyMode::is_enabled()) {
      task.fn->metadata()->print_stack();
    }
    task.base->exception = std::current_exception();
    task.base->has_error = true;
  }
}

static variable_list call_pre_hooks(Function& fn, variable_list inputs) {
  for (const auto& hook : fn.pre_hooks()) {
    inputs = (*hook)(inputs);
  }
  return inputs;
}

static variable_list call_post_hooks(Function& fn, variable_list outputs, variable_list inputs) {
  for (const auto& hook : fn.post_hooks()) {
    outputs = (*hook)(outputs, inputs);
  }
  return outputs;
}

static bool is_compatible_type(const at::Type& expected, const at::Type& actual) {
  // Types are compatible if they exactly match or if the gradient is a sparse
  // version of the expected type.
  return expected == actual || (actual.is_sparse() &&
      expected == actual.toBackend(toDense(actual.backend())));
}

template<typename F>
static void validate_outputs(const edge_list& edges, variable_list& grads, const F& format_error) {
  if (grads.size() != edges.size()) {
    std::stringstream ss;
    ss << "invalid number of gradients - expected ";
    ss << edges.size() << ", but got " << grads.size();
    AT_ERROR(format_error(ss.str()));
  }
  for (size_t i = 0; i < grads.size(); i++) {
    const auto& edge = edges[i];
    if (!edge.is_valid()) continue;

    const auto& metadata = edge.function->input_metadata(edge.input_nr);
    const auto& output = grads[i];
    if (!output.defined()) {
      // FIXME: TestJit.test_ge_optimized fails this assertion.
      // std::stringstream ss;
      // ss << "undefined gradient at index " << i;
      // AT_ERROR(format_error(ss.str()));
      continue;
    }
    if (!grads[i].sizes().equals(metadata.shape())) {
      if (!at::is_expandable_to(metadata.shape(), grads[i].sizes())) {
        std::stringstream ss;
        ss << "invalid gradient at index " << i << " - got ";
        ss << grads[i].sizes() << " but expected shape compatible with ";
        ss << metadata.shape();
        AT_ERROR(format_error(ss.str()));
      }
      grads[i] = at::sum_to(grads[i], metadata.shape());
    }
    if (!is_compatible_type(metadata.type(), grads[i].type())) {
      std::stringstream ss;
      ss << "invalid gradient at index " << i << " - expected type ";
      ss << metadata.type() << " but got " << grads[i].type();
      AT_ERROR(format_error(ss.str()));
    }
    const auto output_device = output.is_cuda() ? output.get_device() : -1;
    if (output_device != metadata.device()) {
      std::stringstream ss;
      ss << "invalid gradient at index " << i << " - expected device ";
      ss << metadata.device() << " but got " << output_device;
      AT_ERROR(format_error(ss.str()));
    }
  }
}

static variable_list call_function(FunctionTask& task) {
  bool prev_checkpoint_valid_state = checkpoint_valid;
  checkpoint_valid = task.base->can_checkpoint() && prev_checkpoint_valid_state;
  auto& fn = *task.fn;
  auto inputs = call_pre_hooks(fn, InputBuffer::variables(std::move(task.inputs)));

  if(!task.base->keep_graph) {
    fn.will_release_variables();
  }

  const auto has_post_hooks = !fn.post_hooks().empty();
  variable_list outputs;

  if(has_post_hooks){
    auto inputs_copy = inputs;
    outputs = fn(std::move(inputs_copy));
  }else{
    outputs = fn(std::move(inputs));
  }

  validate_outputs(fn.next_edges(), outputs, [&](const std::string& msg) {
    std::ostringstream ss;
    ss << "Function "  << fn.name() << " returned an " << msg;
    return ss.str();
  });
  checkpoint_valid = prev_checkpoint_valid_state;

  if(has_post_hooks){
    // NOLINTNEXTLINE(bugprone-use-after-move)
    return call_post_hooks(fn, std::move(outputs), std::move(inputs));
  }
  return outputs;
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
    fn.release_variables();
  }

  int num_outputs = outputs.size();
  if (num_outputs == 0) return; // Don't even acquire the mutex

  if (AnomalyMode::is_enabled()) {
    AutoGradMode grad_mode(false);
    for (int i = 0; i < num_outputs; ++i) {
      auto& output = outputs[i];
      at::OptionalDeviceGuard guard(device_of(output));
      if (output.defined() && output.ne(output).any().item<uint8_t>()) {
        std::stringstream ss;
        ss << "Function '" << fn.name() << "' returned nan values in its " << i << "th output.";
        throw std::runtime_error(ss.str());
      }
    }
  }

  std::lock_guard<std::mutex> lock(task.base->mutex);
  for (int i = 0; i < num_outputs; ++i) {
    auto& output = outputs[i];
    const auto& next = fn.next_edge(i);

    if (!next.is_valid()) continue;

    // Check if the next function is ready to be computed
    bool is_ready = false;
    auto& dependencies = task.base->dependencies;
    auto it = dependencies.find(next.function.get());
    if (it == dependencies.end()) {
      auto name = next.function->name();
      throw std::runtime_error(std::string("dependency not found for ") + name);
    } else if (--it->second == 0) {
      dependencies.erase(it);
      is_ready = true;
    }

    auto& not_ready = task.base->not_ready;
    auto not_ready_it = not_ready.find(next.function.get());
    if (not_ready_it == not_ready.end()) {
      // Skip functions that aren't supposed to be executed
      if (!exec_info.empty()) {
        auto it = exec_info.find(next.function.get());
        if (it == exec_info.end() || !it->second.should_execute()) {
          continue;
        }
      }
      // No buffers have been allocated for the function
      InputBuffer input_buffer(next.function->num_inputs());
      input_buffer.add(next.input_nr, std::move(output));
      if (is_ready) {
        auto& queue = ready_queue(input_buffer.device());
        queue.push(FunctionTask(task.base, next.function, std::move(input_buffer)));
      } else {
        not_ready.emplace(next.function.get(), std::move(input_buffer));
      }
    } else {
      // The function already has a buffer
      auto &input_buffer = not_ready_it->second;
      input_buffer.add(next.input_nr, std::move(output));
      if (is_ready) {
        auto& queue = ready_queue(input_buffer.device());
        queue.push(FunctionTask(task.base, next.function, std::move(input_buffer)));
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
  while (!queue.empty()) {
    auto fn = queue.back(); queue.pop_back();
    for (const auto& edge : fn->next_edges()) {
      if (auto next_ptr = edge.function.get()) {
        dependencies[next_ptr] += 1;
        const bool was_inserted = seen.insert(next_ptr).second;
        if (was_inserted) queue.push_back(next_ptr);
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

auto Engine::execute(const edge_list& roots,
                     const variable_list& inputs,
                     bool keep_graph,
                     bool create_graph,
                     const edge_list& outputs) -> variable_list {
  std::call_once(start_threads_flag, &Engine::start_threads, this);

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  validate_outputs(roots, const_cast<variable_list&>(inputs), [](const std::string& msg) {
    return msg;
  });

  // Callbacks are only valid for the duration of this run and should always be cleared
  ClearCallbacks _cb_guard(final_callbacks, post_callbacks_lock);

  GraphTask graph_task(keep_graph, create_graph);
  std::unique_lock<std::mutex> lock(graph_task.mutex);

  // Now compute the dependencies for all executable functions and queue the root
  auto graph_root = std::make_shared<GraphRoot>(roots, inputs);
  compute_dependencies(graph_root.get(), graph_task);
  if (!outputs.empty()) {
    graph_task.init_to_execute(*graph_root, outputs);
  }
  ready_queue(-1).push(FunctionTask(&graph_task, std::move(graph_root), InputBuffer(0)));

  // Not a worker
  if (worker_device == NO_DEVICE) {
    // Wait for all tasks to complete
    graph_task.not_done.wait(lock, [&graph_task]{
      return graph_task.outstanding_tasks.load() == 0;
    });
  } else {
    // Get back to work while we wait for our new graph_task to
    // complete!
    // See Note [Reentrant backwards]
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
  // WARNING: Don't use a range-for loop here because more callbacks may be
  // added in between callback calls, so iterators may become invalidated.
  // NOLINTNEXTLINE(modernize-loop-convert)
  for (size_t i = 0; i < final_callbacks.size(); ++i) {
    cb_lock.unlock();
    final_callbacks[i]();
    cb_lock.lock();
  }

  return graph_task.captured_vars;
}

// note that when python is present, this base engine will be overriden
// with a PythonEngine. Because this typically happens before get_default_engine
// is called, this base engine will never be created.
static Engine& get_base_engine() {
  static Engine engine;
  return engine;
}

std::atomic<EngineStub> engine_stub(get_base_engine);

void set_default_engine_stub(EngineStub stub) {
  engine_stub.store(stub);
}


Engine& Engine::get_default_engine() {
  return engine_stub.load()();
}

void Engine::queue_callback(std::function<void()> callback) {
  std::lock_guard<std::mutex> lock(post_callbacks_lock);
  final_callbacks.emplace_back(std::move(callback));
}

bool Engine::is_checkpoint_valid() {
  return checkpoint_valid;
}

auto Engine::ready_queue(int device) -> ReadyQueue& {
  return *ready_queues.at(device + 1);
}

auto Engine::start_threads() -> void {
  int num_devices = 0;
#ifdef USE_CUDA
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

void GraphTask::init_to_execute(Function& graph_root, const edge_list& outputs) {
  exec_info[&graph_root].needed = true;

  int output_idx = 0;
  for (auto & output_edge : outputs) {
    Function *output = output_edge.function.get();
    auto & info = exec_info[output];
    if (!info.captures)
      info.captures = make_unique<std::vector<ExecInfo::Capture>>();
    info.captures->emplace_back(output_edge.input_nr, output_idx++);
  }
  captured_vars.resize(output_idx);

  // NB: this is an uglier version (recursion replaced with iteration) of the following code:
  // is_needed = {}
  // def compute_is_needed(fn):
  //   if fn not in is_needed:
  //     is_needed[fn] = any(compute_is_needed(next_edge)
  //                         for next_edge in fn.next_edges)
  //   return is_needed[fn]
  struct Frame {
    Frame (Function *fn) : fn(fn), next_next_fn(0) {}
    Function *fn;
    size_t next_next_fn;

    Function* get_next_fn() {
      const auto & next = fn->next_edges();
      auto num_next = next.size();
      while (next_next_fn < num_next) {
        auto fn = next[next_next_fn++].function.get();
        if (fn) return fn;
      }
      return nullptr;
    }
  };
  std::vector<Frame> stack;
  std::unordered_set<Function*> seen;
  for (const auto & input : graph_root.next_edges()) {
    if (seen.count(input.function.get()) > 0) continue;
    stack.emplace_back(input.function.get());
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
        const auto & next_edges = frame.fn->next_edges();
        const bool needed = std::any_of(
            next_edges.begin(), next_edges.end(), [&](const Edge& edge) {
              auto it = exec_info.find(edge.function.get());
              return it != exec_info.end() && it->second.should_execute();
            });
        exec_info[frame.fn].needed = needed;
        stack.pop_back();
      }
    }
  }
}

}} // namespace torch::autograd
