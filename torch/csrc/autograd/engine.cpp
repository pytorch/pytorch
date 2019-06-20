#include <torch/csrc/autograd/engine.h>

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/anomaly_mode.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/memory.h>

#include <ATen/DeviceGuard.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Parallel.h>
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
  GraphTask* base_;
  std::shared_ptr<Function> fn_;
  // This buffer serves as an implicit "addition" node for all of the
  // gradients flowing here.  Once all the dependencies are finished, we
  // use the contents of this buffer to run the function.
  InputBuffer inputs_;
  // When worker receives a task with isShutdownTask = true, it will immediately
  // exit. The engine sends a shutdown task to every queue upon its destruction.
  bool isShutdownTask_;

  FunctionTask(GraphTask* base, std::shared_ptr<Function> fn, InputBuffer inputs, bool isShutdownTask = false)
    : base_(base)
    , fn_(std::move(fn))
    , inputs_(std::move(inputs))
    , isShutdownTask_(isShutdownTask) {}
};

// Returns true when t2 should be (weakly) BEFORE t1 in the queue.
// Shutdown tasks are first and then empty FunctionTask are next.
struct CompareFunctionTaskTime {
  bool operator()(FunctionTask const & t1, FunctionTask const & t2) {
    if (t2.isShutdownTask_) {
      return true;
    } else if (!t1.fn_ || t1.isShutdownTask_) {
      return false;
    } else if (!t2.fn_) {
      return true;
    } else {
      return t1.fn_->sequence_nr() < t2.fn_->sequence_nr();
    }
  }
};

struct ReadyQueue {
  std::priority_queue<FunctionTask, std::vector<FunctionTask>, CompareFunctionTaskTime> heap_;
  // To notify threads waiting on the ReadyQueue of available tasks on the heap_
  std::condition_variable not_empty_;
  // To protect read and writes to heap_
  std::mutex mutex_;

  void push(FunctionTask item);
  void pushShutdownTask();
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
// We maintain a pool of threads waiting for work to do
// When a reentrant backwards call occurs, the current thread blocks
// and a thread from the pool is woken up to complete the blocking tasks
// If there are no threads avaliable, a new thread is spawned. The new thread
// will continue processing tasks from the same ReadyQueue as the parent worker
//
// When the GraphTask is finished, the parent worker thread that is waiting on
// the task is notified and the current thread returns to the pool.

// GraphTask holds metadata needed for a single execution of backward()
struct GraphTask {
  std::exception_ptr exception_;
  // Indicates if an error occurred while executing any task.  When this is
  // true, it signals all threads to stop executing.
  std::atomic_bool has_error_;
  std::atomic<uint64_t> outstanding_tasks_;
  // It is safe to read grad_mode_ and keep_graph_ without synchronization
  bool keep_graph_;
  bool grad_mode_;

  // To protect reads/writes to no_ready_, dependencies_ , captured_vars_ and
  // exception_
  std::mutex mutex_;
  // Notified when a task finishes executing.  Check outstanding_tasks_ to see
  // if all tasks are done.
  std::condition_variable not_done_;
  std::unordered_map<Function*, InputBuffer> not_ready_;
  std::unordered_map<Function*, int> dependencies_;

  struct ExecInfo {
    struct Capture {
      Capture(int input_idx, int output_idx) : input_idx_(input_idx), output_idx_(output_idx) {}
      int input_idx_; // within Function inputs
      int output_idx_; // within the output vector of a GraphTask
    };

    bool should_execute() const {
      return needed_ || captures_;
    }

    bool needed_ = false;
    std::unique_ptr<std::vector<Capture>> captures_;
  };
  // Exec info has a bit complicated semantics. If it's empty, it means the task is
  // run in a "default" mode, which means that all next_edges we encounter should
  // get executed. If it's not empty, only functions that have an entry and this entry
  // has needed == True should be executed.
  // exec_info_.empty() means it's .backward(), otherwise it's .grad().
  // exec_info_ is safe to read without synchronization
  std::unordered_map<Function*, ExecInfo> exec_info_;
  std::vector<Variable> captured_vars_;

  void init_to_execute(Function& graph_root, const edge_list& outputs);

  // The value of worker_device in the thread that created this task.
  // See Note [Reentrant backwards]
  // Safe to read owner_ without synchronizaton
  int owner_;

  bool can_checkpoint() {
    return exec_info_.empty();
  }

  GraphTask(bool keep_graph, bool grad_mode)
    : has_error_(false)
    , outstanding_tasks_(0)
    , keep_graph_(keep_graph)
    , grad_mode_(grad_mode)
    , owner_(NO_DEVICE) {}
};

auto ReadyQueue::push(FunctionTask item) -> void {
  {
    // Lock mutex for writing to heap_
    std::lock_guard<std::mutex> lock(mutex_);
    ++item.base_->outstanding_tasks_;
    heap_.push(std::move(item));
  }
  not_empty_.notify_one();
}

auto ReadyQueue::pushShutdownTask() -> void {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    heap_.push(FunctionTask(nullptr, nullptr, InputBuffer(0), true));
  }
  not_empty_.notify_one();
}

auto ReadyQueue::pop() -> FunctionTask {
  // Lock mutex for accesses to heap_
  std::unique_lock<std::mutex> lock(mutex_);
  not_empty_.wait(lock, [this]{ return !heap_.empty(); });
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  auto task = std::move(const_cast<FunctionTask&>(heap_.top())); heap_.pop();
  return task;
}

Engine::Engine() = default;

// Send shutdown tasks to all ReadyQueues if no backward tasks are running
// Even though readyQueue should be empty, shutdown tasks have the highest
// priority
Engine::~Engine() {
  bool noBackward = true;
  for (auto& queue: ready_queues_) {
    std::lock_guard<std::mutex> lock(queue->mutex_);
    noBackward =  noBackward && queue->heap_.empty();
  }
  if (noBackward) {
    for (auto& queue : ready_queues_) {
     queue->pushShutdownTask();
    }
  }
  // Othewise threads are leaked
}

void Engine::set_device(int device) {
  // NB: We MUST NOT construct the guard for device -1,
  // as in some settings we compile with cuda, but
  // have lazy stubs for CUDA functionality (so actually
  // attempting to setup a guard(-1) will cause an
  // error, because it will still query cudaGetDevice).
  //
  // Don't use DeviceGuard here because its destructor may be called before the
  // device is reset. This is fine because the device is thread local.
  if (device != -1) {
    for (size_t i = 0; i < static_cast<size_t>(c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES); i++) {
      auto* impl = c10::impl::device_guard_impl_registry[i].load();
      if (impl && device < impl->deviceCount()) {
        impl->setDevice(at::Device(static_cast<c10::DeviceType>(i), device));
      }
    }
  }
  worker_device = device;
}

auto Engine::thread_init(int device) -> void {
  at::init_num_threads();
  // Note [Allocating GPUs to autograd threads]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // What's our strategy here?  Originally, the autograd engine was written
  // with only CUDA in mind.  We allocate one thread to handle all CPU
  // operations, and a thread per CUDA device.
  //
  // But what if we have OTHER devices?  There are two plausible
  // strategies:
  //
  //  - We can allocate threads equal to max(num_cuda_devices, num_xla_devices,
  //    ...) and colocate cuda device 0 with xla device 0
  //  - We can allocate threads equal to sum(num_cuda_devices, num_xla_devices,
  //    ...) keeping everyone separate.
  //
  // We don't have any good reason to prefer one or the other, so we've
  // arbitrarily picked to colocate devices.  Maybe the other approach is
  // better.
  set_device(device);
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
  auto queue = ready_queues_[worker_device + 1];
  // Why the test on graph_task->outstanding_tasks_?  See
  // Note [Reentrant backwards]
  while (!graph_task || graph_task->outstanding_tasks_ > 0) {
    FunctionTask task = queue->pop();
    // This will only work if the worker is running a non backward task
    // TODO Needs to be fixed this to work in all cases
    if (task.isShutdownTask_) {
      C10_LOG_API_USAGE_ONCE("torch.autograd.thread_shutdown");
      break;
    }
    if (task.fn_ && !task.base_->has_error_.load()) {
      GradMode::set_enabled(task.base_->grad_mode_);
      try {
        evaluate_function(task);
      } catch (std::exception& e) {
        thread_on_exception(task, e);
      }
    }
    // Notify downstream about the completion of tasks depending
    // on both where the task was executed, and who owned the overall
    // graph (in case of reentrant execution.)  See Note [Reentrant backwards].
    auto base_owner = task.base_->owner_;
    // Task from a non-worker thread. Easy case.
    if (base_owner == NO_DEVICE) {
      if (--task.base_->outstanding_tasks_ == 0) {
        // Lock mutex to notify the GraphTask waiting on not_done_
        std::lock_guard<std::mutex> lock(task.base_->mutex_);
        task.base_->not_done_.notify_all();
      }
    } else {
      // If it's a task initiated from this thread, decrease the counter, but
      // don't do anything - loop condition will do all checks for us next.
      if (base_owner == worker_device) {
        --task.base_->outstanding_tasks_;
      // Otherwise send a dummy function task to the owning thread just to
      // ensure that it's not sleeping. If it has work, it might see that
      // graph_task->outstanding_tasks_ == 0 before it gets to the task, but
      // it's a no-op anyway.
      } else if (base_owner != worker_device) {
        if (--task.base_->outstanding_tasks_ == 0) {
          // Synchronize outstanding_tasks_ with queue mutex
          std::atomic_thread_fence(std::memory_order_release);
          ready_queue_by_index(base_owner).push(FunctionTask(task.base_, nullptr, InputBuffer(0)));
        }
      }
    }
  }

  if (graph_task) {
    graph_task->not_done_.notify_all();
  }
}

void Engine::reentrant_thread_init() {
  at::init_num_threads();
  auto tp_shared= thread_pool_shared_;
  while(true) {
    std::unique_lock<std::mutex> lk(tp_shared->mutex_);
    tp_shared->work_.wait(lk, [&tp_shared]{ return !tp_shared->graphtasks_queue_.empty();});
    --thread_pool_shared_->num_workers_;
    auto graph_task = tp_shared->graphtasks_queue_.front();
    tp_shared->graphtasks_queue_.pop();
    lk.unlock();
    set_device(graph_task->owner_);
    thread_main(graph_task);
    ++thread_pool_shared_->num_workers_;
  }
}

auto Engine::thread_on_exception(FunctionTask& task, std::exception& e) -> void {
  // Lock mutex for writing to task.base_->exception_
  std::lock_guard<std::mutex> lock(task.base_->mutex_);
  if (!task.base_->has_error_.load()) {
    if (AnomalyMode::is_enabled()) {
      task.fn_->metadata()->print_stack();
    }
    task.base_->exception_ = std::current_exception();
    task.base_->has_error_ = true;
  }
}

static variable_list call_pre_hooks(Function& fn, variable_list inputs) {
  for (const auto& hook : fn.pre_hooks()) {
    inputs = (*hook)(inputs);
  }
  return inputs;
}

static variable_list call_post_hooks(Function& fn, variable_list outputs, const variable_list& inputs) {
  for (const auto& hook : fn.post_hooks()) {
    outputs = (*hook)(outputs, inputs);
  }
  return outputs;
}

static bool is_compatible_type(const at::DeprecatedTypeProperties& expected, const at::DeprecatedTypeProperties& actual) {
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
      grads[i] = at::sum_to(std::move(grads[i]), metadata.shape());
    }
    if (!is_compatible_type(metadata.type(), grads[i].type())) {
      std::stringstream ss;
      ss << "invalid gradient at index " << i << " - expected type ";
      ss << metadata.type() << " but got " << grads[i].type();
      AT_ERROR(format_error(ss.str()));
    }
    auto output_device = output.device();
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
  checkpoint_valid = task.base_->can_checkpoint() && prev_checkpoint_valid_state;
  auto& fn = *task.fn_;
  auto inputs = call_pre_hooks(fn, InputBuffer::variables(std::move(task.inputs_)));

  if(!task.base_->keep_graph_) {
    fn.will_release_variables();
  }

  const auto has_post_hooks = !fn.post_hooks().empty();
  variable_list outputs;

  if(has_post_hooks){
    // In functions/accumulate_grad.cpp, there is some logic to check the conditions under which
    // the incoming gradient can be stolen directly (which elides a deep copy) instead of cloned.
    // One of these conditions is that the incoming gradient's refcount must be 1 (nothing else
    // is referencing the same data).  Stashing inputs_copy here bumps the refcount, so if post hooks
    // are employed, it's actually still ok for accumulate_grad.cpp to steal the gradient if the
    // refcount is 2.
    //
    // "new_grad.use_count() <= 1 + !post_hooks().empty()" in accumulate_grad.cpp accounts for this,
    // but also creates a silent dependency between engine.cpp (ie, this particular engine
    // implementation) and accumulate_grad.cpp.
    //
    // If you change the logic here, make sure it's compatible with accumulate_grad.cpp.
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
    return call_post_hooks(fn, std::move(outputs), inputs);
  }
  return outputs;
}

auto Engine::evaluate_function(FunctionTask& task) -> void {
  // If exec_info_ is not empty, we have to instrument the execution
  auto & exec_info_ = task.base_->exec_info_;
  if (!exec_info_.empty()) {
    auto & fn_info = exec_info_.at(task.fn_.get());
    if (auto *capture_vec = fn_info.captures_.get()) {
      // Lock mutex for writing to task.base_->captured_vars_
      std::lock_guard<std::mutex> lock(task.base_->mutex_);
      for (auto capture : *capture_vec) {
        task.base_->captured_vars_[capture.output_idx_] = task.inputs_[capture.input_idx_];
      }
    }
    if (!fn_info.needed_) return;
  }

  auto outputs = call_function(task);

  auto& fn = *task.fn_;
  if (!task.base_->keep_graph_) {
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

  // Lock mutex for the accesses to GraphTask dependencies_ and not_ready_ below
  std::lock_guard<std::mutex> lock(task.base_->mutex_);
  for (int i = 0; i < num_outputs; ++i) {
    auto& output = outputs[i];
    const auto& next = fn.next_edge(i);

    if (!next.is_valid()) continue;

    // Check if the next function is ready to be computed
    bool is_ready = false;
    auto& dependencies = task.base_->dependencies_;
    auto it = dependencies.find(next.function.get());
    if (it == dependencies.end()) {
      auto name = next.function->name();
      throw std::runtime_error(std::string("dependency not found for ") + name);
    } else if (--it->second == 0) {
      dependencies.erase(it);
      is_ready = true;
    }

    auto& not_ready = task.base_->not_ready_;
    auto not_ready_it = not_ready.find(next.function.get());
    if (not_ready_it == not_ready.end()) {
      // Skip functions that aren't supposed to be executed
      if (!exec_info_.empty()) {
        auto it = exec_info_.find(next.function.get());
        if (it == exec_info_.end() || !it->second.should_execute()) {
          continue;
        }
      }
      // No buffers have been allocated for the function
      InputBuffer input_buffer(next.function->num_inputs());
      input_buffer.add(next.input_nr, std::move(output));
      if (is_ready) {
        auto& queue = ready_queue(input_buffer.device());
        queue.push(FunctionTask(task.base_, next.function, std::move(input_buffer)));
      } else {
        not_ready.emplace(next.function.get(), std::move(input_buffer));
      }
    } else {
      // The function already has a buffer
      auto &input_buffer = not_ready_it->second;
      input_buffer.add(next.input_nr, std::move(output));
      if (is_ready) {
        auto& queue = ready_queue(input_buffer.device());
        queue.push(FunctionTask(task.base_, next.function, std::move(input_buffer)));
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
  auto& dependencies = task.dependencies_;
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
    : callbacks_(callbacks)
    , callbacks_lock_(callbacks_lock) { clear(); }
  ~ClearCallbacks() { clear(); }

  void clear() {
    std::lock_guard<std::mutex> lock(callbacks_lock_);
    callbacks_.clear();
  }

  std::vector<std::function<void()>>& callbacks_;
  std::mutex& callbacks_lock_;
};

auto Engine::execute(const edge_list& roots,
                     const variable_list& inputs,
                     bool keep_graph,
                     bool create_graph,
                     const edge_list& outputs) -> variable_list {
  std::call_once(start_threads_flag_, &Engine::start_threads, this);

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  validate_outputs(roots, const_cast<variable_list&>(inputs), [](const std::string& msg) {
    return msg;
  });

  // Callbacks are only valid for the duration of this run and should always be cleared
  // Lock post_callbacks_lock_ before clearing final_callbacks_
  ClearCallbacks _cb_guard(final_callbacks_, post_callbacks_lock_);

  GraphTask graph_task(keep_graph, create_graph);
  // Lock mutex while GraphTask is being set up
  std::unique_lock<std::mutex> lock(graph_task.mutex_);

  // Now compute the dependencies for all executable functions and queue the root
  auto graph_root = std::make_shared<GraphRoot>(roots, inputs);
  compute_dependencies(graph_root.get(), graph_task);
  if (!outputs.empty()) {
    graph_task.init_to_execute(*graph_root, outputs);
  }
  ready_queue(at::kCPU).push(FunctionTask(&graph_task, std::move(graph_root), InputBuffer(0)));

  // Not a worker
  if (worker_device == NO_DEVICE) {
    // Wait for all tasks to complete
    graph_task.not_done_.wait(lock, [&graph_task]{
      return graph_task.outstanding_tasks_.load() == 0;
    });
  } else {
    // Get back to work while we wait for our new graph_task to
    // complete!
    // See Note [Reentrant backwards]
    // If no extra threads remaining, create a new thread for reentrant call
    graph_task.owner_ = worker_device;
    add_thread_pool_task(&graph_task);
    graph_task.not_done_.wait(lock, [&graph_task]{
      return graph_task.outstanding_tasks_.load() == 0;
    });
  }

  // Check for an exception while running backwards
  if (graph_task.has_error_.load()) {
    std::rethrow_exception(graph_task.exception_);
  }

  if (!graph_task.not_ready_.empty()) {
    throw std::runtime_error("could not compute gradients for some functions");
  }

  // Lock mutex during each iteration for accessing final_callbacks.size()
  // Unlocking is necessary, because the callback can register
  // more callbacks (or they can be registered from other threads
  // while it's waiting.
  std::unique_lock<std::mutex> cb_lock(post_callbacks_lock_);
  // WARNING: Don't use a range-for loop here because more callbacks may be
  // added in between callback calls, so iterators may become invalidated.
  // NOLINTNEXTLINE(modernize-loop-convert)
  for (size_t i = 0; i < final_callbacks_.size(); ++i) {
    cb_lock.unlock();
    final_callbacks_[i]();
    cb_lock.lock();
  }

  return graph_task.captured_vars_;
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
  std::lock_guard<std::mutex> lock(post_callbacks_lock_);
  final_callbacks_.emplace_back(std::move(callback));
}

bool Engine::is_checkpoint_valid() {
  return checkpoint_valid;
}

auto Engine::ready_queue(at::Device device) -> ReadyQueue& {
  // See Note [Allocating GPUs to autograd threads]
  if (device.type() == at::kCPU) {
    return *ready_queues_.at(0);
  } else {
    return *ready_queues_.at(device.index() + 1);
  }
}

// See Note [Allocating GPUs to autograd threads]
// NB: This would become obsolete if we truly allocated a CPU thread
// per device, rather than colocate.
auto Engine::ready_queue_by_index(int device_index) -> ReadyQueue& {
  return *ready_queues_.at(device_index + 1);
}

auto Engine::start_threads() -> void {
  // See Note [Allocating GPUs to autograd threads]
  c10::DeviceIndex num_devices = 0;
  for (const auto& impl_atomic : c10::impl::device_guard_impl_registry) {
    auto* impl = impl_atomic.load();
    if (impl) {
      num_devices = std::max(num_devices, impl->deviceCount());
    }
  }

  // One for CPU, plus one for every GPU device (but colocate GPUs of different
  // types)
  int num_threads = num_devices + 1;
  ready_queues_ = std::vector<std::shared_ptr<ReadyQueue>>(num_threads);
  for (auto& queue : ready_queues_)
    queue.reset(new ReadyQueue());

  thread_pool_shared_ = std::make_shared<ThreadPoolShared>();

  for (int i = 0; i < num_threads; ++i) {
    std::thread t(&Engine::thread_init, this, i - 1);
    t.detach();
  }
}

void Engine::add_thread_pool_task(GraphTask *graph_task) {
  std::lock_guard<std::mutex> lck(thread_pool_shared_->mutex_);
  // There may already be some items on the graphtasks_queue_ added by other
  // threads but not enough workers to get to the the new task that will be
  // added
  if (thread_pool_shared_->num_workers_ <= thread_pool_shared_->graphtasks_queue_.size()) {
    std::thread t(&Engine::reentrant_thread_init, this);
    ++thread_pool_shared_->num_workers_;
    t.detach();
  }
  thread_pool_shared_->graphtasks_queue_.push(graph_task);
  // This works even if new thread is created because wait() will test the
  // predicate before waiting
  thread_pool_shared_->work_.notify_one();
}

void GraphTask::init_to_execute(Function& graph_root, const edge_list& outputs) {
  exec_info_[&graph_root].needed_ = true;

  int output_idx = 0;
  for (auto & output_edge : outputs) {
    Function *output = output_edge.function.get();
    auto & info = exec_info_[output];
    if (!info.captures_)
      info.captures_ = make_unique<std::vector<ExecInfo::Capture>>();
    info.captures_->emplace_back(output_edge.input_nr, output_idx++);
  }
  captured_vars_.resize(output_idx);

  // NB: this is an uglier version (recursion replaced with iteration) of the following code:
  // is_needed = {}
  // def compute_is_needed(fn):
  //   if fn not in is_needed:
  //     is_needed[fn] = any(compute_is_needed(next_edge)
  //                         for next_edge in fn.next_edges)
  //   return is_needed[fn]
  struct Frame {
    Frame (Function *fn) : fn_(fn), next_next_fn_(0) {}
    Function *fn_;
    size_t next_next_fn_;

    Function* get_next_fn() {
      const auto & next = fn_->next_edges();
      auto num_next = next.size();
      while (next_next_fn_ < num_next) {
        auto fn = next[next_next_fn_++].function.get();
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
        const auto & next_edges = frame.fn_->next_edges();
        const bool needed = std::any_of(
            next_edges.begin(), next_edges.end(), [&](const Edge& edge) {
              auto it = exec_info_.find(edge.function.get());
              return it != exec_info_.end() && it->second.should_execute();
            });
        exec_info_[frame.fn_].needed_ = needed;
        stack.pop_back();
      }
    }
  }
}

}} // namespace torch::autograd
