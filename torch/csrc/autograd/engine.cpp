#include <torch/csrc/autograd/engine.h>

#include <torch/csrc/autograd/anomaly_mode.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/dynamo/compiled_autograd.h>

#include <ATen/DeviceAccelerator.h>
#include <ATen/DeviceGuard.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Parallel.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/isnan.h>
#endif

#include <c10/core/DeviceGuard.h>
#include <c10/core/Event.h>
#include <c10/core/Stream.h>
#include <c10/core/StreamGuard.h>
#include <c10/util/AbortHandler.h>
#include <c10/util/Exception.h>
#include <c10/util/ThreadLocal.h>
#include <c10/util/irange.h>
#include <c10/util/thread_name.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>

namespace torch::autograd {

namespace {
static bool in_bad_autograd_fork =
    false; // True for children forked after engine's thread pool init

// Called in the forked child if engine's thread pool has already been
// initialized
static void forked_autograd_child() {
  in_bad_autograd_fork = true;
}

// Should be called before unsafe for forks (thread pool) calls
static void track_bad_autograd_forks() {
#if !defined(WIN32)
  static auto result [[maybe_unused]] =
      pthread_atfork(nullptr, nullptr, forked_autograd_child);
#endif
}

inline bool should_run_in_cpu_ready_queue(c10::DeviceType device) {
  if (device == c10::kCPU || device == c10::kMeta || device == c10::kLazy) {
    return true;
  } else {
    return false;
  }
}

std::atomic<Engine::compiled_autograd_fn> the_compiled_autograd = nullptr;
#define COMPILED_AUTOGRAD_POISON \
  reinterpret_cast<Engine::compiled_autograd_fn>(1)
std::atomic<int32_t> num_threads_in_backwards;
struct CompiledAutogradThreadingDebugCheck {
  CompiledAutogradThreadingDebugCheck() {
    num_threads_in_backwards++;
  }
  ~CompiledAutogradThreadingDebugCheck() {
    release();
  }
  void release() {
    if (std::exchange(incremented, false)) {
      num_threads_in_backwards--;
    }
  }

 private:
  bool incremented{true};
};

} // namespace

// Threads spawned by the engine are assigned a 'worker_device' specifying
// what device they process work for. This variable is initialized at:
// 1. thread creation time for CUDA, XLA device threads, as they are
//    spinning threads waiting for works on their device.
// 2. before the graph task execution for CPU threads, as for each
//    backward call we use the caller thread to drive engine execution.
// This is used when handling reentrant backwards calls;
// See Note [Reentrant backwards]
static thread_local int worker_device = NO_DEVICE;

// This variable is true if ALL invocations in the stack of re-entrant engine
// invocations are imperative backwards. This special variable is needed for the
// gradient checkpointing feature only.
static thread_local bool checkpoint_valid = true;

// Number of nested reentrant backwards calls currently on this thread
static thread_local int current_depth = 0;

// For all device threads (i.e. CUDA, XLA), total_depth represents the total
// nested
//   reentrant backwards depths over all device threads.
// For CPU devices, it is the total depth associated with the original backward
// call.
static thread_local int total_depth = 0;

// The current GraphTask being executed by this thread. This helps
// queue_callback() to find the target GraphTask to append final callbacks.
C10_DEFINE_TLS_static(std::shared_ptr<GraphTask>, tls_current_graph_task);
#define current_graph_task (tls_current_graph_task.get())

// Every autograd worker thread is associated with a ready queue, which
// specifies the stream of work of this thread to do. This shared_ptr is a
// thread_local pointer to each thread's ready_queue, and it should be
// initialized via the Engine::init_local_ready_queue() call in each
// corresponding thread before execution.
//
// The CUDA, XLA threads are shared among all invocations of backwards via
// device_ready_queues_, while the caller thread is dedicated to processing work
// for devices returning true in should_run_in_cpu_ready_queue (most notably the
// CPU device). So any given graph task maintains its own cpu_ready_queue_ where
// you should send work for it to be done.
//
// For reentrant backward calls, if we spawn new thread from the current thread
// because we reached the maximum depth, the new thread will just reuse the same
// ReadyQueue with the parent thread for performance improvement.
// see Note [Reentrant backwards] for more details.
C10_DEFINE_TLS_static(std::shared_ptr<ReadyQueue>, tls_local_ready_queue);
#define local_ready_queue (tls_local_ready_queue.get())

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
// and a thread from the pool is woken up to complete the blocking tasks and an
// any other tasks that would have been assigned to that worker. If there are no
// threads available, a new thread is spawned. The new thread will continue
// processing tasks from the same ReadyQueue as the parent worker
//
// When the GraphTask is finished, the parent worker thread that is waiting on
// the task is notified and the current thread returns to the pool.

// Note [Streaming backwards]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// On CUDA/privateuse1 devices the autograd engine's device operations are run
// on the same stream that ran them in forward. This requires automatically
// syncing the streams so that function A finishes producing its
// output before function B consumes it.
//
// This synchronization occurs when outputs are placed into input buffers.
// The functions corresponding to input buffer positions have metadata
// recording their streams from forward, and during backward this
// data is used to sync the producer's stream with the consumer's.
//
// When a CUDA/privateuse1 function is run either all its inputs were
// accumulated on the stream used to run the function OR the inputs are on
// different devices and the function is responsible for properly acquiring
// them.
//
// User-facing stream semantics of a backward() (or torch.autograd.grad())
// call with respect to surrounding ops are the same as for any other call.
// See "Stream semantics of backward passes" on
// https://pytorch.org/docs/stable/notes/cuda.html
//
// Internally, backward() runs ops (including leaf nodes) on side threads.
// And streams are thread local. So GraphTask achieves the above semantics by
//  1. remembering the current streams on all active CUDA/privateuse1 devices
//     in the user-facing thread (aka, the thread that called execute() to
//     launch the GraphTask)
//  2. remembering the "leaf streams" (streams each backward leaf node ran on)
//  3. during exec_post_processing, for each leaf stream, sync the remembered
//     current streams (on the leaf stream's device) with that
//     leaf stream.

int NodeTask::getReentrantDepth() const {
  std::shared_ptr<GraphTask> graph_task = base_.lock();
  if (graph_task) {
    return graph_task->reentrant_depth_;
  } else {
    // The graph task is no longer valid indicating an error. As a result, we
    // try to move this to the front of the queue to ensure the autograd
    // engine threads pick up this error soon.
    return std::numeric_limits<int>::max();
  }
}

CheckpointValidGuard::CheckpointValidGuard(
    const std::shared_ptr<const GraphTask>& graph_task)
    : prev_checkpoint_valid_state(checkpoint_valid) {
  checkpoint_valid =
      graph_task->can_checkpoint() && prev_checkpoint_valid_state;
}

CheckpointValidGuard::~CheckpointValidGuard() {
  checkpoint_valid = prev_checkpoint_valid_state;
}

auto ReadyQueue::push(NodeTask item, bool incrementOutstandingTasks) -> void {
  {
    // Lock mutex for writing to heap_
    std::lock_guard<std::mutex> lock(mutex_);
    if (incrementOutstandingTasks) {
      std::shared_ptr<GraphTask> graph_task = item.base_.lock();
      TORCH_INTERNAL_ASSERT(graph_task, "GraphTask is no longer valid!");
      ++graph_task->outstanding_tasks_;
    }
    heap_.push(std::move(item));
  }
  not_empty_.notify_one();
}

auto ReadyQueue::pushShutdownTask() -> void {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    heap_.push(NodeTask({}, nullptr, InputBuffer(0), true));
  }
  not_empty_.notify_one();
}

size_t ReadyQueue::size() const {
  // Lock mutex for accesses to heap_
  std::unique_lock<std::mutex> lock(mutex_);
  return heap_.size();
}

auto ReadyQueue::pop() -> NodeTask {
  // Lock mutex for accesses to heap_
  std::unique_lock<std::mutex> lock(mutex_);
  not_empty_.wait(lock, [this] { return !heap_.empty(); });
  auto task = std::move(const_cast<NodeTask&>(heap_.top()));
  heap_.pop();
  return task;
}

bool ReadyQueue::empty() const {
  // Lock mutex for accesses to heap_
  std::unique_lock<std::mutex> lock(mutex_);
  return heap_.empty();
}

Engine::Engine()
    : max_recursion_depth_(MAX_DEPTH), non_reentrant_device_thread_count_(0) {}

Engine::~Engine() {
  stop();
}

// Send shutdown tasks to all device_ready_queues_ if no backward tasks are
// running Even though readyQueue should be empty, shutdown tasks have the
// highest priority
void Engine::stop() {
  if (stopped_) {
    return;
  }
  stopped_ = true;
  // Under some conditions, autograd threads can hang on shutdown
  // Do not wait for them to shutdown indefinitely but rely on timeout
  auto wait_duration_str =
      c10::utils::get_env("TORCH_AUTOGRAD_SHUTDOWN_WAIT_LIMIT");
  auto wait_duration =
      wait_duration_str ? std::atof(wait_duration_str->c_str()) : 10.0;
  bool noBackward = true;
  for (auto& queue : device_ready_queues_) {
    noBackward = noBackward && queue->empty();
  }
  if (noBackward && wait_duration > 0.0f) {
    for (auto& queue : device_ready_queues_) {
      queue->pushShutdownTask();
    }
    // Do not wait for termination of global threads on Windows
    // Because CRT terminates DLL threads before calling
    // global object destructors
#if !defined(_WIN32) || defined(C10_USE_MSVC_STATIC_RUNTIME)

    using namespace std::chrono_literals;
    // Set a deadline for how long it is OK to wait device threads to shutdown
    auto wait_deadline =
        std::chrono::steady_clock::now() + wait_duration * 1.0s;
    std::unique_lock<std::mutex> lk(non_reentrant_device_thread_mutex_);
    while (non_reentrant_device_thread_count_.load() != 0) {
      if (non_reentrant_device_thread_condvar_.wait_until(lk, wait_deadline) ==
          std::cv_status::timeout) {
        break;
      }
    }
#endif
  }
  // Otherwise threads are leaked
}

void Engine::release_workers() {
  std::unique_lock<std::mutex> lk(non_reentrant_device_thread_mutex_);
  non_reentrant_device_thread_count_.store(0);
  non_reentrant_device_thread_condvar_.notify_one();
}

void Engine::increment_non_reentrant_thread_count() {
  std::unique_lock<std::mutex> lk(non_reentrant_device_thread_mutex_);
  non_reentrant_device_thread_count_.fetch_add(1);
  non_reentrant_device_thread_condvar_.notify_one();
}

void Engine::decrement_non_reentrant_thread_count() {
  std::unique_lock<std::mutex> lk(non_reentrant_device_thread_mutex_);
  non_reentrant_device_thread_count_.fetch_sub(1);
  non_reentrant_device_thread_condvar_.notify_one();
}

void Engine::thread_init(
    int device,
    const std::shared_ptr<ReadyQueue>& ready_queue,
    bool should_increment) {
  // pthread_setname_np restricts the name to 16 characters including
  // the null byte.
  std::string thread_name = "pt_autograd_" + std::to_string(device);
  c10::setThreadName(thread_name);

  c10::set_terminate_handler();
  if (should_increment) {
    increment_non_reentrant_thread_count();
  }

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
  worker_device = device;

  // initialize each device thread's thread local ready queue with the ready
  // queue that is created before the thread initialization
  init_local_ready_queue(ready_queue);

  std::shared_ptr<GraphTask> graph_task = nullptr;
  thread_main(graph_task);
  if (should_increment) {
    // Decrement the count during shutdown if we incremented earlier.
    decrement_non_reentrant_thread_count();
  }
}

GraphTaskGuard::GraphTaskGuard(std::shared_ptr<GraphTask> graph_task)
    : last_graph_task_(std::move(current_graph_task)) {
  current_graph_task = std::move(graph_task);
}
GraphTaskGuard::~GraphTaskGuard() {
  restore_current_graph_task();
}

void GraphTaskGuard::restore_current_graph_task() {
  current_graph_task = std::move(last_graph_task_);
}

// The current graph task's exec_info is being used to trim unnecessary edegs
// during node evaluation, see `Node.task_should_compute_output()` function.
const std::unordered_map<Node*, GraphTask::ExecInfo>*
get_current_graph_task_exec_info() {
  return current_graph_task ? &current_graph_task->exec_info_ : nullptr;
}

const std::unordered_set<Node*>* get_current_graph_task_nodes_in_graph() {
  return current_graph_task ? &current_graph_task->nodes_in_graph_ : nullptr;
}

int get_current_graph_task_id() {
  return current_graph_task ? current_graph_task->id_ : -1;
}

bool get_current_graph_task_keep_graph() {
  return current_graph_task ? current_graph_task->keep_graph_ : true;
}

void add_node_to_current_graph_task_exec_info(Node* fn) {
  current_graph_task->exec_info_[fn].needed_ = true;
}

// NB: The engine itself does not use the outputs of this function.
std::vector<Node*> get_current_graph_task_execution_order() {
  std::shared_ptr<GraphTask> task = current_graph_task;
  if (!task) {
    return {};
  }

  // We could potentially check if there is only a single device here
  // but explicitly require this context doesn't seem bad either
  TORCH_CHECK(
      !c10::AutogradState::get_tls_state().get_multithreading_enabled(),
      "get_current_graph_task_execution_order expects the current backward to be "
      "executed with multithreading disabled, e.g. by running:\n\n"
      ">>> with torch.autograd.set_multithreading_enabled(False):\n"
      "...     torch.autograd.grad(...)\n");

  const bool check_exec_info = !task->exec_info_.empty();
  std::vector<Node*> out{};
  // Do a copy since we mutate it later
  std::unordered_map<Node*, int> dependencies = task->dependencies_;

  auto compare_seq_nr = [](Node* n1, Node* n2) {
    return n1->sequence_nr() < n2->sequence_nr();
  };
  std::priority_queue<Node*, std::vector<Node*>, decltype(compare_seq_nr)> heap(
      compare_seq_nr);

  for (Node* ptr : task->graph_roots_) {
    heap.push(ptr);
  }

  // Implementation notes:
  // - We need count dependencies even though we have sequence_nr, because
  //   in the accumulate_grad case we cannot assume the outputs to have higher
  //   sequence_nr than the inputs
  // - Don't need to check topological_nr because we have exec_info
  while (!heap.empty()) {
    Node* fn = heap.top();
    heap.pop();

    out.push_back(fn);
    for (const auto& edge : fn->next_edges()) {
      Node* next_ptr = edge.function.get();
      if (!next_ptr) {
        continue;
      }
      if (check_exec_info) {
        auto it = task->exec_info_.find(next_ptr);
        if (it == task->exec_info_.end() || !it->second.should_execute()) {
          continue;
        }
      }
      auto it = dependencies.find(edge.function.get());
      TORCH_INTERNAL_ASSERT(it != dependencies.end());
      if (--it->second == 0) {
        dependencies.erase(it);
        heap.push(next_ptr);
      }
    }
  }
  return out;
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
//
// thread_main is used by:
// 1). autograd threads for devices (i.e. CUDA, XLA)
// 2). the caller/owning thread of the backward call on CPU (sync mode)
// 3). Renetrant backward that invoked by either 1) or 2)
// The exit conditions are different for the above three cases.
// For 1), we are spinning on running the thread_main on device autograd
//         threads throughout the Engine lifetime, thread_main will get
//         terminated during Engine destruction by pushing shutdown tasks
// For 2), the owning thread of the backward call drives the thread_main
//         synchronously until the graph_task of that owning thread is
//         completed and exit the thread_main to continue executing the
//         result of caller's code.
// For 3), the reentrant backward that invokes
//         thread_main, either from 1) or 2), will not spin and will exit as
//         long as graph_task is completed and notify the owning thread as
//         needed.
auto Engine::thread_main(const std::shared_ptr<GraphTask>& graph_task) -> void {
  // When graph_task is nullptr, this is a long running thread that processes
  // tasks (ex: device threads). When graph_task is non-null (ex: reentrant
  // backwards, user thread), this function is expected to exit once that
  // graph_task complete.

  // local_ready_queue should already been initialized when we get into
  // thread_main
  TORCH_INTERNAL_ASSERT(local_ready_queue != nullptr);
  while (graph_task == nullptr || !graph_task->future_result_->completed()) {
    // local_graph_task represents the graph_task we retrieve from the queue.
    // The outer graph_task represents the overall graph_task we need to execute
    // for reentrant execution.
    std::shared_ptr<GraphTask> local_graph_task;
    {
      // Scope this block of execution since NodeTask is not needed after this
      // block and can be deallocated (release any references to grad tensors
      // as part of inputs_).
      NodeTask task = local_ready_queue->pop();
      // This will only work if the worker is running a non backward task
      // TODO Needs to be fixed this to work in all cases
      if (task.isShutdownTask_) {
        C10_LOG_API_USAGE_ONCE("torch.autograd.thread_shutdown");
        break;
      }

      local_graph_task = task.base_.lock();
      if (!local_graph_task) {
        // GraphTask for function is no longer valid, skipping further
        // execution.
        continue;
      }

      set_device(worker_device);

      if (task.fn_ && !local_graph_task->has_error_.load()) {
        // Set the ThreadLocalState before calling the function.
        // NB: The ThreadLocalStateGuard doesn't set the grad_mode because
        // GraphTask always saves ThreadLocalState without grad_mode.
        at::ThreadLocalStateGuard tls_guard(local_graph_task->thread_locals_);
        c10::WarningUtils::WarningHandlerGuard warnings_guard(
            &local_graph_task->warning_handler_);

        try {
          // The guard sets the thread_local current_graph_task on construction
          // and restores it on exit. The current_graph_task variable helps
          // queue_callback() to find the target GraphTask to append final
          // callbacks.
          GraphTaskGuard guard(local_graph_task);
          NodeGuard ndguard(task.fn_);
          {
            RECORD_FUNCTION(
                c10::str(
                    "autograd::engine::evaluate_function: ",
                    task.fn_.get()->name()),
                c10::ArrayRef<const c10::IValue>());
            evaluate_function(
                local_graph_task,
                task.fn_.get(),
                task.inputs_,
                local_graph_task->cpu_ready_queue_);
          }
        } catch (std::exception& e) {
          // See Note [ Persisting PyErr state across autograd engine threads ]
          thread_on_exception(local_graph_task, task.fn_, e);
        }
      }
    }

    // Decrement the outstanding tasks.
    --local_graph_task->outstanding_tasks_;

    // Check if we've completed execution.
    if (local_graph_task->completed()) {
      local_graph_task->mark_as_completed_and_run_post_processing();

      auto base_owner = local_graph_task->owner_;
      // The current worker thread finish the graph_task, but the owning thread
      // of the graph_task might be sleeping on pop() if it does not have work.
      // So we need to send a dummy function task to the owning thread just to
      // ensure that it's not sleeping, so that we can exit the thread_main.
      // If it has work, it might see that graph_task->outstanding_tasks_ == 0
      // before it gets to the task, but it's a no-op anyway.
      //
      // NB: This is not necessary if the current thread is the owning thread.
      if (worker_device != base_owner) {
        // Synchronize outstanding_tasks_ with queue mutex
        std::atomic_thread_fence(std::memory_order_release);
        ready_queue_by_index(local_graph_task->cpu_ready_queue_, base_owner)
            ->push(NodeTask(local_graph_task, nullptr, InputBuffer(0)));
      }
    }
  }
}

// Reentrant call will re-use the graph_task's owner thread ready_queue for
// queueing tasks (NOTE: this is not true in the async_mode of the engine).
// While we can create separate ready queue for each new reentrant
// thread, but sharing the same cpu_ready_queue with parent thread is a
// performance improvement and cuda thread still have to do the same thing.
void Engine::reentrant_thread_init() {
  c10::set_terminate_handler();
  at::init_num_threads();
  auto tp_shared = thread_pool_shared_;
  while (true) {
    std::unique_lock<std::mutex> lk(tp_shared->mutex_);
    ++thread_pool_shared_->num_workers_;
    tp_shared->work_.wait(
        lk, [&tp_shared] { return !tp_shared->graphtasks_queue_.empty(); });
    --thread_pool_shared_->num_workers_;
    auto task = tp_shared->graphtasks_queue_.front();
    tp_shared->graphtasks_queue_.pop();
    lk.unlock();
    std::shared_ptr<GraphTask> graph_task = task.lock();
    if (!graph_task) {
      LOG(INFO) << "GraphTask has expired, skipping reentrant execution";
      continue;
    }
    set_device(graph_task->owner_);
    // set the local_ready_queue to the ready queue on the graph_task->owner_
    // device
    local_ready_queue =
        ready_queue_by_index(graph_task->cpu_ready_queue_, graph_task->owner_);
    total_depth = graph_task->reentrant_depth_;
    thread_main(graph_task);
  }
}

void Engine::thread_on_exception(
    const std::shared_ptr<GraphTask>& graph_task,
    const std::shared_ptr<Node>& fn,
    std::exception& e) {
  graph_task->set_exception(std::current_exception(), fn);
}

namespace {
std::atomic<uint64_t> graph_task_id{0};
}

GraphTask::GraphTask(
    bool keep_graph,
    bool grad_mode,
    int reentrant_depth,
    std::shared_ptr<ReadyQueue> cpu_ready_queue,
    c10::SmallVector<Node*, 4> graph_roots,
    bool exit_on_error)
    : keep_graph_(keep_graph),
      graph_roots_(std::move(graph_roots)),
      owner_(NO_DEVICE),
      reentrant_depth_(reentrant_depth),
      exit_on_error_(exit_on_error),
      cpu_ready_queue_(std::move(cpu_ready_queue)),
      future_result_(c10::make_intrusive<at::ivalue::Future>(
          c10::ListType::create(c10::TensorType::get()))),
      id_(graph_task_id.fetch_add(1, std::memory_order_relaxed)) {
  thread_locals_.set_grad_mode(grad_mode);
}

bool GraphTask::completed() {
  return outstanding_tasks_.load() == 0 ||
      (exit_on_error_ && has_error_.load());
}

void GraphTask::mark_as_completed_and_run_post_processing() {
  // Allow only one thread one attempt to process this logic.
  if (future_completed_.exchange(true)) {
    // Future is already marked complete, or being marked as such.
    // In case the marking complete is only in progress, we add a
    // wait() to guarantee the future is marked complete on exit.
    future_result_->wait();
    return;
  }

  try {
    // Run post processing, before marking the future as complete.
    // Drop lock prior to completing, to avoid holding across callbacks.
    std::unique_lock<std::mutex> lock(mutex_);

    exec_post_processing();
    std::vector<Variable> vars = std::move(captured_vars_);

    // Need to unlock before we call markCompleted to avoid holding locks
    // when the callbacks are called.
    lock.unlock();
    future_result_->markCompleted(vars);
  } catch (std::exception&) {
    future_result_->setErrorIfNeeded(std::current_exception());
  }
}

void GraphTask::exec_post_processing() {
  if (!not_ready_.empty()) {
    throw std::runtime_error("could not compute gradients for some functions");
  }

  // set the thread_local current_graph_task_ as more callbacks can be installed
  // by existing final callbacks.
  GraphTaskGuard guard(shared_from_this());
  // Lock mutex during each iteration for accessing final_callbacks.size()
  // Unlocking is necessary, because the callback can register
  // more callbacks (or they can be registered from other threads
  // while it's waiting.
  std::unique_lock<std::mutex> cb_lock(final_callbacks_lock_);

  // caller_current_streams_ with nullopt entries removed
  std::vector<c10::Stream> caller_current_streams_filtered;

  // See Note [Streaming backwards].
  // Syncs caller_current_stream with leaf streams, so final_callbacks may use
  // any grad on its device's current stream.
  if (!leaf_streams.empty()) {
    for (const auto& leaf_stream : leaf_streams) {
      // stash_current_cuda/privateuse1_streams() stashed streams for all device
      // IDs that already had a CUDA/privateuse1 context before the GraphTask
      // executed. For inactive devices, it stashed a std::nullopt. I don't
      // expect GraphTask's backward pass ran leaf nodes on any new devices, so
      // the stashed streams should be enough. If leaf_stream.device_index()
      // happens to be for a new device, operator* on the std::nullopt should
      // throw an error.
      const auto& caller_current_stream =
          caller_current_streams_[leaf_stream.device_index()];

      if (caller_current_stream.has_value() &&
          caller_current_stream != leaf_stream) {
        auto event = c10::Event{leaf_stream.device_type()};
        event.record(leaf_stream);
        caller_current_stream->wait(event);
      }
    }

    caller_current_streams_filtered.reserve(caller_current_streams_.size());
    for (const auto& opt_stream : caller_current_streams_) {
      if (opt_stream.has_value()) {
        caller_current_streams_filtered.push_back(*opt_stream);
      }
    }
  }

  {
    // final_callbacks run on the per-device caller_current_streams (the ambient
    // streams surrounding the user's call to backward()). This has two
    // benefits:
    //  1. caller_current_streams have been synced with leaf_streams, so
    //  callbacks may
    //     safely access any grad.
    //  2. The callback's results can safely be used on (user-facing)
    //  caller_current_streams
    //     after backward().
    c10::MultiStreamGuard g(caller_current_streams_filtered);

    // Set the ThreadLocalState before calling the function.
    // NB: The ThreadLocalStateGuard doesn't set the grad_mode because GraphTask
    // always saves ThreadLocalState without grad_mode.
    at::ThreadLocalStateGuard tls_guard(this->thread_locals_);

    // WARNING: Don't use a range-for loop here because more callbacks may be
    // added in between callback calls, so iterators may become invalidated.
    // NOLINTNEXTLINE(modernize-loop-convert)
    for (size_t i = 0; i < final_callbacks_.size(); ++i) {
      cb_lock.unlock();
      final_callbacks_[i]();
      cb_lock.lock();
    }
  }
}

void GraphTask::set_exception_without_signal(const std::shared_ptr<Node>& fn) {
  if (!has_error_.exchange(true)) {
    if (AnomalyMode::is_enabled() && fn) {
      fn->metadata()->print_stack(fn->name());
    }
  }
}

void GraphTask::set_exception(
    std::exception_ptr eptr,
    const std::shared_ptr<Node>& fn) {
  set_exception_without_signal(fn);
  if (!future_completed_.exchange(true)) {
    future_result_->setError(std::move(eptr));
  }
}

static variable_list call_pre_hooks(Node& fn, variable_list inputs) {
  for (const auto& hook : fn.pre_hooks()) {
    inputs = (*hook)(inputs);
  }
  return inputs;
}

static variable_list call_tensor_pre_hooks(Node& fn, variable_list inputs) {
  for (const auto& hook : fn.tensor_pre_hooks()) {
    inputs = (*hook)(inputs);
  }
  for (const auto& pair : fn.retains_grad_hooks()) {
    inputs = (*pair.second)(inputs);
  }
  return inputs;
}

static variable_list call_post_hooks(
    Node& fn,
    variable_list outputs,
    const variable_list& inputs,
    const bool had_post_hooks) {
  for (const auto& hook : fn.post_hooks()) {
    if (had_post_hooks) {
      outputs = (*hook)(outputs, inputs);
    } else {
      variable_list null_inputs;
      outputs = (*hook)(outputs, null_inputs);
    }
  }
  return outputs;
}

void set_device(int device) {
  // NB: We MUST NOT construct the guard for device CPU,
  // as in some settings we compile with cuda, but
  // have lazy stubs for CUDA functionality (so actually
  // attempting to setup a guard(CPU_DEVICE) will cause an
  // error, because it will still query GetDevice).
  //
  // Don't use DeviceGuard here because its destructor may be called before the
  // device is reset. This is fine because the device is thread local.
  if (device != CPU_DEVICE) {
    for (const auto i : c10::irange(static_cast<size_t>(
             c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES))) {
      auto* impl = c10::impl::device_guard_impl_registry[i].load();
      if (impl && device < impl->deviceCount()) {
        impl->setDevice(at::Device(
            static_cast<c10::DeviceType>(i),
            static_cast<c10::DeviceIndex>(device)));
      }
    }
  }
  worker_device = device;
}

// validate_outputs has two overloads, one that accepts edge_list and one that
// accepts vector<optional<InputMetadata>>. The former is stateful (it requires
// the autograd graph to actually use) and the latter is for functional
// autograd. (where we want to be able to take an autograd graph and then
// construct a FX graph out of it without specializing on the properties of the
// gradients).
//
// We do some templating to avoid dynamic allocations in the hot path (the eager
// autograd case). Otherwise, the problem is that we are given a vector<Edge>
// and would need to materialize a vector<optional<InputMetadata>> (or some
// other vector) to pass to a common helper function. The alternative is to use
// C++20's ranges which we don't have access to yet.

// Given an Edge or optional<InputMetdata>, return the InputMetadata
template <typename T>
const static InputMetadata& get_input_metadata(const T& thing);

template <>
const InputMetadata& get_input_metadata<std::optional<InputMetadata>>(
    const std::optional<InputMetadata>& thing) {
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  return thing.value();
}

template <>
const InputMetadata& get_input_metadata<Edge>(const Edge& thing) {
  return thing.function->input_metadata(thing.input_nr);
}

// Given an Edge or optional<InputMetdata>, return if there is an InputMetadata.
template <typename T>
static bool has_input_metadata(const T& thing);

template <>
bool has_input_metadata<std::optional<InputMetadata>>(
    const std::optional<InputMetadata>& thing) {
  return thing.has_value();
}

template <>
bool has_input_metadata<Edge>(const Edge& thing) {
  return thing.is_valid();
}

std::vector<std::optional<InputMetadata>> collect_input_metadata(
    const edge_list& edges) {
  std::vector<std::optional<InputMetadata>> input_metadata;
  for (const auto& edge : edges) {
    if (!edge.is_valid()) {
      input_metadata.emplace_back(std::nullopt);
      continue;
    }
    input_metadata.emplace_back(edge.function->input_metadata(edge.input_nr));
  }
  return input_metadata;
}

// Given an vector<Edge> or vector<optional<InputMetdata>>, validate the
// outputs. This involves using the InputMetadata to check the outputs and also
// potentially calling .sum_to on the outputs.
template <typename T>
static void validate_outputs_impl(
    const std::vector<T>& input_metadata_container,
    variable_list& grads,
    const std::function<std::string(const std::string&)>& format_error) {
  if (grads.size() != input_metadata_container.size()) {
    std::stringstream ss;
    ss << "invalid number of gradients - expected ";
    ss << input_metadata_container.size() << ", but got " << grads.size();
    TORCH_CHECK(false, format_error(ss.str()));
  }
  for (const auto i : c10::irange(grads.size())) {
    if (!has_input_metadata(input_metadata_container[i])) {
      continue;
    }
    const auto& metadata = get_input_metadata(input_metadata_container[i]);
    auto& grad = grads[i];
    if (!grad.defined()) {
      // FIXME: TestJit.test_ge_optimized fails this assertion.
      // std::stringstream ss;
      // ss << "undefined gradient at index " << i;
      // TORCH_CHECK(false, format_error(ss.str()));
      continue;
    }

    grad = metadata.maybe_reduce(i, std::move(grad), format_error);

    bool input_is_complex =
        isComplexType(c10::typeMetaToScalarType(metadata.options().dtype()));
    bool grad_is_complex = isComplexType(grad.scalar_type());

    TORCH_CHECK(
        isFloatingType(grad.scalar_type()) ||
        (input_is_complex == grad_is_complex));
    if (c10::typeMetaToScalarType(metadata.options().dtype()) !=
        grad.scalar_type()) {
      grad = grad.to(c10::typeMetaToScalarType(metadata.options().dtype()));
    }
    if (grad.dtype() != metadata.dtype()) {
      std::stringstream ss;
      ss << "invalid gradient at index " << i << " - expected dtype ";
      ss << metadata.dtype() << " but got " << grad.dtype();
      TORCH_CHECK(false, format_error(ss.str()));
    }
    if (grad.layout() != metadata.layout()) {
      // TODO: Currently we only support (*, Sparse) combination for
      // (tensor.layout(), tensor.grad.layout()) In future, there will be an
      // opportunity to support more combinations of layouts if they are
      // composable (example., operations like addition etc., are well defined
      // between tensors of different layouts.), as well as all parts of
      // autograd like AccumulateGrad correctly handle this. We allow grad to be
      // Strided when metadata is SparseCsr
      if (!grad.is_sparse() &&
          !(grad.layout() == at::kStrided &&
            (at::sparse_csr::is_sparse_compressed(metadata.layout()) ||
             metadata.layout() == at::kSparse))) {
        std::stringstream ss;
        ss << "invalid gradient at index " << i << " - expected layout ";
        ss << metadata.layout() << " but got " << grad.layout();
        TORCH_CHECK(false, format_error(ss.str()));
      }
    }

    if (grad.device() != metadata.device()) {
      // quick hack for: https://github.com/pytorch/pytorch/issues/65016 but
      // should be eventually removed
      if (!(metadata.is_tensor_subclass() ||
            grad.unsafeGetTensorImpl()->is_python_dispatch())) {
        if (grad.dim() == 0) {
          grad = grad.to(metadata.device());
        } else {
          std::stringstream ss;
          ss << "invalid gradient at index " << i << " - expected device ";
          ss << metadata.device() << " but got " << grad.device();
          TORCH_CHECK(false, format_error(ss.str()));
        }
      }
    }
    // We should not build graph for Tensors that are not differentiable
    TORCH_INTERNAL_ASSERT(isDifferentiableType(grad.scalar_type()));
  }
}

void validate_outputs(
    const edge_list& edges,
    variable_list& grads,
    const std::function<std::string(const std::string&)>& format_error) {
  return validate_outputs_impl(edges, grads, format_error);
}

void validate_outputs(
    const std::vector<std::optional<InputMetadata>>& input_metadata,
    variable_list& grads,
    const std::function<std::string(const std::string&)>& format_error) {
  return validate_outputs_impl(input_metadata, grads, format_error);
}

static variable_list call_function(
    std::shared_ptr<GraphTask>& graph_task,
    Node* func,
    InputBuffer& inputBuffer) {
  CheckpointValidGuard cpvguard(graph_task);
  auto& fn = *func;
  auto inputs =
      call_tensor_pre_hooks(fn, InputBuffer::variables(std::move(inputBuffer)));
  inputs = call_pre_hooks(fn, std::move(inputs));
  if (!graph_task->keep_graph_) {
    fn.will_release_variables();
  }

  const auto has_post_hooks = !fn.post_hooks().empty();
  variable_list outputs;

  if (has_post_hooks) {
    // In functions/accumulate_grad.cpp, there is some logic to check the
    // conditions under which the incoming gradient can be stolen directly
    // (which elides a deep copy) instead of cloned. One of these conditions
    // is that the incoming gradient's refcount must be 1 (nothing else is
    // referencing the same data).  Stashing inputs_copy here bumps the
    // refcount, so if post hooks are employed, it's actually still ok for
    // accumulate_grad.cpp to steal the gradient if the refcount is 2.
    //
    // "new_grad.use_count() <= 1 + !post_hooks().empty()" in
    // accumulate_grad.cpp accounts for this, but also creates a silent
    // dependency between engine.cpp (ie, this particular engine
    // implementation) and accumulate_grad.cpp.
    //
    // If you change the logic here, make sure it's compatible with
    // accumulate_grad.cpp.
    auto inputs_copy = inputs;
    outputs = fn(std::move(inputs_copy));
  } else {
    outputs = fn(std::move(inputs));
  }

  validate_outputs(fn.next_edges(), outputs, [&](const std::string& msg) {
    std::ostringstream ss;
    ss << "Function " << fn.name() << " returned an " << msg;
    return ss.str();
  });

  // NOLINTNEXTLINE(bugprone-use-after-move)
  return call_post_hooks(fn, std::move(outputs), inputs, has_post_hooks);
}

void Engine::evaluate_function(
    std::shared_ptr<GraphTask>& graph_task,
    Node* func,
    InputBuffer& inputs,
    const std::shared_ptr<ReadyQueue>& cpu_ready_queue) {
  // The InputBuffer::adds that supplied incoming grads took pains to
  // ensure they're safe to consume in the context of the present
  // func's stream (if applicable). So we guard onto that stream
  // before working with the grads in any capacity.
  auto opt_parent_stream = (*func).stream();
  c10::OptionalStreamGuard parent_stream_guard{opt_parent_stream};

  // If exec_info_ is not empty, we have to instrument the execution
  auto& exec_info_ = graph_task->exec_info_;
  if (!exec_info_.empty()) {
    auto& fn_info = exec_info_.at(func);
    variable_list new_inputs = inputs.buffer;
    if (!fn_info.needed_) {
      // We always want to call tensor pre-hooks, but want to avoid calling it
      // twice. needed_ = True indicates that we will call tensor pre-hooks
      // later.
      //
      // See NOTE [Hooks ordering] for more context.
      new_inputs = call_tensor_pre_hooks(
          *func, InputBuffer::variables(std::move(inputs)));
    }
    if (auto* capture_vec = fn_info.captures_.get()) {
      auto opt_parent_stream = (*func).stream();
      // Lock mutex for writing to graph_task->captured_vars_.
      std::lock_guard<std::mutex> lock(graph_task->mutex_);
      for (const auto& capture : *capture_vec) {
        auto& captured_grad = graph_task->captured_vars_[capture.output_idx_];
        captured_grad = new_inputs[capture.input_idx_];
        // NOTE [Deprecated capture hooks]
        for (const auto& hook :
             capture.DO_NOT_USE_DEPRECATED_get_capture_hooks()) {
          captured_grad = (*hook)(captured_grad);
        }
        if (opt_parent_stream) {
          // No need to take graph_task->mutex_ here, we already hold it
          graph_task->leaf_streams.emplace(*opt_parent_stream);
        }
      }
    }
    if (!fn_info.needed_) {
      // Skip execution if we don't need to execute the function.
      return;
    }
  }

  auto outputs = call_function(graph_task, func, inputs);

  auto& fn = *func;
  if (!graph_task->keep_graph_) {
    fn.release_variables();
  }

  auto num_outputs = outputs.size();
  if (num_outputs == 0) { // Note: doesn't acquire the mutex
    // Records leaf stream (if applicable)
    // See Note [Streaming backwards]
    if (opt_parent_stream) {
      std::lock_guard<std::mutex> lock(graph_task->mutex_);
      graph_task->leaf_streams.emplace(*opt_parent_stream);
    }
    return;
  }

  if (AnomalyMode::is_enabled() && AnomalyMode::should_check_nan()) {
    AutoGradMode grad_mode(false);
    for (const auto i : c10::irange(num_outputs)) {
      auto& output = outputs[i];
      at::OptionalDeviceGuard guard(device_of(output));
      if (output.defined() && isnan(output)._is_any_true().item<bool>()) {
        std::stringstream ss;
        ss << "Function '" << fn.name() << "' returned nan values in its " << i
           << "th output.";
        throw std::runtime_error(ss.str());
      }
    }
  }

  // Lock mutex for the accesses to GraphTask dependencies_, not_ready_ and
  // cpu_ready_queue_ below
  std::lock_guard<std::mutex> lock(graph_task->mutex_);
  for (const auto i : c10::irange(num_outputs)) {
    auto& output = outputs[i];
    const auto& next = fn.next_edge(i);

    if (!next.is_valid())
      continue;

    // Check if the next function is ready to be computed
    bool is_ready = false;
    auto& dependencies = graph_task->dependencies_;
    auto it = dependencies.find(next.function.get());

    if (it == dependencies.end()) {
      auto name = next.function->name();
      throw std::runtime_error(std::string("dependency not found for ") + name);
    } else if (--it->second == 0) {
      dependencies.erase(it);
      is_ready = true;
    }

    auto& not_ready = graph_task->not_ready_;
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

      // Accumulates into buffer
      auto opt_next_stream = next.function->stream();
      input_buffer.add(
          next.input_nr, std::move(output), opt_parent_stream, opt_next_stream);

      if (is_ready) {
        auto queue = ready_queue(cpu_ready_queue, next.function->device());
        queue->push(
            NodeTask(graph_task, next.function, std::move(input_buffer)));
      } else {
        not_ready.emplace(next.function.get(), std::move(input_buffer));
      }
    } else {
      // The function already has a buffer
      auto& input_buffer = not_ready_it->second;

      // Accumulates into buffer
      auto opt_next_stream = next.function->stream();
      input_buffer.add(
          next.input_nr, std::move(output), opt_parent_stream, opt_next_stream);
      if (is_ready) {
        auto queue = ready_queue(cpu_ready_queue, next.function->device());
        queue->push(
            NodeTask(graph_task, next.function, std::move(input_buffer)));
        not_ready.erase(not_ready_it);
      }
    }
  }
}

static uint64_t compute_min_topological_nr(const edge_list& outputs) {
  // Computes the mininum topological number among all the outputs
  if (outputs.empty()) {
    return 0;
  }
  auto min_topo_nr = std::numeric_limits<uint64_t>::max();
  for (auto& output_edge : outputs) {
    auto topo_nr = output_edge.function->topological_nr();
    min_topo_nr = (min_topo_nr < topo_nr) ? min_topo_nr : topo_nr;
  }
  return min_topo_nr;
}

auto Engine::compute_dependencies(
    Node* root,
    GraphTask& task,
    uint64_t min_topo_nr) -> void {
  // Computes the number of dependencies for each function which requires grad
  std::vector<Node*> queue{root};
  bool will_use_accelerator = false;

  // Queue contains all nodes that will start propagating gradients.
  // We no longer have to expand functions that don't require grad.
  auto& dependencies = task.dependencies_;
  while (!queue.empty()) {
    auto fn = queue.back();
    queue.pop_back();
    if (fn->topological_nr() < min_topo_nr) {
      continue;
    }
    if (!will_use_accelerator) {
      will_use_accelerator = fn->stream().has_value();
    }
    for (const auto& edge : fn->next_edges()) {
      if (auto next_ptr = edge.function.get()) {
        dependencies[next_ptr] += 1;
        const bool was_inserted = task.nodes_in_graph_.insert(next_ptr).second;
        if (was_inserted)
          queue.push_back(next_ptr);
      }
    }
  }

  if (will_use_accelerator) {
    // Collects current streams for devices where this process has a
    // context, so GraphTask::exec_post_processing can sync them with
    // leaf_streams.
    task.stash_current_streams();
  }
}

auto Engine::execute(
    const edge_list& root_edges,
    const variable_list& inputs,
    bool keep_graph,
    bool create_graph,
    bool accumulate_grad,
    const edge_list& outputs) -> variable_list {
  validate_outputs(
      root_edges,
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      const_cast<variable_list&>(inputs),
      [](const std::string& msg) { return msg; });
  if (accumulate_grad && create_graph) {
    TORCH_WARN_ONCE(
        "Using backward() with create_graph=True will create a reference cycle "
        "between the parameter and its gradient which can cause a memory leak. "
        "We recommend using autograd.grad when creating the graph to avoid this. "
        "If you have to use this function, make sure to reset the .grad fields of "
        "your parameters to None after use to break the cycle and avoid the leak.");
  }

  // Allows us to assert no other threads are in backwards
  CompiledAutogradThreadingDebugCheck _thread_check;
  auto compiled_autograd = the_compiled_autograd.load();
  TORCH_INTERNAL_ASSERT(compiled_autograd != COMPILED_AUTOGRAD_POISON);

  // accumulate_grad is true if and only if the frontend call was to
  // backward(), not grad(). grad() returns the sum of the gradients
  // w.r.t. the inputs and thus needs the inputs to be present.
  TORCH_CHECK_VALUE(
      accumulate_grad || !outputs.empty(), "grad requires non-empty inputs.");

  // A fresh first time Engine::execute call should start on the CPU device,
  // initialize a new thread local ready queue on CPU or reuse the existing one
  // (if there is one allocated already, i.e. consecutive backward calls,
  // re-entrant backward calls), then memoize the local_ready_queue in GraphTask
  init_local_ready_queue();
  bool not_reentrant_backward_call = worker_device == NO_DEVICE;

  // Store root nodes so we can traverse through the graph later
  // e.g., for get_current_graph_task_execution_order
  c10::SmallVector<Node*, 4> temp_roots{root_edges.size()};
  for (const auto i : c10::irange(root_edges.size())) {
    temp_roots[i] = root_edges[i].function.get();
  }

  auto graph_task = std::make_shared<GraphTask>(
      /* keep_graph */ keep_graph,
      /* create_graph */ create_graph,
      /* depth */ not_reentrant_backward_call ? 0 : total_depth + 1,
      /* cpu_ready_queue */ local_ready_queue,
      /* graph_roots */ std::move(temp_roots));

  // If we receive a single root, skip creating extra root node
  bool skip_dummy_node = root_edges.size() == 1 && compiled_autograd == nullptr;
  auto graph_root = skip_dummy_node
      ? root_edges.at(0).function
      : std::make_shared<GraphRoot>(root_edges, inputs);

  auto min_topo_nr = compute_min_topological_nr(outputs);
  // Now compute the dependencies for all executable functions
  compute_dependencies(graph_root.get(), *graph_task, min_topo_nr);

  if (!outputs.empty()) {
    graph_task->init_to_execute(
        *graph_root, outputs, accumulate_grad, min_topo_nr);
  }

  if (compiled_autograd != nullptr) {
    // see [Note: Compiled Autograd]
    TORCH_CHECK(
        !create_graph, "compiled_autograd does not support create_graph");
    _thread_check.release();
    GraphTaskGuard guard(graph_task);
    CheckpointValidGuard cpvguard(graph_task);
    return (*compiled_autograd)(
        graph_root, *graph_task, accumulate_grad, outputs);
  }

  // Queue the root
  if (skip_dummy_node) {
    InputBuffer input_buffer(root_edges.at(0).function->num_inputs());
    auto input = inputs.at(0);

    const auto input_stream = InputMetadata(input).stream();
    auto opt_next_stream = root_edges.at(0).function->stream();
    input_buffer.add(
        root_edges.at(0).input_nr,
        std::move(input),
        input_stream,
        opt_next_stream);

    execute_with_graph_task(
        graph_task, std::move(graph_root), std::move(input_buffer));
  } else {
    execute_with_graph_task(
        graph_task, std::move(graph_root), InputBuffer(variable_list()));
  }
  // Avoid a refcount bump for the Future, since we check for refcount in
  // DistEngine (see TORCH_INTERNAL_ASSERT(futureGrads.use_count() == 1)
  // in dist_engine.cpp).
  auto& fut = graph_task->future_result_;
  fut->wait();
  graph_task->warning_handler_.replay_warnings();
  return fut->value().toTensorVector();
}

void Engine::initialize_device_threads_pool() {
  TORCH_CHECK(
      !in_bad_autograd_fork,
      "Unable to handle autograd's threading in combination with fork-based multiprocessing. "
      "See https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork");
  // Ensures device_ready_queues_ are initialized only once
  static bool start_device_threads_flag_ [[maybe_unused]] = [this]() {
    this->start_device_threads();
    return true;
  }();
}

c10::intrusive_ptr<at::ivalue::Future> Engine::execute_with_graph_task(
    const std::shared_ptr<GraphTask>& graph_task,
    std::shared_ptr<Node> graph_root,
    InputBuffer&& input_buffer) {
  initialize_device_threads_pool();
  // Lock mutex for GraphTask.
  std::unique_lock<std::mutex> lock(graph_task->mutex_);

  auto queue = ready_queue(graph_task->cpu_ready_queue_, graph_root->device());

  // worker_device == NO_DEVICE it's a CPU thread and it's trying to drive the
  // autograd engine with corresponding GraphTask, and its NOT a re-entrant call
  if (worker_device == NO_DEVICE) {
    // We set the worker_device to CPU_DEVICE only if worker_device was
    // previously NO_DEVICE. Setting it to CPU afterwards allow us to detect
    // whether this is a re-entrant call or not.
    set_device(CPU_DEVICE);

    // set the graph_task owner to the current device
    graph_task->owner_ = worker_device;

    // Now that all the non-thread safe fields of the graph_task have been
    // populated, we can enqueue it.
    queue->push(
        NodeTask(graph_task, std::move(graph_root), std::move(input_buffer)));

    // The owning thread start to drive the engine execution for any CPU task
    // that was just pushed or will be added later from other worker threads
    lock.unlock();
    thread_main(graph_task);
    TORCH_INTERNAL_ASSERT(graph_task->future_result_->completed());
    // reset the worker_device after the completion of the graph_task, this is
    // so that the initial state of the engine remains the same across every
    // backward() or grad() call, we don't need to reset local_ready_queue as we
    // could possibly reuse it for new backward calls.
    worker_device = NO_DEVICE;
  } else {
    // If worker_device is any devices (i.e. CPU, CUDA): this is a re-entrant
    //    backward call from that device.
    graph_task->owner_ = worker_device;

    // Now that all the non-thread safe fields of the graph_task have been
    // populated, we can enqueue it.
    queue->push(
        NodeTask(graph_task, std::move(graph_root), std::move(input_buffer)));

    if (current_depth >= max_recursion_depth_) {
      // See Note [Reentrant backwards]
      // If reached the max depth, switch to a different thread
      add_thread_pool_task(graph_task);
    } else {
      // Total depth needs to be updated only in this codepath, since it is
      // not used in the block above (when we call add_thread_pool_task).
      // In the codepath above, GraphTask.reentrant_depth_ is used to
      // bootstrap total_depth in the other thread.
      ++total_depth;

      // Get back to work while we wait for our new graph_task to
      // complete!
      ++current_depth;
      lock.unlock();
      thread_main(graph_task);
      --current_depth;
      --total_depth;

      // The graph task should have completed and the associated future should
      // be marked completed as well since 'thread_main' above is a call
      // blocking an autograd engine thread.
      TORCH_INTERNAL_ASSERT(graph_task->future_result_->completed());
    }
  }
  // graph_task_exec_post_processing is done when the Future is marked as
  // completed in mark_as_completed_and_run_post_processing.
  return graph_task->future_result_;
}

// note that when python is present, this base engine will be overriden
// with a PythonEngine. Because this typically happens before get_default_engine
// is called, this base engine will never be created.
Engine& Engine::get_base_engine() {
  static Engine engine;
  return engine;
}

static std::atomic<EngineStub> engine_stub(Engine::get_base_engine);

void set_default_engine_stub(EngineStub stub) {
  engine_stub.store(stub);
}

Engine& Engine::get_default_engine() {
  return engine_stub.load()();
}

void Engine::set_compiled_autograd(Engine::compiled_autograd_fn fn) {
  if (the_compiled_autograd.load() == fn) {
    return;
  }
  auto prior = the_compiled_autograd.exchange(COMPILED_AUTOGRAD_POISON);
  TORCH_CHECK(
      num_threads_in_backwards.load() == 0 && prior != COMPILED_AUTOGRAD_POISON,
      "compiled_autograd._enable() requires no threads in backwards()");
  the_compiled_autograd.store(fn);
}

void Engine::queue_callback(std::function<void()> callback) {
  TORCH_CHECK(
      current_graph_task,
      "Final callbacks can only be installed during backward pass.");

  std::lock_guard<std::mutex> lock(current_graph_task->final_callbacks_lock_);
  current_graph_task->final_callbacks_.emplace_back(std::move(callback));
}

bool Engine::is_checkpoint_valid() {
  return checkpoint_valid;
}

void Engine::init_local_ready_queue(std::shared_ptr<ReadyQueue> ready_queue) {
  if (ready_queue) {
    // if ready_queue provided in the caller, use the caller's ready_queue to
    // initialize local_ready_queue
    local_ready_queue = std::move(ready_queue);
  } else if (!local_ready_queue) {
    // otherwise if local_ready_queue not allocated, allocate a new ready_queue
    local_ready_queue = std::make_shared<ReadyQueue>();
  }
}

// CPU ready queue is per GraphTask, but CUDA device ready queues are shared
// across all graph tasks
auto Engine::ready_queue(
    std::shared_ptr<ReadyQueue> cpu_ready_queue,
    at::Device device) -> std::shared_ptr<ReadyQueue> {
  bool multithreading_disabled =
      !c10::AutogradState::get_tls_state().get_multithreading_enabled();
  if (multithreading_disabled || should_run_in_cpu_ready_queue(device.type())) {
    // return the cpu ready queue passed in
    TORCH_INTERNAL_ASSERT(cpu_ready_queue);
    return cpu_ready_queue;
  } else {
    TORCH_INTERNAL_ASSERT(
        0 <= device.index() &&
        device.index() <
            static_cast<c10::DeviceIndex>(device_ready_queues_.size()));
    // See Note [Allocating GPUs to autograd threads]
    return device_ready_queues_.at(device.index());
  }
}

auto Engine::ready_queue_by_index(
    std::shared_ptr<ReadyQueue> cpu_ready_queue,
    int device_index) -> std::shared_ptr<ReadyQueue> {
  if (device_index == CPU_DEVICE) {
    // return the cpu ready queue passed in
    TORCH_INTERNAL_ASSERT(cpu_ready_queue);
    return cpu_ready_queue;
  } else {
    TORCH_INTERNAL_ASSERT(
        0 <= device_index &&
        device_index <
            static_cast<c10::DeviceIndex>(device_ready_queues_.size()));
    // See Note [Allocating GPUs to autograd threads]
    // NB: This function would become obsolete if we truly allocated a CPU
    // thread per device, rather than colocate.
    return device_ready_queues_.at(device_index);
  }
}

auto Engine::start_device_threads() -> void {
  // First always initialize the thread pool for re-entrant threads
  thread_pool_shared_ = std::make_shared<ThreadPoolShared>();

  // Second, create special threads for each non-CPU device
  // See Note [Allocating GPUs to autograd threads]
  c10::DeviceIndex num_devices = 0;
  for (const auto& impl_atomic : c10::impl::device_guard_impl_registry) {
    auto* impl = impl_atomic.load();
    // Only record the number of devices for device that don't run on the
    // cpu ready queue.
    if (impl && !should_run_in_cpu_ready_queue(impl->type())) {
      num_devices = std::max(num_devices, impl->deviceCount());
    }
  }

  // If there are no device except cpu, no need to create worker threads
  if (num_devices == 0) {
    return;
  }

  // Since we're about to create threads, forking is not possible anymore
  track_bad_autograd_forks();

  // allocate one thread for every GPU device (but colocate GPUs of different
  // types), and pre-allocate the device_ready_queues_ to ensure safe reading on
  // it.
  device_ready_queues_ = std::vector<std::shared_ptr<ReadyQueue>>(num_devices);
  for (auto& queue : device_ready_queues_) {
    queue = std::make_shared<ReadyQueue>();
  }

  for (const auto i : c10::irange(num_devices)) {
    std::thread t(&Engine::thread_init, this, i, device_ready_queues_[i], true);
    t.detach();
  }
  // Wait for the threads to start
  {
    std::unique_lock<std::mutex> lk(non_reentrant_device_thread_mutex_);
    while (non_reentrant_device_thread_count_.load() !=
           static_cast<uint32_t>(num_devices)) {
      non_reentrant_device_thread_condvar_.wait(lk);
    }
  }
}

void Engine::add_thread_pool_task(const std::weak_ptr<GraphTask>& graph_task) {
  std::unique_lock<std::mutex> lck(thread_pool_shared_->mutex_);
  // There may already be some items on the graphtasks_queue_ added by other
  // threads but not enough workers to get to the new task that will be
  // added
  bool create_thread =
      (thread_pool_shared_->num_workers_ <=
       thread_pool_shared_->graphtasks_queue_.size());
  thread_pool_shared_->graphtasks_queue_.push(graph_task);
  // Don't need to be holding the lock while actually creating the thread
  lck.unlock();
  if (create_thread) {
    // If we're creating a new thread, forking is not allowed anymore
    track_bad_autograd_forks();
    std::thread t(&Engine::reentrant_thread_init, this);
    t.detach();
  }
  // This works even if new thread is created because wait() will test the
  // predicate before waiting
  thread_pool_shared_->work_.notify_one();
}

// Remembers current streams on all devices where a context has been created for
// This function assumes the accelerator device is available.
void GraphTask::stash_current_streams() {
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  const auto accelerator = at::getAccelerator(true).value();
  const auto guard = c10::impl::VirtualGuardImpl{accelerator};
  auto num_devices = guard.deviceCount();
  caller_current_streams_.resize(num_devices);
  if (num_devices > 0) {
    for (c10::DeviceIndex idx = 0; idx < num_devices; idx++) {
      if (at::globalContext().getAcceleratorHooksInterface().hasPrimaryContext(
              idx)) {
        caller_current_streams_[idx] = guard.getStream({accelerator, idx});
      } else {
        caller_current_streams_[idx] = std::nullopt;
      }
    }
  }
}

void GraphTask::init_to_execute(
    Node& graph_root,
    const edge_list& outputs,
    bool accumulate_grad,
    uint64_t min_topo_nr) {
  // Populates exec_info so nodes that should be executed have
  // `exec_info[node].needed_ = true` Only nodes that have a path to any edge in
  // `outputs` should be executed. The code below populates exec_info using
  // recursion, but the actual code does this iteratively. Refer to the
  // numbering to see how the actual code corresponds. A difference to note is
  // that in the iterative version, when you are working with the current Node,
  // you are responsible to update your parent's is_needed after all your
  // children have been updated.
  //
  // is_needed = {fn: True for fn in outputs}             # (0)
  // seen = {}
  // def compute_is_needed(fn):
  //   for next_edge in fn.next_edges:
  //     child_fn = next_edge.fn
  //     if child_fn in seen and is_needed[child_fn]:     # (1)
  //       is_needed[fn] = true
  //     else:
  //       seen.add(child_fn)
  //       if compute_is_needed(child_fn):
  //         is_needed[fn] = true                         # (2)
  //                                                      # (3) exit for-loop
  //   return is_needed[fn]
  // compute_is_needed(graph_root)
  //
  // NB: you might be wondering why we don't populate `seen` with outputs. We
  // cannot because in the case where two outputs lie on the same path, we still
  // need to explore past the first output or we would miss the nodes that are
  // required to compute the second output.
  int output_idx = 0;
  for (auto& output_edge : outputs) {
    // (0) `is_needed` above corresponds to `exec_info_[fn].needed_`
    Node* output = output_edge.function.get();
    auto& info = exec_info_[output];
    if (accumulate_grad) {
      // if called through `.backward()` we directly set `needed_` for all the
      // outputs to true
      info.needed_ = true;
    } else {
      // otherwise it is `.grad()` and we set exec_info[fn].captures_ instead
      // In terms of populating the rest of exec_info though, you can basically
      // think of this as the same as setting `needed_` is true directly.
      if (!info.captures_) {
        info.captures_ = std::make_unique<std::vector<ExecInfo::Capture>>();
      }
      info.captures_->emplace_back(output_edge.input_nr, output_idx++);
    }
  }
  captured_vars_.resize(output_idx);

  struct Frame {
    Frame(Node* fn) : fn_(fn) {}
    Node* fn_{};
    size_t next_next_fn_{};

    Node* get_next_fn() {
      const auto& next = fn_->next_edges();
      auto num_next = next.size();
      while (next_next_fn_ < num_next) {
        auto fn = next[next_next_fn_++].function.get();
        if (fn)
          return fn;
      }
      return nullptr;
    }
  };

  auto nodeShouldExecute = [this](Node* fn) {
    auto it = exec_info_.find(fn);
    return it != exec_info_.end() && it->second.should_execute();
  };

  std::vector<Frame> stack;
  std::unordered_set<Node*> seen;
  stack.emplace_back(&graph_root);
  exec_info_.emplace(stack.back().fn_, ExecInfo());

  while (!stack.empty()) {
    auto& frame = stack.back();
    const auto fn = frame.fn_;

    Node* child_fn = nullptr;
    while ((child_fn = frame.get_next_fn()) && !seen.emplace(child_fn).second) {
      // (1) next child exists AND has already been seen
      if (nodeShouldExecute(child_fn)) {
        exec_info_[fn].needed_ = true;
      }
    }

    if (child_fn) {
      // (2) next child exists but has not been seen
      if (child_fn->topological_nr() < min_topo_nr) {
        // child created before the first output means this child cannot have
        // an edge to output
        continue;
      }
      stack.emplace_back(child_fn);
    } else {
      // (3) no next child exists for `fn` means its `needed` has already been
      // finalized. pop stack and update parent
      stack.pop_back();
      if (nodeShouldExecute(fn) && !stack.empty()) {
        exec_info_[stack.back().fn_].needed_ = true;
      }
    }
  }
}

} // namespace torch::autograd
