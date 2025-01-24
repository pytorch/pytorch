#pragma once

// Engine implements backpropagation from output variables and their gradients
// to "root" variables (variables created by the user with requires_grad=True).

#include <ATen/Tensor.h>
#include <ATen/ThreadLocalState.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/autograd/anomaly_mode.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/graph_task.h>
#include <torch/csrc/autograd/input_buffer.h>
#include <torch/csrc/autograd/saved_variable_hooks.h>
#include <torch/csrc/autograd/utils/warnings.h>

#include <c10/util/CallOnce.h>

#include <exception>
#include <functional>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

namespace torch::autograd {
struct ReadyQueue;
}

namespace torch::autograd {

// Maximum reentrant backward depth before switching to a new thread
// This limit is based on the TSAN's deadlock detector, where it will
// fail if a program hold more than 65 locks in one thread at once.
// As we hold mutex in every of our custom C++ autograd Node, we would
// like to avoid TSAN complains on this when doing reentrant backwards
// For reference, see https://github.com/google/sanitizers/issues/950
static constexpr int MAX_DEPTH = 60;

void set_device(int device);
TORCH_API void validate_outputs(
    const edge_list& edges,
    variable_list& grads,
    const std::function<std::string(const std::string&)>& format_error);
TORCH_API void validate_outputs(
    const std::vector<std::optional<InputMetadata>>& input_metadata,
    variable_list& grads,
    const std::function<std::string(const std::string&)>& format_error);

struct NodeTask {
  std::weak_ptr<GraphTask> base_;
  std::shared_ptr<Node> fn_;
  // This buffer serves as an implicit "addition" node for all of the
  // gradients flowing here.  Once all the dependencies are finished, we
  // use the contents of this buffer to run the function.
  InputBuffer inputs_;
  // When worker receives a task with isShutdownTask = true, it will immediately
  // exit. The engine sends a shutdown task to every queue upon its destruction.
  bool isShutdownTask_;

  int getReentrantDepth() const;

  NodeTask(
      std::weak_ptr<GraphTask> base,
      std::shared_ptr<Node> fn,
      InputBuffer inputs,
      bool isShutdownTask = false)
      : base_(std::move(base)),
        fn_(std::move(fn)),
        inputs_(std::move(inputs)),
        isShutdownTask_(isShutdownTask) {}
};

// Guard that sets and restores checkpoint_valid
class CheckpointValidGuard {
 public:
  explicit CheckpointValidGuard(
      const std::shared_ptr<const GraphTask>& graph_task);
  ~CheckpointValidGuard();

 private:
  bool prev_checkpoint_valid_state;
};

struct ReadyQueue {
 private:
  // Returns true when t2 should be (weakly) BEFORE t1 in the queue.
  // Shutdown tasks are first and then empty NodeTask are next.
  struct CompareNodeTaskTime {
    bool operator()(NodeTask const& t1, NodeTask const& t2) {
      // NOLINTNEXTLINE(bugprone-branch-clone)
      if (t2.isShutdownTask_) {
        return true;
      } else if (!t1.fn_ || t1.isShutdownTask_) {
        return false;
      } else if (!t2.fn_) {
        return true;
      } else if (t1.getReentrantDepth() == t2.getReentrantDepth()) {
        return t1.fn_->sequence_nr() < t2.fn_->sequence_nr();
      } else {
        return t1.getReentrantDepth() < t2.getReentrantDepth();
      }
    }
  };

  // To notify threads waiting on the ReadyQueue of available tasks on the heap_
  std::condition_variable not_empty_;
  // To protect read and writes to heap_
  mutable std::mutex mutex_;

  std::priority_queue<NodeTask, std::vector<NodeTask>, CompareNodeTaskTime>
      heap_;

 public:
  // incrementOutstandingTasks indicates whether or not we should increment
  // 'outstanding_tasks_' for the associated GraphTask. This should mostly
  // always be true and is only set false in certain cases (see docs for
  // DistEngine.execute_graph_task_until_ready_queue_empty)
  void push(NodeTask item, bool incrementOutstandingTasks = true);
  void pushShutdownTask();
  NodeTask pop();
  bool empty() const;
  size_t size() const;
};

// A single instance of this struct should be created through the whole process
// lifetime. The worker thread creation logic and Engine's destructor rely on
// this.
struct TORCH_API Engine {
  /// Returns a reference to a static `Engine` instance.
  static Engine& get_default_engine();

  static Engine& get_base_engine();

  // compiled_autograd needs to live in a different .so file so that it
  // can have python symbols, so we add a layer of indirection
  // see [Note: Compiled Autograd]
  typedef variable_list (*compiled_autograd_fn)(
      const std::shared_ptr<Node>& graph_root,
      GraphTask& graph_task,
      bool accumulate_grad,
      const edge_list& outputs);
  static void set_compiled_autograd(compiled_autograd_fn fn);

  Engine(const Engine&) = delete;
  Engine(Engine&&) = delete;
  virtual ~Engine();

  // Given a list of (Node, input number) pairs computes the value of the graph
  // by following next_edge references.
  virtual variable_list execute(
      const edge_list& roots,
      const variable_list& inputs,
      bool keep_graph,
      bool create_graph,
      bool accumulate_grad,
      const edge_list& outputs = {});

  // Given a pre-populated GraphTask and GraphRoot, computes the backward pass
  // for the graph.
  //
  // NB: This API should only be used by internal autograd specific
  // machinery and shouldn't be exposed to users in anyway.
  virtual c10::intrusive_ptr<at::ivalue::Future> execute_with_graph_task(
      const std::shared_ptr<GraphTask>& graph_task,
      std::shared_ptr<Node> graph_root,
      InputBuffer&& input_buffer);

  virtual std::unique_ptr<AnomalyMetadata> make_anomaly_metadata() {
    return std::make_unique<AnomalyMetadata>();
  }

  virtual std::unique_ptr<SavedVariableHooks> get_default_saved_variable_hooks() {
    return nullptr;
  }

  // We pass cpu_ready_queue to evaluate_function, so that it knows
  // the correct ready queue to push to after a NodeTask is ready
  void evaluate_function(
      std::shared_ptr<GraphTask>& graph_task,
      Node* func,
      InputBuffer& inputs,
      const std::shared_ptr<ReadyQueue>& cpu_ready_queue);

  void initialize_device_threads_pool();
  virtual void thread_on_exception(
      const std::shared_ptr<GraphTask>& graph_task,
      const std::shared_ptr<Node>& fn,
      std::exception& e);

  void queue_callback(std::function<void()> callback);

  bool is_checkpoint_valid();

  // Should be called after fork to notify that worker threads are gone
  void release_workers();

  // Must be called by subclass before destructing to avoid a data-race-on-vptr.
  void stop();

  // Initializes a device thread for the autograd engine.
  virtual void thread_init(
      int device,
      const std::shared_ptr<ReadyQueue>& ready_queue,
      bool should_increment = true);

 protected:
  Engine();
  void compute_dependencies(Node* root, GraphTask& task, uint64_t min_topo_nr);

  // initialize the thread local ready queue with the ready queue that is
  // created elsewhere (i.e. thread_init, Engine::execute, etc), or create a new
  // ready queue if ready_queue is not provided.
  void init_local_ready_queue(
      std::shared_ptr<ReadyQueue> ready_queue = nullptr);

  std::shared_ptr<ReadyQueue> ready_queue(
      std::shared_ptr<ReadyQueue> cpu_ready_queue,
      at::Device device);
  std::shared_ptr<ReadyQueue> ready_queue_by_index(
      std::shared_ptr<ReadyQueue> cpu_ready_queue,
      int device_index);
  // start device threads (CUDA, XLA, etc.) in Engine,
  // note that it does NOT start CPU thread.
  void start_device_threads();
  void increment_non_reentrant_thread_count();
  void decrement_non_reentrant_thread_count();
  virtual void thread_main(const std::shared_ptr<GraphTask>& task);
  void reentrant_thread_init();
  void add_thread_pool_task(const std::weak_ptr<GraphTask>& graph_task);

  // Safe to read device_ready_queues_ without synchronization after
  // initialization
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::vector<std::shared_ptr<ReadyQueue>> device_ready_queues_;

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::vector<std::function<void()>> final_callbacks_;
  // To protect reads and writes to final_callbacks_
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::mutex post_callbacks_lock_;

  // How many nested reentrant calls are allowed until a new thread is used
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  int max_recursion_depth_;

  struct ThreadPoolShared {
    // Data structures used by the threads for executing reentrant backwards
    // tasks. See Note [Reentrant backwards]
    // Number of available threads for processing new GraphTasks.
    unsigned int num_workers_{0};
    // The threads will wait on work_ to be notified of GraphTasks
    std::condition_variable work_;
    // To protect reads and writes to graphtask_queue_ and num_workers_
    // and for synchronizing creating new threads when needed
    std::mutex mutex_;
    // Workers will process the GraphTasks added to this queue. A GraphTask is
    // allocated inside Engine::execute and lives for the duration of execute
    std::queue<std::weak_ptr<GraphTask>> graphtasks_queue_;

    ThreadPoolShared() = default;
  };

  // Temporary workaround until shutting down threads is done
  // We need shared ownership of all these objects because the threads are
  // leaked when Engine shuts down, so there may be threads waiting on work_ for
  // the graphtasks_queue_ to be nonempty.
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::shared_ptr<ThreadPoolShared> thread_pool_shared_;

 private:
  // Number of non-reentrant threads
  std::atomic<uint32_t> non_reentrant_device_thread_count_;
  // Destructor will wait for non-reentrant threads to finish
  std::condition_variable non_reentrant_device_thread_condvar_;
  std::mutex non_reentrant_device_thread_mutex_;
  // stop() must be called before the destruction path goes down to the base
  // class, in order to avoid a data-race-on-vptr. Use this boolean to guard
  // whether stop() has already been called, so we can call this in every
  // destructor of the class hierarchy.
  bool stopped_{false};
};

// allow python_engine to override the default engine when it loads
using EngineStub = Engine& (*)();
TORCH_API void set_default_engine_stub(EngineStub stub);

} // namespace torch::autograd
