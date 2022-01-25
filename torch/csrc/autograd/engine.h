#pragma once

// Engine implements backpropagation from output variables and their gradients
// to "root" variables (variables created by the user with requires_grad=True).

#include <ATen/Tensor.h>
#include <ATen/core/ivalue.h>
#include <ATen/ThreadLocalState.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/autograd/anomaly_mode.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/input_buffer.h>
#include <torch/csrc/autograd/saved_variable_hooks.h>
#include <torch/csrc/autograd/utils/warnings.h>

#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>
#include <thread>

namespace torch { namespace autograd {
struct ReadyQueue;
}} // namespace torch::autograd

namespace torch { namespace autograd {

static constexpr int NO_DEVICE = -2;
static constexpr int CPU_DEVICE = -1;

// Maximum reentrant backward depth before switching to a new thread
// This limit is based on the TSAN's deadlock detector, where it will
// fail if a program hold more than 65 locks in one thread at once.
// As we hold mutex in every of our custom C++ autograd Node, we would
// like to avoid TSAN complains on this when doing reentrant backwards
// For reference, see https://github.com/google/sanitizers/issues/950
static constexpr int MAX_DEPTH = 60;

void set_device(int device);
void validate_outputs(
    const edge_list& edges,
    variable_list& grads,
    const std::function<std::string(const std::string&)>& format_error);

// GraphTask holds metadata needed for a single execution of backward()
struct GraphTask: std::enable_shared_from_this<GraphTask> {
  std::atomic<uint64_t> outstanding_tasks_{0};
  // Indicates if an error occurred while executing any task.  When this is
  // true, it signals all threads to stop executing.
  std::atomic_bool has_error_{false};
  std::atomic_bool future_completed_{false};
  // It is safe to read keep_graph_ without synchronization
  bool keep_graph_;

  // To protect reads/writes to not_ready_, dependencies_, captured_vars_,
  // has_error_, future_result_, cpu_ready_queue_, and leaf_streams.
  std::mutex mutex_;
  std::unordered_map<Node*, InputBuffer> not_ready_;
  std::unordered_map<Node*, int> dependencies_;

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  struct ExecInfo {
    struct Capture {
      Capture(const Capture&) = delete;
      Capture(Capture&&) = default;

      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
      Capture(int input_idx, int output_idx)
          : input_idx_(input_idx), output_idx_(output_idx) {}
      int input_idx_; // within Node inputs
      int output_idx_; // within the output vector of a GraphTask

      // This hook will be executed after a grad is captured. The captured
      // grad will be replaced by the return value of the hook.
      struct GradCaptureHook {
        virtual ~GradCaptureHook() = default;
        virtual at::Tensor operator()(const at::Tensor& grad) = 0;
      };
      // The hooks will be called one by one in the order as they were added.
      // The input grad of a hook will be the output of its preceding hook. The
      // first hook will take the captured grad as the input. The output of the
      // last hook will replace the captured grad.
      std::vector<std::unique_ptr<GradCaptureHook>> hooks_;
    };

    bool should_execute() const {
      return needed_ || captures_;
    }

    bool needed_ = false;
    std::unique_ptr<std::vector<Capture>> captures_;
  };
  // Exec info has a bit complicated semantics. If it's empty, it means the task
  // is run in a "default" mode, which means that all next_edges we encounter
  // should get executed. If it's not empty, only functions that have an entry
  // and this entry has needed == True should be executed. exec_info is only empty
  // when the graph is executed via .backward() and the inputs parameter is not passed.
  // Otherwise, when executed through .grad(), or when inputs arg is specified for
  // .backward(), exec_info will be non-empty.
  //
  // exec_info_ is safe to read without synchronization
  std::unordered_map<Node*, ExecInfo> exec_info_;
  // Captures variables are grads captured that we return to the user. After
  // execution of the GraphTask is completed, the captured_vars_ are moved
  // out of the GraphTask and are no longer valid.
  std::vector<Variable> captured_vars_;

  // Note: this field is not ready to be used until the proper `thread_locals_.set_grad_mode()`
  // call in the constructor.
  at::ThreadLocalState thread_locals_ = at::ThreadLocalState();

  std::unordered_set<c10::Stream> leaf_streams;

  // Per-device current streams of the execute() that called this GraphTask.
  // These will be synced with leaf_streams in exec_post_processing.
  std::vector<c10::optional<c10::Stream>> caller_current_streams_;

  // Collects caller_current_streams_
  void stash_current_streams();

  void init_to_execute(Node& graph_root, const edge_list& outputs, bool accumulate_grad, uint64_t min_topo_nr);

  // The value of worker_device in the thread that created this task.
  // See Note [Reentrant backwards]
  // Safe to read owner_ and reentrant_depth_ without synchronizaton
  int owner_;
  // The number of parent graph tasks for this graph task
  const int reentrant_depth_;

  bool can_checkpoint() const {
    return exec_info_.empty();
  }

  // check if the GraphTask is completed or not
  bool completed();
  // mark the graph task as completed and trigger post processing
  void mark_as_completed_and_run_post_processing();

  // Set an appropriate exception on this graph_task which was encountered while
  // running the provided function.
  void set_exception(std::exception_ptr eptr, const std::shared_ptr<Node>& fn);

  // Set an appropriate exception on this graph_task which was encountered while
  // running the provided function. But doesn't signal completion on
  // 'future_result_' right away. The user needs to explicitly mark
  // 'future_result_' completed with an appropriate exception.
  void set_exception_without_signal(const std::shared_ptr<Node>& fn);

  // Whether or not to stop execution for this GraphTask when an error is
  // encountered. When set to true, this would cause Engine::execute() to throw
  // an exception as soon as the autograd engine receives an exception.
  bool exit_on_error_;

  // CPU threads are dedicated to processing CPU work for the backward they invoked.
  // So any given graph task maintains its own cpu_ready_queue_ where you should send
  // work for it to be done. We memoize the cpu_ready_queue_ per GraphTask so that
  // we know which ready queue we should push to if we are on device thread (i.e. GPU)
  // and but next NodeTask should be run on CPU.
  std::shared_ptr<ReadyQueue> cpu_ready_queue_;

  // Future representing the completion of the graph task. Notified when all
  // tasks are done.
  c10::intrusive_ptr<at::ivalue::Future> future_result_;

  // Final callbacks installed during execution of this GraphTask
  std::vector<std::function<void()>> final_callbacks_;
  // To protect reads and writes to final_callbacks_. Intentionally no reusing
  // mutex_ as the two are protecting different data structures.
  std::mutex final_callbacks_lock_;

  utils::DelayWarningHandler warning_handler_;

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  GraphTask(
      bool keep_graph,
      bool grad_mode,
      int reentrant_depth,
      std::shared_ptr<ReadyQueue> cpu_ready_queue,
      bool exit_on_error = false)
      : keep_graph_(keep_graph),
        owner_(NO_DEVICE),
        reentrant_depth_(reentrant_depth),
        exit_on_error_(exit_on_error),
        cpu_ready_queue_(std::move(cpu_ready_queue)),
        future_result_(c10::make_intrusive<at::ivalue::Future>(c10::ListType::create(c10::TensorType::get()))) {
    thread_locals_.set_grad_mode(grad_mode);
        }
 private:
  // run GraphTask post processing
  void exec_post_processing();
};

// The guard that sets and restores current_graph_task.
class GraphTaskGuard {
 public:
  explicit GraphTaskGuard(std::shared_ptr<GraphTask> graph_task);
  ~GraphTaskGuard();

  void restore_current_graph_task();

 private:
  std::shared_ptr<GraphTask> last_graph_task_;
};

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
      // NOLINTNEXTLINE(modernize-pass-by-value)
      std::weak_ptr<GraphTask> base,
      std::shared_ptr<Node> fn,
      InputBuffer inputs,
      bool isShutdownTask = false)
      : base_(base),
        fn_(std::move(fn)),
        inputs_(std::move(inputs)),
        isShutdownTask_(isShutdownTask) {}
};

// Guard that sets and restores checkpoint_valid
class CheckpointValidGuard {
 public:
  explicit CheckpointValidGuard(const std::shared_ptr<const GraphTask>& graph_task);
  ~CheckpointValidGuard();
 private:
  bool prev_checkpoint_valid_state;
};


struct ReadyQueue {
 private:
  // Returns true when t2 should be (weakly) BEFORE t1 in the queue.
  // Shutdown tasks are first and then empty NodeTask are next.
  struct CompareNodeTaskTime {
    bool operator()(NodeTask const & t1, NodeTask const & t2) {
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

  std::priority_queue<NodeTask, std::vector<NodeTask>, CompareNodeTaskTime> heap_;

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

// A single instance of this struct should be created through the whole process lifetime.
// The worker thread creation logic and Engine's destructor rely on this.
struct TORCH_API Engine {
  /// Returns a reference to a static `Engine` instance.
  static Engine& get_default_engine();

  static Engine& get_base_engine();

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
      std::shared_ptr<GraphTask> graph_task,
      const std::shared_ptr<Node>& fn,
      std::exception& e);

  void queue_callback(std::function<void()> callback);

  bool is_checkpoint_valid();

  size_t ready_queue_size(const std::shared_ptr<GraphTask>& graph_task, at::Device device);

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

  // initialize the thread local ready queue with the ready queue that is created
  // elsewhere (i.e. thread_init, Engine::execute, etc), or create a new
  // ready queue if ready_queue is not provided.
  void init_local_ready_queue(std::shared_ptr<ReadyQueue> ready_queue = nullptr);

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

  // Ensures device_ready_queues_ are initialized only once
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::once_flag start_device_threads_flag_;
  // Safe to read device_ready_queues_ without synchronization after initialization
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
    unsigned int num_workers_;
    // The threads will wait on work_ to be notified of GraphTasks
    std::condition_variable work_;
    // To protect reads and writes to graphtask_queue_ and num_workers_
    // and for synchronizing creating new threads when needed
    std::mutex mutex_;
    // Workers will process the GraphTasks added to this queue. A GraphTask is
    // allocated inside Engine::execute and lives for the duration of execute
    std::queue<std::weak_ptr<GraphTask>> graphtasks_queue_;

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    ThreadPoolShared() : num_workers_(0) {}
 };

 // Temporary workaround until shutting down threads is done
 // We need shared ownership of all these objects because the threads are leaked
 // when Engine shuts down, so there may be threads waiting on work_
 // for the graphtasks_queue_ to be nonempty.
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

}} // namespace torch::autograd
