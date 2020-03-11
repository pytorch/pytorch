#pragma once

// Engine implements backpropagation from output variables and their gradients
// to "root" variables (variables created by the user with requires_grad=True).

#include <ATen/ThreadLocalDebugInfo.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/autograd/anomaly_mode.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/input_buffer.h>
#include <torch/csrc/utils/future.h>

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

using FutureVariableList = torch::utils::Future<variable_list>;

void validate_outputs(
    const edge_list& edges,
    variable_list& grads,
    const std::function<std::string(const std::string&)>& format_error);

static constexpr int NO_DEVICE = -2;
static constexpr int CPU_DEVICE = -1;

// GraphTask holds metadata needed for a single execution of backward()
struct GraphTask {
  // Indicates if an error occurred while executing any task.  When this is
  // true, it signals all threads to stop executing.
  std::atomic_bool has_error_;
  std::atomic<uint64_t> outstanding_tasks_;
  // It is safe to read grad_mode_ and keep_graph_ without synchronization
  bool keep_graph_;
  bool grad_mode_;

  // To protect reads/writes to not_ready_, dependencies_, captured_vars_,
  // has_error_, future_result_, cpu_ready_queue_, and leaf_streams.
  std::mutex mutex_;
  std::unordered_map<Node*, InputBuffer> not_ready_;
  std::unordered_map<Node*, int> dependencies_;

  struct ExecInfo {
    struct Capture {
      Capture(int input_idx, int output_idx)
          : input_idx_(input_idx), output_idx_(output_idx) {}
      int input_idx_; // within Node inputs
      int output_idx_; // within the output vector of a GraphTask
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
  // and this entry has needed == True should be executed. exec_info_.empty()
  // means it's .backward(), otherwise it's .grad(). exec_info_ is safe to read
  // without synchronization
  std::unordered_map<Node*, ExecInfo> exec_info_;
  // Captures variables are grads captured that we return to the user. After
  // execution of the GraphTask is completed, the captured_vars_ are moved
  // out of the GraphTask and are no longer valid.
  std::vector<Variable> captured_vars_;
  std::shared_ptr<at::ThreadLocalDebugInfoBase> debug_info_ =
      at::getThreadLocalDebugInfo();
  std::unordered_set<c10::Stream> leaf_streams;

  void init_to_execute(Node& graph_root, const edge_list& outputs);

  // The value of worker_device in the thread that created this task.
  // See Note [Reentrant backwards]
  // Safe to read owner_ and reentrant_depth_ without synchronizaton
  int owner_;
  // The number of parent graph tasks for this graph task
  const int reentrant_depth_;

  bool can_checkpoint() {
    return exec_info_.empty();
  }

  // Set an appropriate exception on this graph_task which was encountered while
  // running the provided function.
  void set_exception(std::exception& e, const std::shared_ptr<Node>& fn);

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
  // work for it to be done. We memorize the cpu_ready_queue_ per GraphTask so that
  // we know which ready queue we should push to if we are on device thread (i.e. GPU)
  // and but next NodeTask should be run on CPU.
  std::shared_ptr<ReadyQueue> cpu_ready_queue_;

  // Future representing the completion of the graph task. Notified when all
  // tasks are done.
  std::shared_ptr<FutureVariableList> future_result_;

  GraphTask(
      bool keep_graph,
      bool grad_mode,
      int reentrant_depth,
      std::shared_ptr<ReadyQueue> cpu_ready_queue,
      bool exit_on_error = false)
      : has_error_(false),
        outstanding_tasks_(0),
        keep_graph_(keep_graph),
        grad_mode_(grad_mode),
        owner_(NO_DEVICE),
        reentrant_depth_(reentrant_depth),
        exit_on_error_(exit_on_error),
        cpu_ready_queue_(std::move(cpu_ready_queue)),
        future_result_(std::make_shared<FutureVariableList>()) {
          TORCH_INTERNAL_ASSERT(cpu_ready_queue_ != nullptr);
        }
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
      std::weak_ptr<GraphTask> base,
      std::shared_ptr<Node> fn,
      InputBuffer inputs,
      bool isShutdownTask = false)
      : base_(base),
        fn_(std::move(fn)),
        inputs_(std::move(inputs)),
        isShutdownTask_(isShutdownTask) {}
};


struct ReadyQueue {
 private:
  // Returns true when t2 should be (weakly) BEFORE t1 in the queue.
  // Shutdown tasks are first and then empty NodeTask are next.
  struct CompareNodeTaskTime {
    bool operator()(NodeTask const & t1, NodeTask const & t2) {
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
  // always be true, see the doc for 'enqueue_blocked_task_on_cpu' for when we
  // might set this to false.
  void push(NodeTask item, bool incrementOutstandingTasks = true);
  void pushShutdownTask();
  std::unique_ptr<NodeTask> pop();
  bool empty() const;
  size_t size() const;
};

// A single instance of this struct should be created through the whole process lifetime.
// The worker thread creation logic and Engine's destructor rely on this.
struct TORCH_API Engine {
  /// Returns a reference to a static `Engine` instance.
  static Engine& get_default_engine();

  Engine();
  virtual ~Engine();

  // Given a list of (Node, input number) pairs computes the value of the graph
  // by following next_edge references.
  virtual variable_list execute(
      const edge_list& roots,
      const variable_list& inputs,
      bool keep_graph,
      bool create_graph,
      const edge_list& outputs = {});

  // Given a pre-populated GraphTask and GraphRoot, computes the backward pass
  // for the graph. This API should only be used by internal autograd specific
  // machinery and shouldn't be exposed to users in anyway.
  virtual std::shared_ptr<FutureVariableList> execute_with_graph_task(
      const std::shared_ptr<GraphTask>& graph_task,
      std::shared_ptr<Node> graph_root,
      bool async_mode = false);

  // Enqueues a blocked task for execution on the CPU thread. A blocked task is
  // basically a task that isn't triggered automatically to be
  // 'ready to execute' by the autograd engine. This task needs to be unblocked
  // for execution via an external mechanism. This method assumes that
  // the appropriate GraphTask has already been initialized appropriately.
  // Another important part is that this does not increment 'outstanding_tasks_'
  // in the appropriate GraphTask. It is assumed we've already done this before
  // hand for this task (to ensure we block for its execution). This is useful
  // in the distributed autograd case where we need to increment
  // 'outstanding_tasks_' first to indicate the local autograd engine needs to
  // wait for this task, but the task might actually be received later over the
  // network for execution.
  void enqueue_blocked_task_on_cpu(NodeTask task);

  virtual std::unique_ptr<AnomalyMetadata> make_anomaly_metadata() {
    return nullptr;
  }

  void queue_callback(std::function<void()> callback);

  bool is_checkpoint_valid();

  // initialize the thread local ready queue with the ready queue that is created
  // elsewhere, i.e. thread_init, reentrant_thread_init, initialization in the
  // distributed autograd engine, etc.
  std::shared_ptr<ReadyQueue> init_local_ready_queue(std::shared_ptr<ReadyQueue> ready_queue = nullptr);

  size_t ready_queue_size(const std::shared_ptr<GraphTask>& graph_task, at::Device device);

 protected:
  void compute_dependencies(Node* root, GraphTask& task);
  void evaluate_function(
      std::shared_ptr<GraphTask>& graph_task,
      Node* func,
      InputBuffer& inputs);
  ReadyQueue& ready_queue(
      const std::shared_ptr<GraphTask>& graph_task,
      at::Device device);
  ReadyQueue& ready_queue_by_index(
      const std::shared_ptr<GraphTask>& graph_task,
      int device_index);
  // start device threads (CUDA, XLA, etc.) in Engine,
  // note that it does NOT start CPU thread.
  void start_device_threads();
  virtual void thread_init(int device, const std::shared_ptr<ReadyQueue>& ready_queue);
  virtual void thread_on_exception(
      std::shared_ptr<GraphTask> graph_task,
      const std::shared_ptr<Node>& fn,
      std::exception& e);
  virtual void thread_main(
      const std::shared_ptr<GraphTask>& task,
      bool reentrant_thread);
  void reentrant_thread_init(const std::shared_ptr<ReadyQueue>& parent_ready_queue);
  void add_thread_pool_task(const std::weak_ptr<GraphTask>& graph_task);
  void set_device(int device);
  void initialize_device_threads_pool();

  // Ensures ready_queues_ are initialized only once
  std::once_flag start_device_threads_flag_;
  // Safe to read ready_queues_ without synchronization after intialization
  std::vector<std::shared_ptr<ReadyQueue>> device_ready_queues_;
  std::vector<std::function<void()>> final_callbacks_;
  // To protect reads and writes to final_callbacks_
  std::mutex post_callbacks_lock_;

  // How many nested reentrant calls are allowed until a new thread is used
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

    ThreadPoolShared() : num_workers_(0) {}
 };

 // Temporary workaround until shutting down threads is done
 // We need shared ownership of all these objects because the threads are leaked
 // when Engine shuts down, so there may be threads waiting on work_
 // for the graphtasks_queue_ to be nonempty.
 std::shared_ptr<ThreadPoolShared> thread_pool_shared_;

private:
 void execute_until_ready_queue_empty(const std::shared_ptr<GraphTask>& graph_task);
 void graph_task_exec_post_processing(
     const std::shared_ptr<GraphTask>& graph_task);
 void mark_graph_task_completed(const std::shared_ptr<GraphTask>& graph_task);
};

// allow python_engine to override the default engine when it loads
using EngineStub = Engine& (*)();
TORCH_API void set_default_engine_stub(EngineStub stub);

}} // namespace torch::autograd
