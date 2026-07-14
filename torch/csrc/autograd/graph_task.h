#pragma once
#include <ATen/ThreadLocalState.h>
#include <ATen/core/Tensor.h>
#include <c10/util/ThreadLocal.h>
#include <torch/csrc/autograd/input_buffer.h>
#include <torch/csrc/autograd/utils/warnings.h>
#include <vector>

namespace torch::autograd {

using edge_list = std::vector<Edge>;
struct ReadyQueue;

static constexpr int NO_DEVICE = -2;
static constexpr int CPU_DEVICE = -1;

// GraphTask holds metadata needed for a single execution of backward()
struct GraphTask : std::enable_shared_from_this<GraphTask> {
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

  // Records the nodes that are in the graph
  std::unordered_set<Node*> nodes_in_graph_;
  c10::SmallVector<Node*, 4> graph_roots_;
  // Note [Exec info]
  // Exec info is created for each GraphTask, which allows filtering paths on
  // the graph that are not needed. It has a bit complicated semantics. If it's
  // empty, it means the task is run in a "default" mode, which means that all
  // next_edges we encounter should get executed. If it's not empty, only
  // functions that have an entry and this entry has needed == True should be
  // executed. exec_info is only empty when the graph is executed via
  // .backward() and the inputs parameter is not passed. Otherwise, when
  // executed through .grad(), or when inputs arg is specified for .backward(),
  // exec_info will be non-empty.
  //
  struct ExecInfo {
    struct Capture {
      Capture(const Capture&) = delete;
      Capture(Capture&&) = default;
      Capture& operator=(const Capture&) = delete;
      Capture& operator=(Capture&&) = default;
      ~Capture() = default;

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
      // NOTE [Deprecated capture hooks]
      //
      // The current status of capture hooks is that we continue to support
      // the single usage of it by distributed in the dist_engine. If anyone
      // else needs to use it for other purposes, they should file an issue.
      //
      // Capture hooks were originally created because there did not exist
      // any way to register pre/post hooks to grad_fn in a way such that it
      // would still be executed even if that is the grad_fn of a Tensor
      // passed as input= of .grad. As far as I know, only dist_engine uses
      // this hook.
      //
      // However, there are other alternatives today like tensor hooks that can
      // replace the usage that originally motivated its creation. Also,
      // Captures hooks are an outlier in terms of the types of hook that
      // autograd offers in how it is registered and behaves, e.g. it is a hook
      // registered not to the graph, but to a particular graph_task! This makes
      // it a burden to maintain.
      //
      // It would be very nice to clean up/do a migration from pre/post
      // hooks used in distributed to use tensor hooks, but for now we just
      // mark this method as deprecated to prevent additional usage.
      //
      // If you still think you really need to capture hooks, please file an
      // issue (and tag autograd).
      const std::vector<std::unique_ptr<GradCaptureHook>>&
      DO_NOT_USE_DEPRECATED_get_capture_hooks() const {
        return hooks_;
      }
      // See NOTE [deprecated capture hooks]
      void DO_NOT_USE_DEPRECATED_register_capture_hook(
          std::unique_ptr<GradCaptureHook> hook) {
        hooks_.push_back(std::move(hook));
      }

     private:
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
  // exec_info_ is safe to read without synchronization
  std::unordered_map<Node*, ExecInfo> exec_info_;
  // Captures variables are grads captured that we return to the user. After
  // execution of the GraphTask is completed, the captured_vars_ are moved
  // out of the GraphTask and are no longer valid.
  std::vector<Variable> captured_vars_;

  // Note: this field is not ready to be used until the proper
  // `thread_locals_.set_grad_mode()` call in the constructor.
  at::ThreadLocalState thread_locals_;

  std::unordered_set<c10::Stream> leaf_streams;

  // Per-device current streams of the execute() that called this GraphTask.
  // These will be synced with leaf_streams in exec_post_processing.
  std::vector<std::optional<c10::Stream>> caller_current_streams_;

  // Collects caller_current_streams_ for the accelerator device.
  void stash_current_streams();

  void init_to_execute(
      Node& graph_root,
      const edge_list& outputs,
      bool accumulate_grad,
      uint64_t min_topo_nr);

  // The value of worker_device in the thread that created this task.
  // See Note [Reentrant backwards]
  // Safe to read owner_ and reentrant_depth_ without synchronization
  int owner_{NO_DEVICE};
  // The number of parent graph tasks for this graph task
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
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

  // CPU threads are dedicated to processing CPU work for the backward they
  // invoked. So any given graph task maintains its own cpu_ready_queue_ where
  // you should send work for it to be done. We memoize the cpu_ready_queue_ per
  // GraphTask so that we know which ready queue we should push to if we are on
  // device thread (i.e. GPU) and but next NodeTask should be run on CPU.
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

  uint64_t id_;

  GraphTask(
      bool keep_graph,
      bool grad_mode,
      int reentrant_depth,
      std::shared_ptr<ReadyQueue> cpu_ready_queue,
      c10::SmallVector<Node*, 4> graph_roots,
      bool exit_on_error = false);

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

TORCH_API const std::unordered_map<Node*, GraphTask::ExecInfo>*
get_current_graph_task_exec_info();
TORCH_API const std::unordered_set<Node*>*
get_current_graph_task_nodes_in_graph();
TORCH_API bool get_current_graph_task_keep_graph();
TORCH_API std::vector<Node*> get_current_graph_task_execution_order();
TORCH_API int get_current_graph_task_id();
void add_node_to_current_graph_task_exec_info(Node* fn);

} // namespace torch::autograd
