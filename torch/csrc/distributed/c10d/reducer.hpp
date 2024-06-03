#pragma once

#include <c10/core/ScalarType.h>
#include <atomic>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <ATen/core/ivalue_inl.h>
#include <c10/macros/Macros.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/comm.hpp>
#include <torch/csrc/distributed/c10d/debug.h>
#include <torch/csrc/distributed/c10d/default_comm_hooks.hpp>
#include <torch/csrc/distributed/c10d/reducer_timer.hpp>
#ifndef _WIN32
#include <torch/csrc/distributed/autograd/context/context.h>
#endif

namespace c10d {

constexpr int kDefaultFirstBucketBytes = int(1024 * 1024);
constexpr int kDefaultBucketBytesCap = int(25 * 1024 * 1024);
// Collect runtime stats once for every kDDPRuntimeLoggingSampleRate iterations.
constexpr int kDDPRuntimeLoggingSampleRate = 100;

// Forward declaration
class Logger;

// Local accumulator type for a single bucket.
struct BucketAccumulator {
  std::vector<size_t> indices;
  size_t size = 0;
  size_t size_limit = 0;
};

class TORCH_API Reducer {
 public:
  // The constructor takes a list of variables (i.e. parameters) for this
  // process's single model replica (as DDP assumes single-process
  // single-device). The bucket assignment for this reducer, `bucket_indices`,
  // is specified as a list of buckets, each of which is specified as a list of
  // indices into the bucket's `variables` list.
  explicit Reducer(
      std::vector<at::Tensor> params,
      std::vector<std::vector<size_t>> bucket_indices,
      const std::vector<size_t>& per_bucket_size_limits,
      c10::intrusive_ptr<c10d::ProcessGroup> process_group,
      std::vector<bool> expect_sparse_gradients,
      int64_t bucket_bytes_cap,
      bool find_unused_parameters,
      bool gradient_as_bucket_view,
      std::unordered_map<size_t, std::string> param_names,
      int64_t first_bucket_bytes_cap);

  ~Reducer() noexcept(false);

  // To (re-)initialize bucket assignment, pass a list of buckets, each of
  // which is specified by a list of indices in the bucket's `variables` list.
  // This function performs validation that the variables within a bucket
  // all live on the same device and have the same dimensionality.
  void initialize_buckets(std::vector<std::vector<size_t>> bucket_indices);

  void autograd_hook(size_t index);

  // This function is called when the forward function has produced an output,
  // and the user wishes to reduce gradients in the backwards pass.
  // If they don't, and wish to accumulate gradients before reducing them,
  // a call to this function can simply be omitted.
  void prepare_for_backward(const std::vector<at::Tensor>& outputs);

  // Called at the beginning of forward() inside DistributedDataParallel,
  // right now it captures the starting time of forward in each iteration.
  void prepare_for_forward();

  // Returns the relative time in nanoseconds when gradients were ready,
  // with respect to the time `prepare_for_backward` was called. The
  // vector is for parameters for a single model replica.
  std::vector<int64_t> get_backward_stats() const {
    return backward_stats_;
  }

  // Registers a hook to the reducer. The hook is `CommHookInterface`
  // type to allow both Python and CPP hooks. This function can only
  // be called once before calling backward.
  // Cannot combine with the call of `register_builtin_comm_hook`.
  void register_comm_hook(std::unique_ptr<CommHookInterface> iface);

  // Registers a built-in C++ comm hook to the reducer. This function can only
  // be called once before calling backward.
  // Cannot combine with the call of `register_comm_hook`.
  void register_builtin_comm_hook(c10d::BuiltinCommHookType comm_hook_type);

  // Informs reducer that optimizer is running in backward, so gradients
  // don't need to be copied from buckets as the optimizer would've already
  // been applied.
  void set_optimizer_in_backward() {
    optim_in_backward_ = true;
  };

  // Runs allreduce or installed communication hook given GradBucket instance.
  c10::intrusive_ptr<c10::ivalue::Future> run_comm_hook(
      GradBucket& grad_bucket);

  // Runs default allreduce hook.
  c10::intrusive_ptr<c10::ivalue::Future> run_allreduce_hook(
      GradBucket& grad_bucket);

  // Returns gradient buckets in sequential order of buckets_. This is the order
  // in which buckets are reduced across processes. If return_zero_tensors=true,
  // will return zero tensors of the same shape instead of the true tensors.
  std::vector<c10d::GradBucket> get_grad_buckets(
      bool return_zero_tensors = true) const;

  // Rebuild buckets based on rebuilt_params_ and rebuilt_param_indices_
  // according to when tensors received grads in the backward pass.
  // TODO this function makes broadcast communication call and
  // could be overlapped with next forward() call, thus
  // it could be async. Will make it async when rebuilding buckets for
  // find_unused_parameters = true case, as we could rebuild buckets more than
  // once for find_unused_parameters = true case, where subgraphs are trained
  // and parameter indices order may change more frequently.
  // For find_unused_parameters = false case, buckets are only rebuilt once,
  // the performance cost is negligible. Returns true if the buckets were
  // rebuilt.
  bool rebuild_buckets();

  void setSparseMetadata(std::map<std::string, at::Tensor>& metadata);

  // Install futures that should be awaited at end of backwards. Currently these
  // are only used by user-defined custom buffer reduction hooks, but can be
  // generalized to any user-originating futures that need to be awaited.
  void install_futures(c10::List<c10::intrusive_ptr<c10::ivalue::Future>> futs);

  // Returns true if we should rebuild buckets, else false. We only rebuild
  // buckets once after the first iteration and never rebuild them if
  // find_unused_parameters_.
  inline bool should_rebuild_buckets() const {
    return (static_graph_ || !find_unused_parameters_) && !has_rebuilt_bucket_;
  }

  // Pushes all parameters to be rebuilt.
  void push_rebuilt_params_for_all_indices();

  // Creates and sets ForwardPassWorkHandle given a Work and the
  // corresponding tensor being reduced.
  void set_forward_pass_work_handle(
      c10::intrusive_ptr<c10d::Work> forwardPassWorkHandle,
      bool useStaticWorldSize);

  // Retrieve on-device tensors used to track locally unused parameters. It is
  // a tensor where index i = 1 if the Variable with that index has been used.
  at::Tensor get_local_used_map_on_device() const;

  // An function for users to set sample_rate of collecting
  // runtime stats. The time stats will be recorded for the
  // first 10 iterations, after 10 iterations time stats will be
  // recorded once every "sample_rate" training iterations.
  void set_ddp_runtime_logging_sample_rate(int sample_rate);

  // Specify the training graph is static.
  void set_static_graph();

  // Delay all reduce to be after all gradients' calculation is complete.
  void delay_all_reduce();

  void set_mixed_precision_param_dtype(c10::ScalarType dtype);

  // Weak reference to associated DDP logger. The reference is weak to avoid
  // refcycle between reducer and logger.
  void set_logger(std::weak_ptr<c10d::Logger> logger);

  // When graph is not explicitly set by user as static and has unused
  // parameters, this will return whether the graph has been static until the
  // current iteration, which means unused params set has not changed.
  bool ddp_graph_static();

  // Removes autograd hooks registered by the Reducer on the model parameters.
  void remove_autograd_hooks();

  // Checks whether or not the reducer has finalized the current backward
  // iteration.
  void check_finalized();

  // Updates the underlying process group used by DDP with the new process
  // group.
  void update_process_group(
      c10::intrusive_ptr<c10d::ProcessGroup> new_process_group);

  // Resets reducer state.
  void reset_state();

 protected:
  // Forward declaration.
  struct Bucket;

  void push_rebuilt_params(const size_t& index);

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  mutable std::mutex mutex_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  const std::vector<at::Tensor> params_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  c10::intrusive_ptr<::c10d::ProcessGroup> process_group_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::vector<bool> expect_sparse_gradients_;

  std::vector<std::shared_ptr<torch::autograd::Node>>
      grad_accumulators_; // NOLINT(cppcoreguidelines-non-private-member-variables-in-classes)
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::unordered_map<torch::autograd::Node*, size_t> gradAccToVariableMap_;
  std::vector<std::pair<uintptr_t, std::shared_ptr<torch::autograd::Node>>>
      hooks_; // NOLINT(cppcoreguidelines-non-private-member-variables-in-classes)

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  bool expect_autograd_hooks_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  bool require_finalize_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  size_t next_bucket_;

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  bool has_marked_unused_parameters_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  const bool find_unused_parameters_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  const bool gradient_as_bucket_view_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::vector<size_t> unused_parameters_;
  // Previous iteration's unused params, used for checking if unused parameters
  // change between iterations. Only filled during the first backwards call.
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::vector<size_t> prev_iteration_unused_parameters_;
  // Whether graph is static or not. When user does not explicitly set static
  // graph, the only possible dynamism is set of unused parameters changing
  // between iterations which is tracked by this flag.
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  bool ddp_graph_static_{true};
  // Locally used parameter maps indicating if parameters are used locally
  // during the current iteration or no_sync session if no_sync is on.
  // Each map is a one-dim int32 tensor of number of parameters. These tensors
  // are marked in autograd_hook to indicate the corresponding param has been
  // used, and get allreduced in the end of backward step of current iteration
  // or no_sync session for figuring out the globally unused parameters.
  //
  // local_used_map_:     CPU tensor for bookkeeping locally used params
  // local_used_map_dev_: dev tensor for reducing globally unused params
  at::Tensor local_used_map_;
  at::Tensor local_used_map_dev_;
  // Indicate that reduction is done and D2H copy is done as well.
  bool local_used_map_reduced_;

  // Weak pointer to associated DDP logger.
  std::weak_ptr<c10d::Logger> logger_;
  // List of futures installed by Reducer::install_futures that should be
  // awaited at the end of backwards pass.
  std::optional<c10::List<c10::intrusive_ptr<c10::ivalue::Future>>>
      installed_futures_{c10::nullopt};
  // Mixed precision parameter dtype for bucket type checking.
  std::optional<c10::ScalarType> mixed_precision_param_dtype_{c10::nullopt};

  // Work handle for allreduce on local_used_map_
  c10::intrusive_ptr<c10d::Work> local_used_work_;

  void mark_variable_ready_dense(size_t variable_index);

  void mark_variable_ready_sparse(size_t variable_index);

  void mark_variable_ready(size_t variable_index);

  void mark_bucket_ready(size_t bucket_index);

  void finalize_bucket_dense(Bucket& bucket);

  void finalize_backward();

  // Returns list of model parameters corresponding to the given bucket.
  // bucket_index is a key to cache after buckets are rebuilt, after which this
  // mapping never changes.
  std::vector<at::Tensor> get_variables_for_bucket(
      size_t bucket_index,
      const Bucket& bucket) const;

  // Asserts that the reduction for the previous iteration has finished before
  // rebuilding buckets or kicking off the next one.
  void ensure_prior_reduction_finished();

  // Broadcast rebuilt buckets from rank 0 to other ranks before initializing
  // the buckets
  void sync_bucket_indices(std::vector<std::vector<size_t>>& bucket_indices);

  // We'd like to use DistAutogradContext::GradCallback here but dist autograd
  // doesn't exist under Windows. So we just directly use the concrete type but
  // to preserve and enforce our original intent we do a static assert when dist
  // autograd is available.
  using GradCallback = std::function<bool(at::Tensor&)>;
#ifndef _WIN32
  static_assert(
      std::is_same_v<
          GradCallback,
          torch::distributed::autograd::DistAutogradContext::GradCallback>);
#endif
  void runGradCallbackForVariable(at::Tensor& variable, GradCallback&& cb);

  // This function is called inside `initialize_buckets()`. It initializes both
  // `bucket_views_in` and `bucket_views_out` with views for each variable's
  // gradient into the bucket's flattened `gradients` tensor. Views serve as
  // entry points to `copy_()` each grad's data in/out of the flattened
  // `gradients` tensor.
  void initialize_bucket_views(Bucket& bucket);

  // This function is called inside `finalize_backward`, it happens only if
  // DDP communication hook was registered to recreate just bucket_views_out
  // with the result of `future_work`.
  void populate_bucket_views_out(Bucket& bucket, at::Tensor& tensor);

  // If gradient_as_bucket_view_ is false, after allreduce buckets,
  // copy bucket results back to grads.
  void copy_bucket_to_grad(
      at::Tensor& variable,
      Reducer::Bucket& bucket,
      size_t intra_bucket_index,
      bool global_unused);
  // Check layout of grad and bucket_view before copying the grad to bucket.
  void check_grad_layout(const at::Tensor& grad, const at::Tensor& bucket_view);

  // A bucket contains [1..N] gradients to be reduced, where the gradients
  // have the same dtype and device.
  // Coalescing gradients together before reducing can result in lower overhead
  // and/or faster time to completion. Coalescing requires the constituent
  // gradients to have the same dtype and device, and the resulting flattened
  // tensor uses that common dtype and device. The flattened tensor is filled
  // as the corresponding gradients are computed (triggered by autograd hooks),
  // and the buckets are reduced in a predetermined order consistent across
  // processes.
  struct Bucket {
    // Gradients of the bucket flattened into a 1-dimensional tensor
    at::Tensor gradients;

    // Views into the `gradients` tensor for each individual gradient
    // Each view is created with layout (size and stride) matching the
    // gradient's expected layout (see the "Gradient Layout Contract" in
    // torch/csrc/autograd/functions/accumulate_grad.h).
    // `bucket_views_in[i].copy_(grad)` and `grad.copy_(bucket_views_out[i])`
    // provide convenient ways to copy gradient data in/out of `gradients`,
    // respectively.
    // We keep both `bucket_views_in` and `bucket_views_out` because
    // registering a DDP communication hook may re-initialize
    // `bucket_views_out` with the value of the hook's `future_work` but we
    // still need separate views into the bucket's original flattened gradient
    // to copy in gradient data.
    std::vector<at::Tensor> bucket_views_in;
    std::vector<at::Tensor> bucket_views_out;

    // Variables whose gradients are held in this bucket
    // We use refcounted tensors here so that we can easily unflatten the
    // bucket's flattened `gradients` tensor into the participating variables
    // after reduction has completed.
    std::vector<at::Tensor> variables;

    // Per-variable offset/length into the flattened `gradients` tensor and
    // the corresponding `GradBucket` instance for communication hooks
    std::vector<size_t> offsets;
    std::vector<size_t> lengths;

    // Per-variable sizes slicing into the bucket's `gradients` tensor
    std::vector<c10::IntArrayRef> sizes_vec;

    // Number of gradients left to be computed before the bucket is ready to
    // be reduced
    size_t pending;

    // Global indices of participating variables in the bucket
    std::vector<size_t> variable_indices;

    // Future work handle for DDP communication hook
    // If no hook is registered, a temporary vanilla allreduce hook is used.
    c10::intrusive_ptr<at::ivalue::Future> future_work;

    // If this bucket should expect a single sparse gradient
    // If `true`, then this implies that `bucket.variables.size() == 1`.
    bool expect_sparse_gradient = false;

    // Sparse indices tensor
    std::optional<at::Tensor> sparse_tensor_indices = c10::nullopt;

    // TODO(@pietern)
    // Memory copies from gradient tensors into the bucket are potentially
    // done on different CUDA streams. We record an event for every copy
    // so that we can synchronize with them prior to kicking off the reduction.
    // std::vector<at::cuda::CUDAEvent> events;
  };

  std::vector<Bucket> buckets_;

  // A variable locator locates a particular variable in the reducer's buckets
  struct VariableLocator {
    // Index of the bucket containing the variable in the `buckets_` vector
    size_t bucket_index;
    // Index of the variable in the bucket, which may be used consistently
    // across `bucket_views_in`, `bucket_views_out`, `variables`, `offsets`,
    // `lengths`, `sizes_vec`, and `variable_indices` in `Bucket`
    size_t intra_bucket_index;

    VariableLocator() = default;

    VariableLocator(size_t bucket_index_, size_t intra_bucket_index_)
        : bucket_index(bucket_index_),
          intra_bucket_index(intra_bucket_index_) {}
  };

  // Map the index of a variable to its location in the bucket structure.
  std::vector<VariableLocator> variable_locators_;

  // track the number of iterations to synchronize grads in training so far.
  long num_iterations_;
  // track distinct iteration of backward call. This is distinct from
  // num_iterations_, for example in the case of multiple forward before
  // backward.
  long num_bwd_calls_;
  // whether the first autograd hook for a distinct backward pass has been
  // called.
  bool first_autograd_hook_called_;
  // track the number of buckets that have been ready for
  // communication calls like allReduce or communication hooks.
  int num_buckets_ready_;

  // Timing information.
  int64_t backward_compute_start_time_ = -1;
  std::unique_ptr<Timer> timer_;

  // We collect the relative timestamp of every gradient being ready
  // when executing autograd. This can be used to derive a timeline of
  // the point in time buckets were ready, or ideal bucket assignment/ordering.
  std::vector<int64_t> backward_stats_;

  bool should_collect_runtime_stats();
  void record_forward_compute_start_time();
  void record_backward_compute_start_time();
  void record_backward_compute_end_time();
  void record_backward_comm_start_time();
  void record_backward_comm_end_time();

  int get_ddp_runtime_logging_sample_rate();
  int ddp_runtime_logging_sample_rate_ = kDDPRuntimeLoggingSampleRate;

  bool is_multi_device_module_ = false;

  // Following variables are to help build dynamic bucket order
  bool has_rebuilt_bucket_;
  std::vector<at::Tensor> rebuilt_params_;
  std::vector<int64_t> rebuilt_param_indices_;
  const int64_t bucket_bytes_cap_;

#ifndef _WIN32
  struct RpcContext {
    using ContextPtr = torch::distributed::autograd::ContextPtr;
    // The shared_ptr is to hold the context instance.
    ContextPtr context_ptr_holder;
    std::atomic<ContextPtr::element_type*> context_ptr{nullptr};

    void set(ContextPtr&& new_context_ptr);
  };
  RpcContext rpc_context_;
#endif

  // A struct containing work handle and tensor for allreduce scheduled in
  // forward pass, if applicable.
  struct ForwardPassAllreduceWork {
    c10::intrusive_ptr<c10d::Work> workHandle;
    at::Tensor resultTensor;
    // whether we should divide by the initial world_size or the no. of
    // remaining DDP ranks.
    bool useStaticWorldSize;
  };

  // Handle for the currently scheduled allreduce in the forward pass, if
  // applicable.
  ForwardPassAllreduceWork forwardPassWorkHandle_;

  // Division factor for reduction of gradients.
  // Equal to the process group size, with an exception of handling uneven
  // input.
  int div_factor_;

  bool static_graph_;

  // Key: size_t (index), Value: the number of times that a variable's
  // autograd_hook() should be triggered before marking this variable's grad as
  // ready for communication. Map will not change after 1st iteration.
  std::unordered_map<size_t, int> numGradHooksTriggeredMap_;
  // Key: size_t (index), Value: the number of times that a variable's
  // autograd_hook() are left to be triggered before marking this variable's
  // grad as ready for communication. Map will change after 1st iteration to
  // track a grad is ready for communication or not.
  std::unordered_map<size_t, int> numGradHooksTriggeredMapPerIteration_;

 private:
  // reset counting for buckets before backward starts
  void reset_bucket_counting();
  // search unused parameters beore backward starts
  void search_unused_parameters(
      const std::vector<torch::autograd::Variable>& outputs);
  void set_divide_factor();
  // kick off all reduce for the ready bucket
  void all_reduce_bucket(Bucket& bucket);
  // kick off all reduce to local used map, it can help find global unused
  // parameters
  void all_reduce_local_used_map();
  // initialize locally used parameter maps
  void initialize_local_used_map();
  // get current cuda stream
  const c10::Stream get_current_stream();
  bool dynamic_graph_find_unused();
  bool static_graph_first_iteration();
  bool static_graph_after_first_iteration();

  // comm_hook_ is used to access the DDP communication hook if registered.
  std::unique_ptr<CommHookInterface> comm_hook_;

  // Sparse metadata contains the indices that will be used
  // when calling into sparse allreduce.
  // This is only used in the sparse allreduce collective calls
  std::unique_ptr<std::map<std::string, at::Tensor>> sparse_metadata_;

  // Debug level setting. It is parsed once when Reducer is constructed, and
  // remains the same across a single invocation of DDP training.
  DebugLevel ddp_debug_level_;
  // Mapping of variable index to fully qualified name of model to notify users
  // about errors when certain parameters do not get gradient.
  std::unordered_map<size_t, std::string> param_names_;
  // Variable indices stored sequentially in order of when the gradient is ready
  // for the current backwards pass.
  std::vector<int64_t> grad_ready_order_indices_;
  // Bytes capacity of first bucket, can be configured by user
  int64_t first_bucket_bytes_cap_;
  // Per iteration set of parameter indices that have been marked ready.
  std::unordered_set<size_t> perIterationReadyParams_;
  // Retrieves parameter names that have not been marked as ready as part of
  // previous iteration.
  std::vector<std::string> getUnmarkedParamsForIteration();
  // Retrieves parameter indices that have not been marked as ready as part of
  // previous iteration.
  std::vector<size_t> getUnmarkedParamIndicesForIteration();
  // Raises appropriate error if mark_variable_ready is called on the same
  // variable twice, which is unexpected.
  void checkAndRaiseMarkedTwiceError(size_t curVariableIndex);
  // Retrieves parameter corresponding to the given VariableIndex.
  at::Tensor& get_param_from_index(size_t index);

  // Cached bucket index to model parameter mapping. Populated after buckets
  // are rebuilt after which this mapping is static.
  mutable std::unordered_map<size_t, std::vector<at::Tensor>>
      cached_variables_for_bucket_;

  bool optim_in_backward_{false};
  friend class Logger;
};

// This is equivalent to take_tensors but returns indices into the
// tensor list argument for bucket assignment. Also, it is aware
// of device placement and will not allow buckets to span devices.
// The index of tensors[i] assigned to bucket is tensor_indices[i],
// when tensor_indices is empty, the index of tensors[i] assigned to
// bucket is i.
TORCH_API std::tuple<std::vector<std::vector<size_t>>, std::vector<size_t>>
compute_bucket_assignment_by_size(
    const std::vector<at::Tensor>& tensors,
    const std::vector<size_t>& bucket_size,
    const std::vector<bool>& expect_sparse_gradient = {},
    const std::vector<int64_t>& tensor_indices = {},
    const std::optional<std::weak_ptr<c10d::Logger>>& logger = {});

// Verify models across all processes are the same as model on rank 0 with
// respect to no. of params and matching dtype/size/layout.
TORCH_API void verify_params_across_processes(
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    const std::vector<at::Tensor>& params,
    const std::optional<std::weak_ptr<c10d::Logger>>& logger);
} // namespace c10d
