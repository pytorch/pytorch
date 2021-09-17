#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <ATen/core/ivalue_inl.h>
#include <c10/macros/Macros.h>
#include <c10/util/intrusive_ptr.h>
#include <c10d/ProcessGroup.hpp>
#include <c10d/Utils.hpp>
#include <c10d/comm.hpp>
#include <c10d/default_comm_hooks.hpp>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#ifndef _WIN32
#include <torch/csrc/distributed/autograd/context/context.h>
#endif

namespace c10d {

constexpr int kDefaultFirstBucketBytes = int(1024 * 1024);
constexpr int kDefaultBucketBytesCap = int(25 * 1024 * 1024);
// Collect runtime stats once for every kDDPRuntimeLoggingSampleRate iterations.
constexpr int kDDPRuntimeLoggingSampleRate = 100;
constexpr int kUnsetTime = -1;

inline int64_t current_time_in_nanos() {
  return torch::autograd::profiler::getTime();
}

// Forward declaration
class Logger;

class TORCH_API Timer {
  private:
    // The timestamp of forward call start time in each iteration.
    int64_t forward_start_time = kUnsetTime;
    // The timestamp of backward computation start and end time in each
    // iteration.
    int64_t backward_compute_start_time = kUnsetTime;
    int64_t backward_compute_end_time = kUnsetTime;
    // The timestamp of first communication call start time in each iteration.
    int64_t backward_comm_start_time = kUnsetTime;
    // The timestamp of last communication call end time in each iteration.
  int64_t backward_comm_end_time = kUnsetTime;
 public:
  enum class Event {
    kForwardStart,
    kBackwardComputeStart,
    kBackwardComputeEnd,
    kBackwardCommStart,
    kBackwardCommEnd,
  };

  // Record the current event, i.e., mark it as having occurred now. Default
  // CPU implementation.
  virtual void record(Event event) {
    getTimeRef(event) = current_time_in_nanos();
  }

  // Return the difference between when two events occurred, in nanoseconds.
  // Or nullopt if one of them hasn't been recorded.
  virtual c10::optional<int64_t> measureDifference(Event start, Event end) = 0;

  virtual ~Timer() = default;

  // Return host-side timestamp, or nullopt if it has not yet been recorded.
  c10::optional<int64_t> getTimestamp(Event event) {
    auto time = getTimeRef(event);
    if (time == kUnsetTime) {
      return c10::nullopt;
    } else {
      return time;
    }
  }

  // Return host-side time member variable corresponding to the given event.
  int64_t& getTimeRef(Event event) {
    switch (event) {
      case Event::kForwardStart:
        return forward_start_time;
      case Event::kBackwardComputeStart:
        return backward_compute_start_time;
      case Event::kBackwardComputeEnd:
        return backward_compute_end_time;
      case Event::kBackwardCommStart:
        return backward_comm_start_time;
      case Event::kBackwardCommEnd:
        return backward_comm_end_time;
      default:
        TORCH_INTERNAL_ASSERT(false);
    }
  }
};

// Local accumulator type for a single bucket.
struct BucketAccumulator {
  std::vector<size_t> indices;
  size_t size = 0;
  size_t size_limit = 0;
};

C10_DECLARE_TYPED_REGISTRY(TimerRegistry, c10::DeviceType, Timer, std::unique_ptr, c10::Device);

class TORCH_API Reducer {
 public:
  // The constructor takes a list of variables for every model replica.
  // The bucket assignment for this reducer is specified as a list of
  // buckets, each of which is specified as a list of indices into the
  // variables list for **a single replica** (i.e. `variables[0]`).
  explicit Reducer(
      std::vector<std::vector<at::Tensor>> replicas,
      std::vector<std::vector<size_t>> bucket_indices,
      std::vector<size_t> per_bucket_size_limits,
      c10::intrusive_ptr<c10d::ProcessGroup> process_group,
      std::vector<std::vector<bool>> expect_sparse_gradients,
      int64_t bucket_bytes_cap,
      bool find_unused_parameters,
      bool gradient_as_bucket_view,
      std::unordered_map<size_t, std::string> paramNames,
      int64_t first_bucket_bytes_cap);

  ~Reducer() noexcept(false);

  // To (re-)initialize bucket assignment, pass a list of buckets, each
  // of which is specified by a list of indices in the variables list.
  // This function performs validation that the variables within a bucket
  // all live on the same device and have the same dimensionality.
  void initialize_buckets(
      std::vector<std::vector<size_t>> bucket_indices,
      std::vector<size_t> per_bucket_sizes);

  // This function is called when the forward function has produced an output,
  // and the user wishes to reduce gradients in the backwards pass.
  // If they don't, and wish to accumulate gradients before reducing them,
  // a call to this function can simply be omitted.
  void prepare_for_backward(const std::vector<at::Tensor>& outputs);

  // Called at the begginning of forward() inside DistributedDataParallel,
  // right now it caputures the starting time of forward in each iteration.
  void prepare_for_forward();

  // Returns the relative time in nanoseconds when gradients were ready,
  // with respect to the time `prepare_for_backward` was called. The outer
  // vector is for model replicas and the inner vector is for parameters.
  std::vector<std::vector<int64_t>> get_backward_stats() const {
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

  // Runs allreduce or installed communication hook given GradBucket instance.
  c10::intrusive_ptr<c10::ivalue::Future> run_comm_hook(
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

  // Returns true if we should rebuild buckets, else false. We only rebuild
  // buckets once after the first iteration and never rebuild them if
  // find_unused_parameters_.
  inline bool should_rebuild_buckets() const {
    return (static_graph_ || !find_unused_parameters_) && !has_rebuilt_bucket_;
  }

  // Pushes all parameters to be rebuilt.
  void push_rebuilt_params_for_all_indices();

  // Creates and sets ForwardPassWorkHandle given a ProcessGroup::Work and the
  // corresponding tensor being reduced.
  void set_forward_pass_work_handle(
      c10::intrusive_ptr<c10d::ProcessGroup::Work> forwardPassWorkHandle,
      bool useStaticWorldSize);

  // Retrieve on-device tensors used to track locally unused parameters. For
  // each replica, it is a tensor where index i = 1 if the Variable with that
  // index has been used.
  std::vector<at::Tensor> get_local_used_maps_on_device() const;

  // An function for users to set sample_rate of collecting
  // runtime stats. The time stats will be recorded for the
  // first 10 iterations, after 10 iteratons time stats will be
  // recorded once every "sample_rate" training iterations.
  void set_ddp_runtime_logging_sample_rate(int sample_rate);

  // Specify the training graph is static.
  void set_static_graph();

  // Delay all reduce to be after all gradients' calculation is complete.
  void delay_all_reduce();

  // Weak reference to associated DDP logger. The reference is weak to avoid
  // refcycle between reducer and logger.
  void set_logger(std::weak_ptr<c10d::Logger> logger);

  // When graph is not explicitly set by user as static and has unused
  // parameters, this will return whether the graph has been static until the
  // current iteration, which means unused params set has not changed.
  bool ddp_graph_static();

 protected:
  // Forward declaration.
  struct Bucket;

  void push_rebuilt_params(const size_t& index);

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  mutable std::mutex mutex_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  const std::vector<std::vector<at::Tensor>> replicas_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  const c10::intrusive_ptr<::c10d::ProcessGroup> process_group_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::vector<std::vector<bool>> expect_sparse_gradients_;

  std::vector<std::vector<std::shared_ptr<torch::autograd::Node>>>
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
  // during the current iteration or no_sync session if no_sync is on. One
  // tensor for each model replica and each tensor is one-dim int32 tensor of
  // number of parameters. These tensors are marked in autograd_hook to indicate
  // the corresponding param has been used, and get allreduced in the end of
  // backward of current iteration or no_sync session for figuring out the
  // globally unused parameters.
  //
  // local_used_maps_:     CPU tensors for bookkeeping locally used params
  // local_used_maps_dev_: dev tensors for reducing globally unused params
  std::vector<at::Tensor> local_used_maps_;
  std::vector<at::Tensor> local_used_maps_dev_;
  // Indicate that reduction is done and D2H copy is done as well.
  bool local_used_maps_reduced_;

  // Weak pointer to associated DDP logger.
  std::weak_ptr<c10d::Logger> logger_;

  // Work handle for allreduce on local_used_maps_
  c10::intrusive_ptr<c10d::ProcessGroup::Work> local_used_work_;

  void mark_variable_ready_dense(size_t variable_index);

  void mark_variable_ready_sparse(size_t variable_index);

  void mark_variable_ready(size_t variable_index);

  void autograd_hook(size_t index);

  void mark_bucket_ready(size_t bucket_index);

  void finalize_bucket_dense(Bucket& replica);

  void finalize_backward();

  // Returns list of model parameters corresponding to the given bucket.
  // bucket_index is a key to cache after buckets are rebuilt, after which this
  // mapping never changes.
  std::vector<at::Tensor> get_variables_for_bucket(
      size_t bucket_index, const Bucket& bucket) const;

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
      std::is_same<
          GradCallback,
          torch::distributed::autograd::DistAutogradContext::GradCallback>::
          value,
      "");
#endif
  void runGradCallbackForVariable(at::Tensor& variable, GradCallback&& cb);

  // A bucket replica represents [1..N] gradients to be reduced,
  // with the same dtype, on the same device.
  //
  // Batching gradients together before reducing them can result in lower
  // overhead and/or faster time to completion. Only gradients of the same type
  // and on the same device can be batched. The tensor that represents the
  // flattened gradient uses the same type and is placed on the same device.
  // Buckets are filled as the gradients they hold are computed (triggered by
  // autograd hooks). Buckets are reduced in a predetermined order that is
  // identical across processes.
  struct BucketReplica {
    // Flattened (1 dimensional) contents of bucket.
    at::Tensor contents;

    // Views into contents for each grad.  Each view will be created with
    // layout (sizes + strides) matching the grad's expected layout
    // ("Gradient Layout Contract" in torch/csrc/autograd/AccumulateGrad.h).
    // `bucket_views_in[i].copy_(grad)` and
    // `grad.copy_(bucket_views_out[i])`
    // provide convenient ways to move grad data in/out of contents.
    // The reason we keep two states for bucket_views is that if DDP
    // communication hook was registered, `bucket_views_out` could be
    // re-initialized with the value of hook's `future_work`. We still need to
    // keep a separate view reference to replica's original contents for
    // `bucket_views_in[i].copy_(grad)` call.
    std::vector<at::Tensor> bucket_views_in;
    std::vector<at::Tensor> bucket_views_out;

    // Variables that contribute to this bucket replica. Use refcounted value
    // here so that we can easily unflatten the bucket contents into the
    // participating variables after reduction has completed.
    std::vector<at::Tensor> variables;

    // Per-variable offset/length into the flat bucket contents tensor and grad
    // bucket.
    std::vector<size_t> offsets;
    std::vector<size_t> lengths;

    // Per-variable sizes into the grad bucekt.
    std::vector<c10::IntArrayRef> sizes_vec;

    // Number of tensors to be added before this bucket is complete.
    // This is reset to `variables.size()` every iteration.
    size_t pending;

    // TODO(@pietern)
    // Memory copies from gradient tensors into the bucket are potentially
    // done on different CUDA streams. We record an event for every copy
    // so that we can synchronize with them prior to kicking off the reduction.
    // std::vector<at::cuda::CUDAEvent> events;
  };

  // This function is called inside `initialize_buckets`, it initializes both
  // bucket_views_in and bucket_views_out into the contents tensor for each
  // variable's grad. Views serve as entry points to copy_ each grad's data
  // in/out of the flat contents tensor.
  void initialize_bucket_views(BucketReplica& replica, at::Tensor& contents);

  // This function is called inside `finalize_backward`, it happens only if
  // DDP communication hook was registered to recreate just bucket_views_out
  // with the result of `future_work`.
  void populate_bucket_views_out(BucketReplica& replica, at::Tensor& tensor);

  // If gradient_as_bucket_view_ is false, after allreduce buckets,
  // copy bucket results back to grads.
  void copy_bucket_to_grad(
      at::Tensor& variable,
      Reducer::BucketReplica& replica,
      size_t intra_bucket_index,
      bool global_unused);
  // Check layout of grad and bucket_view before copying the grad to bucket.
  void check_grad_layout(const at::Tensor& grad, const at::Tensor& bucket_view);

  // A bucket holds N bucket replicas (1 per model replica).
  //
  // If every bucket in this struct is ready, the reduction can be kicked off.
  // One bucket per replica. Reduction is kicked off when every bucket is ready.
  //
  struct Bucket {
    std::vector<BucketReplica> replicas;

    // Global indices of participating variables in the bucket
    std::vector<size_t> variable_indices;

    // Number of replicas to be marked done before this bucket is ready.
    size_t pending;

    // Keep future work handle around DDP comm hook.
    // If no hook is registered, a temporary vanilla allreduce hook will be
    // used.
    c10::intrusive_ptr<at::ivalue::Future> future_work;

    // If this bucket should expect a single sparse gradient.
    // Implies: replicas[i].variables.size() == 1.
    bool expect_sparse_gradient = false;
    // "Limit" of cumulative parameter sizes that this bucket manages. It is
    // actually a soft limit because we don't shard parameters across buckets
    // so a single parameter may push it over the cap.
    size_t bucket_size_limit;
  };

  std::vector<Bucket> buckets_;

  // A variable locator locates a particular variable in the bucket
  // structure. The `bucket_index` field points to the bucket in the `buckets_`
  // vector. The `intra_bucket_index` field points to the index of the variable
  // in any of the vector fields in the bucket replica.
  struct VariableLocator {
    // Index into the `buckets_` variable.
    size_t bucket_index;
    // Index of parameter in single bucket replica.
    size_t intra_bucket_index;

    VariableLocator() = default;

    VariableLocator(size_t bucket_index_, size_t intra_bucket_index_) {
      bucket_index = bucket_index_;
      intra_bucket_index = intra_bucket_index_;
    }
  };

  // Map the index of a variable to its location in the bucket structure.
  std::vector<VariableLocator> variable_locators_;

  // track the number of iterations to synchronize grads in training so far.
  long num_iterations_;
  // track the number of buckets that have been ready for
  // communication calls like allReduce or communication hooks.
  int num_buckets_ready_;

  // Timing information.
  int64_t backward_compute_start_time_ = -1;
  std::unique_ptr<Timer> timer_;

  // We collect the relative timestamp of every gradient being ready
  // when executing autograd. This can be used to derive a timeline of
  // the point in time buckets were ready, or ideal bucket assignment/ordering.
  std::vector<std::vector<int64_t>> backward_stats_;

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
    c10::intrusive_ptr<c10d::ProcessGroup::Work> workHandle;
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
  // Debug level setting. It is parsed once when Reducer is constructed, and
  // remains the same across a single invocation of DDP training.
  DistributedDebugLevel ddp_debug_level_;
  // Mapping of variable index to fully qualified name of model to notify users
  // about errors when certain parameters do not get gradient.
  std::unordered_map<size_t, std::string> param_names_;
  // Variable indices stored sequentially in order of when the gradient is ready
  // for the current backwards pass.
  std::vector<int> grad_ready_order_indices_;
  // Bytes capacity of first bucket, can be configured by user
  int64_t first_bucket_bytes_cap_;
  // Per iteration set of parameter indices that have been marked ready.
  std::unordered_set<size_t> perIterationReadyParams_;
  // Retrieves parameter names that have not been marked as ready as part of
  // previous iteration.
  std::vector<std::string> getUnmarkedParamsForIteration();
  // Retrives parameter indices that have not been marked as ready as part of
  // previous iteration.
  std::vector<size_t> getUnmarkedParamIndicesForIteration();
  // Raises appropriate error if mark_variable_ready is called on the same
  // variable twice, which is unexpected.
  void checkAndRaiseMarkedTwiceError(size_t curVariableIndex);
  // Retrieves parameter corresponding to the given VariableIndex.
  at::Tensor& get_param_from_index(size_t index);

  // Cached bucket index to model parameter mapping. Populated after buckets
  // are rebuilt after which this mapping is static.
  mutable std::unordered_map<size_t, std::vector<at::Tensor>> cached_variables_for_bucket_;

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
    const c10::optional<std::weak_ptr<c10d::Logger>>& logger = {});

// Verify models across all processes are the same as model on rank 0 with
// respect to no. of params and matching dtype/size/layout.
TORCH_API void verify_replica0_across_processes(
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    const std::vector<std::vector<at::Tensor>>& model_replicas,
    const c10::optional<std::weak_ptr<c10d::Logger>>& logger);
} // namespace c10d
