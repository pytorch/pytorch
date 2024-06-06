#include <torch/csrc/distributed/c10d/reducer.hpp>

#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/default_comm_hooks.hpp>

#include <functional>

#include <c10/core/DeviceGuard.h>
#include <c10/core/ScalarType.h>
#include <c10/core/StreamGuard.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <c10/util/hash.h>
#include <c10/util/irange.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function_hook.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/autograd/utils/grad_layout_contract.h>
#include <torch/csrc/autograd/utils/lambda_post_hook.h>
#include <torch/csrc/distributed/c10d/comm.hpp>
#include <torch/csrc/distributed/c10d/logger.hpp>
#include <utility>

namespace c10d {
namespace {

constexpr int kUnsetDivFactor = -1;

// Macro that wraps TORCH_CHECK with DDP logging.
#define REDUCER_CHECK(cond, logger_, ...)             \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {               \
    if (!logger_.expired()) {                         \
      logger_.lock()->set_error_and_log(__VA_ARGS__); \
    }                                                 \
    TORCH_CHECK(false, ##__VA_ARGS__);                \
  }

} // namespace

C10_DEFINE_TYPED_REGISTRY( // NOLINT
    TimerRegistry,
    c10::DeviceType,
    Timer,
    std::unique_ptr,
    c10::Device);

namespace {

class CpuTimer : public Timer {
 public:
  explicit CpuTimer(c10::Device /* unused */) {}

  std::optional<int64_t> measureDifference(Event start, Event end) override {
    int64_t start_time = getTimeRef(start);
    int64_t end_time = getTimeRef(end);
    // If cpu_end_time is not recorded in this iteration,
    // avg_time will return invalid value.
    // For some cases like DDP runs on non-sync mode, backward compute
    // end time can not be recorded in this iteration and thus can not
    // calculate the valid avg_time.
    // In this case, skip calculating the avg_time and return.
    if (end_time < start_time) {
      return c10::nullopt;
    }
    return end_time - start_time;
  }
};

C10_REGISTER_TYPED_CLASS(TimerRegistry, c10::kCPU, CpuTimer);

std::vector<at::Tensor> extractTensors(const c10::IValue& result) {
  if (result.isPyObject()) {
    return result.toPyObjectHolder()->extractTensors();
  }
  TORCH_INTERNAL_ASSERT(
      result.isTensor() || result.isTensorList(),
      "expected the hook result is either a Tensor or a TensorList found ",
      result.tagKind());

  if (result.isTensor()) {
    return {result.toTensor()};
  }

  return result.toTensorVector();
}

} // namespace

Reducer::Reducer(
    std::vector<at::Tensor> params,
    std::vector<std::vector<size_t>> bucket_indices,
    const std::vector<size_t>& per_bucket_size_limits,
    c10::intrusive_ptr<c10d::ProcessGroup> process_group,
    std::vector<bool> expect_sparse_gradients,
    int64_t bucket_bytes_cap,
    bool find_unused_parameters,
    bool gradient_as_bucket_view,
    std::unordered_map<size_t, std::string> param_names,
    int64_t first_bucket_bytes_cap)
    : params_(std::move(params)),
      process_group_(std::move(process_group)),
      expect_sparse_gradients_(std::move(expect_sparse_gradients)),
      expect_autograd_hooks_(false),
      require_finalize_(false),
      next_bucket_(0),
      has_marked_unused_parameters_(false),
      find_unused_parameters_(find_unused_parameters),
      gradient_as_bucket_view_(gradient_as_bucket_view),
      local_used_map_reduced_(false),
      num_iterations_(0),
      num_bwd_calls_(0),
      first_autograd_hook_called_(false),
      num_buckets_ready_(0),
      has_rebuilt_bucket_(false),
      bucket_bytes_cap_(bucket_bytes_cap),
      div_factor_(kUnsetDivFactor),
      static_graph_(false),
      comm_hook_(nullptr),
      ddp_debug_level_(debug_level()),
      param_names_(std::move(param_names)),
      first_bucket_bytes_cap_(first_bucket_bytes_cap) {
  C10_LOG_API_USAGE_ONCE("torch.distributed.ddp.reducer");
  TORCH_INTERNAL_ASSERT(!params_.empty(), "Expected at least one parameter.");

  if (ddp_debug_level_ != c10d::DebugLevel::Off) {
    LOG(INFO) << "Reducer initialized with bucket_bytes_cap: "
              << bucket_bytes_cap_
              << " first_bucket_bytes_cap: " << first_bucket_bytes_cap;
  }
  // Check whether the module is multi_device_module
  {
    std::set<int> unique_devices;
    for (const auto& v : params_) {
      auto device_idx = int(v.device().index());
      if (unique_devices.find(device_idx) == unique_devices.end()) {
        unique_devices.insert(device_idx);
        if (unique_devices.size() > 1) {
          is_multi_device_module_ = true;
          break;
        }
      }
    }
  }

  // For CUDA, record events only for single device module.
  c10::Device device = params_[0].device();
  if (!(device.is_cuda() && is_multi_device_module_)) {
    timer_ = TimerRegistry()->Create(device.type(), device);
  }

  // If `expect_sparse_gradients` is not specified, initialize it such that
  // we do not expect sparse gradients for any parameter.
  if (expect_sparse_gradients_.empty()) {
    expect_sparse_gradients_ = std::vector<bool>(params_.size(), false);
  }
  TORCH_INTERNAL_ASSERT(expect_sparse_gradients_.size() == params_.size());

  // Initialize variable bucketing.
  // This can be reinitialized later after capturing runtime information.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    initialize_buckets(std::move(bucket_indices));
  }

  // All variables are expected to have their `grad_fn` set to the gradient
  // accumulation function (since they are leafs in the autograd graph).
  // We store pointers to these functions such that we can check if they are
  // used in an autograd pass. If they are not, we know their grad tensors
  // can be marked as ready for reduction.
  {
    const auto variable_count = params_.size();
    grad_accumulators_.resize(variable_count);
    for (const auto variable_index : c10::irange(variable_count)) {
      auto& variable = params_[variable_index];

      // The gradient accumulator function is lazily initialized once.
      // Therefore we can use its presence in the autograd graph as
      // evidence that the parameter has participated in an iteration.
      auto grad_accumulator = torch::autograd::impl::grad_accumulator(variable);

#ifndef _WIN32
      using torch::distributed::autograd::ThreadLocalDistAutogradContext;
#endif
      // Hook to execute after the gradient accumulator has executed.
      hooks_.emplace_back(
          grad_accumulator->add_post_hook(
              std::make_unique<torch::autograd::utils::LambdaPostHook>(
                  [this, variable_index](
                      const torch::autograd::variable_list& outputs,
                      const torch::autograd::variable_list& /* unused */) {
#ifndef _WIN32
                    this->rpc_context_.set(
                        ThreadLocalDistAutogradContext::getContextPtr());
#endif
                    this->autograd_hook(variable_index);
                    return outputs;
                  },
                  [=](torch::autograd::CompiledNodeArgs& args) {
                    // Make post_hook an noop if compiled_autograds is enabled.
                  })),
          grad_accumulator);

      // Map raw function pointer to parameter index.
      // This is used later on when the autograd graph is traversed
      // to check for parameters for which no gradient is computed, if
      // find_unused_parameters=True.
      // Note that the mapping of gradient accumulator to variable should be
      // one to one as we deduplicate shared parameters before constructing
      // Reducer.
      if (find_unused_parameters_) {
        gradAccToVariableMap_[grad_accumulator.get()] = variable_index;
      }

      numGradHooksTriggeredMap_[variable_index] = 0;

      // The gradient accumulator is stored as weak_ptr in the autograd
      // metadata of the variable, so we have to keep it alive here for
      // the raw pointer to be valid.
      REDUCER_CHECK(
          grad_accumulators_[variable_index] == nullptr,
          logger_,
          c10::str(
              "Reducer tried to register duplicate grad accumulator for variable ",
              variable_index));

      grad_accumulators_[variable_index] = std::move(grad_accumulator);
    }
  }

  // Initialize backward stats vector.
  {
    const auto variable_count = params_.size();
    backward_stats_.resize(variable_count);
  }

  // See Note [Skip allreducing local_used_map_dev]
  if (find_unused_parameters_) {
    initialize_local_used_map();
  }
}

// Note [Skip allreducing local_used_map_dev]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// If find_unused_parameters_ is set to false, there is no need to allreduce
// local_used_map_dev_, because all parameters will be reduced anyway.
// Therefore, we can avoid allocating memory for local_used_map and
// local_used_map_dev_ if find_unused_parameters_ is false.

// Note [DDP Communication Hook]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// If DDP communication hook is not registered, the reducer reduces the buckets
// by just calling allreduce. If registered, it calls the hook and uses future
// work handle. If registered, reducer also skips dividing grads by world size.
// The reason for this is that the communication hook is expected to completely
// override how we perform communication and the user should have complete
// control over how the grads are handled.
//
// DDP communication hook is an enhancement that provides a hook which can be
// used to override how DDP communicates gradients across ranks, this can be
// used for algorithms like Gradient Compression/GossipGrad. This hook can be
// registered from Python API using `register_comm_hook`. `PythonCommHook`
// enables registering a Python hook and is a subclass of `CommHookInterface`.
// Additionally, there are also some built-in C++ hook implementations that can
// be specified by calling `register_builtin_comm_hook` from Python API.

Reducer::~Reducer() noexcept(false) {
  remove_autograd_hooks();
}

bool Reducer::dynamic_graph_find_unused() {
  return !static_graph_ && find_unused_parameters_;
}

bool Reducer::static_graph_first_iteration() {
  return static_graph_ && num_bwd_calls_ == 1;
}

bool Reducer::static_graph_after_first_iteration() {
  return static_graph_ && num_bwd_calls_ > 1;
}

bool Reducer::ddp_graph_static() {
  std::lock_guard<std::mutex> lock(mutex_);
  return ddp_graph_static_;
}

void Reducer::initialize_local_used_map() {
  const auto variable_count = params_.size();
  at::TensorOptions options;
  options = options.dtype(at::kInt);

  // Deliberately don't pin the memory even if local_used_map_dev_ will
  // be cuda. See Note [local_used_map_ -> local_used_map_dev copying]
  local_used_map_ = at::zeros({static_cast<long>(variable_count)}, options);

  // This tensor needs to be on the same device as the replica params because
  // backend such as NCCL may not support CPU tensors, and hence it might not
  // work if we always put it on CPU. The dist backend for MTIA doesn't support
  // int32 allreduce for now, so it has to be placed on CPU.
  options = options.device(
      (params_[0].is_mtia()) ? c10::Device(c10::DeviceType::CPU)
                             : params_[0].device());
  local_used_map_dev_ = at::empty({static_cast<long>(variable_count)}, options);
}

void Reducer::check_grad_layout(
    const at::Tensor& grad,
    const at::Tensor& bucket_view) {
  // Ensure that the gradient type matches the bucket type, or mixed precision
  // type if we are training with mixed precision.
  auto type = mixed_precision_param_dtype_
      ? *mixed_precision_param_dtype_
      : bucket_view.options().dtype().toScalarType();
  REDUCER_CHECK(
      grad.options().dtype().toScalarType() == type,
      logger_,
      c10::str(
          "Expected ", type, ", got ", grad.options().dtype().toScalarType()));

  TORCH_INTERNAL_ASSERT(grad.device() == bucket_view.device());
  TORCH_INTERNAL_ASSERT(grad.numel() == bucket_view.numel());
  // AccumulateGrad doesn't HAVE to obey the grad layout contract.
  // The penalty for disobedience is reduced performance, not numerical
  // death. Warnings here help diagnose poor DDP performance.
  if (grad.strides() != bucket_view.strides()) {
    TORCH_WARN_ONCE(
        "Grad strides do not match bucket view strides. "
        "This may indicate grad was not created according to the "
        "gradient layout contract, or that the param's strides "
        "changed since DDP was constructed.  This is not an error, "
        "but may impair performance.\n"
        "grad.sizes() = ",
        grad.sizes(),
        ", strides() = ",
        grad.strides(),
        "\n",
        "bucket_view.sizes() = ",
        bucket_view.sizes(),
        ", strides() = ",
        bucket_view.strides());
  }
  if (!gradient_as_bucket_view_) {
    TORCH_INTERNAL_ASSERT(!grad.is_alias_of(bucket_view));
  }
}

void Reducer::mark_variable_ready_dense(size_t variable_index) {
  const auto& bucket_index = variable_locators_[variable_index];
  auto& bucket = buckets_[bucket_index.bucket_index];
  auto& variable = bucket.variables[bucket_index.intra_bucket_index];
  auto& bucket_view = bucket.bucket_views_in[bucket_index.intra_bucket_index];

  // Copy the contents of the gradient tensor to the corresponding part of the
  // bucket's flattened gradient tensor.
  // If the gradient is not set, we assume it wasn't computed as part of the
  // current backwards pass, and we zero the part of the bucket it would
  // otherwise hold.
  runGradCallbackForVariable(variable, [&](auto& grad) {
    if (grad.defined()) {
      this->check_grad_layout(grad, bucket_view);
      // When gradient_as_bucket_view_ is false, or even when
      // gradient_as_bucket_view_ is true, in rare cases users may set grad to
      // be None after every iteration. In these cases, grad and bucket_view are
      // pointing to different storages and thus need to copy grads to
      // bucket_view. If gradient_as_bucket_view_ is set as true, let grad point
      // to bucket_view. If grad has already been set as views of buckets in
      // previous iterations, no copy is needed.
      if (!grad.is_alias_of(bucket_view)) {
        if (comm_hook_ == nullptr) {
          auto wrapped =
              at::native::wrapped_scalar_tensor(double(1.) / div_factor_);
          if (!grad.requires_grad()) {
            // Divides while copying into the bucket view to save one scan over
            // all the input parameters.
            RECORD_FUNCTION(
                "torch::distributed::reducer::mul_out",
                std::vector<c10::IValue>({bucket_view}))
            at::mul_out(bucket_view, grad, wrapped);
          } else {
            // If DDP is running with create_graph=True, gradients require_grad
            // themselves in order to compute higher order derivatives. However,
            // DDP will not sync up these gradients currently (see
            // https://github.com/pytorch/pytorch/issues/63812).
            C10_LOG_EVERY_N(WARNING, 1000)
                << "Using DistributedDataParallel with create_graph=True "
                << " is not well-supported. The higher-order gradient will "
                << " not be synchronized across ranks, and backpropagation "
                << " through all_reduce operations will not occur. If you require "
                << " DDP to work with higher-order gradients for your use case, "
                << " please ping https://github.com/pytorch/pytorch/issues/63929";
            auto div_result = at::mul(grad, wrapped);
            RECORD_FUNCTION(
                "torch::distributed::reducer::copy_",
                std::vector<c10::IValue>({bucket_view}))
            bucket_view.copy_(div_result);
          }
        } else {
          RECORD_FUNCTION(
              "torch::distributed::reducer::copy_",
              std::vector<c10::IValue>({bucket_view}))
          bucket_view.copy_(grad);
        }

        if (gradient_as_bucket_view_) {
          // Let grad point to bucket_view buffer.
          grad = bucket_view;
          // The grad is modified and need to be written back.
          return true;
        }
      } else {
        // If grad and bucket view point to the same storage, no need to copy.
        if (comm_hook_ == nullptr) {
          bucket_view.div_(div_factor_);
        }
      }
    } else {
      // Gradient is undefined. When find_unused_parameters=True, ensure it is
      // not marked as locally used, otherwise we will be allreducing zero's
      // instead of not touching .grad field of parameter.
      if (this->dynamic_graph_find_unused() ||
          this->static_graph_first_iteration()) {
        REDUCER_CHECK(
            local_used_map_[variable_index].item<int>() == 0,
            logger_,
            "Encountered gradient which is undefined, but still allreduced by "
            "DDP reducer. This indicates a bug in DDP implementation, please "
            "report a bug with a repro to PyTorch.");
      }
      bucket_view.zero_();
    }
    // The grad is not modified and doesn't need to be written back.
    return false;
  });
}

void Reducer::mark_variable_ready_sparse(size_t variable_index) {
  const auto& bucket_index = variable_locators_[variable_index];
  auto& bucket = buckets_[bucket_index.bucket_index];
  auto& variable = bucket.variables[bucket_index.intra_bucket_index];

  runGradCallbackForVariable(variable, [&](auto& grad) {
    REDUCER_CHECK(
        grad.defined(), logger_, "Expected sparse gradient to be defined.");
    REDUCER_CHECK(
        grad.options().layout() == c10::kSparse,
        logger_,
        "Expected variable to have sparse gradient.");

    // Copy the indices of sparse metadata
    if (sparse_metadata_) {
      grad = grad.coalesce();
      REDUCER_CHECK(
          !param_names_.empty(), logger_, "No parameter names were found");
      std::string& param_name = param_names_[variable_index];
      auto iter = sparse_metadata_->find(param_name);
      REDUCER_CHECK(
          iter != sparse_metadata_->end(),
          logger_,
          "param: " + param_name + " not found in sparse metadata");
      bucket.sparse_tensor_indices =
          iter->second.to(at::kLong).unsqueeze(0).to(grad.device());
      auto indices = at::searchsorted(
          bucket.sparse_tensor_indices.value(), grad.indices(), false, false);
      // For indices we are using the ones set by sparse_metadata
      grad = at::sparse_coo_tensor(indices, grad.values(), grad.sizes());
    }

    // Sparse tensors cannot be grouped together with other sparse tensors in a
    // single reduction operation like we can for dense tensors. Therefore, the
    // `offsets` and `lengths` vectors in the bucket struct are empty, and
    // there is no pre-existing accumulation tensor.
    // Directly assign the sparse tensor to the `gradients` field.
    bucket.gradients = grad;
    // If no DDP comm hook is registered, the allreduce only sums up the
    // value, and a separate division is required.
    if (comm_hook_ == nullptr) {
      bucket.gradients.div_(div_factor_);
    }
    // The grad is modified in place and needs to be written back.
    return true;
  });
}

std::vector<c10d::GradBucket> Reducer::get_grad_buckets(
    bool return_zero_tensors) const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<c10d::GradBucket> gradBuckets;
  gradBuckets.reserve(buckets_.size());
  for (const auto i : c10::irange(buckets_.size())) {
    auto& bucket = buckets_[i];
    auto variables_for_bucket = get_variables_for_bucket(i, bucket);
    gradBuckets.emplace_back(
        i,
        buckets_.size(),
        return_zero_tensors ? at::zeros_like(bucket.gradients)
                            : bucket.gradients,
        bucket.offsets,
        bucket.lengths,
        bucket.sizes_vec,
        variables_for_bucket,
        c10::nullopt);
  }
  return gradBuckets;
}

void Reducer::set_forward_pass_work_handle(
    c10::intrusive_ptr<c10d::Work> forwardPassWorkHandle,
    bool useStaticWorldSize) {
  std::lock_guard<std::mutex> lock(mutex_);
  forwardPassWorkHandle_.workHandle = std::move(forwardPassWorkHandle);
  forwardPassWorkHandle_.useStaticWorldSize = useStaticWorldSize;
}

at::Tensor Reducer::get_local_used_map_on_device() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return local_used_map_dev_;
}

void Reducer::push_rebuilt_params_for_all_indices() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!should_rebuild_buckets() || !rebuilt_param_indices_.empty()) {
    return;
  }
  const auto variable_count = params_.size();
  for (const auto variable_index : c10::irange(variable_count)) {
    push_rebuilt_params(variable_index);
  }
}

void Reducer::push_rebuilt_params(const size_t& index) {
  rebuilt_params_.push_back(params_[index]);
  rebuilt_param_indices_.push_back(static_cast<int64_t>(index));
}

void Reducer::set_divide_factor() {
  // If it was scheduled, wait on allreduce in forward pass that tells us
  // division factor based on no. of currently participating processes.
  if (div_factor_ == kUnsetDivFactor) {
    div_factor_ = process_group_->getSize();
    auto& workHandle = forwardPassWorkHandle_.workHandle;
    if (workHandle && !forwardPassWorkHandle_.useStaticWorldSize) {
      workHandle->wait();
      // PyProcessGroup::PyWork doesn't expose value, so fetch it from the
      // future
      auto results = extractTensors(workHandle->getFuture()->value());

      // Guard against the results being empty
      TORCH_INTERNAL_ASSERT(!results.empty());
      at::Tensor& res = results.front();
      div_factor_ = res.item().to<int>();
    }
  }
}

// This is called before training and converts the gradients to the dtype they
// should be reduced in.
void Reducer::set_mixed_precision_param_dtype(c10::ScalarType dtype) {
  mixed_precision_param_dtype_ = dtype;
  for (auto& bucket : buckets_) {
    bucket.gradients = bucket.gradients.to(dtype);
  }
}

// Right now delay_all_reduce is only called when static_graph_=true and
// num_iterations_==1.
void Reducer::delay_all_reduce() {
  std::lock_guard<std::mutex> lock(this->mutex_);

  if (should_collect_runtime_stats()) {
    record_backward_compute_end_time();
    record_backward_comm_start_time();
  }

  // launch all reduce local used map
  all_reduce_local_used_map();

  // prepare to set unused_parameters_, if it is static graph,
  // unused_parameters_ will not change after 1st iteration.
  unused_parameters_.clear();

  require_finalize_ = true;
  // copy all gradients to buckets
  for (const auto variable_index : c10::irange(params_.size())) {
    // set unused_parameters_
    if (numGradHooksTriggeredMap_[variable_index] == 0) {
      unused_parameters_.push_back(variable_index);
    }
    set_divide_factor();
    if (expect_sparse_gradients_[variable_index]) {
      mark_variable_ready_sparse(variable_index);
    } else {
      mark_variable_ready_dense(variable_index);
    }
  }

  // To avoid confusion around why static graph is picking up
  // some parameters as unused on a rank vs not, we log
  // unused parameter names for each rank for better
  // debugability when TORCH_DISTRIBUTED_DEBUG is set to
  // INFO or DETAIL
  if (ddp_debug_level_ != c10d::DebugLevel::Off) {
    // construct one string to output
    std::ostringstream unused_params_stream;

    for (const auto& unused_index : unused_parameters_) {
      auto param_name = param_names_.find(unused_index);
      TORCH_INTERNAL_ASSERT(
          param_name != param_names_.end(),
          "Expected to find parameter name from unused parameters map in debug mode.");
      // Add the param_name
      unused_params_stream << "{" << param_name->second << "," << unused_index
                           << "}";
    }

    // Each rank prints out all the unused parameters detected
    if (!unused_parameters_.empty()) {
      LOG(INFO) << "[Rank " << process_group_->getRank() << "]: "
                << "Parameter(s) (in the format of {param_name, index}): "
                << unused_params_stream.str()
                << " is(are) unused during first iteration. Since"
                << " static_graph=True is enabled for DDP, we expect"
                << " this set of unused parameters to remain consistent"
                << " on this rank throughout the training.";
    }
  }

  // launch all reduces for all buckets
  for (auto& bucket : buckets_) {
    all_reduce_bucket(bucket);
  }

  finalize_backward();
}

void Reducer::set_logger(std::weak_ptr<c10d::Logger> logger) {
  logger_ = std::move(logger);
}

// The function `autograd_hook` is called after the gradient for a
// model parameter has been accumulated into its gradient tensor.
// This function is only to be called from the autograd thread.
void Reducer::autograd_hook(size_t index) {
  std::lock_guard<std::mutex> lock(this->mutex_);
  if (!first_autograd_hook_called_) {
    first_autograd_hook_called_ = true;
    num_bwd_calls_++;
  }

  // See Note [Skip allreducing local_used_map_dev]
  if (dynamic_graph_find_unused() || static_graph_first_iteration()) {
    // Since it gets here, this param has been used for this iteration. We want
    // to mark it in local_used_map_. During no_sync session, the same var can
    // be set multiple times, which is OK as does not affect correctness. As
    // long as it is used once during no_sync session, it is marked as used.
    // Only set it as locally used if the grad is defined. Otherwise, hooks can
    // be fired  with undefined grads, such as when not all outputs are used in
    // DDP when computing loss. In this case, we don't want to mark it as
    // locally used to ensure we don't touch the parameter's .grad field.
    auto& variable = get_param_from_index(index);
    runGradCallbackForVariable(variable, [&](auto& grad) {
      if (grad.defined()) {
        local_used_map_[static_cast<int64_t>(index)] = 1;
      }
      // The gradient is never modified.
      return false;
    });
  }

  if (static_graph_first_iteration()) {
    numGradHooksTriggeredMap_[index] += 1;
    return;
  }

  // Ignore if we don't expect to be called.
  // This may be the case if the user wants to accumulate gradients
  // for number of iterations before reducing them.
  if (!expect_autograd_hooks_) {
    return;
  }

  grad_ready_order_indices_.push_back(static_cast<int64_t>(index));

  // If `find_unused_parameters_` is true there may be model parameters that
  // went unused when computing the model output, they won't be part of the
  // autograd graph, and won't receive gradients. These parameters are
  // discovered in the `prepare_for_backward` function and their indexes stored
  // in the `unused_parameters_` vector.
  if (!has_marked_unused_parameters_) {
    has_marked_unused_parameters_ = true;
    for (const auto& unused_index : unused_parameters_) {
      mark_variable_ready(unused_index);
    }
  }

  // Rebuild bucket only if 1) it is the first time to rebuild bucket 2)
  // static_graph_ is true or find_unused_parameters_ is false,
  // 3) this backward pass needs to run allreduce.
  // Here, we just dump tensors and their parameter indices into
  // rebuilt_params_ and rebuilt_param_indices_ based on gradient arriving
  // order, and then at the end of finalize_backward(), buckets will be
  // rebuilt based on rebuilt_params_ and rebuilt_param_indices_, and then
  // will be broadcasted and initialized.
  // If it is static graph, after 1st iteration, check if a variable
  // is ready for communication based on numGradHooksTriggeredMap_.
  if (static_graph_after_first_iteration()) {
    REDUCER_CHECK(
        numGradHooksTriggeredMapPerIteration_[index] > 0,
        logger_,
        "Your training graph has changed in this iteration, ",
        "e.g., one parameter is unused in first iteration, but ",
        "then got used in the second iteration. this is not ",
        "compatible with static_graph set to True.");
    if (--numGradHooksTriggeredMapPerIteration_[index] == 0) {
      if (should_rebuild_buckets()) {
        push_rebuilt_params(index);
      }
      // Finally mark variable for which this function was originally called.
      mark_variable_ready(index);
    }
  } else {
    if (should_rebuild_buckets()) {
      push_rebuilt_params(index);
    }
    // Finally mark variable for which this function was originally called.
    mark_variable_ready(index);
  }
}

void Reducer::all_reduce_local_used_map() {
  // See Note [Skip allreducing local_used_map_dev]
  // H2D from local_used_map_ to local_used_map_dev_
  if (local_used_map_dev_.is_cuda() || local_used_map_dev_.is_privateuseone()) {
    // Note [local_used_map_ -> local_used_map_dev copying]
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // We do async H2D to avoid the blocking overhead. The async copy and
    // allreduce respect the current stream, so will be sequenced
    // correctly.
    //
    // Correct sequencing with respect to host operations is also
    // essential. The H2D copy_ is stream ordered, while the host's
    // changes to local_used_map_ are host ordered. If a large backlog of
    // cuda/privateuseone-stream work pushes the copy_ far into the future, and
    // if no blocking calls occur between now and finalize_backward()** such
    // that finalize_backward() re-zeroes local_used_map_ on the host
    // before the stream executes the copy_, copy_ will read those zeros
    // instead of the values we thought we told it to read here. Copying
    // local_used_map_ to a pinned temporary (which the pinned caching
    // allocator should supply asynchronously) avoids this nasty, rare
    // race condition.
    //
    // ** In the hoped-for case where all params are used, DDP itself
    // won't do any blocking work between now and the re-zeroing, so the
    // danger is real.
    //
    // Defensively ensures local_used_map_tmp is distinct from
    // local_used_map_
    auto local_used_map_tmp = at::native::empty_like(
        local_used_map_,
        c10::optTypeMetaToScalarType(local_used_map_.options().dtype_opt()),
        local_used_map_.options().layout_opt(),
        local_used_map_.options().device_opt(),
        true /* pinned_memory */);
    // Paranoid asserts here because in some workloads, the pinned
    // allocator behaves in a way we don't understand, and may be bugged.
    // See https://github.com/pytorch/pytorch/pull/54474
    TORCH_INTERNAL_ASSERT(local_used_map_tmp.is_pinned());
    TORCH_INTERNAL_ASSERT(
        local_used_map_tmp.data_ptr() != local_used_map_.data_ptr());
    local_used_map_tmp.copy_(local_used_map_);
    local_used_map_dev_.copy_(local_used_map_tmp, true);
  } else if (local_used_map_dev_.is_mtia()) {
    // MTIA probably will have special logic in the future, following code might
    // be changed drastically. Therefore, a new if case is created for MTIA, for
    // now, the implementation is similar to the CUDA/privateuseone one, except
    // for the pin memory step.
    auto local_used_map_tmp = at::native::empty_like(
        local_used_map_,
        c10::optTypeMetaToScalarType(local_used_map_.options().dtype_opt()),
        local_used_map_.options().layout_opt(),
        local_used_map_.options().device_opt());
    local_used_map_tmp.copy_(local_used_map_);
    local_used_map_dev_.copy_(local_used_map_tmp, true);
  } else {
    local_used_map_dev_.copy_(local_used_map_, true);
  }
  std::vector<at::Tensor> temp_local_used_map_dev_vec_ = {local_used_map_dev_};
  local_used_work_ = process_group_->allreduce(temp_local_used_map_dev_vec_);
}

at::Tensor& Reducer::get_param_from_index(size_t index) {
  const auto& bucket_index = variable_locators_[index];
  auto& bucket = buckets_[bucket_index.bucket_index];
  // Cannot simply access variable via `bucket.variables[variable_index]` since
  // return value is used in `runGradCallbackForVariable()` which does not
  // accept const tensors.
  auto& variable = bucket.variables[bucket_index.intra_bucket_index];
  return variable;
}

void Reducer::checkAndRaiseMarkedTwiceError(size_t index) {
  // Something is wrong if all variables contained in this bucket have
  // already been marked as ready.
  // We don't expect the same variable to be marked ready twice.
  bool marked_twice =
      perIterationReadyParams_.find(index) != perIterationReadyParams_.end();

  if (marked_twice) {
    // Report index of param that has been marked twice. In debug mode, also
    // report fully qualified parameter name.
    auto param_name = param_names_.find(index);
    const bool found_param_name = param_name != param_names_.end();
    TORCH_INTERNAL_ASSERT(
        ddp_debug_level_ == c10d::DebugLevel::Off || found_param_name,
        "Expected to find parameter name in debug mode.");
    std::string paramInfo = c10::str(
        "Parameter at index ",
        index,
        found_param_name ? c10::str(" with name ", param_name->second) : "",
        " has been marked as ready twice. This means that multiple autograd engine ",
        " hooks have fired for this particular parameter during this iteration.");
    // param_names_ is empty in debug mode.
    if (!found_param_name) {
      paramInfo += c10::str(
          " You can set the environment variable TORCH_DISTRIBUTED_DEBUG to either",
          " INFO or DETAIL to print parameter names for further debugging.");
    }
    std::string common_error = c10::str(
        "Expected to mark a variable ready only once. ",
        "",
        "This error is caused by one of the following reasons: ",
        "1) Use of a module parameter outside the `forward` function. ",
        "Please make sure model parameters are not shared across multiple ",
        "concurrent forward-backward passes. or try to use _set_static_graph() ",
        "as a workaround if this module graph does not change ",
        "during training loop.",
        "2) Reused parameters in multiple reentrant backward passes. For ",
        "example, if you use multiple `checkpoint` functions to wrap the ",
        "same part of your model, it would result in the same set of ",
        "parameters been used by different reentrant backward passes ",
        "multiple times, and hence marking a variable ready multiple times. ",
        "DDP does not support such use cases in default. You can try to ",
        "use _set_static_graph() as a workaround if your module graph ",
        "does not change over iterations.");

    common_error += c10::str("\n", paramInfo);

    REDUCER_CHECK(
        has_marked_unused_parameters_,
        logger_,
        common_error,
        "3) Incorrect unused parameter detection. The return value of the ",
        "`forward` function is inspected by the distributed data parallel ",
        "wrapper to figure out if any of the module's parameters went ",
        "unused. For unused parameters, DDP would not expect gradients from ",
        "then. However, if an unused parameter becomes part of the autograd ",
        "graph at a later point in time (e.g., in a reentrant backward when ",
        "using `checkpoint`), the gradient will show up unexpectedly. If all ",
        "parameters in the model participate in the backward pass, you can ",
        "disable unused parameter detection by passing the keyword argument ",
        "`find_unused_parameters=False` to ",
        "`torch.nn.parallel.DistributedDataParallel`. If unused parameters ",
        "in the model do not change over iterations, You can try to use ",
        "_set_static_graph() as a workaround if this module graph does not ",
        "change during training loop.");
    REDUCER_CHECK(!has_marked_unused_parameters_, logger_, common_error);
  }
}

void Reducer::mark_variable_ready(size_t variable_index) {
  REDUCER_CHECK(
      variable_index < variable_locators_.size(),
      logger_,
      "Out of range variable index.");

  checkAndRaiseMarkedTwiceError(variable_index);
  perIterationReadyParams_.insert(variable_index);
  backward_stats_[variable_index] =
      current_time_in_nanos() - backward_compute_start_time_;

  // Any time we mark a variable ready (be it in line due to unused parameters,
  // or via an autograd hook), we require a call to the finalize function. If
  // this doesn't happen before the next iteration (or call to
  // `prepare_for_backwards`), we know something is wrong.
  require_finalize_ = true;

  const auto& bucket_index = variable_locators_[variable_index];
  auto& bucket = buckets_[bucket_index.bucket_index];

  set_divide_factor();

  if (bucket.expect_sparse_gradient) {
    mark_variable_ready_sparse(variable_index);
  } else {
    mark_variable_ready_dense(variable_index);
  }

  // TODO(@pietern): Make this work for both CPU/CUDA tensors.
  // When using CPU tensors we don't need to do this.
  // Record event so that we can wait for all of them.
  // auto& event = bucket.events[bucket_index.intra_bucket_index];
  // event.record();

  // Check if this was the final gradient for this bucket.
  if (--bucket.pending == 0) {
    mark_bucket_ready(bucket_index.bucket_index);
  }

  // Run finalizer function and kick off reduction for local_used_map once the
  // final bucket was marked ready.
  if (next_bucket_ == buckets_.size()) {
    if (dynamic_graph_find_unused()) {
      all_reduce_local_used_map();
    }

    torch::autograd::Engine::get_default_engine().queue_callback([this] {
      std::lock_guard<std::mutex> lock(this->mutex_);
      if (should_collect_runtime_stats()) {
        record_backward_compute_end_time();
      }
      // Check that all buckets were completed and had their work kicked off.
      TORCH_INTERNAL_ASSERT(next_bucket_ == buckets_.size());
      if (static_graph_after_first_iteration() && should_rebuild_buckets()) {
        for (const auto& unused_index : unused_parameters_) {
          push_rebuilt_params(unused_index);
        }
      }
      this->finalize_backward();
    });
  }
}

c10::intrusive_ptr<c10::ivalue::Future> Reducer::run_comm_hook(
    GradBucket& grad_bucket) {
  if (comm_hook_ == nullptr) {
    return run_allreduce_hook(grad_bucket);
  } else {
    return comm_hook_->runHook(grad_bucket);
  }
}

c10::intrusive_ptr<c10::ivalue::Future> Reducer::run_allreduce_hook(
    GradBucket& grad_bucket) {
  _AllReduceBySumCommHook allreduce_hook(process_group_);
  return allreduce_hook.runHook(grad_bucket);
}

void Reducer::all_reduce_bucket(Bucket& bucket) {
  auto variables_for_bucket = get_variables_for_bucket(next_bucket_, bucket);
  // TODO(@pietern): Ensure proper synchronization with the CUDA events
  // that recorded copies into this `gradients` tensor. If these copies are
  // executed on non-default streams, the current stream for the device
  // that holds the `gradients` tensor must wait on these events.
  //
  // As long as autograd uses the default stream for every device,
  // these operations are implicitly sequenced, and we don't need to
  // do any extra synchronization here.
  const auto& tensor = bucket.gradients;

  // TODO(@egienvalue): remove special case after view ops are fully
  // supported on MTIA.
  // If the bucket.gradients is on MTIA, bucket.bucket_views_in might not
  // point to the same storage as bucket.gradients due to the special
  // memory layout. It has to explicitly copy the data back to 1-D gradients.
  if (tensor.is_mtia()) {
    for (const auto i : c10::irange(bucket.variables.size())) {
      const auto offset = bucket.offsets[i];
      const auto length = bucket.lengths[i];
      if (!bucket.bucket_views_in[i].is_alias_of(tensor)) {
        tensor
            .narrow(
                0, static_cast<int64_t>(offset), static_cast<int64_t>(length))
            .copy_(bucket.bucket_views_in[i].flatten());
      }
    }
  }

  GradBucket grad_bucket(
      next_bucket_,
      buckets_.size(),
      tensor,
      bucket.offsets,
      bucket.lengths,
      bucket.sizes_vec,
      variables_for_bucket,
      bucket.sparse_tensor_indices);
  bucket.future_work = run_comm_hook(grad_bucket);
}

std::vector<at::Tensor> Reducer::get_variables_for_bucket(
    size_t bucket_index,
    const Bucket& bucket) const {
  // Check if we have cached mapping previously.
  if (has_rebuilt_bucket_ &&
      cached_variables_for_bucket_.find(bucket_index) !=
          cached_variables_for_bucket_.end()) {
    return cached_variables_for_bucket_[bucket_index];
  }
  std::vector<at::Tensor> variables_for_bucket;
  variables_for_bucket.reserve(bucket.variable_indices.size());
  for (const auto& variable_index : bucket.variable_indices) {
    // Grab bucket index where gradient is located using variable_locators_.
    auto& bucket_index_for_variable = variable_locators_[variable_index];
    // Grab the actual model parameter.
    auto& variable =
        bucket.variables[bucket_index_for_variable.intra_bucket_index];
    variables_for_bucket.emplace_back(variable);
  }

  if (has_rebuilt_bucket_) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        cached_variables_for_bucket_.find(bucket_index) ==
        cached_variables_for_bucket_.end());
    cached_variables_for_bucket_.insert(
        {bucket_index, std::move(variables_for_bucket)});
    return cached_variables_for_bucket_[bucket_index];
  } else {
    return variables_for_bucket;
  }
}

// Called when the bucket at the specified index is ready to be reduced.
void Reducer::mark_bucket_ready(size_t bucket_index) {
  TORCH_INTERNAL_ASSERT(bucket_index >= next_bucket_);

  // Buckets are reduced in sequence. Ignore this bucket if
  // it's not its turn to be reduced.
  if (bucket_index > next_bucket_) {
    return;
  }

  // Keep going, until we either:
  // - have kicked off reduction for all buckets, or
  // - found a bucket that's not yet ready for reduction.
  for (; next_bucket_ < buckets_.size() && buckets_[next_bucket_].pending == 0;
       next_bucket_++) {
    num_buckets_ready_++;
    if (num_buckets_ready_ == 1 && should_collect_runtime_stats()) {
      record_backward_comm_start_time();
    }
    auto& bucket = buckets_[next_bucket_];
    all_reduce_bucket(bucket);
  }
}

void Reducer::install_futures(
    c10::List<c10::intrusive_ptr<c10::ivalue::Future>> futs) {
  // Append instead of overwrite so that this method can be called multiple
  // times in one iteration.
  if (!installed_futures_) {
    installed_futures_ = std::move(futs);
  } else {
    installed_futures_->append(futs);
  }
}

void Reducer::initialize_buckets(
    std::vector<std::vector<size_t>> bucket_indices) {
  // If initialize_buckets is called inside DDP constructor, then
  // it does not matter rpc context ptr is nullptr or not, as grad
  // will not be mutated.
  // If initialize_buckets is called during training loop, e.g, inside
  // rebuild_buckets(), since grad could be mutated and be pointed to
  // bucket_view, then it needs to check rpc context ptr is nullptr or not,
  // If rpc context ptr is nullptr, mutate variable.grad(); otherwise,
  // mutate grad in rpc context.
#ifndef _WIN32
  using torch::distributed::autograd::ThreadLocalDistAutogradContext;
  this->rpc_context_.set(ThreadLocalDistAutogradContext::getContextPtr());
#endif

  // This shouldn't be called if we're expecting autograd hooks to fire.
  REDUCER_CHECK(
      !expect_autograd_hooks_,
      logger_,
      "`initialize_buckets` must NOT be called during autograd execution.");

  // Clear current bucket assignment.
  buckets_.clear();
  variable_locators_.clear();

  // Ensure we have a bucket index for every variable.
  variable_locators_.resize(params_.size());

  // Iterate over buckets.
  const auto bucket_count = bucket_indices.size();
  buckets_.reserve(bucket_count);
  for (const auto bucket_index : c10::irange(bucket_count)) {
    Bucket bucket;

    // TODO(@pietern): Validate indices.
    // Must be non-empty, unique, and unique across buckets.
    REDUCER_CHECK(
        !bucket_indices[bucket_index].empty(),
        logger_,
        "Empty bucket specified.");

    // Variables that expect sparse gradients must have their own bucket.
    if (bucket_indices[bucket_index].size() == 1) {
      const auto variable_index = bucket_indices[bucket_index].front();
      bucket.expect_sparse_gradient = expect_sparse_gradients_[variable_index];
    } else {
      for (const auto variable_index : bucket_indices[bucket_index]) {
        REDUCER_CHECK(
            !expect_sparse_gradients_[variable_index],
            logger_,
            "Buckets with more than one variable cannot include variables ",
            "that expect a sparse gradient.");
      }
    }

    if (bucket.expect_sparse_gradient) {
      const auto variable_index = bucket_indices[bucket_index].front();
      const auto& variable = params_[variable_index];
      TORCH_INTERNAL_ASSERT(bucket_indices[bucket_index].size() == 1);
      bucket.variables = {variable};
    } else {
      at::TensorOptions options;
      // The start index of the variable in the flattened tensor.
      size_t offset = 0;

      // Reserve enough space for the per-variable fields stored in the bucket
      // for efficiency.
      const size_t num_variables = bucket_indices[bucket_index].size();
      bucket.variables.reserve(num_variables);
      bucket.offsets.reserve(num_variables);
      bucket.lengths.reserve(num_variables);
      bucket.sizes_vec.reserve(num_variables);

      // Iterate over bucket variables.
      for (const auto variable_index : bucket_indices[bucket_index]) {
        TORCH_INTERNAL_ASSERT(
            variable_index < params_.size(),
            "Out of range variable index specified.");
        const auto& variable = params_[variable_index];
        if (!options.has_device()) {
          options = options.device(variable.device());
        } else {
          REDUCER_CHECK(
              variable.device() == options.device(),
              logger_,
              "All parameters in a bucket must be ",
              "placed on the same device.");
        }
        if (!options.has_dtype()) {
          options = options.dtype(variable.dtype());
        } else {
          REDUCER_CHECK(
              variable.dtype() == options.dtype(),
              logger_,
              "All parameters in a bucket must have the same dtype.");
        }
        const auto length = variable.numel();
        bucket.variables.push_back(variable);
        bucket.offsets.push_back(offset);
        bucket.lengths.push_back(length);
        bucket.sizes_vec.push_back(variable.sizes());
        offset += length;
      }

      // Allocate the bucket's flattened `gradients` tensor.
      // Make gradient type in the reduced precision if mixed precision is
      // enabled. This ensures that the type is correct when e.g. rebuilding
      // buckets.
      if (mixed_precision_param_dtype_) {
        options = options.dtype(*mixed_precision_param_dtype_);
      }
      bucket.gradients = at::empty({static_cast<long>(offset)}, options);

      // Note:  "Gradient Layout Contract"
      //
      // Here, create views into the `gradients` tensor for each variable's
      // grad. Views serve as entry points to `copy_()` each grad's data in/out
      // of the flattened `gradients` tensor.
      //
      // Gradients may have dense memory but non-row-major-contiguous strides
      // (e.g. channels_last or channels_last_3d). For coalesced accesses
      // during copy_s, it's beneficial for each view's layout to match its
      // grad's layout.
      //
      // Specifically, we expect torch/csrc/autograd/functions/accumulate_grad.h
      // produces grads that obey the "Gradient Layout Contract":
      //   (1) if variable.is_non_overlapping_and_dense(), the stashed grad's
      //       strides match variable.
      //   (2) else, stashed grad is rowmajor contiguous.
      // and create views to match.
      //
      // If AccumulateGrad breaks the contract, and produces a grad with an
      // unexpected layout, performance will degrade due to poor memory access
      // patterns when copy_ing grad data in and out of its bucket view.
      // However, numerics remain correct, because the bucket view is the same
      // on either end of the raw allreduce.  bucket_view_in.copy(grad)
      // tranposes
      // (+ densifies) to the bucket view's layout, the data is allreduced,
      // then grad.copy_(bucket_view_out) transposes it back to grad's layout.
      //
      // The only way the numerics can go haywire is if the bucket views
      // themselves have different layouts across processes.
      // Bucket views' sizes and strides are set based on param layouts, using
      // the same logic that (we expect) AccumulateGrad uses for their grads.
      // Therefore, the only way a bucket view could have different layouts in
      // different processes is if its param has a different layout in
      // different processes. We can check that param layouts match across
      // processes in Reducer's constructor by allreducing some metadata.
      // Checking just once won't catch if someone messes with
      // param layouts over time, but not messing with params after DDP
      // construction is already a documented constraint.
      initialize_bucket_views(bucket);
    }

    // Map participating variables to this bucket.
    size_t intra_bucket_index = 0;
    for (const auto variable_index : bucket_indices[bucket_index]) {
      TORCH_INTERNAL_ASSERT(
          variable_index < variable_locators_.size(),
          "Out of range variable index specified.");
      variable_locators_[variable_index] =
          VariableLocator(bucket_index, intra_bucket_index++);
    }
    bucket.variable_indices = std::move(bucket_indices[bucket_index]);

    buckets_.push_back(std::move(bucket));
  }
}

// (see Note:  "Gradient Layout Contract" in initialize_buckets).
void Reducer::initialize_bucket_views(Reducer::Bucket& bucket) {
  const auto& gradients = bucket.gradients;
  for (const auto i : c10::irange(bucket.variables.size())) {
    auto& v = bucket.variables[i];
    const auto offset = bucket.offsets[i];
    const auto length = bucket.lengths[i];
    // TODO(@egienvalue): remove special case after view ops are fully
    // supported on MTIA.
    // In general, on MTIA, due to the special memory layout, it doesn't
    // support as_strided which creates a view tensor and aten::view will
    // create a new tensor on MTIA for now.
    if (v.is_non_overlapping_and_dense() && !v.is_mtia()) {
      // If the param's memory is dense, match its layout, anticipating
      // the autograd engine (AccumulateGrad) will also create gradients
      // matching its layout.
      bucket.bucket_views_in.push_back(
          gradients.as_strided(v.sizes(), v.strides(), offset));
    } else {
      // Fall back to a C-style contiguous view, again anticipating
      // AccumulateGrad will do the same when stashing grads for non-dense
      // params.
      bucket.bucket_views_in.push_back(
          gradients
              .narrow(
                  0, static_cast<int64_t>(offset), static_cast<int64_t>(length))
              .view(v.sizes()));
    }
    // By default `bucket_views_out` and `bucket_views_in` are
    // essentially the same thing.
    bucket.bucket_views_out = bucket.bucket_views_in;

    // If gradient_as_bucket_view_ is set as true, then there are two cases to
    // handle: initialize_bucket_views could be called inside initialize_buckets
    // when rebuild_buckets, if grad has already been defined/calculated in
    // previous iteration, old grad needs to be copied into new bucket_view and
    // let grad point to the new bucket_view, initialize_bucket_views could also
    // be called inside initialize_buckets during construction. Grads are not
    // defined during construction time, in this case, do not let grad point to
    // bucket_view, because grads should be kept as being undefined for globally
    // unused parameters.
    if (gradient_as_bucket_view_) {
      auto& bucket_view = bucket.bucket_views_in.back();
      runGradCallbackForVariable(v, [&](auto& grad) {
        if (grad.defined() && !grad.is_alias_of(bucket_view)) {
          bucket_view.copy_(grad);
          grad = bucket_view;
          // The grad is modified and needs to be written back.
          return true;
        }
        // The grad is not modified and does not need to be written back.
        return false;
      });
    }
  }
}

// (see Note:  "Gradient Layout Contract" in initialize_buckets).
void Reducer::populate_bucket_views_out(
    Reducer::Bucket& bucket,
    at::Tensor& tensor) {
  bucket.bucket_views_out.clear();
  for (const auto i : c10::irange(bucket.variables.size())) {
    const auto& v = bucket.variables[i];
    const auto offset = bucket.offsets[i];
    const auto length = bucket.lengths[i];
    // TODO(@egienvalue): remove special case after view ops are fully
    // supported on MTIA.
    // In general, on MTIA, due to the special memory layout, it doesn't
    // support as_strided which creates a view tensor and aten::view will
    // create a new tensor on MTIA for now.
    if (v.is_non_overlapping_and_dense() && !v.is_mtia()) {
      // If the param's memory is dense, match its layout, anticipating
      // the autograd engine (AccumulateGrad) will also create gradients
      // matching its layout.
      bucket.bucket_views_out.push_back(
          tensor.as_strided(v.sizes(), v.strides(), offset));
    } else {
      // Fall back to a C-style contiguous view, again anticipating
      // AccumulateGrad will do the same when stashing grads for non-dense
      // params.
      bucket.bucket_views_out.push_back(
          tensor
              .narrow(
                  0, static_cast<int64_t>(offset), static_cast<int64_t>(length))
              .view(v.sizes()));
    }
  }
}

void Reducer::prepare_for_forward() {
  std::lock_guard<std::mutex> lock(mutex_);
  num_iterations_++;
  if (should_collect_runtime_stats()) {
    record_forward_compute_start_time();
  }
}

void Reducer::reset_bucket_counting() {
  next_bucket_ = 0;
  // Reset num_buckets_ready_ at the beginning of backward computation
  // in each iteration.
  num_buckets_ready_ = 0;

  for (auto& bucket : buckets_) {
    bucket.pending = bucket.variables.size();
  }

  if (static_graph_) {
    numGradHooksTriggeredMapPerIteration_ = numGradHooksTriggeredMap_;
  }
}

// Traverse the autograd graph starting at the specified output.
// All parameters for which we have a pointer to their gradient accumulation
// functions, but don't show up in the autograd graph will be marked ready for
// for reduction as soon as the first autograd hook is called. This is not
// done immediately because the model output may be ignored, and we only
// want to start performing reductions on `torch.autograd.backward()`.
void Reducer::search_unused_parameters(
    const std::vector<torch::autograd::Variable>& outputs) {
  std::unordered_set<torch::autograd::Node*> seen;
  std::vector<torch::autograd::Node*> queue;

  RECORD_FUNCTION(
      "torch.distributed.ddp.reducer::search_unused_parameters",
      std::vector<c10::IValue>());

  // Seed queue with the grad functions of all outputs.
  for (const auto& output : outputs) {
    const auto& grad_fn = output.grad_fn();
    if (grad_fn) {
      queue.push_back(grad_fn.get());
    }
  }

  // Traverse the autograd graph starting at the specified output.
  while (!queue.empty()) {
    auto fn = queue.back();
    queue.pop_back();
    for (const auto& edge : fn->next_edges()) {
      if (auto next_ptr = edge.function.get()) {
        const bool was_inserted = seen.insert(next_ptr).second;
        if (was_inserted) {
          queue.push_back(next_ptr);
        }
      }
    }
  }

  // Find accumulator functions that don't show up in this graph.
  for (const auto& it : gradAccToVariableMap_) {
    // If the accumulator function is present in the graph, we know
    // a gradient will be computed for the corresponding parameter.
    if (seen.count(it.first) == 0) {
      if (ddp_debug_level_ == c10d::DebugLevel::Detail) {
        const auto param_info = param_names_.find(it.second);
        TORCH_INTERNAL_ASSERT(
            param_info != param_names_.end(),
            "Did not find variable index ",
            it.second,
            " in DDP parameter name mapping!");
        const auto param_name = param_info->second;
        LOG(INFO) << "[Rank " << process_group_->getRank() << "]: "
                  << "Parameter " << param_name << " at index " << it.second
                  << " is marked as unused.";
      }
      unused_parameters_.push_back(it.second);
    }
  }

  // Warn user about unnecessary perf hit if all parameters were used in
  // forward.
  if (unused_parameters_.empty()) {
    TORCH_WARN_ONCE(
        "find_unused_parameters=True was specified in DDP constructor, "
        "but did not find any unused parameters in the forward pass. This flag "
        "results in an extra traversal of the autograd graph every iteration, "
        " which can adversely affect performance. If your model indeed never "
        "has any unused parameters in the forward pass, consider turning this "
        "flag off. Note that this warning may be a false positive if your model "
        "has flow control causing later iterations to have unused parameters.");
  }
  if (!static_graph_ && ddp_graph_static_) {
    if (num_iterations_ > 1) {
      // Graph is still static if the set of unused parameters did not change.
      ddp_graph_static_ =
          prev_iteration_unused_parameters_ == unused_parameters_;

      if (!ddp_graph_static_) {
        // Log graph is not static. Logger takes care of ensuring this is done
        // only once to avoid overhead.
        logger_.lock()->log_if_graph_static(false);
      }
    }
    prev_iteration_unused_parameters_ = unused_parameters_;
  }
}

void Reducer::prepare_for_backward(
    const std::vector<torch::autograd::Variable>& outputs) {
  std::lock_guard<std::mutex> lock(mutex_);

  backward_compute_start_time_ = current_time_in_nanos();
  if (should_collect_runtime_stats()) {
    record_backward_compute_start_time();
  }

  // Reset accounting.
  expect_autograd_hooks_ = true;
  // Clear gradient ready order as it can be different in the next iteration.
  grad_ready_order_indices_.clear();

  reset_bucket_counting();

  // Reset unused parameter accounting.
  has_marked_unused_parameters_ = false;
  // Reset per iteration marked ready parameters.
  perIterationReadyParams_.clear();

  // If static graph is not set, search graph to detect unused parameters.
  // When static graph is set, unused_parameters_ will be detected and will
  // not change after 1st iteration.
  // If static_graph_ = false and find_unused_parameters_ is false,
  // we assume that autograd hooks for ALL variables will be called,
  // and we don't have to search the autograd graph for presence of these hooks.
  if (dynamic_graph_find_unused()) {
    unused_parameters_.clear();
    search_unused_parameters(outputs);
  }
}

void Reducer::copy_bucket_to_grad(
    at::Tensor& variable,
    Reducer::Bucket& bucket,
    size_t intra_bucket_index,
    bool global_unused) {
  const auto& bucket_view = bucket.bucket_views_out[intra_bucket_index];
  runGradCallbackForVariable(variable, [&](auto& grad) {
    // If a parameter is globally unused, we keep its grad untouched.
    if (!global_unused) {
      if (!grad.defined()) {
        // Creates grad according to the "Gradient Layout Contract"
        // (see torch/csrc/autograd/functions/accumulate_grad.h)
        grad =
            torch::autograd::utils::clone_obey_contract(bucket_view, variable);
      } else {
        grad.copy_(bucket_view);
      }
      // The grad is modified and needs to be written back.
      return true;
    }
    // The grad is not modified.
    return false;
  });
}

std::vector<std::string> Reducer::getUnmarkedParamsForIteration() {
  std::vector<std::string> unMarkedParamNames;
  for (const auto& it : param_names_) {
    if (perIterationReadyParams_.find(it.first) ==
        perIterationReadyParams_.end()) {
      unMarkedParamNames.push_back(it.second);
    }
  }
  return unMarkedParamNames;
}

std::vector<size_t> Reducer::getUnmarkedParamIndicesForIteration() {
  std::vector<size_t> unmarked_param_indices;
  const auto variable_count = params_.size();
  for (const auto variable_index : c10::irange(variable_count)) {
    if (perIterationReadyParams_.find(variable_index) ==
        perIterationReadyParams_.end()) {
      unmarked_param_indices.push_back(variable_index);
    }
  }
  return unmarked_param_indices;
}

// A bucket with one or more dense tensors needs to be unflattened.
void Reducer::finalize_bucket_dense(Bucket& bucket) {
  for (const auto intra_bucket_index : c10::irange(bucket.variables.size())) {
    auto& variable = bucket.variables[intra_bucket_index];

    bool global_unused = false;
    // See Note [Skip allreducing local_used_map_dev]
    if (static_graph_ || find_unused_parameters_) {
      // Determine if this param has been used globally or not.
      //
      // If the variable was used locally, it is also used globally and then
      // we don't need to wait for the reduction. Otherwise we lazily wait for
      // the reduction to complete, only when we see a variable that was
      // unused locally. Then we end up delaying the synchronization point
      // that local_used_work_->wait() implies. If we don't have any unused
      // parameters at all, we can skip waiting for the work to complete
      // altogether, and cause negligible performance overhead for models
      // where all parameters are used. Such lazily waiting means minimizing
      // performance impact for the big majority of models where all
      // parameters are always used. Then we only pay the overhead cost if
      // there is indeed a parameter that is locally unused, because we need
      // to check if it's also globally unused.
      int64_t variable_index =
          static_cast<int64_t>(bucket.variable_indices[intra_bucket_index]);
      // Note: global_unused might not be global yet. As we lazily wait for
      // the reduction to complete, it becomes really global only if we get to
      // the point as below where we wait for the reduction work, make D2H
      // copy, and update global_unused with the real global consensus, i.e.
      // local_used_map_reduced_ is true.
      global_unused = local_used_map_[variable_index].item<int>() == 0;
      if (global_unused && !local_used_map_reduced_) {
        // Wait for local_used_map reduction to complete.
        local_used_work_->wait();
        // D2H from local_used_map_dev_ to local_used_map_
        // Blocking copy, if local_used_map_dev_ is cuda
        local_used_map_.copy_(local_used_map_dev_);

        global_unused = local_used_map_[variable_index].item<int>() == 0;
        local_used_map_reduced_ = true;
      }
    }

    if (!gradient_as_bucket_view_) {
      if (optim_in_backward_) {
        // Return early if optimizer has already run.
        runGradCallbackForVariable(variable, [&](auto& grad) { return true; });
      } else {
        RECORD_FUNCTION(
            "torch.distributed.ddp.reducer::copy_bucket_to_grad",
            std::vector<c10::IValue>({variable}));
        copy_bucket_to_grad(
            variable, bucket, intra_bucket_index, global_unused);
      }
    } else {
      const auto& bucket_view_out = bucket.bucket_views_out[intra_bucket_index];
      auto& bucket_view_in = bucket.bucket_views_in[intra_bucket_index];
      // If a communication hook is registered, then `bucket_view_out` stores
      // the allreduced results in a newly allocated tensor, so we copy
      // `bucket_view_out` back to `bucket_view_in` for this gradient.
      if (!bucket_view_in.is_alias_of(bucket_view_out)) {
        bucket_view_in.copy_(bucket_view_out);
      }
      runGradCallbackForVariable(variable, [&](auto& grad) {
        if (optim_in_backward_) {
          // Return early if optimizer has already run.
          return true;
        }
        // If a parameter is globally unused, we keep its grad untouched.
        if (!global_unused) {
          // If grad is globally used but locally unused, let grad point to
          // bucket_view_in
          if (!grad.defined()) {
            grad = bucket_view_in;
          } else {
            if (!grad.is_alias_of(bucket_view_in)) {
              REDUCER_CHECK(
                  false,
                  logger_,
                  "Detected at least one parameter gradient is not the "
                  "expected DDP bucket view with gradient_as_bucket_view=True. "
                  "This may happen (for example) if multiple allreduce hooks "
                  "were registered onto the same parameter. If you hit this error, "
                  "please file an issue with a minimal repro.");
            }
          }
          // The grad is modified and needs to be written back.
          return true;
        }
        // The grad is not modified.
        return false;
      });
    }
  }
}

void Reducer::finalize_backward() {
  // No longer expect autograd hooks to fire after this function returns.
  TORCH_INTERNAL_ASSERT(expect_autograd_hooks_);
  expect_autograd_hooks_ = false;
  // reset for the next iteration
  first_autograd_hook_called_ = false;

  // No longer require call to finalize after this function returns.
  TORCH_INTERNAL_ASSERT(require_finalize_);
  require_finalize_ = false;

  // Wait for asynchronous reduction to complete, and unflatten the bucket's
  // flattened `gradients` tensor.
  for (auto& bucket : buckets_) {
    // See Note [DDP Communication Hook]
    TORCH_INTERNAL_ASSERT(
        bucket.future_work,
        "Expected bucket.future_work not to be null. "
        "This may indicate that communication hook was not properly installed.");
    bucket.future_work->wait();
    auto future_result = comm_hook_ == nullptr
        ? detail::parseCppCommHookResult(bucket.future_work->value())
        : comm_hook_->parseHookResult(bucket.future_work->value());
    if (bucket.expect_sparse_gradient) {
      // sparse metadata is set so the bucket should have sparse_tensor_indices
      if (sparse_metadata_) {
        REDUCER_CHECK(
            bucket.sparse_tensor_indices.value().numel() ==
                bucket.gradients.sizes()[0],
            logger_,
            "Sparse metadata and gradient size mismatch");
        auto sparse_result = at::sparse_coo_tensor(
            bucket.sparse_tensor_indices.value(),
            future_result,
            bucket.gradients.sizes());
        bucket.gradients.copy_(sparse_result);
      } else {
        bucket.gradients.copy_(future_result);
      }
    } else {
      // Reinitialize only `bucket_views_out` with the future_result by
      // following the same logic in `initialize_buckets`.
      populate_bucket_views_out(bucket, future_result);
    }

    // Unset allreduce division factor, as it may change in next backwards pass
    // when running with DDP join mode.
    div_factor_ = kUnsetDivFactor;

    if (!bucket.expect_sparse_gradient) {
      // We don't need to finalize the sparse bucket since the sparse grad and
      // the bucket essentially point to the same storage. As a result, once
      // the allreduce is done, the sparse grads are automatically updated.
      finalize_bucket_dense(bucket);
    }
  }

  if (installed_futures_ != c10::nullopt) {
    c10::collectAll(*installed_futures_)->wait();
    installed_futures_ = c10::nullopt;
  }

  // See Note [Skip allreducing local_used_maps_dev]
  if (dynamic_graph_find_unused() || static_graph_first_iteration()) {
    // Due to the lazy wait, it is possible that reduction of the current
    // iteration is still going when the one for next iteration gets kicked off.
    // For such case, we want to wait explicitly to make sure the reduction does
    // complete before kicking off next one. Otherwise the previous one may
    // interfere, write to the device-side memory and clobber the content of
    // local_unused_maps_dev_.
    if (!local_used_map_reduced_) {
      local_used_work_->wait();
    }
  }

  if (dynamic_graph_find_unused()) {
    // Reset unused parameter accounting.
    // See Note [local_used_map_ -> local_used_map_dev copying]
    local_used_map_.fill_(0);
    local_used_map_reduced_ = false;
  }

  if (should_collect_runtime_stats()) {
    record_backward_comm_end_time();
  }

  sparse_metadata_.reset();
}

void Reducer::runGradCallbackForVariable(
    at::Tensor& variable,
    GradCallback&& cb) {
#ifdef _WIN32
  cb(variable.mutable_grad());
#else
  auto context_ptr = rpc_context_.context_ptr.load();
  if (context_ptr == nullptr) {
    cb(variable.mutable_grad());
  } else {
    // Under distributed autograd
    context_ptr->runGradCallbackForVariable(variable, std::move(cb));
  }
#endif
}

#ifndef _WIN32
void Reducer::RpcContext::set(ContextPtr&& new_context_ptr) {
  // We should set 'new_context_ptr' even if it's nullptr. That means the
  // reducer is under a local backward run.
  const auto new_context_raw_ptr = new_context_ptr.get();
  if (context_ptr.exchange(new_context_raw_ptr) != new_context_raw_ptr) {
    // Set the shared ptr to the context only if it's set first time.
    // All call sites should use the same context ptr.
    // Use an atomic to avoid data race from multiple threads.
    context_ptr_holder = std::move(new_context_ptr);
  }
}
#endif

void Reducer::sync_bucket_indices(
    std::vector<std::vector<size_t>>& bucket_indices) {
  auto num_buckets = bucket_indices.size();
  std::vector<size_t> bucket_sizes;
  bucket_sizes.reserve(num_buckets);
  int64_t total_size = 0;
  for (const auto i : c10::irange(num_buckets)) {
    auto bucket_size = bucket_indices.at(i).size();
    bucket_sizes.push_back(bucket_size);
    total_size += static_cast<int64_t>(bucket_size);
  }

  at::TensorOptions options;
  options = options.dtype(at::kInt);
  options = options.device(params_[0].device());

  // Group indices and num_bucket together into indices_tensor
  // Broadcast this tensor first, as its size is equal among all processes
  auto indices_tensor = at::empty({total_size + 1}, at::kInt);
  auto indices_accessor = indices_tensor.accessor<int, 1>();
  auto indices_accessor_Index = 0;
  for (const auto i : c10::irange(num_buckets)) {
    const auto& bucket_size = bucket_indices.at(i).size();
    for (const auto j : c10::irange(bucket_size)) {
      indices_accessor[indices_accessor_Index++] =
          static_cast<int>(bucket_indices[i][j]);
    }
  }
  indices_accessor[indices_accessor_Index] = static_cast<int>(num_buckets);

  // Copy CPU tensor to device tensor, as the process_group_ could be NCCL and
  // it can only broadcast device tensors.
  auto indices_tensor_device = at::empty({total_size + 1}, options);
  indices_tensor_device.copy_(indices_tensor, /*non_blocking=*/true);
  std::vector<at::Tensor> indices_tensor_list = {indices_tensor_device};
  process_group_->broadcast(indices_tensor_list)->wait();
  indices_tensor.copy_(indices_tensor_list.front(), /*non_blocking=*/false);

  // Update num_buckets after receiving it from rank 0
  num_buckets = indices_accessor[indices_accessor_Index];

  // Broadcast bucket_sizes
  auto bucket_sizes_tensor = at::empty({(int64_t)num_buckets}, at::kInt);
  auto bucket_sizes_accessor = bucket_sizes_tensor.accessor<int, 1>();
  for (const auto i : c10::irange(num_buckets)) {
    // For rank != 0, it is possible that local num buckets bucket_sizes.size()
    // is smaller than broadcasted num_buckets
    bucket_sizes_accessor[i] =
        bucket_sizes.at(std::min(i, (bucket_sizes.size() - 1)));
  }
  auto bucket_sizes_tensor_device = at::empty({(int64_t)num_buckets}, options);
  bucket_sizes_tensor_device.copy_(bucket_sizes_tensor, /*non_blocking=*/true);
  std::vector<at::Tensor> bucket_sizes_tensor_list = {
      bucket_sizes_tensor_device};
  process_group_->broadcast(bucket_sizes_tensor_list)->wait();
  bucket_sizes_tensor.copy_(
      bucket_sizes_tensor_list.front(), /*non_blocking=*/false);

  // Clear bucket_indices first, and then update bucket_indices using received
  // num_buckets, bucket_sizes_tensor and indices_tensor from rank 0
  bucket_indices.clear();
  bucket_indices.reserve(num_buckets);
  indices_accessor_Index = 0;
  for (const auto i : c10::irange(num_buckets)) {
    const auto& bucket_size = bucket_sizes_accessor[static_cast<int64_t>(i)];
    std::vector<size_t> bucket;
    bucket.reserve(bucket_size);
    for (const auto j : c10::irange(bucket_size)) {
      (void)j;
      bucket.push_back(indices_accessor[indices_accessor_Index++]);
    }
    bucket_indices.emplace_back(std::move(bucket));
  }
}

bool Reducer::rebuild_buckets() {
  // Ensure reduction for previous backwards pass is finished. If user's model
  // has unused parameters for example, this will raise an error recommending to
  // run with find_unused_parameters=True, instead of the size mismatch
  // exception below.
  std::lock_guard<std::mutex> lock(mutex_);
  ensure_prior_reduction_finished();
  if (!should_rebuild_buckets() || rebuilt_params_.empty()) {
    return false;
  }

  TORCH_INTERNAL_ASSERT(
      rebuilt_params_.size() == rebuilt_param_indices_.size(),
      c10::str(
          "rebuilt parameter tensors size is not same as rebuilt parameter indices size: ",
          rebuilt_params_.size(),
          " versus ",
          rebuilt_param_indices_.size()));
  TORCH_INTERNAL_ASSERT(
      params_.size() == rebuilt_param_indices_.size(),
      c10::str(
          "rebuilt parameter indices size is not same as original model parameters size.",
          "Original model param size is: ",
          params_.size(),
          " versus rebuilt params size of: ",
          rebuilt_param_indices_.size()));
  std::vector<size_t> bucket_size_limits;
  bucket_size_limits.push_back(first_bucket_bytes_cap_);
  bucket_size_limits.push_back(bucket_bytes_cap_);
  auto ddp_set_last_bucket_as_small =
      (getCvarString({"DDP_SET_LAST_BUCKET_CAP"}, "N/A") == "1");

  if (ddp_set_last_bucket_as_small) {
    // Reverse so that first_bucket_bytes_cap_ (smaller bucket) becomes the last
    // bucket. We cannot simply pass in {bucket_bytes_cap_,
    // first_bucket_bytes_cap} as the bucket order as we would immediately
    // advance to the 2nd element after the first bucket, whereas we only want
    // the last bucket to have a smaller size.
    std::reverse(rebuilt_params_.begin(), rebuilt_params_.end());
    std::reverse(rebuilt_param_indices_.begin(), rebuilt_param_indices_.end());
  }
  auto [rebuilt_bucket_indices, per_bucket_size_limits] =
      compute_bucket_assignment_by_size(
          rebuilt_params_,
          bucket_size_limits,
          expect_sparse_gradients_,
          rebuilt_param_indices_,
          logger_);

  if (ddp_set_last_bucket_as_small) {
    // Reverse again because buckets were rebuilt in the opposite of gradient
    // ready order.
    std::reverse(rebuilt_bucket_indices.begin(), rebuilt_bucket_indices.end());
    std::reverse(per_bucket_size_limits.begin(), per_bucket_size_limits.end());
  }

  if (ddp_debug_level_ != c10d::DebugLevel::Off) {
    TORCH_INTERNAL_ASSERT(
        rebuilt_bucket_indices.size() == per_bucket_size_limits.size())
    LOG(INFO) << rebuilt_bucket_indices.size()
              << " buckets rebuilt with size limits: "
              << c10::Join(", ", per_bucket_size_limits) << " bytes.";
  }

  // For rebuilt bucket indices, it needs to be synced across all ranks.
  // Broadcast the newly rebuilt bucket indices from rank 0 in default.
  // After syncing up rebuilt bucket indices, initialize buckets for reducer.
  sync_bucket_indices(rebuilt_bucket_indices);

  has_rebuilt_bucket_ = true;
  rebuilt_params_.clear();
  rebuilt_param_indices_.clear();

  initialize_buckets(std::move(rebuilt_bucket_indices));

  return true;
}

void Reducer::setSparseMetadata(std::map<std::string, at::Tensor>& metadata) {
  sparse_metadata_ =
      std::make_unique<std::map<std::string, at::Tensor>>(metadata);
}

// See Note [DDP Communication Hook]
void Reducer::register_comm_hook(std::unique_ptr<CommHookInterface> iface) {
  REDUCER_CHECK(
      comm_hook_ == nullptr,
      logger_,
      "register_comm_hook or register_builtin_comm_hook can only be called once.");

  comm_hook_ = std::move(iface);
}

// See Note [DDP Communication Hook]
void Reducer::register_builtin_comm_hook(
    c10d::BuiltinCommHookType comm_hook_type) {
  REDUCER_CHECK(
      comm_hook_ == nullptr,
      logger_,
      "register_builtin_comm_hook or register_comm_hook can only be called once.");

  switch (comm_hook_type) {
    case c10d::BuiltinCommHookType::ALLREDUCE:
      comm_hook_ = std::make_unique<c10d::AllReduceCommHook>(process_group_);
      LOG(INFO) << "Built-in communication hook ALLREDUCE is registered.";
      break;
    case c10d::BuiltinCommHookType::FP16_COMPRESS:
      comm_hook_ = std::make_unique<c10d::FP16CompressCommHook>(process_group_);
      LOG(INFO) << "Built-in communication hook FP16_COMPRESS is registered.";
      break;
    default:
      TORCH_WARN_ONCE(
          "Unknown built-in DDP comm hook type is provided. No comm hook will be used.");
  }
}

void Reducer::ensure_prior_reduction_finished() {
  // Check that any prior reduction has finished.
  // The variable `require_finalize_` is true until all gradients
  // have been computed and reduction of all buckets has been kicked off.
  if (require_finalize_) {
    // Collect unmarked parameter indices, additionally, in debug mode retrieve
    // parameter names.
    auto unmarked_param_indices = getUnmarkedParamIndicesForIteration();
    // We should have some unmarked parameter indices, otherwise we would not
    // have run into this error branch.
    TORCH_INTERNAL_ASSERT(!unmarked_param_indices.empty());

    std::string kBaseErrorMsg =
        "Expected to have finished reduction in the prior iteration before "
        "starting a new one. "
        ""
        "This error indicates that your module has parameters that were "
        "not used in producing loss. ";
    std::string kOutputsNotUsedInLossErrorMsg =
        "making sure all "
        "`forward` function outputs participate in calculating loss. ";
    std::string kDDPBugErrorMsg =
        "\nIf you already have done the above, then the distributed "
        "data parallel module wasn't able to locate the output tensors in the "
        "return value of your module's `forward` function. "
        "Please include the loss function and the structure of the return "
        "value of `forward` of your module when reporting this issue (e.g. "
        "list, dict, iterable).";

    if (static_graph_) {
      kBaseErrorMsg =
          "Expected to have finished reduction in the prior iteration before "
          "starting a new one. "
          "This error indicates that your training graph has changed "
          "in this iteration, e.g., one parameter is used in first "
          "iteration, but then got unused in the second iteration. "
          "this is not compatible with static_graph set to True.";
    } else if (!find_unused_parameters_) {
      // Parameters may have been unused in forward pass, or not all outputs
      // were used in producing loss.
      kBaseErrorMsg +=
          "You can enable unused parameter detection by passing the "
          "keyword argument `find_unused_parameters=True` to "
          "`torch.nn.parallel.DistributedDataParallel`, and by \n";
      kBaseErrorMsg += kOutputsNotUsedInLossErrorMsg;
      kBaseErrorMsg += kDDPBugErrorMsg;
    } else {
      // Note that it does not really matter whether unused_parameters_.empty(),
      // since user may have enabled detection but this particular iteration
      // could have used or not used all parameters.
      kBaseErrorMsg +=
          "Since `find_unused_parameters=True` is enabled, this likely "
          " means that not all `forward` outputs participate in computing loss. You can fix this by ";
      kBaseErrorMsg += kOutputsNotUsedInLossErrorMsg;
      kBaseErrorMsg += kDDPBugErrorMsg;
    }

    const std::string unmarked_param_indices_info = c10::str(
        "\n",
        "Parameter indices which did not receive grad for rank ",
        process_group_->getRank(),
        ": ",
        unmarked_param_indices);

    if (ddp_debug_level_ == DebugLevel::Off) {
      // Without debug mode, log unmarked_param_indices, as well as
      // recommendation to use debug mode to print parameter names.
      kBaseErrorMsg += unmarked_param_indices_info;
      kBaseErrorMsg +=
          "\n In addition, you can set the environment variable "
          "TORCH_DISTRIBUTED_DEBUG to either INFO or DETAIL to print out information "
          "about which particular parameters did not receive gradient on this rank "
          "as part of this error";
    } else {
      // Retrieve set of parameter names that did not receive gradient.
      auto unmarkedParams = getUnmarkedParamsForIteration();
      TORCH_INTERNAL_ASSERT(!unmarkedParams.empty());
      for (const auto& s : unmarkedParams) {
        LOG(INFO) << "[Rank " << process_group_->getRank() << "] "
                  << "Parameter: " << s
                  << " did not get gradient in backwards pass.";
      }
      const std::string unmarkedParamInfo = c10::Join(", ", unmarkedParams);
      // In debug mode, log param names and indices that went unused.
      kBaseErrorMsg += c10::str(
          "\n",
          "Parameters which did not receive grad for rank ",
          process_group_->getRank(),
          ": ",
          unmarkedParamInfo);
      kBaseErrorMsg += unmarked_param_indices_info;
    }
    REDUCER_CHECK(false, logger_, kBaseErrorMsg);
  }
}

void Reducer::set_ddp_runtime_logging_sample_rate(int sample_rate) {
  ddp_runtime_logging_sample_rate_ = sample_rate;
}

int Reducer::get_ddp_runtime_logging_sample_rate() {
  return ddp_runtime_logging_sample_rate_;
}

bool Reducer::should_collect_runtime_stats() {
  if (num_iterations_ > 0 &&
      (num_iterations_ <= 10 ||
       num_iterations_ % get_ddp_runtime_logging_sample_rate() == 0)) {
    return true;
  }
  return false;
}

void Reducer::record_forward_compute_start_time() {
  if (timer_) {
    timer_->record(Timer::Event::kForwardStart);
  }
}

void Reducer::record_backward_compute_start_time() {
  if (timer_) {
    timer_->record(Timer::Event::kBackwardComputeStart);
  }
}

void Reducer::record_backward_compute_end_time() {
  if (timer_) {
    timer_->record(Timer::Event::kBackwardComputeEnd);
  }
}

void Reducer::record_backward_comm_start_time() {
  if (timer_) {
    timer_->record(Timer::Event::kBackwardCommStart);
  }
}

void Reducer::record_backward_comm_end_time() {
  if (timer_) {
    timer_->record(Timer::Event::kBackwardCommEnd);
  }
}

void Reducer::set_static_graph() {
  std::lock_guard<std::mutex> lock(mutex_);
  REDUCER_CHECK(
      num_iterations_ == 0,
      logger_,
      "set_static_graph() should be called before training loop starts "
      "and after DistributedDataParallel is constructed.");
  static_graph_ = true;
  // when static_graph_ is set as true, always initialize_local_used_map
  // and detect the global unused parameters in the first iteration.
  initialize_local_used_map();
}

namespace {

// Tensors may be coalesced into buckets. Buckets must contain tensors of
// the same type, on the same device, so a bucket can identified by a
// composite key of a tensor's type identifier and its device.
struct BucketKey {
  BucketKey(c10::ScalarType type, c10::Device device)
      : type(type), device(device) {}

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const*)
  const c10::ScalarType type;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const*)
  const c10::Device device;

  // See torch/csrc/utils/hash.h for dispatch code.
  static size_t hash(const BucketKey& key) {
    return c10::get_hash(key.type, key.device);
  }
};

inline bool operator==(const BucketKey& lhs, const BucketKey& rhs) {
  return lhs.type == rhs.type && lhs.device == rhs.device;
}

} // namespace

std::tuple<std::vector<std::vector<size_t>>, std::vector<size_t>>
compute_bucket_assignment_by_size(
    const std::vector<at::Tensor>& tensors,
    const std::vector<size_t>& bucket_size_limits,
    const std::vector<bool>& expect_sparse_gradient,
    const std::vector<int64_t>& tensor_indices,
    const std::optional<std::weak_ptr<c10d::Logger>>& logger) {
  // Either expect_sparse_gradient is not specified or it has as many elements
  // as the vector with tensors.
  TORCH_INTERNAL_ASSERT(
      expect_sparse_gradient.empty() ||
      (tensors.size() == expect_sparse_gradient.size()));
  TORCH_INTERNAL_ASSERT(!tensors.empty());
  // Store bucket indices and their sizes together, because we later sort the
  // resulting indices by minimum tensor index and want to keep sizes
  // consistent.
  std::vector<std::tuple<std::vector<size_t>, size_t>> result;
  // Sparse tensors go in their own bucket, so they do not have an enforced size
  // limit.
  size_t kNoSizeLimit = 0;
  result.reserve(tensors.size());

  // Keep iterator into the size_limit vector by tensor type and device.
  // This is done so that we can use the consecutive bucket limits per type.
  std::unordered_map<
      BucketKey,
      std::vector<size_t>::const_iterator,
      c10::hash<BucketKey>>
      bucket_size_limit_iterators;

  // Keep vector of indices and size accumulator by tensor type and device.
  std::unordered_map<BucketKey, BucketAccumulator, c10::hash<BucketKey>>
      buckets;

  for (const auto i : c10::irange(tensors.size())) {
    const auto& tensor = tensors[i];
    auto msg = std::string("No support for sparse tensors.");
    if (logger.has_value()) {
      REDUCER_CHECK(!tensor.is_sparse(), logger.value(), msg);
    } else {
      TORCH_CHECK(!tensor.is_sparse(), msg);
    }

    // when tensor_indices is empty, the index of tensors[i] assigned to
    // bucket is i, otherwise the tensor index is tensor_indices[i].
    auto tensor_index = i;
    if (!tensor_indices.empty()) {
      tensor_index = tensor_indices[i];
    }
    // If we expect a sparse gradient to be produced for this tensor, it cannot
    // be grouped together with other gradients and gets its own bucket.
    if (!expect_sparse_gradient.empty() &&
        expect_sparse_gradient[tensor_index]) {
      result.emplace_back(std::vector<size_t>({tensor_index}), kNoSizeLimit);
      continue;
    }

    auto key = BucketKey(tensor.scalar_type(), tensor.device());
    auto& bucket = buckets[key];
    bucket.indices.push_back(tensor_index);
    bucket.size += tensor.numel() * tensor.element_size();

    // Initialize bucket size limit iterator if necessary.
    if (bucket_size_limit_iterators.count(key) == 0) {
      bucket_size_limit_iterators[key] = bucket_size_limits.begin();
    }

    auto& bucket_size_limit_iterator = bucket_size_limit_iterators[key];
    const auto bucket_size_limit = *bucket_size_limit_iterator;
    bucket.size_limit = bucket_size_limit;
    if (bucket.size >= bucket_size_limit) {
      result.emplace_back(std::move(bucket.indices), bucket.size_limit);
      bucket = BucketAccumulator();

      // Advance to the next bucket size limit for this type/device.
      auto next = bucket_size_limit_iterator + 1;
      if (next != bucket_size_limits.end()) {
        bucket_size_limit_iterator = next;
      }
    }
  }

  // Add remaining buckets.
  for (auto& it : buckets) {
    auto& bucket = it.second;
    if (!bucket.indices.empty()) {
      result.emplace_back(std::move(bucket.indices), bucket.size_limit);
    }
  }

  // If tensor_indices is not empty, the order of the tensors is in the gradient
  // ready order, so no need to sort.
  // If tensor_indices is empty, sort resulting buckets by the minimum tensor
  // index they include. We assume that the order of the tensors is the order in
  // which they are used (or the reverse order in which their gradients are
  // produced). This sorting step ensures that the buckets are ready in
  // consecutive order.
  if (tensor_indices.empty()) {
    std::sort(
        result.begin(),
        result.end(),
        [](const std::tuple<std::vector<size_t>, size_t>& a,
           const std::tuple<std::vector<size_t>, size_t>& b) {
          auto indices_a = std::get<0>(a);
          auto indices_b = std::get<0>(b);
          const auto amin =
              std::min_element(indices_a.begin(), indices_a.end());
          const auto bmin =
              std::min_element(indices_b.begin(), indices_b.end());
          return *amin < *bmin;
        });
  }

  // Return bucket indices and size limits as separate entries in tuple, as some
  // APIs only need to consume bucket indices.
  std::vector<std::vector<size_t>> bucket_indices;
  bucket_indices.reserve(result.size());
  std::vector<size_t> per_bucket_size_limits;
  per_bucket_size_limits.reserve(result.size());
  for (const auto& bucket_indices_with_size : result) {
    bucket_indices.emplace_back(std::get<0>(bucket_indices_with_size));
    per_bucket_size_limits.emplace_back(std::get<1>(bucket_indices_with_size));
  }
  return std::make_tuple(bucket_indices, per_bucket_size_limits);
}

// Verifies corresponding params in the model replica have the same
// sizes/strides across processes.
void verify_params_across_processes(
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    const std::vector<at::Tensor>& params,
    const std::optional<std::weak_ptr<c10d::Logger>>& logger) {
  // First verify number of parameters to avoid inconsistent inputs into
  // broadcast which can cause a crash.
  // See https://github.com/pytorch/pytorch/issues/73547

  at::TensorOptions param_size_options;
  param_size_options = param_size_options.dtype(at::kLong);
  param_size_options = param_size_options.device(params[0].device());
  // Note: Not using tensor building API because of
  // https://github.com/pytorch/pytorch/issues/74114
  at::Tensor param_size_tensor =
      at::tensor({static_cast<int64_t>(params.size())}, param_size_options);

  // Allgather and verify parameter size.
  std::vector<std::vector<at::Tensor>> param_size_output_tensors;
  param_size_output_tensors.emplace_back();
  auto world_size = process_group->getSize();
  for (C10_UNUSED const auto i : c10::irange(world_size)) {
    param_size_output_tensors.front().emplace_back(
        at::empty_like(param_size_tensor));
  }

  std::vector<at::Tensor> param_size_vec{param_size_tensor};
  process_group->allgather(param_size_output_tensors, param_size_vec)->wait();
  auto result_size_tensors = param_size_output_tensors.front();
  for (const auto i : c10::irange(world_size)) {
    auto param_size_for_rank = result_size_tensors[i][0].item<int>();
    TORCH_CHECK(
        static_cast<size_t>(param_size_for_rank) == params.size(),
        c10::str(
            "DDP expects same model across all ranks, but Rank ",
            process_group->getRank(),
            " has ",
            params.size(),
            " params, while rank ",
            i,
            " has inconsistent ",
            param_size_for_rank,
            " params."));
  }

  // Continue with parameter shape verification.
  size_t i = 0;
  for (const auto& t : params) {
    i += 2 * t.dim();
  }
  at::TensorOptions options;
  options = options.dtype(at::kLong);
  auto metadata = at::empty({static_cast<long>(i)}, options);

  // Technically, process 0 is the broadcast source, so only process 0 needs
  // to populate metadata.  But no harm keeping work aligned across processes.
  auto metadata_accessor = metadata.accessor<int64_t, 1>();
  i = 0;
  for (const auto& t : params) {
    for (const auto& sz : t.sizes()) {
      metadata_accessor[static_cast<int64_t>(i++)] = sz;
    }
    for (const auto& str : t.strides()) {
      metadata_accessor[static_cast<int64_t>(i++)] = str;
    }
  }

  auto metadata_dev = metadata.clone().to(params[0].device());
  std::vector<at::Tensor> vec{metadata_dev};
  process_group->broadcast(vec)->wait();

  // Technically, process 0 doesn't need to double-check metadata, because it
  // was the source.  But no harm keeping work aligned.
  auto control = at::empty({static_cast<long>(i)}, options);
  control.copy_(metadata_dev, /*non_blocking=*/false);
  auto control_accessor = control.accessor<int64_t, 1>();
  i = 0;
  for (const auto p : c10::irange(params.size())) {
    const auto& t = params[p];
    for (const auto& sz : t.sizes()) {
      auto msg = c10::str(
          "[",
          process_group->getRank(),
          "]: params[",
          p,
          "] in this process",
          " with sizes ",
          t.sizes(),
          " appears not to match sizes of the same param in process 0.");
      if (logger.has_value()) {
        REDUCER_CHECK(sz == control_accessor[i++], logger.value(), msg)
      } else {
        TORCH_CHECK(sz == control_accessor[i++], msg)
      }
    }
    for (const auto& str : t.strides()) {
      auto msg = c10::str(
          "params[",
          p,
          "] in this process",
          " with sizes ",
          t.sizes(),
          " appears not to match strides of the same param in process 0.");
      if (logger.has_value()) {
        REDUCER_CHECK(str == control_accessor[i++], logger.value(), msg)
      } else {
        TORCH_CHECK(str == control_accessor[i++], msg)
      }
    }
  }
}

void Reducer::remove_autograd_hooks() {
  // Remove all hooks on variables registered by this Reducer. This is necessary
  // to make DDP failure recoverable. Otherwise, multiple Reducer instances
  // (from recoveries) will add their hooks to the original model, and those
  // hooks will try to invoke methods on a deleted Reducer objects.
  for (auto& hook : hooks_) {
    auto& key = hook.first;
    auto& grad_accumulator = hook.second;

    TORCH_INTERNAL_ASSERT(
        grad_accumulator->del_post_hook(key),
        "Reducer attempts to delete a non-existing hook.");
  }
  hooks_.clear();
}

void Reducer::check_finalized() {
  std::lock_guard<std::mutex> lock(mutex_);
  ensure_prior_reduction_finished();
}

void Reducer::update_process_group(
    c10::intrusive_ptr<c10d::ProcessGroup> new_process_group) {
  std::lock_guard<std::mutex> lock(mutex_);
  process_group_ = std::move(new_process_group);
}

void Reducer::reset_state() {
  std::lock_guard<std::mutex> lock(mutex_);
  // Force rebuild of buckets.
  has_rebuilt_bucket_ = false;
  rebuilt_params_.clear();
  rebuilt_param_indices_.clear();

  // Ensure forward can run despite previous backward not succeeding.
  expect_autograd_hooks_ = false;
  require_finalize_ = false;
  first_autograd_hook_called_ = false;

  // Unset allreduce division factor, as it may change in next backwards pass
  // when running with DDP join mode.
  div_factor_ = kUnsetDivFactor;

  // Reset unused parameter accounting.
  // See Note [local_used_map_ -> local_used_map_dev copying]
  local_used_map_.fill_(0);
  local_used_map_reduced_ = false;
}

} // namespace c10d
