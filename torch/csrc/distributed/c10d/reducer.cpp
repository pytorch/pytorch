#include <torch/csrc/distributed/c10d/reducer.h>

#include <functional>

#include <c10/util/Exception.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function_hook.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/utils/hash.h>
#include <torch/csrc/utils/memory.h>

namespace c10d {
namespace {

// Turns lambda without input/output into a torch::autograd::FunctionPostHook.
class LambdaPostHook : public torch::autograd::FunctionPostHook {
  using variable_list = std::vector<torch::autograd::Variable>;

 public:
  /* implicit */ LambdaPostHook(std::function<void(void)> fn)
      : fn_(std::move(fn)) {}

  variable_list operator()(
      const variable_list& outputs,
      const variable_list& /* unused */) override {
    fn_();
    return outputs;
  }

 protected:
  std::function<void(void)> fn_;
};

inline int64_t current_time_in_nanos() {
  return torch::autograd::profiler::getTime();
}

} // namespace

Reducer::Reducer(
    std::vector<std::vector<torch::autograd::Variable>> replicas,
    std::vector<std::vector<size_t>> bucket_indices,
    std::shared_ptr<c10d::ProcessGroup> process_group)
    : replicas_(std::move(replicas)),
      process_group_(std::move(process_group)),
      expect_autograd_hooks_(false),
      require_finalize_(false),
      has_marked_unused_parameters_(false),
      next_bucket_(0),
      backward_stats_base_(0) {
  AT_ASSERTM(replicas_.size() >= 1, "Expected at least one model replica.");
  AT_ASSERTM(replicas_[0].size() >= 1, "Expected at least one parameter.");

  // Verify that all specified variables require gradients,
  // and that they have the same size across replicas.
  {
    const auto replica_count = replicas_.size();
    for (size_t replica_index = 0; replica_index < replica_count;
         replica_index++) {
      const auto variable_count = replicas_[replica_index].size();
      AT_ASSERTM(
          replicas_[replica_index].size() == replicas_[0].size(),
          "Model replicas must have an equal number of parameters.");
      for (size_t variable_index = 0; variable_index < variable_count;
           variable_index++) {
        AT_ASSERTM(
            replicas_[replica_index][variable_index].requires_grad(),
            "Variables must require gradients (have `requires_grad` set).");
        AT_ASSERTM(
            replicas_[replica_index][variable_index].sizes() ==
                replicas_[0][variable_index].sizes(),
            "Variables across model replicas must have identical sizes.");
        AT_ASSERTM(
            replicas_[replica_index][variable_index].dtype() ==
                replicas_[0][variable_index].dtype(),
            "Variables across model replicas must have identical dtype.");
      }
    }
  }

  // Initialize variable bucketing.
  // This can be reinitialized later after capturing runtime information.
  initialize_buckets(std::move(bucket_indices));

  // All variables are expected to have their `grad_fn` set to the gradient
  // accumulation function (since they are leafs in the autograd graph).
  // We store pointers to these functions such that we can check if they are
  // used in an autograd pass. If they are not, we know their grad tensors
  // can be marked as ready for reduction.
  {
    const auto replica_count = replicas_.size();
    grad_accumulators_.resize(replica_count);
    for (size_t replica_index = 0; replica_index < replica_count;
         replica_index++) {
      const auto variable_count = replicas_[replica_index].size();
      grad_accumulators_[replica_index].resize(variable_count);
      for (size_t variable_index = 0; variable_index < variable_count;
           variable_index++) {
        auto& variable = replicas_[replica_index][variable_index];

        // The gradient accumulator function is lazily initialized once.
        // Therefore we can use its presence in the autograd graph as
        // evidence that the parameter has participated in an iteration.
        auto grad_accumulator = variable.grad_accumulator();

        // Hook to execute after the gradient accumulator has executed.
        hooks_.emplace_back(
            grad_accumulator->add_post_hook(
                torch::make_unique<LambdaPostHook>([=] {
                    std::lock_guard<std::mutex> lock(this->mutex_);
                    this->mark_variable_ready(
                        replica_index,
                        variable_index,
                        /* called_from_autograd= */ true);
                })),
            grad_accumulator
        );

        // Map raw function pointer to replica index and parameter index.
        // This is used later on when the autograd graph is traversed
        // to check for parameters for which no gradient is computed.
        func_[grad_accumulator.get()] =
            std::make_tuple(replica_index, variable_index);

        // The gradient accumulator is stored as weak_ptr in the autograd
        // metadata of the variable, so we have to keep it alive here for
        // the raw pointer to be valid.
        grad_accumulators_[replica_index][variable_index] =
            std::move(grad_accumulator);
      }
    }
  }

  // Initialize backward stats vector.
  {
    const auto replica_count = replicas_.size();
    backward_stats_.resize(replica_count);
    const auto variable_count = replicas_[0].size();
    std::for_each(
        backward_stats_.begin(),
        backward_stats_.end(),
        [=](std::vector<int64_t>& v) { v.resize(variable_count); });
  }
}

Reducer::~Reducer() noexcept(false) {
  // Remove all hooks on variables registered by this Reducer. This is necessary
  // to make DDP failure recoverable. Otherwise, multiple Reducer instances
  // (from recoveries) will add their hooks to the original model, and those
  // hooks will try to invoke methods on a deleted Reducer objects.
  for (auto& hook : hooks_) {
    auto& key = hook.first;
    auto& grad_accumulator = hook.second;
    AT_ASSERTM(grad_accumulator->del_post_hook(key),
        "Reducer attempts to delete a non-existing hook.");
  }
}

// Called when the gradient for the specified variable is ready.
// It can be called from two places:
// - By an autograd thread after executing a gradient accumulator function.
// - By the `Reducer::prepare_for_backward` function if the variable doesn't
//   show up in the autograd graph (and it wouldn't be called by autograd).
void Reducer::mark_variable_ready(
    size_t replica_index,
    size_t variable_index,
    bool called_from_autograd) {
  // Ignore if we don't expect to be called.
  // This may be the case if the user wants to accumulate gradients
  // for number of iterations before reducing them.
  if (!expect_autograd_hooks_) {
    return;
  }

  AT_ASSERTM(replica_index < replicas_.size(), "Out of range replica index.");
  AT_ASSERTM(
      variable_index < variable_locators_.size(),
      "Out of range variable index.");
  backward_stats_[replica_index][variable_index] =
      current_time_in_nanos() - backward_stats_base_;

  // Any time we mark a variable ready (be it in line due to unused parameters,
  // or via an autograd hook), we require a call to the finalize function. If
  // this doesn't happen before the next iteration (or call to
  // `prepare_for_backwards`), we know something is wrong.
  require_finalize_ = true;

  const auto& bucket_index = variable_locators_[variable_index];
  auto& bucket = buckets_[bucket_index.bucket_index];
  auto& replica = bucket.replicas[replica_index];

  // Something is wrong if all variables contained in this bucket replica have
  // already been marked as ready.
  if (replica.pending == 0) {
    // Receiving a call to `mark_variable_ready` twice for the same variable
    // is only possible if the variable was initially deemed unused, and was
    // marked ready from the `prepare_for_backward` function, only to become
    // part of the autograd graph at a later point in time.
    AT_ASSERT(has_marked_unused_parameters_);
    AT_ERROR(
        "Expected to mark a variable ready only once. ",
        "",
        "This error is caused by use of a module parameter outside the ",
        "`forward` function. The return value of the `forward` function ",
        "is inspected by the distributed data parallel wrapper to figure ",
        "out if any of the module's parameters went unused. If this is the ",
        "case, it knows they won't receive gradients in a backward pass. ",
        "If any of those parameters are then used outside `forward`, this ",
        "error condition is triggered. ",
        "",
        "You can disable unused parameter detection by passing the keyword "
        "argument `find_unused_parameters=False` to ",
        "`torch.nn.parallel.DistributedDataParallel`.");
  }

  auto& variable = replica.variables[bucket_index.intra_bucket_index];
  const auto offset = replica.offsets[bucket_index.intra_bucket_index];
  const auto length = replica.lengths[bucket_index.intra_bucket_index];

  // Copy contents of gradient tensor to bucket tensor.
  // If the gradient is not set, we assume it wasn't computed
  // as part of the current backwards pass, and zero the part
  // of the bucket it would otherwise hold.
  auto bucket_view = replica.contents.narrow(0, offset, length);
  auto& grad = variable.grad();
  if (grad.defined()) {
    // Assert that the grad tensor and the bucket don't share storage.
    // If they did, we could avoid the copy altogether.
    // The reason for not doing this is that existing code calls
    // `detach_` from `zero_grad`, which is incompatible with views.
    AT_ASSERT(!grad.is_alias_of(bucket_view));
    AT_ASSERT(grad.type() == variable.type());
    AT_ASSERT(grad.device() == variable.device());
    AT_ASSERT(grad.numel() == length);
    bucket_view.copy_(grad.view({-1}), /* non_blocking */ true);
  } else {
    bucket_view.zero_();
  }

  // TODO(@pietern): Make this work for both CPU/CUDA tensors.
  // When using CPU tensors we don't need to do this.
  // // Record event so that we can wait for all of them.
  // auto& event = replica.events[bucket_index.intra_bucket_index];
  // event.record();

  // Check if this was the final gradient for this bucket.
  if (--replica.pending == 0) {
    // Prescale bucket contents to turn the global sum into the global average.
    replica.contents.div_(process_group_->getSize());
    // Kick off reduction if all replicas for this bucket are ready.
    if (--bucket.pending == 0) {
      mark_bucket_ready(bucket_index.bucket_index);
    }
  }

  // Run finalizer function once the final bucket was marked ready.
  if (next_bucket_ == buckets_.size()) {
    if (called_from_autograd) {
      torch::autograd::Engine::get_default_engine().queue_callback([=] {
        std::lock_guard<std::mutex> lock(this->mutex_);
        this->finalize_backward();
      });
    } else {
      finalize_backward();
    }
  }
}

// Called when the bucket at the specified index is ready to be reduced.
void Reducer::mark_bucket_ready(size_t bucket_index) {
  AT_ASSERT(bucket_index >= next_bucket_);

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
    auto& bucket = buckets_[next_bucket_];
    std::vector<at::Tensor> tensors;
    tensors.reserve(bucket.replicas.size());
    for (const auto& replica : bucket.replicas) {
      // TODO(@pietern): Ensure proper synchronization with the CUDA events
      // that recorded copies into this contents tensor. If these copies are
      // executed on non-default streams, the current stream for the device
      // that holds the contents tensor must wait on these events.
      //
      // As long as autograd uses the default stream for every device,
      // these operations are implicitly sequenced, and we don't need to
      // do any extra synchronization here.
      //
      tensors.push_back(replica.contents);
    }
    bucket.work = process_group_->allreduce(tensors);
  }
}

void Reducer::initialize_buckets(
    std::vector<std::vector<size_t>> bucket_indices) {
  std::lock_guard<std::mutex> lock(mutex_);

  // This shouldn't be called if we're expecting autograd hooks to fire.
  AT_ASSERTM(
      !expect_autograd_hooks_,
      "`initialize_buckets` must NOT be called during autograd execution.");

  // Clear current bucket assignment.
  buckets_.clear();
  variable_locators_.clear();

  // Ensure we have a bucket index for every variable.
  variable_locators_.resize(replicas_[0].size());

  // Iterate over buckets.
  const auto bucket_count = bucket_indices.size();
  const auto replica_count = replicas_.size();
  buckets_.reserve(bucket_count);
  for (size_t bucket_index = 0; bucket_index < bucket_count; bucket_index++) {
    Bucket bucket;

    // TODO(@pietern): Validate indices.
    // Must be non-empty, unique, and unique across buckets.
    AT_ASSERTM(
        bucket_indices[bucket_index].size() > 0, "Empty bucket specified.");

    // Iterate over model replicas.
    for (size_t replica_index = 0; replica_index < replica_count;
         replica_index++) {
      at::TensorOptions options;
      BucketReplica replica;
      size_t offset = 0;

      // Iterate over bucket variables.
      for (const auto variable_index : bucket_indices[bucket_index]) {
        AT_ASSERTM(
            variable_index < replicas_[replica_index].size(),
            "Out of range variable index specified.");
        const auto& variable = replicas_[replica_index][variable_index];
        if (!options.has_device()) {
          options = options.device(variable.device());
        } else {
          AT_ASSERTM(
              variable.device() == options.device(),
              "All parameters in a bucket must be placed on the same device.");
        }
        if (!options.has_dtype()) {
          options = options.dtype(variable.dtype());
        } else {
          AT_ASSERTM(
              variable.dtype() == options.dtype(),
              "All parameters in a bucket must have the same dtype.");
        }
        const auto length = variable.numel();
        replica.variables.push_back(variable);
        replica.offsets.push_back(offset);
        replica.lengths.push_back(length);
        offset += length;
      }

      // Allocate bucket contents tensor.
      // This must be a Variable because as of Apr 2019 there is still
      // a distinction between the Tensor and Variable types, and it
      // is not recommended (or sometimes even possible) to mix and match.
      replica.contents = torch::autograd::make_variable_consuming(
          at::empty({static_cast<long>(offset)}, options));

      // Add bucket replica to enclosing bucket.
      bucket.replicas.push_back(std::move(replica));
    }

    // Map participating variables to this bucket.
    // This is identical across replicas so we only need to do this once.
    size_t intra_bucket_index = 0;
    for (const auto variable_index : bucket_indices[bucket_index]) {
      AT_ASSERTM(
          variable_index < variable_locators_.size(),
          "Out of range variable index specified.");
      variable_locators_[variable_index] = VariableLocator{
          .bucket_index = bucket_index,
          .intra_bucket_index = intra_bucket_index++,
      };
    }

    buckets_.push_back(std::move(bucket));
  }
}

// Traverse the autograd graph starting at the specified output.
// All parameters for which we have a pointer to their gradient accumulation
// functions and don't show up in this graph can be marked as ready
// for reduction immediately. Not doing this means we would deadlock waiting
// on a gradient for those parameters that will never be computed.
//
// Rough copy of torch::autograd::Engine::compute_dependencies.
//
void Reducer::prepare_for_backward(
    const std::vector<torch::autograd::Variable>& outputs) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::unordered_set<torch::autograd::Function*> seen;
  std::vector<torch::autograd::Function*> queue;

  // Check that any prior reduction has finished.
  // The variable `expect_autograd_hooks` is true until gradients for all
  // parameters have been received and all buckets are ready.
  if (require_finalize_) {
    AT_ERROR(
        "Expected to have finished reduction in the prior iteration before ",
        "starting a new one. ",
        "",
        "This error indicates that your module has parameters that were ",
        "not used in producing loss. ",
        "",
        "You can enable unused parameter detection by (1) passing the keyword "
        "argument `find_unused_parameters=True` to ",
        "`torch.nn.parallel.DistributedDataParallel`; (2) making sure all ",
        "`forward` function outputs participate in calculating loss. "
        "",
        "If you already have done the above two steps, then the distributed ",
        "data parallel module wasn't able to locate the output tensors in the ",
        "return value of your module's `forward` function. ",
        "Please include the loss function and the structure of the return ",
        "value of `forward` of your module when reporting this issue (e.g. ",
        "list, dict, iterable).");
  }

  // Reset accounting.
  has_marked_unused_parameters_ = true;
  expect_autograd_hooks_ = true;
  next_bucket_ = 0;
  backward_stats_base_ = current_time_in_nanos();
  for (auto& bucket : buckets_) {
    for (auto& replica : bucket.replicas) {
      replica.pending = replica.variables.size();
    }
    bucket.pending = bucket.replicas.size();
  }

  // If no outputs are specified, we assume that autograd hooks for ALL
  // variables will be called, and we don't have to search the autograd graph
  // for presence of these hooks.
  if (outputs.empty()) {
    has_marked_unused_parameters_ = false;
    return;
  }

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
  for (const auto& it : func_) {
    // If the accumulator function is present in the graph, we know
    // a gradient will be computed for the corresponding parameter.
    if (seen.count(it.first) > 0) {
      continue;
    }

    size_t replica_index;
    size_t variable_index;
    std::tie(replica_index, variable_index) = it.second;
    mark_variable_ready(replica_index, variable_index);
  }
}

void Reducer::finalize_backward() {
  // No longer expect autograd hooks to fire after this function returns.
  AT_ASSERT(expect_autograd_hooks_);
  expect_autograd_hooks_ = false;

  // No longer require call to finalize after this function returns.
  AT_ASSERT(require_finalize_);
  require_finalize_ = false;

  // Check that all buckets were completed and had their work kicked off.
  AT_ASSERT(next_bucket_ == buckets_.size());

  // Wait for asynchronous reduction to complete and unflatten contents.
  for (auto& bucket : buckets_) {
    AT_ASSERT(bucket.work);
    bucket.work->wait();
    for (auto& replica : bucket.replicas) {
      for (size_t intra_bucket_index = 0;
           intra_bucket_index < replica.variables.size();
           intra_bucket_index++) {
        auto& variable = replica.variables[intra_bucket_index];
        const auto offset = replica.offsets[intra_bucket_index];
        const auto length = replica.lengths[intra_bucket_index];
        auto bucket_view =
            replica.contents.narrow(0, offset, length).view(variable.sizes());
        auto& grad = variable.grad();
        if (!grad.defined()) {
          grad = at::empty(bucket_view.sizes(), bucket_view.options());
        }
        grad.copy_(bucket_view);
      }
    }
  }
}

namespace {

// Tensors may be coalesced into buckets. Buckets must contain tensors of
// the same type, on the same device, so a bucket can identified by a
// composite key of a tensor's type identifier and its device.
struct BucketKey {
  BucketKey(c10::ScalarType type, c10::Device device)
      : type(std::move(type)), device(std::move(device)) {}

  const c10::ScalarType type;
  const c10::Device device;

  // See torch/csrc/utils/hash.h for dispatch code.
  static size_t hash(const BucketKey& key) {
    return torch::get_hash(key.type, key.device);
  }
};

inline bool operator==(const BucketKey& lhs, const BucketKey& rhs) {
  return lhs.type == rhs.type && lhs.device == rhs.device;
}

} // namespace

// This is equivalent to take_tensors but returns indices into the
// tensor list argument for bucket assignment. Also, it is aware
// of device placement and will not allow buckets to span devices.
std::vector<std::vector<size_t>> compute_bucket_assignment_by_size(
    const std::vector<at::Tensor>& tensors,
    std::vector<size_t> bucket_size_limits) {
  std::vector<std::vector<size_t>> result;
  result.reserve(tensors.size());

  // Keep iterator into the size_limit vector by tensor type and device.
  // This is done so that we can use the consecutive bucket limits per type.
  std::unordered_map<
      BucketKey,
      std::vector<size_t>::iterator,
      torch::hash<BucketKey>>
      bucket_size_limit_iterators;

  // Local accumulator type for a single bucket.
  struct BucketAccumulator {
    std::vector<size_t> indices;
    size_t size = 0;
  };

  // Keep vector of indices and size accumulator by tensor type and device.
  std::unordered_map<BucketKey, BucketAccumulator, torch::hash<BucketKey>>
      buckets;

  for (size_t i = 0; i < tensors.size(); i++) {
    const auto& tensor = tensors[i];
    AT_ASSERTM(!tensor.is_sparse(), "No support for sparse tensors.");
    auto key = BucketKey(tensor.scalar_type(), tensor.device());
    auto& bucket = buckets[key];
    bucket.indices.push_back(i);
    bucket.size += tensor.numel() * tensor.element_size();

    // Initialize bucket size limit iterator if necessary.
    if (bucket_size_limit_iterators.count(key) == 0) {
      bucket_size_limit_iterators[key] = bucket_size_limits.begin();
    }

    auto& bucket_size_limit_iterator = bucket_size_limit_iterators[key];
    const auto bucket_size_limit = *bucket_size_limit_iterator;
    if (bucket.size >= bucket_size_limit) {
      result.emplace_back(std::move(bucket.indices));
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
      result.emplace_back(std::move(bucket.indices));
    }
  }

  // Sort resulting buckets by the minimum tensor index they include.
  // We assume that the order of the tensors is the order in which they are
  // used (or the reverse order in which their gradients are produced).
  // This sorting step ensures that the buckets are ready in consecutive order.
  std::sort(
      result.begin(),
      result.end(),
      [](const std::vector<size_t>& a, const std::vector<size_t>& b) {
        const auto amin = std::min_element(a.begin(), a.end());
        const auto bmin = std::min_element(b.begin(), b.end());
        return *amin < *bmin;
      });

  return result;
}

} // namespace c10d
