#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>

#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/parser.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/registry.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

#include <c10/util/irange.h>
#include <torch/csrc/jit/jit_log.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

// Check device of TensorType in all inputs ensure all tensors are on cuda
// devices.
// return common device index (or -1 if device differs).
int getCommonDeviceCUDA(const at::ArrayRef<IValue>& inputs) {
  int index = -1;
  for (const auto& input : inputs) {
    if (!input.isTensor()) {
      continue;
    }
    const auto& device = input.toTensor().device();
    // skip cpu scalar tensor as they'll be promoted to scalar later
    if (device.is_cpu() && is_cpu_scalar(input.toTensor())) {
      continue;
    }
    TORCH_CHECK(device.is_cuda(), "nvfuser only supports cuda device");
    auto cur_index = device.index();
    if (index != -1 && index != cur_index) {
      return -1;
    }
    index = (int)cur_index; // NOLINT
  }
  return index;
}

// TODO: temporary hack to resolve my is_constructible issue;
std::vector<size_t> toVector(const at::DimVector& small_vec) {
  return std::vector<size_t>(small_vec.begin(), small_vec.end());
}

void encodeBuffer(size_t value, std::string& buffer) {
  const char* v = reinterpret_cast<char*>(&value);
  for (const auto i : c10::irange(sizeof(size_t))) {
    (void)i; // Suppress unused variable warning
    buffer.push_back(*(v++));
  }
}

} // namespace

InputsIdLookup::IdLookupReturn InputsIdLookup::lookupId(
    const at::ArrayRef<IValue>& inputs,
    const SchedulerRuntimeInfo* additional_info) {
  IdLookupReturn ret;

  // lock mutex_ because we are touching encoding_
  std::lock_guard<std::mutex> guard(mutex_);
  encoding_.clear();
  for (const auto& input : inputs) {
    if (input.isTensor()) {
      auto& input_tensor = input.toTensor();

      for (auto size : input_tensor.sizes()) {
        encodeBuffer(size, encoding_);
        encoding_.push_back(' ');
      }
      encoding_.push_back('X');
      encoding_.push_back(' ');
      for (auto stride : input_tensor.strides()) {
        encodeBuffer(stride, encoding_);
        encoding_.push_back(' ');
      }
      encoding_.push_back('d');
      encodeBuffer(input_tensor.device().index(), encoding_);
    } else {
      // encode s for scalar;
      encoding_.push_back('s');
    }
    encoding_.push_back(';');
  }
  if (additional_info) {
    encodeBuffer(additional_info->getCommonAlignmentSize(), encoding_);
  }

  auto& entry = encoding_lookup_[encoding_];

  if (entry.id == 0) {
    // no entry existed for given input set, set id for given entry
    entry.id = current_id_++;
    if (used_entry_.size() == max_cache_size_) {
      // pop least recently used cache;
      const auto& remove_iter = encoding_lookup_.find(used_entry_.back());
      used_entry_.pop_back();
      ret.evict_id = remove_iter->second.id;
      ret.eviction = true;
      encoding_lookup_.erase(remove_iter);
    }
  } else {
    // short-cut to leave LRU entry as is
    if (entry.lru_iter == used_entry_.begin()) {
      ret.id = entry.id;
      return ret;
    }

    used_entry_.erase(entry.lru_iter);
  }

  ret.id = entry.id;
  entry.lru_iter = used_entry_.insert(used_entry_.begin(), encoding_);
  return ret;
}

FusionExecutorCache::FusionExecutorCache(std::unique_ptr<Fusion> fusion)
    : fusion_(std::move(fusion)) {
  for (const auto& indices : fusion_->getOutputAliasIndices()) {
    aliased_output_indices_.insert(indices);
  }
}

// Note [ Permutation support in nvfuser ]
//
// Background:
// To support permutation in nvfuser with optimal performance, we would want to
// allow dimension collapsing in generated code on channels-last tensors, which
// greatly simplifies indexing. Current API in codegen only allows dimensional
// collapsing on neighboring axes. The unfortunate thing is that memory format
// design in PyTorch is implicitly marked by strides, while the semantics
// meaning of axes remain unchanged. i.e. A 4d tensor with axes [N, C, H, W]
// would have the same shape in both format, while contiguous tensor carries
// strides [C*H*W, H*W, W, 1] and channels-last tensor [H*W*C, 1, W*C, C]
//
// Approach:
// Part_1. To allow axes collapsing for permuted tensor in codegen, we can
// permute input tensor to have axes in decending order by their strides, so
// they would be viewed as `contiguous` in codegen, hence collapsed to simple
// indexing. Part_2. To ensure correct result, we need to ensure computation in
// nvfuser carries same semantics as with TorchScript graph. We need to
//   Part_2_1. Maintain a bookkeeping where each codegen tensor is tagged with
//   either their permutation. Part_2_2. Parsing rule should handle and
//   propagate the tag properly, e.g. batch normalization has special rules for
//   `channels_last` input tensor and mark output in its right permutation.
// Part_3. Codegen output tensor that has been permuted should be restored to
// original layout before returning to TorchScript
//
// For details  on Part_2, refer to implementation Note [ Permutation
// Bookkeeping and Propagation in Parser ]
std::vector<at::Tensor> FusionExecutorCache::runFusionWithInputs(
    const at::ArrayRef<IValue>& inputs) {
  FUSER_PERF_SCOPE("FusionExecutorCache::runFusionWithInputs");

  // permute input tensor for kernel execution. See Part_1 in Note [ Channels
  // Last support in nvfuser ]
  at::ArrayRef<IValue> perm_inputs = inputs;
  const auto& to_be_permuted_inputs = fusion_->getPermutationInputMap();
  std::vector<IValue> inputs_vec;
  if (!to_be_permuted_inputs.empty()) {
    inputs_vec = inputs.vec();
    for (const auto& pair : to_be_permuted_inputs) {
      auto v = inputs_vec[pair.first];
      TORCH_CHECK(
          v.isTensor(), "input permutation can only be applied at tensor");
      auto tensor = v.toTensor();
      inputs_vec[pair.first] = tensor.permute(pair.second);
    }
    perm_inputs = inputs_vec;
  }

  SchedulerRuntimeInfo runtime_info(fusion(), perm_inputs);

  auto id_lookup_ret = inputs_id_lookup_.lookupId(perm_inputs, &runtime_info);
  if (id_lookup_ret.eviction) {
    evictCache(id_lookup_ret.evict_id);
  }

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  const size_t unique_id = id_lookup_ret.id;
  auto kernel_runtime = getKernelRuntimeFor(perm_inputs, unique_id);
  most_recent_runtime_ = kernel_runtime;
  auto outputs = kernel_runtime->runWithInput(perm_inputs, unique_id);

  // permute output tensor returned by kernel execution. See Part_3 in Note [
  // Permutation support in nvfuser ]
  for (const auto& pair : fusion_->getPermutationOutputMap()) {
    outputs[pair.first] = outputs[pair.first].permute(pair.second);
  }

  int offset = 0;
  for (const auto& v : aliased_output_indices_) {
    outputs.erase(outputs.begin() + v - offset);
    offset++;
  }

  return outputs;
}

void FusionExecutorCache::evictCache(size_t cache_id) {
  auto it = id_to_kernel_runtime_.find(cache_id);
  TORCH_INTERNAL_ASSERT(it != id_to_kernel_runtime_.end());
  it->second->evictCache(cache_id);
  id_to_kernel_runtime_.erase(it);
}

FusionKernelRuntime* FusionExecutorCache::getKernelRuntimeFor(
    const at::ArrayRef<IValue>& inputs,
    size_t unique_id) {
  // Check for id hit case
  auto id_it = id_to_kernel_runtime_.find(unique_id);
  if (id_it != id_to_kernel_runtime_.end()) {
    return id_it->second;
  }

  // Access kernels associated with the common device id
  auto device_index = getCommonDeviceCUDA(inputs);
  TORCH_CHECK(device_index >= 0, "device is not coherent for fusion inputs");
  auto& kernel_runtimes = kernel_runtimes_[device_index];

  // Check for re-use hit case
  //  a kernel runtime is re-usable if all the compiled
  //  kernels have the same heuristic parameters
  std::unique_ptr<FusionHeuristics> new_heuristics;

  auto reuse_it = std::find_if(
      kernel_runtimes.begin(),
      kernel_runtimes.end(),
      [&inputs, &new_heuristics](auto& kernel_runtime) {
        auto maybe_heuristics = kernel_runtime->getMaybeHeuristicsFor(inputs);
        if (!maybe_heuristics.has_value()) {
          return false;
        }
        new_heuristics = std::move(maybe_heuristics.value());
        return true;
      });

  FusionKernelRuntime* kernel_runtime = nullptr;
  if (reuse_it != kernel_runtimes.end()) {
    kernel_runtime = reuse_it->get();
    kernel_runtime->updateHeuristicsLaunchParams(new_heuristics.get());
  } else {
    // graph miss, need to re-build an optimized graph for this case
    kernel_runtimes.emplace_back(
        std::make_unique<FusionKernelRuntime>(fusion_.get(), inputs));
    kernel_runtime = kernel_runtimes.back().get();
    if (profiling_) {
      kernel_runtime->profile(true);
    }
  }

  id_to_kernel_runtime_[unique_id] = kernel_runtime;
  return kernel_runtime;
}

FusionKernelRuntime::FusionKernelRuntime(
    Fusion* fusion,
    const at::ArrayRef<IValue>& inputs) {
  FUSER_PERF_SCOPE("FusionKernelRuntime::FusionKernelRuntime");

  // Make a copy of fusion and do segmentation and translation
  //  on this copy
  auto fusion_copy = std::make_unique<Fusion>(*fusion);

  // Run segmentation on the copied fusion
  SchedulerRuntimeInfo runtime_info(fusion_copy.get(), inputs, true);

  // Initialize the evaluator simplifer
  precomputed_integers_ =
      std::make_unique<FusionPrecomputedIntegers>(fusion_copy.get());

  //! Try to schedule the complete fusion
  const auto maybe_complete_fusion_heuristic =
      SchedulerEntry::proposeHeuristics(fusion_copy.get(), runtime_info);

  //! Decide if this fusion is segmented or not
  const bool segmented = !maybe_complete_fusion_heuristic.has_value();

  if (segmented) {
    // Take ownership and segment transformed fusion
    segmented_fusion_ =
        SegmentCandidateFinder::segment(std::move(fusion_copy), inputs);
    heuristics_ = segmented_fusion_->makeInitialHeuristics(inputs);
    executors_ =
        std::vector<FusionExecutor>(segmented_fusion_->groups().size());
    if (isDebugDumpEnabled(DebugDumpOption::FusionSegments)) {
      segmented_fusion_->print();
    }
  } else {
    auto complete_fusion_heuristic = maybe_complete_fusion_heuristic.value();

    // Take ownership of the transformed fusion
    single_kernel_fusion_ = std::move(fusion_copy);

    single_kernel_fusion_data_cache_ = std::make_unique<HeuristicSummary>(
        single_kernel_fusion_.get(), complete_fusion_heuristic, runtime_info);

    heuristics_ = std::make_unique<FusionHeuristics>(
        complete_fusion_heuristic,
        runtime_info,
        single_kernel_fusion_data_cache_.get());

    executors_ = std::vector<FusionExecutor>(1);
    // In the case that the fusion isn't segmented but user
    //  wants segmented fusion in the debug print. Will
    //  print math of the composite fusion as placeholder
    if (isDebugDumpEnabled(DebugDumpOption::FusionSegments)) {
      single_kernel_fusion_->printMath();
    }
  }

  is_segmented_ = segmented;

  if (is_segmented_) {
    prepareRuntimeOrder();
  }
}

std::vector<at::Tensor> FusionKernelRuntime::runKernelWithInput(
    const at::ArrayRef<IValue>& inputs,
    size_t input_id,
    SegmentedGroup* sg) {
  FUSER_PERF_SCOPE("FusionKernelRuntime::runKernelWithInput");
  // This function will be called once on un-segmented fusion,
  //  for segmented fusion, this function will be called on each segment
  //  In the case of segmented fusion, segmented group needs to be given so
  //   a kernel is compiled and run for a segmented group
  //  In the case of complete fusion, sg = nullptr, and the original fusion
  //   is complied and run
  auto group_id = sg ? sg->groupId() : 0;
  const int device_index = getCommonDeviceCUDA(inputs);
  TORCH_CHECK(device_index >= 0, "device is not coherent for fusion inputs");

  LaunchParams launch_params;

  auto scheduler_entry = schedulers()[group_id].get();

  // Check that the heuristics are matched, in the case of segmented fusion
  TORCH_INTERNAL_ASSERT(!sg || scheduler_entry->heuristc() == sg->heuristic());

  if (!executors_[group_id].compiled()) {
    FUSER_PERF_SCOPE("FusionKernelRuntime::runKernelWithInput::Compile");
    std::unique_ptr<Fusion> fusion_to_run;
    if (sg) {
      // Running a segment group as a single kernel,
      //  make a fusion to run from segmented fusion
      fusion_to_run = segmented_fusion_->makeFusion(sg);
    } else {
      // Without a segmented group defaults to compiling the
      //  complete fusion
      fusion_to_run = std::make_unique<Fusion>(*single_kernel_fusion_);
    }
    CompileOptions options;
    options.device = c10::Device(DeviceType::CUDA, device_index);
    options.index_mode = scheduler_entry->indexMode();
    FusionGuard fg(fusion_to_run.get());
    scheduler_entry->schedule(fusion_to_run.get());
    // Load launch params for reduction and normalization kernels
    if (scheduler_entry->hasReductionParam()) {
      launch_params = scheduler_entry->reductionParams().lparams;
    } else {
      launch_params = scheduler_entry->pointwiseParams().lparams;
    }
    executors_[group_id].compileFusion(
        fusion_to_run.get(), inputs, launch_params, options);
  } else {
    // Load launch params for reduction and normalization kernels
    if (scheduler_entry->hasReductionParam()) {
      launch_params = scheduler_entry->reductionParams().lparams;
    } else {
      launch_params = scheduler_entry->pointwiseParams().lparams;
    }
  }

  if (profiling_) {
    most_recent_executor_log_.fusion_executor = &executors_[group_id];
    most_recent_executor_log_.launch_constraints = launch_params;
    if (scheduler_entry->hasReductionParam()) {
      most_recent_executor_log_.reduction_params =
          scheduler_entry->reductionParams();
    } else {
      most_recent_executor_log_.pointwise_params =
          scheduler_entry->pointwiseParams();
    }
  }

  return executors_[group_id].runFusion(inputs, launch_params, input_id);
}

void FusionKernelRuntime::prepareRuntimeOrder() {
  // Setup group run order:
  std::unordered_set<Val*> available_input;

  // setup the order tensor dimensions are bound
  for (const size_t i : c10::irange(segmented_fusion_->inputs().size())) {
    auto input_val = segmented_fusion_->inputs()[i];
    available_input.insert(input_val);

    if (auto input_tv = dynamic_cast<TensorView*>(input_val)) {
      auto root_dom = TensorDomain::noReductions(input_tv->getRootDomain());
      for (const size_t dim : c10::irange(root_dom.size())) {
        const auto extent = root_dom[dim]->extent();
        available_input.insert(extent);
        runtime_workspace_.group_extent_binding_order.push_back(extent);
      }
    }
  }

  // Keep track of groups that has run
  std::vector<bool> group_ran(segmented_fusion_->groups().size(), false);

  while (!std::all_of(
      group_ran.begin(), group_ran.end(), [](bool b) { return b; })) {
    bool one_ran = false;

    // Find the first segment with all inputs available to run
    for (const size_t group_i :
         c10::irange(segmented_fusion_->groups().size())) {
      auto& group = segmented_fusion_->groups()[group_i];
      if (group_ran[group_i]) {
        continue;
      }
      const auto& group_inputs = group->inputs();
      bool ready_to_run = std::all_of(
          group_inputs.begin(),
          group_inputs.end(),
          [&available_input](Val* val) { return available_input.count(val); });

      if (ready_to_run) {
        runtime_workspace_.group_run_order.push_back(group);
        const auto& group_outputs = group->outputs();

        // Insert graph segment output to tensor map
        for (const size_t group_out_i : c10::irange(group_outputs.size())) {
          available_input.insert(group_outputs[group_out_i]);
        }
        group_ran[group_i] = true;
        one_ran = true;
      }
    }
    TORCH_INTERNAL_ASSERT(
        one_ran,
        "Couldn't run all groups, something must have gone wrong in segmentation.");
  }
}

std::vector<at::Tensor> FusionKernelRuntime::runWithInput(
    const at::ArrayRef<IValue>& inputs,
    size_t input_id) {
  if (is_segmented_) {
    FUSER_PERF_SCOPE("FusionKernelRuntime::runMultiKernelWithInput");

    TORCH_INTERNAL_ASSERT(
        inputs.size() == segmented_fusion_->inputs().size(),
        "Inputs were not set up correctly, recieved ",
        inputs.size(),
        " inputs but expecting ",
        segmented_fusion_->inputs().size());

    c10::Device device(c10::DeviceType::CUDA, 0);
    int extent_index_ = 0;
    // Bind input in the tensor_map
    for (const auto i : c10::irange(inputs.size())) {
      runtime_workspace_.tensor_map.emplace(
          segmented_fusion_->inputs()[i], inputs[i]);

      // Bind tensorview inputs values in case some segmented group
      //  needs it down the road.
      // TODO: we probably have done this already up to this point
      //      should consider caching the expression evaluators, both
      //      more convenient and safer than replication
      if (inputs[i].isTensor()) {
        auto aten_tensor = inputs[i].toTensor();
        device = aten_tensor.device();
        for (auto dim_size : aten_tensor.sizes()) {
          runtime_workspace_.tensor_map.emplace(
              runtime_workspace_.group_extent_binding_order[extent_index_++],
              dim_size);
        }
      }
    }

    for (auto group_to_run : runtime_workspace_.group_run_order) {
      // Prepare input vector
      for (auto input : group_to_run->inputs()) {
        runtime_workspace_.group_runtime_inputs.push_back(
            runtime_workspace_.tensor_map.at(input));
      }
      // Run graph segment
      runtime_workspace_.group_runtime_outputs = runKernelWithInput(
          runtime_workspace_.group_runtime_inputs, input_id, group_to_run);

      const auto& group_outputs = group_to_run->outputs();

      // Insert graph segment output to tensor map
      for (unsigned int group_out_i = 0; group_out_i < group_outputs.size();
           group_out_i++) {
        runtime_workspace_.tensor_map.emplace(
            group_outputs[group_out_i],
            runtime_workspace_.group_runtime_outputs[group_out_i]);
      }
      runtime_workspace_.group_runtime_inputs.clear();
      runtime_workspace_.group_runtime_outputs.clear();
    }

    // Produce final global output
    std::vector<IValue> fusion_outputs;
    for (auto output : segmented_fusion_->outputs()) {
      const auto iter = runtime_workspace_.tensor_map.find(output);
      if (iter != runtime_workspace_.tensor_map.end()) {
        fusion_outputs.push_back(iter->second);
      } else {
        bool empty_type_check = output->getDataType().has_value() &&
            output->getDataType().value() == DataType::Float;

        // Only support two cases of empty tensor here, since
        //   this is hot path.
        auto out_tv = output->as<TensorView>();

        // TODO: should be only one of the two once the "empty"
        //  definition has been unified throughout the ops.
        bool empty_tensor_check =
            out_tv->isZeroDim() || out_tv->isEmptyTensor();

        // This is the check for an empty tensor;
        TORCH_INTERNAL_ASSERT(
            empty_tensor_check && empty_type_check,
            "Non empty tensor cannot be found at tensor_map in ",
            __FUNCTION__);

        // TODO: would need to clean up this part when
        //   we have a unified and consistent way to generate
        //   size-0 tensors.
        const auto tensor_options =
            at::TensorOptions().dtype(at::kFloat).device(device);
        fusion_outputs.emplace_back(at::empty({0}, tensor_options));
      }
    }

    std::vector<at::Tensor> fusion_output_tensors;
    std::transform(
        fusion_outputs.begin(),
        fusion_outputs.end(),
        std::back_inserter(fusion_output_tensors),
        [](IValue ival) {
          TORCH_INTERNAL_ASSERT(
              ival.isTensor(),
              "Cannot output non-tensor objects from a fusion.");
          return ival.toTensor();
        });

    runtime_workspace_.tensor_map.clear();
    return fusion_output_tensors;
  } else {
    return runKernelWithInput(inputs, input_id);
  }
}

const std::vector<FusionKernelRuntime::SchedulerEntryPtr>& FusionKernelRuntime::
    schedulers() {
  return heuristics_->heuristicsList();
}

void FusionKernelRuntime::updateHeuristicsLaunchParams(
    FusionHeuristics* update_heuristics) {
  FUSER_PERF_SCOPE("FusionKernelRuntime::updateHeuristicsLaunchParams");
  auto scheduler_list_length = heuristics_->heuristicsList().size();
  TORCH_INTERNAL_ASSERT(
      update_heuristics->heuristicsList().size() == scheduler_list_length);
  for (const auto i : c10::irange(scheduler_list_length)) {
    auto& schedulerPtr = heuristics_->heuristicsList()[i];
    if (schedulerPtr->hasReductionParam()) {
      schedulerPtr->updateLaunchConstraint(
          update_heuristics->heuristicsList()[i]->reductionParams().lparams);
    } else {
      schedulerPtr->updateLaunchConstraint(
          update_heuristics->heuristicsList()[i]->pointwiseParams().lparams);
    }
  }
}

c10::optional<FusionKernelRuntime::HeuristicsPtr> FusionKernelRuntime::
    getMaybeHeuristicsFor(const at::ArrayRef<IValue>& inputs) {
  FUSER_PERF_SCOPE("FusionKernelRuntime::getMaybeHeuristicsFor");
  auto complete_fusion = is_segmented_ ? segmented_fusion_->completeFusion()
                                       : single_kernel_fusion_.get();
  SchedulerRuntimeInfo runtime_info(complete_fusion, inputs);
  precomputed_integers_->bindFusionInputs(inputs);
  precomputed_integers_->evaluate();
  runtime_info.expressionEvaluator().bindPrecomputedIntegers(
      precomputed_integers_.get());

  c10::optional<FusionKernelRuntime::HeuristicsPtr> ret;
  // Segmented case, need to iterate over all segmented groups
  if (is_segmented_) {
    ret = std::make_unique<FusionHeuristics>();
    size_t total_groups = segmented_fusion_->groups().size();
    for (const auto group_index : c10::irange(total_groups)) {
      auto group = segmented_fusion_->groups()[group_index];

      auto maybe_scheduler_entry = group->getMaybeSchedulerEntry(runtime_info);
      if (!maybe_scheduler_entry.has_value()) {
        return c10::nullopt;
      }
      auto scheduler_entry = std::move(maybe_scheduler_entry.value());
      if (!scheduler_entry->sameAs(
              heuristics_->heuristicsList()[group_index].get())) {
        return c10::nullopt;
      }
      ret.value()->emplaceBack(std::move(scheduler_entry));
    }

    return ret;
  }

  // Un-segmented case, just check the complete fusion
  auto& complete_fusion_scheduler = schedulers()[0];
  auto complete_fusion_heuristic = complete_fusion_scheduler->heuristc();
  if (!SchedulerEntry::canSchedule(
          complete_fusion_heuristic,
          complete_fusion,
          runtime_info,
          single_kernel_fusion_data_cache_.get())) {
    return c10::nullopt;
  }

  ret = std::make_unique<FusionHeuristics>(
      complete_fusion_heuristic,
      runtime_info,
      single_kernel_fusion_data_cache_.get());
  if (!complete_fusion_scheduler->sameAs(
          ret.value()->heuristicsList()[0].get())) {
    return c10::nullopt;
  }

  return ret;
}

void GraphCache::createFusion(const std::shared_ptr<Graph>& graph) {
  FUSER_PERF_SCOPE("GraphCache::createFusion");

  fusion_executor_cache_ =
      std::make_unique<FusionExecutorCache>(parseJitIR(graph));

  num_of_outputs_ = graph->outputs().size();
}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
GraphCache::GraphCache(const std::shared_ptr<Graph>& graph) {
  FUSER_PERF_SCOPE("GraphCache::GraphCache");
  TORCH_INTERNAL_ASSERT(
      IsNewExecutorEnabled(), "legacy executor is not supported by nvfuser");

  GRAPH_DEBUG("GraphCache constructor: ", this);
  GRAPH_DUMP("GraphCache created for graph", graph);
  createFusion(graph);
}

std::vector<at::Tensor> GraphCache::runGraphWithInputs(
    const at::ArrayRef<IValue>& inputs) {
  FUSER_PERF_SCOPE("GraphCache::runGraphWithInputs");

  GRAPH_DEBUG("running GraphCache: ", this);
  auto outputs = fusion_executor_cache_->runFusionWithInputs(inputs);
  TORCH_INTERNAL_ASSERT(
      outputs.size() == num_of_outputs_,
      "FusionExecutorCache returned ",
      outputs.size(),
      " outputs, doesn't match computational graph, which requires ",
      num_of_outputs_);

  return outputs;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
