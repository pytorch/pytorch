#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>

#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/parser.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

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

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
void debugPrint(const TensorTypePtr& type) {
  std::stringstream sizes_s;
  if (auto sizes = type->symbolic_sizes().sizes()) {
    for (const auto& shape_symbol : *sizes) {
      if (shape_symbol.is_static()) {
        sizes_s << shape_symbol.static_size() << ", ";
      } else {
        sizes_s << "s(" << *reinterpret_cast<const int64_t*>(&shape_symbol)
                << "), ";
      }
    }
  } else {
    sizes_s << "no size available";
  }
  std::cout << "sizes:" << sizes_s.str() << std::endl;
  if (const auto& stride_properties = type->stride_properties().sizes()) {
    std::stringstream stride_s;
    std::stringstream index_s;
    std::stringstream contig_s;

    for (const auto& stride_property : *stride_properties) {
      if (stride_property.has_value() && stride_property->stride_.has_value()) {
        stride_s << *stride_property->stride_ << ", ";
      } else {
        stride_s << "?, ";
      }
      if (stride_property.has_value() &&
          stride_property->stride_index_.has_value()) {
        index_s << *stride_property->stride_index_ << ", ";
      } else {
        index_s << "?, ";
      }
      if (stride_property.has_value() &&
          stride_property->contiguous_.has_value()) {
        contig_s << *stride_property->contiguous_ << ", ";
      } else {
        contig_s << "?, ";
      }
    }
    std::cout << "stride: " << stride_s.str() << std::endl;
    std::cout << "stride index: " << index_s.str() << std::endl;
    std::cout << "contiguous: " << contig_s.str() << std::endl;
  } else {
    std::cout << "no stride properties available" << std::endl;
  }
}
#pragma clang diagnostic pop

at::DimVector graphReductionAxes(
    const std::shared_ptr<Graph>& graph,
    bool& simple_reduction) {
  FUSER_PERF_SCOPE("graphReductionAxes");
  simple_reduction = true;

  at::DimVector reduction_axes;
  // TODO: let check that we have only single reduction node in the graph.
  for (const auto& n : graph->nodes()) {
    if (isReductionToSizeNode(n)) {
      // TODO: we don't support permutation with ReductionToSize;
      simple_reduction = false;
      reduction_axes.clear();
      return reduction_axes;
    } else if (isReductionNode(n)) {
      // TODO: we should return empty when `keepdim` is True?
      auto dims_list = constant_as<c10::List<int64_t>>(n->input(1));
      TORCH_INTERNAL_ASSERT(
          dims_list.has_value(), "reduction axes should be constant");
      for (const auto dim : dims_list->vec()) {
        reduction_axes.emplace_back(static_cast<int>(dim));
      }
      // we should return here, but we don't!
      // We continue the traversal and check for other reduction node. Because
      // our permutation doesn't really support intermediate reduction; Continue
      // traversal would trigger the `TORCH_INTERNAL_ASSERT`, it's not ideal but
      // at least it's not silent error.
    }
    // TODO: this doesn't apply any more, clean it up
  }
  return reduction_axes;
}

// TODO(CONTIGUITY)
at::DimVector getPermutationPerSortedStride(const TensorTypePtr& type) {
  FUSER_PERF_SCOPE("getPermutationPerSortedStride");

  // `permute_seq` is the returned permutation to achieve sorted stride;
  at::DimVector permute_seq;

  auto stride_properties = type->stride_properties().sizes();

  // no consistent permutation available, we just don't do it;
  if (!stride_properties.has_value()) {
    return permute_seq;
  }

  // TODO: reuse this;
  const int rank = static_cast<int>(stride_properties->size());

  // stores axes with stride_index;
  std::set<int> ordered_axes;

  // TODO: this does not support broadcast yet;
  for (int i = 0; i < rank; i++) {
    if ((*stride_properties)[i].has_value() &&
        (*stride_properties)[i]->stride_index_.has_value()) {
      ordered_axes.insert((*stride_properties)[i]->stride_index_.value());
    }
  }

  int unallocated_axis = 0;
  // we push from slowest to fastest
  for (int i = rank - 1; i >= 0; i--) {
    if ((*stride_properties)[i].has_value() &&
        (*stride_properties)[i]->stride_index_.has_value()) {
      permute_seq.emplace_back((*stride_properties)[i]->stride_index_.value());
    } else {
      // no designated axis for this slot, so we push an axis w/o designated
      // order;
      while (ordered_axes.count(unallocated_axis) != 0) {
        ++unallocated_axis;
      }
      permute_seq.emplace_back(unallocated_axis++);
    }
  }
  return permute_seq;
}

at::DimVector inversePermutation(
    const at::DimVector& permuted,
    const std::vector<size_t>& reduction_axes) {
  if (permuted.empty()) {
    return permuted;
  }
  int rank = static_cast<int>(permuted.size());

  if (!reduction_axes.empty()) {
    int red_rank = rank - static_cast<int>(reduction_axes.size());

    // see [ NOTE - reduction in graph ] part 1.
    // a. we skip axes that were eliminated by reduction;
    // b. we adjust axes index that were affected by reduction;
    at::DimVector adjusted_permutation;
    for (const auto& dim : permuted) {
      int adjusted_offset = 0;
      for (const auto& red_dim : reduction_axes) {
        if (red_dim < (unsigned long)dim) {
          adjusted_offset++; // 1.b
        } else if (red_dim == (unsigned long)dim) {
          adjusted_offset = -1; // 1.a
          break;
        }
      }
      if (adjusted_offset >= 0) {
        adjusted_permutation.emplace_back(dim - adjusted_offset);
      }
    }

    at::DimVector permutation(red_rank, -1);
    for (int i = 0; i < red_rank; i++) {
      permutation[adjusted_permutation[i]] = i;
    }
    return permutation;
  } else {
    at::DimVector permutation(rank, -1);
    for (int i = 0; i < rank; i++) {
      permutation[permuted[i]] = i;
    }
    return permutation;
  }
}

void encodeBuffer(size_t value, std::string& buffer) {
  const char* v = reinterpret_cast<char*>(&value);
  for (size_t i = 0; i < sizeof(size_t); i++) {
    buffer.push_back(*(v++));
  }
}

} // namespace

InputsIdLookup::IdLookupReturn InputsIdLookup::lookupId(
    const at::ArrayRef<IValue>& inputs) {
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

FusionExecutorCache::FusionExecutorCache(std::unique_ptr<Fusion>&& fusion)
    : fusion_(std::move(fusion)) {
  FUSER_PERF_SCOPE("FusionExecutorCache::FusionExecutorCache");

  // case of segmented fusion
  // TODO: might be worthwhile re-using the SchedulerEntry infrastructure for
  //       single-kernel fusion as well.
  const bool segmented =
      !SchedulerEntry::proposeHeuristics(fusion_.get()).has_value();

  if (segmented) {
    fusion_segments_ = fusion_->segment();
    fusion_segment_runtime_cache_.initCache(fusion_segments_.get());
    if (isDebugDumpEnabled(DebugDumpOption::FusionSegments)) {
      fusion_segments_->print();
    }
    return;
  }

  // In the case that the fusion isn't segmented but user
  //  wants segmented fusion in the debug print. Will
  //  print math of the composite fusion as placeholder
  if (isDebugDumpEnabled(DebugDumpOption::FusionSegments)) {
    fusion_->printMath();
  }

  // avoid putting `has_nontrivial_reduction_` in the initializer list
  has_nontrivial_reduction_ = fusion_->hasReduction();

  if (has_nontrivial_reduction_) {
    FusionGuard fg(fusion_.get());

    // Use dependency check to find the reduction tv as it returns used values
    // instead of exprs.

    // The call is relatively heavy weight, consider caching
    auto all_values = DependencyCheck::getAllValsBetween(
        {fusion_->inputs().begin(), fusion_->inputs().end()},
        fusion_->outputs());

    // Separate the reduction TensorViews from the other TensorViews
    // Ignore input TensorViews
    for (auto tv : ir_utils::filterByType<TensorView>(all_values)) {
      if (tv->hasReduction()) {
        reduction_tv_.push_back(tv);
      }
    }

    TORCH_INTERNAL_ASSERT(
        !reduction_tv_.empty(),
        "Could not find any reduction TensorViews in the fusion.");
  }
}

std::vector<at::Tensor> FusionExecutorCache::runFusionWithInputs(
    const at::ArrayRef<IValue>& inputs) {
  FUSER_PERF_SCOPE("runFusionWithInputs");

  // TODO: This seems overly conservative to send to normalization scheduler. We
  // may want to check there's a "residual path" around the reduction.
  auto detect_normalization_fusion = [&]() {
    for (auto expr : fusion_->exprs()) {
      if (expr->getExprType() == ExprType::BroadcastOp) {
        auto output = expr->output(0);
        auto input_def_expr = expr->input(0)->definition();
        if (!fusion_->unordered_uses(output).empty() &&
            input_def_expr != nullptr &&
            input_def_expr->getExprType() == ExprType::ReductionOp) {
          return true;
        }
      }
    }
    return false;
  };

  LaunchParams launch_params;

  // get unique id `unique_id` for given input set `inputs`;
  auto id_lookup_ret = inputs_id_lookup_.lookupId(inputs);
  if (id_lookup_ret.eviction) {
    evictCache(id_lookup_ret.evict_id);
  }

  const size_t unique_id = id_lookup_ret.id;
  const int device_index = getCommonDeviceCUDA(inputs);
  TORCH_CHECK(device_index >= 0, "device is not coherent for fusion inputs");

  // Manage Segmented Fusion through FusionSegmentRuntimeCache
  if (isSegmented()) {
    auto seg_runtime = fusion_segment_runtime_cache_.getRt(inputs, unique_id);
    // Propagate the unique_id so the contained fusionExecutors in the runtime
    //  entry will cache the buffer sizes and launch params based on this id.
    return seg_runtime->runWithInput(inputs, unique_id);
  }

  if (code_to_fe_lookup_.count(unique_id) == 0) {
    // enter when we get a new input set. We need to search for compatible
    // entries in cached `FusionExecutor` or compile new one as needed.

    // caching strategy is different for pw-fusion and reduction-fusion.
    if (has_nontrivial_reduction_) {
      bool isNormalizationFusion = detect_normalization_fusion();
      // Generate the reduction parameters
      auto reduction_params = (isNormalizationFusion)
          ? getNormalizationHeuristics(fusion_.get(), inputs, reduction_tv_)
          : getReductionHeuristics(
                fusion_.get(), inputs, reduction_tv_.front());

      TORCH_INTERNAL_ASSERT(
          reduction_params.has_value(),
          "Error getting reduction heuristics for scheduling.");

      launch_params = reduction_params.value().lparams;

      // cache based on launch parameters
      auto fusion_executor =
          &red_fusion_executor_cache_[device_index][reduction_params.value()];

      if (!fusion_executor->compiled()) {
        // HEURISTIC NOT COMPILED, COMPILE A KERNEL

        // We clone *fusion_ to fusion so we can leave the unscheduled
        // computational graph intact for future compilation.
        Fusion fusion_clone = *fusion_;
        FusionGuard fg(&fusion_clone);

        // Separate the reduction TensorViews from the other TensorViews
        // Ignore input TensorViews
        std::vector<TensorView*> clone_reduction_tv;
        std::vector<TensorView*> clone_other_tv;
        auto all_values = DependencyCheck::getAllValsBetween(
            {fusion_clone.inputs().begin(), fusion_clone.inputs().end()},
            fusion_clone.outputs());

        for (auto tv : ir_utils::filterByType<TensorView>(all_values)) {
          if (tv->hasReduction()) {
            clone_reduction_tv.push_back(tv);
          } else if (!fusion_clone.hasInput(tv)) {
            clone_other_tv.push_back(tv);
          }
        }

        if (isNormalizationFusion) {
          scheduleNormalization(
              &fusion_clone,
              reduction_params.value(),
              clone_reduction_tv,
              clone_other_tv);
        } else {
          auto single_reduction_tv = clone_reduction_tv.front();

          // Heavy weight call
          auto outputs_of_reduction =
              DependencyCheck::getAllOutputsOf({single_reduction_tv});

          auto tv_entries =
              ir_utils::filterByType<TensorView>(outputs_of_reduction);

          std::vector<TensorView*> tv_outputs_of_reduction(
              tv_entries.begin(), tv_entries.end());

          scheduleReduction(
              &fusion_clone,
              reduction_params.value(),
              single_reduction_tv,
              tv_outputs_of_reduction);
        }

        // This means we have not found a previously generated kernel that is
        // compatible with the new reduction params. We need to finish codegen.
        CompileOptions options;
        options.device = c10::Device(DeviceType::CUDA, device_index);
        fusion_executor->compileFusion(&fusion_clone, options);
      }
      // record new short cut to `FusionExecutor`
      code_to_fe_lookup_[unique_id] = fusion_executor;

    } else {
      // Handle pointwise operations
      if (pw_fusion_executor_cache_.count(device_index) == 0) {
        pw_fusion_executor_cache_[device_index] =
            std::make_unique<FusionExecutor>();
        CompileOptions options;
        options.device = c10::Device(DeviceType::CUDA, device_index);
        // We do not need to copy fusion_ because we are not generating
        // multiple kernels for point-wise operations.
        auto fusion_clone = *fusion_;
        scheduleFusion(&fusion_clone, inputs);
        pw_fusion_executor_cache_[device_index]->compileFusion(
            &fusion_clone, options);
      }
      // record new short cut to `FusionExecutor`
      code_to_fe_lookup_[unique_id] =
          pw_fusion_executor_cache_[device_index].get();
    }
  }

  return code_to_fe_lookup_[unique_id]->runFusion(
      inputs, launch_params, unique_id);
}

FusionSegmentRuntime::FusionSegmentRuntime(
    SegmentedFusion* segmented_fusion,
    std::unique_ptr<SegmentHeuristics>& heuristics,
    size_t input_id)
    : executors_(segmented_fusion->groups().size()),
      heuristics_(std::move(heuristics)),
      segmented_fusion_(segmented_fusion) {}

// Largely duplicated from FusionExecutorCache
std::vector<at::Tensor> FusionSegmentRuntime::runSegmentWithInput(
    SegmentedGroup* sg,
    const at::ArrayRef<IValue>& inputs,
    size_t input_id) {
  auto group_id = sg->groupId();
  const int device_index = getCommonDeviceCUDA(inputs);
  LaunchParams launch_params;

  auto scheduler_entry = schedulers()[group_id].get();

  // Check that the heuristics are matched
  TORCH_INTERNAL_ASSERT(scheduler_entry->heuristc() == sg->heuristic());

  if (!executors_[group_id].compiled()) {
    std::unique_ptr<Fusion> fusion_seg = segmented_fusion_->makeFusion(sg);
    CompileOptions options;
    options.device = c10::Device(DeviceType::CUDA, device_index);
    FusionGuard fg(fusion_seg.get());
    scheduler_entry->schedule(fusion_seg.get());
    executors_[group_id].compileFusion(fusion_seg.get(), options);
  }

  // Load launch params for reduction and normalization kernels
  if (scheduler_entry->hasParam()) {
    launch_params = scheduler_entry->params().lparams;
  }

  return executors_[group_id].runFusion(inputs, launch_params, input_id);
}

std::vector<at::Tensor> FusionSegmentRuntime::runWithInput(
    const at::ArrayRef<IValue>& inputs,
    size_t input_id) {
  TORCH_INTERNAL_ASSERT(
      inputs.size() == segmented_fusion_->inputs().size(),
      "Inputs were not set up correctly, recieved ",
      inputs.size(),
      " inputs but expecting ",
      segmented_fusion_->inputs().size());

  // Map to keep track of currently available tensors
  std::unordered_map<Val*, IValue> tensor_map;

  // Bind input in the tensor_map
  for (size_t i = 0; i < inputs.size(); i++) {
    tensor_map.emplace(segmented_fusion_->inputs()[i], inputs[i]);

    // Bind tensorview inputs values in case some segmented group
    //  needs it down the road.
    // TODO: we probably have done this already up to this point
    //      should consider caching the expression evaluators, both
    //      more convenient and safer than replication
    if (inputs[i].isTensor()) {
      auto aten_tensor = inputs[i].toTensor();
      TORCH_INTERNAL_ASSERT(
          segmented_fusion_->inputs()[i]->getValType() == ValType::TensorView);
      auto input_tv = segmented_fusion_->inputs()[i]->as<TensorView>();
      auto root_dom = TensorDomain::noReductions(input_tv->getRootDomain());
      for (size_t dim = 0; dim < root_dom.size(); dim++) {
        const auto extent = root_dom[dim]->extent();
        const auto value = aten_tensor.sizes()[dim];
        tensor_map.emplace(extent, value);
      }
    }
  }

  // Keep track of groups that has run
  std::vector<bool> group_ran(segmented_fusion_->groups().size(), false);

  while (!std::all_of(
      group_ran.begin(), group_ran.end(), [](bool b) { return b; })) {
    bool one_ran = false;

    // Find the first segment with all inputs available to run
    for (size_t group_i = 0; group_i < segmented_fusion_->groups().size();
         group_i++) {
      auto& group = segmented_fusion_->groups()[group_i];
      if (group_ran[group_i]) {
        continue;
      }
      const auto& group_inputs = group->inputs();
      bool ready_to_run = std::all_of(
          group_inputs.begin(), group_inputs.end(), [&tensor_map](Val* val) {
            return tensor_map.find(val) != tensor_map.end();
          });

      if (ready_to_run) {
        std::vector<IValue> group_runtime_inputs;
        group_runtime_inputs.reserve(group_inputs.size());

        // Prepare input vector
        for (auto input : group_inputs) {
          group_runtime_inputs.push_back(tensor_map.at(input));
        }

        // Run graph segment
        auto group_runtime_outputs =
            runSegmentWithInput(group, group_runtime_inputs, input_id);

        const auto& group_outputs = group->outputs();

        // Insert graph segment output to tensor map
        for (size_t group_out_i = 0; group_out_i < group_outputs.size();
             group_out_i++) {
          tensor_map.emplace(
              group_outputs[group_out_i], group_runtime_outputs[group_out_i]);
        }
        group_ran[group_i] = true;
        one_ran = true;
      }
    }
    TORCH_INTERNAL_ASSERT(
        one_ran,
        "Couldn't run all groups, something must have gone wrong in segmentation.");
  }

  // Produce final global output
  std::vector<IValue> fusion_outputs;
  for (auto output : segmented_fusion_->outputs()) {
    const auto iter = tensor_map.find(output);
    if (iter != tensor_map.end()) {
      fusion_outputs.push_back(iter->second);
    } else {
      // This is the check for an empty tensor;
      TORCH_INTERNAL_ASSERT(
          output->as<TensorView>()->nDims() == 0 &&
              output->getDataType().has_value() &&
              output->getDataType().value() == DataType::Float,
          "Non empty tensor cannot be found at tensor_map in ",
          __FUNCTION__);
      fusion_outputs.emplace_back(at::Tensor());
    }
  }

  std::vector<at::Tensor> fusion_output_tensors;
  std::transform(
      fusion_outputs.begin(),
      fusion_outputs.end(),
      std::back_inserter(fusion_output_tensors),
      [](IValue ival) {
        TORCH_INTERNAL_ASSERT(
            ival.isTensor(), "Cannot output non-tensor objects from a fusion.");
        return ival.toTensor();
      });

  return fusion_output_tensors;
}

const std::vector<FusionSegmentRuntime::SchedulerEntryPtr>&
FusionSegmentRuntime::schedulers() {
  return heuristics_->heuristics();
}

namespace {
using HashType = FusionSegmentRuntime::HashType;
// Use a slightly more nontrivial combine to avoid collision
//  (from Boost)
inline HashType combineHash(HashType a, HashType b) {
  return a ^
      (b + 0x9e3779b9 + // NOLINT(cppcoreguidelines-avoid-magic-numbers)
       (a << 6) + // NOLINT(cppcoreguidelines-avoid-magic-numbers)
       (a >> 2)); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
}
} // namespace

FusionSegmentRuntime::HashType FusionSegmentRuntime::getHash(
    SegmentHeuristics* sh) {
  HashType h = 0;
  for (auto& se_pt : sh->heuristics()) {
    h = combineHash(h, SchedulerEntryHash()(*se_pt));
  }
  return h;
}

FusionSegmentRuntime::HeuristicTag::HeuristicTag(SegmentHeuristics* sh) {
  heuristics_ = sh;
  hash_ = FusionSegmentRuntime::getHash(sh);
}

bool FusionSegmentRuntime::HeuristicTag::operator==(
    const FusionSegmentRuntime::HeuristicTag& other) const {
  if (heuristics_->heuristics().size() !=
      other.heuristics_->heuristics().size()) {
    return false;
  }

  auto& heuristics = heuristics_->heuristics();
  return std::equal(
      heuristics.begin(),
      heuristics.end(),
      other.heuristics_->heuristics().begin(),
      [](const SchedulerEntryPtr& a, const SchedulerEntryPtr& b) {
        return a->sameAs(b.get());
      });
}

void FusionSegmentRuntimeCache::evictId(size_t input_id) {
  TORCH_INTERNAL_ASSERT(id_to_rt_.count(input_id) != 0);

  // Evict the stored input tensor meta data
  //  corresponding to input_id
  id_to_rt_.at(input_id)->evictCache(input_id);
  id_to_rt_.erase(input_id);
}

FusionSegmentRuntime* FusionSegmentRuntimeCache::getRt(
    const at::ArrayRef<IValue>& inputs,
    size_t input_id) {
  // Look up by input_id first
  auto seg_runtime = getRtById(input_id);
  if (seg_runtime == nullptr) {
    // if id misses, lookup by heuristics
    //  this will create new entry if not found
    seg_runtime = getRtByHeuristics(inputs, input_id);
  }
  return seg_runtime;
}

FusionSegmentRuntime* FusionSegmentRuntimeCache::getRtById(size_t input_id) {
  if (id_to_rt_.count(input_id) == 0) {
    return nullptr;
  }
  return id_to_rt_.at(input_id);
}

FusionSegmentRuntime* FusionSegmentRuntimeCache::getRtByHeuristics(
    const at::ArrayRef<IValue>& inputs,
    size_t input_id) {
  auto dev_id = getCommonDeviceCUDA(inputs);
  auto heuristics = segmented_fusion_->makeHeuristics(inputs);
  HeuristicTag tag(heuristics.get());
  auto rt = at(dev_id, tag);

  // Heuristics miss
  if (rt == nullptr) {
    // Construct new runtime instance
    auto new_rt = std::make_unique<FusionSegmentRuntime>(
        segmented_fusion_, heuristics, input_id);
    rt = new_rt.get();

    // Cache the new instance
    insertEntry(dev_id, tag, std::move(new_rt));
  }

  // Cache this new id
  id_to_rt_[input_id] = rt;

  return rt;
}

void FusionSegmentRuntimeCache::initCache(SegmentedFusion* sf) {
  segmented_fusion_ = sf;
}

FusionSegmentRuntime* FusionSegmentRuntimeCache::at(
    int dev_id,
    HeuristicTag tag) {
  // Get cache for the device id
  auto& run_time_cache_ptr = seg_runtime_cache_group_[dev_id];

  // Check empty
  if (!run_time_cache_ptr) {
    return nullptr;
  }

  // Get entry from cache
  auto& cache_entry_ptr = run_time_cache_ptr->operator[](tag);

  // Check empty
  if (!cache_entry_ptr) {
    return nullptr;
  }

  // Return non-empty entry
  return cache_entry_ptr.get();
}

void FusionSegmentRuntimeCache::insertEntry(
    int dev_id,
    HeuristicTag tag,
    SegRuntimePtr&& rt_pt) {
  auto& run_time_cache_ptr = seg_runtime_cache_group_[dev_id];

  if (!run_time_cache_ptr) {
    // First time seeing this device
    // run_time_cache_ptr is a reference so will be auto updated
    // could have updated run_time_cache_ptr to save
    // one hashing but too confusing to read
    seg_runtime_cache_group_[dev_id] = std::make_unique<SegRuntimeCache>();
  }

  run_time_cache_ptr->operator[](tag) = std::move(rt_pt);
}

bool GraphCache::requiresPermutation() {
  if (!support_permutation_) {
    return false;
  }

  const size_t input_rank = input_permutation_.size();
  for (size_t i = 0; i < input_rank; i++) {
    if (input_permutation_[i] != (long)i) {
      return true;
    }
  }
  // Check if output agrees
  const size_t pw_output_rank = pw_output_permutation_.size();
  for (size_t i = 0; i < pw_output_rank; i++) {
    TORCH_INTERNAL_ASSERT(
        pw_output_permutation_[i] == (long)i,
        "permutation of output and input is not consistent");
  }
  const size_t reduction_output_rank = reduction_output_permutation_.size();
  for (size_t i = 0; i < reduction_output_rank; i++) {
    TORCH_INTERNAL_ASSERT(
        reduction_output_permutation_[i] == (long)i,
        "permutation of output and input is not consistent");
  }
  return false;
}

void GraphCache::extractPermutation(const TensorTypePtr& acc_type) {
  input_permutation_ = getPermutationPerSortedStride(acc_type);
  reduction_output_permutation_ =
      inversePermutation(input_permutation_, toVector(reduction_axes_));
  pw_output_permutation_ = inversePermutation(input_permutation_, {});
}

void GraphCache::createFusion(const std::shared_ptr<Graph>& graph) {
  FUSER_PERF_SCOPE("GraphCache::createFusion");

  // permute inputs on `Graph` to sort dimensions on common stride order;
  if (requiresPermutation()) {
    // TODO: lambda is a bad idea, the logic in this function is too tricky and
    //       should be properly tested to ensure correctness.
    // lambda to permute `TensorType` axes per `input_permutation_`
    auto type_permute_fn = [this](const TensorTypePtr& type) {
      // std::vector<c10::ShapeSymbol> vec_shape_symbol =
      // type->symbolic_sizes().sizes().value();
      auto vec_shape_symbol = type->symbolic_sizes().sizes().value();
      // std::vector<c10::optional<c10::Stride>> vec_optional_stride =
      // type->stride_properties().sizes().value();
      auto vec_optional_stride = type->stride_properties().sizes().value();

      int rank = static_cast<int>(type->dim().value());

      std::vector<c10::ShapeSymbol> permuted_vec_ss;
      std::vector<c10::optional<c10::Stride>> permuted_vec_optional_stride;
      for (int i = 0; i < rank; i++) {
        permuted_vec_ss.emplace_back(
            vec_shape_symbol[this->input_permutation_[i]]);
        // permutation doesn't change contiguity info, nor does it change
        // stride; The only thing affected is stride_index_;
        if (vec_optional_stride[i].has_value()) {
          c10::optional<size_t> index = vec_optional_stride[i]->stride_index_;
          if (index.has_value()) {
            for (int j = 0; j < rank; j++) {
              // follow the permutation to resolve the new stride_index;
              if (this->input_permutation_[j] == (long)index.value()) {
                index = j;
                break;
              }
            }
          }
          permuted_vec_optional_stride.emplace_back(c10::Stride(
              /*stride_index=*/index,
              /*contiguous=*/vec_optional_stride[i]->contiguous_,
              /*stride=*/vec_optional_stride[i]->stride_));
        } else {
          permuted_vec_optional_stride.emplace_back(c10::nullopt);
        }
      }

      return TensorType::create(
          type->scalarType(),
          type->device(),
          permuted_vec_ss,
          permuted_vec_optional_stride,
          type->requires_grad());
    }; // closing lambda
    for (auto input : graph->inputs()) {
      if (auto input_type = input->type()->cast<TensorType>()) {
        input->setType(type_permute_fn(input_type));
      }
    }

    if (!reduction_axes_.empty()) {
      // see [ NOTE - reduction in graph ] part 2.
      for (auto n : graph->nodes()) {
        if (isReductionNode(n)) {
          auto dims_list = constant_as<c10::List<int64_t>>(n->input(1));
          TORCH_INTERNAL_ASSERT(
              dims_list.has_value(), "reduction axes should be constant");
          std::vector<int64_t> adjusted_reduction_axes;
          for (const auto dim : dims_list->vec()) {
            // adjust reduction axis to be the permuted axis;
            for (size_t j = 0; j < input_permutation_.size(); j++) {
              // follow the permutation to resolve the new reduction axes;
              if (input_permutation_[j] == dim) {
                adjusted_reduction_axes.emplace_back(j);
                break;
              }
            }
          }
          graph->setInsertPoint(n);
          auto const_ival_axes =
              graph->insertConstant(IValue(adjusted_reduction_axes));
          n->replaceInput(1, const_ival_axes);
        }
      }
    }
  }

  fusion_executor_cache_ =
      std::make_unique<FusionExecutorCache>(parseJitIR(graph));
}

GraphCache::GraphCache(const std::shared_ptr<Graph>& graph) {
  FUSER_PERF_SCOPE("GraphCache::GraphCache");
  TORCH_INTERNAL_ASSERT(
      IsNewExecutorEnabled(), "legacy executor is not supported by nvfuser");

  // [ NOTE - reduction in graph ]
  //
  // reduction complicates our permutation in integration, it addes two things:
  // 1. we need to adjust xxx_output_permutation_;
  //    because of dimension elimination during permutation (not necessarily,
  //    given the `keepdim` argument.) this needs to be accommodated later when
  //    we added the support.
  // 2. adjust reduction axes for the permutation;
  //    permute changes the semantics of axes, we need to update the reduction
  //    axes in the graph in order to match the behavior;
  reduction_axes_ = graphReductionAxes(graph, support_permutation_);

  // TODO: reduction with permutation is tricky now as we might support complex
  // topology in graph with segmented fusion.
  if (support_permutation_) {
    // run over inputs to extract common types;
    TensorTypePtr acc_type = TensorType::get();
    for (const auto& input : graph->inputs()) {
      // only check tensor types;
      if (auto input_type = input->type()->cast<TensorType>()) {
        if (acc_type->dim().has_value()) {
          // TODO: I think merge cannot handle broadcast - Go verify it later;
          // TODO: Since we are only handling permutation here, we should just
          //       merge the stride_index_;
          acc_type = acc_type->merge(*input_type);
        } else {
          acc_type = input_type;
        }
      }
    }
    extractPermutation(acc_type);
  }
  createFusion(graph);
}

std::vector<at::Tensor> GraphCache::runGraphWithInputs(
    const at::ArrayRef<IValue>& inputs) {
  FUSER_PERF_SCOPE("runGraphWithInputs");

  // GraphCache need to permute inputs/outputs to accommodate dimension
  // coalescing
  if (requiresPermutation()) {
    std::vector<IValue> permuted_inputs;
    permuted_inputs.reserve(inputs.size());
    for (const auto& input : inputs) {
      if (input.isTensor()) {
        permuted_inputs.emplace_back(
            input.toTensor().permute(input_permutation_));
      } else {
        permuted_inputs.emplace_back(input);
      }
    }
    auto outputs = fusion_executor_cache_->runFusionWithInputs(permuted_inputs);
    std::vector<at::Tensor> permuted_outputs;
    permuted_outputs.reserve(outputs.size());
    for (const auto& output : outputs) {
      // This is to address the issue that not all outputs from a reduction
      // fusion are reduced tensor; We support intermediate tensors to be output
      if (static_cast<size_t>(output.dim()) == pw_output_permutation_.size()) {
        permuted_outputs.emplace_back(output.permute(pw_output_permutation_));
      } else if (
          static_cast<size_t>(output.dim()) ==
          reduction_output_permutation_.size()) {
        permuted_outputs.emplace_back(
            output.permute(reduction_output_permutation_));
      } else {
        TORCH_INTERNAL_ASSERT(
            false,
            "Something went wrong with integration permutation, can't find a consistent permutation for output in fusion");
      }
    }
    return permuted_outputs;
  } else {
    return fusion_executor_cache_->runFusionWithInputs(inputs);
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
