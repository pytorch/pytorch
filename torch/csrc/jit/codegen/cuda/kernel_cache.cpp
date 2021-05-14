#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>

#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/parser.h>
#include <torch/csrc/jit/codegen/cuda/scheduler.h>
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
    // NOLINTNEXTLINE(bugprone-signed-char-misuse)
    index = cur_index;
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

at::DimVector graphReductionAxes(const std::shared_ptr<Graph>& graph) {
  FUSER_PERF_SCOPE("graphReductionAxes");

  at::DimVector reduction_axes;
  // TODO: let check that we have only single reduction node in the graph.
  for (const auto& n : graph->nodes()) {
    if (isReductionNode(n)) {
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

} // namespace

InputsIdLookup::IdLookupReturn InputsIdLookup::lookupId(
    const at::ArrayRef<IValue>& inputs) {
  IdLookupReturn ret;
  std::stringstream encoded_inputs;
  for (const auto& input : inputs) {
    if (input.isTensor()) {
      auto& input_tensor = input.toTensor();

      encoded_inputs << ";";
      auto sep = "";
      for (auto size : input_tensor.sizes()) {
        encoded_inputs << sep << size;
        sep = ",";
      }
      encoded_inputs << "@";
      sep = "";
      for (auto stride : input_tensor.strides()) {
        encoded_inputs << sep << stride;
        sep = ",";
      }
      encoded_inputs << "@" << input_tensor.device().str();
    } else {
      // encode s for scalar;
      encoded_inputs << ";s";
    }
  }
  auto& id_iter_pair = encoding_lookup_[encoded_inputs.str()];

  // short-cut to leave LRU entry as is;
  if (id_iter_pair.lru_iter == used_entry_.begin()) {
    ret.id = id_iter_pair.id;
    return ret;
  }

  if (id_iter_pair.id == 0) {
    // no entry existed for given input set, set id for given entry
    id_iter_pair.id = current_id_++;
    if (used_entry_.size() == max_cache_size_) {
      // pop least recently used cache;
      const auto& remove_iter = encoding_lookup_.find(used_entry_.back());
      used_entry_.pop_back();
      ret.evict_id = remove_iter->second.id;
      ret.eviction = true;
      encoding_lookup_.erase(remove_iter);
    }
  } else {
    used_entry_.erase(id_iter_pair.lru_iter);
  }

  ret.id = id_iter_pair.id;
  id_iter_pair.lru_iter =
      used_entry_.insert(used_entry_.begin(), encoded_inputs.str());
  return ret;
}

FusionExecutorCache::FusionExecutorCache(std::unique_ptr<Fusion>&& fusion)
    : fusion_(std::move(fusion)) {
  FUSER_PERF_SCOPE("FusionExecutorCache::FusionExecutorCache");
  // avoid putting `has_reduction_` in the initializer list
  has_reduction_ = fusion_->hasReduction();
}

std::vector<at::Tensor> FusionExecutorCache::runFusionWithInputs(
    const at::ArrayRef<IValue>& inputs) {
  FUSER_PERF_SCOPE("runFusionWithInputs");

  // get unique id `unique_id` for given input set `inputs`;
  auto id_lookup_ret = inputs_id_lookup_.lookupId(inputs);
  if (id_lookup_ret.eviction) {
    evictCache(id_lookup_ret.evict_id);
  }

  const size_t unique_id = id_lookup_ret.id;
  const int device_index = getCommonDeviceCUDA(inputs);
  TORCH_CHECK(device_index >= 0, "device is not coherent for fusion inputs");

  LaunchParams launch_params;
  if (code_to_fe_lookup_.count(unique_id) == 0) {
    // enter when we get a new input set. We need to search for compatible
    // entries in cached `FusionExecutor` or compile new one as needed.

    // caching strategy is different for pw-fusion and reduction-fusion.
    if (has_reduction_) {
      // Grab the fusion to analyze for heuristics
      FusionGuard fg(fusion_.get());

      TensorView* reduction_tv = nullptr;
      // Use dependency check to find the reduction tv as it returns used values
      // instead of exprs.

      // The call is relatively heavy weight, consider caching
      auto used_vals = DependencyCheck::getAllValsBetween(
          {fusion_->inputs().begin(), fusion_->inputs().end()},
          fusion_->outputs());

      // Find the reduction tensor view, make sure there's only one
      for (auto val : used_vals) {
        if (val->getValType().value() == ValType::TensorView) {
          auto tv = val->as<TensorView>();
          if (tv->hasReduction()) {
            TORCH_INTERNAL_ASSERT(
                reduction_tv == nullptr,
                "Already found a reduction tensorview, cannot handle fusion of multiple reductions.");
            reduction_tv = tv;
          }
        }
      }

      TORCH_INTERNAL_ASSERT(
          reduction_tv != nullptr,
          "Could not find the reduction tensor view in the fusion.");

      // Generate the reduction parameters
      auto reduction_params =
          getReductionHeuristics(fusion_.get(), inputs, reduction_tv);

      TORCH_INTERNAL_ASSERT(
          reduction_params.has_value(),
          "Error getting reduction heuristics for scheduling.");

      launch_params = reduction_params.value().lparams;

      auto fusion_executor =
          &red_fusion_executor_cache_[device_index][reduction_params.value()];

      if (!fusion_executor->compiled()) {
        // HEURISTIC NOT COMPILED, COMPILE A KERNEL
        Fusion fusion = *fusion_;

        FusionGuard fg(&fusion);

        // Heavy weight call
        auto used_vals = DependencyCheck::getAllValsBetween(
            {fusion.inputs().begin(), fusion.inputs().end()}, fusion.outputs());

        TensorView* reduction_tv = nullptr;

        for (auto val : used_vals) {
          if (val->getValType().value() == ValType::TensorView) {
            auto tv = val->as<TensorView>();
            if (tv->hasReduction()) {
              TORCH_INTERNAL_ASSERT(
                  reduction_tv == nullptr,
                  "Already found a reduction tensorview, cannot handle fusion of multiple reductions.");
              reduction_tv = tv;
            }
          }
        }

        TORCH_INTERNAL_ASSERT(
            reduction_tv != nullptr,
            "Could not find the reduction tensor view in the fusion.");

        // Heavy weight call
        auto outputsOfReduction =
            DependencyCheck::getAllOutputsOf({reduction_tv});

        auto tv_entries =
            ir_utils::filterByType<TensorView>(outputsOfReduction);

        std::vector<TensorView*> tvOutputsOfReduction(
            tv_entries.begin(), tv_entries.end());

        scheduleReduction(
            &fusion,
            reduction_params.value(),
            reduction_tv,
            tvOutputsOfReduction);

        // This means we have not found a previously generated kernel that's
        // compatible with the new reduction params. We need to finish codegen.
        CompileOptions options;
        options.device = c10::Device(DeviceType::CUDA, device_index);
        fusion_executor->compileFusion(&fusion, options);
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
        // no need to copy fusion_, as we are not generating more than 1 kernel
        // for PW.
        scheduleFusion(fusion_.get(), inputs);
        pw_fusion_executor_cache_[device_index]->compileFusion(
            fusion_.get(), options);
      }
      // record new short cut to `FusionExecutor`
      code_to_fe_lookup_[unique_id] =
          pw_fusion_executor_cache_[device_index].get();
    }
  }

  return code_to_fe_lookup_[unique_id]->runFusion(
      inputs, launch_params, unique_id);
}

bool GraphCache::requiresPermutation() {
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
  reduction_axes_ = graphReductionAxes(graph);

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
