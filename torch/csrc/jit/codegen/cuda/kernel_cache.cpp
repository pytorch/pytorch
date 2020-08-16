#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/parser.h>
#include <torch/csrc/jit/codegen/cuda/scheduler.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

// TODO: This class is dead at the moment, but we need to figure out a generic
// cacheing system that will suite our needs.

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

// TODO: temporary hack to resolve my is_constructible issue;
std::vector<size_t> toVector(const at::DimVector& small_vec) {
  return std::vector<size_t>(small_vec.begin(), small_vec.end());
}

void debugPrint(const TensorTypePtr& type) {
  printf("\nsizes:");
  if (auto sizes = type->symbolic_sizes().sizes()) {
    // for (const auto& shape_symbol : sizes.value()) {
    int rank = static_cast<int>(sizes->size());
    for (int i = 0; i < rank; i++) {
      const auto& shape_symbol = sizes.value()[i];
      if (shape_symbol.is_static()) {
        printf("%ld, ", shape_symbol.static_size());
      } else {
        printf("s(%ld), ", *reinterpret_cast<const int64_t*>(&shape_symbol));
      }
    }
  } else {
    printf("no size available\n");
  }
  if (const auto& stride_properties = type->stride_properties().sizes()) {
    int rank = static_cast<int>(stride_properties->size());
    printf("\nstride: ");
    for (int i = 0; i < rank; i++) {
      if ((*stride_properties)[i].has_value() &&
          (*stride_properties)[i]->stride_.has_value()) {
        printf("%ld, ", (*stride_properties)[i]->stride_.value());
      } else {
        printf("?, ");
      }
    }
    printf("\nstride index: ");
    for (int i = 0; i < rank; i++) {
      if ((*stride_properties)[i].has_value() &&
          (*stride_properties)[i]->stride_index_.has_value()) {
        printf("%ld, ", (*stride_properties)[i]->stride_index_.value());
      } else {
        printf("?, ");
      }
    }
    printf("\ncontiguous: ");
    for (int i = 0; i < rank; i++) {
      if ((*stride_properties)[i].has_value() &&
          (*stride_properties)[i]->contiguous_.has_value()) {
        printf("%d, ", (*stride_properties)[i]->contiguous_.value());
      } else {
        printf("?, ");
      }
    }
  } else {
    printf("no stride properties available\n");
  }
}

at::DimVector graphReductionAxes(const std::shared_ptr<Graph>& graph) {
  at::DimVector reduction_axes;
  for (const auto& n : graph->nodes()) {
    if (isReductionNode(n)) {
      // TODO: I think this is enough to detect reduction that's not the output
      // as well. Since we go in topological order, we would run into
      // intermediate reduction, if there's any.
      TORCH_INTERNAL_ASSERT(
          graph->outputs().size() == 1 && graph->outputs()[0] == n->output(),
          "support for graph with reduction is limited to single output from reduction node");

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

at::DimVector getPermutationPerSortedStride(const TensorTypePtr& type) {
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
        if (red_dim < (const unsigned long)dim) {
          adjusted_offset++; // 1.b
        } else if (red_dim == (const unsigned long)dim) {
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

FusionExecutorCache::FusionExecutorCache(
    std::unique_ptr<Fusion>&& fusion,
    at::Device device)
    : device_(device), fusion_(std::move(fusion)) {}

// TODO: dummy cache
std::vector<at::Tensor> FusionExecutorCache::runFusionWithInputs(
    const at::ArrayRef<IValue>& inputs) {
  // caching strategy is different for pw-fusion and reduction-fusion.
  if (fusion_->hasReduction()) {
    // copy the fusion, since each FusionExecutor needs to manipulate the fusion
    // in order to generate kernel.
    Fusion fusion = *fusion_;
    FusionGuard fg(&fusion);
    TensorView* red_tv = nullptr;
    for (auto expr : fusion.exprs()) {
      if (expr->getExprType().has_value() &&
          expr->getExprType().value() == ExprType::ReductionOp) {
        red_tv = expr->outputs()[0]->as<TensorView>();
        break;
      }
    }
    auto reduction_params = scheduleReduction(&fusion, inputs, red_tv);
    TORCH_INTERNAL_ASSERT(
        reduction_params.has_value(),
        "reduction schedule failed in `scheduleReduction`");
    auto& fusion_executor =
        red_fusion_executor_cache_[reduction_params.value()];
    if (!fusion_executor.compiled()) {
      // This means we have not found a previously generated kernel that's
      // compatible with the new reduction params. We need to finish codegen.
      CompileOptions options;
      options.device = device_;
      fusion_executor.compileFusion(&fusion, options);
    }
    return fusion_executor.runFusion(inputs);
  } else {
    if (!pw_fusion_executor_cache_) {
      pw_fusion_executor_cache_ = std::make_unique<FusionExecutor>();
      CompileOptions options;
      options.device = device_;
      // no need to copy fusion_, as we are not generating more than 1 kernel
      // for PW.
      scheduleFusion(fusion_.get(), inputs);
      pw_fusion_executor_cache_->compileFusion(fusion_.get(), options);
    }
    return pw_fusion_executor_cache_->runFusion(inputs);
  }
}

GraphCache::InputsRequirement::InputsRequirement(
    const std::shared_ptr<Graph>& graph,
    const std::vector<size_t>& reduction_axes) {
  // run over inputs to extract common types;
  TensorTypePtr acc_type = TensorType::get();
  for (const auto& input : graph->inputs()) {
    // only check tensor types;
    if (auto input_type = input->type()->cast<TensorType>()) {
      vec_optional_ttp.emplace_back(input_type);
      if (acc_type->dim().has_value()) {
        // TODO: I think merge cannot handle broadcast - Go verify it later;
        // TODO: Since we are only handling permutation here, we should just
        //       merge the stride_index_;
        acc_type = acc_type->merge(input_type);
      } else {
        acc_type = input_type;
      }
    } else {
      vec_optional_ttp.emplace_back(c10::nullopt);
    }
  }
  input_permutation_ = getPermutationPerSortedStride(acc_type);
  output_permutation_ = inversePermutation(input_permutation_, reduction_axes);
  TORCH_CHECK(
      acc_type->device().has_value(), "requires fixed device for all inputs");
  device_ = acc_type->device();
}

GraphCache::InputsRequirement::InputsRequirement(
    const at::ArrayRef<IValue>& inputs,
    const std::vector<size_t>& reduction_axes) {
  // run over inputs to extract common types;
  TensorTypePtr acc_type = TensorType::get();
  for (const auto& input : inputs) {
    // only check tensor types;
    if (input.isTensor()) {
      // TensorType::create populates stride properties;
      // auto input_type = TensorType::create(input.toTensor());
      // vec_optional_ttp.emplace_back(input_type);
      vec_optional_ttp.emplace_back(TensorType::create(input.toTensor()));
      if (acc_type->dim().has_value()) {
        // TODO: I think merge cannot handle broadcast - Go verify it later;
        // TODO: Since we are only handling permutation here, we should just
        //       merge the stride_index_;
        acc_type = acc_type->merge(vec_optional_ttp.back().value());
      } else {
        acc_type = vec_optional_ttp.back().value();
      }
    } else {
      vec_optional_ttp.emplace_back(c10::nullopt);
    }
  }
  input_permutation_ = getPermutationPerSortedStride(acc_type);
  output_permutation_ = inversePermutation(input_permutation_, reduction_axes);
  TORCH_CHECK(
      acc_type->device().has_value(), "requires fixed device for all inputs");
  device_ = acc_type->device();
}

bool GraphCache::InputsRequirement::requiresPermutation() {
  const size_t input_rank = input_permutation_.size();
  for (size_t i = 0; i < input_rank; i++) {
    if (input_permutation_[i] != (long)i) {
      return true;
    }
  }
  // Check if output agrees
  const size_t output_rank = output_permutation_.size();
  for (size_t i = 0; i < output_rank; i++) {
    TORCH_INTERNAL_ASSERT(
        output_permutation_[i] == (long)i,
        "permutation of output and input is not consistent");
  }
  return false;
}

// TODO: tests!
bool GraphCache::InputsRequirement::complyWith(
    const InputsRequirement& expect) {
  if (device_ != expect.device_ ||
      input_permutation_ != expect.input_permutation_ ||
      output_permutation_ != expect.output_permutation_ ||
      vec_optional_ttp.size() != expect.vec_optional_ttp.size()) {
    return false;
  }

  // trick here is, `this` is always well defined while `expect` could has
  // missing options;
  for (size_t i = 0; i < vec_optional_ttp.size(); i++) {
    // TensorType has to match, otherwise it's not compatible to our graph.
    auto expect_vec_optional_ttp_i = expect.vec_optional_ttp[i];
    TORCH_INTERNAL_ASSERT(
        vec_optional_ttp[i].has_value() ==
        expect_vec_optional_ttp_i.has_value());
    if (expect_vec_optional_ttp_i.has_value()) {
      // We assume that dimensionality should always match.
      TORCH_INTERNAL_ASSERT(
          (*expect_vec_optional_ttp_i)->symbolic_sizes().sizes().has_value() &&
              (*expect_vec_optional_ttp_i)
                  ->stride_properties()
                  .sizes()
                  .has_value() &&
              (*expect_vec_optional_ttp_i)->dim().has_value() &&
              (*vec_optional_ttp[i])->dim().value() &&
              (*expect_vec_optional_ttp_i)->dim().value() ==
                  (*vec_optional_ttp[i])->dim().value(),
          "expect fixed rank of tensors");

      int rank = static_cast<int>((*expect_vec_optional_ttp_i)->dim().value());
      auto vec_shape_symbol_ex =
          (*expect_vec_optional_ttp_i)->symbolic_sizes().sizes().value();
      auto vec_optional_stride_ex =
          (*expect_vec_optional_ttp_i)->stride_properties().sizes().value();
      auto vec_shape_symbol =
          (*vec_optional_ttp[i])->symbolic_sizes().sizes().value();
      auto vec_optional_stride =
          (*vec_optional_ttp[i])->stride_properties().sizes().value();
      for (int j = 0; j < rank; j++) {
        // if broadcast rule differs, compliance is broken;
        if ((vec_shape_symbol_ex[j].is_static() &&
             vec_shape_symbol_ex[j].static_size() == 1) ^
            (vec_shape_symbol[j].is_static() &&
             vec_shape_symbol[j].static_size() == 1)) {
          return false;
        }

        const auto& vec_optional_stride_ex_j = vec_optional_stride_ex[j];
        const auto& vec_optional_stride_j = vec_optional_stride[j];
        // if contiguity / stride index differ, compliance is broken;
        if (vec_optional_stride_ex_j.has_value() !=
            vec_optional_stride_j.has_value()) {
          return false;
        }
        if (vec_optional_stride_ex_j.has_value() &&
            (vec_optional_stride_ex_j->stride_index_ !=
                 vec_optional_stride_j->stride_index_ ||
             vec_optional_stride_ex_j->contiguous_ !=
                 vec_optional_stride_j->contiguous_)) {
          return false;
        }
      }
    }
  }
  return true;
}

FusionExecutorCache* GraphCache::createFusionExecutorCache(
    const InputsRequirement& input_stack) {
  input_stacks_.emplace_back(input_stack);
  std::shared_ptr<Graph> parsing_graph = graph_->copy();
  // assign inputs on parsing_graph to accommodate legacy executor, where input
  // type might be missing/incomplete;
  // This is purely overhead for profiling executor;
  for (size_t i = 0; i < input_stack.vec_optional_ttp.size(); i++) {
    // skip scalar inputs;
    if (input_stack.vec_optional_ttp[i].has_value()) {
      parsing_graph->inputs()[i]->setType(
          input_stack.vec_optional_ttp[i].value());
    }
  }

  // permute inputs on `Graph` to sort dimensions on common stride order;
  if (input_stacks_.back().requiresPermutation()) {
    auto input_permutation = input_stacks_.back().input_permutation_;

    // TODO: lambda is a bad idea, the logic in this function is too tricky and
    //       should be properly tested to ensure correctness.
    // lambda to permute `TensorType` axes per `input_permutation`
    auto type_permute_fn = [&input_permutation](const TensorTypePtr& type) {
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
        permuted_vec_ss.emplace_back(vec_shape_symbol[input_permutation[i]]);
        // permutation doesn't change contiguity info, nor does it change
        // stride; The only thing affected is stride_index_;
        if (vec_optional_stride[i].has_value()) {
          c10::optional<size_t> index = vec_optional_stride[i]->stride_index_;
          if (index.has_value()) {
            for (int j = 0; j < rank; j++) {
              // follow the permutation to resolve the new stride_index;
              if (input_permutation[j] == (long)index.value()) {
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

    for (auto input : parsing_graph->inputs()) {
      if (auto input_type = input->type()->cast<TensorType>()) {
        input->setType(type_permute_fn(input_type));
      }
    }

    if (!reduction_axes_.empty()) {
      // see [ NOTE - reduction in graph ] part 2.
      for (auto n : parsing_graph->nodes()) {
        if (isReductionNode(n)) {
          // TODO: this is mostly redundant check, but it's compile time, we
          //       leave it here to be safe;
          TORCH_INTERNAL_ASSERT(
              parsing_graph->outputs().size() == 1 &&
                  parsing_graph->outputs()[0] == n->output(),
              "supporfor graph with reduction is limited to single output from reduction node");
          auto dims_list = constant_as<c10::List<int64_t>>(n->input(1));
          TORCH_INTERNAL_ASSERT(
              dims_list.has_value(), "reduction axes should be constant");
          std::vector<int64_t> adjusted_reduction_axes;
          for (const auto dim : dims_list->vec()) {
            // adjust reduction axis to be the permuted axis;
            for (size_t j = 0; j < input_permutation.size(); j++) {
              // follow the permutation to resolve the new reduction axes;
              if (input_permutation[j] == dim) {
                adjusted_reduction_axes.emplace_back(j);
                break;
              }
            }
          }
          parsing_graph->setInsertPoint(n);
          auto const_ival_axes =
              parsing_graph->insertConstant(IValue(adjusted_reduction_axes));
          n->replaceInput(1, const_ival_axes);
        }
      }
    }
  }

  TORCH_INTERNAL_ASSERT(
      input_stacks_.back().device_.has_value(),
      "device is not set for fusion executor, something went wrong in NvFuser");
  fe_cache_.emplace_back(std::make_unique<FusionExecutorCache>(
      parseJitIR(parsing_graph), input_stacks_.back().device_.value()));
  return fe_cache_.back().get();
}

GraphCache::GraphCache(std::shared_ptr<Graph> graph)
    : graph_(std::move(graph)) {
  // [ NOTE - reduction in graph ]
  //
  // reduction complicates our permutation in integration, it addes two things:
  // 1. we need to adjust output_permutation_;
  //    because of dimension elimination during permutation (not necessarily,
  //    given the `keepdim` argument.) this needs to be accommodated later when
  //    we added the support.
  // 2. adjust reduction axes for the permutation;
  //    permute changes the semantics of axes, we need to update the reduction
  //    axes in the graph in order to match the behavior;
  reduction_axes_ = graphReductionAxes(graph_);

  // compile a kernel if we have enough information from graph (profiling
  // record)
  if (IsNewExecutorEnabled()) {
    createFusionExecutorCache(
        InputsRequirement(graph_, toVector(reduction_axes_)));
  }
}

std::vector<at::Tensor> GraphCache::runGraphWithInputs(
    const at::ArrayRef<IValue>& inputs) {
  InputsRequirement input_stack(inputs, toVector(reduction_axes_));
  FusionExecutorCache* fusion_executor_cache = nullptr;

  // TODO: hash indexing;
  for (size_t i = 0; i < fe_cache_.size(); i++) {
    if (input_stack.complyWith(input_stacks_[i])) {
      fusion_executor_cache = fe_cache_[i].get();
      break;
    }
  }
  if (!fusion_executor_cache) {
    fusion_executor_cache = createFusionExecutorCache(input_stack);
  }

  // GraphCache need to permute inputs/outputs to accommodate dimension
  // coalescing
  if (input_stack.requiresPermutation()) {
    std::vector<IValue> permuted_inputs;
    permuted_inputs.reserve(inputs.size());
    for (const auto& input : inputs) {
      if (input.isTensor()) {
        permuted_inputs.emplace_back(
            input.toTensor().permute(input_stack.input_permutation_));
      } else {
        permuted_inputs.emplace_back(input);
      }
    }
    auto outputs = fusion_executor_cache->runFusionWithInputs(permuted_inputs);
    std::vector<at::Tensor> permuted_outputs;
    permuted_outputs.reserve(outputs.size());
    for (const auto& output : outputs) {
      permuted_outputs.emplace_back(
          output.permute(input_stack.output_permutation_));
    }
    return permuted_outputs;
  } else {
    return fusion_executor_cache->runFusionWithInputs(inputs);
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
