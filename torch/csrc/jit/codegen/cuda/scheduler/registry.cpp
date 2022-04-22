#include <c10/util/irange.h>
#include <torch/csrc/jit/codegen/cuda/disjoint_set.h>
#include <torch/csrc/jit/codegen/cuda/executor_utils.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/pointwise.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/registry.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>

#include <limits>

#include <ATen/cuda/CUDAContext.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {
// TODO: Deduplicate from compute_at.cpp
std::deque<std::deque<TensorView*>> tvChains(
    std::deque<std::deque<Val*>> val_chains) {
  std::deque<std::deque<TensorView*>> tv_chains(val_chains.size());
  for (const auto i : c10::irange(val_chains.size())) {
    auto tv_iterable = ir_utils::filterByType<TensorView>(val_chains[i]);
    tv_chains[i] =
        std::deque<TensorView*>(tv_iterable.begin(), tv_iterable.end());
  }
  return tv_chains;
}

class SchedulerTopologyChecker {
 public:
  // Checks if any broadcasts are resolved after a reduction that don't follow
  // the normalization pattern
  static bool hasNonNormalizePostReductionBCast(Fusion* fusion) {
    auto all_vals = fusion->usedMathVals();
    std::vector<TensorView*> reduction_tvs;
    for (auto tv : ir_utils::filterByType<TensorView>(all_vals)) {
      if (tv->hasReduction() &&
          !(fusion == tv->fusion() && tv->isFusionInput())) {
        reduction_tvs.push_back(tv);
      }
    }

    // All tensor views that are eventually consumed to produce a reduction,
    // includes reduction tensor views.
    std::unordered_set<TensorView*> pre_reduction_tvs;

    {
      auto pre_reduction_vals = DependencyCheck::getAllValsBetween(
          {fusion->inputs().begin(), fusion->inputs().end()},
          {reduction_tvs.begin(), reduction_tvs.end()});
      auto pre_reduction_tv_vector =
          ir_utils::filterByType<TensorView>(pre_reduction_vals);
      pre_reduction_tvs = std::unordered_set<TensorView*>(
          pre_reduction_tv_vector.begin(), pre_reduction_tv_vector.end());
    }

    // Track which tensor views we've validated so we don't do it again.
    std::unordered_set<TensorView*> validated_resolved_tvs;

    // Run forward (towards outputs) from reductions on any path that isn't
    // before another reduction. Look for resolved broadcasts. If a resolved
    // broadcast is found, start there and propagate backwards. Track the id's
    // that were resolved and make sure there's a mapping to a TensorView before
    // a reduction.
    for (auto red_tv : reduction_tvs) {
      auto forward_tv_chains =
          tvChains(DependencyCheck::getAllUseChains(red_tv));
      // Propagate forward from reduction through all uses of the reduction
      for (auto forward_tv_dep_chain : forward_tv_chains) {
        TensorView* forward_running_producer = nullptr;
        TensorView* forward_running_consumer = forward_tv_dep_chain.front();
        forward_tv_dep_chain.pop_front();
        while (!forward_tv_dep_chain.empty()) {
          forward_running_producer = forward_running_consumer;
          forward_running_consumer = forward_tv_dep_chain.front();
          forward_tv_dep_chain.pop_front();

          if (std::none_of(
                  forward_running_producer->getMaybeRFactorDomain().begin(),
                  forward_running_producer->getMaybeRFactorDomain().end(),
                  [](IterDomain* id) { return id->isBroadcast(); })) {
            // If there's no broadcast axes in producer it doesn't need to be
            // checked
            continue;
          }

          // If consumer is before another reduction it doesn't need to be
          // checked
          if (pre_reduction_tvs.count(forward_running_consumer)) {
            break;
          }

          // If consumer was already validated it doesn't need to be checked
          if (validated_resolved_tvs.count(forward_running_consumer)) {
            continue;
          }

          auto forward_pairwise_root_map = PairwiseRootDomainMap(
              forward_running_producer, forward_running_consumer);
          auto forward_p2c_root_map =
              forward_pairwise_root_map.mapProducerToConsumer(
                  forward_running_producer->domain(),
                  forward_running_consumer->domain());

          // These are the ids we will have to resolve. As we resolve them we'll
          // remove them from this vector. If this vector ends up empty, then
          // we've resolved everything we need to. This is a pair so as we
          // traverse we can map the id through the traversal. The first entry
          // in the pair will be the original id so we can reset it if it's not
          // resolved before the next traversal. The second ID will be
          // propagated as we map the IDs through the backward traversal.
          std::vector<std::pair<IterDomain*, IterDomain*>> ids_to_resolve;

          // Check if any TensorViews have a resolved broadcast
          for (auto entry : forward_p2c_root_map) {
            auto p_id = entry.first;
            auto c_id = entry.second;
            if (p_id->isBroadcast() &&
                (!c_id->isBroadcast() && !c_id->isTrivialReduction())) {
              ids_to_resolve.emplace_back(std::make_pair(c_id, c_id));
            }
          }

          if (ids_to_resolve.empty()) {
            continue;
          }

          // Only because of api limitations in getAllDependencyChains
          auto inputs_of_forward_running_consumer =
              IterVisitor::getInputsTo({forward_running_consumer});
          auto tv_inputs_of_forward_running_consumer =
              ir_utils::filterByType<TensorView>(
                  inputs_of_forward_running_consumer);

          for (auto input_of_forward_running_consumer :
               tv_inputs_of_forward_running_consumer) {
            if (pre_reduction_tvs.find(input_of_forward_running_consumer) ==
                pre_reduction_tvs.end()) {
              // If this input isn't an input to a reduction, no point
              // traversing the dependency chains as we know we can't validate
              // this broadcast through chains to this input
              continue;
            }

            auto backward_tv_chains =
                tvChains(DependencyCheck::getAllDependencyChains(
                    input_of_forward_running_consumer,
                    forward_running_consumer));

            for (auto backward_tv_chain : backward_tv_chains) {
              if (ids_to_resolve.empty()) {
                break;
              }

              for (auto& pair : ids_to_resolve) {
                pair.second = pair.first;
              }

              TensorView* backward_running_producer = backward_tv_chain.back();
              TensorView* backward_running_consumer = nullptr;
              backward_tv_chain.pop_back();

              TORCH_INTERNAL_ASSERT(
                  backward_running_producer == forward_running_consumer);

              while (!backward_tv_chain.empty()) {
                backward_running_consumer = backward_running_producer;
                backward_running_producer = backward_tv_chain.back();
                backward_tv_chain.pop_back();

                std::vector<IterDomain*> running_resolved_ids;

                auto backward_pairwise_root_map = PairwiseRootDomainMap(
                    backward_running_producer, backward_running_consumer);

                auto backward_c2p_root_map =
                    backward_pairwise_root_map.mapConsumerToProducer(
                        backward_running_consumer->domain(),
                        backward_running_producer->domain());

                // Mark if producer is a producer of a reduction
                bool producer_resolves =
                    pre_reduction_tvs.count(backward_running_producer);

                bool at_leat_one_id_mapped = false;
                for (size_t entry_i = ids_to_resolve.size(); entry_i > 0;
                     entry_i--) {
                  auto orig_id = ids_to_resolve[entry_i - 1].first;
                  auto running_id = ids_to_resolve[entry_i - 1].second;
                  if (backward_c2p_root_map.find(running_id) !=
                      backward_c2p_root_map.end()) {
                    at_leat_one_id_mapped = true;
                    if (producer_resolves &&
                        !backward_c2p_root_map.at(running_id)->isBroadcast()) {
                      // If mapped, and producer is a producer of a reduction,
                      // we can resolve this id
                      ids_to_resolve.erase(
                          ids_to_resolve.begin() + (entry_i - 1));
                    } else {
                      ids_to_resolve[entry_i - 1] = std::make_pair(
                          orig_id, backward_c2p_root_map.at(running_id));
                    }
                  }
                }
                if (!at_leat_one_id_mapped) {
                  // If no id's map any more, go to the next chain
                  break;
                }

                if (ids_to_resolve.empty()) {
                  break;
                }
              }
            }
          } // for(auto input_of_forward_running_consumer :
            // tv_inputs_of_forward_running_consumer){

          // if all ids were not resolved, then we've found an instance of a
          // bad broadcast resolution after reduction
          if (ids_to_resolve.size()) {
            return true;
          }

        } // while (!forward_tv_dep_chain.empty()) {
      } // for (auto forward_tv_dep_chain : forward_tv_chains) {
    } // for (auto red_tv : reduction_tvs)
    return false;
  }

  // Checks if any broadcasts are resolved after a reduction, this shouldn't be
  // accepted in the single reduction or multi-reduction scheduler
  static bool hasPostReductionBCast(Fusion* fusion) {
    auto all_vals = fusion->usedMathVals();
    for (auto tv : ir_utils::filterByType<TensorView>(all_vals)) {
      // Welford can have 2 outputs, so do this on all found reduction tensor
      // views
      if (tv->hasReduction() && !tv->isFusionInput()) {
        auto tv_chains = tvChains(DependencyCheck::getAllUseChains(tv));
        // Propagate forward from reduction through all uses of the reduction
        for (auto tv_dep_chain : tv_chains) {
          TensorView* running_producer = nullptr;
          TensorView* running_consumer = tv_dep_chain.front();
          tv_dep_chain.pop_front();
          while (!tv_dep_chain.empty()) {
            running_producer = running_consumer;
            running_consumer = tv_dep_chain.front();
            tv_dep_chain.pop_front();

            auto pairwise_root_map =
                PairwiseRootDomainMap(running_producer, running_consumer);
            auto p2c_root_map = pairwise_root_map.mapProducerToConsumer(
                running_producer->domain(), running_consumer->domain());

            // Check if any TensorViews have a resolved broadcast
            for (auto entry : p2c_root_map) {
              auto p_id = entry.first;
              auto c_id = entry.second;
              if (p_id->isBroadcast() &&
                  (!c_id->isBroadcast() && !c_id->isTrivialReduction())) {
                return true;
              }
            }
          }
        }
      }
    }
    return false;
  }

  // Checks if there's any unsupported operations post reduction. If outer
  // reduction we can fuse some pointwise ops if they don't require
  // broadcasting (checked in hasPostReductionBCast). For inner reductions we
  // cannot fuse any binary like operation (includes operations like shift that
  // we're not fusing right now) involving "new" inputs (not going through a
  // reduction).
  static bool supportedPostReductionFusion(
      Fusion* fusion,
      std::vector<TensorView*> reduction_tvs) {
    TORCH_INTERNAL_ASSERT(reduction_tvs.size());
    bool fastest_dim_reduction = true;
    auto red_root_dom = reduction_tvs[0]->getRootDomain();
    for (size_t i = red_root_dom.size(); i > 0; i--) {
      if (red_root_dom[i - 1]->isBroadcast() ||
          red_root_dom[i - 1]->isTrivialReduction()) {
        continue;
      } else if (red_root_dom[i - 1]->isReduction()) {
        fastest_dim_reduction = true;
        break;
      } else {
        fastest_dim_reduction = false;
        break;
      }
    }

    // When checking post reduction vals, we need to make sure
    //  we are really checking paths starting from all outputs
    //  of multi-output reductions, i.e. welford. The reduction_tv
    //  vector is assumed to only have one of them.
    std::unordered_set<Val*> reduction_tv_set(
        reduction_tvs.begin(), reduction_tvs.end());

    for (auto red : reduction_tvs) {
      if (red->definition()) {
        if (auto wop = dynamic_cast<WelfordOp*>(red->definition())) {
          for (auto wop_output : wop->outputs()) {
            if (wop_output->isA<TensorView>()) {
              reduction_tv_set.insert(wop_output);
            }
          }
        }
      }
    }

    // If reductions are on fastest dim, don't fuse any operations (after
    // reductions) that requires an input that is not an input to the
    // reductions.
    if (fastest_dim_reduction) {
      auto post_reduction_vals = DependencyCheck::getAllValsBetween(
          reduction_tv_set,
          {fusion->outputs().begin(), fusion->outputs().end()});

      if (post_reduction_vals.empty()) {
        return true;
      }

      auto reduction_inputs = IterVisitor::getInputsTo(
          {reduction_tvs.begin(), reduction_tvs.end()});

      for (auto tv : ir_utils::filterByType<TensorView>(
               post_reduction_vals.begin(), post_reduction_vals.end())) {
        if (tv->definition() == nullptr) {
          continue;
        }

        auto tv_inputs = IterVisitor::getInputsTo({tv});

        if (std::any_of(
                tv_inputs.begin(),
                tv_inputs.end(),
                [&reduction_inputs](Val* inp) {
                  return inp->isA<TensorView>() &&
                      std::find(
                          reduction_inputs.begin(),
                          reduction_inputs.end(),
                          inp) == reduction_inputs.end();
                })) {
          return false;
        }
      }
    }

    return true;
  }
};

bool isConnectedFusionGraph(Fusion* fusion) {
  if (fusion->outputs().empty()) {
    // Trivial case interpreted as connected
    return true;
  }

  // A set of connected components on the fusion graph
  DisjointSet<Val*> component_sets;

  // Iterate through all used exprs
  for (auto expr : fusion->exprs()) {
    TORCH_INTERNAL_ASSERT(
        !expr->inputs().empty(), "unknown expr with zero input");

    // Each expr joins all its inputs and
    //  outputs to the same component
    auto input0 = expr->inputs()[0];
    for (auto input : expr->inputs()) {
      component_sets.join(input0, input);
    }
    for (auto output : expr->outputs()) {
      component_sets.join(input0, output);
    }
  }

  // Join aliased outputs
  for (auto alias_it : fusion->ioAlias()) {
    component_sets.join(alias_it.first, alias_it.second);
  }

  // Check connected-ness:
  //  If there is no independent compute flow
  // on this fusion graph, all outputs will be
  // equivalent/connected to the first output.
  auto output0 = fusion->outputs()[0];
  for (auto output : fusion->outputs()) {
    if (!component_sets.areEquivalent(output0, output)) {
      return false;
    }
  }
  return true;
}

} // namespace

SchedulerRuntimeInfo::SchedulerRuntimeInfo(
    Fusion* complete_fusion,
    const at::ArrayRef<IValue>& inputs,
    bool create_expr_evaluator)
    : complete_fusion_(complete_fusion) {
  collectVectorizationInfo(inputs);
  expression_evaluator_ =
      std::make_unique<ExpressionEvaluator>(complete_fusion_);
  if (create_expr_evaluator) {
    initializeExpressionEvaluator(inputs);
  }
  collectIndexModeInfo(inputs);
}

SchedulerRuntimeInfo::SchedulerRuntimeInfo(
    const SchedulerRuntimeInfo& copy_from)
    : complete_fusion_(copy_from.complete_fusion_),
      alignment_map_(copy_from.alignment_map_),
      common_alignment_size_(copy_from.common_alignment_size_) {
  expression_evaluator_ =
      std::make_unique<ExpressionEvaluator>(complete_fusion_);
}

size_t SchedulerRuntimeInfo::getAlignmentSize(TensorView* tv) {
  auto alignment_entry = alignment_map_.find(tv);
  if (alignment_entry == alignment_map_.end()) {
    return max_alignment_size_in_byte;
  } else {
    return alignment_entry->second;
  }
}

void SchedulerRuntimeInfo::initializeExpressionEvaluator(
    const at::ArrayRef<IValue>& inputs) {
  // TODO: refactor bindFusionInputs to better support this
  //  use case, i.e. support construct and bind input.
  *expression_evaluator_ =
      executor_utils::bindFusionInputs(inputs, complete_fusion_);
}

size_t SchedulerRuntimeInfo::collectAlignmentSize(
    const at::Tensor& tensor) const {
  const size_t address = reinterpret_cast<size_t>(tensor.data_ptr());
  size_t alignment_size = 1;
  size_t next_alignment_size = 2;

  while (alignment_size <= max_alignment_size_in_byte &&
         address % next_alignment_size == 0) {
    alignment_size = next_alignment_size;
    next_alignment_size *= 2;
  }

  return alignment_size;
}

void SchedulerRuntimeInfo::collectVectorizationInfo(
    const at::ArrayRef<IValue>& inputs) {
  common_alignment_size_ = max_alignment_size_in_byte;
  size_t number_of_inputs = complete_fusion_->inputs().size();
  std::unordered_map<TensorView*, size_t> cg_tensor_to_at_tensor_index;

  for (auto input_index : c10::irange(number_of_inputs)) {
    if (auto input_tensor = dynamic_cast<TensorView*>(
            complete_fusion_->inputs()[input_index])) {
      if (input_tensor->nDims() == 0) {
        // A 0-dim tensor input would not need vectorization
        continue;
      }
      if (input_tensor->domain()
              ->domain()[input_tensor->nDims() - 1]
              ->isBroadcast()) {
        // skip the tensors with innermost iterdomain broadcasted,
        //  as we will not vectorize these.
        continue;
      }

      // Collect strides of the input tensor
      TORCH_INTERNAL_ASSERT(inputs[input_index].isTensor());
      const auto& at_tensor = inputs[input_index].toTensor();

      cg_tensor_to_at_tensor_index.emplace(
          std::make_pair(input_tensor, input_index));

      // Collect alignment of the input tensor
      auto alignment_size = collectAlignmentSize(at_tensor);
      common_alignment_size_ = std::min(alignment_size, common_alignment_size_);
      alignment_map_[input_tensor] = alignment_size;
    }
  }

  // Compute max vector word size for each input,
  //  tensors with inner most broadcast already
  //  filtered out.  common_alignment_size_ is
  //  computed up to this point.
  for (auto it : cg_tensor_to_at_tensor_index) {
    vectorword_map_[it.first] = collectMaxVectorizeSize(
        inputs[it.second].toTensor(), common_alignment_size_);
  }
}

size_t SchedulerRuntimeInfo::collectMaxVectorizeSize(
    const at::Tensor& tensor,
    size_t max_vector_size_in_byte) {
  size_t vector_size = 1;
  size_t next_vector_size = 2;
  bool next_size_compatible = true;

  while (next_size_compatible &&
         next_vector_size * tensor.itemsize() <= max_vector_size_in_byte) {
    // If inner most dimension size is not divisible by new word size
    //  then we cannot vectorize with this width. But we do not
    //  care if all dimensions of this tensor is 1, i.e.
    //  input is actually a un-squeezed 0-dim tensor.
    for (size_t i = tensor.ndimension(); i > 0; i--) {
      if (tensor.size(i - 1) != 1) {
        if (tensor.size(tensor.ndimension() - 1) % next_vector_size != 0 ||
            tensor.stride(tensor.ndimension() - 1) != 1) {
          next_size_compatible = false;
        }
        break;
      }
    }

    if (!next_size_compatible) {
      break;
    }

    // If any stride is not divisible by the next word size,
    //  we cannot vectorize with this width.
    for (auto stride : tensor.strides()) {
      if (stride != 1 && stride % next_vector_size != 0) {
        next_size_compatible = false;
        break;
      }
    }

    if (next_size_compatible) {
      vector_size = next_vector_size;
      next_vector_size *= 2;
    }
  }

  return vector_size;
}

size_t SchedulerRuntimeInfo::getVectorizableWidth(TensorView* tv) {
  auto recorded_size_it = vectorword_map_.find(tv);
  if (recorded_size_it != vectorword_map_.end()) {
    return recorded_size_it->second;
  }

  // If we don't have an record, either it is a tv with innermost
  //  broadcast, or it is an intermediate tensor allocated by fuser
  auto tv_root = TensorDomain::noReductions(tv->getMaybeRFactorDomain());
  auto tv_root_size = tv_root.size();

  // Filter out 0-dim tensors
  if (tv_root_size < 1) {
    return 1;
  }

  // Filter out mismatched contiguity info
  if (tv_root_size != tv->domain()->contiguity().size()) {
    return 1;
  }

  // Filter out innermost broadcast tensors
  auto inner_dimension = tv_root[tv_root_size - 1];
  if (inner_dimension->isBroadcast()) {
    return 1;
  }

  // Handle intermediate or output tensors that
  //  will be allocated by fuser
  auto maybe_data_type = tv->getDataType();

  // Do not vectorize on data with unknown type
  if (!maybe_data_type.has_value()) {
    return 1;
  }

  size_t item_size = dataTypeSize(maybe_data_type.value());
  // Assume we don't have non-divisible types for now.
  TORCH_INTERNAL_ASSERT(max_alignment_size_in_byte % item_size == 0);
  size_t max_vector_size = max_alignment_size_in_byte / item_size;

  // Assuming intermediate tensors have friendly alignment, and
  //  all contiguity true. Determine the largest power of 2 below
  //  innermost dimension size for the word size of vectorizaiton
  size_t vector_size = 1;
  size_t next_vector_size = 2;
  auto maybe_inner_dimension_size =
      expression_evaluator_->evaluate(inner_dimension->extent());
  TORCH_INTERNAL_ASSERT(maybe_inner_dimension_size.has_value());
  size_t inner_dimension_size = maybe_inner_dimension_size.value();

  while (next_vector_size <= max_vector_size &&
         next_vector_size <= inner_dimension_size &&
         inner_dimension_size % next_vector_size == 0) {
    vector_size = next_vector_size;
    next_vector_size *= 2;
  }

  // save output to avoid re-compute
  vectorword_map_[tv] = vector_size;

  return vector_size;
}

void SchedulerRuntimeInfo::collectIndexModeInfo(
    const at::ArrayRef<at::IValue>& inputs) {
  // TODO: Need to check the output sizes as well.

  // Save 1 more bit besides the sign bit to be conservative
  constexpr int64_t most_positive_int32_index =
      std::numeric_limits<int>::max() / 2;
  constexpr int64_t most_negative_int32_index =
      std::numeric_limits<int>::min() / 2;

  // Start by setting index mode to int32
  index_mode_ = KernelIndexMode::INT32;

  // Check all runtime inputs, and if any one of
  //  the input's index exceeds max_int32 will
  //  fall back to int64 indexing
  for (auto ivalue_input : inputs) {
    if (ivalue_input.isTensor()) {
      auto tensor_input = ivalue_input.toTensor();
      int64_t tensor_most_positive_index = 0;
      int64_t tensor_most_negative_index = 0;
      for (auto dim_i = 0; dim_i < tensor_input.ndimension(); dim_i++) {
        // Ignore broadcast dimensions
        if (tensor_input.size(dim_i) > 1) {
          // accumulate based on the sign of stride
          if (tensor_input.stride(dim_i) > 0) {
            // Acuumulate positive stride
            tensor_most_positive_index +=
                (tensor_input.size(dim_i) - 1) * tensor_input.stride(dim_i);
          } else {
            // Acuumulate negative stride
            tensor_most_negative_index +=
                (tensor_input.size(dim_i) - 1) * tensor_input.stride(dim_i);
          }
        }
      }

      // Fall back to int64 if it can be either too positive
      //  or too negative.
      if (tensor_most_positive_index > most_positive_int32_index ||
          tensor_most_negative_index < most_negative_int32_index) {
        index_mode_ = KernelIndexMode::INT64;
        return;
      }
    }
  }
}

bool SchedulerEntry::sameAs(const SchedulerEntry* other) {
  if (heuristc_ != other->heuristc_) {
    return false;
  }
  if (index_mode_ != other->index_mode_) {
    return false;
  }
  // Heuristic equal should imply has_reduction_param_ equal,
  //  need to double check if it is the case before removing
  //  the below one.
  if (has_reduction_param_ != other->has_reduction_param_) {
    return false;
  }
  if (has_reduction_param_) {
    return rparams_ == other->rparams_;
  } else {
    return pparams_ == other->pparams_;
  }
  return true;
}

namespace {
std::vector<TransposeOp*> findTransposeOps(Fusion* fusion) {
  auto exprs = fusion->exprs();
  auto transpose_ops = ir_utils::filterByType<TransposeOp>(exprs);
  return std::vector<TransposeOp*>(transpose_ops.begin(), transpose_ops.end());
}

static bool checkPatternEquivalence(
    TensorView* out_tv0,
    TensorView* out_tv1,
    const ComputeAtRootDomainMap& root_map) {
  const auto& out_root0 = out_tv0->getRootDomain();
  const auto& out_root1 = out_tv1->getRootDomain();
  const auto domain0 = out_tv0->domain();
  const auto domain1 = out_tv1->domain();

  auto it0 = out_root0.begin();
  auto it1 = out_root1.begin();

  auto skip_broadcast = [&]() {
    while (it0 != out_root0.end() && (*it0)->isBroadcast()) {
      it0++;
    }
    while (it1 != out_root1.end() && (*it1)->isBroadcast()) {
      it1++;
    }
  };

  skip_broadcast();
  while (it0 != out_root0.end() && it1 != out_root1.end()) {
    if ((*it0)->isReduction() != (*it1)->isReduction()) {
      return false;
    }
    if (!root_map.canMap(domain0, (*it0), domain1, (*it1))) {
      return false;
    }
    it0++;
    it1++;
    skip_broadcast();
  }

  return it0 == out_root0.end() && it1 == out_root1.end();
}

//! Scheduler interface:
//!    Each of the scheduler needs to provide 3 interface functions:
//!
//!      1. canScheduleCompileTime(Fusion* fusion) :
//!
//!        This function contains compiled-time checks on the graph itself
//!        without runtime input information. Only `fusion` is given in the
//!        argument to make sure only compile-time available info is needed in
//!        the check.
//!
//!        This function is to be called exactly once on each segmented group
//!        created in a segmented fusion so this part will not contribute to
//!        dynamic shape latency.
//!
//!     2. canScheduleRunTime(
//!            Fusion* fusion,
//!            SchedulerRuntimeInfo& runtime_info,
//!           HeuristicSummary* data_cache = nullptr):
//!        This function contains all canSchedule checks that will have to
//!        involve runtime input information, and will be run both by the
//!        segmenter and the kernel cache. The latency of this function will
//!        contribute to dynamic shape latency so `data_cache` should be used as
//!        much as possible to save re-computation.
//!
//!     3. schedule(fusion):
//!
//!        This function will be called when compiling a kernel. It should apply
//!        scheduling to the given fusion

class ReductionScheduler : public SchedulerEntry {
 public:
  explicit ReductionScheduler(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr)
      : SchedulerEntry(ScheduleHeuristic::Reduction, true) {
    computeHeuristics(fusion, runtime_info, data_cache);
  }

  //! Check if the reduction heuristics apply in given fusion
  static bool canScheduleCompileTime(Fusion* fusion) {
    // Temporarily allow view in reduction scheduler
    // TODO Add more testing before enabling
    auto view_tvs = scheduler_utils::getViewTVs(fusion);
    if (view_tvs.size() > 0) {
      return false;
    }

    // Needs at least one non-trivial reduction to consider.
    if (ir_utils::getReductionOps(fusion, true /* ignore_trivial */).empty()) {
      return false;
    }

    auto reduction_tvs =
        scheduler_utils::getReductionTvs(fusion, false /* ignore_trivial */);

    if (reduction_tvs.size() == 0) {
      // Use pointwise logic
      return false;
    }

    if (findTransposeOps(fusion).size() > 0) {
      // Use pointwise logic
      return false;
    }

    // Make sure reduction axes are consistent through the fusion
    auto reduction_ops =
        ir_utils::getReductionOps(fusion, false /* ignore_trivial */);
    if (reduction_ops.size() > 1) {
      // Before examining the reduction axes want to quickly
      //   check the reductions have the same axis width
      //   to avoid building root domain map in easier cases
      bool valid_axis_count = false;
      size_t axis_count = 0;
      auto reduction_root_size = [](TensorView* red_tv) {
        size_t count = 0;
        for (auto id : red_tv->getRootDomain()) {
          if (!id->isBroadcast()) {
            count++;
          }
        }
        return count;
      };

      for (auto red : reduction_tvs) {
        if (!valid_axis_count) {
          valid_axis_count = true;
          axis_count = reduction_root_size(red);
        } else {
          if (reduction_root_size(red) != axis_count) {
            return false;
          }
        }
      }

      // Use root domain map to check the reduction ops have the same axes
      FusionGuard fg(fusion);
      ComputeAtRootDomainMap root_map;
      root_map.build(true);

      // red_ops.size()>1 checked before
      for (size_t it = 1; it < reduction_tvs.size(); it++) {
        if (!checkPatternEquivalence(
                reduction_tvs[it - 1], reduction_tvs[it], root_map)) {
          return false;
        }
      }
    }

    // Doesn't allow persistent kernels in this scheduler
    auto persistent_buffer_info = scheduler_utils::persistentBuffers(fusion);
    if (persistent_buffer_info.persistent_buffers.size() > 0) {
      return false;
    }

    if (!SchedulerTopologyChecker::supportedPostReductionFusion(
            fusion, reduction_tvs) ||
        SchedulerTopologyChecker::hasPostReductionBCast(fusion)) {
      return false;
    }

    return true;
  }

  static bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    return true;
  }

  void schedule(Fusion* fusion) override {
    FUSER_PERF_SCOPE("Schedule Single Reduction");
    scheduleReduction(fusion, rparams());
  }

 private:
  void computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    auto param = getReductionHeuristics(fusion, runtime_info, data_cache);
    TORCH_INTERNAL_ASSERT(param.has_value());
    rparams() = param.value();
  }
};

class PointWiseScheduler : public SchedulerEntry {
 public:
  explicit PointWiseScheduler(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr)
      : SchedulerEntry(ScheduleHeuristic::PointWise, false) {
    computeHeuristics(fusion, runtime_info, data_cache);
  }

  static bool canScheduleCompileTime(Fusion* fusion) {
    //   Currently using the same path as the scheduler
    // to eliminate mismatch between canSchedule and
    // schedule pointwise.
    if (!hasReferenceTensorView(fusion)) {
      return false;
    }

    auto reduction_ops =
        ir_utils::getReductionOps(fusion, true /* ignore_trivial */);
    auto welford_ops = ir_utils::filterByType<WelfordOp>(reduction_ops);
    return reduction_ops.empty() && welford_ops.empty();
  }

  static bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    return true;
  }

  void schedule(Fusion* fusion) override {
    FUSER_PERF_SCOPE("Schedule PointWise Fusion");
    schedulePointwise(fusion, pparams());
  }

  void computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    auto pparam = getPointwiseHeuristics(fusion, runtime_info, data_cache);
    TORCH_INTERNAL_ASSERT(pparam.has_value());
    pparams() = pparam.value();
  }
};

class PersistentKernelScheduler : public SchedulerEntry {
 public:
  explicit PersistentKernelScheduler(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr)
      : SchedulerEntry(ScheduleHeuristic::Persistent, true) {
    computeHeuristics(fusion, runtime_info, data_cache);
  }

  void schedule(Fusion* fusion) override {
    FUSER_PERF_SCOPE("Schedule Persistent Fusion");
    schedulePersistentKernel(fusion, rparams());
  }

  static bool canScheduleCompileTime(Fusion* fusion) {
    // Needs at least one non-trivial reduction to consider.
    if (ir_utils::getReductionOps(fusion, true /* ignore_trivial */).empty()) {
      return false;
    }

    auto reduction_ops =
        ir_utils::getReductionOps(fusion, false /* ignore_trivial */);
    auto welford_ops = ir_utils::filterByType<WelfordOp>(reduction_ops);
    // For persistent schedule we want welford translated to average and
    // standard deviation reductions.
    if (welford_ops.begin() != welford_ops.end()) {
      return false;
    }

    auto view_tvs = scheduler_utils::getViewTVs(fusion);
    if (view_tvs.size() > 0) {
      return false;
    }

    auto reduction_tvs =
        scheduler_utils::getReductionTvs(fusion, false /* ignore_trivial */);

    if (reduction_tvs.size() == 0) {
      // Use pointwise logic
      return false;
    }

    if (findTransposeOps(fusion).size() > 0) {
      // Use pointwise logic
      return false;
    }

    // Before examining the reduction axes want to quickly
    //   check the reductions have the same axis width
    //   to avoid building root domain map in easier cases
    bool valid_axis_count = false;
    size_t axis_count = 0;
    auto reduction_root_size = [](TensorView* red_tv) {
      size_t count = 0;
      for (auto id : red_tv->getRootDomain()) {
        if (!id->isBroadcast()) {
          count++;
        }
      }
      return count;
    };

    for (auto red : reduction_tvs) {
      if (!valid_axis_count) {
        valid_axis_count = true;
        axis_count = reduction_root_size(red);
      } else {
        if (reduction_root_size(red) != axis_count) {
          return false;
        }
      }
    }

    // Use root domain map to check the reduction ops have the same axes
    FusionGuard fg(fusion);
    ComputeAtRootDomainMap root_map;
    root_map.build(true);

    // red_ops.size()>1 checked before
    for (const auto it : c10::irange(1, reduction_tvs.size())) {
      if (!checkPatternEquivalence(
              reduction_tvs[it - 1], reduction_tvs[it], root_map)) {
        return false;
      }
    }

    // Only accept persistent kernels
    auto persistent_buffer_info = scheduler_utils::persistentBuffers(fusion);
    if (persistent_buffer_info.persistent_buffers.size() == 0) {
      return false;
    }

    if (SchedulerTopologyChecker::hasNonNormalizePostReductionBCast(fusion)) {
      return false;
    }

    return true;
  }

  static bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    FUSER_PERF_SCOPE("PersistentKernelScheduler::canSchedule");

    auto reduction_tv_entry =
        HeuristicSummaryEntry<HeuristicCompileTime::ReductionTVs>(
            data_cache, [&fusion]() {
              return std::make_unique<std::vector<TensorView*>>(
                  scheduler_utils::getReductionTvs(
                      fusion /*, ignore_trivial = true*/));
            });

    auto& reduction_tvs = reduction_tv_entry.get();

    auto persistent_buffer_info_entry =
        HeuristicSummaryEntry<HeuristicCompileTime::PersistentBufferInfo>(
            data_cache, [&fusion]() {
              return std::make_unique<scheduler_utils::PersistentBufferInfo>(
                  scheduler_utils::persistentBuffers(fusion));
            });

    auto& persistent_buffer_info = persistent_buffer_info_entry.get();

    auto persistent_buffer_size_info = scheduler_utils::persistentBufferSize(
        fusion, runtime_info, persistent_buffer_info, data_cache);

    auto persistent_buffer_size = std::min(
        persistent_buffer_size_info.persistent_buffer_size,
        persistent_buffer_size_info.projected_persistent_buffer_size);

    if (persistent_buffer_size > scheduler_utils::register_file_size) {
      return false;
    }

    // If there's a small iteration dimension but a large reduction dimension it
    // may not make sense to make a persistent kernel
    auto properties =
        scheduler_utils::getProperties(fusion, runtime_info, reduction_tvs[0]);

    const int64_t device_max_threads_per_multiprocessor =
        (int64_t)at::cuda::getCurrentDeviceProperties()
            ->maxThreadsPerMultiProcessor;

    const int64_t device_multiprocessor_count =
        (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

    const int64_t warp_size = at::cuda::warp_size();

    // Maximum number of iteration dimensions we can have and still be
    // persistent.
    const int64_t max_multi_reduction_factor = std::max(
        scheduler_utils::register_file_size / persistent_buffer_size,
        (int64_t)1);

    // If outer reduction, and we have few iteration numel but large reduction
    // numel, don't generate kernel because we don't support cross grid
    // persistence
    if (
        // Don't go persistent if we can't fit half a warp on an SM
        (!properties.fastest_dim_reduction &&
         max_multi_reduction_factor < warp_size / 2) ||
        ( // Don't go persistent if we can't use a small fraction of the
          // available SMs yet have a large reduction size
            properties.total_iteration_numel <
                (properties.fastest_dim_reduction
                     ? std::max(device_multiprocessor_count / 8, (int64_t)1)
                     // Make sure we at least use a quarter of the device * a
                     // half warp
                     : (warp_size / 8) * device_multiprocessor_count) &&
            // Reduction count is larger than max thread count * 4
            properties.total_reduction_numel >=
                device_max_threads_per_multiprocessor * 4)) {
      return false;
    }

    return true;
  }

 private:
  void computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    auto param = getPersistentHeuristics(fusion, runtime_info, data_cache);
    TORCH_INTERNAL_ASSERT(param.has_value());
    rparams() = param.value();
  }
};

// Schedule Table
const std::vector<ScheduleHeuristic>& all_heuristics() {
  static const std::vector<ScheduleHeuristic> hlist = {
      ScheduleHeuristic::Reduction,
      ScheduleHeuristic::PointWise,
      ScheduleHeuristic::Persistent};
  return hlist;
}

//! A Utility for checking both dynamic and static part of
//!  can schedule
template <typename SchedulerType>
bool checkCanSchedule(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache = nullptr) {
  // If a data cache is given, the compile time part doesn't need to be checked,
  // since for all current use cases
  //  it has to pass all the compile time checks to create a data cache for this
  //  fusion.
  if (!data_cache) {
    if (!isConnectedFusionGraph(fusion)) {
      return false;
    }
    if (!SchedulerType::canScheduleCompileTime(fusion)) {
      return false;
    }
  }

  return SchedulerType::canScheduleRunTime(fusion, runtime_info, data_cache);
}

} // namespace

// Simple dispatcher interface
bool SchedulerEntry::canSchedule(
    ScheduleHeuristic sh,
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  switch (sh) {
    case ScheduleHeuristic::PointWise:
      return checkCanSchedule<PointWiseScheduler>(
          fusion, runtime_info, data_cache);
    case ScheduleHeuristic::Reduction:
      return checkCanSchedule<ReductionScheduler>(
          fusion, runtime_info, data_cache);
    case ScheduleHeuristic::Persistent:
      return checkCanSchedule<PersistentKernelScheduler>(
          fusion, runtime_info, data_cache);
    default:
      TORCH_INTERNAL_ASSERT(false, "unreachable");
      return false;
  }
  return false;
}

std::unique_ptr<SchedulerEntry> SchedulerEntry::makeEntry(
    ScheduleHeuristic sh,
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  std::unique_ptr<SchedulerEntry> scheduler_entry = nullptr;
  switch (sh) {
    case ScheduleHeuristic::PointWise:
      scheduler_entry = std::make_unique<PointWiseScheduler>(
          fusion, runtime_info, data_cache);
      break;
    case ScheduleHeuristic::Reduction:
      scheduler_entry = std::make_unique<ReductionScheduler>(
          fusion, runtime_info, data_cache);
      break;
    case ScheduleHeuristic::Persistent:
      scheduler_entry = std::make_unique<PersistentKernelScheduler>(
          fusion, runtime_info, data_cache);
      break;
    default:
      TORCH_INTERNAL_ASSERT(false, "unreachable");
  }

  scheduler_entry->index_mode_ = runtime_info.getIndexMode();
  return scheduler_entry;
}

// Simply loop through the list as baseline strategy
c10::optional<ScheduleHeuristic> SchedulerEntry::proposeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info) {
  for (auto sh : all_heuristics()) {
    if (canSchedule(sh, fusion, runtime_info)) {
      return sh;
    }
  }
  return c10::nullopt;
}

size_t SchedulerEntryHash::operator()(const SchedulerEntry& se) const {
  if (se.hasReductionParam()) {
    return ReductionParamsHash()(se.reductionParams());
  } else {
    return PointwiseParamsHash()(se.pointwiseParams());
  }
}

std::string toString(ScheduleHeuristic sh) {
  switch (sh) {
    case ScheduleHeuristic::PointWise:
      return "pointwise";
    case ScheduleHeuristic::Reduction:
      return "reduction";
    case ScheduleHeuristic::Persistent:
      return "persistent";
    default:
      TORCH_INTERNAL_ASSERT(false, "undefined schedule");
  }
  return "";
}

namespace {

//! CompileTimeInfo is the actual subclass of CompileTimeInfoBase that will
//!  be stored in the data cache. It owns a data_ state internally of the
//!  dataType defined within the entry class, which are listed in compile
//!  time info header.
template <typename EntryClass>
class CompileTimeInfo : public HeuristicCompileTime::CompileTimeInfoBase {
 public:
  CompileTimeInfo(std::unique_ptr<typename EntryClass::DataType> data)
      : CompileTimeInfoBase(EntryClass::EntryType), data_(std::move(data)) {}

  typename EntryClass::DataType* get() {
    return data_.get();
  }

 private:
  std::unique_ptr<typename EntryClass::DataType> data_;
};

} // namespace

HeuristicSummary::HeuristicSummary(
    Fusion* fusion,
    ScheduleHeuristic heuristic,
    SchedulerRuntimeInfo& runtime_info)
    : heuristic_(heuristic) {
  recording_ = true;
  switch (heuristic) {
    case ScheduleHeuristic::PointWise:
      getPointwiseHeuristics(fusion, runtime_info, this);
      PointWiseScheduler::canScheduleRunTime(fusion, runtime_info, this);
      break;
    case ScheduleHeuristic::Reduction:
      getReductionHeuristics(fusion, runtime_info, this);
      ReductionScheduler::canScheduleRunTime(fusion, runtime_info, this);
      break;
    case ScheduleHeuristic::Persistent:
      getPersistentHeuristics(fusion, runtime_info, this);
      PersistentKernelScheduler::canScheduleRunTime(fusion, runtime_info, this);
      break;
    default:
      TORCH_INTERNAL_ASSERT(false, "unknown heuristic");
  }
  validate();
  recording_ = false;
}

void HeuristicSummary::validate() const {
  switch (heuristic_) {
    case ScheduleHeuristic::PointWise: {
      TORCH_INTERNAL_ASSERT(
          entry_type_map_.count(EntryType::VECTORIZABLE_INPUTS_AND_OUTPUTS));
      TORCH_INTERNAL_ASSERT(
          entry_type_map_.count(EntryType::BROADCAST_BYTE_MULTIPLES));
      break;
    }
    case ScheduleHeuristic::Reduction: {
      TORCH_INTERNAL_ASSERT(entry_type_map_.count(EntryType::REDUCTION_TVS));
      TORCH_INTERNAL_ASSERT(
          entry_type_map_.count(EntryType::VECTORIZABLE_INPUTS_AND_OUTPUTS));
      TORCH_INTERNAL_ASSERT(
          entry_type_map_.count(EntryType::UNROLLABLE_INPUTS_AND_OUTPUTS));
      break;
    }
    case ScheduleHeuristic::Persistent: {
      TORCH_INTERNAL_ASSERT(entry_type_map_.count(EntryType::REDUCTION_TVS));
      TORCH_INTERNAL_ASSERT(
          entry_type_map_.count(EntryType::VECTORIZABLE_INPUTS_AND_OUTPUTS));
      TORCH_INTERNAL_ASSERT(
          entry_type_map_.count(EntryType::UNROLLABLE_INPUTS_AND_OUTPUTS));
      TORCH_INTERNAL_ASSERT(
          entry_type_map_.count(EntryType::PERSISTENT_BUFFER_INFO));
      // If check persistent factor only when persistent buffers needed.
      auto persistent_buffer_info =
          entry_type_map_.at(EntryType::PERSISTENT_BUFFER_INFO)
              ->as<
                  CompileTimeInfo<HeuristicCompileTime::PersistentBufferInfo>>()
              ->get();
      TORCH_INTERNAL_ASSERT(
          !persistent_buffer_info->persistent_buffers.empty() &&
          entry_type_map_.count(EntryType::SCOPE_PERSISTENT_FACTOR_INFO));
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(false, "unknown heuristic");
  }
}

void HeuristicSummary::insert(HeuristicSummary::EntryOwningPtr new_entry) {
  TORCH_INTERNAL_ASSERT(
      recording_, "should only insert entries at recording phase");
  // Just override when insertion duplicates, equality not checked.
  entry_type_map_[new_entry->type()] = new_entry.get();
  entries_.emplace_back(std::move(new_entry));
}

template <typename EntryClass>
HeuristicSummaryEntry<EntryClass>::HeuristicSummaryEntry(
    HeuristicSummary* data_cache,
    MakerFnType fn) {
  using InfoType = CompileTimeInfo<EntryClass>;

  if (!data_cache || data_cache->isRecording()) {
    owned_data_ = fn();
    data_ptr_ = owned_data_.get();

    if (data_cache) {
      std::unique_ptr<HeuristicCompileTime::CompileTimeInfoBase> new_entry =
          std::make_unique<InfoType>(std::move(owned_data_));
      data_cache->insert(std::move(new_entry));
    }
  } else {
    data_ptr_ =
        data_cache->at(EntryClass::EntryType)->template as<InfoType>()->get();
  }
}

// Template instantiation for pre-defined cache entries
template class HeuristicSummaryEntry<
    HeuristicCompileTime::VectorizableInputsAndOutputs>;
template class HeuristicSummaryEntry<
    HeuristicCompileTime::UnrollableInputsAndOutputs>;
template class HeuristicSummaryEntry<HeuristicCompileTime::ReductionTVs>;
template class HeuristicSummaryEntry<
    HeuristicCompileTime::PersistentBufferInfo>;
template class HeuristicSummaryEntry<
    HeuristicCompileTime::ScopePersistentFactorInfo>;
template class HeuristicSummaryEntry<HeuristicCompileTime::BroadcastMultiples>;

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
