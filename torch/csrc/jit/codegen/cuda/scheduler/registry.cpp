#include <c10/util/irange.h>
#include <torch/csrc/jit/codegen/cuda/executor_utils.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/registry.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {
// TODO: Deduplicate from compute_at.cpp
std::deque<std::deque<TensorView*>> tvChains(
    std::deque<std::deque<Val*>> val_chains) {
  std::deque<std::deque<TensorView*>> tv_chains(val_chains.size());
  for (size_t i = 0; i < val_chains.size(); i++) {
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
      if (tv->hasReduction()) {
        reduction_tvs.push_back(tv);
      }
    }

    // All tensor views that are eventually consumed to produce a reduction
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
  // accepted in the single reduction scheduler
  static bool hasPostReductionBCast(Fusion* fusion) {
    auto all_vals = fusion->usedMathVals();
    for (auto tv : ir_utils::filterByType<TensorView>(all_vals)) {
      // Welford can have 2 outputs, so do this on all found reduction tensor
      // views
      if (tv->hasReduction()) {
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
};
} // namespace

SchedulerRuntimeInfo::SchedulerRuntimeInfo(
    Fusion* complete_fusion,
    const at::ArrayRef<IValue>& inputs,
    bool create_expr_evaluator)
    : complete_fusion_(complete_fusion) {
  collectVectorizationInfo(inputs);
  if (create_expr_evaluator) {
    initializeExpressionEvaluator(inputs);
  }
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
  expression_evaluator_ =
      std::make_unique<ExpressionEvaluator>(complete_fusion_);
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
  auto tv_root = TensorDomain::noReductions(tv->getRootDomain());
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

bool SchedulerEntry::sameAs(const SchedulerEntry* other) {
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
inline bool isTrivialReduction(ReductionOp* red) {
  auto o_tv = red->out()->as<TensorView>();
  // Assuming graph unscheduled at this point.
  for (auto id : o_tv->getRootDomain()) {
    if (id->isReduction() && !id->extent()->isOneInt()) {
      return false;
    }
  }
  return true;
}

std::vector<ReductionOp*> findReductionOps(Fusion* fusion) {
  std::vector<ReductionOp*> red_ops;
  for (auto expr : fusion->exprs()) {
    if (auto red = dynamic_cast<ReductionOp*>(expr)) {
      if (!isTrivialReduction(red)) {
        red_ops.push_back(red);
      }
    }
  }
  return red_ops;
}

class SingleReductionScheduler : public SchedulerEntry {
 public:
  explicit SingleReductionScheduler(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info)
      : SchedulerEntry(ScheduleHeuristic::Reduction, true) {
    computeHeuristics(fusion, runtime_info);
  }

  //! Check if the reduction heuristics apply in given fusion
  static bool canSchedule(Fusion* fusion, SchedulerRuntimeInfo& runtime_info) {
    auto red_ops = findReductionOps(fusion);
    if (red_ops.size() != 1) {
      return false;
    }

    if (SchedulerTopologyChecker::hasPostReductionBCast(fusion)) {
      return false;
    }

    auto red_tv = red_ops[0]->out()->as<TensorView>();

    // Not allowing broadcasting reduction result to support
    //  grid reduction. This is an overkill might want to consider
    //  trying to get the heuristics and check only if grid reduction is
    //  required.
    //  TODO: We can actually allow broadcasts that doesn't get resolved
    //        in the same fusion, temporarily use a simplified detection
    //        where broadcast is allowed if it's at output and has no use
    auto dependent_vals = DependencyCheck::getAllDependentVals({red_tv});
    for (auto val : dependent_vals) {
      if (val->definition()->isA<BroadcastOp>() && !val->uses().empty()) {
        return false;
      }
    }

    return true;
  }

  void schedule(Fusion* fusion) override {
    FUSER_PERF_SCOPE("Schedule Single Reduction");
    scheduleReduction(fusion, rparams_);
  }

 private:
  void computeHeuristics(Fusion* fusion, SchedulerRuntimeInfo& runtime_info) {
    auto& expr_evaluator = runtime_info.expressionEvaluator();
    auto param = getReductionHeuristics(fusion, expr_evaluator);
    TORCH_INTERNAL_ASSERT(param.has_value());
    rparams_ = param.value();
  }
};

class PointWiseScheduler : public SchedulerEntry {
 public:
  explicit PointWiseScheduler(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info)
      : SchedulerEntry(ScheduleHeuristic::PointWise, false) {
    computeHeuristics(fusion, runtime_info.expressionEvaluator());
  }

  static bool canSchedule(Fusion* fusion, SchedulerRuntimeInfo& runtime_info) {
    auto red_ops = findReductionOps(fusion);
    return red_ops.empty();
  }

  void schedule(Fusion* fusion) override {
    FUSER_PERF_SCOPE("Schedule PointWise Fusion");
    schedulePointwise(fusion, pparams_);
  }

  void computeHeuristics(Fusion* fusion, ExpressionEvaluator& ee) {
    auto pparam = getPointwiseHeuristics(fusion, {}, ee);
    TORCH_INTERNAL_ASSERT(pparam.has_value());
    pparams_ = pparam.value();
  }
};

class NormalizationScheduler : public SchedulerEntry {
 public:
  explicit NormalizationScheduler(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info)
      : SchedulerEntry(ScheduleHeuristic::Normalization, true) {
    computeHeuristics(fusion, runtime_info);
  }

  void schedule(Fusion* fusion) override {
    FUSER_PERF_SCOPE("Schedule Normalization Fusion");
    scheduleNormalization(fusion, rparams_);
  }

  static bool canSchedule(Fusion* fusion, SchedulerRuntimeInfo& runtime_info) {
    // auto & expr_evaluator = runtime_info.expressionEvaluator();
    std::vector<TensorView*> reduction_tv;
    for (auto tv : scheduler_utils::allTvs(fusion)) {
      if (tv->hasReduction() && !fusion->hasInput(tv)) {
        reduction_tv.push_back(tv);
      }
    }

    if (reduction_tv.size() == 0) {
      // Use single reduction or pointwise logic
      return false;
    }

    if (SchedulerTopologyChecker::hasNonNormalizePostReductionBCast(fusion)) {
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

    for (auto red : reduction_tv) {
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
    for (size_t it = 1; it < reduction_tv.size(); it++) {
      if (!checkEquivalence(reduction_tv[it - 1], reduction_tv[it], root_map)) {
        return false;
      }
    }
    return true;
  }

 private:
  void computeHeuristics(Fusion* fusion, SchedulerRuntimeInfo& runtime_info) {
    auto& expr_evaluator = runtime_info.expressionEvaluator();
    auto rparams = getNormalizationHeuristics(fusion, expr_evaluator);
    TORCH_INTERNAL_ASSERT(rparams.has_value());
    rparams_ = rparams.value();
  }

  static bool checkEquivalence(
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
};

// Schedule Table
const std::vector<ScheduleHeuristic>& all_heuristics() {
  static const std::vector<ScheduleHeuristic> hlist = {
      ScheduleHeuristic::Reduction,
      ScheduleHeuristic::PointWise,
      ScheduleHeuristic::Normalization};
  return hlist;
}

} // namespace

// Simple dispatcher interface
bool SchedulerEntry::canSchedule(
    ScheduleHeuristic sh,
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info) {
  switch (sh) {
    case ScheduleHeuristic::PointWise:
      return PointWiseScheduler::canSchedule(fusion, runtime_info);
    case ScheduleHeuristic::Reduction:
      return SingleReductionScheduler::canSchedule(fusion, runtime_info);
    case ScheduleHeuristic::Normalization:
      return NormalizationScheduler::canSchedule(fusion, runtime_info);
    default:
      TORCH_INTERNAL_ASSERT(false, "unreachable");
      return false;
  }
  return false;
}

std::unique_ptr<SchedulerEntry> SchedulerEntry::makeEntry(
    ScheduleHeuristic sh,
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info) {
  switch (sh) {
    case ScheduleHeuristic::PointWise:
      return std::make_unique<PointWiseScheduler>(fusion, runtime_info);
    case ScheduleHeuristic::Reduction:
      return std::make_unique<SingleReductionScheduler>(fusion, runtime_info);
    case ScheduleHeuristic::Normalization:
      return std::make_unique<NormalizationScheduler>(fusion, runtime_info);
    default:
      TORCH_INTERNAL_ASSERT(false, "unreachable");
  }
  return nullptr;
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
    case ScheduleHeuristic::Normalization:
      return "normalization";
    default:
      TORCH_INTERNAL_ASSERT(false, "undefined schedule");
  }
  return "";
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
