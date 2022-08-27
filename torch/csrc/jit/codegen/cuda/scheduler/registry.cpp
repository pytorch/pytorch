#include <c10/util/irange.h>
#include <torch/csrc/jit/codegen/cuda/disjoint_set.h>
#include <torch/csrc/jit/codegen/cuda/executor_utils.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/debug_utils.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/pointwise.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/registry.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/transpose.h>
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
      // Reductions can have multiple outputs, so do this on all found reduction
      // tensor views
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
    //  of multi-output reductions, i.e. welford/grouped reduction. The
    //  reduction_tv vector is assumed to only have one of them.
    std::unordered_set<Val*> reduction_tv_set(
        reduction_tvs.begin(), reduction_tvs.end());

    for (auto red : reduction_tvs) {
      if (red->definition()) {
        if (ir_utils::isReductionOp(red->definition())) {
          auto outs = red->definition()->outputs();
          for (auto out_tv : ir_utils::filterByType<TensorView>(outs)) {
            reduction_tv_set.insert(out_tv);
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
  DisjointSets<Val*> component_sets;

  // Iterate through all used exprs
  for (auto expr : fusion->exprs()) {
    TORCH_INTERNAL_ASSERT(
        !expr->outputs().empty(), "unknown expr with zero output");

    // Each expr maps all its inputs and
    //  outputs to the same component
    auto output0 = expr->output(0);
    for (auto input : ir_utils::filterByType<TensorView>(expr->inputs())) {
      component_sets.mapEntries(output0, input);
    }
    for (auto output : expr->outputs()) {
      component_sets.mapEntries(output0, output);
    }
  }

  // Map aliased outputs
  for (auto alias_it : fusion->ioAlias()) {
    component_sets.mapEntries(alias_it.first, alias_it.second);
  }

  // Check connected-ness:
  //  If there is no independent compute flow
  // on this fusion graph, all outputs will be
  // equivalent/connected to the first output.
  auto output0 = fusion->outputs()[0];
  for (auto output : fusion->outputs()) {
    if (!component_sets.strictAreMapped(output0, output)) {
      return false;
    }
  }
  return true;
}

} // namespace

void SchedulerRuntimeInfo::initialize(
    const KernelArgumentHolder& args,
    bool create_expr_evaluator) {
  TORCH_INTERNAL_ASSERT(
      complete_fusion_->inputs().size() == args.size(),
      "Invalid number of arguments passed in for provided fusion group.");

  for (auto inp_i : c10::irange(args.size())) {
    auto kernel_arg = args[inp_i];
    // Note: we are skipping CpuScalar tensor here
    if (auto tensor_arg_abstract =
            dynamic_cast<const TensorArgAbstract*>(kernel_arg)) {
      auto fusion_inp = complete_fusion_->inputs()[inp_i];
      auto data_ptr = tensor_arg_abstract->getPointer();
      input_ptrs_[fusion_inp] = (size_t)data_ptr;
    }
  }

  expression_evaluator_ =
      std::make_unique<ExpressionEvaluator>(complete_fusion_);
  if (create_expr_evaluator) {
    initializeExpressionEvaluator(args);
  }
  index_mode_ = args.getIndexMode();
}

SchedulerRuntimeInfo::SchedulerRuntimeInfo(
    Fusion* complete_fusion,
    const KernelArgumentHolder& args,
    bool create_expr_evaluator)
    : complete_fusion_(complete_fusion) {
  initialize(args, create_expr_evaluator);
}

// TODO: remove this one
SchedulerRuntimeInfo::SchedulerRuntimeInfo(
    Fusion* complete_fusion,
    const at::ArrayRef<at::IValue>& aten_inputs,
    bool create_expr_evaluator)
    : complete_fusion_(complete_fusion) {
  KernelArgumentHolder args =
      KernelArgumentHolder::createKernelArgumentHolder(aten_inputs);
  initialize(args, create_expr_evaluator);
}

// TODO: Output tensors could have an alignment that is not 16 Bytes passed in
// from user.
size_t SchedulerRuntimeInfo::ptrOf(TensorView* tv) {
  if (input_ptrs_.find(tv) != input_ptrs_.end()) {
    return input_ptrs_.at(tv);
  }
  return max_alignment_size_in_byte;
}

void SchedulerRuntimeInfo::initializeExpressionEvaluator(
    const KernelArgumentHolder& args) {
  // TODO: refactor bindFusionInputs to better support this
  //  use case, i.e. support construct and bind input.
  *expression_evaluator_ =
      executor_utils::bindFusionInputs(args, complete_fusion_);
}

size_t SchedulerRuntimeInfo::computeAlignmentSize(size_t ptr_address) {
  size_t alignment_size = 1;
  size_t next_alignment_size = 2;

  while (next_alignment_size <= max_alignment_size_in_byte &&
         ptr_address % next_alignment_size == 0) {
    alignment_size = next_alignment_size;
    next_alignment_size *= 2;
  }
  return alignment_size;
}

size_t SchedulerRuntimeInfo::getAlignmentSize(TensorView* tv) {
  auto alignment_entry = alignment_map_.find(tv);
  if (alignment_entry != alignment_map_.end()) {
    return alignment_entry->second;
  }

  auto alignment_size = SchedulerRuntimeInfo::computeAlignmentSize(ptrOf(tv));
  alignment_map_[tv] = alignment_size;
  return alignment_size;
}

// Gets maximum vectorizable width of tv, assumes we can merge across all
// iteration domains if contiguous. Cannot permute the dimensions to fix
// contiguity.
size_t SchedulerRuntimeInfo::getMaxVectorizableWidth(TensorView* tv) {
  // Gets the vectorizable width of the tv starting from the inner most
  // dimension, working its way towards the outer most dimension, if they're
  // contiguous. Ignores broadcast and reduction domains.
  auto max_vectorword_map_it_ = max_vectorword_map_.find(tv);
  if (max_vectorword_map_it_ != max_vectorword_map_.end()) {
    return max_vectorword_map_it_->second;
  }

  // If we don't have an record, either it is a tv with innermost broadcast,
  // or it is an intermediate tensor allocated by fuser. Logic copied to get
  // root according to scheduler_utils::innerMostRootDim.
  auto tv_root = tv->hasReduction() && tv->hasRFactor()
      ? tv->getRootDomain()
      : tv->getMaybeRFactorDomain();

  auto tv_root_no_reductions = TensorDomain::noReductions(tv_root);

  auto contiguity = tv->domain()->contiguity();
  // Appears after reductions the reduction domain often has a contiguity entry.
  // This only matters if the result of the reduction is an output
  if (contiguity.size() == tv_root.size() &&
      contiguity.size() != tv_root_no_reductions.size()) {
    std::vector<bool> new_contiguity;
    for (auto i : c10::irange(tv_root.size())) {
      if (!tv_root[i]->isReduction()) {
        new_contiguity.push_back(contiguity[i]);
      }
    }
    contiguity = new_contiguity;
  }
  tv_root = tv_root_no_reductions;

  auto tv_root_size = tv_root.size();

  // Filter out 0-dim tensors
  if (tv_root_size < 1) {
    return 1;
  }

  // Filter out mismatched contiguity info
  if (tv_root_size != contiguity.size()) {
    return 1;
  }

  size_t item_size =
      dataTypeSize(tv->dtype(), indexModeToDtype(getIndexMode()));

  // Alignment should always at least be the data type size
  TORCH_INTERNAL_ASSERT(getAlignmentSize(tv) % item_size == 0);
  size_t max_vector_size = getAlignmentSize(tv) / item_size;

  if (max_vector_size == 1) {
    return 1;
  }

  auto numel = 1;
  for (auto i : c10::irange(tv_root_size)) {
    auto root_i = tv_root_size - i - 1;
    auto root_id = tv_root[root_i];

    if (root_id->extent()->isOneInt() || root_id->isBroadcast()) {
      continue;
    }

    // Not contiguous
    if (!contiguity[root_i]) {
      break;
    }

    auto dim_size = expression_evaluator_->evaluate(root_id->extent());
    // Inference failed for some reason, assume not-contiguous at this point
    if (!dim_size.has_value()) {
      break;
    }

    // Still contiguous
    numel *= dim_size->as<int64_t>();
  }

  // Assuming intermediate tensors have friendly alignment, and
  //  all contiguity true. Determine the largest power of 2 below
  //  innermost dimension size for the word size of vectorizaiton
  size_t vector_size = 1;
  size_t next_vector_size = 2;
  while (next_vector_size <= max_vector_size && next_vector_size <= numel &&
         numel % next_vector_size == 0) {
    vector_size = next_vector_size;
    next_vector_size *= 2;
  }

  // save output to avoid re-compute
  max_vectorword_map_[tv] = vector_size;

  return vector_size;
}

// Gets the vectorizable width of the inner most dimension of tv if it's
// contiguous. Ignores inner most dimensions that are broadcast or reduction.
size_t SchedulerRuntimeInfo::getInnerDimVectorizableWidth(TensorView* tv) {
  auto inner_vectorword_map_it_ = inner_vectorword_map_.find(tv);
  if (inner_vectorword_map_it_ != inner_vectorword_map_.end()) {
    return inner_vectorword_map_it_->second;
  }

  // If we don't have an record, either it is a tv with innermost broadcast,
  // or it is an intermediate tensor allocated by fuser. Logic copied to get
  // root according to scheduler_utils::innerMostRootDim.
  auto tv_root = tv->hasReduction() && tv->hasRFactor()
      ? tv->getRootDomain()
      : tv->getMaybeRFactorDomain();

  auto tv_root_no_reductions = TensorDomain::noReductions(tv_root);

  auto contiguity = tv->domain()->contiguity();
  // Appears after reductions the reduction domain often has a contiguity entry.
  // This only matters if the result of the reduction is an output
  if (contiguity.size() == tv_root.size() &&
      contiguity.size() != tv_root_no_reductions.size()) {
    std::vector<bool> new_contiguity;
    for (auto i : c10::irange(tv_root.size())) {
      if (!tv_root[i]->isReduction()) {
        new_contiguity.push_back(contiguity[i]);
      }
    }
    contiguity = new_contiguity;
  }
  tv_root = tv_root_no_reductions;

  auto tv_root_size = tv_root.size();

  // Filter out 0-dim tensors
  if (tv_root_size < 1) {
    return 1;
  }

  // Filter out mismatched contiguity info
  if (tv_root_size != contiguity.size()) {
    return 1;
  }

  auto inner_most_dim = scheduler_utils::innerMostRootDim(tv);

  int id_pos = -1;
  for (auto root_i : c10::irange(tv_root_size)) {
    if (tv_root[root_i] == inner_most_dim) {
      id_pos = root_i;
      break;
    }
  }

  // Something went wrong with finding the inner most dimension, just
  // return 1.
  if (id_pos == -1) {
    return 1;
  }

  // If the inner most dimension is not contiguous return 1
  if (!contiguity[id_pos]) {
    return 1;
  }

  size_t item_size =
      dataTypeSize(tv->dtype(), indexModeToDtype(getIndexMode()));

  // Alignment should always at least be the data type size
  TORCH_INTERNAL_ASSERT(getAlignmentSize(tv) % item_size == 0);
  size_t max_vector_size = getAlignmentSize(tv) / item_size;

  // Assuming intermediate tensors have friendly alignment, and
  //  all contiguity true. Determine the largest power of 2 below
  //  innermost dimension size for the word size of vectorizaiton
  size_t vector_size = 1;
  size_t next_vector_size = 2;
  auto maybe_inner_dimension_size =
      expression_evaluator_->evaluate(inner_most_dim->extent());
  TORCH_INTERNAL_ASSERT(maybe_inner_dimension_size.has_value());
  size_t inner_dimension_size = maybe_inner_dimension_size->as<int64_t>();

  while (next_vector_size <= max_vector_size &&
         next_vector_size <= inner_dimension_size &&
         inner_dimension_size % next_vector_size == 0) {
    vector_size = next_vector_size;
    next_vector_size *= 2;
  }

  // save output to avoid re-compute
  inner_vectorword_map_[tv] = vector_size;

  return vector_size;
}

bool SchedulerEntry::sameAs(const SchedulerEntry* other) {
  if (heuristc_ != other->heuristc_) {
    return false;
  }
  if (index_mode_ != other->index_mode_) {
    return false;
  }
  return params_->sameAs(other->params_);
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

// Reusing some code from lowering specifically in lower_trivial_broadcast.cpp
// ConcretizedBroadcastDomains::maybeNonUniquelyConcretized this checks if
// there's a broadcast iteration domain that's being broadcasted to seemingly
// different extents, meaning we don't know in the kernel if the dimension is
// being broadcasted to one size multiple times or different sizes. This is a
// hard to optimize problem and likely indicates we shouldn't be fusing.
bool hasNonUniqueBcast(Fusion* fusion) {
  ConcretizedBroadcastDomains concretize_info;
  concretize_info.build(fusion);

  for (auto tv : ir_utils::allTvs(fusion)) {
    for (auto id : tv->getRootDomain()) {
      if (concretize_info.maybeNonUniquelyConcretized(id)) {
        return true;
      }
    }
  }
  return false;
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
      : SchedulerEntry(ScheduleHeuristic::Reduction) {
    computeHeuristics(fusion, runtime_info, data_cache);
  }

  //! Check if the reduction heuristics apply in given fusion
  static bool canScheduleCompileTime(Fusion* fusion) {
    // Temporarily disallow view in reduction scheduler
    // TODO Add more testing before enabling
    auto view_tvs = scheduler_utils::getViewTVs(fusion);
    if (view_tvs.size() > 0) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Reduction, "No support for view op");
      return false;
    }

    // Needs at least one non-trivial reduction to consider.
    if (ir_utils::getReductionOps(fusion, true /* ignore_trivial */).empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Reduction, "No reduction op to schedule");
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
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Reduction, "No support for transpose op");
      return false;
    }

    if (hasNonUniqueBcast(fusion)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Reduction,
          "Broadcasting dimension might be broadcasting to multiple sizes.");
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
            scheduler_debug_utils::canScheduleRejectReason(
                ScheduleHeuristic::Reduction,
                "Inconsistent reduction axes ",
                red,
                "is not ",
                axis_count);
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
          scheduler_debug_utils::canScheduleRejectReason(
              ScheduleHeuristic::Reduction,
              "Un-mapped multi-reduction: ",
              reduction_tvs[it - 1],
              " ",
              reduction_tvs[it]);
          return false;
        }
      }
    }

    // Doesn't allow persistent kernels in this scheduler
    auto persistent_buffer_info = scheduler_utils::persistentBuffers(fusion);
    if (persistent_buffer_info.persistent_buffers.size() > 0) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Reduction,
          "need persistent buffers that reduction scheduler doesn't handle");
      return false;
    }

    if (!SchedulerTopologyChecker::supportedPostReductionFusion(
            fusion, reduction_tvs) ||
        SchedulerTopologyChecker::hasPostReductionBCast(fusion)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Reduction,
          "has unsupported post reduction fusion");
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
    scheduleReduction(fusion, reductionParams());
  }

 private:
  void computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    params_ = getReductionHeuristics(fusion, runtime_info, data_cache);
    TORCH_INTERNAL_ASSERT(params_ != nullptr);
  }
};

class PointWiseScheduler : public SchedulerEntry {
 public:
  explicit PointWiseScheduler(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr)
      : SchedulerEntry(ScheduleHeuristic::PointWise) {
    computeHeuristics(fusion, runtime_info, data_cache);
  }

  static bool canScheduleCompileTime(Fusion* fusion) {
    //   Currently using the same path as the scheduler
    // to eliminate mismatch between canSchedule and
    // schedule pointwise.
    if (!hasReferenceTensorView(fusion)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::PointWise, "cannot find reference tensor");
      return false;
    }

    auto reduction_ops =
        ir_utils::getReductionOps(fusion, true /* ignore_trivial */);

    if (!reduction_ops.empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::PointWise, "no support for reduction ops");
      return false;
    }

    if (hasNonUniqueBcast(fusion)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::PointWise,
          "Broadcasting dimension might be broadcasting to multiple sizes.");
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
    FUSER_PERF_SCOPE("Schedule PointWise Fusion");
    schedulePointwise(fusion, pointwiseParams());
  }

  void computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    params_ = getPointwiseHeuristics(fusion, runtime_info, data_cache);
    TORCH_INTERNAL_ASSERT(params_ != nullptr);
  }
};

class PersistentKernelScheduler : public SchedulerEntry {
 public:
  explicit PersistentKernelScheduler(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr)
      : SchedulerEntry(ScheduleHeuristic::Persistent) {
    computeHeuristics(fusion, runtime_info, data_cache);
  }

  void schedule(Fusion* fusion) override {
    FUSER_PERF_SCOPE("Schedule Persistent Fusion");
    schedulePersistentKernel(fusion, reductionParams());
  }

  static bool canScheduleCompileTime(Fusion* fusion) {
    // Needs at least one non-trivial reduction to consider.
    if (ir_utils::getReductionOps(fusion, true /* ignore_trivial */).empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent, "needs a reduction op");
      return false;
    }

    auto reduction_ops =
        ir_utils::getReductionOps(fusion, false /* ignore_trivial */);

    auto view_tvs = scheduler_utils::getViewTVs(fusion);
    if (view_tvs.size() > 0) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent, "no support for view");
      return false;
    }

    if (hasNonUniqueBcast(fusion)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent,
          "Broadcasting dimension might be broadcasting to multiple sizes.");
      return false;
    }

    auto reduction_tvs =
        scheduler_utils::getReductionTvs(fusion, false /* ignore_trivial */);

    if (reduction_tvs.size() == 0) {
      // Use pointwise logic
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent, "no reduction tv");
      return false;
    }

    if (findTransposeOps(fusion).size() > 0) {
      // Use pointwise logic
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent, "no support for transpose");
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
          scheduler_debug_utils::canScheduleRejectReason(
              ScheduleHeuristic::Persistent,
              "inconsistent reduction root size");
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
        scheduler_debug_utils::canScheduleRejectReason(
            ScheduleHeuristic::Persistent,
            "unmapped reduction ",
            reduction_tvs[it - 1],
            " and ",
            reduction_tvs[it]);
        return false;
      }
    }

    // Only accept persistent kernels
    auto persistent_buffer_info = scheduler_utils::persistentBuffers(fusion);
    if (persistent_buffer_info.persistent_buffers.size() == 0) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent, "no persistent buffer identified");
      return false;
    }

    if (SchedulerTopologyChecker::hasNonNormalizePostReductionBCast(fusion)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent,
          "unsupported post reduction normalization");
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
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent,
          "not enough registers for persistence");
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
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent, "unsupported cross grid persistence");

      return false;
    }

    return true;
  }

 private:
  void computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    params_ = getPersistentHeuristics(fusion, runtime_info, data_cache);
    TORCH_INTERNAL_ASSERT(params_ != nullptr);
  }
};

class TransposeScheduler : public SchedulerEntry {
 public:
  explicit TransposeScheduler(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr)
      : SchedulerEntry(ScheduleHeuristic::Transpose) {
    computeHeuristics(fusion, runtime_info, data_cache);
  }

  static bool canScheduleCompileTime(Fusion* fusion) {
    if (!isOptionEnabled(EnableOption::TransposeScheduler)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Transpose, "not enabled");
      return false;
    }

    // Temporarily disallow view in transpose scheduler
    // TODO Add more testing before enabling
    auto view_tvs = scheduler_utils::getViewTVs(fusion);
    if (view_tvs.size() > 0) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Transpose, "No support for view op");
      return false;
    }

    if (!hasAtLeastTwoValidGroups(fusion)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Transpose,
          "cannot find two mismatching inner most dimensions");
      return false;
    }

    // TODO: add support for trivial reduction
    auto reduction_ops =
        ir_utils::getReductionOps(fusion, false /* ignore_trivial */);

    if (!reduction_ops.empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Transpose, "no support for reduction ops");
      return false;
    }

    if (hasNonUniqueBcast(fusion)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Transpose,
          "Broadcasting dimension might be broadcasting to multiple sizes.");
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
    FUSER_PERF_SCOPE("Schedule Transpose Fusion");
    scheduleTranspose(fusion, transposeParams());
  }

 private:
  void computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    params_ = getTransposeHeuristics(fusion, runtime_info, data_cache);
    TORCH_INTERNAL_ASSERT(params_ != nullptr);
  }
};

// Schedule Table
const std::vector<ScheduleHeuristic>& all_heuristics() {
  static const std::vector<ScheduleHeuristic> hlist = {
      ScheduleHeuristic::Reduction,
      ScheduleHeuristic::Transpose,
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
    case ScheduleHeuristic::Transpose:
      return checkCanSchedule<TransposeScheduler>(
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
    case ScheduleHeuristic::Transpose:
      scheduler_entry = std::make_unique<TransposeScheduler>(
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
      scheduler_debug_utils::canScheduleMessage("***Accepted*** as: ", sh);
      return sh;
    }
  }
  return c10::nullopt;
}

size_t SchedulerEntryHash::operator()(const SchedulerEntry& se) const {
  return se.params()->hash();
}

std::string toString(ScheduleHeuristic sh) {
  switch (sh) {
    case ScheduleHeuristic::PointWise:
      return "pointwise";
    case ScheduleHeuristic::Reduction:
      return "reduction";
    case ScheduleHeuristic::Persistent:
      return "persistent";
    case ScheduleHeuristic::Transpose:
      return "transpose";
    default:
      TORCH_INTERNAL_ASSERT(false, "undefined schedule");
  }
  return "";
}

std::ostream& operator<<(std::ostream& os, ScheduleHeuristic sh) {
  os << toString(sh);
  return os;
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
    case ScheduleHeuristic::Transpose:
      getTransposeHeuristics(fusion, runtime_info, this);
      TransposeScheduler::canScheduleRunTime(fusion, runtime_info, this);
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
      TORCH_INTERNAL_ASSERT(entry_type_map_.count(EntryType::DOMAIN_MAP));
      TORCH_INTERNAL_ASSERT(
          entry_type_map_.count(EntryType::REFERENCE_TENSORS));
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
    case ScheduleHeuristic::Transpose: {
      TORCH_INTERNAL_ASSERT(entry_type_map_.count(
          EntryType::INPUTS_AND_OUTPUTS_INNER_DIM_GROUPS));
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
template class HeuristicSummaryEntry<HeuristicCompileTime::DomainMap>;
template class HeuristicSummaryEntry<HeuristicCompileTime::ReferenceTensors>;
template class HeuristicSummaryEntry<
    HeuristicCompileTime::VectorizableInputsAndOutputs>;
template class HeuristicSummaryEntry<
    HeuristicCompileTime::InputsOutputsInnerDimGroups>;
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
