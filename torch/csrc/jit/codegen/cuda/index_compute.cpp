#include <torch/csrc/jit/codegen/cuda/index_compute.h>

#include <c10/util/Exception.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

// A merge is contiguous if:
//   Inputs of outer are to the left in the root domain of the inputs of RHS.
//   All inputs are contiguous in the root domain:
//     - All marked as contiguous
//     - Only gaps between inputs are broadcast or reductoin dims
//   There are no split transformations performed on outer or inner
//   All transformations on outer or inner are contiguous merges
// If this criteria holds, then we can index the input root domains of this
// merge with the indexing provided to the output of the merge in the backward
// index pass

class ContigIDs : public OptInDispatch {
 private:
  using OptInDispatch::handle;

  // Mark if ids are result of contigous merges
  std::unordered_set<kir::IterDomain*> contig_ids;
  // Given contiguous domain, return all iter domains within its history.
  std::unordered_map<kir::IterDomain*, std::unordered_set<kir::IterDomain*>>
      within_contig_ids;
  const std::vector<IterDomain*>& root_domain_;
  const std::vector<bool>& root_contiguity_;
  std::unordered_map<IterDomain*, bool> is_contig_root;

  bool inRoot(const std::vector<IterDomain*>& ids) {
    return std::all_of(ids.begin(), ids.end(), [this](IterDomain* id) {
      return is_contig_root.find(id) != is_contig_root.end();
    });
  }

  bool isContig(kir::IterDomain* id) {
    return contig_ids.find(id) != contig_ids.end();
  }

  // Split outputs are not conitguous, don't need to do anything.
  void handle(Split*) override {}

  void handle(Merge* merge) override {
    // If either input is non-contiguous so is output.
    auto inner = merge->inner();
    auto outer = merge->outer();
    if (!isContig(GpuLower::lowerValue(inner)->as<kir::IterDomain>()) ||
        !isContig(GpuLower::lowerValue(outer)->as<kir::IterDomain>())) {
      return;
    }

    // Grab inputs, make sure they're in root domain, check if they're
    // contiguous.

    auto lhs_inputs =
        ir_utils::iterDomainInputsOfOrderedAs({outer}, root_domain_);
    auto rhs_inputs =
        ir_utils::iterDomainInputsOfOrderedAs({inner}, root_domain_);

    TORCH_INTERNAL_ASSERT(
        inRoot(lhs_inputs) && inRoot(rhs_inputs),
        "Found an invalid merge operation, inputs of its arguments are not in the root domain.");

    std::deque<IterDomain*> ordered_inputs(
        lhs_inputs.begin(), lhs_inputs.end());
    ordered_inputs.insert(
        ordered_inputs.end(), rhs_inputs.begin(), rhs_inputs.end());

    // If any root input is not contig, output is not contig
    if (!(std::all_of(
            ordered_inputs.begin(),
            ordered_inputs.end(),
            [this](IterDomain* id) {
              return is_contig_root.at(id) && !id->isBroadcast() &&
                  !id->isReduction();
            }))) {
      return;
    }

    std::deque<IterDomain*> root_copy(root_domain_.begin(), root_domain_.end());

    // Forward to first matching argument
    while (!root_copy.empty() && !ordered_inputs.empty()) {
      if (root_copy.front() != ordered_inputs.front()) {
        root_copy.pop_front();
      } else {
        break;
      }
    }

    // Forward through all matching arguments
    while (!root_copy.empty() && !ordered_inputs.empty()) {
      if (root_copy.front() == ordered_inputs.front()) {
        root_copy.pop_front();
        ordered_inputs.pop_front();
        // We probably should be able to make access contiguous through
        // reduction domains, however, for now it's causing issues in predicate
        // generation. See test: ReductionSchedulerMultiDimNonFastest
        //  } else if (
        //     root_copy.front()->isReduction() ||
        //     root_copy.front()->isBroadcast()) {
        //   root_copy.pop_front();
      } else {
        break;
      }
    }

    // If we matched all inputs, the output is contiguous. Only want to keep the
    // top contig ID, lower ids should be placed in the "within_contig_ids" map
    // of top id.
    auto kir_inner =
        GpuLower::lowerValue(merge->inner())->as<kir::IterDomain>();
    auto kir_outer =
        GpuLower::lowerValue(merge->outer())->as<kir::IterDomain>();
    auto kir_out = GpuLower::lowerValue(merge->out())->as<kir::IterDomain>();
    if (ordered_inputs.empty()) {
      if (contig_ids.find(kir_inner) != contig_ids.end()) {
        contig_ids.erase(kir_inner);
      }

      if (contig_ids.find(kir_outer) != contig_ids.end()) {
        contig_ids.erase(kir_outer);
      }

      contig_ids.emplace(kir_out);

      std::unordered_set<kir::IterDomain*> within_out;
      within_out.emplace(kir_inner);
      if (within_contig_ids.find(kir_inner) != within_contig_ids.end()) {
        auto in_inner = within_contig_ids.at(kir_inner);
        within_out.insert(in_inner.begin(), in_inner.end());
        within_contig_ids.erase(kir_inner);
      }

      within_out.emplace(kir_outer);
      if (within_contig_ids.find(kir_outer) != within_contig_ids.end()) {
        auto in_outer = within_contig_ids.at(kir_outer);
        within_out.insert(in_outer.begin(), in_outer.end());
        within_contig_ids.erase(kir_outer);
      }

      within_contig_ids[kir_out] = within_out;
    }
  }

 public:
  ContigIDs() = delete;

  // Check through thie history of ids whose inputs map to root_domain with
  // contiguity root_contiguity. Return unordered_set of all merges that are
  // contiguous.
  ContigIDs(
      const std::vector<IterDomain*>& ids,
      const std::vector<IterDomain*>& _root_domain,
      const std::vector<bool>& _root_contiguity)
      : root_domain_(_root_domain), root_contiguity_(_root_contiguity) {
    if (ids.empty()) {
      return;
    }

    TORCH_INTERNAL_ASSERT(
        root_domain_.size() == root_contiguity_.size(),
        "Arguments don't match ",
        root_domain_.size(),
        " != ",
        root_contiguity_.size());

    for (size_t i = 0; i < root_domain_.size(); i++) {
      if (root_contiguity_[i]) {
        auto kir_root_domain_i =
            GpuLower::lowerValue(root_domain_[i])->as<kir::IterDomain>();
        contig_ids.emplace(kir_root_domain_i);
        within_contig_ids[kir_root_domain_i] =
            std::unordered_set<kir::IterDomain*>();
      }
      is_contig_root[root_domain_[i]] = root_contiguity_[i];
    }

    auto exprs = ExprSort::getExprs(ids[0]->fusion(), {ids.begin(), ids.end()});

    for (auto expr : exprs) {
      handle(expr);
    }
  }

  const std::unordered_set<kir::IterDomain*> contigIDs() const {
    return contig_ids;
  }

  const std::
      unordered_map<kir::IterDomain*, std::unordered_set<kir::IterDomain*>>
      withinContigIDs() const {
    return within_contig_ids;
  }
};

} // namespace

void IndexCompute::handle(Split* split) {
  auto in_id = GpuLower::lowerValue(split->in())->as<kir::IterDomain>();
  auto outer_id = GpuLower::lowerValue(split->outer())->as<kir::IterDomain>();
  auto inner_id = GpuLower::lowerValue(split->inner())->as<kir::IterDomain>();

  auto outer_it = index_map_.find(outer_id);
  auto inner_it = index_map_.find(inner_id);
  if (outer_it == index_map_.end() || inner_it == index_map_.end())
    return;

  auto outer_ind = outer_it->second;
  auto inner_ind = inner_it->second;

  bool outer_zero = outer_ind->isZeroInt();
  bool inner_zero = inner_ind->isZeroInt();

  bool outer_bcast = outer_id->isBroadcast();
  bool inner_bcast = inner_id->isBroadcast();

  // Zero inds because a dim is bcast is part of normal traversal, if it's not
  // bcast but is zero ind then it's from local or smem. In the latter case we
  // want to propagate this property.
  if ((outer_zero && !outer_bcast) || (inner_zero && !inner_bcast) ||
      hasZeroMerged(inner_id) || hasZeroMerged(outer_id)) {
    zero_merged_in_.emplace(in_id);
  } else {
    // Maybe clear in_id as it could have been mapped over from another
    // IndexCompute. Uncertain if this is needed but seems to be safe.
    if (hasZeroMerged(in_id)) {
      zero_merged_in_.erase(in_id);
    }
  }

  kir::IrBuilder ir_builder(GpuLower::current()->kernel());

  if (outer_zero && inner_zero) {
    index_map_[in_id] = ir_builder.create<kir::Int>(0);
    extent_map_[in_id] = ir_builder.create<kir::Int>(0);
  } else if (outer_zero) {
    index_map_[in_id] = inner_ind;
    zero_merged_in_.emplace(in_id);
    extent_map_[in_id] = getExtent(inner_id);
  } else if (inner_zero) {
    index_map_[in_id] = outer_ind;
    zero_merged_in_.emplace(in_id);
    extent_map_[in_id] = getExtent(outer_id);
  } else {
    index_map_[in_id] = ir_builder.addExpr(
        ir_builder.mulExpr(outer_ind, getExtent(inner_id)), inner_ind);
    if (extent_map_.find(outer_id) != extent_map_.end() ||
        extent_map_.find(inner_id) != extent_map_.end()) {
      extent_map_[in_id] =
          ir_builder.mulExpr(getExtent(outer_id), getExtent(inner_id));
    }
  }
}

void IndexCompute::handle(Merge* merge) {
  auto out_id = GpuLower::lowerValue(merge->out())->as<kir::IterDomain>();
  auto outer_id = GpuLower::lowerValue(merge->outer())->as<kir::IterDomain>();
  auto inner_id = GpuLower::lowerValue(merge->inner())->as<kir::IterDomain>();

  auto out_it = index_map_.find(out_id);
  if (out_it == index_map_.end())
    return;

  auto out_ind = out_it->second;

  kir::IrBuilder ir_builder(GpuLower::current()->kernel());
  auto zero = ir_builder.create<kir::Int>(0);

  if (out_ind->isZeroInt()) {
    index_map_[outer_id] = zero;
    index_map_[inner_id] = zero;
    extent_map_[outer_id] = zero;
    extent_map_[inner_id] = zero;
    return;
  }

  if (!hasZeroMerged(out_id) && contig_ids.find(out_id) != contig_ids.end()) {
    auto input_ids = ir_utils::iterDomainInputsOfOrderedAs(
        {merge->out()}, td_->getRootDomain());

    // Shouldn't hit this, but don't want to segfault if somehow we do.
    TORCH_INTERNAL_ASSERT(!input_ids.empty());

    for (auto root_id : input_ids) {
      index_map_[GpuLower::lowerValue(root_id)->as<kir::IterDomain>()] = zero;
    }

    index_map_[GpuLower::lowerValue(*(input_ids.end() - 1))
                   // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
                   ->as<kir::IterDomain>()] = out_ind;
    return;
  }

  Val* inner_extent = getExtent(inner_id);
  Val* outer_extent = getExtent(outer_id);

  if (inner_id->isBroadcast() && inner_extent->isOneInt()) {
    index_map_[outer_id] = out_ind;
    index_map_[inner_id] = zero;

    extent_map_[outer_id] = getExtent(out_id);
  } else if (outer_id->isBroadcast() && outer_extent->isOneInt()) {
    index_map_[outer_id] = zero;
    index_map_[inner_id] = out_ind;

    extent_map_[inner_id] = getExtent(out_id);
  } else if (hasZeroMerged(out_id)) {
    index_map_[inner_id] = out_ind;
    extent_map_[inner_id] = getExtent(out_id);

    index_map_[outer_id] = zero;
    extent_map_[outer_id] = zero;

    zero_merged_in_.emplace(inner_id);
    zero_merged_in_.emplace(outer_id);
  } else {
    Val* I = inner_extent;

    Val* outer_ind = ir_builder.divExpr(out_ind, I);
    Val* inner_ind = ir_builder.modExpr(out_ind, I);

    index_map_[outer_id] = outer_ind;
    index_map_[inner_id] = inner_ind;
  }
}

void IndexCompute::handle(Expr* e) {
  switch (e->getExprType().value()) {
    case (ExprType::Split):
    case (ExprType::Merge):
      break;
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Invalid expr type found in transform traversal.");
  }
  BackwardVisitor::handle(e);
}

// Otherwise warning on runBackward as it hides an overloaded virtual
// using TransformIter::runBackward;
IndexCompute::IndexCompute(
    const TensorDomain* _td,
    std::unordered_map<kir::IterDomain*, Val*> initial_index_map,
    std::unordered_map<kir::IterDomain*, Val*> _extent_map,
    std::unordered_set<kir::IterDomain*> _zero_merged_in,
    const std::vector<bool>& root_contiguity)
    : td_(_td),
      index_map_(std::move(initial_index_map)),
      extent_map_(std::move(_extent_map)),
      zero_merged_in_(std::move(_zero_merged_in)) {
  FUSER_PERF_SCOPE("IndexCompute::IndexCompute");

  // Make sure we recompute any indices we can that map to a contiguous access
  // in physical memory.
  if (std::any_of(root_contiguity.begin(), root_contiguity.end(), [](bool b) {
        return b;
      })) {
    ContigIDs contig_finder(
        td_->domain(), td_->getRootDomain(), root_contiguity);
    contig_ids = contig_finder.contigIDs();
    auto within_contig = contig_finder.withinContigIDs();
    for (auto contig_id : contig_ids) {
      if (index_map_.find(contig_id) != index_map_.end()) {
        TORCH_INTERNAL_ASSERT(
            within_contig.find(contig_id) != within_contig.end());
        for (auto id : within_contig.at(contig_id)) {
          index_map_.erase(id);
        }
      }
    }
  }

  const std::vector<Val*> domain_vals(
      td_->domain().begin(), td_->domain().end());

  traverseFrom(td_->fusion(), domain_vals, false);
}

Val* IndexCompute::getExtent(kir::IterDomain* id) {
  if (extent_map_.find(id) != extent_map_.end()) {
    return extent_map_.at(id);
  } else {
    return id->extent();
  }
}

bool IndexCompute::hasZeroMerged(kir::IterDomain* id) {
  return zero_merged_in_.find(id) != zero_merged_in_.end();
}

IndexCompute IndexCompute::updateIndexCompute(
    const TensorDomain* new_td,
    const std::unordered_map<IterDomain*, IterDomain*>& id_map,
    std::unordered_map<kir::IterDomain*, Val*> new_index_entries,
    const std::vector<bool>& root_contiguity) {
  FUSER_PERF_SCOPE("updateIndexCompute");

  std::unordered_map<kir::IterDomain*, Val*> updated_index_map =
      std::move(new_index_entries);
  std::unordered_map<kir::IterDomain*, Val*> updated_extent_map;
  std::unordered_set<kir::IterDomain*> updated_zero_merged_in;

  for (auto id_entry : id_map) {
    kir::IterDomain* prev_id =
        GpuLower::lowerValue(id_entry.first)->as<kir::IterDomain>();
    kir::IterDomain* new_id =
        GpuLower::lowerValue(id_entry.second)->as<kir::IterDomain>();

    if (index_map_.find(prev_id) != index_map_.end()) {
      updated_index_map[new_id] = index_map_.at(prev_id);
    }

    if (extent_map_.find(prev_id) != extent_map_.end()) {
      updated_extent_map[new_id] = extent_map_.at(prev_id);
    }

    if (zero_merged_in_.find(prev_id) != zero_merged_in_.end()) {
      updated_zero_merged_in.emplace(new_id);
    }
  }

  return IndexCompute(
      new_td,
      updated_index_map,
      updated_extent_map,
      updated_zero_merged_in,
      root_contiguity);
}

std::vector<bool> IndexCompute::contiguityAnd(
    const std::vector<bool>& contig1,
    const std::vector<bool>& contig2) {
  TORCH_INTERNAL_ASSERT(
      contig1.size() == contig2.size(),
      "Called contiguityAnd with mismatched vectors.");

  std::vector<bool> contig_result;
  std::transform(
      contig1.begin(),
      contig1.end(),
      contig2.begin(),
      std::back_inserter(contig_result),
      std::logical_and<>());
  return contig_result;
}

// TODO: use new mapping functions
// This mapping might need to go through rfactor, unclear
std::vector<bool> IndexCompute::contiguityPasC(
    TensorDomain* producer,
    TensorDomain* consumer) {
  FUSER_PERF_SCOPE("contiguityPasC");

  const std::vector<bool>& producer_contiguity = producer->contiguity();
  std::vector<bool> as_consumer_contiguity;

  auto c_root = consumer->getRootDomain();
  auto p_root = producer->getRootDomain();

  size_t p_ind = 0;
  size_t c_ind = 0;
  while (p_ind < p_root.size()) {
    if (p_root[p_ind]->isReduction()) {
      p_ind++;
    } else if (
        c_root[c_ind]->isBroadcast() &&
        p_root[p_ind]->getIterType() != c_root[c_ind]->getIterType()) {
      c_ind++;
      as_consumer_contiguity.push_back(false);
    } else {
      as_consumer_contiguity.push_back(producer_contiguity[p_ind]);
      c_ind++;
      p_ind++;
    }
  }

  while (c_ind < c_root.size()) {
    as_consumer_contiguity.push_back(false);
    c_ind++;
  }

  return as_consumer_contiguity;
}

namespace {

std::deque<TensorView*> getComputeAtTVStackFrom(TensorView* from_tv) {
  // What's the computeAt root tensor view in this operation
  // This tensor is the terminating tensor in the computeAT dag from consumer
  auto end_tv = from_tv->getComputeAtAxis(0).second;

  // grab all tensor views from producer_tv -> computeAtRoot
  std::deque<TensorView*> tv_stack;

  // Then immediate consumer
  auto running_tv = from_tv;

  // Follow computeAt path until we hit end_tv
  while (running_tv != end_tv) {
    TORCH_INTERNAL_ASSERT(running_tv->hasComputeAt());
    tv_stack.push_front(running_tv);
    running_tv = running_tv->getComputeAtView();
  }

  tv_stack.push_front(end_tv);

  return tv_stack;
}

std::pair<
    std::unordered_map<kir::IterDomain*, Val*>,
    std::unordered_map<kir::IterDomain*, Val*>>
generateIndexAndExtentMap(
    std::deque<TensorView*> c2p_tv_stack,
    std::deque<kir::ForLoop*> loops,
    const std::unordered_map<kir::ForLoop*, Val*>& loop_to_ind_map,
    const std::vector<bool>& last_tv_root_contiguity) {
  if (c2p_tv_stack.empty())
    return std::make_pair(
        std::unordered_map<kir::IterDomain*, Val*>(),
        std::unordered_map<kir::IterDomain*, Val*>());

  // Go through our stack, and map the intermediate IterDomains from common
  // transformations from consumer to producer
  std::deque<std::unordered_map<IterDomain*, IterDomain*>> c2p_ID_maps;
  std::deque<std::unordered_map<IterDomain*, IterDomain*>> p2c_ID_maps;

  // c2p_tv_stack comes in as consumer -> producer
  // Realized we may want to actually do a pass from producer->consumer first to
  // propagate iterators outside the compute at position back into consumers, so
  // we can repropagate back to producer. The need for this was exposed in
  // https://github.com/csarofeen/pytorch/issues/286

  for (size_t i = 0; i + 1 < c2p_tv_stack.size(); i++) {
    auto c_tv = c2p_tv_stack[i];
    auto p_tv = c2p_tv_stack[i + 1];

    // Map root ID's from consumer to producer
    auto c2p_root_map =
        TensorDomain::mapRootCtoP(c_tv->domain(), p_tv->domain());

    // Look for matching ID transformations in producer and consumer...
    BestEffortReplay replay(
        p_tv->domain()->domain(), c_tv->domain()->domain(), c2p_root_map);

    // and grab the intermediate IterDomain map.
    c2p_ID_maps.push_back(replay.getReplay());

    // Something wasn't symmetric when using:
    //
    // auto p2c_root_map = TensorDomain::mapRootPtoC(p_tv->domain(),
    // c_tv->domain());
    //
    // replay = BestEffortReplay(
    //     c_tv->domain()->domain(), p_tv->domain()->domain(), p2c_root_map,
    //     true);

    BestEffortReplay replay_p2c(
        p_tv->domain()->domain(), c_tv->domain()->domain(), c2p_root_map, true);

    std::unordered_map<IterDomain*, IterDomain*> p2c_id_map;

    for (auto ent : replay_p2c.getReplay()) {
      p2c_id_map[ent.second] = ent.first;
    }

    // and grab the intermediate IterDomain map.
    p2c_ID_maps.push_front(p2c_id_map);
  }

  // Maps to be used in the c2p propagation
  std::unordered_map<TensorView*, std::unordered_map<kir::IterDomain*, Val*>>
      p2c_index_maps;

  // PROPAGATE PRODUCER -> CONSUMER START

  std::deque<TensorView*> p2c_tv_stack(
      c2p_tv_stack.rbegin(), c2p_tv_stack.rend());

  // Setup initial IndexCompute:
  auto tv = p2c_tv_stack.front();
  p2c_tv_stack.pop_front();
  auto td = tv->domain()->domain();

  std::vector<kir::IterDomain*> kir_td;

  std::transform(
      td.begin(), td.end(), std::back_inserter(kir_td), [](IterDomain* id) {
        return GpuLower::lowerValue(id)->as<kir::IterDomain>();
      });

  // Map from all IterDomain's to corresponding index as we process each tv in
  // the stack
  std::unordered_map<kir::IterDomain*, Val*> initial_index_map;

  // Match loops to this TV if the loop matchis this TV's ID (could reduce
  // complexity here)

  while (
      !loops.empty() &&
      std::find(kir_td.rbegin(), kir_td.rend(), loops.back()->iter_domain()) !=
          kir_td.rend()) {
    TORCH_INTERNAL_ASSERT(
        loop_to_ind_map.find(loops.back()) != loop_to_ind_map.end());
    initial_index_map[loops.back()->iter_domain()] =
        loop_to_ind_map.at(loops.back());
    loops.pop_back();
  }

  IndexCompute index_compute(
      tv->domain(),
      initial_index_map,
      std::unordered_map<kir::IterDomain*, Val*>(),
      std::unordered_set<kir::IterDomain*>(),
      std::vector<bool>(tv->getRootDomain().size(), false));

  p2c_index_maps[tv] = index_compute.indexMap();

  // Go through the tv entire stack
  while (!p2c_tv_stack.empty()) {
    // Grab the TV
    tv = p2c_tv_stack.front();
    p2c_tv_stack.pop_front();
    td = tv->domain()->domain();
    kir_td.clear();
    std::transform(
        td.begin(), td.end(), std::back_inserter(kir_td), [](IterDomain* id) {
          return GpuLower::lowerValue(id)->as<kir::IterDomain>();
        });

    // Match loops to this TV if the loop matchis this TV's ID (could reduce
    // complexity here)

    // Map from all IterDomain's to corresponding index as we process each tv in
    // the stack
    std::unordered_map<kir::IterDomain*, Val*> new_indices;

    while (!loops.empty() &&
           std::find(
               kir_td.rbegin(), kir_td.rend(), loops.back()->iter_domain()) !=
               kir_td.rend()) {
      TORCH_INTERNAL_ASSERT(
          loop_to_ind_map.find(loops.back()) != loop_to_ind_map.end());
      new_indices[loops.back()->iter_domain()] =
          loop_to_ind_map.at(loops.back());
      loops.pop_back();
    }

    if (!p2c_ID_maps.empty()) {
      index_compute = index_compute.updateIndexCompute(
          tv->domain(),
          p2c_ID_maps.front(),
          new_indices,
          std::vector<bool>(tv->getRootDomain().size(), false));

      p2c_index_maps[tv] = index_compute.indexMap();

      p2c_ID_maps.pop_front();
    }
  }

  // PROPAGATE PRODUCER -> CONSUMER END

  // PROPAGATE CONSUMER -> PRODUCER START

  // Setup initial IndexCompute:
  tv = c2p_tv_stack.front();
  c2p_tv_stack.pop_front();

  // Map from all IterDomain's to corresponding index as we process each tv in
  // the stack
  initial_index_map = p2c_index_maps.at(tv);

  std::unordered_map<kir::IterDomain*, Val*> initial_extent_map;
  if (!c2p_ID_maps.empty()) {
    auto first_id_map = c2p_ID_maps.front();
    for (auto id_entry : first_id_map) {
      kir::IterDomain* this_id =
          GpuLower::lowerValue(id_entry.first)->as<kir::IterDomain>();
      if (initial_extent_map.find(this_id) == initial_extent_map.end()) {
        initial_extent_map[this_id] = this_id->extent();
      }
    }
  }

  index_compute = IndexCompute(
      tv->domain(),
      initial_index_map,
      initial_extent_map,
      std::unordered_set<kir::IterDomain*>(),
      c2p_tv_stack.empty()
          ? last_tv_root_contiguity
          : std::vector<bool>(tv->getRootDomain().size(), false));

  // Go through the tv entire stack
  while (!c2p_tv_stack.empty()) {
    // Grab the TV
    tv = c2p_tv_stack.front();
    c2p_tv_stack.pop_front();

    if (!c2p_ID_maps.empty()) {
      index_compute = index_compute.updateIndexCompute(
          tv->domain(),
          c2p_ID_maps.front(),
          p2c_index_maps.at(tv),
          c2p_tv_stack.empty()
              ? last_tv_root_contiguity
              : std::vector<bool>(tv->getRootDomain().size(), false));

      c2p_ID_maps.pop_front();
    }
  }

  // PROPAGATE CONSUMER -> PRODUCER END

  // Fill in extent map as some mapped indices may not have their extent filled
  // in it, but consumers of this function expect it to be there

  std::unordered_map<kir::IterDomain*, Val*> extent_map(
      index_compute.extentMap());
  for (auto ind_entry : index_compute.indexMap()) {
    auto id = ind_entry.first;
    if (extent_map.find(id) == extent_map.end()) {
      extent_map[id] = id->extent();
    }
  }

  return std::make_pair(index_compute.indexMap(), extent_map);
}

} // namespace

kir::TensorIndex* Index::getGlobalProducerIndex(
    TensorView* producer_tv,
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops) {
  FUSER_PERF_SCOPE("getGlobalProducerIndex");

  kir::IrBuilder ir_builder(GpuLower::current()->kernel());

  // Replay producer to look like consumer so we can index on producer since our
  // loop nests look like consumer
  auto producerAsC = TransformReplay::replayPasC(
                         producer_tv->domain(), consumer_tv->domain(), -1)
                         .first;

  // Make the actual producer_tv look like consumer while we do the indexing
  // math in this function
  ir_utils::TVDomainGuard domain_guard(producer_tv, producerAsC);

  // grab all tensor views from producer_tv <- computeAtRoot
  std::deque<TensorView*> tv_stack = getComputeAtTVStackFrom(consumer_tv);
  tv_stack.push_back(producer_tv);

  std::unordered_map<kir::ForLoop*, Val*> loop_to_ind_map;
  std::transform(
      loops.begin(),
      loops.end(),
      std::inserter(loop_to_ind_map, loop_to_ind_map.begin()),
      [](kir::ForLoop* fl) { return std::make_pair(fl, fl->index()); });

  auto index_map = generateIndexAndExtentMap(
                       tv_stack,
                       std::deque<kir::ForLoop*>(loops.begin(), loops.end()),
                       loop_to_ind_map,
                       producer_tv->domain()->contiguity())
                       .first;

  // Indices should now be mapped onto IterDomains in producer, so just grab
  // and use them.
  auto root_dom = producer_tv->getMaybeRFactorDomain();

  bool inner_most_dim_contig =
      root_dom[root_dom.size() - 1]->getIterType() == IterType::Iteration &&
      producer_tv->domain()->contiguity()[root_dom.size() - 1];

  // Global striding
  int64_t stride_i = 0;
  std::vector<Val*> strided_inds;
  for (size_t i = 0; i < root_dom.size(); i++) {
    if (root_dom[i]->isReduction() ||
        root_dom[i]->getIterType() == IterType::BroadcastWithoutStride) {
      continue;
    } else if (root_dom[i]->getIterType() == IterType::BroadcastWithStride) {
      stride_i++;
      continue;
    }

    auto kir_root_dom_i =
        GpuLower::lowerValue(root_dom[i])->as<kir::IterDomain>();

    TORCH_INTERNAL_ASSERT(
        index_map.find(kir_root_dom_i) != index_map.end(),
        "Couldn't find root mapping for TV",
        producer_tv->name(),
        " dim: ",
        i,
        " id: ",
        kir::toString(kir_root_dom_i));

    auto root_ind = index_map.at(kir_root_dom_i);
    TORCH_INTERNAL_ASSERT(kir::isLoweredScalar(root_ind));

    if (i == root_dom.size() - 1 && inner_most_dim_contig) {
      strided_inds.push_back(root_ind);
    } else if (root_ind->isZeroInt()) {
      stride_i++;
    } else {
      std::stringstream ss;
      ss << "T" << producer_tv->name() << ".stride[" << stride_i++ << "]";
      strided_inds.push_back(ir_builder.mulExpr(
          root_ind,
          ir_builder.create<kir::NamedScalar>(ss.str(), DataType::Int)));
    }
  }

  if (strided_inds.size() == 0)
    strided_inds.push_back(ir_builder.create<kir::Int>(0));

  return ir_builder.create<kir::TensorIndex>(producer_tv, strided_inds);
}

namespace {

std::unordered_map<kir::ForLoop*, Val*> indexMapFromTV(
    TensorView* tv,
    const std::vector<kir::ForLoop*>& loops) {
  auto alloc_point = loop_utils::getAllocPoint(tv, loops);
  auto alloc_loop = alloc_point.first;

  bool within_alloc = false;
  if (alloc_loop == nullptr) {
    within_alloc = true;
  }

  kir::IrBuilder ir_builder(GpuLower::current()->kernel());
  Val* zero = ir_builder.create<kir::Int>(0);

  bool is_shared = tv->getMemoryType() == MemoryType::Shared;
  bool is_local = tv->getMemoryType() == MemoryType::Local;

  std::unordered_map<kir::ForLoop*, Val*> loop_to_ind_map;

  for (auto loop : loops) {
    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (!within_alloc) {
      loop_to_ind_map[loop] = zero;
    } else if (loop->iter_domain()->isBlockDim() && is_shared) {
      loop_to_ind_map[loop] = zero;
    } else if (loop->iter_domain()->isThread() && is_local) {
      loop_to_ind_map[loop] = zero;
    } else {
      loop_to_ind_map[loop] = loop->index();
    }

    if (!within_alloc && loop == alloc_loop) {
      within_alloc = true;
    }
  }
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  return loop_to_ind_map;
}

} // namespace

// Producer index for either shared or local memory
kir::TensorIndex* Index::getProducerIndex_impl(
    TensorView* producer_tv,
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops) {
  kir::IrBuilder ir_builder(GpuLower::current()->kernel());

  // producer_tv->domain() is not replayed as the loop strucutre we were
  // provided, so replay it to match consumer_tv which is.
  auto producerAsC = TransformReplay::replayPasC(
                         producer_tv->domain(), consumer_tv->domain(), -1)
                         .first;

  // Set producer_tv with the domain replayed as consumer to grab the right
  // indices. The guard will reset the domain when this scope ends.
  ir_utils::TVDomainGuard domain_guard(producer_tv, producerAsC);

  // grab all tensor views from producer_tv <- computeAtRoot
  std::deque<TensorView*> tv_stack = getComputeAtTVStackFrom(consumer_tv);
  tv_stack.push_back(producer_tv);

  std::unordered_map<kir::ForLoop*, Val*> loop_to_ind_map =
      indexMapFromTV(producer_tv, loops);

  auto index_and_extent_map = generateIndexAndExtentMap(
      tv_stack,
      std::deque<kir::ForLoop*>(loops.begin(), loops.end()),
      loop_to_ind_map,
      std::vector<bool>(producer_tv->getRootDomain().size(), false));
  auto index_map = index_and_extent_map.first;
  auto extent_map = index_and_extent_map.second;

  // Indices should now be mapped onto IterDomains in producer, so just grab
  // and use them.
  auto root_dom = producer_tv->getMaybeRFactorDomain();

  std::vector<Val*> strided_inds;

  for (size_t i = 0; i < root_dom.size(); i++) {
    if (root_dom[i]->isReduction() || root_dom[i]->isBroadcast()) {
      continue;
    }

    auto kir_root_dom_i =
        GpuLower::lowerValue(root_dom[i])->as<kir::IterDomain>();

    TORCH_INTERNAL_ASSERT(
        index_map.find(kir_root_dom_i) != index_map.end(),
        "Couldn't find root mapping for TV",
        producer_tv->name(),
        " dim: ",
        i,
        " id: ",
        kir::toString(kir_root_dom_i));

    auto root_ind_i = index_map.at(kir_root_dom_i);
    TORCH_INTERNAL_ASSERT(kir::isLoweredScalar(root_ind_i));

    if (root_ind_i->isZeroInt()) {
      continue;
    }

    // Compute striding for this index.
    Val* stride = nullptr;
    for (size_t j = i + 1; j < root_dom.size(); j++) {
      if (root_dom[j]->isBroadcast() || root_dom[j]->isReduction()) {
        continue;
      }

      auto kir_root_dom_j =
          GpuLower::lowerValue(root_dom[j])->as<kir::IterDomain>();

      TORCH_INTERNAL_ASSERT(
          index_map.find(kir_root_dom_j) != index_map.end() &&
              extent_map.find(kir_root_dom_j) != extent_map.end(),
          "Couldn't find root mapping for TV",
          consumer_tv->name(),
          " dim: ",
          i,
          " id: ",
          root_dom[i]);

      auto root_ind_j = index_map.at(kir_root_dom_j);
      auto root_ext_j = extent_map.at(kir_root_dom_j);

      TORCH_INTERNAL_ASSERT(kir::isLoweredScalar(root_ext_j));

      if (!root_ind_j->isZeroInt()) {
        if (stride == nullptr) {
          stride = root_ext_j;
        } else {
          stride = ir_builder.mulExpr(stride, root_ext_j);
        }
      }
    }

    if (stride != nullptr) {
      strided_inds.push_back(ir_builder.mulExpr(root_ind_i, stride));
    } else {
      strided_inds.push_back(root_ind_i);
    }
  }

  if (strided_inds.size() == 0)
    strided_inds.push_back(ir_builder.create<kir::Int>(0));

  return ir_builder.create<kir::TensorIndex>(producer_tv, strided_inds);
}

kir::TensorIndex* Index::getGlobalConsumerIndex(
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops) {
  FUSER_PERF_SCOPE("getGlobalConsumerIndex");

  kir::IrBuilder ir_builder(GpuLower::current()->kernel());

  // grab all tensor views from producer_tv <- computeAtRoot
  std::deque<TensorView*> tv_stack = getComputeAtTVStackFrom(consumer_tv);

  std::unordered_map<kir::ForLoop*, Val*> loop_to_ind_map;
  std::transform(
      loops.begin(),
      loops.end(),
      std::inserter(loop_to_ind_map, loop_to_ind_map.begin()),
      [](kir::ForLoop* fl) { return std::make_pair(fl, fl->index()); });

  auto index_map = generateIndexAndExtentMap(
                       tv_stack,
                       std::deque<kir::ForLoop*>(loops.begin(), loops.end()),
                       loop_to_ind_map,
                       consumer_tv->domain()->contiguity())
                       .first;

  // Indices should now be mapped onto IterDomains in consumer, so just grab
  // and use them.
  auto root_dom = consumer_tv->getMaybeRFactorDomain();

  bool inner_most_dim_contig =
      root_dom[root_dom.size() - 1]->getIterType() == IterType::Iteration &&
      consumer_tv->domain()->contiguity()[root_dom.size() - 1];

  int64_t stride_i = 0;
  std::vector<Val*> strided_inds;
  for (size_t i = 0; i < root_dom.size(); i++) {
    if (root_dom[i]->isReduction() ||
        root_dom[i]->getIterType() == IterType::BroadcastWithoutStride) {
      continue;
    } else if (root_dom[i]->getIterType() == IterType::BroadcastWithStride) {
      stride_i++;
      continue;
    }

    auto kir_root_dom_i =
        GpuLower::lowerValue(root_dom[i])->as<kir::IterDomain>();

    TORCH_INTERNAL_ASSERT(
        index_map.find(kir_root_dom_i) != index_map.end(),
        "Couldn't find root mapping for TV",
        consumer_tv->name(),
        " dim: ",
        i,
        " id: ",
        kir::toString(kir_root_dom_i));
    auto ind = index_map.at(kir_root_dom_i);

    if (i == root_dom.size() - 1 && inner_most_dim_contig) {
      strided_inds.push_back(ind);
    } else if (ind->isZeroInt()) {
      stride_i++;
    } else {
      std::stringstream ss;
      ss << "T" << consumer_tv->name() << ".stride[" << stride_i++ << "]";
      strided_inds.push_back(ir_builder.mulExpr(
          ind, ir_builder.create<kir::NamedScalar>(ss.str(), DataType::Int)));
    }
  }

  if (strided_inds.size() == 0)
    strided_inds.push_back(ir_builder.create<kir::Int>(0));

  return ir_builder.create<kir::TensorIndex>(consumer_tv, strided_inds);
}

// Consumer index for either shared or local memory
kir::TensorIndex* Index::getConsumerIndex_impl(
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops) {
  kir::IrBuilder ir_builder(GpuLower::current()->kernel());

  // grab all tensor views from consumer_tv <- computeAtRoot
  std::deque<TensorView*> tv_stack = getComputeAtTVStackFrom(consumer_tv);

  std::unordered_map<kir::ForLoop*, Val*> loop_to_ind_map =
      indexMapFromTV(consumer_tv, loops);

  auto index_and_extent_map = generateIndexAndExtentMap(
      tv_stack,
      std::deque<kir::ForLoop*>(loops.begin(), loops.end()),
      loop_to_ind_map,
      std::vector<bool>(consumer_tv->getRootDomain().size(), false));

  auto index_map = index_and_extent_map.first;
  auto extent_map = index_and_extent_map.second;

  // Indices should now be mapped onto IterDomains in consumer, so just grab
  // and use them.
  auto root_dom = consumer_tv->getMaybeRFactorDomain();

  std::vector<Val*> strided_inds;
  for (size_t i = 0; i < root_dom.size(); i++) {
    if (root_dom[i]->isReduction() || root_dom[i]->isBroadcast()) {
      continue;
    }

    auto kir_root_dom_i =
        GpuLower::lowerValue(root_dom[i])->as<kir::IterDomain>();

    TORCH_INTERNAL_ASSERT(
        index_map.find(kir_root_dom_i) != index_map.end(),
        "Couldn't find root mapping for TV",
        consumer_tv->name(),
        " dim: ",
        i,
        " id: ",
        kir::toString(kir_root_dom_i));
    auto root_ind_i = index_map.at(kir_root_dom_i);
    TORCH_INTERNAL_ASSERT(kir::isLoweredScalar(root_ind_i));

    if (root_ind_i->isZeroInt()) {
      continue;
    }

    // Compute striding for this index.
    Val* stride = nullptr;
    for (size_t j = i + 1; j < root_dom.size(); j++) {
      if (root_dom[j]->isBroadcast() || root_dom[j]->isReduction()) {
        continue;
      }

      auto kir_root_dom_j =
          GpuLower::lowerValue(root_dom[j])->as<kir::IterDomain>();

      TORCH_INTERNAL_ASSERT(
          index_map.find(kir_root_dom_j) != index_map.end() &&
              extent_map.find(kir_root_dom_j) != extent_map.end(),
          "Couldn't find root mapping for TV",
          consumer_tv->name(),
          " dim: ",
          i,
          " id: ",
          root_dom[i]);

      auto root_ind_j = index_map.at(kir_root_dom_j);
      auto root_ext_j = extent_map.at(kir_root_dom_j);
      TORCH_INTERNAL_ASSERT(kir::isLoweredScalar(root_ext_j));
      if (!root_ind_j->isZeroInt()) {
        if (stride == nullptr) {
          stride = root_ext_j;
        } else {
          stride = ir_builder.mulExpr(stride, root_ext_j);
        }
      }
    }

    if (stride != nullptr) {
      strided_inds.push_back(ir_builder.mulExpr(root_ind_i, stride));
    } else {
      strided_inds.push_back(root_ind_i);
    }
  }

  if (strided_inds.size() == 0)
    strided_inds.push_back(ir_builder.create<kir::Int>(0));

  return ir_builder.create<kir::TensorIndex>(consumer_tv, strided_inds);
}

// Producer is the inputs of an expression
kir::TensorIndex* Index::getProducerIndex(
    TensorView* producer,
    TensorView* consumer,
    const std::vector<kir::ForLoop*>& loops) {
  FUSER_PERF_SCOPE("Index::getProducerIndex");

  kir::IrBuilder ir_builder(GpuLower::current()->kernel());

  if (producer->domain()->noReductions().size() == 0) {
    return ir_builder.create<kir::TensorIndex>(producer, std::vector<Val*>{});
  }

  if (producer->getMemoryType() == MemoryType::Global) {
    return getGlobalProducerIndex(producer, consumer, loops);
  }

  return getProducerIndex_impl(producer, consumer, loops);
}

// Consumer is the output of an expression
kir::TensorIndex* Index::getConsumerIndex(
    TensorView* consumer,
    const std::vector<kir::ForLoop*>& loops) {
  FUSER_PERF_SCOPE("Index::getConsumerIndex");

  kir::IrBuilder ir_builder(GpuLower::current()->kernel());

  if (consumer->domain()->noReductions().size() == 0) {
    return ir_builder.create<kir::TensorIndex>(consumer, std::vector<Val*>{});
  }

  if (consumer->getMemoryType() == MemoryType::Global) {
    return getGlobalConsumerIndex(consumer, loops);
  }

  return getConsumerIndex_impl(consumer, loops);
}

// Basically just copy getGlobalConsumerIndex, just don't do the striding and
// return std::vector of Vals
std::pair<std::vector<Val*>, bool> Index::getConsumerRootPredIndices(
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::vector<bool>& root_contiguity,
    bool unroll) {
  FUSER_PERF_SCOPE("Index::getConsumerRootPredIndices");

  kir::IrBuilder ir_builder(GpuLower::current()->kernel());

  // grab all tensor views from producer_tv <- computeAtRoot
  std::deque<TensorView*> tv_stack = getComputeAtTVStackFrom(consumer_tv);

  std::unordered_map<kir::ForLoop*, Val*> loop_to_ind_map;

  std::transform(
      loops.begin(),
      loops.end(),
      std::inserter(loop_to_ind_map, loop_to_ind_map.begin()),
      [](kir::ForLoop* fl) { return std::make_pair(fl, fl->index()); });

  if (unroll) {
    bool within_unroll = false;
    Val* one = ir_builder.create<kir::Int>(1);
    for (auto loop : loops) {
      if (loop->iter_domain()->getParallelType() == ParallelType::Unroll) {
        within_unroll = true;
      }

      if (within_unroll && !loop->iter_domain()->isThread()) {
        loop_to_ind_map[loop] =
            ir_builder.subExpr(loop->iter_domain()->extent(), one);
      }
    }
  }

  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  auto index_map = generateIndexAndExtentMap(
                       tv_stack,
                       std::deque<kir::ForLoop*>(loops.begin(), loops.end()),
                       loop_to_ind_map,
                       root_contiguity)
                       .first;

  // Indices should now be mapped onto IterDomains in consumer, so just grab
  // and use them.

  // If we are generating a predicate for initialization check if we should use
  // rfactor instead of root_dom
  bool use_rfactor = true;
  if (consumer_tv->hasRFactor()) {
    auto rfactor_dom = consumer_tv->getMaybeRFactorDomain();
    for (auto rfactor_id : rfactor_dom) {
      if (rfactor_id->isReduction()) {
        auto kir_rfactor_id =
            GpuLower::lowerValue(rfactor_id)->as<kir::IterDomain>();
        if (index_map.find(kir_rfactor_id) != index_map.end()) {
          if (!index_map.at(kir_rfactor_id)->isZeroInt()) {
            use_rfactor = false;
            break;
          }
        }
      }
    }
  }

  auto root_dom = use_rfactor ? consumer_tv->getMaybeRFactorDomain()
                              : consumer_tv->getRootDomain();

  std::vector<Val*> root_inds(root_dom.size(), ir_builder.create<kir::Int>(0));
  for (size_t i = 0; i < root_dom.size(); i++) {
    if (root_dom[i]->isBroadcast()) {
      continue;
    }

    auto kir_root_dom_i =
        GpuLower::lowerValue(root_dom[i])->as<kir::IterDomain>();
    if (index_map.find(kir_root_dom_i) != index_map.end()) {
      auto ind = index_map.at(kir_root_dom_i);
      TORCH_INTERNAL_ASSERT(kir::isLoweredScalar(ind))
      root_inds[i] = ind;
    }
  }

  return std::make_pair(root_inds, use_rfactor);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
