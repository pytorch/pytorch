#include <torch/csrc/jit/codegen/cuda/index_reference_replay.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

IterDomain* IndexReferenceReplay::concreteToRefId(IterDomain* concrete_id) {
  TORCH_INTERNAL_ASSERT(toConcrete(concrete_id) == concrete_id);
  // If a reference id doesn't exist for the provided concrete id, make a new
  // one and add it to the ref<->concrete maps
  if (concrete_to_ref_id_.find(concrete_id) == concrete_to_ref_id_.end()) {
    auto ref_id = idCopy(concrete_id);
    ref_id_to_concrete_[ref_id] = concrete_id;
    concrete_to_ref_id_[concrete_id] = ref_id;
    return ref_id;
  }
  return concrete_to_ref_id_.at(concrete_id);
}

IterDomain* IndexReferenceReplay::refIdToConcrete(IterDomain* ref_id) {
  // Assert the ref id is associated with a concrete id and return it
  TORCH_INTERNAL_ASSERT(
      ref_id_to_concrete_.find(ref_id) != ref_id_to_concrete_.end(),
      "Could not find ",
      ref_id,
      " in reference replay.");
  return ref_id_to_concrete_.at(ref_id);
}

IterDomain* IndexReferenceReplay::idCopy(IterDomain* id) {
  // Make a new copy of the provided id for the reference to "own". Reference
  // iteration domains should always be "iteration" type, not broadcast or
  // reduction. All we care about are the transformations, and trying to make
  // sure we track correctly a replaying with consistent reduction/broadcast
  // domains is challenging and unnecessary.
  auto copied_id = SimplifyingIrBuilder::create<IterDomain>(
      id->container(), id->start(), id->extent(), id->getParallelType());
  replayed_ids_.emplace_back(copied_id);
  return copied_id;
}

IterDomain* IndexReferenceReplay::toConcrete(IterDomain* id) {
  return ca_map_.getConcreteMappedID(id);
}

void IndexReferenceReplay::handle(Split* split) {
  // Don't consume the same values multiple times
  auto ref_in = concreteToRefId(toConcrete(split->in()));
  if (ref_id_consumed_.find(ref_in) != ref_id_consumed_.end()) {
    return;
  }
  // Don't produce the same values multiple times
  auto ref_outer = concreteToRefId(toConcrete(split->outer()));
  auto ref_inner = concreteToRefId(toConcrete(split->inner()));
  if (ref_id_produced_.find(ref_outer) != ref_id_produced_.end() ||
      ref_id_produced_.find(ref_inner) != ref_id_produced_.end()) {
    return;
  }

  // Replay the provided split operation and add it to the reference DAG
  SimplifyingIrBuilder::create<Split>(
      split->container(),
      ref_outer,
      ref_inner,
      ref_in,
      split->factor(),
      split->innerSplit(),
      split->startOffset(),
      split->stopOffset());

  // Mark producers and consumers
  ref_id_consumed_.emplace(ref_in);
  ref_id_produced_.emplace(ref_outer);
  ref_id_produced_.emplace(ref_inner);
}

void IndexReferenceReplay::handle(Merge* merge) {
  // Don't consume the same values multiple times
  auto ref_outer = concreteToRefId(toConcrete(merge->outer()));
  auto ref_inner = concreteToRefId(toConcrete(merge->inner()));
  if (ref_id_consumed_.find(ref_outer) != ref_id_consumed_.end() ||
      ref_id_consumed_.find(ref_inner) != ref_id_consumed_.end()) {
    return;
  }

  // Don't produce the same values multiple times
  auto ref_out = concreteToRefId(toConcrete(merge->out()));
  if (ref_id_produced_.find(ref_out) != ref_id_produced_.end()) {
    return;
  }

  // Replay the provided merge operation and add it to the reference DAG
  SimplifyingIrBuilder::create<Merge>(
      merge->container(), ref_out, ref_outer, ref_inner);

  // Mark producers and consumers
  ref_id_consumed_.emplace(ref_outer);
  ref_id_consumed_.emplace(ref_inner);
  ref_id_produced_.emplace(ref_out);
}

void IndexReferenceReplay::handle(Expr* e) {
  // Simple expression dispatch
  switch (e->getExprType().value()) {
    case (ExprType::Split):
    case (ExprType::Merge):
      break;
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Invalid expr type found in transform traversal.");
  }
  OptInDispatch::handle(e);
}

TensorDomain* IndexReferenceReplay::computeReplay() {
  // Throw an error when two loops are mapped with each other, which
  // violates an assumption that unique mappings between concrete
  // IterDomains and the IterDomains of the loop structure must be
  // established. It should be a reasonable assumption, but fusions
  // like below won't work:
  // tv0 = [I0]
  // tv1 = broadcast(tv0, {true, false});
  // tv2 = broadcast(tv0, {false, true});
  // tv3 = tv1 + tv2
  // Notice that the two axes of each of tv1, tv2 and tv3 are mapped
  // with each other. We believe it is unlikely this limitation
  // becomes a real concern in practice.
  for (auto it_i = loop_structure_.begin(); it_i != loop_structure_.end();
       ++it_i) {
    for (auto it_j = it_i + 1; it_j != loop_structure_.end(); ++it_j) {
      TORCH_INTERNAL_ASSERT(
          !ca_map_.areMapped((*it_i)->iter_domain(), (*it_j)->iter_domain()),
          "Unsupported loop structure. Two loops are mapped together.");
    }
  }

  std::vector<IterDomain*> domain_ids;
  std::transform(
      loop_structure_.begin(),
      loop_structure_.end(),
      std::back_inserter(domain_ids),
      [](kir::ForLoop* fl) { return fl->iter_domain(); });

  // IterVisitor based traversals don't work because we don't have all outputs.
  // backward traversal's traverseFrom(domain_ids) will throw "Invalid backward
  // traversal found. Some output paths were not provided". Therefore manaully
  // do the backward traversal

  // Order is really important here, start with outer most for loops in a depth
  // first manner. The outer most loops are topologically closer to the outputs,
  // so their broadcast dimensions are "more" resolved than those towards the
  // inner most loops.
  std::deque<IterDomain*> to_visit(domain_ids.begin(), domain_ids.end());
  std::unordered_set<Expr*> visited;
  while (!to_visit.empty()) {
    auto out_id = to_visit.front();
    to_visit.pop_front();

    auto expr = out_id->definition();

    // ID's will be copied for the reference as we replay transformations. If
    // there was no transformations on an iteration domain, a copy of the
    // iteration domain for the reference is made here.
    if (expr == nullptr) {
      if (std::find(domain_ids.begin(), domain_ids.end(), out_id) !=
          domain_ids.end()) {
        concreteToRefId(toConcrete(out_id));
      }
      continue;
    }

    if (!visited.emplace(expr).second) {
      continue;
    }

    handle(expr);

    auto inp_ids = ir_utils::filterByType<IterDomain>(expr->inputs());
    // Make sure to put at the begining of the deque to maintain correct
    // ordering.
    to_visit.insert(to_visit.begin(), inp_ids.begin(), inp_ids.end());
  }

  // Construct a tensor that's representitive of the replayed loop structure.
  std::vector<IterDomain*> loops_replayed_domain;
  for (auto loop : loop_structure_) {
    auto loop_id = loop->iter_domain();
    // Map to loops with the loop map, but make sure the replayed id is actually
    // a leaf in the replay.
    auto ref_id_it = std::find_if(
        replayed_ids_.begin(), replayed_ids_.end(), [&](IterDomain* ref_id) {
          return ref_id->uses().empty() &&
              GpuLower::current()->caLoopMap().areMapped(
                  refIdToConcrete(ref_id), loop_id);
        });

    TORCH_INTERNAL_ASSERT(
        ref_id_it != replayed_ids_.end(),
        "Could not find required iter domain in reference replay: ",
        loop_id);

    auto ref_id = *ref_id_it;
    loops_replayed_domain.emplace_back(ref_id);

    // Preserve parallelization
    ref_id->parallelize(loop_id->getParallelType());
  }

  // If no domains were replayed to make the reference, just return the root
  // domain.
  if (std::none_of(
          loops_replayed_domain.begin(),
          loops_replayed_domain.end(),
          [](IterDomain* id) { return id->definition() != nullptr; })) {
    auto domain = SimplifyingIrBuilder::create<TensorDomain>(
        // If there was no replay only return a domain with a root domain.
        loops_replayed_domain);
    return domain;
  } else {
    // Construct the root domain as the inputs of the replayed domain
    auto loops_replayed_domain_vals =
        ir_utils::filterByType<Val>(loops_replayed_domain);
    auto root_domain_vals = IterVisitor::getInputsTo(
        {loops_replayed_domain_vals.begin(), loops_replayed_domain_vals.end()});
    auto root_domain_ids = ir_utils::filterByType<IterDomain>(root_domain_vals);

    auto all_replayed_vals = ir_utils::filterByType<Val>(replayed_ids_);

    // The domain may have dangling iteration domains, i.e. the inner output of
    // a split but not the outer. Find which replayed vals are dependant on the
    // root domains.
    auto all_ids_from_root = DependencyCheck::getAllValsBetween(
        {root_domain_vals.begin(), root_domain_vals.end()},
        {all_replayed_vals.begin(), all_replayed_vals.end()});

    // Fill all dangling outputs as otherwise backwards visitor in index compute
    // will complain for not having all outputs of the traversal.
    for (auto id : ir_utils::filterByType<IterDomain>(all_ids_from_root)) {
      if (id->uses().empty()) {
        if (std::find(
                loops_replayed_domain.begin(),
                loops_replayed_domain.end(),
                id) == loops_replayed_domain.end()) {
          loops_replayed_domain.emplace_back(id);
        }
      }
    }

    // Create and return the reference.
    auto domain = SimplifyingIrBuilder::create<TensorDomain>(
        std::vector<IterDomain*>(
            root_domain_ids.begin(), root_domain_ids.end()),
        loops_replayed_domain);
    return domain;
  }
}

IndexCompute getReferenceIndexing(
    const std::vector<kir::ForLoop*>& loop_structure,
    TensorDomain* reference_tensor,
    kir::ForLoop* double_buffer_loop) {
  // Create a simple index mapping from loop iter domains to their local index.
  // This is only applicable to global memory buffers.
  std::unordered_map<IterDomain*, Val*> initial_index_map;

  TORCH_INTERNAL_ASSERT(loop_structure.size() <= reference_tensor->nDims());
  int magic_zero_loop = -1;
  for (const auto loop_i : c10::irange(loop_structure.size())) {
    auto ref_axis = reference_tensor->axis(loop_i);
    auto loop = loop_structure[loop_i];
    auto ind = loop->index();

    // If the loop is trivial, only the start value is used
    if (loop->isTrivial()) {
      initial_index_map[ref_axis] = loop->start();
    } else {
      initial_index_map[ref_axis] = ind;
    }

    if (double_buffer_loop == loop) {
      TORCH_INTERNAL_ASSERT(
          !loop->isTrivial(), "The double buffer loop must be materialized");
      // This version of getReferenceIndexing is only used for
      // indexing global tensors. When indexing global producers, the
      // index for a double buffered loop needs to be incremented. The
      // parameter double_buffer_loop should be nullptr when indexing
      // global consumers tensors.
      initial_index_map[ref_axis] = SimplifyingIrBuilder::addExpr(
          initial_index_map[ref_axis], GpuLower::current()->kernel()->oneVal());
    }

    if (Index::protectWithMagicZero(loop, ref_axis, ind)) {
      magic_zero_loop = (int)loop_i;
    }
  }

  // Add magic zero to a fairly inner most index
  if (magic_zero_loop >= 0) {
    auto ref_id = reference_tensor->axis(magic_zero_loop);
    initial_index_map[ref_id] = SimplifyingIrBuilder::addExpr(
        initial_index_map[ref_id], FusionGuard::getCurFusion()->magicZeroVal());
  }

  // Send to the other version of reference indexing that directly takes the
  // index map
  return getReferenceIndexing(
      loop_structure, reference_tensor, initial_index_map, {}, {});
}

IndexCompute getReferenceIndexing(
    const std::vector<kir::ForLoop*>& loop_structure,
    TensorDomain* reference_tensor,
    std::unordered_map<IterDomain*, Val*> index_map,
    std::unordered_set<IterDomain*> zero_domains,
    std::unordered_set<IterDomain*> preferred_paths,
    std::unordered_map<IterDomain*, Val*> halo_extent_map) {
  // I thought this might be necesasry, but turns out it's not. I think it's
  // because of the root ordering above, however leaving it in case we find
  // out it is necessary in some cases. At the time of commiting, cuda-memcheck
  // passed without this.
  //
  // std::unordered_map<IterDomain*,
  // Val*> reference_extent_map; for (auto loop : loop_structure) {
  //   // If there's a broadcast merged in the for loop ID we want to track its
  //   // extent
  //   auto inputs = InputsOf::outputs(
  //       FusionGuard::getCurFusion(),
  //       {toFusionID(loop->iter_domain())});

  //   auto iter_inputs = ir_utils::filterByType<IterDomain>(inputs);

  //   // If any of the inputs are a broadcast, explicitly mark the loop id's
  //   // extent
  //   if (std::any_of(iter_inputs.begin(), iter_inputs.end(), [](IterDomain*
  //   id) {
  //         return id->isBroadcast();
  //       })) {
  //     reference_extent_map[loop->iter_domain()] =
  //     loop->iter_domain()->extent();
  //   }
  // }

  IndexCompute compute(
      reference_tensor,
      index_map, // NOLINT
      // reference_extent_map, // Seems this is not necessary, see comment above
      // in this function
      {},
      zero_domains,
      std::unordered_set<IterDomain*>(),
      reference_tensor->contiguity(),
      preferred_paths,
      halo_extent_map);

  compute.run();

  return compute;
}

namespace {

// Class to track through the reference what path to take for zero merged in
// indices if we're indexing shared memory or local memory. Use marked root
// domains and traverse through the replay to mark paths to get to them during a
// backward replay.
class PreferredPathCompute : public IterVisitor {
 private:
  void handle(Expr* e) override {
    // If an input ID is marked, propagate the marking to outputs of the
    // expression
    auto all_iter_inputs = ir_utils::filterByType<IterDomain>(e->inputs());
    if (std::any_of(
            all_iter_inputs.begin(),
            all_iter_inputs.end(),
            [&](IterDomain* inp_id) {
              return this->preferred_path.find(inp_id) !=
                  this->preferred_path.end();
            })) {
      auto all_iter_outputs = ir_utils::filterByType<IterDomain>(e->outputs());
      preferred_path.insert(all_iter_outputs.begin(), all_iter_outputs.end());
    }
  }

 private:
  // If making a choice these are the iter domains to prefer when traversing
  // backward.
  std::unordered_set<IterDomain*> preferred_path;

 public:
  static std::unordered_set<IterDomain*> compute(
      TensorDomain* reference_domain,
      const std::unordered_set<IterDomain*>& preferred_roots) {
    // TODO: assert all provided preferred roots are in the history of reference
    // domain.

    PreferredPathCompute compute;
    // Init preferred path
    compute.preferred_path = preferred_roots;

    // Propagate
    compute.traverseFrom(
        FusionGuard::getCurFusion(),
        std::vector<Val*>(
            reference_domain->domain().begin(),
            reference_domain->domain().end()));

    return compute.preferred_path;
  }
};
} // namespace

// External interface for preferred path propagation.
std::unordered_set<IterDomain*> buildPreferredPaths(
    TensorDomain* reference_tensor,
    const std::unordered_set<IterDomain*>& preferred_roots) {
  return PreferredPathCompute::compute(reference_tensor, preferred_roots);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
