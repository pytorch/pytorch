#include <torch/csrc/jit/codegen/cuda/index_reference_replay.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>

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
  auto copied_id =
      new IterDomain(id->start(), id->extent(), id->getParallelType());
  replayed_ids_.emplace_back(copied_id);
  return copied_id;
}

IterDomain* IndexReferenceReplay::toFusionID(kir::IterDomain* kir_id) {
  return ca_map_.toFusion(kir_id);
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
  if (ref_id_produced_.find(ref_outer) != ref_id_consumed_.end() ||
      ref_id_produced_.find(ref_inner) != ref_id_consumed_.end()) {
    return;
  }

  // Replay the provided split operation and add it to the reference DAG
  new Split(ref_outer, ref_inner, ref_in, split->factor(), split->innerSplit());

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
  if (ref_id_produced_.find(ref_out) != ref_id_consumed_.end()) {
    return;
  }

  // Replay the provided merge operation and add it to the reference DAG
  new Merge(ref_out, ref_outer, ref_inner);

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
      [this](kir::ForLoop* fl) { return toFusionID(fl->iter_domain()); });

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
    auto loop_id = toFusionID(loop->iter_domain());
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
    auto domain = new TensorDomain(
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
    auto domain = new TensorDomain(
        {root_domain_ids.begin(), root_domain_ids.end()},
        loops_replayed_domain);
    return domain;
  }
}

IndexCompute getReferenceIndexing(
    const std::vector<kir::ForLoop*>& loop_structure,
    TensorDomain* reference_tensor) {
  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  // Create a simple index mapping from loop iter domains to their local index.
  // This is only applicable to global memory buffers.
  std::unordered_map<kir::IterDomain*, kir::Val*> initial_index_map;

  TORCH_INTERNAL_ASSERT(loop_structure.size() <= reference_tensor->nDims());
  int magic_zero_loop = -1;
  for (size_t loop_i = 0; loop_i < loop_structure.size(); loop_i++) {
    auto ref_axis = reference_tensor->axis(loop_i);
    auto kir_ref_axis = gpu_lower->lowerValue(ref_axis)->as<kir::IterDomain>();
    auto loop = loop_structure[loop_i];
    auto ind = loop->index();
    ;

    initial_index_map[kir_ref_axis] = ind;
    if (loop->vectorize()) {
      initial_index_map[kir_ref_axis] = ir_builder.create<kir::Int>(0);
    }

    if (Index::protectWithMagicZero(loop, ref_axis, ind)) {
      magic_zero_loop = (int)loop_i;
    }
  }

  // Add magic zero to a fairly inner most index
  if (magic_zero_loop >= 0) {
    auto ref_id = gpu_lower->lowerValue(reference_tensor->axis(magic_zero_loop))
                      ->as<kir::IterDomain>();
    initial_index_map[ref_id] = ir_builder.addExpr(
        initial_index_map[ref_id], ir_builder.magicZeroVal());
  }

  // Send to the other version of reference indexing that directly takes the
  // index map
  return getReferenceIndexing(
      loop_structure, reference_tensor, initial_index_map, {});
}

IndexCompute getReferenceIndexing(
    const std::vector<kir::ForLoop*>& loop_structure,
    TensorDomain* reference_tensor,
    std::unordered_map<kir::IterDomain*, kir::Val*> index_map,
    std::unordered_set<IterDomain*> preferred_paths,
    std::unordered_map<kir::IterDomain*, kir::Val*> halo_extent_map) {
  auto gpu_lower = GpuLower::current();

  // I thought this might be necesasry, but turns out it's not. I think it's
  // because of the root ordering above, however leaving it in case we find
  // out it is necessary in some cases. At the time of commiting, cuda-memcheck
  // passed without this.
  //
  // std::unordered_map<kir::IterDomain*,
  // kir::Val*> reference_extent_map; for (auto loop : loop_structure) {
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

  // Convert to preferred_path to kir::IterDomain for IndexCompute
  std::unordered_set<kir::IterDomain*> kir_preferred_path;
  std::transform(
      preferred_paths.begin(),
      preferred_paths.end(),
      std::inserter(kir_preferred_path, kir_preferred_path.begin()),
      [&gpu_lower](IterDomain* id) {
        return gpu_lower->lowerValue(id)->as<kir::IterDomain>();
      });

  IndexCompute compute(
      reference_tensor,
      index_map, // NOLINT
      // reference_extent_map, // Seems this is not necessary, see comment above
      // in this function
      {},
      std::unordered_set<kir::IterDomain*>(),
      reference_tensor->contiguity(),
      kir_preferred_path,
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
