#include <torch/csrc/jit/codegen/cuda/index_reference_replay.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// We're going to replay this split operation on the corresponding ID
void IndexReferenceReplay::handle(Split* s) {
  auto in = s->in();

  auto concrete_in = GpuLower::current()->caIndexMap().getConcreteMappedID(in);
  auto mapped_in_it = concrete_to_id_.find(concrete_in);
  if (mapped_in_it == concrete_to_id_.end()) {
    // If we can't find the concrete IDs in our local map, don't do anything.
    return;
  }

  auto mapped_in = mapped_in_it->second;

  if (leaf_ids_.find(mapped_in) == leaf_ids_.end()) {
    // If ID has already been replayed, don't do anything.
    return;
  }

  auto replayed_outs =
      IterDomain::split(mapped_in, s->factor(), s->innerSplit());

  auto concrete_outer =
      GpuLower::current()->caIndexMap().getConcreteMappedID(s->outer());
  auto concrete_inner =
      GpuLower::current()->caIndexMap().getConcreteMappedID(s->inner());

  // Update leaf id set and concrete id map
  leaf_ids_.erase(mapped_in);
  leaf_ids_.emplace(replayed_outs.first);
  leaf_ids_.emplace(replayed_outs.second);
  concrete_to_id_[concrete_outer] = replayed_outs.first;
  concrete_to_id_[concrete_inner] = replayed_outs.second;
}

// We're going to replay this merge operation on the corresponding IDs
void IndexReferenceReplay::handle(Merge* m) {
  auto in_outer = m->outer();
  auto in_inner = m->inner();

  auto concrete_in_outer =
      GpuLower::current()->caIndexMap().getConcreteMappedID(in_outer);
  auto concrete_in_inner =
      GpuLower::current()->caIndexMap().getConcreteMappedID(in_inner);

  auto mapped_in_outer_it = concrete_to_id_.find(concrete_in_outer);
  auto mapped_in_inner_it = concrete_to_id_.find(concrete_in_inner);

  if (mapped_in_outer_it == concrete_to_id_.end() ||
      mapped_in_inner_it == concrete_to_id_.end()) {
    // If we can't find the concrete IDs in our local map, don't do anything.
    return;
  }

  auto mapped_in_outer = mapped_in_outer_it->second;
  auto mapped_in_inner = mapped_in_inner_it->second;

  if (leaf_ids_.find(mapped_in_outer) == leaf_ids_.end() &&
      leaf_ids_.find(mapped_in_inner) == leaf_ids_.end()) {
    // If ID has already been replayed, don't do anything.
    return;
  }
  auto replayed = IterDomain::merge(mapped_in_outer, mapped_in_inner);

  auto concrete_out =
      GpuLower::current()->caIndexMap().getConcreteMappedID(m->out());

  // Update leaf id set and concrete id map
  leaf_ids_.erase(mapped_in_outer);
  leaf_ids_.erase(mapped_in_inner);
  leaf_ids_.emplace(replayed);
  concrete_to_id_[concrete_out] = replayed;
}

TensorDomain* IndexReferenceReplay::computeReplay() {
  auto gpu_lower = GpuLower::current();
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
          !gpu_lower->caIndexMap().areMapped(
              (*it_i)->iter_domain(), (*it_j)->iter_domain()),
          "Unsupported loop structure. Two loops are mapped together.");
    }
  }

  // Grab the iter domain's from the loop structure
  std::vector<IterDomain*> fusion_loop_structure;

  std::transform(
      loop_structure_.begin(),
      loop_structure_.end(),
      std::back_inserter(fusion_loop_structure),
      [&](kir::ForLoop* fl) {
        auto fid = gpu_lower->caIndexMap().toFusion(fl->iter_domain());
        return fid;
      });

  // Get any and all inputs that generated the provided loop structure, some
  // root inputs may be mapped to eachother but not identical
  auto all_inputs = InputsOf::outputs(
      FusionGuard::getCurFusion(),
      std::vector<Val*>(
          fusion_loop_structure.begin(), fusion_loop_structure.end()));

  // Make sure all inputs are iter domains, ignoring anything like split factor
  // inputs
  auto all_iter_inputs = ir_utils::filterByType<IterDomain>(all_inputs);

  // Sort out the inputs as there could be entires that map to eachother, and
  // they can be a combiantion of iteration, reduction, and broadcast. Order as
  // iter, reduction, then broadcast for iterating and removing duplicate mapped
  // entries. Since these are input IterDomains we mainly want to prioritize
  // non-broadcast "versions" of the iter domain if it shows up more than once.
  // We could get both if we have a compute at structure where a consumer has a
  // concrete iter domain but it's producer has a broadcast domain, and the
  // compute at axis is across a split on this domain. The producer would give a
  // broadcast input, consumer would have iter domain input.
  // Additionally, we prefer non-reduction iter domains over reduciton
  // domains, but this is just optional and not necessary for correctness.
  std::vector<IterDomain*> sorted_inputs;
  std::copy_if(
      all_iter_inputs.begin(),
      all_iter_inputs.end(),
      std::back_inserter(sorted_inputs),
      [](IterDomain* id) { return !id->isBroadcast() && !id->isReduction(); });
  std::copy_if(
      all_iter_inputs.begin(),
      all_iter_inputs.end(),
      std::back_inserter(sorted_inputs),
      [](IterDomain* id) { return id->isReduction(); });
  std::copy_if(
      all_iter_inputs.begin(),
      all_iter_inputs.end(),
      std::back_inserter(sorted_inputs),
      [](IterDomain* id) { return id->isBroadcast(); });

  // Produce a non repetitive set of inputs. Remove "duplicate" IterDomains that
  // map to eachother.
  std::vector<IterDomain*> root_axes;
  for (auto root_id : sorted_inputs) {
    auto concrete_id = gpu_lower->caIndexMap().getConcreteMappedID(root_id);
    if (concrete_to_id_.find(concrete_id) != concrete_to_id_.end()) {
      continue;
    }

    // Make a copy of the root_id for the reference to "own"
    IterDomain* root_id_copy = root_id->clone();

    // Initialize root axes, concrete map, and leaf map for replay.
    root_axes.push_back(root_id_copy);
    concrete_to_id_[concrete_id] = root_id_copy;
    leaf_ids_.emplace(root_id_copy);
  }

  // Order is important here, replay expressions from loops outside to inside.
  auto replay_exprs = ExprSort::getExprs(
      FusionGuard::getCurFusion(),
      {fusion_loop_structure.begin(), fusion_loop_structure.end()});

  // Run the reference replay
  for (auto expr : replay_exprs) {
    OptInDispatch::handle(expr);
  }

  // Construct a tensor that's representitive of the replayed loop structure.
  std::vector<IterDomain*> loops_replayed_domain;

  // Grab a set of concrete leaf ids to make it easier to search which for loop
  // matches the leaf id from the replay.
  std::unordered_set<IterDomain*> concrete_leaf_ids;
  for (auto entry : concrete_to_id_) {
    if (leaf_ids_.find(entry.second) != leaf_ids_.end()) {
      concrete_leaf_ids.emplace(entry.first);
    }
  }

  // Figure out which ID's that were replayed correspond to the respective loops
  // that were replayed.
  std::transform(
      fusion_loop_structure.begin(),
      fusion_loop_structure.end(),
      std::back_inserter(loops_replayed_domain),
      [&](IterDomain* loop_id) {
        for (auto id : concrete_leaf_ids) {
          // Matching has to be done on loop map, though replay was done in ID
          // map, so we need to manually check that things are mapped in the
          // loop map. Cannot simply look up concrete IDs to match them as index
          // map and loop map do not have the same concrete id mapping. We also
          // allow matching explicitly through the index map. Index map is not
          // gauranteed to be contained in loop map, therefore if we generate
          // mappings to conrete id's through the index map, the mapping from
          // those ID's to the ID's we replay are not gauranteed to be in loop
          // map. The reverse is also true, so for validation make sure one of
          // the mappings exist. For reference check the difference between:
          // AdvancedLowering5 test and AdvancedIndexing1.
          if (gpu_lower->caLoopMap().areMapped(id, loop_id) ||
              gpu_lower->caIndexMap().areMapped(id, loop_id)) {
            concrete_leaf_ids.erase(id);
            auto replayed_id = concrete_to_id_.at(id);
            // Propagate parallelization and vectorization. Necessary
            // for indexing. IndexCompute::getExtent depends on the
            // propagated parallelization.
            if (isParallelTypeVectorize(loop_id->getParallelType()) ||
                isParallelTypeThread(loop_id->getParallelType())) {
              replayed_id->parallelize(loop_id->getParallelType());
            }
            return replayed_id;
          }
        }

        TORCH_INTERNAL_ASSERT(
            false,
            "Could not find required iter domain in reference replay: ",
            loop_id);
      });

  // Add any remaining leaf iter domains, this can happen from rfactor patterns.
  for (auto entry : concrete_leaf_ids) {
    loops_replayed_domain.push_back(concrete_to_id_.at(entry));
  }
  if (replay_exprs.empty()) {
    auto domain = new TensorDomain(
        // If there was no replay only return a domain with a root domain.
        loops_replayed_domain);
    return domain;
  } else {
    auto domain = new TensorDomain(root_axes, loops_replayed_domain);
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
  //       {gpu_lower->caIndexMap().toFusion(loop->iter_domain())});

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
