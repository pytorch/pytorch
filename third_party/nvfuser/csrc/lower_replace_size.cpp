#include <instrumentation.h>
#include <ir_builder.h>
#include <ir_iostream.h>
#include <ir_utils.h>
#include <lower_utils.h>
#include <root_domain_map.h>

#include <lower_replace_size.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {
// Going to generate a map of tensor view root domain extents to reduce the
// number used during lowering. For example if we have:
//
// T2[i0, i1] = T1[i0, i1] + T2[i2, i3]
//
// We know it would be safe to use:
//
// T2[i0, i1] = T1[i0, i1] + T2[i0, i1]
//
// And that way we don't generate T2.size[0] and T2.size[1], instead we will
// reuse T1.size[0] and T1.size[1]
// This is important when doing CSE as T2 and T1 would otherwise look like
// they're using different values, even though we know they're the same
//
// There's some duplicate logic here that's in computeAt map, but it's not so
// concice there to pull out. May want to consider making this mapping its own
// class especially as it may be useful during scheduling.
std::unordered_map<Val*, Val*> getSimplificationMap(Fusion* fusion) {
  std::list<std::unordered_set<IterDomain*>> disjoint_root_sets;
  std::unordered_map<IterDomain*, std::unordered_set<IterDomain*>*>
      id_to_disjoint_root_set;

  auto map_root_ids = [&disjoint_root_sets, &id_to_disjoint_root_set](
                          IterDomain* id0, IterDomain* id1) {
    if (id0->isBroadcast() || id1->isBroadcast()) {
      return;
    }

    auto disjoint_set_0_it = id_to_disjoint_root_set.find(id0);
    auto disjoint_set_1_it = id_to_disjoint_root_set.find(id1);
    bool set_0_found = disjoint_set_0_it != id_to_disjoint_root_set.end();
    bool set_1_found = disjoint_set_1_it != id_to_disjoint_root_set.end();

    if (set_0_found && set_1_found) {
      if (disjoint_set_0_it->second == disjoint_set_1_it->second) {
        return;
      }
      // merge second disjoint set into first
      auto* set_0 = disjoint_set_0_it->second;
      auto* set_1 = disjoint_set_1_it->second;
      for (auto id : *set_1) {
        set_0->emplace(id);
        id_to_disjoint_root_set[id] = set_0;
      }
      // remove second set from disjoint_root_sets
      disjoint_root_sets.erase(std::find(
          disjoint_root_sets.begin(), disjoint_root_sets.end(), *set_1));
    } else if (set_0_found || set_1_found) {
      auto existing_set =
          set_0_found ? disjoint_set_0_it->second : disjoint_set_1_it->second;
      auto to_add_id = set_0_found ? id1 : id0;
      existing_set->emplace(to_add_id);
      id_to_disjoint_root_set[to_add_id] = existing_set;
      // add entry into existing set
    } else {
      // create new set entry
      disjoint_root_sets.emplace_back(std::unordered_set<IterDomain*>());
      auto* new_set = &disjoint_root_sets.back();
      new_set->emplace(id0);
      new_set->emplace(id1);
      id_to_disjoint_root_set[id0] = new_set;
      id_to_disjoint_root_set[id1] = new_set;
    }
  };

  auto fusion_vals = fusion->usedMathVals();
  for (auto producer_tv : ir_utils::filterByType<TensorView>(fusion_vals)) {
    auto consumer_tvs = ir_utils::consumerTvsOf(producer_tv);
    for (auto consumer_tv : consumer_tvs) {
      auto pairwise_map = PairwiseRootDomainMap(producer_tv, consumer_tv);
      auto c2p_root_map = pairwise_map.mapConsumerToProducer(
          consumer_tv->domain(), producer_tv->domain());
      for (auto entry : c2p_root_map) {
        auto c_id = entry.first;
        auto p_id = entry.second;
        map_root_ids(p_id, c_id);
      }
    }
  }

  // Map each set to an input ID (if it exists) that has the smallest ->name()
  // entry value
  std::unordered_map<std::unordered_set<IterDomain*>*, IterDomain*>
      set_to_input_id;

  // Loop over the root domains, of the inputs to the fusion. Pick an input ID
  // to use as the representative ID of the collected sets. Only consider inputs
  // as those are the ones that map to values like "T0.size[1]". They are he
  // ID's that propagated their extents into the problem. We could also check
  // the outputs as we do have C++ examples of using output dimensions for the
  // problem size instead of inputs. However, we don't do anything where we can
  // translate to those kinds of kernels integrated into PyTorch.
  for (auto input_tv : ir_utils::filterByType<TensorView>(fusion->inputs())) {
    for (auto id :
         TensorDomain::noReductions(input_tv->getMaybeRFactorDomain())) {
      auto id_set_it = id_to_disjoint_root_set.find(id);
      if (id_set_it == id_to_disjoint_root_set.end()) {
        continue;
      }
      auto* id_set = id_set_it->second;
      if (set_to_input_id.find(id_set) == set_to_input_id.end()) {
        set_to_input_id[id_set] = id;
      } else {
        auto input_id_of_set = set_to_input_id.at(id_set);
        // Swap id's if new name is less than previously set
        bool swap_ids = id->name() < input_id_of_set->name();
        // If new id is a const scalar but previously was'nt use the const
        // scalar
        swap_ids = swap_ids ||
            (id->extent()->isConstScalar() &&
             !input_id_of_set->extent()->isConstScalar());
        // If previous scalar was const and new isn't, don't swap
        swap_ids = swap_ids &&
            !(input_id_of_set->extent()->isConstScalar() &&
              !id->extent()->isConstScalar());

        if (swap_ids) {
          set_to_input_id[id_set] = id;
        }
      }
    }
  }

  // Finally make map from ID extents to the representitive ID extent.
  std::unordered_map<Val*, Val*> extent_to_min_input_id_extent;
  for (auto entry : set_to_input_id) {
    auto* set = entry.first;
    auto input_id = entry.second;
    for (auto id : *set) {
      extent_to_min_input_id_extent[id->extent()] = input_id->extent();
    }
  }
  return extent_to_min_input_id_extent;
}

} // namespace

void replaceSymbolicSizes(Fusion* fusion) {
  FUSER_PERF_SCOPE("GpuLower::Lower::replaceSymbolicSizes");
  std::unordered_map<Val*, Val*> tensor_dim_map;

  // Grab inputs and outputs
  std::vector<TensorView*> inputs_and_outputs;
  for (auto val : fusion->inputs()) {
    if (ir_utils::isTV(val)) {
      inputs_and_outputs.push_back(val->as<TensorView>());
    }
  }
  // Symbolic size is necessary for outputs if there are no inputs.
  // Otherwise infer output sizes from the inputs via expression evaluation.
  if (fusion->inputs().empty()) {
    for (auto val : fusion->outputs()) {
      if (ir_utils::isTV(val)) {
        inputs_and_outputs.push_back(val->as<TensorView>());
      }
    }
  }

  // Generate map for all tensorview root domain values to map them to symbolic
  // values. i.e. T0->getRootDomain()[0] would map to a named scalar
  // "T0.size[0]". This map will be used when lowering fusion ir to kernel ir.
  for (TensorView* tv : inputs_and_outputs) {
    // Replace the domain with one based on Ti.size[j]
    const std::vector<IterDomain*>& root_td = tv->getRootDomain();

    size_t dim = 0;
    for (auto id : root_td) {
      Val* orig_size = id->extent();
      // Output sizes could have reduction axes, which isn't what gets output.
      // NOLINTNEXTLINE(bugprone-branch-clone)
      if (id->isReduction()) {
        continue;
      } else if (orig_size->isConstScalar()) {
        dim++;
        continue;
      }

      // Currently turn off this part for inputs of segmented fusion,
      //  since FusionKernelRuntime will provide these as integer inputs
      if (tensor_dim_map.find(orig_size) == tensor_dim_map.end() &&
          !orig_size->isFusionInput() && !orig_size->isConstScalar()) {
        std::stringstream ss;
        ss << "T" << tv->name() << ".size[" << dim++ << "]";
        tensor_dim_map[orig_size] = IrBuilder::create<NamedScalar>(
            ss.str(), orig_size->getDataType().value());
      } else {
        dim++;
      }
    }
  }

  // Use a minimal number of sizes from provided tensors.
  auto extent_simplification_map = getSimplificationMap(fusion);
  for (auto extent_entry : extent_simplification_map) {
    auto orig_extent = extent_entry.first;
    auto simplified_extent = extent_entry.second;
    if (tensor_dim_map.count(orig_extent)) {
      if (tensor_dim_map.count(simplified_extent)) {
        tensor_dim_map[orig_extent] = tensor_dim_map[simplified_extent];
      } else {
        tensor_dim_map[orig_extent] = simplified_extent;
      }
    }
  }

  // Run mutation on the fusion with the tensor_dim_map
  ir_utils::replaceValue(fusion, tensor_dim_map);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
