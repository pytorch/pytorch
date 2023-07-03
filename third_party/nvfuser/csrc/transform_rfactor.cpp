#include <transform_rfactor.h>

#include <arith.h>
#include <fusion.h>
#include <instrumentation.h>
#include <ir_builder.h>
#include <ir_iostream.h>
#include <ir_utils.h>
#include <iter_visitor.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

// This class replays the root domains of the producer of an rfactor domain.
// Axes must be replayed to mark rfactor iter domains as being reductions in the
// producer, but converting the other reductions in the producer as iter
// domains. Those (previously reductions in the producer) iter domains are then
// converted to reduction domains in the consumer. This breaks up the reduction
// into two stages, but maintains the correct values are reduced across those
// stages.
//
// The rfactor domain of the producer must match the consumers root domain to
// maintain producer-consumer mappings. The following uses the original domain
// being rfactored and marked iter domains as "static_rfactor_ids". These static
// IDs cannot be changed in the producer as it would invalidate the rfactor, no
// longer matching the consumer.
//
// To find the rfactor domain in the producer which will be used as the root
// domain in the consumer, we start at the roots of producer, and replay forward
// the root iter domains if that iter domain is marked as a "static_rfactor_id".
// To do this we maintain the ordering of the iter domains. For example:
//
//       I1
//       /\           //
//     I2  \          //
//     /\  I3
//    / I4  /
//   /    \/
//  I5    I6
//
// If rfactor_axes = {I6}, then "static_rfactor_id" IDs will be {I6, I4, I3, I2,
// I1}. Then, as we perform the replay the rfactor domain will be updated as:
// [I1] -> [I2, I3] -> [I5, I4, I3] -> [I5, I6]
//
// ReplayTransformations typically updates the leaf ids, but we'll simply use
// the mapping from the original tensor domain so we won't bother updating them
// in this replay.
class ReplayRFactor : public ReplayTransformations {
 private:
  // Perform the update of the rfactor domain by replacing "replace0" with
  // "with0" and if not nullptr "with1", also removes "replace1" if not nullptr.
  void updateRFactorDomain(
      IterDomain* replace0,
      IterDomain* replace1,
      IterDomain* with0,
      IterDomain* with1) {
    TORCH_INTERNAL_ASSERT(
        with0 != nullptr,
        "The first provided IterDomain should be a real pointer,",
        " the second iter domain provided can be a nullptr.");
    auto pos =
        std::find(rfactor_domain_.begin(), rfactor_domain_.end(), replace0);
    TORCH_INTERNAL_ASSERT(
        pos != rfactor_domain_.end(),
        "Could not find iter domain: ",
        replace0->toString(),
        " in the rfactor domain to replace.");
    rfactor_domain_.insert(pos, with0);
    if (with1 != nullptr) {
      pos = std::find(rfactor_domain_.begin(), rfactor_domain_.end(), replace0);
      rfactor_domain_.insert(pos, with1);
    }
    pos = std::find(rfactor_domain_.begin(), rfactor_domain_.end(), replace0);
    rfactor_domain_.erase(pos);
    if (replace1 != nullptr) {
      pos = std::find(rfactor_domain_.begin(), rfactor_domain_.end(), replace1);
      TORCH_INTERNAL_ASSERT(
          pos != rfactor_domain_.end(),
          "Wanted to replace ",
          replace1->toString(),
          " but it's not in the rfactor domain.");
      rfactor_domain_.erase(pos);
    }
  }

  // Took a good bit of this from ReplayTransformations::handle(Split...)
  void handle(Split* s) override {
    // Grab input to the split operation
    auto id_in = s->in();
    // Grab our mapping of that ID to the one we're replaying
    auto it = id_map_.find(id_in);
    // Make sure it exists in the map
    TORCH_INTERNAL_ASSERT(
        it != id_map_.end(),
        "Transform traversal failed, dependencies not met.");
    // Grab the ID we're going to replay on
    auto mapped = (*it).second;
    // This ID should be a leaf ID (meaning it has no uses we generated)
    TORCH_INTERNAL_ASSERT(
        leaf_ids_.find(mapped) != leaf_ids_.end(),
        "Transform traversal failed, modified a node but it was not a leaf node.");

    // outer loop size
    Val* remainder = ceilDiv(mapped->extent(), s->factor());

    // Check if we need to mark the outputs as an rfactor domain meaning this
    // transformation must be present in replays otherwise it breaks the compute
    // definition of the fusion. Iter domains are actually not static, its the
    // transformation that's static or not, so if one output is marked as a
    // static id, then both must be.
    bool static_rfactor_outputs = static_rfactor_ids_.count(s->outer()) ||
        static_rfactor_ids_.count(s->inner());

    // Manually replay the split, making reduction = false and rfactor = true
    // outer IterDomain
    IterDomain* ido =
        IterDomainBuilder(
            s->container()->zeroVal(),
            s->innerSplit() ? remainder->as<Int>() : s->factor())
            .iter_type(
                rfactor_axes_.count(s->outer()) ? IterType::Reduction
                                                : IterType::Iteration)
            .is_rfactor_domain(static_rfactor_outputs)
            .build();

    // inner IterDomain
    IterDomain* idi =
        IterDomainBuilder(
            s->container()->zeroVal(),
            s->innerSplit() ? s->factor() : remainder->as<Int>())
            .iter_type(
                rfactor_axes_.count(s->inner()) ? IterType::Reduction
                                                : IterType::Iteration)
            .is_rfactor_domain(static_rfactor_outputs)
            .build();

    // Generate the split node
    IrBuilder::create<Split>(
        s->container(), ido, idi, mapped, s->factor(), s->innerSplit());

    // Remove mapped id from leaf IDs
    leaf_ids_.erase(mapped);
    // Add outputs to leaf IDs
    leaf_ids_[ido] = counter++;
    leaf_ids_[idi] = counter++;

    // Update our ID map to include these outputs
    id_map_[s->outer()] = ido;
    id_map_[s->inner()] = idi;

    if (static_rfactor_ids_.count(s->in())) {
      updateRFactorDomain(s->in(), nullptr, s->outer(), s->inner());
    }
  }

  void handle(Merge* m) override {
    auto id_outer = m->outer();
    auto id_inner = m->inner();
    auto it_outer = id_map_.find(id_outer);
    auto it_inner = id_map_.find(id_inner);
    TORCH_INTERNAL_ASSERT(
        it_outer != id_map_.end() && it_inner != id_map_.end(),
        "Transform traversal failed, dependencies not met.");

    auto id_outer_mapped = (*it_outer).second;
    auto id_inner_mapped = (*it_inner).second;

    TORCH_INTERNAL_ASSERT(
        leaf_ids_.find(id_outer_mapped) != leaf_ids_.end() &&
            leaf_ids_.find(id_inner_mapped) != leaf_ids_.end(),
        "Transform traversal failed, modified ",
        id_outer_mapped,
        " and ",
        id_inner_mapped,
        " however one or both are not leaf nodes.");

    Val* merged_id_size =
        mul(id_outer_mapped->extent(), id_inner_mapped->extent());

    IterDomain* merged_id =
        IterDomainBuilder(m->container()->zeroVal(), merged_id_size->as<Int>())
            .iter_type(
                rfactor_axes_.count(m->out()) ? IterType::Reduction
                                              : IterType::Iteration)
            .is_rfactor_domain(static_rfactor_ids_.count(m->out()))
            .build();

    IrBuilder::create<Merge>(
        m->container(), merged_id, id_outer_mapped, id_inner_mapped);

    // Remove inputs from the leaf IDs
    leaf_ids_.erase(id_outer_mapped);
    leaf_ids_.erase(id_inner_mapped);

    // Add the output to the leaf IDs
    leaf_ids_[merged_id] = counter++;

    id_map_[m->out()] = merged_id;

    // Similar to split replay above, check if output needs to be marked as
    // rfactor indicating this transofrmation is static.
    if (static_rfactor_ids_.count(m->inner()) ||
        static_rfactor_ids_.count(m->outer())) {
      TORCH_INTERNAL_ASSERT(
          static_rfactor_ids_.count(m->inner()) ==
              static_rfactor_ids_.count(m->outer()),
          "If one input to a merge is a static rfactor id, the other must be as well.");
      updateRFactorDomain(m->outer(), m->inner(), m->out(), nullptr);
    }
  }

  // The IterDomains in the original_domain that are being factored into the
  // first stage of the two stage reduction (the producer).
  std::unordered_set<IterDomain*> rfactor_axes_;
  // Iter domains whose history cannot be changed as it would break rfactor
  // dependencies.
  std::unordered_set<IterDomain*> static_rfactor_ids_;

 public:
  // The updated domain matching the producer's rfactor domain. This rfactor
  // domain is relative to the iter domains in the origianl_domain and must be
  // updated to grab the mapped id's later.
  std::vector<IterDomain*> rfactor_domain_;

  ReplayRFactor(
      // Original domain the rfactor is in reference to.
      TensorDomain* original_domain,
      // The root mapping from the original root domain, to the roots of the
      // domain to be replayed.
      std::unordered_map<IterDomain*, IterDomain*> id_map,
      // The rfactor axes in original_domain->domain() to be factored into the
      // two stage reduction.
      std::unordered_set<IterDomain*> rfactor_axes,
      // All the iter domains in original_domain that the rfactor axes are
      // dependant on.
      std::unordered_set<IterDomain*> static_rfactor_ids)
      : ReplayTransformations(
            original_domain->domain(),
            std::move(id_map),
            false),
        rfactor_axes_(std::move(rfactor_axes)),
        static_rfactor_ids_(static_rfactor_ids),
        rfactor_domain_(original_domain->getMaybeRFactorDomain()) {}
};

} // namespace

std::pair<TensorDomain*, TensorDomain*> TransformRFactor::runReplay(
    TensorDomain* original_td,
    std::vector<int> axes) {
  FUSER_PERF_SCOPE("TransformRFactor::runReplay");

  TORCH_CHECK(!axes.empty(), "No axes provided to rfactor replay.");

  int ndims = (int)original_td->nDims();

  // Adjust and check provided axes
  std::transform(axes.begin(), axes.end(), axes.begin(), [ndims](int i) {
    TORCH_CHECK(
        i >= -ndims && i < ndims,
        "Rfactor replay received an axis outside the number of dims in the tensor, acceptable inclusive range is ",
        -ndims,
        " to ",
        ndims - 1);
    return i < 0 ? i + ndims : i;
  });

  // remove duplicates, and put into a set for searching
  std::unordered_set<int> axes_set(axes.begin(), axes.end());

  TORCH_INTERNAL_ASSERT(
      std::all_of(
          axes_set.begin(),
          axes_set.end(),
          [original_td](int i) { return original_td->axis(i)->isReduction(); }),
      "Cannot rfactor axes that are not reduction axes.");

  // RFactor requires at least one reduction axis to be marked as factored out,
  // and at least one reduction axis that won't. Otherwise it's just a pointwise
  // cacheing operation.
  bool found_non_rfactor_reduction = false;

  // Make a set of final axes that are marked to be rfactored
  std::unordered_set<IterDomain*> rfactor_axes(axes_set.size());
  {
    size_t i = 0;
    for (auto id : original_td->domain()) {
      if (axes_set.find(i++) != axes_set.end()) {
        rfactor_axes.emplace(id);
      } else if (id->isReduction()) {
        found_non_rfactor_reduction = true;
      }
    }
  }

  TORCH_CHECK(
      found_non_rfactor_reduction,
      "Must have at least one reduction axis not marked as rfactor.");

  // Get root IterDomains of the rfactor domains, these will be the ones we will
  // replay marked as rfactor axes, those marked in the axes set will be
  // reduction=false
  auto rfactor_root_vals = IterVisitor::getInputsTo(
      std::vector<Val*>(rfactor_axes.begin(), rfactor_axes.end()));
  auto rfactor_root_ids = ir_utils::filterByType<IterDomain>(rfactor_root_vals);

  // Put in a set to make searching easy
  std::unordered_set<IterDomain*> rfactor_root_axes(
      rfactor_root_ids.begin(), rfactor_root_ids.end());

  TORCH_INTERNAL_ASSERT(
      std::none_of(
          rfactor_root_ids.begin(),
          rfactor_root_ids.end(),
          [](IterDomain* id) { return id->maybePartial(); }),
      "rFactor of partial domains not allowed, but at least one found.");

  auto original_td_root = original_td->getMaybeRFactorDomain();

  // Generate a new TensorDomain and set up map from one root to this one.
  std::vector<IterDomain*> new_producer_root(original_td_root.size(), nullptr);
  std::unordered_map<IterDomain*, IterDomain*> original_to_producer_root_map;

  {
    for (auto i : c10::irange(original_td_root.size())) {
      auto id = original_td_root[i];
      // If this is an rfactor root, it will be a reduction in this stage
      if (rfactor_root_axes.find(id) != rfactor_root_axes.end()) {
        new_producer_root[i] = IterDomainBuilder(id->start(), id->extent())
                                   .stop_offset(id->stopOffset())
                                   .iter_type(IterType::Reduction)
                                   .is_rfactor_domain(true)
                                   .build();
        // If this is not an rfactor root, but a reduction root, it should be
        // turned into an iteration domain
      } else if (id->isReduction()) {
        new_producer_root[i] = IterDomainBuilder(id->start(), id->extent())
                                   .stop_offset(id->stopOffset())
                                   .build();
      } else {
        new_producer_root[i] = id->cloneWithoutRFactor();
      }
      original_to_producer_root_map[id] = new_producer_root[i++];
    }
  }

  // Axes in the original_td that are in the history of the rfactored domains.
  // These will mark which iter domains must be preserved as static
  // transformations to preserve compute semantics.
  auto all_deps_of_rfactor = DependencyCheck::getAllValsBetween(
      {original_td->getMaybeRFactorDomain().begin(),
       original_td->getMaybeRFactorDomain().end()},
      {rfactor_axes.begin(), rfactor_axes.end()});

  auto all_id_deps_of_rfactor =
      ir_utils::filterByType<IterDomain>(all_deps_of_rfactor);

  std::unordered_set<IterDomain*> static_rfactor_ids(
      {all_id_deps_of_rfactor.begin(), all_id_deps_of_rfactor.end()});

  // Replay producer dimensions.
  ReplayRFactor replay_rfactor(
      original_td,
      original_to_producer_root_map,
      rfactor_axes,
      static_rfactor_ids);

  std::unordered_map<IterDomain*, IterDomain*> original_to_producer_id_map =
      replay_rfactor.getReplay();

  std::vector<IterDomain*> new_producer_domain(original_td->nDims(), nullptr);
  {
    for (auto i : c10::irange(original_td->nDims())) {
      auto orig_id = original_td->axis(i);
      auto replayed_id_it = original_to_producer_id_map.find(orig_id);
      TORCH_INTERNAL_ASSERT(
          replayed_id_it != original_to_producer_id_map.end(),
          "Error during rfactor replay, missing an axis.");
      auto replayed_id = replayed_id_it->second;
      replayed_id->parallelize(orig_id->getParallelType());
      if (orig_id->hasPaddingToMultipleOfWarp()) {
        replayed_id->padToMultipleOfWarp(orig_id->getMaybeSizeAfterPadding());
      }
      new_producer_domain[i++] = replayed_id;
    }
  }

  // Specify the rfactor domain of the producer which will match the consumer
  // root domain.
  std::vector<IterDomain*> new_producer_rfactor_domain;
  new_producer_rfactor_domain.reserve(replay_rfactor.rfactor_domain_.size());
  std::transform(
      replay_rfactor.rfactor_domain_.begin(),
      replay_rfactor.rfactor_domain_.end(),
      std::back_inserter(new_producer_rfactor_domain),
      [&](IterDomain* id) {
        auto replayed_id_it = original_to_producer_id_map.find(id);
        TORCH_INTERNAL_ASSERT(
            replayed_id_it != original_to_producer_id_map.end(),
            "Error during rfactor replay, missing an axis.");
        return replayed_id_it->second;
      });

  TensorDomain* producer_domain = IrBuilder::create<TensorDomain>(
      original_td->container(),
      new_producer_root,
      new_producer_rfactor_domain,
      new_producer_domain,
      std::vector<bool>(new_producer_rfactor_domain.size(), true));

  // Producer has been finished, now work on consumer.

  // For convenience flip the original to producer map
  std::unordered_map<IterDomain*, IterDomain*> producer_to_original_map;
  for (auto entry : original_to_producer_id_map) {
    producer_to_original_map[entry.second] = entry.first;
  }

  std::vector<IterDomain*> new_consumer_root_domain;
  new_consumer_root_domain.reserve(new_producer_rfactor_domain.size());
  std::unordered_map<IterDomain*, IterDomain*> original_to_consumer_root_map;
  for (auto p_root_id : new_producer_rfactor_domain) {
    if (p_root_id->isReduction()) {
      continue;
    }
    auto p2o_it = producer_to_original_map.find(p_root_id);
    TORCH_INTERNAL_ASSERT(
        p2o_it != producer_to_original_map.end(),
        "Missing mapping from original tensor domain to producer tensor domain.");
    auto original_id = p2o_it->second;
    auto new_consumer_root =
        IterDomainBuilder(original_id->start(), original_id->extent())
            .stop_offset(original_id->stopOffset())
            .iter_type(original_id->getIterType())
            .build();
    new_consumer_root_domain.push_back(new_consumer_root);
    original_to_consumer_root_map[original_id] = new_consumer_root;
  }

  ReplayTransformations consumer_replay(
      original_td->domain(), original_to_consumer_root_map, false);
  auto original_to_consumer_map = consumer_replay.getReplay();

  std::vector<IterDomain*> new_consumer_domain;

  {
    // Construct the new consumer domain
    for (auto i : c10::irange(original_td->nDims())) {
      auto orig_id = original_td->axis(i);
      auto replayed_id_it = original_to_consumer_map.find(orig_id);
      if (replayed_id_it != original_to_consumer_map.end()) {
        auto replayed_id = replayed_id_it->second;
        new_consumer_domain.push_back(replayed_id);
        replayed_id->parallelize(orig_id->getParallelType());
        if (orig_id->hasPaddingToMultipleOfWarp()) {
          replayed_id->padToMultipleOfWarp(orig_id->getMaybeSizeAfterPadding());
        }
      }
    }
  }

  auto consumer_domain = IrBuilder::create<TensorDomain>(
      original_td->container(),
      new_consumer_root_domain,
      new_consumer_domain,
      std::vector<bool>(new_consumer_root_domain.size(), true));

  return std::make_pair(producer_domain, consumer_domain);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
