#include <torch/csrc/jit/codegen/cuda/transform_rfactor.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

class ReplayRFactor : public ReplayTransformations {
 private:
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

    // Check if either outputs of the split are going to be an rfactored axis
    bool rfactor_outer = false;
    bool rfactor_inner = false;
    if (rfactor_axes_.find(s->outer()) != rfactor_axes_.end())
      rfactor_outer = true;

    if (rfactor_axes_.find(s->inner()) != rfactor_axes_.end())
      rfactor_inner = true;

    bool rfactor_input = mapped->isRFactorProduct();

    // If nothing is going to be rfactored replay a normal split
    if (!rfactor_inner && !rfactor_outer && !rfactor_input)
      return ReplayTransformations::handle(s);

    // outer loop size
    Val* oe = ceilDiv(mapped->extent(), s->factor());

    // Manually replay the split, making reduction = false and rfactor = true
    // outer IterDomain
    IterDomain* ido = new IterDomain(
        new Int(0),
        oe->as<Int>(),
        mapped->getParallelType(),
        rfactor_outer ? IterType::Reduction : IterType::Iteration,
        true); // broadcast

    // inner IterDomain
    IterDomain* idi = new IterDomain(
        new Int(0),
        s->factor(),
        mapped->getParallelType(),
        rfactor_inner ? IterType::Reduction : IterType::Iteration,
        true);

    // Generate the split node
    new Split(ido, idi, mapped, s->factor());

    // Remove mapped id from leaf IDs
    leaf_ids_.erase(mapped);
    // Add outputs to leaf IDs
    leaf_ids_[ido] = counter++;
    leaf_ids_[idi] = counter++;

    // Update our ID map to include these outputs
    id_map_[s->outer()] = ido;
    id_map_[s->inner()] = idi;
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

    bool rfactor_output = false;
    if (rfactor_axes_.find(m->out()) != rfactor_axes_.end())
      rfactor_output = true;

    bool rfactor_input = id_inner_mapped->isRFactorProduct() ||
        id_outer_mapped->isRFactorProduct();

    if (!rfactor_output && !rfactor_input)
      return ReplayTransformations::handle(m);

    Val* merged_id_size =
        mul(id_outer_mapped->extent(), id_inner_mapped->extent());

    IterDomain* merged_id = new IterDomain(
        new Int(0),
        merged_id_size->as<Int>(),
        id_outer_mapped->getParallelType(),
        rfactor_output ? IterType::Reduction : IterType::Iteration,
        true);

    new Merge(merged_id, id_outer_mapped, id_inner_mapped);

    // Remove inputs from the leaf IDs
    leaf_ids_.erase(id_outer_mapped);
    leaf_ids_.erase(id_inner_mapped);

    // Add the output to the leaf IDs
    leaf_ids_[merged_id] = counter++;

    id_map_[m->out()] = merged_id;
  }

  std::unordered_set<IterDomain*> rfactor_axes_;

 public:
  ReplayRFactor(
      const std::vector<IterDomain*>& _target_domain,
      std::unordered_map<IterDomain*, IterDomain*> _id_map,
      std::unordered_set<IterDomain*> _rfactor_axes)
      : ReplayTransformations(_target_domain, std::move(_id_map), false),
        rfactor_axes_(std::move(_rfactor_axes)) {}
};

} // namespace

// Take any axes not provided, that are reductions, and convert them to
// iteration axes. Any axes that share inputs to the axes provided should be
// marked as rfactorProduct.
TensorDomain* TransformRFactor::runReplay(
    TensorDomain* orig_td,
    std::vector<int> axes) {
  FUSER_PERF_SCOPE("runReplay");

  TORCH_CHECK(!axes.empty(), "No axes provided to rfactor replay.");

  int ndims = (int)orig_td->nDims();

  // Adjust and check provided axes
  std::transform(axes.begin(), axes.end(), axes.begin(), [ndims](int i) {
    TORCH_CHECK(
        i >= -ndims && i < ndims,
        "Rfactor replay recieved an axis outside the number of dims in the tensor, acceptable inclusive range is ",
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
          [orig_td](int i) { return orig_td->axis(i)->isReduction(); }),
      "Cannot rfactor axes that are not reduction axes.");

  // RFactor requires at least one reduction axis to be marked as factored out,
  // and at least one reduction axis that won't. Otherwise it's just a pointwise
  // cacheing operation.
  bool found_non_rfactor_reduction = false;

  // Make a set of final axes that are marked to be rfactored
  std::unordered_set<IterDomain*> rfactor_axes(axes_set.size());
  {
    size_t i = 0;
    for (auto id : orig_td->domain()) {
      if (axes_set.find(i++) != axes_set.end())
        rfactor_axes.emplace(id);
      if (id->isReduction())
        found_non_rfactor_reduction = true;
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

  // Make sure they're all IterDomains.
  TORCH_INTERNAL_ASSERT(
      std::all_of(
          rfactor_root_vals.begin(),
          rfactor_root_vals.end(),
          [](Val* v) {
            return v->getValType().value() == ValType::IterDomain;
          }),
      "Found invalid input domain axes.");

  // Put in a set to make searching easy
  std::unordered_set<IterDomain*> rfactor_root_axes;
  std::transform(
      rfactor_root_vals.begin(),
      rfactor_root_vals.end(),
      std::inserter(rfactor_root_axes, rfactor_root_axes.end()),
      [](Val* val) {
        TORCH_INTERNAL_ASSERT(
            val->getValType().value() == ValType::IterDomain,
            "Invalid value type found in rfactor axes inputs.");
        return val->as<IterDomain>();
      });

  auto orig_td_root = orig_td->getRootDomain();

  // Generate a new TensorDomain and set up map from one root to this one.
  std::vector<IterDomain*> new_root(orig_td_root.size(), nullptr);
  std::unordered_map<IterDomain*, IterDomain*> replay_map;

  {
    size_t i = 0;
    for (auto id : orig_td_root) {
      // If this is an rfactor root, it will be a reduction in this stage
      if (rfactor_root_axes.find(id) != rfactor_root_axes.end()) {
        new_root[i] = new IterDomain(
            id->start(),
            id->extent(),
            id->getParallelType(),
            IterType::Reduction,
            true);
        // If this is not an rfactor root, but a reduction root, it should be
        // turned into an iteration domain
      } else if (id->isReduction()) {
        new_root[i] = new IterDomain(
            id->start(),
            id->extent(),
            id->getParallelType(),
            IterType::Iteration,
            false);
      } else {
        new_root[i] = id->clone();
      }
      replay_map[id] = new_root[i++];
    }
  }

  // Replay producer dimensions.
  ReplayRFactor replay_rfactor(orig_td->domain(), replay_map, rfactor_axes);

  std::unordered_map<IterDomain*, IterDomain*> replayed =
      replay_rfactor.getReplay();

  std::vector<IterDomain*> new_domain(orig_td->nDims(), nullptr);
  {
    size_t i = 0;
    for (auto id : orig_td->domain()) {
      TORCH_INTERNAL_ASSERT(
          replayed.find(id) != replayed.end(),
          "Error during rfactor replay, missing an axis.");
      new_domain[i++] = replayed[id];
    }
  }

  // We need a root to match up with the consumer of this domain, it should have
  // rfactor axes after transformations, but not other axes.
  std::vector<IterDomain*> rfactor_root;
  for (auto dom : new_root)
    if (!dom->isRFactorProduct())
      rfactor_root.push_back(dom);

  for (auto dom : new_domain)
    if (dom->isRFactorProduct())
      rfactor_root.push_back(dom);

  return new TensorDomain(
      new_root,
      rfactor_root,
      new_domain,
      std::vector<bool>(new_root.size(), true));
}

// We want to take any axes marked in axes and remove them from the TensorDomain
// completely, any other reduction axes found should remain.
TensorDomain* TransformRFactor::runReplay2(
    TensorDomain* orig_td,
    std::vector<int> axes) {
  FUSER_PERF_SCOPE("runReplay2");

  int ndims = (int)orig_td->nDims();

  // Adjust and check provided axes
  std::transform(axes.begin(), axes.end(), axes.begin(), [ndims](int i) {
    TORCH_CHECK(
        i >= -ndims && i < ndims,
        "Rfactor replay recieved an axis outside the number of dims in the tensor, acceptable inclusive range is ",
        -ndims,
        " to ",
        ndims - 1);
    return i < 0 ? i + ndims : i;
  });

  // remove duplicates, and put into a set for searching
  std::set<int> axes_set(axes.begin(), axes.end());

  // Grab the axes in the rfactor, these were converted to iter domains in the
  // producer of this domain, and will be reduced in this domain
  std::unordered_set<IterDomain*> rfactor_axes(axes_set.size());
  {
    size_t i = 0;
    for (auto id : orig_td->domain()) {
      if (axes_set.find(i++) != axes_set.end())
        rfactor_axes.emplace(id);
    }
  }

  auto rfactor_root_vals = IterVisitor::getInputsTo(
      std::vector<Val*>(rfactor_axes.begin(), rfactor_axes.end()));

  // Make sure they're all IterDomains.
  TORCH_INTERNAL_ASSERT(
      std::all_of(
          rfactor_root_vals.begin(),
          rfactor_root_vals.end(),
          [](Val* v) {
            return v->getValType().value() == ValType::IterDomain;
          }),
      "Found invalid input domain axes.");

  // Put in a set to make searching easy
  std::unordered_set<IterDomain*> rfactor_root_axes;
  std::transform(
      rfactor_root_vals.begin(),
      rfactor_root_vals.end(),
      std::inserter(rfactor_root_axes, rfactor_root_axes.end()),
      [](Val* val) {
        TORCH_INTERNAL_ASSERT(
            val->getValType().value() == ValType::IterDomain,
            "Invalid value type found in rfactor axes inputs.");
        return val->as<IterDomain>();
      });

  // Replay all other root domains that are iter domains, as these will match in
  // the domain we're creating
  std::vector<IterDomain*> new_root;
  std::unordered_map<IterDomain*, IterDomain*> replay_root_map;
  for (auto id : orig_td->getRootDomain()) {
    if (rfactor_root_axes.find(id) == rfactor_root_axes.end()) {
      new_root.push_back(id->clone());
      replay_root_map[id] = new_root.back();
    }
  }

  ReplayTransformations rt(orig_td->domain(), replay_root_map, false);
  auto replayed = rt.getReplay();

  std::vector<IterDomain*> new_domain;

  {
    // Construct the new domain, and append rfactor axes to the new root domain
    size_t i = 0;
    for (auto id : orig_td->domain()) {
      if (replayed.find(id) != replayed.end()) {
        new_domain.push_back(replayed[id]);
      } else if (axes_set.find(i) == axes_set.end()) {
        IterDomain* new_id = id->clone();
        new_domain.push_back(new_id);
        new_root.push_back(new_id);
      }
      i++;
    }
  }

  return new TensorDomain(
      new_root, new_domain, std::vector<bool>(new_root.size(), true));
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
