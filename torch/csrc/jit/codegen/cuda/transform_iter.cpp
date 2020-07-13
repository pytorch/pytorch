#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

namespace torch {
namespace jit {
namespace fuser {

// Transform dispatch
void ReplayTransformations::handle(Expr* e) {
  switch (e->getExprType().value()) {
    case (ExprType::Split):
    case (ExprType::Merge):
      break;
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Invalid expr type found in transform traversal.");
  }
  IterVisitor::handle(e);
}

// We're going to replay this split operation on the corresponding ID
void ReplayTransformations::handle(Split* s) {
  // Grab our input to the split node
  auto id_in = s->in();

  // Make sure we have a corresponding entry in our map pointing to the ID we're
  // going to replay the split on
  auto it = id_map_.find(id_in);
  if (it == id_map_.end()) {
    if (check_all_ops_run_) {
      TORCH_INTERNAL_ASSERT(
          false, "Transform traversal failed, dependencies not met.");
    } else {
      return;
    }
  }

  auto mapped = (*it).second;
  TORCH_INTERNAL_ASSERT(
      s->factor()->isConst(),
      "Transform traversal does not support splitting on non-const values.");
  // Make sure this ID is a leaf ID (meaning it has no uses we generated)
  TORCH_INTERNAL_ASSERT(
      leaf_ids_.find(mapped) != leaf_ids_.end(),
      "Transform traversal failed, modified a node but it was not a leaf node.");

  // Replay the split onto mapped
  auto outs = IterDomain::split(mapped, s->factor()->value().value());
  // Remove mapped from the leaf IDs
  leaf_ids_.erase(mapped);

  // Add outputs to leaf IDs
  leaf_ids_[outs.first] = counter++;
  leaf_ids_[outs.second] = counter++;

  // Update our ID map to include these outputs
  id_map_[s->outer()] = outs.first;
  id_map_[s->inner()] = outs.second;
}

// We're going to replay this merge operation on the corresponding IDs
void ReplayTransformations::handle(Merge* m) {
  // Grab the inputs to the merge node
  auto id_outer = m->outer();
  auto id_inner = m->inner();

  // Make sure we have a corresponding entry in our map pointing to the IDs
  // we're going to replay the merge on
  auto it_outer = id_map_.find(id_outer);
  auto it_inner = id_map_.find(id_inner);
  if (it_outer == id_map_.end() || it_inner == id_map_.end()) {
    if (check_all_ops_run_) {
      TORCH_INTERNAL_ASSERT(
          false, "Transform traversal failed, dependencies not met.");
    } else {
      return;
    }
  }

  // Grab the IDs we're going to replay this merge on
  auto id_outer_mapped = (*it_outer).second;
  auto id_inner_mapped = (*it_inner).second;

  // Make sure these IDs are leaf IDs (meaning they have no uses we generated)
  TORCH_INTERNAL_ASSERT(
      leaf_ids_.find(id_outer_mapped) != leaf_ids_.end() &&
          leaf_ids_.find(id_inner_mapped) != leaf_ids_.end(),
      "Transform traversal failed, tried to replay with ",
      id_outer_mapped,
      " and ",
      id_inner_mapped,
      " however one or both are not leaf nodes.");

  // Replay the merge operation
  auto out = IterDomain::merge(id_outer_mapped, id_inner_mapped);

  // Remove inputs from the leaf IDs
  leaf_ids_.erase(id_outer_mapped);
  leaf_ids_.erase(id_inner_mapped);

  // Add the output to the leaf IDs
  leaf_ids_[out] = counter++;

  // Update our ID map with the replayed output
  id_map_[m->out()] = out;
}

ReplayTransformations::ReplayTransformations(
    const std::vector<IterDomain*>& _target_domain,
    std::unordered_map<IterDomain*, IterDomain*> _id_map,
    bool _check_all_ops_run)
    : target_domain_(_target_domain),
      id_map_(std::move(_id_map)),
      check_all_ops_run_(_check_all_ops_run) {
  // Make sure id_map has all the inputs needed to replay target_domain
  auto inps = IterVisitor::getInputsTo(
      std::vector<Val*>(target_domain_.begin(), target_domain_.end()));

  if (check_all_ops_run_)
    std::for_each(inps.begin(), inps.end(), [this](Val* val) {
      TORCH_INTERNAL_ASSERT(
          val->getValType().value() == ValType::IterDomain,
          "Expected IterDomain only for Replay Transformations, but found ",
          val);
      IterDomain* id = static_cast<IterDomain*>(val);
      TORCH_INTERNAL_ASSERT(
          this->id_map_.find(id) != this->id_map_.end(),
          "Could not find required input: ",
          id,
          " in provided id_map.");
    });

  // Set all the leaf nodes for tracking, all ids start as a leaf and will be
  // updated based on the transformations
  for (auto entry : id_map_)
    leaf_ids_[entry.second] = counter++;
}

// Replays outputs that were generated from ids.first on ids.second
void ReplayTransformations::runReplay() {
  TORCH_INTERNAL_ASSERT(
      !ran_replay,
      "Cannot run replay twice without creating a new Replay Class.");
  ran_replay = true;
  if (target_domain_.empty() || id_map_.empty())
    return;

  // Switch outDomain to a vector to start the traversal
  std::vector<Val*> traversal_vals(
      target_domain_.begin(), target_domain_.end());
  traverseFrom(traversal_vals[0]->fusion(), traversal_vals);

  if (check_all_ops_run_)
    TORCH_INTERNAL_ASSERT(
        leaf_ids_.size() >= target_domain_.size(),
        "Transform traversal failed, did not find enough output IterDomains.");

  // Validate replay
  for (auto out : target_domain_) {
    auto it_replayed = id_map_.find(out);
    if (it_replayed == id_map_.end()) {
      if (check_all_ops_run_) {
        TORCH_INTERNAL_ASSERT(
            false,
            "Transform traversal failed, could not find expected output.");
      }
      continue;
    }

    auto id_replayed = (*it_replayed).second;
    auto it_leaf = leaf_ids_.find(id_replayed);
    TORCH_INTERNAL_ASSERT(
        it_leaf != leaf_ids_.end(),
        "Transform Traversal failed, expected matched output to be a leaf of the replay, but was not.");
  }

  // Populate leaf_vec_ in a deterministic manner. This is deterministic
  // because size_t in leaf_ids is filled based on operation order.
  std::set<std::pair<IterDomain*, size_t>, id_int_lt> ordered_set;
  for (auto entry : leaf_ids_)
    ordered_set.emplace(entry);

  leaf_vec_.clear();
  leaf_vec_.resize(ordered_set.size());
  std::transform(
      ordered_set.begin(),
      ordered_set.end(),
      leaf_vec_.begin(),
      [](std::pair<IterDomain*, size_t> entry) { return entry.first; });
}

BestEffortReplay::BestEffortReplay(
    const std::vector<IterDomain*>& replay_domain,
    const std::vector<IterDomain*>& target_domain,
    std::unordered_map<IterDomain*, IterDomain*> replay_map)
    : id_map_(std::move(replay_map)) {
  for (auto entry : id_map_)
    leaf_ids_[entry.second] = counter++;

  std::vector<Expr*> t_exprs = Exprs::getFrom(
      std::vector<Val*>(target_domain.begin(), target_domain.end()));

  // If we check how an IterDomain was generated, it should only use an
  // IterDomain in an expression once. We pull a map from the input
  // IterDomains to the expression consuming them to generate the
  // replay_domain domain. This will be used to propagate the target_domain to
  // replay_domain map.

  std::vector<Expr*> r_exprs = Exprs::getFrom(
      std::vector<Val*>(replay_domain.begin(), replay_domain.end()));
  std::unordered_map<IterDomain*, Expr*> replay_expr_map;
  for (auto r_expr : r_exprs)
    for (auto inp : r_expr->inputs())
      if (inp->getValType().value() == ValType::IterDomain) {
        auto id = static_cast<IterDomain*>(inp);
        TORCH_INTERNAL_ASSERT(
            replay_expr_map.find(id) == replay_expr_map.end(),
            "Error trying to map rfactor root domain during replay. IterDomain's shouldn't have more than one use.");
        // Only want to forward rfactor in map
        replay_expr_map[id] = r_expr;
      }

  std::string err_str(
      "Error during replay, a computeAt was called that conflicts with an rfactor call.");

  for (auto t_expr : t_exprs) {
    // Going to map the target_domain inputs/outputs to replay_domain
    // inputs/outputs
    std::vector<IterDomain*> r_inps;
    std::vector<IterDomain*> t_inps;

    for (auto inp : t_expr->inputs()) {
      if (inp->getValType() == ValType::IterDomain) {
        auto t_inp = static_cast<IterDomain*>(inp);
        t_inps.push_back(t_inp);
        // There might not be a mapping, that could be okay.
        auto it = id_map_.find(t_inp);
        if (it != id_map_.end())
          r_inps.push_back(it->second);
      }
    }

    bool has_rfactor =
        std::any_of(r_inps.begin(), r_inps.end(), [](IterDomain* id) {
          return id->isRFactorProduct();
        });

    if (r_inps.size() != t_inps.size() || r_inps.empty()) {
      // If any replay_domain inputs are an rfactor product, all inputs should
      // match.
      TORCH_INTERNAL_ASSERT(!has_rfactor, err_str);
      continue;
    }

    if (replay_expr_map.find(r_inps[0]) == replay_expr_map.end()) {
      TORCH_INTERNAL_ASSERT(!has_rfactor, err_str);
      continue;
    }

    auto r_expr = replay_expr_map[r_inps[0]];
    bool mismatched_inputs = false;
    {
      size_t i = 0;
      for (auto r_inp : r_expr->inputs()) {
        if (i > r_inps.size()) {
          mismatched_inputs = true;
          break;
        }
        mismatched_inputs = mismatched_inputs || r_inp != r_inps[i];
        i++;
      }
    }

    if (mismatched_inputs) {
      TORCH_INTERNAL_ASSERT(!has_rfactor, err_str);
      continue;
    }

    if (t_expr->nOutputs() != r_expr->nOutputs()) {
      TORCH_INTERNAL_ASSERT(!has_rfactor, err_str);
      continue;
    }

    if (r_expr->getExprType().value() != t_expr->getExprType().value()) {
      TORCH_INTERNAL_ASSERT(!has_rfactor, err_str);
      continue;
    }

    // If the expression is a split, make sure it's split by the same ammount.
    if (r_expr->getExprType().value() == ExprType::Split) {
      if (!static_cast<Split*>(r_expr)->factor()->sameAs(
              static_cast<Split*>(r_expr)->factor())) {
        TORCH_INTERNAL_ASSERT(!has_rfactor, err_str);
        continue;
      }
    }

    bool missing_input = std::any_of(
        t_expr->inputs().begin(), t_expr->inputs().end(), [this](Val* inp) {
          if (inp->getValType() == ValType::IterDomain) {
            return id_map_.find(static_cast<IterDomain*>(inp)) == id_map_.end();
          }
          return false;
        });

    if (missing_input) {
      TORCH_INTERNAL_ASSERT(!has_rfactor, err_str);
      continue;
    }
    // Take target_domain inputs out of map:
    for (auto inp : t_expr->inputs()) {
      if (inp->getValType() == ValType::IterDomain) {
        auto t_inp = static_cast<IterDomain*>(inp);
        auto it = id_map_.find(t_inp);
        if (leaf_ids_.find(it->second) != leaf_ids_.end()) {
          leaf_ids_.erase(it->second);
        }
      }
    }

    // Add outputs to map.
    for (size_t i = 0; i < t_expr->nOutputs(); i++) {
      auto t_out = t_expr->output(i);
      auto r_out = r_expr->output(i);
      if (t_out->getValType() == ValType::IterDomain &&
          r_out->getValType() == ValType::IterDomain) {
        id_map_[static_cast<IterDomain*>(t_out)] =
            static_cast<IterDomain*>(r_out);
        leaf_ids_[static_cast<IterDomain*>(r_out)] = counter++;
      }
    }
  }
}

} // namespace fuser
} // namespace jit
} // namespace torch