#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

#include <torch/csrc/jit/codegen/cuda/ir_utils.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

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
    if (error_on_failure_) {
      TORCH_INTERNAL_ASSERT(
          false, "Transform traversal failed, dependencies not met.");
    } else {
      return;
    }
  }

  auto mapped = (*it).second;
  // Make sure this ID is a leaf ID (meaning it has no uses we generated)
  TORCH_INTERNAL_ASSERT(
      leaf_ids_.find(mapped) != leaf_ids_.end(),
      "Transform traversal failed, modified a node but it was not a leaf node.");

  // Replay the split onto mapped
  auto outs = IterDomain::split(mapped, s->factor());
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

  const bool outer_found = it_outer != id_map_.end();
  const bool outer_bcast = id_outer->isBroadcast();
  const bool inner_found = it_inner != id_map_.end();
  const bool inner_bcast = id_inner->isBroadcast();

  // If either are not found
  if (!outer_found || !inner_found) {
    // If both aren't found, it's a failure
    // If outer is found && inner is bcast it is not a failure
    // If inner is found && outer is bcast it is not a failure
    if (!(outer_found || inner_found) || (outer_found && !inner_bcast) ||
        (inner_found && !outer_bcast)) {
      if (error_on_failure_) {
        TORCH_INTERNAL_ASSERT(
            false, "Transform traversal failed, dependencies not met.");
      } else {
        return;
      }
    }
  }

  // If we merge a broadcast dim with a non-broadcast dim, just remap the output
  // to the non-broadcast dim.
  if (inner_found && !outer_found && outer_bcast) {
    id_map_[m->out()] = it_inner->second;
    return;
  }
  if (outer_found && !inner_found && inner_bcast) {
    id_map_[m->out()] = it_outer->second;
    return;
  }

  // Grab the IDs we're going to replay this merge on
  const auto id_outer_mapped = it_outer->second;
  const auto id_inner_mapped = it_inner->second;

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
    bool _error_on_failure)
    : target_domain_(_target_domain),
      id_map_(std::move(_id_map)),
      error_on_failure_(_error_on_failure) {
  // Make sure id_map has all the inputs needed to replay target_domain
  auto inps = IterVisitor::getInputsTo(
      std::vector<Val*>(target_domain_.begin(), target_domain_.end()));

  if (error_on_failure_)
    std::for_each(inps.begin(), inps.end(), [this](Val* val) {
      TORCH_INTERNAL_ASSERT(
          val->getValType().value() == ValType::IterDomain,
          "Expected IterDomain only for Replay Transformations, but found ",
          val);
      IterDomain* id = val->as<IterDomain>();
      TORCH_INTERNAL_ASSERT(
          id_map_.find(id) != id_map_.end(),
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

  if (error_on_failure_)
    TORCH_INTERNAL_ASSERT(
        leaf_ids_.size() >= target_domain_.size(),
        "Transform traversal failed, did not find enough output IterDomains.");

  // Validate replay
  for (auto out : target_domain_) {
    auto it_replayed = id_map_.find(out);
    if (it_replayed == id_map_.end()) {
      if (error_on_failure_) {
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
        "Transform Traversal failed, expected a replayed dim for ",
        out,
        " but one was not created.");
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
    std::unordered_map<IterDomain*, IterDomain*> replay_map,
    bool forward_bcast_mismatch)
    : id_map_(std::move(replay_map)) {
  for (auto entry : id_map_)
    leaf_ids_[entry.second] = counter++;

  // Grab expr history of iter domains in target_domain
  std::vector<Expr*> t_exprs = ExprSort::getExprs(
      FusionGuard::getCurFusion(),
      std::vector<Val*>(target_domain.begin(), target_domain.end()));

  // If we check how an IterDomain was generated, it should only use an
  // IterDomain in an expression once. We pull a map from the input
  // IterDomains to the expression consuming them to generate the
  // replay_domain domain. This will be used to propagate the target_domain to
  // replay_domain map.

  // Maps replay domain's IterDomains to the Exprs they're used in
  std::vector<Expr*> r_exprs = ExprSort::getExprs(
      FusionGuard::getCurFusion(),
      std::vector<Val*>(replay_domain.begin(), replay_domain.end()));
  std::unordered_map<IterDomain*, Expr*> replay_expr_map;
  for (auto r_expr : r_exprs) {
    for (auto id : ir_utils::filterByType<IterDomain>(r_expr->inputs())) {
      TORCH_INTERNAL_ASSERT(
          replay_expr_map.find(id) == replay_expr_map.end(),
          "Error trying to map rfactor root domain during replay. IterDomain's shouldn't have more than one use.");
      // Only want to forward rfactor in map
      replay_expr_map[id] = r_expr;
    }
  }

  std::string err_str(
      "Error during replay, a computeAt was called that conflicts with an rfactor call.");

  // Iterate through target IterDomains' history and compare with what we
  // recorded from replay_domain
  for (auto t_expr : t_exprs) {
    auto t_inps_filtered = ir_utils::filterByType<IterDomain>(t_expr->inputs());
    std::vector<IterDomain*> t_inps(
        t_inps_filtered.begin(), t_inps_filtered.end());

    std::vector<IterDomain*> r_inps =
        std::vector<IterDomain*>(t_inps.size(), nullptr);

    // Map t_expr inputs to replay domain directly
    for (size_t t_i = 0; t_i < t_inps.size(); t_i++) {
      // There might not be a mapping, that could be okay.
      auto it = id_map_.find(t_inps[t_i]);
      if (it != id_map_.end())
        r_inps[t_i] = it->second;
    }

    bool has_rfactor =
        std::any_of(r_inps.begin(), r_inps.end(), [](IterDomain* id) {
          return id == nullptr ? false : id->isRFactorProduct();
        });

    if (has_rfactor) {
      bool no_missing_exprs = std::none_of(
          r_inps.begin(), r_inps.end(), [&replay_expr_map](IterDomain* id) {
            if (id == nullptr) {
              return true;
            } else {
              return replay_expr_map.find(id) == replay_expr_map.end();
            }
          });
      TORCH_INTERNAL_ASSERT(no_missing_exprs, err_str);
    }

    // I would like to have this more generic or have this whole function go
    // through dispatch, but trying to make quick forward progress on
    // https://github.com/csarofeen/pytorch/issues/286 This mapping reflects
    // more closely what is done in ReplayTransform with mismatched
    // broadcast/merge
    if (forward_bcast_mismatch && !has_rfactor &&
        t_expr->getExprType().value() == ExprType::Merge) {
      auto t_merge = t_expr->as<Merge>();
      auto t_outer = t_merge->outer();
      auto t_inner = t_merge->inner();
      IterDomain* r_outer = id_map_.find(t_outer) != id_map_.end()
          ? id_map_.at(t_outer)
          : nullptr;
      IterDomain* r_inner = id_map_.find(t_inner) != id_map_.end()
          ? id_map_.at(t_inner)
          : nullptr;
      if (r_outer != nullptr && r_inner == nullptr && t_inner->isBroadcast()) {
        id_map_[t_merge->out()] = r_outer;
      } else if (
          r_inner != nullptr && r_outer == nullptr && t_outer->isBroadcast()) {
        id_map_[t_merge->out()] = r_inner;
      }
    }

    Expr* r_expr = nullptr;
    for (auto r_inp : r_inps) {
      if (r_inp != nullptr) {
        auto it = replay_expr_map.find(r_inp);
        if (it != replay_expr_map.end()) {
          r_expr = it->second;
          break;
        }
      }
    }

    if (r_expr == nullptr) {
      TORCH_INTERNAL_ASSERT(!has_rfactor, err_str);
      continue;
    }

    bool mismatched_inputs = r_inps.size() != r_expr->inputs().size();
    for (size_t i = 0; i < r_inps.size() && !mismatched_inputs; i++) {
      if (r_inps[i] == nullptr) {
        mismatched_inputs = true;
      } else {
        mismatched_inputs =
            mismatched_inputs || r_expr->inputs()[i] != r_inps[i];
      }
    }

    if (mismatched_inputs) {
      TORCH_INTERNAL_ASSERT(!has_rfactor, err_str);
      continue;
    }

    if (t_expr->outputs().size() != r_expr->outputs().size()) {
      TORCH_INTERNAL_ASSERT(!has_rfactor, err_str);
      continue;
    }

    if (r_expr->getExprType().value() != t_expr->getExprType().value()) {
      TORCH_INTERNAL_ASSERT(!has_rfactor, err_str);
      continue;
    }

    // If the expression is a split, make sure it's split by the same ammount.
    if (r_expr->getExprType().value() == ExprType::Split) {
      if (!r_expr->as<Split>()->factor()->sameAs(
              r_expr->as<Split>()->factor())) {
        TORCH_INTERNAL_ASSERT(!has_rfactor, err_str);
        continue;
      }
    }

    bool missing_input = std::any_of(
        t_expr->inputs().begin(), t_expr->inputs().end(), [this](Val* inp) {
          if (inp->getValType() == ValType::IterDomain) {
            return id_map_.find(inp->as<IterDomain>()) == id_map_.end();
          }
          return false;
        });

    if (missing_input) {
      TORCH_INTERNAL_ASSERT(!has_rfactor, err_str);
      continue;
    }
    // Take target_domain inputs out of map:
    for (auto t_inp : ir_utils::filterByType<IterDomain>(t_expr->inputs())) {
      auto it = id_map_.find(t_inp);
      if (leaf_ids_.find(it->second) != leaf_ids_.end()) {
        leaf_ids_.erase(it->second);
      }
    }

    // Add outputs to map.
    for (size_t i = 0; i < t_expr->outputs().size(); i++) {
      auto t_out = t_expr->output(i);
      auto r_out = r_expr->output(i);
      if (t_out->getValType() == ValType::IterDomain &&
          r_out->getValType() == ValType::IterDomain) {
        id_map_[t_out->as<IterDomain>()] = r_out->as<IterDomain>();
        leaf_ids_[r_out->as<IterDomain>()] = counter++;
      }
    }
  }
}

// Find the first position i where td1[i] is not the same as td2[i].
// "Same" means the DAG to generate td1[i] and td2[i] are the
// equivelent.
int BestEffortReplay::findFirstMismatchedID(
    const TensorDomain* td1,
    const TensorDomain* td2) {
  std::unordered_map<IterDomain*, IterDomain*> id_map;
  auto rd1 = td1->getRootDomain();
  auto rd2 = td2->getRootDomain();
  std::unordered_set<IterDomain*> rd2_set(
      td2->getRootDomain().begin(), td2->getRootDomain().end());

  // Find matching root IterDomains, we could make this O(nlog(n)) if we could
  // sort IterDomains.
  for (auto rd1i : rd1) {
    for (auto rd2i : rd2) {
      if (rd1i->sameAs(rd2i) && rd2_set.find(rd2i) != rd2_set.end()) {
        id_map[rd1i] = rd2i;
        rd2_set.erase(rd2i);
        break;
      }
    }
  }

  BestEffortReplay ber(td2->domain(), td1->domain(), id_map);

  for (size_t i = 0; i < td1->domain().size(); i++) {
    if (ber.getReplay().find(td1->axis(i)) == ber.getReplay().end()) {
      return i;
    }
    // Order is important.
    auto td2_axis = ber.getReplay().at(td1->axis(i));
    if (td2->axis(i) != td2_axis) {
      return i;
    }
  }
  return td1->nDims();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
