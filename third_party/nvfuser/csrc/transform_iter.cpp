#include <transform_iter.h>

#include <c10/util/irange.h>
#include <ir_utils.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// Transform dispatch
void ReplayTransformations::handle(Expr* e) {
  switch (e->getExprType().value()) {
    case (ExprType::Split):
    case (ExprType::Merge):
    case (ExprType::Swizzle2D):
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
  auto outs = IterDomain::split(
      mapped, s->factor(), s->innerSplit(), s->startOffset(), s->stopOffset());
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

void ReplayTransformations::handle(Swizzle2D* swizzle_2d) {
  // Grab our input to the split node
  auto id_in_x = swizzle_2d->inX();
  auto id_in_y = swizzle_2d->inY();

  // Make sure we have a corresponding entry in our map pointing to the ID we're
  // going to replay the swizzle on
  auto it_x = id_map_.find(id_in_x);
  auto it_y = id_map_.find(id_in_y);

  if (it_x == id_map_.end() || it_y == id_map_.end()) {
    if (error_on_failure_) {
      TORCH_INTERNAL_ASSERT(
          false, "Transform traversal failed, dependencies not met.");
    } else {
      return;
    }
  }

  auto mapped_x = (*it_x).second;
  auto mapped_y = (*it_y).second;

  // Make sure this ID is a leaf ID (meaning it has no uses we generated)
  TORCH_INTERNAL_ASSERT(
      leaf_ids_.find(mapped_x) != leaf_ids_.end() &&
          leaf_ids_.find(mapped_y) != leaf_ids_.end(),
      "Transform traversal failed, modified a node but it was not a leaf node.");

  auto outs = std::make_pair(mapped_x, mapped_y);

  if (replay_swizzle_) {
    // Replay the swizzle onto mapped
    outs = IterDomain::swizzle(swizzle_2d->swizzleType(), mapped_x, mapped_y);

    // Remove mapped from the leaf IDs
    leaf_ids_.erase(mapped_x);
    leaf_ids_.erase(mapped_y);
  }

  // Add outputs to leaf IDs
  leaf_ids_[outs.first] = counter++;
  leaf_ids_[outs.second] = counter++;

  // Update our ID map to include these outputs
  id_map_[swizzle_2d->outX()] = outs.first;
  id_map_[swizzle_2d->outY()] = outs.second;
}

ReplayTransformations::ReplayTransformations(
    const std::vector<IterDomain*>& _target_domain,
    std::unordered_map<IterDomain*, IterDomain*> _id_map,
    bool _error_on_failure,
    bool replay_swizzle)
    : target_domain_(_target_domain),
      id_map_(std::move(_id_map)),
      error_on_failure_(_error_on_failure),
      replay_swizzle_(replay_swizzle) {
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
  traverseTo(traversal_vals[0]->fusion(), traversal_vals);

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
    std::unordered_map<IterDomain*, IterDomain*> target2replay_map,
    std::unordered_map<IterDomain*, IterDomain*> replay_forward_id_map,
    std::unordered_map<IterDomain*, IterDomain*> target_forward_id_map,
    bool skip_swizzle)
    : target2replay_id_map_(std::move(target2replay_map)),
      replay_forward_id_map_(std::move(replay_forward_id_map)),
      target_forward_id_map_(std::move(target_forward_id_map)),
      skip_swizzle_(skip_swizzle) {
  for (auto entry : target2replay_id_map_) {
    leaf_ids_[entry.second] = counter++;
  }

  // Grab expr history of iter domains in target_domain
  std::vector<Expr*> target_exprs = StmtSort::getExprs(
      FusionGuard::getCurFusion(),
      std::vector<Val*>(target_domain.begin(), target_domain.end()));

  // If we check how an IterDomain was generated, it should only use an
  // IterDomain in an expression once. We pull a map from the input
  // IterDomains to the expression consuming them to generate the
  // replay_domain domain. This will be used to propagate the target_domain to
  // replay_domain map.

  // Map replay domain's IterDomains to the Exprs they're used in
  std::vector<Expr*> replay_exprs = StmtSort::getExprs(
      FusionGuard::getCurFusion(),
      std::vector<Val*>(replay_domain.begin(), replay_domain.end()));

  // Track which id's in replay have to be replayed to guarantee rfactor
  // transformations. The iteration domains in the rfactor axes don't have
  // to be used in a matching expression in target, so we want to exclude those.
  // Only the iteration domains [root_domains, rfactor) domains have to be used
  // in matching transformation to guarantee rfactor domain is consistent.
  // However, if any rfactor id was used to produce the rfactor domain, we need
  // transformations on them to match the target exactly.
  std::unordered_set<IterDomain*> replay_rfactor_ids;

  // Track which expressions iteration domains are used, they should only be
  // used in one expression.
  std::unordered_map<IterDomain*, Expr*> replay_id2expr_map;
  for (auto replay_expr : replay_exprs) {
    for (auto id : ir_utils::filterByType<IterDomain>(replay_expr->inputs())) {
      TORCH_INTERNAL_ASSERT(
          replay_id2expr_map.find(id) == replay_id2expr_map.end(),
          "Error trying to map rfactor root domain during replay.",
          " An IterDomain was found to be used in more than one expression.");

      replay_id2expr_map[id] = replay_expr;
    }

    // Only want to forward rfactor in map
    auto out_ids = ir_utils::filterByType<IterDomain>(replay_expr->outputs());
    if (std::any_of(out_ids.begin(), out_ids.end(), [](IterDomain* id) {
          return id->isRFactorProduct();
        })) {
      auto inp_ids = ir_utils::filterByType<IterDomain>(replay_expr->inputs());
      replay_rfactor_ids.insert(inp_ids.begin(), inp_ids.end());
    }
  }

  std::unordered_map<IterDomain*, Expr*> target_id2expr_map;
  for (auto target_expr : target_exprs) {
    for (auto id : ir_utils::filterByType<IterDomain>(target_expr->inputs())) {
      TORCH_INTERNAL_ASSERT(
          target_id2expr_map.insert({id, target_expr}).second,
          "BestEffortReplay : Unexpected multi-use of id",
          id);
    }
  }

  if (skip_swizzle_) {
    // Progress through all swizzle ops if we are skipping
    //  swizzles on the mapping.
    skipSwizzles(target_id2expr_map, replay_id2expr_map);
  }

  std::string err_str(
      "Error during replay, a transformation was called that conflicts with an rfactor call.");

  bool any_target_expr_contains_broadcast_id = false;

  // Iterate through target IterDomains' history and compare with what we
  // recorded from replay_domain
  for (auto target_expr : target_exprs) {
    auto target_inps_filtered =
        ir_utils::filterByType<IterDomain>(target_expr->inputs());

    // If any input argument in target expression is in the forward map then
    // forward the mapped IterDomains in replay and continue to the next
    // expression as target_expr cannot match a replay_expr
    if (std::any_of(
            target_inps_filtered.begin(),
            target_inps_filtered.end(),
            [&](IterDomain* target_inp) {
              return this->inTargetForwardMap(target_inp);
            })) {
      for (auto target_inp : target_inps_filtered) {
        if (inTargetForwardMap(target_inp)) {
          auto target2replay_it = target2replay_id_map_.find(target_inp);
          if (target2replay_it != target2replay_id_map_.end()) {
            // Replace target_inp entry in target2replay_id_map_ with forwarded
            // id
            target2replay_id_map_[getTargetForwardedId(target_inp)] =
                target2replay_it->second;
            target2replay_id_map_.erase(target_inp);
          }
        }
      }
      // Continue to next target_expr
      continue;
    }

    std::vector<IterDomain*> target_id_inps(
        target_inps_filtered.begin(), target_inps_filtered.end());

    bool target_expr_contains_broadcast_id = std::any_of(
        target_inps_filtered.begin(),
        target_inps_filtered.end(),
        [](IterDomain* id) { return id->isBroadcast(); });
    any_target_expr_contains_broadcast_id =
        any_target_expr_contains_broadcast_id ||
        target_expr_contains_broadcast_id;

    std::vector<IterDomain*> replay_inps =
        std::vector<IterDomain*>(target_id_inps.size(), nullptr);

    bool missing_replay_input = false;

    // Map target_expr inputs to replay domain directly
    for (const auto t_i : c10::irange(target_id_inps.size())) {
      // There might not be a mapping, that could be okay (depends on rfactor
      // checking).
      auto it = target2replay_id_map_.find(target_id_inps[t_i]);
      if (it != target2replay_id_map_.end()) {
        replay_inps[t_i] = getReplayForwardedId(it->second);
      } else {
        missing_replay_input = true;
      }
    }

    // Check if any of the associated replay id's are part of an rfactor domain
    bool replay_has_rfactor_inp = std::any_of(
        replay_inps.begin(),
        replay_inps.end(),
        [&replay_rfactor_ids](IterDomain* id) {
          return id == nullptr ? false
                               : id->isRFactorProduct() &&
                  (replay_rfactor_ids.find(id) != replay_rfactor_ids.end());
        });

    // If some replay id inputs are part of rfactor, make sure all target
    // expression inputs map to a replay input
    if (replay_has_rfactor_inp) {
      bool no_missing_exprs = std::none_of(
          replay_inps.begin(),
          replay_inps.end(),
          [&replay_id2expr_map](IterDomain* id) {
            if (id == nullptr) {
              return true;
            } else {
              return replay_id2expr_map.find(id) == replay_id2expr_map.end();
            }
          });
      // View operation creates a TensorView with rfactor. After view, broadcast
      // operation adds iterDomains for any size-1 dimensions. Therefore, the
      // target domain (broadcast) may contain broadcast ids that are not
      // present in the replay domain (view). In this case, we skip any target
      // expressions that contain broadcast ids.
      TORCH_INTERNAL_ASSERT(
          no_missing_exprs || any_target_expr_contains_broadcast_id, err_str);
    }

    // If any inputs are missing, continue as this expr doesn't match.
    if (missing_replay_input) {
      TORCH_INTERNAL_ASSERT(
          !replay_has_rfactor_inp || any_target_expr_contains_broadcast_id,
          err_str);
      continue;
    }

    // Find which replay_expr maps to the target_expr
    Expr* replay_expr = nullptr;
    // Check if all inputs have the same expression
    bool mismatched_replay_exprs = false;
    for (auto replay_inp : replay_inps) {
      auto it = replay_id2expr_map.find(replay_inp);
      if (it != replay_id2expr_map.end()) {
        if (replay_expr == nullptr) {
          replay_expr = it->second;
        } else {
          mismatched_replay_exprs =
              mismatched_replay_exprs || replay_expr != it->second;
        }
      } else {
        // If no expr is mapped then set mismatched epxrs to go to continue to
        // the next target expr
        mismatched_replay_exprs = true;
      }
    }

    // If expressions of mapped inputs don't match, then continue to next target
    // expr
    if (mismatched_replay_exprs || replay_expr == nullptr) {
      TORCH_INTERNAL_ASSERT(!replay_has_rfactor_inp, err_str);
      continue;
    }

    bool mismatched_inputs = replay_inps.size() != replay_expr->inputs().size();
    for (size_t i = 0; i < replay_inps.size() && !mismatched_inputs; i++) {
      mismatched_inputs =
          mismatched_inputs || replay_expr->inputs()[i] != replay_inps[i];
    }

    // If there isn't an rfactor id in the replay's inputs and there's a
    // mismatched input, continue
    if (mismatched_inputs) {
      TORCH_INTERNAL_ASSERT(!replay_has_rfactor_inp, err_str);
      continue;
    }

    // If there isn't an rfactor id in the replay's inputs and there's a
    // mismatch in replay_expr's and target_expr's outputs, continue
    if (target_expr->outputs().size() != replay_expr->outputs().size()) {
      TORCH_INTERNAL_ASSERT(!replay_has_rfactor_inp, err_str);
      continue;
    }

    // If there isn't an rfactor id in the replay's inputs and there's a
    // mismatch in replay_expr's and target_expr's expression type, continue
    if (replay_expr->getExprType().value() !=
        target_expr->getExprType().value()) {
      TORCH_INTERNAL_ASSERT(!replay_has_rfactor_inp, err_str);
      continue;
    }

    // If there isn't an rfactor id in the replay's inputs and there's a
    // mismatch in replay_expr's and target_expr's split factor (if a split
    // expr), continue
    if (replay_expr->getExprType().value() == ExprType::Split) {
      auto r_split = replay_expr->as<Split>();
      auto t_split = target_expr->as<Split>();
      if (!r_split->factor()->sameAs(t_split->factor()) ||
          r_split->innerSplit() != t_split->innerSplit() ||
          !r_split->startOffset()->sameAs(t_split->startOffset()) ||
          !r_split->stopOffset()->sameAs(t_split->stopOffset())) {
        TORCH_INTERNAL_ASSERT(!replay_has_rfactor_inp, err_str);
        continue;
      }
    }

    // Need to match swizzle type and parameters if
    //  not skipping swizzles in this mapping pass.
    if (!skip_swizzle_ && replay_expr->etype() == ExprType::Swizzle2D) {
      auto r_swizzle_2d = replay_expr->as<Swizzle2D>();
      auto t_swizzle_2d = target_expr->as<Swizzle2D>();
      if (!(r_swizzle_2d->swizzleType() == t_swizzle_2d->swizzleType())) {
        TORCH_INTERNAL_ASSERT(!replay_has_rfactor_inp, err_str);
        continue;
      }
    }

    // Take replay expr inputs out of map:
    for (const auto t_i : c10::irange(target_id_inps.size())) {
      auto t_inp = target_id_inps[t_i];
      auto r_orig_inp = target2replay_id_map_.at(t_inp);
      auto r_maybe_forwarded_inp = replay_inps[t_i];

      // Remove original target2replay_it->second if it's in leaf_ids
      if (leaf_ids_.find(r_orig_inp) != leaf_ids_.end()) {
        leaf_ids_.erase(r_orig_inp);
      }

      // Check if we used a forwarded id, if so add forwarded id's to tracking.
      if (r_orig_inp != r_maybe_forwarded_inp) {
        forwarded_ids_.emplace_back(r_orig_inp);
      }
    }

    // Add outputs to map.
    for (const auto i : c10::irange(target_expr->outputs().size())) {
      auto t_out = target_expr->output(i);
      auto r_out = replay_expr->output(i);
      if (t_out->getValType() == ValType::IterDomain &&
          r_out->getValType() == ValType::IterDomain) {
        target2replay_id_map_[t_out->as<IterDomain>()] =
            r_out->as<IterDomain>();
        leaf_ids_[r_out->as<IterDomain>()] = counter++;
      }
    }

    if (skip_swizzle_) {
      // Progress through all swizzle ops if we are skipping
      //  swizzles on the mapping.
      skipSwizzles(target_id2expr_map, replay_id2expr_map);
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
  for (const auto i :
       c10::irange(std::max(td1->domain().size(), td2->domain().size()))) {
    if (ber.getReplay().find(td1->axis(i)) == ber.getReplay().end()) {
      return i;
    }
    // Order is important.
    auto td2_axis = ber.getReplay().at(td1->axis(i));
    if (td2->axis(i) != td2_axis) {
      return i;
    }
  }
  return std::min(td1->nDims(), td2->nDims());
}

namespace {

// Maps that track information relevant to best effort replay about broadcast
// axes in consumer that are not in producer
//
// For example if we have consumer: T0[i0, b1, b2, i3] and producer:
// T1[i0, i3]
//
// If consumer transformations are:
// -> T[i0, b1o, b1i, b2o, b2i, i3]
// -> T[i0*b1i, b1o, b2o, b2i, i3]
// -> T[i0*b1i*b2o, b1o, b2i, i3]
// -> T[i0*b1i*b2o*i3, b1o, b2i]
//
// forwarding_map would forward i0->i0*b1i and i0*b1i->i0*b1i*b2o
// compliment_map would have the entry i0->b1i and i0*b1i->b2o
//
// The first is to fast forward transformations in consumer involving broadcast
// axes not in producer. The compliment map is to use later to compute what leaf
// nodes we may have after the forwarding process is finished. Leaf nodes are
// only important for replayCasP, so look there to see how this is done. Forward
// map is used for replayCasP and replayPasC.
struct ConsumerForwardingInfo {
 public:
  // Map IterDomain* axes that can safely be forwarded to their output.
  std::unordered_map<IterDomain*, IterDomain*> forwarding_map;

  // Given a forward id map id_input -> id_forwarded
  // Track the other inputs in the expr that id_input is an input to. These will
  // be used to adjust the replay's leaf tracking. Don't need to track one to
  // many as currently transformations on IterDomains can only have maximum 2
  // inputs, but maybe in the future we'll have more.
  std::unordered_map<IterDomain*, std::vector<IterDomain*>> compliment_map;

  ConsumerForwardingInfo(
      const TensorView* producer,
      const TensorView* consumer) {
    // Collect which root axes are in consumer that are not in producer because
    // of broadcasting
    std::unordered_set<IterDomain*> consumer_bcast_roots_not_in_producer;

    const auto c2p_root_map =
        PairwiseRootDomainMap(producer, consumer)
            .mapConsumerToProducer(consumer->domain(), producer->domain());

    for (auto consumer_root_id : consumer->getRootDomain()) {
      if (consumer_root_id->isBroadcast()) {
        if (c2p_root_map.find(consumer_root_id) == c2p_root_map.end()) {
          consumer_bcast_roots_not_in_producer.emplace(consumer_root_id);
        }
      }
    }

    // We have root axes in consumer that don't exist in producer, now forward
    // those to include all id's in consumer comprised of only axes not in
    // producer.
    auto consumer_bcast_ids_not_in_producer =
        consumer_bcast_roots_not_in_producer;

    std::vector<Expr*> consumer_history = StmtSort::getExprs(
        FusionGuard::getCurFusion(),
        std::vector<Val*>(
            consumer->domain()->domain().begin(),
            consumer->domain()->domain().end()));

    auto isIdOnlyInConsumer =
        [&consumer_bcast_ids_not_in_producer](IterDomain* input_id) {
          return consumer_bcast_ids_not_in_producer.find(input_id) !=
              consumer_bcast_ids_not_in_producer.end();
        };

    for (auto expr : consumer_history) {
      auto input_ids = ir_utils::filterByType<IterDomain>(expr->inputs());
      // If expr inputs are all in consumer_bcast_ids_not_in_producer, than so
      // are all outputs
      if (std::all_of(input_ids.begin(), input_ids.end(), isIdOnlyInConsumer)) {
        // add all outputs to not being in producer
        for (auto output_ids :
             ir_utils::filterByType<IterDomain>(expr->outputs())) {
          consumer_bcast_ids_not_in_producer.emplace(output_ids);
        }
      } else if (
          expr->isA<Merge>() &&
          std::any_of(input_ids.begin(), input_ids.end(), isIdOnlyInConsumer)) {
        auto merge_expr = expr->as<Merge>();
        // If
        // - one of the inputs is made of id's in consumer that don't map to
        // producer (bcast axes),
        // - && the other input maps to an id in both consumer and producer
        // - && this is a merge
        //   for the sake of BestEffortReplay we can forward the input mapping
        //   to both consumer and producer to the output of the expression
        std::vector<IterDomain*> forwarded_ids;
        std::vector<IterDomain*> compliment_ids;

        for (auto input_id : input_ids) {
          if (!isIdOnlyInConsumer(input_id)) {
            forwarded_ids.emplace_back(input_id);
            forwarding_map.emplace(std::make_pair(input_id, merge_expr->out()));
          } else {
            compliment_ids.push_back(input_id);
          }
        }

        // Set up compliment map
        for (auto forwarded_id : forwarded_ids) {
          compliment_map.emplace(std::make_pair(forwarded_id, compliment_ids));
        }
      }
    }
  }
};

// Maps that track information relevant to best effort replay about
// trivial-reduction axes in producer
//
// For example if we have producer: T0[i0, r1, r2, i3] and consumer:
// T1[i0, i3]
//
// If producer transformations are:
// -> T[i0, r1, r2, i3]
// -> T[i0*r1, r2, i3]
// -> T[i0*r1*r2, i3]
//
// forwarding_map would forward i0->i0*r1 and i0*r1->i0*r1*r2
// compliment_map would have the i0->r1 and i0*r1->r2
//
// These two maps are used similarly as ConsumerForwardingInfo. See
// its comments as well.
struct ProducerForwardingInfo {
 public:
  // Map IterDomain* axes that can safely be forwarded to their output.
  std::unordered_map<IterDomain*, IterDomain*> forwarding_map;

  // Given a forward id map id_input -> id_forwarded
  // Track the other inputs in the expr that id_input is an input to. These will
  // be used to adjust the replay's leaf tracking. Don't need to track one to
  // many as currently transformations on IterDomains can only have maximum 2
  // inputs, but maybe in the future we'll have more.
  std::unordered_map<IterDomain*, std::vector<IterDomain*>> compliment_map;

  ProducerForwardingInfo(const TensorView* producer) {
    std::vector<Expr*> producer_history = StmtSort::getExprs(
        FusionGuard::getCurFusion(),
        std::vector<Val*>(
            producer->domain()->domain().begin(),
            producer->domain()->domain().end()));

    for (auto merge : ir_utils::filterByType<Merge>(producer_history)) {
      auto inner = merge->inner();
      auto outer = merge->outer();
      if ((inner->isTrivialReduction() && !outer->isReduction()) ||
          (outer->isTrivialReduction() && !inner->isReduction())) {
        auto compliment_id = inner->isTrivialReduction() ? inner : outer;
        auto forwarded_id = inner->isTrivialReduction() ? outer : inner;
        forwarding_map.emplace(std::make_pair(forwarded_id, merge->out()));
        compliment_map.emplace(std::make_pair(
            forwarded_id, std::vector<IterDomain*>{compliment_id}));
      }
    }
  }
};

// Trace chain of swizzles until reaching
//  an IterDomain that's either a leaf or
//  not a producer of any swizzle.
IterDomain* getSwizzleFinalOutput(
    IterDomain* id,
    const std::unordered_map<IterDomain*, Expr*>& id2expr) {
  bool is_swizzle_input = true;

  // Note: currently not supporting swizzling consumer of another
  //  swizzle id, so this should terminate in 1 iter, but eventually
  //  will try to support stacked swizzles so keeping this pass
  //  generic.
  while (is_swizzle_input) {
    auto expr_it = id2expr.find(id);

    // This means id is a leaf that doesn't
    //  have any consumers. Stop iteration in this case.
    if (expr_it == id2expr.end()) {
      is_swizzle_input = false;
      break;
    }

    if (expr_it->second->etype() == ExprType::Swizzle2D) {
      // In the case of 2D swizzle ops, just forward
      //  inX to outX and inY to outY.
      auto expr = expr_it->second->as<Swizzle2D>();
      if (id == expr->inX()) {
        id = expr->outX();
      } else {
        TORCH_INTERNAL_ASSERT(
            id == expr->inY(),
            "unknown input to swizzle op",
            id->toString(),
            expr->toString());
        id = expr->outY();
      }
    } else {
      // Probably unreachable but if the expression
      //  is unknown type assume it is not a swizzle op.
      is_swizzle_input = false;
    }
  }

  return id;
}

bool isSwizzleInput(
    IterDomain* input_id,
    const std::unordered_map<IterDomain*, Expr*>& id2expr) {
  auto user_expr_it = id2expr.find(input_id);

  if (user_expr_it == id2expr.end()) {
    return false;
  }

  return user_expr_it->second->etype() == ExprType::Swizzle2D;
}

} // namespace

void BestEffortReplay::addComplimentLeafIDs(
    const std::unordered_map<IterDomain*, IterDomain*>& forwarding_map,
    const std::unordered_map<IterDomain*, std::vector<IterDomain*>>&
        compliment_map) {
  // ID's could go through more than one forward iteration in the map before it
  // terminates. Grab every id between the forwarded id, and what it was
  // forwarded to
  std::function<void(IterDomain*, std::vector<IterDomain*>&)>
      collectForwardedIds =
          [&forwarding_map, &collectForwardedIds](
              IterDomain* forward_id,
              std::vector<IterDomain*>& forwarded_ids) -> void {
    if (forwarding_map.find(forward_id) != forwarding_map.end()) {
      forwarded_ids.emplace_back(forward_id);
      collectForwardedIds(forwarding_map.at(forward_id), forwarded_ids);
    }
  };

  std::vector<IterDomain*> expanded_forwarded_ids;
  for (auto forwarded_id : forwarded_ids_) {
    collectForwardedIds(forwarded_id, expanded_forwarded_ids);
  }

  // Grab all compliments of forwarded ids.
  std::vector<IterDomain*> compliments;
  for (auto forwarded_id : expanded_forwarded_ids) {
    auto compliment_map_it = compliment_map.find(forwarded_id);
    TORCH_INTERNAL_ASSERT(
        compliment_map_it != compliment_map.end(),
        "Issue tracking forwarded broadcast merges in best effort replay.");
    compliments.insert(
        compliments.end(),
        compliment_map_it->second.begin(),
        compliment_map_it->second.end());
  }

  // Grab all exprs used to make the forwarded compliments
  auto compliment_exprs = StmtSort::getExprs(
      FusionGuard::getCurFusion(), {compliments.begin(), compliments.end()});

  // Figure out if there are any leaves in compliment_exprs that aren't
  // the forwarded id
  std::unordered_map<IterDomain*, size_t> leaf_ids;

  for (auto expr : compliment_exprs) {
    for (auto inp : ir_utils::filterByType<IterDomain>(expr->inputs())) {
      leaf_ids.erase(inp);
    }
    for (auto out : ir_utils::filterByType<IterDomain>(expr->outputs())) {
      // If we used the comliment for forwarded don't add to leaf nodes.
      if (std::find(compliments.begin(), compliments.end(), out) ==
          compliments.end()) {
        leaf_ids.emplace(std::make_pair(out, counter++));
      }
    }
  }

  leaf_ids_.insert(leaf_ids.begin(), leaf_ids.end());
}

BestEffortReplay BestEffortReplay::replayCasP(
    const TensorView* consumer,
    const TensorView* producer,
    int producer_compute_at_axis,
    const RootDomainMap& root_map) {
  if (producer_compute_at_axis < 0)
    producer_compute_at_axis += (int)producer->nDims() + 1;

  TORCH_INTERNAL_ASSERT(
      producer_compute_at_axis >= 0 &&
          (unsigned int)producer_compute_at_axis <= producer->nDims(),
      "Invalid axis provided to BestEffortReplay::replayCasP.");

  // producer ids we need to match in consumer
  std::vector<IterDomain*> producer_CA_ids(
      producer->domain()->domain().begin(),
      producer->domain()->domain().begin() + producer_compute_at_axis);
  producer_CA_ids = TensorDomain::noReductions(producer_CA_ids);

  // If producer has an rfactor root, that's what will match the consumer
  std::vector<IterDomain*> producer_root = producer->getMaybeRFactorDomain();

  // Figure out all inputs required to generate the compute_at dimensions. We
  // need all deps because inputs on producer may be in getRootDomain, but we
  // may need in rFactorDomain
  auto all_CA_id_deps = DependencyCheck::getAllValsBetween(
      {producer_root.begin(), producer_root.end()},
      {producer_CA_ids.begin(), producer_CA_ids.end()});

  // Figure out minimal set of root IDs needed to produce producer_CA_ids:
  std::unordered_set<IterDomain*> producer_CA_root_ids;
  for (IterDomain* id : producer_root) {
    if (std::find(all_CA_id_deps.begin(), all_CA_id_deps.end(), id) !=
        all_CA_id_deps.end()) {
      producer_CA_root_ids.emplace(id);
    }
  }

  const auto p2c_root_map = root_map.mapProducerToConsumer(
      producer->domain(), consumer->domain(), producer_CA_root_ids);

  // See FusionAdvancedComputeAt7 for an example of the forwarding logic
  ConsumerForwardingInfo consumer_forwarding_info(producer, consumer);

  ProducerForwardingInfo producer_forwarding_info(producer);

  auto consumer_replay = BestEffortReplay(
      consumer->domain()->domain(),
      producer_CA_ids,
      p2c_root_map,
      consumer_forwarding_info.forwarding_map,
      producer_forwarding_info.forwarding_map);

  consumer_replay.addComplimentLeafIDs(
      consumer_forwarding_info.forwarding_map,
      consumer_forwarding_info.compliment_map);

  return consumer_replay;
}

// Runs a best effort replay that ignores broadcast axes that appear in
// consumer that are not mapped to producer in root_map.
BestEffortReplay BestEffortReplay::replayPasC(
    const TensorView* producer,
    const TensorView* consumer,
    int consumer_compute_at_axis,
    const RootDomainMap& root_map) {
  if (consumer_compute_at_axis < 0)
    consumer_compute_at_axis += (int)consumer->nDims() + 1;
  TORCH_INTERNAL_ASSERT(
      consumer_compute_at_axis >= 0 &&
          (unsigned int)consumer_compute_at_axis <= consumer->nDims(),
      "Invalid axis provided to BestEffortReplay::replayPasC.");

  // consumer ids we need to match in producer
  std::vector<IterDomain*> consumer_CA_ids(
      consumer->domain()->domain().begin(),
      consumer->domain()->domain().begin() + consumer_compute_at_axis);

  // Figure out all inputs required to generate the compute_at dimensions
  auto consumer_CA_root_vals = IterVisitor::getInputsTo(
      std::vector<Val*>(consumer_CA_ids.begin(), consumer_CA_ids.end()));

  std::unordered_set<IterDomain*> consumer_CA_root_ids;
  for (auto val : consumer_CA_root_vals) {
    if (val->getValType().value() == ValType::IterDomain) {
      consumer_CA_root_ids.emplace(val->as<IterDomain>());
    }
  }

  const auto c2p_root_map = root_map.mapConsumerToProducer(
      consumer->domain(), producer->domain(), consumer_CA_root_ids);

  ConsumerForwardingInfo consumer_forwarding_info(producer, consumer);

  ProducerForwardingInfo producer_forwarding_info(producer);

  // Instead of replaying from the root, lets try to play forward the history
  // of producer if they match ops on consumer. Enforce if we modify an
  // rfactor axis that those ops must match.
  auto producer_replay = BestEffortReplay(
      producer->domain()->domain(),
      consumer_CA_ids,
      c2p_root_map,
      producer_forwarding_info.forwarding_map,
      consumer_forwarding_info.forwarding_map);

  producer_replay.addComplimentLeafIDs(
      producer_forwarding_info.forwarding_map,
      producer_forwarding_info.compliment_map);

  return producer_replay;
}

void BestEffortReplay::skipSwizzles(
    const std::unordered_map<IterDomain*, Expr*>& target_id2expr,
    const std::unordered_map<IterDomain*, Expr*>& replay_id2expr) {
  // Update target2replay map
  bool updated = true;

  while (updated) {
    updated = false;
    for (auto it : target2replay_id_map_) {
      if (isSwizzleInput(it.first, target_id2expr) ||
          isSwizzleInput(it.second, replay_id2expr)) {
        updated = true;
        auto new_target = getSwizzleFinalOutput(it.first, target_id2expr);
        auto new_replay = getSwizzleFinalOutput(it.second, replay_id2expr);

        // new_target and new_replay will now be the final output
        //  skipping all swizzles in between. We'd need to
        //  update the mapping and leaf ids to the final outputs.
        target2replay_id_map_.erase(it.first);
        TORCH_INTERNAL_ASSERT(
            target2replay_id_map_.insert(std::make_pair(new_target, new_replay))
                .second,
            "Unexpected replay leaf");
        // Progress the leaf ids if the replay is updated
        if (it.second != new_replay &&
            leaf_ids_.find(it.second) != leaf_ids_.end()) {
          leaf_ids_.erase(it.second);
          leaf_ids_[new_replay] = counter++;
        }
        break;
      }
    }
  }
}

DisjointSets<IterDomain*> BestEffortReplay::getDisjointSets() {
  DisjointSets<IterDomain*> result;
  const std::unordered_map<IterDomain*, IterDomain*>* maps[3] = {
      &target2replay_id_map_, &replay_forward_id_map_, &target_forward_id_map_};
  for (auto map : maps) {
    for (auto entry : *map) {
      result.mapEntries(entry.first, entry.second);
    }
  }
  return result;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
