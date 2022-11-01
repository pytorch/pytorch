#include <torch/csrc/jit/codegen/cuda/compute_at_map.h>

#include <torch/csrc/jit/codegen/cuda/disjoint_set.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

#include <tuple>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace {

// Is the provided IterDomain an Leaf of provided TensorView and within its
// computeAtPosition.
// If outside computeAt axis, we don't want to directly map consumer/producer in
// the loop mapping as they are not sharing the same loop.
bool idIsAComputeAtLeafDomain(IterDomain* id, TensorView* tv) {
  auto begin = tv->domain()->domain().begin();
  auto end = tv->domain()->domain().begin() + tv->getComputeAtPosition();
  return std::find(begin, end, id) != end;
}

// Is the provided IterDomain an Leaf of provided TensorView
bool idIsALeafDomain(IterDomain* id, TensorView* tv) {
  auto begin = tv->domain()->domain().begin();
  auto end = tv->domain()->domain().end();
  return std::find(begin, end, id) != end;
}

} // namespace

IterDomainGraph::IterDomainGraph(Fusion* fusion, bool allow_self_mapping) {
  build(fusion);

  if (!allow_self_mapping) {
    TORCH_INTERNAL_ASSERT(
        !hasSelfMapping(),
        "Unsupported domain mapping detected in ",
        std::get<0>(*self_mapping_info_)->toString(),
        ". ",
        std::get<3>(*self_mapping_info_),
        " domains, ",
        std::get<1>(*self_mapping_info_)->toString(),
        " and ",
        std::get<2>(*self_mapping_info_)->toString(),
        ", are mapped with each other.");
  }
}

//! Map corresponding inputs and outputs of swizzle op together
//!  on the given disjoint set, if the given id is an output
//!  of a swizzle operator.
//!
//! The current usage of swizzle operator is local to each tensor
//!  itself, so they should not affect exact or permissive mapping
//!  between iterdomains on different tensor domains.
//! TODO:
//!   Exact mapping based index hoisting of swizzled iterdomains
//!   is disabled currently and will be re-enabled in the next
//!   few build out steps.
void mapMaybeSwizzleOp(
    DisjointSets<IterDomain*>& disjoint_sets,
    IterDomain* id) {
  if (auto swizzle_2d = dynamic_cast<Swizzle2D*>(id->definition())) {
    // Map each input to its corresponding output on the given
    //  disjoint set.
    disjoint_sets.mapEntries(swizzle_2d->inX(), swizzle_2d->outX());
    disjoint_sets.mapEntries(swizzle_2d->inY(), swizzle_2d->outY());
  }
}

bool IterDomainGraph::exprsMap(
    Expr* first,
    Expr* second,
    bool forward,
    const DisjointSets<IterDomain*>& id_map) {
  if (first == nullptr || second == nullptr) {
    return false;
  }

  if (first->etype() != second->etype()) {
    return false;
  }

  TORCH_INTERNAL_ASSERT(
      first->etype() == ExprType::Merge || first->etype() == ExprType::Split,
      "Merge and split are the only expressions supported through rfactor operations in compute at map, but found:\n",
      first->toString());

  auto first_ids = ir_utils::filterByType<IterDomain>(
                       forward ? first->inputs() : first->outputs())
                       .vector();

  auto second_ids = ir_utils::filterByType<IterDomain>(
                        forward ? second->inputs() : second->outputs())
                        .vector();

  TORCH_INTERNAL_ASSERT(
      first_ids.size() == second_ids.size(),
      "Expected number of ",
      (forward ? "inputs" : "outputs"),
      " to match for\n",
      first->toString(),
      second->toString());

  {
    std::vector<std::pair<IterDomain*, IterDomain*>> zipped_ids;

    std::transform(
        first_ids.begin(),
        first_ids.end(),
        second_ids.begin(),
        std::back_inserter(zipped_ids),
        [](IterDomain* first, IterDomain* second) {
          return std::make_pair(first, second);
        });

    if (std::any_of(
            zipped_ids.begin(),
            zipped_ids.end(),
            [&](std::pair<IterDomain*, IterDomain*> id_pair) {
              return !id_map.strictAreMapped(id_pair.first, id_pair.second);
            })) {
      return false;
    }
  }

  if (first->isA<Merge>() && !forward) {
    // Can't back prop through merge without making sure one dimension actually
    // is identical extents.
    auto merge0 = first->as<Merge>();
    auto merge1 = second->as<Merge>();

    auto extent_0o = merge0->outer()->extent();
    auto extent_0i = merge0->inner()->extent();
    auto extent_1o = merge1->outer()->extent();
    auto extent_1i = merge1->inner()->extent();

    auto extent_0_match = extent_0o->sameAs(extent_1o) ||
        (extent_0o->isConstInt() && extent_1o->isConstInt() &&
         extent_0o->evaluateInt() == extent_1o->evaluateInt());

    auto extent_1_match = extent_0i->sameAs(extent_1i) ||
        (extent_0i->isConstInt() && extent_1i->isConstInt() &&
         extent_0i->evaluateInt() == extent_1i->evaluateInt());

    if (!(extent_0_match || extent_1_match)) {
      return false;
    }
  }

  if (first->isA<Split>()) {
    auto first_split = first->as<Split>();
    auto second_split = second->as<Split>();
    if (!first_split->factor()->sameAs(second_split->factor()) ||
        first_split->innerSplit() != second_split->innerSplit() ||
        !first_split->startOffset()->sameAs(second_split->startOffset()) ||
        !first_split->stopOffset()->sameAs(second_split->stopOffset())) {
      return false;
    }
  }

  return true;
}

// Given first and second Exprs "match"
//   Expr type matches
//   IterDomain's in the inputs and outputs exact match, (including argument
//     position positions)
//   Paramters like Split's factor "match" (exact match on integers could be
//     better, as today it will just check it's the same symbol or evaluated to
//     the same constant. However, we know all the extents of all the
//     IterDomain's that exact map with eachother are the same value.
void IterDomainGraph::mapThroughExpr(Expr* first, Expr* second, bool forward) {
  if (first == nullptr || second == nullptr) {
    return;
  }

  if (!exprsMap(first, second, forward, exact_nodes_)) {
    return;
  }

  auto first_ids = ir_utils::filterByType<IterDomain>(
                       forward ? first->outputs() : first->inputs())
                       .vector();
  auto second_ids = ir_utils::filterByType<IterDomain>(
                        forward ? second->outputs() : second->inputs())
                        .vector();
  TORCH_INTERNAL_ASSERT(
      first_ids.size() == second_ids.size(),
      "This should be unreachable, if transformation expressions match, their number of inputs and outputs should as well.\n However found:\n",
      first->toString(),
      "\nand\n",
      second->toString());
  for (auto out_i : c10::irange(first_ids.size())) {
    exact_nodes_.mapEntries(first_ids[out_i], second_ids[out_i]);
    permissive_nodes_.mapEntries(first_ids[out_i], second_ids[out_i]);
  }
}

namespace {

// Returns the first pair of id's in ids detected to match eachother on the
// permissive map of the ID graph. TODO: what this is really looking for is if
// there's any overlapping between the iter domains in the provided set.
//
// i.e. if we have:
// tv0 = arange(6).view({3, 2})
// tv1 = tv0[3, 2].t()
// tv2 = tv0[3, 2].view({2, 3})
// tv3 = tv1 + tv2
//
// Then we can see this overlap in the tv3 expression as:
//
// tv0 = { {0, 1, 2},
//         {3, 4, 5} }
//
// tv1 = { {0, 3},
//         {1, 4},
//         {2, 5} }
//
// tv2 = { {0, 1},
//         {2, 3},
//         {4, 5} }
//
// The elements in tv1 {3, 1, 4, 2}, map respectively to the elements in tv2 {1,
// 2, 3, 4}. The reason this is so important is it means that generating tv3 is
// no longer a trivially parallelizable problem (if we include the dag all the
// way to tv0). So tv0's axes cannot be inlined across both the tv0 and tv1
// path. This breaks some assumptions we have today in schedulers that will
// assume tv2 can be trivially inlined/parallelized. Instead we'd need to take
// into consideration the effective communication going on here, so that we pull
// multiple values of tv0 to compute tv3.
c10::optional<std::pair<IterDomain*, IterDomain*>> detectMappablePair(
    const std::vector<IterDomain*>& ids,
    const IterDomainGraph& id_graph,
    IdMappingMode mode) {
  for (auto id1 : ids) {
    for (auto id2 : ids) {
      if (id1 == id2) {
        continue;
      }
      if (mode == IdMappingMode::EXACT) {
        if (id_graph.exactNodes().disjointSetMap().at(id1)->has(id2)) {
          return std::make_pair(id1, id2);
        }
      } else if (mode == IdMappingMode::PERMISSIVE) {
        if (id_graph.permissiveNodes().disjointSetMap().at(id1)->has(id2)) {
          return std::make_pair(id1, id2);
        }
      } else if (mode == IdMappingMode::LOOP) {
        if (id_graph.loopNodes().disjointSetMap().at(id1)->has(id2)) {
          return std::make_pair(id1, id2);
        }
      } else {
        TORCH_INTERNAL_ASSERT(false, "Unrecognized IdMappingMode mode.");
      }
    }
  }

  return {};
}

// It is assumed that for any tensor represented by a list of domains,
// those domains should never be mapped with each other. It may be
// possible to lift this assumption, but it's unclear if it could
// matter in practice.
c10::optional<std::tuple<TensorView*, IterDomain*, IterDomain*, std::string>>
findFirstSelfMapping(Fusion* fusion, const IterDomainGraph& id_graph) {
  for (auto tv : ir_utils::allTvs(fusion)) {
    // For each tensor, make sure root, rfactor and leaf domains
    // should not include domains that are mapped with another domain
    // in the same set of domains. This may be overly conservative,
    // and it maybe enough to check the root domains.

    // Root domains
    auto self_mappped_root_pair =
        detectMappablePair(tv->getRootDomain(), id_graph, IdMappingMode::EXACT);
    if (self_mappped_root_pair.has_value()) {
      return std::make_tuple(
          tv,
          self_mappped_root_pair->first,
          self_mappped_root_pair->second,
          "Root");
    }

    // Rfactor domains
    if (tv->hasRFactor()) {
      auto self_mappped_rf_pair = detectMappablePair(
          tv->getRFactorDomain(), id_graph, IdMappingMode::EXACT);
      if (self_mappped_rf_pair.has_value()) {
        return std::make_tuple(
            tv,
            self_mappped_rf_pair->first,
            self_mappped_rf_pair->second,
            "RFactor");
      }
    }

    // Leaf domains
    auto self_mappped_leaf_pair = detectMappablePair(
        tv->domain()->domain(), id_graph, IdMappingMode::LOOP);
    if (self_mappped_leaf_pair.has_value()) {
      return std::make_tuple(
          tv,
          self_mappped_leaf_pair->first,
          self_mappped_leaf_pair->second,
          "Leaf");
    }
  }
  return c10::nullopt;
}

} // namespace

void IterDomainGraph::build(Fusion* fusion) {
  FusionGuard fg(fusion);

  // Initialize a node for every iteration domain
  for (auto tv : ir_utils::allTvs(fusion)) {
    const auto& domain = tv->domain()->domain();
    auto all_ids = ir_utils::allIDsOf(tv);

    // Check is this domain is a consumer of a view-like operation
    bool view_like_domain = tv->domain()->hasViewLikeRFactor();

    for (auto id : all_ids) {
      // Check if this id is a view like rfactor id
      bool is_view_rfactor_id = false;
      if (view_like_domain && id->isRFactorProduct()) {
        // If the tensor domain is a view like domain, and the iteration domain
        // is marked as an rfactor product and is in the rfactor domain, it's a
        // view like rfactor iteration domain
        const auto& rfactor_domain = tv->domain()->getMaybeRFactorDomain();
        if (std::find(rfactor_domain.begin(), rfactor_domain.end(), id) !=
            rfactor_domain.end()) {
          is_view_rfactor_id = true;
        }
      }
      bool is_leaf_id =
          std::find(domain.begin(), domain.end(), id) != domain.end();
      initializeId(id, is_view_rfactor_id, is_leaf_id);
    }
  }

  // All ID's are initialized, start connecting them on the permissive, exact,
  // and loop dimensions.

  for (auto expr : fusion->exprs()) {
    if (!ir_utils::isTvOp(expr)) {
      continue;
    }

    auto tv_outputs = ir_utils::filterByType<TensorView>(expr->outputs());
    TensorView* first_output_tv = nullptr;

    for (auto c_tv : tv_outputs) {
      if (first_output_tv == nullptr) {
        first_output_tv = c_tv;
      } else {
        // Map multi outputs of an expression to each other. c is current
        // output, and f as first output. Keep consistent with the later section
        // of producer and consumers. Which here producer is now "first output",
        // and consumer is still consumer. One exception is how the
        // domains left of CA positions are handled in the Parallel
        // map. Those domains are not mapped in producer and consumer
        // mappings as they do not share loops, but are mapped in the
        // case of mapping multiple outputs since they do share the
        // same loops.

        TORCH_INTERNAL_ASSERT(
            c_tv->getRootDomain().size() ==
                first_output_tv->getRootDomain().size(),
            "Multiple outputs with mismatched dimensions is not supported. ",
            "Only supported case is welford op where all outputs tvs have idential domains.");
        // p->f, c->c
        std::unordered_map<IterDomain*, IterDomain*> c2f_root_map;
        for (const auto i :
             c10::irange(first_output_tv->getRootDomain().size())) {
          c2f_root_map.insert(std::make_pair(
              c_tv->getRootDomain()[i], first_output_tv->getRootDomain()[i]));
        }

        // Multi output mapping, outputs are required to have the same domain
        // and same transformations, so they can be mapped in permissive/exact,
        // and when within compute at position of domain()->domain() in the
        // parallel map.
        auto replay_FasC = BestEffortReplay(
            first_output_tv->domain()->domain(),
            c_tv->domain()->domain(),
            c2f_root_map);

        auto c2f_map = replay_FasC.getReplay();

        // Map the entire replay map between the multiple
        // consumers even for the Parallel map as they share the same
        // loop.
        for (auto c_id : getSortedKeys(c2f_map, Statement::lessThan)) {
          auto f_id = c2f_map.at(c_id);
          // Map the id's together
          permissive_nodes_.mapEntries(f_id, c_id);
          exact_nodes_.mapEntries(f_id, c_id);
          if (idIsALeafDomain(f_id, first_output_tv)) {
            loop_nodes_.mapEntries(f_id, c_id);
          }
          sibling_sets_.mapEntries(f_id, c_id);
        }
      }

      auto tv_inputs = ir_utils::filterByType<TensorView>(expr->inputs());

      for (auto p_tv : tv_inputs) {
        auto pairwise_map = PairwiseRootDomainMap(p_tv, c_tv);

        // Look for matching ID transformations in producer and consumer, replay
        // producer as consumer. We use the symmetric API of BestEffortReplay so
        // that both broadcast and squeeze are handled correctly.
        const auto permissive_disjoint_sets =
            BestEffortReplay::replayPasC(p_tv, c_tv, -1, pairwise_map)
                .getIterDomainEquivalence();

        // For exact mapings do not map any broadcast dimensions to
        // non-broadcast dimensions. Prevent any broadcasted axes being mapped
        // to non-broadcasted axes.
        auto exact_c2p_root_map =
            PairwiseRootDomainMap(p_tv, c_tv, true)
                .mapConsumerToProducer(c_tv->domain(), p_tv->domain());

        // Same as permissive above but for exact
        auto exact_replay_PasC = BestEffortReplay(
            p_tv->domain()->domain(),
            c_tv->domain()->domain(),
            exact_c2p_root_map);

        const auto& exact_c2p_map = exact_replay_PasC.getReplay();

        for (auto c_id : getSortedKeys(exact_c2p_map, Statement::lessThan)) {
          auto p_id = exact_c2p_map.at(c_id);
          exact_nodes_.mapEntries(c_id, p_id);
          consumers_.at(p_id).pushBack(c_id);
          producers_.at(c_id).pushBack(p_id);

          // Add the swizzle inputs to the same
          //  disjoint set as well if either c_id
          //  or p_id is swizzle output.
          mapMaybeSwizzleOp(exact_nodes_, p_id);
          mapMaybeSwizzleOp(exact_nodes_, c_id);
        }

        auto p_ids_vec = ir_utils::allIDsOf(p_tv);
        auto c_ids_vec = ir_utils::allIDsOf(c_tv);
        std::unordered_set<IterDomain*> p_ids(
            p_ids_vec.begin(), p_ids_vec.end());
        std::unordered_set<IterDomain*> c_ids(
            c_ids_vec.begin(), c_ids_vec.end());

        for (auto& dset : permissive_disjoint_sets.disjointSets()) {
          auto& vec = dset->vector();
          for (auto i : c10::irange(vec.size())) {
            auto id1 = vec[i];
            permissive_nodes_.mapEntries(id1, vec[0]);

            // Add the swizzle inputs to the same
            //  disjoint set as well if either c_id
            //  or p_id is swizzle output.
            mapMaybeSwizzleOp(permissive_nodes_, id1);

            for (auto j : c10::irange(i + 1, vec.size())) {
              auto id2 = vec[j];
              if (p_ids.count(id1) && c_ids.count(id2)) {
                consumers_.at(id1).pushBack(id2);
                producers_.at(id2).pushBack(id1);
                if (idIsAComputeAtLeafDomain(id1, p_tv) &&
                    idIsALeafDomain(id2, c_tv)) {
                  loop_nodes_.mapEntries(id1, id2);
                }
              }
              if (c_ids.count(id1) && p_ids.count(id2)) {
                producers_.at(id1).pushBack(id2);
                consumers_.at(id2).pushBack(id1);
                if (idIsAComputeAtLeafDomain(id2, p_tv) &&
                    idIsALeafDomain(id1, c_tv)) {
                  loop_nodes_.mapEntries(id1, id2);
                }
              }
            }
          }
        }
      }
    }
  }

  // Explicitly map through rfactor transformations, if we have an op like:
  //
  // T1[x, y*z] = view(T0[x*y, z])
  // T3[x, y*z] = view(T2[x*y, z])
  // T4 = T0 + T2
  //
  // We want to map T1 and T3's rfactor transformations together by playing the
  // transformations forward since their root domains map. If instead we have:
  //
  // T1[x, y*z] = view(T0[x*y, z])
  // T3[x, y*z] = view(T2[x*y, z])
  // T4 = T1 + T3
  //
  // Then we wouldn't have a mapping of T1 and T3's root domain, we'd have a
  // mapping of their rfactor domain, so we would want to map T1 and T3's
  // rfactor transformations starting at their rfactor domains.
  //
  // Therefore we'll explicitly map rfactor transformation iteration domains
  // forward and backwards. Something similar could happen with rfactor of root
  // domains, though it seems mapping rfactor reduction domains aren't that
  // important. Mapping view transformations is more important since view is
  // part of the compute definition so having the map through the
  // transformations makes it easy to check if different view operations are
  // consistent with eachother.

  auto all_tvs = ir_utils::allTvs(fusion);
  std::vector<TensorView*> all_consumer_tvs;
  std::copy_if(
      all_tvs.begin(),
      all_tvs.end(),
      std::back_inserter(all_consumer_tvs),
      [](TensorView* tv) { return !tv->isFusionInput() && tv->hasRFactor(); });

  // IterDomains could have multiple uses defined in the fusion if multiple
  // transformations were redefined (more than one transform propagation pass
  // was run and retransformed sections of the graph). We're going to make a new
  // uses map so we can easily process the actual uses of IterDomains. We
  // actually only need rfactor uses for this section of mapping, so we'll limit
  // this map to only rfactor transformations.
  std::unordered_map<IterDomain*, Expr*> rfactor_id_uses;

  // Order of traversal is important for processing all the rfactor ids as the
  // first pass will go forward through expressions and the second pass will
  // traverse backwards through them. ID's will be unique in this vector,
  // enforced when building it since it's built with rfactor_id_uses.
  std::vector<IterDomain*> rfactor_id_order;

  // Grab all the rfactor ids.
  for (auto consumer_tv : all_consumer_tvs) {
    auto exprs = StmtSort::getExprs(
        fusion,
        {consumer_tv->getMaybeRFactorDomain().begin(),
         consumer_tv->getMaybeRFactorDomain().end()});
    for (auto expr : exprs) {
      auto rfactor_inp_ids = ir_utils::filterByType<IterDomain>(expr->inputs());
      TORCH_INTERNAL_ASSERT(
          expr->isA<Split>() || expr->isA<Merge>(),
          "Wasn't expecting the expression type of:\n",
          expr->toString(),
          "\nto be an expression defined in an rfactor transformation.");
      for (auto rfactor_inp_id : rfactor_inp_ids) {
        TORCH_INTERNAL_ASSERT(
            rfactor_id_uses.find(rfactor_inp_id) == rfactor_id_uses.end(),
            "Was expecting iter domains to only have one active transformation but found id ",
            rfactor_inp_id->toString(),
            " used in\n",
            rfactor_id_uses.at(rfactor_inp_id),
            "\nand\n",
            expr->toString());
        rfactor_id_uses.emplace(std::make_pair(rfactor_inp_id, expr));
        rfactor_id_order.push_back(rfactor_inp_id);
      }
    }
    for (auto rfactor_id : consumer_tv->getMaybeRFactorDomain()) {
      if (rfactor_id->isRFactorProduct()) {
        rfactor_id_uses.emplace(std::make_pair(rfactor_id, nullptr));
        rfactor_id_order.push_back(rfactor_id);
      }
    }
  }

  // if prop_forward we're going forward through transformations and
  // expressions, meaning if inputs of expressions map then we map their
  // outputs, otherwise we're traversing backwards, meaning if outputs of
  // expressions map then we map their inputs.
  for (auto prop_forward : {true, false}) {
    std::unordered_set<Expr*> visited_exprs;

    for (auto rfactor_id_i : c10::irange(rfactor_id_order.size())) {
      auto first_rfactor_id = prop_forward
          ? rfactor_id_order[rfactor_id_i]
          : rfactor_id_order[rfactor_id_order.size() - 1 - rfactor_id_i];

      // At should be safe since we made rfactor_id_order and rfactor_id_uses at
      // the same time so they should have the same exact entries.
      auto first_expr = prop_forward ? rfactor_id_uses.at(first_rfactor_id)
                                     : first_rfactor_id->definition();

      if (first_expr == nullptr) {
        continue;
      }

      if (visited_exprs.find(first_expr) != visited_exprs.end()) {
        continue;
      }
      visited_exprs.emplace(first_expr);

      // Only need to be concerned here with mapping across rfactor iter
      // domains, so isolate out those.
      auto all_exact_map_ids = exact_nodes_.getDisjointSetOf(first_rfactor_id);
      std::vector<IterDomain*> exact_map_rf_ids;
      std::copy_if(
          all_exact_map_ids.vector().begin(),
          all_exact_map_ids.vector().end(),
          std::back_inserter(exact_map_rf_ids),
          [](IterDomain* id) { return id->isRFactorProduct(); });

      for (auto exact_map_rf_id : exact_map_rf_ids) {
        if (exact_map_rf_id == first_rfactor_id) {
          continue;
        }
        // If there's an input with an rfactor domain we could have an exact
        // mapped rfactor id that's on the input meaning it wouldn't have an
        // entry in rfactor_id_uses
        auto other_use =
            rfactor_id_uses.find(exact_map_rf_id) == rfactor_id_uses.end()
            ? nullptr
            : rfactor_id_uses.at(exact_map_rf_id);
        auto other_expr =
            prop_forward ? other_use : exact_map_rf_id->definition();

        if (other_expr == nullptr) {
          continue;
        }

        if (visited_exprs.find(other_expr) != visited_exprs.end()) {
          continue;
        }

        mapThroughExpr(first_expr, other_expr, prop_forward);
      }
    }
  }

  // Build almost exact map by forwarding through broadcast axes
  almost_exact_nodes_ = exact_nodes_;
  std::unordered_set<Expr*> visited;
  auto all_elements = exact_nodes_.getAllElements();
  for (auto entry : all_elements.vector()) {
    if (entry->definition() == nullptr) {
      continue;
    }
    auto def = entry->definition();
    if (!visited.emplace(def).second) {
      continue;
    }
    if (auto merge = dynamic_cast<Merge*>(def)) {
      if (merge->inner()->extent()->isOneInt()) {
        almost_exact_nodes_.mapEntries(merge->outer(), merge->out());
      }
      if (merge->outer()->extent()->isOneInt()) {
        almost_exact_nodes_.mapEntries(merge->inner(), merge->out());
      }
    } else if (auto split = dynamic_cast<Split*>(def)) {
      if (split->factor()->isOneInt() && split->startOffset()->isZeroInt() &&
          split->stopOffset()->isZeroInt()) {
        if (split->innerSplit()) {
          almost_exact_nodes_.mapEntries(split->in(), split->outer());
        } else {
          almost_exact_nodes_.mapEntries(split->in(), split->inner());
        }
      }
    }
  }

  self_mapping_info_ = findFirstSelfMapping(fusion, *this);
}

void IterDomainGraph::initializeId(
    IterDomain* id,
    bool is_view_rfactor_id,
    bool is_leaf_id) {
  permissive_nodes_.initializeSet(id);
  exact_nodes_.initializeSet(id);
  if (is_leaf_id) {
    loop_nodes_.initializeSet(id);
  }
  consumers_[id] = {};
  producers_[id] = {};
  sibling_sets_.initializeSet(id);

  all_ids_.pushBack(id);

  if (is_view_rfactor_id) {
    view_rfactor_ids_.emplace(id);
  }
}

ComputeAtMap::ComputeAtMap(Fusion* fusion)
    : id_graph_(fusion), concretized_bcasts_(fusion), fusion_(fusion) {
  build(fusion);
}

void ComputeAtMap::build(Fusion* fusion) {
  buildUniqueExactExprMaps();
  buildConcreteIds();
  buildUniqueExactExprMaps();
}

void ComputeAtMap::validateAndPropagatePType() {
  for (const auto& loop_disjoint_set : id_graph_.loopNodes().disjointSets()) {
    ParallelType common_ptype = ParallelType::Serial;
    for (auto id : loop_disjoint_set->vector()) {
      auto id_ptype = id->getParallelType();
      TORCH_INTERNAL_ASSERT(
          id_ptype == common_ptype || id_ptype == ParallelType::Serial ||
              common_ptype == ParallelType::Serial,
          "Issue validating parallel type disjoint ptype is, ",
          common_ptype,
          " but found in the set the id: ",
          id->toString());
      common_ptype =
          common_ptype == ParallelType::Serial ? id_ptype : common_ptype;
    }

    for (auto id : loop_disjoint_set->vector()) {
      id->parallelize(common_ptype);
    }
  }
}

void ComputeAtMap::allocateIndexVariables() {
  // Run through all disjoint sets registered in loop map,
  //  all lowered kir::ForLoop will correspond to one of the disjoint sets
  //  and we only need one index variable for each set.
  for (const auto& loop_disjoint_set : id_graph_.loopNodes().disjointSets()) {
    ParallelType ptype;
    // first allocate thread and grid parallel indices:
    //  The validation pass will check that the parallel bindings within the
    //  loop nodes are consistent so all the loops within this disjoint set
    //  will be realized implicitly using parallel index variables.
    if (std::any_of(
            loop_disjoint_set->vector().begin(),
            loop_disjoint_set->vector().end(),
            [&ptype](IterDomain* id) {
              if (id->isThread() &&
                  // Halo extended parallel loops currently are handled
                  // differently and an index variable would still
                  // be allocated in this case.
                  (GpuLower::current()->haloInfo()->getExtent(id) == nullptr)) {
                ptype = id->getParallelType();
                return true;
              }
              return false;
            })) {
      loop_index_variable_map_[loop_disjoint_set.get()] =
          NamedScalar::getParallelIndex(ptype);
      continue;
    }

    // All loops in this set are non-parallel, non-concretized broadcast
    //  iterdomains, their "index variable" should be zero.
    if (std::all_of(
            loop_disjoint_set->vector().begin(),
            loop_disjoint_set->vector().end(),
            [](IterDomain* id) { return id->isBroadcast(); })) {
      loop_index_variable_map_[loop_disjoint_set.get()] = fusion_->zeroVal();
      continue;
    }

    // Allocate variable for the iterdomains:
    auto concrete_loop_id_it = concrete_id_cache_.find(loop_disjoint_set);
    TORCH_INTERNAL_ASSERT(
        concrete_loop_id_it != concrete_id_cache_.end(),
        "Concrete id not computed");

    auto concrete_loop_id = concrete_loop_id_it->second;

    // Need to allocate double buffered loop differently.
    if (GpuLower::current()->doubleBufferInfo().isDoubleBufferedIterDomain(
            concrete_loop_id)) {
      // Allocate index variable for each stage of the double buffered loop.
      double_buffered_loop_index_variable_map_[loop_disjoint_set.get()] =
          std::make_unique<DoubleBufferIndices>(DoubleBufferIndices(
              {{DoubleBufferLoopStage::Prolog,
                IrBuilder::create<Int>(c10::nullopt)},
               {DoubleBufferLoopStage::Main,
                IrBuilder::create<Int>(c10::nullopt)},
               {DoubleBufferLoopStage::Epilog,
                IrBuilder::create<Int>(c10::nullopt)}}));
    } else {
      // Everything now should be serial concrete loops,
      //   we just allocate a loop index integer for each set of loops.
      loop_index_variable_map_[loop_disjoint_set.get()] =
          IrBuilder::create<Int>(c10::nullopt);
    }
  }
}

Val* ComputeAtMap::getIndexVariable(
    IterDomain* id,
    DoubleBufferLoopStage double_buffer_loop_stage) const {
  TORCH_INTERNAL_ASSERT(
      id_graph_.loopNodes().mappingExists(id),
      "Index Variable: no index variable allocated as ",
      id->toString(),
      " is not registered in loop map");
  const auto* loop_set = &(id_graph_.loopNodes().getDisjointSetOf(id));

  // Check if this loop was modified by double buffer pass.
  bool is_double_buffer_iterdomain =
      GpuLower::current()->doubleBufferInfo().isDoubleBufferedIterDomain(id);

  if (is_double_buffer_iterdomain) {
    // Use dedicated double buffer index variable if the loop is double buffer
    // loop
    if (double_buffer_loop_stage == DoubleBufferLoopStage::NotApplicable) {
      // The double buffered loop stages are created after the loop nest
      //  lowering phase so this function will be querried before the double
      //  buffer pass. At that point, no forloop has any double buffer
      //  stage defined, and we just default to using the main stage index.
      double_buffer_loop_stage = DoubleBufferLoopStage::Main;
    }
    return double_buffered_loop_index_variable_map_.at(loop_set)->at(
        double_buffer_loop_stage);
  } else {
    return loop_index_variable_map_.at(loop_set);
  }
}

bool ComputeAtMap::areMapped(
    IterDomain* id0,
    IterDomain* id1,
    IdMappingMode mode) const {
  return disjointSetOf(id0, mode)->has(id1);
}

IterDomain* ComputeAtMap::computeConcreteId(
    IterDomain* id,
    IdMappingMode mode) {
  const auto& disjoint_set_shared_ptr = disjointSetOf(id, mode);

  TORCH_INTERNAL_ASSERT(
      disjoint_set_shared_ptr->vector().size(),
      "Empty disjoint set found for ",
      id->toString());

  if (disjoint_set_shared_ptr->vector().size() == 1) {
    // If only one entry in the disjoint set, by definition the existing ID has
    // to be the concrete ID.
    return disjoint_set_shared_ptr->vector().front();
  }

  // Grab a set of candidate concrete_ids, we track towards the consumers in the
  // ID group as one of those is guaranteed to be a valid concrete id.
  VectorOfUniqueEntries<IterDomain*> maybe_concrete_ids;
  for (auto id : disjoint_set_shared_ptr->vector()) {
    bool id_output = true;
    for (auto consumer_id : id_graph_.consumers().at(id).vector()) {
      if (disjoint_set_shared_ptr->has(consumer_id)) {
        id_output = false;
        break;
      }
    }
    if (id_output) {
      maybe_concrete_ids.pushBack(id);
    }
  }

  // Shouldn't ever happen, it would mean there's an error somewhere in the
  // graph.
  TORCH_INTERNAL_ASSERT(
      maybe_concrete_ids.vector().size() > 0,
      "No potential concrete_id's found for ",
      id->toString());

  if (maybe_concrete_ids.vector().size() == 1) {
    return maybe_concrete_ids.vector().front();
  }

  // Broadcast resolution is what we have to figure out here. So if we traverse
  // back from leaves to rfactor inputs through the exact map, if there's an
  // operation with a broadcast input that's resolved within the history all of
  // the domains in all of the maybe_rfactor_ids, then the concrete ID must
  // resolve that broadcast.
  //
  // (1) Compute "traversed IDs" which is every exact disjoint set starting at
  // all maybe concrete ID's traversing back through exact map.
  //
  // (2) Check all broadcast sets, remove from "traversed IDs" any broadcast set
  // that has its broadcast resolved ID within "traversed IDs", and all
  // IterDomains dependant on that broadcast.
  //
  // (3) Start at all "traversed IDs" set that has an rfactor domain, traverse
  // backwards to inputs and remove every exact ID set from "traversed IDs".
  //
  // Remove (2) and (3) from (1) and we have the iteration domains we must
  // resolve. The concrete ID must be in that set.
  //
  // Find any maybe concrete ID through the same iter/broadcast counting as
  // before as it should work fine.

  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      maybe_concrete_exact_sets;

  for (auto maybe_concrete_id : maybe_concrete_ids) {
    maybe_concrete_exact_sets.pushBack(
        disjointSetOf(maybe_concrete_id, IdMappingMode::EXACT));
  }

  // Going to iteratively modify this to be all sets that the concrete ID needs
  // to cover
  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      all_exact_sets_covered =
          getAllDisjointSetProducers(maybe_concrete_exact_sets);

  // Remove all broadcast domains that are resolved within the history of any of
  // the maybe concrete sets.
  {
    // All broadcast exact sets in all_exact_sets_covered that are resolved by
    // IterDomains in all_exact_sets_covered
    VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
        resolved_broadcasts;

    for (auto exact_set : all_exact_sets_covered) {
      TORCH_INTERNAL_ASSERT(
          exact_set->vector().size(),
          "Cannot compute concrete id of empty set.");
      auto c_id = getConcreteMappedID(
          exact_set->vector().front(), IdMappingMode::EXACT);

      if (!c_id->isBroadcast()) {
        continue;
      }

      bool concretized_in_group = false;
      for (auto bcast_id : exact_set->vector()) {
        auto concretized_ids =
            concretized_bcasts_.allConcretizedDomains(bcast_id);
        for (auto concretized_id : concretized_ids) {
          if (all_exact_sets_covered.has(
                  disjointSetOf(concretized_id, IdMappingMode::EXACT))) {
            concretized_in_group = true;
            break;
          }
        }
        if (concretized_in_group) {
          break;
        }
      }

      if (concretized_in_group) {
        resolved_broadcasts.pushBack(exact_set);
      }
    }

    // Need to remove all uses of broadcast dims that are resolved in this
    // group, and all their uses.
    auto all_resolved_broadcast_uses =
        getAllDisjointSetConsumers(resolved_broadcasts);

    // Remove broadcast resolved sets from all_exact_sets_covered by effectively
    // doing an inplace copy_if
    VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
        tmp_all_exact_sets_covered;
    std::swap(tmp_all_exact_sets_covered, all_exact_sets_covered);
    for (auto entry : tmp_all_exact_sets_covered) {
      if (all_resolved_broadcast_uses.has(entry)) {
        continue;
      }
      all_exact_sets_covered.pushBack(entry);
    }
  }

  // Remove all domains in the history of sets marked as rfactor.
  {
    // All exact sets in the history of an rfactored domain
    VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
        produces_rfactor_dom;
    for (auto exact_set : all_exact_sets_covered) {
      if (produces_rfactor_dom.has(exact_set)) {
        // Already processed
        continue;
      }
      if (std::none_of(
              exact_set->vector().begin(),
              exact_set->vector().end(),
              [&](IterDomain* id) { return isViewRfactor(id); })) {
        continue;
      }
      VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
          rfactor_history = getAllDisjointSetProducers({exact_set});
      for (auto entry : rfactor_history) {
        // Leave rfactor exact set, unless it's in the history of another
        // rfactor domain.
        if (entry != exact_set) {
          produces_rfactor_dom.pushBack(entry);
        }
      }
    }

    // Remove all sets in rfactor history from all_exact_sets_covered by
    // effectively doing an inplace copy_if
    VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
        tmp_all_exact_sets_covered;
    std::swap(tmp_all_exact_sets_covered, all_exact_sets_covered);
    for (auto entry : tmp_all_exact_sets_covered) {
      if (produces_rfactor_dom.has(entry)) {
        continue;
      }
      all_exact_sets_covered.pushBack(entry);
    }
  }

  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      input_ids;

  {
    // Remove any concrete id that's not still in all_exact_sets_covered,
    // basically copy_if
    decltype(maybe_concrete_ids) tmp_maybe_concrete_ids;
    std::swap(maybe_concrete_ids, tmp_maybe_concrete_ids);
    for (auto entry : tmp_maybe_concrete_ids) {
      if (all_exact_sets_covered.has(
              disjointSetOf(entry, IdMappingMode::EXACT))) {
        maybe_concrete_ids.pushBack(entry);
      }
    }
  }

  TORCH_INTERNAL_ASSERT(
      maybe_concrete_ids.vector().size() > 0,
      "No potential concrete_id's found for disjoint set ",
      disjoint_set_shared_ptr->toString());

  if (maybe_concrete_ids.vector().size() == 1) {
    return maybe_concrete_ids.vector().front();
  }

  // The concrete_id should have the most roots it can trace back to that are
  // iter domains, (non-broadcast/non-reduction). We don't trace back through
  // view operations, so the one with the most iter root domains is the concrete
  // ID.
  IterDomain* concrete_id = nullptr;
  int max_iter_root_count = 0;
  int max_bcast_root_count = 0;

  for (auto maybe_concrete_id : maybe_concrete_ids.vector()) {
    auto concrete_id_root_sets = getInputDisjointSetsOf(maybe_concrete_id);

    int bcast_root_count = std::count_if(
        concrete_id_root_sets.vector().begin(),
        concrete_id_root_sets.vector().end(),
        [&](std::shared_ptr<VectorOfUniqueEntries<IterDomain*>> set) {
          return set->vector()[0]->isBroadcast();
        });

    int iter_root_count =
        (int)concrete_id_root_sets.vector().size() - bcast_root_count;
    if (iter_root_count > max_iter_root_count ||
        (iter_root_count == max_iter_root_count &&
         bcast_root_count > max_bcast_root_count)) {
      max_iter_root_count = iter_root_count;
      max_bcast_root_count = bcast_root_count;
      concrete_id = maybe_concrete_id;
    }
  }

  TORCH_INTERNAL_ASSERT(
      concrete_id != nullptr,
      "No concrete_id found for disjoint set ",
      disjoint_set_shared_ptr->toString());

  return concrete_id;
}

void ComputeAtMap::buildConcreteIds() {
  // For the exact map just select the first ID since they're all exactly the
  // same size, it doesn't matter which is selected. This should be run-to-run
  // deterministic but which ID gets selected her depends on the traversal order
  // generating the set (compute at map build).
  for (const auto& disjoint_set_shared_ptr :
       id_graph_.exactNodes().disjointSets()) {
    TORCH_INTERNAL_ASSERT(
        disjoint_set_shared_ptr->vector().size(),
        "Cannot compute concrete id of empty set.");
    auto first_id = disjoint_set_shared_ptr->vector().front();
    concrete_id_cache_[disjoint_set_shared_ptr] = first_id;
  }

  // The following two algorithms seem quite wasteful. Should find a more
  // efficient way to compute concrete IDs.
  for (const auto& disjoint_set_shared_ptr :
       id_graph_.permissiveNodes().disjointSets()) {
    TORCH_INTERNAL_ASSERT(
        disjoint_set_shared_ptr->vector().size(),
        "Cannot compute concrete id of empty set.");
    auto first_id = disjoint_set_shared_ptr->vector().front();
    auto concrete_id = computeConcreteId(first_id, IdMappingMode::PERMISSIVE);
    concrete_id_cache_[disjoint_set_shared_ptr] = concrete_id;
  }

  // Same as exact computation
  for (const auto& disjoint_set_shared_ptr :
       id_graph_.almostExactNodes().disjointSets()) {
    TORCH_INTERNAL_ASSERT(
        disjoint_set_shared_ptr->vector().size(),
        "Cannot compute concrete id of empty set.");
    auto first_id = disjoint_set_shared_ptr->vector().front();
    auto concrete_id = computeConcreteId(first_id, IdMappingMode::ALMOSTEXACT);
    concrete_id_cache_[disjoint_set_shared_ptr] = concrete_id;
  }

  for (const auto& disjoint_set_shared_ptr :
       id_graph_.loopNodes().disjointSets()) {
    TORCH_INTERNAL_ASSERT(
        disjoint_set_shared_ptr->vector().size(),
        "Cannot compute concrete id of empty set.");
    auto first_id = disjoint_set_shared_ptr->vector().front();
    auto concrete_id = computeConcreteId(first_id, IdMappingMode::LOOP);
    concrete_id_cache_[disjoint_set_shared_ptr] = concrete_id;
  }
}

bool ComputeAtMap::areExactExprs(Expr* expr_1, Expr* expr_2) {
  if (expr_1->getExprType() != expr_2->getExprType()) {
    return false;
  }

  if (expr_1->isA<Swizzle2D>()) {
    auto swizzle_1 = expr_1->as<Swizzle2D>();
    auto swizzle_2 = expr_2->as<Swizzle2D>();
    if (swizzle_1->swizzleType() != swizzle_2->swizzleType() ||
        swizzle_1->swizzleMode() != swizzle_2->swizzleMode()) {
      return false;
    }
  }

  TORCH_INTERNAL_ASSERT(
      expr_1->inputs().size() == expr_2->inputs().size() &&
          expr_1->outputs().size() == expr_2->outputs().size(),
      "Expr traversal doesn't support variable number of inputs and outputs.");

  for (auto input_i : c10::irange(expr_1->inputs().size())) {
    if (expr_1->inputs()[input_i]->isA<IterDomain>() &&
        !areMapped(
            expr_1->inputs()[input_i]->as<IterDomain>(),
            expr_2->inputs()[input_i]->as<IterDomain>(),
            IdMappingMode::EXACT)) {
      // Inputs don't exact map in the right order
      return false;
    }
  }

  for (auto output_i : c10::irange(expr_1->outputs().size())) {
    if (expr_1->outputs()[output_i]->isA<IterDomain>() &&
        !areMapped(
            expr_1->outputs()[output_i]->as<IterDomain>(),
            expr_2->outputs()[output_i]->as<IterDomain>(),
            IdMappingMode::EXACT)) {
      // Outputs don't exact map in the right order
      return false;
    }
  }
  // Expr's have exact mapped inputs and outputs, including parameters of the
  // transformation.
  return true;
}

void ComputeAtMap::buildUniqueExactExprMaps() {
  // Start by building definitions
  for (const auto& disjoint_set_shared_ptr :
       id_graph_.exactNodes().disjointSets()) {
    std::vector<Expr*> definitions;

    // N^2 in number of unique transformations, this might be better to do
    // when generating the map.
    for (auto id : disjoint_set_shared_ptr->vector()) {
      if (id->definition() != nullptr) {
        auto id_inputs =
            ir_utils::filterByType<IterDomain>(id->definition()->inputs());
        if (std::any_of(id_inputs.begin(), id_inputs.end(), [&](auto id_input) {
              return disjoint_set_shared_ptr->has(id_input);
            })) {
          // Definition to this exact map, shouldn't be marked as a definition
          // to traverse on the exact map.

          // This is a WAR for FusionSimpleSwizzle2_CUDA wher there is a pattern
          // like:
          //
          // tv0[32, 32]
          // tv0->swizzle(Swizzle2DType::ZShape, 0, 1);
          //
          // each root domain is exact mapped with the outputs of the swizzle.
          // So the pre and post swizzle ID is in an exact set, but that exact
          // set also has the swizzle as a definition that leads to itself.
          //
          // TODO: Try to formalize this better in the exact ID traversal. Right
          // now its just interfering with concrete ID detection.
          continue;
        }
        bool match = false;
        for (auto recorded_def : definitions) {
          if (areExactExprs(id->definition(), recorded_def)) {
            match = true;
            break;
          }
        }
        if (!match) {
          definitions.push_back(id->definition());
        }
      }
    }
    unique_exact_definitions_[disjoint_set_shared_ptr] = definitions;
  }

  // Use definitions to build uses
  for (const auto& disjoint_set_shared_ptr :
       id_graph_.exactNodes().disjointSets()) {
    // Make sure uses is always initialized even there are no uses.
    if (unique_exact_uses_.find(disjoint_set_shared_ptr) ==
        unique_exact_uses_.end()) {
      unique_exact_uses_[disjoint_set_shared_ptr] = {};
    }

    auto definition_it =
        unique_exact_definitions_.find(disjoint_set_shared_ptr);

    if (definition_it == unique_exact_definitions_.end()) {
      continue;
    }

    const auto& definitions = definition_it->second;

    for (auto definition : definitions) {
      auto inp_ids = ir_utils::filterByType<IterDomain>(definition->inputs());
      for (auto inp : inp_ids) {
        auto inp_disjoint_set_shared_ptr =
            disjointSetOf(inp, IdMappingMode::EXACT);
        // Initialize uses entry
        if (unique_exact_uses_.find(inp_disjoint_set_shared_ptr) ==
            unique_exact_uses_.end()) {
          unique_exact_uses_[inp_disjoint_set_shared_ptr] = {};
        }

        auto& uses = unique_exact_uses_.at(inp_disjoint_set_shared_ptr);

        bool already_added = false;
        for (auto other_use : uses) {
          if (areExactExprs(definition, other_use)) {
            already_added = true;
            break;
          }
        }
        if (already_added) {
          continue;
        }

        if (!already_added) {
          uses.push_back(definition);
        }
      }
    }
  }
}

IterDomain* ComputeAtMap::getConcreteMappedID(
    IterDomain* id,
    IdMappingMode mode) const {
  auto disjoint_set_shared_ptr = disjointSetOf(id, mode);

  TORCH_INTERNAL_ASSERT(
      disjoint_set_shared_ptr->vector().size() > 0,
      "Empty disjoint set found for ",
      id->toString());

  auto cache_it = concrete_id_cache_.find(disjoint_set_shared_ptr);

  TORCH_INTERNAL_ASSERT(
      cache_it != concrete_id_cache_.end(),
      "Could not find concrete id for: ",
      id->toString(),
      " with mode ",
      mode);

  return cache_it->second;
}

namespace {

std::string idGraphNodesToString(
    const ComputeAtMap& ca_map,
    IdMappingMode mode) {
  std::stringstream ss;
  // Sort vectors before printing so that the resulting output is
  // printed deterministically
  auto disjoint_sets = ca_map.getIdSets(mode).disjointSets();
  std::sort(
      disjoint_sets.begin(),
      disjoint_sets.end(),
      [&](const auto& set1, const auto& set2) {
        if (set1->empty()) {
          return true;
        } else if (set2->empty()) {
          return false;
        } else {
          auto concrete_id1 = ca_map.getConcreteMappedID(set1->front(), mode);
          auto concrete_id2 = ca_map.getConcreteMappedID(set2->front(), mode);
          return Statement::lessThan(concrete_id1, concrete_id2);
        }
      });
  for (const auto& s_ptr : disjoint_sets) {
    const auto& set = *s_ptr;
    IterDomain* concrete_id = nullptr;
    if (!set.empty()) {
      auto id = set.front();
      concrete_id = ca_map.getConcreteMappedID(id, mode);
    }
    ss << "  {";
    for (auto entry : set.vector()) {
      ss << abstractToString(entry);
      if (entry == concrete_id) {
        ss << "*";
      }
      if (entry != set.back()) {
        ss << "; ";
      }
    }
    ss << " }\n";
  }
  return ss.str();
}

} // namespace

std::string ComputeAtMap::toString() const {
  std::stringstream ss;
  ss << "Compute at map { \n";
  ss << "Exact map:\n" << idGraphNodesToString(*this, IdMappingMode::EXACT);
  ss << "Almost Exact map:\n"
     << idGraphNodesToString(*this, IdMappingMode::ALMOSTEXACT);
  ss << "Loop map:\n" << idGraphNodesToString(*this, IdMappingMode::LOOP);
  ss << "Permissive map:\n"
     << idGraphNodesToString(*this, IdMappingMode::PERMISSIVE);
  ss << "Consumer maps:\n";
  for (auto key : getSortedKeys(id_graph_.consumers(), Statement::lessThan)) {
    auto consumers = id_graph_.consumers().at(key);
    std::sort(consumers.begin(), consumers.end(), Statement::lessThan);
    ss << "  " << key->toString() << " :: " << consumers.toString() << "\n";
  }

  ss << "Producer maps:\n";
  for (auto key : getSortedKeys(id_graph_.producers(), Statement::lessThan)) {
    VectorOfUniqueEntries<IterDomain*> producers =
        id_graph_.producers().at(key);
    std::sort(producers.begin(), producers.end(), Statement::lessThan);
    ss << "  " << key->toString() << " :: " << producers.toString() << "\n";
  }

  ss << "Sibling map:\n" << id_graph_.siblings().toString() << "\n";

  ss << "} compute at map" << std::endl;
  return ss.str();
}

bool ComputeAtMap::isViewRfactor(IterDomain* ref_id) const {
  return id_graph_.viewRfactorIds().find(ref_id) !=
      id_graph_.viewRfactorIds().end();
}

std::vector<IterDomain*> ComputeAtMap::getViewRfactorDomainsOfIdGroup(
    IterDomain* ref_id,
    IdMappingMode mode) const {
  auto disjoint_set = disjointSetOf(ref_id, mode);
  std::vector<IterDomain*> rfactor_ids;
  for (auto disjoint_id : disjoint_set->vector()) {
    if (id_graph_.viewRfactorIds().find(disjoint_id) !=
        id_graph_.viewRfactorIds().end()) {
      rfactor_ids.push_back(disjoint_id);
    }
  }
  return rfactor_ids;
}

const std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>& ComputeAtMap::
    disjointSetOf(IterDomain* id, IdMappingMode mode) const {
  TORCH_INTERNAL_ASSERT(
      idExistsInMap(id),
      id->toString(),
      " has not been processed in this Compute At Map, yet the disjoint set for it was requested.");
  return getIdSets(mode).disjointSetMap().at(id);
}

const DisjointSets<IterDomain*>& ComputeAtMap::getIdSets(
    IdMappingMode mode) const {
  switch (mode) {
    case IdMappingMode::EXACT:
      return id_graph_.exactNodes();
    case IdMappingMode::ALMOSTEXACT:
      return id_graph_.almostExactNodes();
    case IdMappingMode::LOOP:
      return id_graph_.loopNodes();
    case IdMappingMode::PERMISSIVE:
      return id_graph_.permissiveNodes();
  }
  TORCH_INTERNAL_ASSERT(false, "Error with mapping mode provided.");
}

bool ComputeAtMap::idExistsInMap(IterDomain* id) const {
  return getIdSets(IdMappingMode::EXACT).disjointSetMap().find(id) !=
      getIdSets(IdMappingMode::EXACT).disjointSetMap().end();
}

VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
ComputeAtMap::getInputDisjointSetsOf(IterDomain* of_id, bool stop_at_rfactor) {
  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      input_disjoint_sets;

  VectorOfUniqueEntries<IterDomain*> inputs;
  // This deque could be VectorOfUniqueEntries
  std::deque<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>> to_visit(
      {disjointSetOf(of_id, IdMappingMode::EXACT)});
  std::unordered_set<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      visited;
  while (!to_visit.empty()) {
    auto currently_visiting = to_visit.front();
    to_visit.pop_front();
    if (!visited.emplace(currently_visiting).second) {
      continue;
    }
    auto defs_it = unique_exact_definitions_.find(currently_visiting);
    TORCH_INTERNAL_ASSERT(
        defs_it != unique_exact_definitions_.end(),
        "unique_exact_definitions_ wasn't correctly generated, missing the disjoint set:\n",
        currently_visiting->toString());

    // If there's no definition, we've found an input.
    if (defs_it->second.empty()) {
      input_disjoint_sets.pushBack(currently_visiting);
      continue;
    }

    if (stop_at_rfactor &&
        std::any_of(
            currently_visiting->vector().begin(),
            currently_visiting->vector().end(),
            [&](IterDomain* id) { return isViewRfactor(id); })) {
      input_disjoint_sets.pushBack(currently_visiting);
      continue;
    }

    // Traverse producers of current disjoint set and collect unique exact
    // disjoint set producers
    VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
        producers_of_currently_visiting;

    for (auto def : defs_it->second) {
      auto id_inps = ir_utils::filterByType<IterDomain>(def->inputs());
      for (auto id_inp : id_inps) {
        producers_of_currently_visiting.pushBack(
            disjointSetOf(id_inp, IdMappingMode::EXACT));
      }
    }

    // Add producers to visit if not already there
    for (auto producer : producers_of_currently_visiting.vector()) {
      if (visited.find(producer) == visited.end()) {
        to_visit.push_back(producer);
      }
    }
  }

  return input_disjoint_sets;
}

VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
ComputeAtMap::getAllDisjointSetProducers(
    const VectorOfUniqueEntries<
        std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>& exact_sets) {
  // This deque could be VectorOfUniqueEntries
  std::deque<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>> to_visit(
      {exact_sets.vector().begin(), exact_sets.vector().end()});

  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      visited;

  while (!to_visit.empty()) {
    auto currently_visiting = to_visit.front();
    to_visit.pop_front();
    if (!visited.pushBack(currently_visiting)) {
      continue;
    }
    auto defs_it = unique_exact_definitions_.find(currently_visiting);
    TORCH_INTERNAL_ASSERT(
        defs_it != unique_exact_definitions_.end(),
        "unique_exact_definitions_ wasn't correctly generated, missing the disjoint set:\n",
        currently_visiting->toString());

    // Traverse producers of current disjoint set and collect unique exact
    // disjoint set producers
    VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
        producers_of_currently_visiting;

    for (auto def : defs_it->second) {
      auto id_inps = ir_utils::filterByType<IterDomain>(def->inputs());
      for (auto id_inp : id_inps) {
        producers_of_currently_visiting.pushBack(
            disjointSetOf(id_inp, IdMappingMode::EXACT));
      }
    }

    // Add producers to visit if not already there
    for (auto producer : producers_of_currently_visiting.vector()) {
      if (!visited.has(producer)) {
        to_visit.push_back(producer);
      }
    }
  }

  return visited;
}

VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
ComputeAtMap::getAllDisjointSetConsumers(
    const VectorOfUniqueEntries<
        std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>& exact_sets) {
  // This deque could be VectorOfUniqueEntries
  std::deque<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>> to_visit(
      {exact_sets.vector().begin(), exact_sets.vector().end()});

  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      visited;

  while (!to_visit.empty()) {
    auto currently_visiting = to_visit.front();
    to_visit.pop_front();
    if (!visited.pushBack(currently_visiting)) {
      continue;
    }
    auto uses_it = unique_exact_uses_.find(currently_visiting);
    TORCH_INTERNAL_ASSERT(
        uses_it != unique_exact_uses_.end(),
        "unique_exact_uses_ wasn't correctly generated, missing the disjoint set:\n",
        currently_visiting->toString());

    // Traverse consumers of current disjoint set and collect unique exact
    // disjoint set consumers
    VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
        consumers_of_currently_visiting;

    for (auto uses : uses_it->second) {
      auto id_outs = ir_utils::filterByType<IterDomain>(uses->outputs());
      for (auto id_out : id_outs) {
        consumers_of_currently_visiting.pushBack(
            disjointSetOf(id_out, IdMappingMode::EXACT));
      }
    }

    // Add consumers to visit if not already there
    for (auto consumer : consumers_of_currently_visiting.vector()) {
      if (!visited.has(consumer)) {
        to_visit.push_back(consumer);
      }
    }
  }

  return visited;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
