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
// computeAtPosition
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

// Returns a pair of mapped IDs
c10::optional<std::pair<IterDomain*, IterDomain*>> detectMappablePair(
    const std::vector<IterDomain*>& ids,
    const IterDomainGraph& id_graph) {
  for (auto id1 : ids) {
    for (auto id2 : ids) {
      if (id1 == id2) {
        continue;
      }
      if (id_graph.permissiveNodes().disjointSetMap().at(id1)->has(id2)) {
        return std::make_pair(id1, id2);
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
        detectMappablePair(tv->getRootDomain(), id_graph);
    if (self_mappped_root_pair.has_value()) {
      return std::make_tuple(
          tv,
          self_mappped_root_pair->first,
          self_mappped_root_pair->second,
          "Root");
    }

    // Rfactor domains
    if (tv->hasRFactor()) {
      auto self_mappped_rf_pair =
          detectMappablePair(tv->getRFactorDomain(), id_graph);
      if (self_mappped_rf_pair.has_value()) {
        return std::make_tuple(
            tv,
            self_mappped_rf_pair->first,
            self_mappped_rf_pair->second,
            "RFactor");
      }
    }

    // Leaf domains
    auto self_mappped_leaf_pair =
        detectMappablePair(tv->domain()->domain(), id_graph);
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
    const auto& root_domain = tv->getRootDomain();
    const auto& domain = tv->domain()->domain();

    // Grab all values in the history of the tensor view's domain
    auto all_vals = DependencyCheck::getAllValsBetween(
        {root_domain.begin(), root_domain.end()},
        {domain.begin(), domain.end()});

    // Filter so we only have iteration domains (ignore Ints used in split)
    auto all_ids = ir_utils::filterByType<IterDomain>(all_vals);

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
        for (auto entry : c2f_map) {
          auto c_id = entry.first;
          auto f_id = entry.second;
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
        // If outside computeAt axis, we don't want to directly map
        // consumer/producer as their thread mappings could change as long as
        // it's across shared/global memory.
        auto pairwise_map = PairwiseRootDomainMap(p_tv, c_tv);
        const auto& permissive_c2p_root_map =
            pairwise_map.mapConsumerToProducer(c_tv->domain(), p_tv->domain());

        // Look for matching ID transformations in producer and consumer, replay
        // producer as consumer. We want to replay producer as consumer instead
        // of the other way around since consumer may have some broadcasted axes
        // producer doesn't have merged into loops producer may use. If we did
        // consumer as producer we wouldn't have this information in the
        // mapping. If we're using this map for indexing, we do not want to
        // propagate broadcast mismatches. If we're using it to identify loop
        // nests, we do want to propagate mismatches.
        auto permissive_replay_PasC =
            BestEffortReplay::replayPasC(p_tv, c_tv, -1, pairwise_map);

        const auto& permissive_c2p_map = permissive_replay_PasC.getReplay();
        const auto permissive_disjoint_sets =
            permissive_replay_PasC.getDisjointSets();

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

        for (auto entry : exact_c2p_map) {
          auto c_id = entry.first;
          auto p_id = entry.second;
          exact_nodes_.mapEntries(c_id, p_id);
          consumers_.at(p_id).pushBack(c_id);
          producers_.at(c_id).pushBack(p_id);

          // Add the swizzle inputs to the same
          //  disjoint set as well if either c_id
          //  or p_id is swizzle output.
          mapMaybeSwizzleOp(exact_nodes_, p_id);
          mapMaybeSwizzleOp(exact_nodes_, c_id);
        }

        for (auto entry : permissive_c2p_map) {
          auto c_id = entry.first;
          auto p_id = entry.second;
          if (idIsAComputeAtLeafDomain(p_id, p_tv)) {
            loop_nodes_.mapEntries(c_id, p_id);
          } else {
            // When there are trivial reductions merged with other dims, `p_id`
            // might not be a compute at leaf domain of `p_tv`, but it actually
            // has an equivalent compute at leaf domain. For that case, we map
            // the equivalent compute at leaf domain.
            for (int i = 0; i < p_tv->getComputeAtPosition(); i++) {
              auto id = p_tv->axis(i);
              if (permissive_disjoint_sets.permissiveAreMapped(p_id, id)) {
                loop_nodes_.mapEntries(c_id, id);
              }
            }
          }
          permissive_nodes_.mapEntries(c_id, p_id);
          consumers_.at(p_id).pushBack(c_id);
          producers_.at(c_id).pushBack(p_id);

          // Add the swizzle inputs to the same
          //  disjoint set as well if either c_id
          //  or p_id is swizzle output.
          mapMaybeSwizzleOp(permissive_nodes_, p_id);
          mapMaybeSwizzleOp(permissive_nodes_, c_id);
        }

        // Make sure we always get root mapping for the permissive map.
        // Because of forwarding we could otherwise miss some root mappings.
        for (auto entry : permissive_c2p_root_map) {
          auto c_id = entry.first;
          auto p_id = entry.second;
          // Map the id's together
          permissive_nodes_.mapEntries(c_id, p_id);
          consumers_.at(p_id).pushBack(c_id);
          producers_.at(c_id).pushBack(p_id);
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
    : id_graph_(fusion), fusion_(fusion) {
  build(fusion);
}

void ComputeAtMap::build(Fusion* fusion) {
  trivial_reduction_info_.build(fusion);
  buildConcreteIds();
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

namespace {

// Validate a LOOP concrete ID has the complete ID set required for
// indexing. See issue #1655 and FusionIncompleteConcreteID for an
// example fusion that fails with this validation. Fixing this issue
// would require creating a reference IterDomain with all the
// necessary root ID for for loop extent generation, for indexing, and for
// predication.
//
// root_ids_of_all_ids and root_ids_of_concrete_id consist of EXACT
// concrete IDs.
void validateCompletenessOfLoopConcreteID(
    IterDomain* concrete_id,
    const ComputeAtMap& ca_map,
    const TrivialReductionInfo& trivial_reduction_info,
    // All root id's of all IDs in the disjoint id set
    const std::unordered_set<IterDomain*>& root_ids_of_all_ids,
    // Map from a root id to the concrete id's it's represented in
    const std::unordered_set<IterDomain*>& root_ids_of_concrete_id,
    const std::unordered_map<IterDomain*, std::vector<IterDomain*>>&
        root_id_to_maybe_concrete_ids,
    // Disjoint set just for printing
    const std::vector<IterDomain*>& id_set,
    // All the candidate concrete IDs found for this disjoint id set
    const std::vector<IterDomain*>& maybe_concrete_ids) {
  std::vector<IterDomain*> root_ids_not_found_with_concrete_id;

  for (auto root_id : root_ids_of_all_ids) {
    if (root_ids_of_concrete_id.find(root_id) !=
        root_ids_of_concrete_id.end()) {
      continue;
    }

    // None of the root IDs of the conrete ID is exactly mapped with
    // root_id.

    // It is still a valid concrete ID if it has a non-broadcast
    // root ID that is mapped with root_id.
    if ((root_id->isBroadcast() || trivial_reduction_info.isDerived(root_id)) &&
        std::any_of(
            root_ids_of_concrete_id.begin(),
            root_ids_of_concrete_id.end(),
            [&](auto root_id_of_concrete_id) {
              return !root_id_of_concrete_id->isBroadcast() &&
                  !trivial_reduction_info.isDerived(root_id_of_concrete_id) &&
                  ca_map.areMapped(
                      root_id,
                      root_id_of_concrete_id,
                      IdMappingMode::PERMISSIVE);
            })) {
      continue;
    }

    // If all of the corresponding maybe-concrete IDs are exactly
    // mapped with the concrete ID, this missing root_id is not a
    // problem. This can happen with reduction rfactor, e.g.,
    // FusionAdvancedLowering1.
    if (std::all_of(
            root_id_to_maybe_concrete_ids.at(root_id).begin(),
            root_id_to_maybe_concrete_ids.at(root_id).end(),
            [&](auto maybe_concrete_id) {
              return ca_map.areMapped(
                  concrete_id, maybe_concrete_id, IdMappingMode::EXACT);
            })) {
      continue;
    }

    root_ids_not_found_with_concrete_id.push_back(root_id);
  }

  if (root_ids_not_found_with_concrete_id.empty()) {
    return;
  }

  // Error detected as some root IDs are not accounted for by the
  // concrete ID.
  std::stringstream error_msg;
  error_msg << "IDs: " << ir_utils::toString(id_set);
  error_msg << ", concrete ID: " << concrete_id->toString();
  error_msg << ", maybe concrete IDs: "
            << ir_utils::toString(maybe_concrete_ids);
  error_msg << ", all root IDs:";
  for (auto root_id : root_ids_of_all_ids) {
    error_msg << " " << root_id->toString();
  }
  error_msg << ", root IDs not found with concrete ID: ";
  for (auto id : root_ids_not_found_with_concrete_id) {
    error_msg << " " << id->toString();
  }
  TORCH_INTERNAL_ASSERT(
      false, "Concrete ID failed to cover all root IDs. ", error_msg.str());
}

} // namespace

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
      maybe_concrete_ids.vector().size(),
      "No potential concrete_id's found for ",
      id->toString());

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

  // For the LOOP map, the concrete ID must account for all root IDs
  // of all of the IDs in each disjoit set. At least those ID's that are
  // non-broadcast/non-reduction. As broadcast is only important here if it's
  // concretized in the set. Track information so we can later make sure the
  // concrete id has accounted for all iter domains meaning it has a correct
  // loop size.
  std::unordered_set<IterDomain*> root_ids_of_all_ids;
  std::unordered_set<IterDomain*> root_ids_of_concrete_id;
  std::unordered_map<IterDomain*, std::vector<IterDomain*>>
      root_id_to_maybe_concrete_ids;

  // Populate the above information, look for the concrete id, validate the loop
  // concrete ID.
  for (auto maybe_concrete_id : maybe_concrete_ids.vector()) {
    std::unordered_set<IterDomain*> root_ids;
    std::deque<IterDomain*> to_visit;

    to_visit.push_back(maybe_concrete_id);
    while (to_visit.size()) {
      auto current_id = to_visit.front();
      to_visit.pop_front();
      if (isViewRfactor(current_id)) {
        root_ids.emplace(current_id);
        continue;
      }

      // push back producer IterDomains or add root if they don't exist
      auto producer_vals = ir_utils::producerValsOf(current_id);
      auto producer_ids = ir_utils::filterByType<IterDomain>(producer_vals);

      if (producer_ids.empty()) {
        root_ids.emplace(current_id);
      } else {
        to_visit.insert(
            to_visit.end(), producer_ids.begin(), producer_ids.end());
      }
    }

    if (mode == IdMappingMode::LOOP) {
      std::transform(
          root_ids.begin(),
          root_ids.end(),
          std::inserter(root_ids_of_all_ids, root_ids_of_all_ids.end()),
          [&](const auto root_id) {
            auto exact_concrete_id =
                getConcreteMappedID(root_id, IdMappingMode::EXACT);
            root_id_to_maybe_concrete_ids[exact_concrete_id].push_back(
                maybe_concrete_id);
            return exact_concrete_id;
          });
    }

    int bcast_root_count = std::count_if(
        root_ids.begin(), root_ids.end(), [&](IterDomain* root_id) {
          return root_id->isBroadcast()
              // TODO: This shouldn't have a negative impact, but (emperically)
              // might not be necessary
              || trivial_reduction_info_.isDerived(root_id);
        });
    int iter_root_count = (int)root_ids.size() - bcast_root_count;
    if (iter_root_count > max_iter_root_count ||
        (iter_root_count == max_iter_root_count &&
         bcast_root_count > max_bcast_root_count)) {
      max_iter_root_count = iter_root_count;
      max_bcast_root_count = bcast_root_count;
      concrete_id = maybe_concrete_id;

      // If we update the concrete_id, then update the root_ids_of_concrete_id
      // to reflect this id
      if (mode == IdMappingMode::LOOP) {
        root_ids_of_concrete_id.clear();
        std::transform(
            root_ids.begin(),
            root_ids.end(),
            std::inserter(
                root_ids_of_concrete_id, root_ids_of_concrete_id.end()),
            [&](const auto root_id) {
              return getConcreteMappedID(root_id, IdMappingMode::EXACT);
            });
      }
    }
  } // end maybe_concrete_id

  TORCH_INTERNAL_ASSERT(
      concrete_id != nullptr,
      "Something went wrong, could not find a concrete id.");

  if (mode == IdMappingMode::LOOP) {
    // Validate the concrete id has influence from all the roots of all the
    // consumers that will map to this concete id in the loop map. This means
    // all the consumers in all expressions of the loop nest generated based on
    // this concrete ID will have their roots mapping to this concrete ID
    // represented in the extent of this concrete id.
    validateCompletenessOfLoopConcreteID(
        concrete_id,
        *this,
        trivial_reduction_info_,
        root_ids_of_all_ids,
        root_ids_of_concrete_id,
        root_id_to_maybe_concrete_ids,
        disjoint_set_shared_ptr->vector(),
        maybe_concrete_ids.vector());
  }

  return concrete_id;
}

void ComputeAtMap::buildConcreteIds() {
  for (const auto& disjoint_set_shared_ptr :
       id_graph_.permissiveNodes().disjointSets()) {
    TORCH_INTERNAL_ASSERT(
        disjoint_set_shared_ptr->vector().size(),
        "Cannot compute concrete id of empty set.");
    auto first_id = disjoint_set_shared_ptr->vector().front();
    auto concrete_id = computeConcreteId(first_id, IdMappingMode::PERMISSIVE);
    concrete_id_cache_[disjoint_set_shared_ptr] = concrete_id;
  }

  for (const auto& disjoint_set_shared_ptr :
       id_graph_.exactNodes().disjointSets()) {
    TORCH_INTERNAL_ASSERT(
        disjoint_set_shared_ptr->vector().size(),
        "Cannot compute concrete id of empty set.");
    auto first_id = disjoint_set_shared_ptr->vector().front();
    auto concrete_id = computeConcreteId(first_id, IdMappingMode::EXACT);
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
  const auto& disjoint_sets = ca_map.getIdSets(mode);
  for (const auto& s_ptr : disjoint_sets.disjointSets()) {
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
  ss << "Permissive map:\n"
     << idGraphNodesToString(*this, IdMappingMode::PERMISSIVE);
  ss << "Exact map:\n" << idGraphNodesToString(*this, IdMappingMode::EXACT);
  ss << "Loop map:\n" << idGraphNodesToString(*this, IdMappingMode::LOOP);
  ss << "Consumer maps:\n";
  for (auto entry : id_graph_.consumers()) {
    ss << "  " << entry.first->toString() << " :: " << entry.second.toString()
       << "\n";
  }

  ss << "Producer maps:\n";
  for (auto entry : id_graph_.producers()) {
    ss << "  " << entry.first->toString() << " :: " << entry.second.toString()
       << "\n";
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
    case IdMappingMode::PERMISSIVE:
      return id_graph_.permissiveNodes();
    case IdMappingMode::EXACT:
      return id_graph_.exactNodes();
    case IdMappingMode::LOOP:
      return id_graph_.loopNodes();
  }
  TORCH_INTERNAL_ASSERT(false, "Error with mapping mode provided.");
}

bool ComputeAtMap::idExistsInMap(IterDomain* id) const {
  return getIdSets(IdMappingMode::EXACT).disjointSetMap().find(id) !=
      getIdSets(IdMappingMode::EXACT).disjointSetMap().end();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
