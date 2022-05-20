#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>

#include <torch/csrc/jit/codegen/cuda/lower_trivial_broadcast.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

void ConcretizedBroadcastDomains::build(Fusion* fusion) {
  // Initialize the origin map with input broadcast domains
  for (const auto fusion_input_tv :
       ir_utils::filterByType<TensorView>(fusion->inputs())) {
    for (auto root_id : fusion_input_tv->getRootDomain()) {
      if (root_id->isBroadcast()) {
        broadcast_origin_map_.emplace(
            root_id, std::unordered_set<IterDomain*>({root_id}));
      }
    }
  }
  traverse(fusion);
}

bool ConcretizedBroadcastDomains::isConcretized(IterDomain* id) const {
  auto it = concretized_domains_.find(id);
  return it != concretized_domains_.end();
}

void ConcretizedBroadcastDomains::handle(BroadcastOp* bop) {
  // Create a new entry for each of new broadcast domains
  auto out = bop->out()->as<TensorView>();
  for (const auto i : c10::irange(out->getRootDomain().size())) {
    if (bop->getBroadcastDimFlags().at(i)) {
      auto new_bcast_id = out->getRootDomain().at(i);
      broadcast_origin_map_.emplace(
          new_bcast_id, std::unordered_set<IterDomain*>({new_bcast_id}));
    }
  }
}

void ConcretizedBroadcastDomains::handle(Expr* expr) {
  IterVisitor::handle(expr);

  // Propagate broadcast origin info from producers to consumers
  for (auto producer : ir_utils::filterByType<TensorView>(expr->inputs())) {
    std::unordered_set<IterDomain*> producer_broadcasts;
    // This assumes there's no merged broadcast axes between root and rfactor
    // domains which is not possible at the moment. If this assumption is ever
    // invalidated we would need to manaually propagate root IDs to rfactor IDs.
    for (auto producer_id : producer->getMaybeRFactorDomain()) {
      if (producer_id->isBroadcast()) {
        producer_broadcasts.insert(producer_id);
      }
    }
    if (producer_broadcasts.empty()) {
      continue;
    }

    for (auto consumer : ir_utils::filterByType<TensorView>(expr->outputs())) {
      auto p2c_map =
          PairwiseRootDomainMap(producer, consumer)
              .mapProducerToConsumer(
                  producer->domain(), consumer->domain(), producer_broadcasts);
      for (const auto& kv : p2c_map) {
        auto p_id = kv.first;
        auto c_id = kv.second;
        const bool is_concretized = !c_id->isBroadcast();
        auto it = broadcast_origin_map_.find(p_id);
        TORCH_INTERNAL_ASSERT(
            it != broadcast_origin_map_.end(),
            "Broadcast origin info not found for producer broadcast domain: ",
            p_id->toString(),
            " of ",
            producer->toString());
        const auto& producer_origins = it->second;
        if (is_concretized) {
          // Keep track of all the origin domains as concretized
          for (auto origin : producer_origins) {
            // concretized_root_domains_.insert(origin);
            markAsConcretized(origin);
          }
        } else {
          // Not concretized yet. Propagate forward the origin info.
          auto& consumer_origins = broadcast_origin_map_[c_id];
          for (auto origin : producer_origins) {
            consumer_origins.insert(origin);
          }
          consumer_origins.insert(c_id);
        }
      }
    }
  }
}

void ConcretizedBroadcastDomains::markAsConcretized(IterDomain* root_domain) {
  std::deque<IterDomain*> child_domains({root_domain});
  while (!child_domains.empty()) {
    auto child = child_domains.front();
    child_domains.pop_front();
    if (!concretized_domains_.emplace(child).second) {
      continue;
    }
    const auto& child_uses = child->uses();
    for (auto child_use : child_uses) {
      for (auto out_id :
           ir_utils::filterByType<IterDomain>(child_use->outputs())) {
        child_domains.push_back(out_id);
      }
    }
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
