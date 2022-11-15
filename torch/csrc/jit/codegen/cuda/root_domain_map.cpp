#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>

#include <sstream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

std::unordered_map<IterDomain*, IterDomain*> RootDomainMap::
    mapProducerToConsumer(
        const TensorDomain* producer,
        const TensorDomain* consumer,
        const std::unordered_set<IterDomain*>& root_dims_to_map) const {
  return map(producer, consumer, root_dims_to_map, true);
}

std::unordered_map<IterDomain*, IterDomain*> RootDomainMap::
    mapProducerToConsumer(
        const TensorDomain* producer,
        const TensorDomain* consumer) const {
  std::unordered_set<IterDomain*> root_dims_to_map(
      producer->getMaybeRFactorDomain().begin(),
      producer->getMaybeRFactorDomain().end());
  return mapProducerToConsumer(producer, consumer, root_dims_to_map);
}

std::unordered_map<IterDomain*, IterDomain*> RootDomainMap::
    mapConsumerToProducer(
        const TensorDomain* consumer,
        const TensorDomain* producer,
        const std::unordered_set<IterDomain*>& root_dims_to_map) const {
  return map(producer, consumer, root_dims_to_map, false);
}

std::unordered_map<IterDomain*, IterDomain*> RootDomainMap::
    mapConsumerToProducer(
        const TensorDomain* consumer,
        const TensorDomain* producer) const {
  std::unordered_set<IterDomain*> root_dims_to_map(
      consumer->getRootDomain().begin(), consumer->getRootDomain().end());
  return mapConsumerToProducer(consumer, producer, root_dims_to_map);
}

PairwiseRootDomainMap::PairwiseRootDomainMap(
    const TensorView* producer,
    const TensorView* consumer,
    bool is_exact)
    : producer_tv_(producer), consumer_tv_(consumer), is_exact_(is_exact) {
  TORCH_INTERNAL_ASSERT(producer != nullptr);
  TORCH_INTERNAL_ASSERT(consumer != nullptr);
  TORCH_INTERNAL_ASSERT(producer->fusion() == consumer->fusion());
  // Make sure they are really a producer and its consumer
  TORCH_INTERNAL_ASSERT(
      producer->isConsumerOf(consumer),
      "Not a producer-consumer pair: ",
      producer,
      ", ",
      consumer);
}

std::unordered_map<IterDomain*, IterDomain*> PairwiseRootDomainMap::map(
    const TensorDomain* producer,
    const TensorDomain* consumer,
    const std::unordered_set<IterDomain*>& root_dims_to_map,
    bool producer_to_consumer) const {
  // Sanity check that the given producer and consumer domains are
  // really the TensorDomains of the producer and consumer TensorViews
  // given to the constructor.
  TORCH_INTERNAL_ASSERT(producer_tv_->domain() == producer);
  TORCH_INTERNAL_ASSERT(consumer_tv_->domain() == consumer);

  if (consumer_tv_->definition()->isA<TransposeOp>()) {
    return mapTranspose(
        producer, consumer, root_dims_to_map, producer_to_consumer);
  }

  std::vector<bool> broadcast_flags;
  if (BroadcastOp* bop =
          dynamic_cast<BroadcastOp*>(consumer_tv_->definition())) {
    broadcast_flags = bop->getBroadcastDimFlags();
  }

  std::vector<bool> squeeze_flags;
  if (SqueezeOp* sop = dynamic_cast<SqueezeOp*>(consumer_tv_->definition())) {
    squeeze_flags = sop->getSqueezeDimFlags();
  }

  IterDomain* selected_id = nullptr;
  if (SelectOp* sop = dynamic_cast<SelectOp*>(consumer_tv_->definition())) {
    selected_id = sop->getSelectAxis();
  }

  std::unordered_map<IterDomain*, IterDomain*> dom_map;
  const auto producer_root =
      TensorDomain::noReductions(producer->getMaybeRFactorDomain());
  const auto& consumer_root = consumer->getRootDomain();
  size_t itc = 0, itp = 0;
  while (itc < consumer_root.size() && itp < producer_root.size()) {
    IterDomain* producer_id = producer_root[itp];
    IterDomain* consumer_id = consumer_root[itc];

    // When the producer ID is the dim of a SelectOp, there is no
    // mapping for it.
    if (producer_id == selected_id) {
      itp++;
      continue;
    }

    // When the consumer ID is a new broadcast domain, there is no
    // mapping for it.
    if (!broadcast_flags.empty() && broadcast_flags.at(itc)) {
      TORCH_INTERNAL_ASSERT(consumer_id->isBroadcast());
      itc++;
      continue;
    }

    // When the producer ID is a removed broadcast domain, there is no
    // mapping for it.
    if (!squeeze_flags.empty() && squeeze_flags.at(itp)) {
      TORCH_INTERNAL_ASSERT(producer_id->isBroadcast());
      itp++;
      continue;
    }

    // In exact mapping, do not map broadcast domains with
    // non-broadcast domains, except with a broadcast producer and a
    // consumer trivial reduction
    if (is_exact_ && producer_id->isBroadcast() != consumer_id->isBroadcast()) {
      itc++;
      itp++;
      continue;
    }

    IterDomain* map_key_id = producer_id;
    IterDomain* map_value_id = consumer_id;
    if (!producer_to_consumer) {
      std::swap(map_key_id, map_value_id);
    }

    if (root_dims_to_map.find(map_key_id) != root_dims_to_map.end()) {
      dom_map.insert(std::make_pair(map_key_id, map_value_id));
    }
    itc++;
    itp++;
  }
  return dom_map;
}

std::unordered_map<IterDomain*, IterDomain*> PairwiseRootDomainMap::
    mapTranspose(
        const TensorDomain* producer,
        const TensorDomain* consumer,
        const std::unordered_set<IterDomain*>& root_dims_to_map,
        bool producer_to_consumer) const {
  const auto producer_root =
      TensorDomain::noReductions(producer->getMaybeRFactorDomain());
  const auto& consumer_root = consumer->getRootDomain();

  std::unordered_map<IterDomain*, IterDomain*> dom_map;

  TransposeOp* top = dynamic_cast<TransposeOp*>(consumer_tv_->definition());
  TORCH_INTERNAL_ASSERT(top != nullptr);

  const auto& new2old = top->new2old();
  for (const auto i : c10::irange(consumer_root.size())) {
    IterDomain* producer_id = producer_root[new2old[i]];
    IterDomain* consumer_id = consumer_root[i];

    // In exact mapping, do not map broadcast domains with
    // non-broadcast domains
    if (is_exact_ && producer_id->isBroadcast() != consumer_id->isBroadcast()) {
      continue;
    }

    if (!producer_to_consumer) {
      std::swap(producer_id, consumer_id);
    }

    if (root_dims_to_map.find(producer_id) != root_dims_to_map.end()) {
      dom_map.insert(std::make_pair(producer_id, consumer_id));
    }
  }
  return dom_map;
}

std::string PairwiseRootDomainMap::toString() const {
  std::stringstream ss;
  ss << "{producer: " << producer() << ", consumer: " << consumer();
  auto p2c = mapProducerToConsumer(producer()->domain(), consumer()->domain());
  for (auto pair : p2c) {
    ss << ", " << pair.first->toString() << " -> " << pair.second->toString();
  }
  ss << "}";
  return ss.str();
}

namespace {

template <typename T>
auto ensureMapping(
    T& m,
    const typename T::key_type& key,
    const typename T::mapped_type& init_value) {
  auto it = m.find(key);
  if (it == m.end()) {
    it = m.insert({key, init_value}).first;
  }
  return it;
}

TensorView* lookUpTv(const TensorDomain* td) {
  Fusion* fusion = FusionGuard::getCurFusion();
  for (auto tv : ir_utils::filterByType<TensorView>(fusion->vals())) {
    if (tv->domain() == td) {
      return tv;
    }
  }
  return nullptr;
}

} // namespace

std::string DomainKey::toString() const {
  std::stringstream ss;
  if (id()) {
    ss << id();
  } else {
    ss << "null";
  }
  if (concreteId()) {
    ss << " (concrete: " << concreteId() << ")";
  }
  ss << " in ";
  if (td()) {
    auto tv = lookUpTv(td());
    TORCH_INTERNAL_ASSERT(tv != nullptr, "No TV found for ", td()->toString());
    ss << "T" << tv->name() << "[ " << td()->getRootDomain() << " ]";
    if (td()->hasRFactor()) {
      ss << " (Rfactor: [ " << td()->getMaybeRFactorDomain() << " ])";
    }
  } else {
    ss << "null";
  }
  return ss.str();
}

UnmappableReductionDomains::UnmappableReductionDomains() {
  Fusion* fusion = FusionGuard::getCurFusion();
  traverse(fusion);
}

namespace {

//! Find all domains that a given domain is dependent on
class FindInputDomains : BackwardVisitor {
 private:
  FindInputDomains(TensorView* tv, const IterDomain* id)
      : BackwardVisitor(false), tv_(tv) {
    input_keys_.insert(DomainKey(tv_->domain(), id));
  }

  DomainKeySet find() {
    traverseTo(tv_->fusion(), {tv_});
    return input_keys_;
  }

  void handle(Expr* expr) override {
    for (auto output : expr->outputs()) {
      if (!output->isA<TensorView>()) {
        continue;
      }
      for (auto input : expr->inputs()) {
        if (!input->isA<TensorView>()) {
          continue;
        }
        propagate(input->as<TensorView>(), output->as<TensorView>());
      }
    }
  }

  void propagate(TensorView* in_tv, TensorView* out_tv) {
    auto c2p = PairwiseRootDomainMap(in_tv, out_tv)
                   .mapConsumerToProducer(out_tv->domain(), in_tv->domain());
    for (auto root_dom : out_tv->getRootDomain()) {
      DomainKey out_key({out_tv->domain(), root_dom});
      if (input_keys_.find(out_key) == input_keys_.end()) {
        continue;
      }
      auto input_id_it = c2p.find(root_dom);
      if (input_id_it == c2p.end()) {
        continue;
      }
      DomainKey input_key(in_tv->domain(), input_id_it->second);
      input_keys_.insert(input_key);
    }
  }

 private:
  TensorView* tv_ = nullptr;
  DomainKeySet input_keys_;

 public:
  static DomainKeySet find(TensorView* tv, const IterDomain* id) {
    return FindInputDomains(tv, id).find();
  }
};

} // namespace

void UnmappableReductionDomains::handleReductionOutput(TensorView* out_tv) {
  std::vector<DomainKey> reduction_keys;
  for (const auto id : out_tv->getRootDomain()) {
    if (id->isReduction()) {
      DomainKey key(out_tv->domain(), id);
      reduction_keys.push_back(key);
      reduction_domains_.insert({key, {}});
    }
  }
  auto use_chains = DependencyCheck::getAllUseChains(out_tv);
  for (const auto& chain : use_chains) {
    for (const auto& tv : ir_utils::filterByType<TensorView>(chain)) {
      // Do not include the tensor itself in its consumers
      if (tv == out_tv) {
        continue;
      }
      const auto& root_domain = tv->getRootDomain();
      for (const auto& id : root_domain) {
        DomainKey consumer_key(tv->domain(), id);
        for (const auto& reduction_key : reduction_keys) {
          reduction_domains_.at(reduction_key).insert(consumer_key);
        }
      }
    }
  }
  for (const auto& reduction_key : reduction_keys) {
    reduction_domain_inputs_.insert(
        {reduction_key, FindInputDomains::find(out_tv, reduction_key.id())});
  }
}

void UnmappableReductionDomains::handle(ReductionOp* op) {
  // Builds a map from reduction domains to consumer domains.
  TensorView* out_tv = op->out()->as<TensorView>();
  handleReductionOutput(out_tv);
}

void UnmappableReductionDomains::handle(GroupedReductionOp* op) {
  // Builds a map from reduction domains to consumer domains.
  for (auto out : op->outputs()) {
    handleReductionOutput(out->as<TensorView>());
  }
}

void UnmappableReductionDomains::handle(MmaOp* mma) {
  // Builds a map from reduction domains to consumer domains.
  TensorView* out_tv = mma->out()->as<TensorView>();
  handleReductionOutput(out_tv);
}

void UnmappableReductionDomains::handle(WelfordOp* op) {
  // Builds a map from reduction domains to consumer domains.
  handleReductionOutput(op->outAvg()->as<TensorView>());
  handleReductionOutput(op->outVar()->as<TensorView>());
  handleReductionOutput(op->outN()->as<TensorView>());
}

bool UnmappableReductionDomains::isReductionOutputMapped(
    const DomainKeySet& consumer_domains,
    const ComputeAtRootDomainMap& root_map) const {
  // Check each reduction domain if any of the consumer domains
  // conflicts with it
  for (const auto& kv : reduction_domains_) {
    const DomainKey& reduction_domain = kv.first;
    // Domains that must not be mapped with the reduction domain
    const DomainKeySet& incompatible_domains = kv.second;
    // Input domains to the reduction domain
    const auto& input_keys = reduction_domain_inputs_.at(reduction_domain);
    // Check if any of the consumer domains is an input to the
    // reduction
    auto it = std::find_if(
        consumer_domains.begin(),
        consumer_domains.end(),
        [&](const auto& consumer_domain) {
          return std::find(
                     input_keys.begin(), input_keys.end(), consumer_domain) !=
              input_keys.end();
        });
    // None of the consumer domains is used for the reduction
    // domain. They should be safe with respect to this reduction
    // domain
    if (it == consumer_domains.end()) {
      continue;
    }

    // A consumer domain that is an input to the reduction domain
    const DomainKey& input_to_reduction = *it;

    // Check if mapping input_to_reduction with the other domains in
    // consumer_domains. If there's a domain that is a consumer of the
    // reduction, they must not be mapped together
    for (const auto& consumer_domain : consumer_domains) {
      if (consumer_domain == input_to_reduction) {
        continue;
      }
      if (std::any_of(
              incompatible_domains.begin(),
              incompatible_domains.end(),
              [&](const DomainKey& incompatible_domain) {
                return root_map.canMap(
                    consumer_domain.td(),
                    consumer_domain.id(),
                    incompatible_domain.td(),
                    incompatible_domain.id());
              })) {
        return true;
      }
    }
  }
  return false;
}

std::string UnmappableReductionDomains::toString() const {
  std::stringstream ss;
  ss << "Reduction-to-consumer map\n";
  for (const auto& kv : reduction_domains_) {
    ss << "\tReduction: " << kv.first.toString() << "\n";
    for (const auto& mapped_val : kv.second) {
      ss << "\t\tConsumer domain: " << mapped_val.toString() << "\n";
    }
  }

  ss << "Reduction-to-producer map\n";
  for (const auto& kv : reduction_domain_inputs_) {
    ss << "\tReduction: " << kv.first.toString() << "\n";
    for (const auto& mapped_val : kv.second) {
      ss << "\t\tProducer domain: " << mapped_val.toString() << "\n";
    }
  }

  return ss.str();
}

void ComputeAtRootDomainMap::build(bool map_through_reduction) {
  // Make sure we start from scratch. Throw away previous results.
  eq_set_.clear();
  bcast_map_.clear();
  new_broadcast_domains_.clear();
  removed_broadcast_domains_.clear();
  ComputeAtRootDomainMapBuilder builder(*this, map_through_reduction);
}

bool ComputeAtRootDomainMap::canMap(
    const TensorDomain* td_a,
    const IterDomain* id_a,
    const TensorDomain* td_b,
    const IterDomain* id_b) const {
  TORCH_INTERNAL_ASSERT(
      id_a->definition() == nullptr || id_a->isRFactorProduct(),
      "Non-root domain is not supported: ",
      id_a);
  TORCH_INTERNAL_ASSERT(
      id_b->definition() == nullptr || id_b->isRFactorProduct(),
      "Non-root domain is not supported: ",
      id_b);

  // Forward to overloaded functions
  if (!id_a->isBroadcast() && !id_b->isBroadcast()) {
    return canMap(DomainKey(td_a, id_a), DomainKey(td_b, id_b));
  } else if (!id_a->isBroadcast()) {
    return canMap(DomainKey(td_a, id_a), td_b, id_b);
  } else if (!id_b->isBroadcast()) {
    return canMap(DomainKey(td_b, id_b), td_a, id_a);
  }

  // At this point, both are broadcast. Every pair of concrete IDs of
  // both id_a and id_b needs to be looked at. Whether they are
  // mappable depends on whether the concrete IDs are broadcast or
  // not. Note that a broadcast axis is used a concrete ID when it is
  // part of an output tensor domain, i.e., when it never gets
  // concretized with any non-broadcast axis.

  // If there exists a pair of non-broadcast concrete IDs is not
  // mappable, id_a and id_b can't be mapped together. Otherwise, they
  // can be mapped when there is any mappable pair is found.
  bool mappable_pair_found = false;
  for (const auto& key_a : getConcretizedKeys(td_a, id_a)) {
    for (const auto& key_b : getConcretizedKeys(td_b, id_b)) {
      const bool mappable = canMap(key_a, key_b);
      mappable_pair_found = mappable_pair_found || mappable;
      // If both concrete IDs are not broadcast, they must be
      // mappable. Also, if either of the concrete IDs is a reduction,
      // that means a trivial reduction (i.e., broadcast immediately
      // followed by reduction), which does not prevent any mapping.
      if (!key_a.concreteId()->isBroadcast() &&
          !key_b.concreteId()->isBroadcast() &&
          !key_a.concreteId()->isReduction() &&
          !key_b.concreteId()->isReduction() && !mappable) {
        return false;
      }
    }
  }

  return mappable_pair_found;
}

bool ComputeAtRootDomainMap::canMap(
    const DomainKey& key_a,
    const TensorDomain* td_b,
    const IterDomain* id_b) const {
  TORCH_INTERNAL_ASSERT(
      id_b->definition() == nullptr || id_b->isRFactorProduct(),
      "Non-root domain is not supproted: ",
      id_b);

  if (!id_b->isBroadcast()) {
    return canMap(key_a, DomainKey(td_b, id_b));
  }

  // If id_b is broadcast, look at all the concrete IDs that id_b may
  // be concretized to. Whether it is mappable with key_a depends on
  // whether key_a's concrete ID is also broadcast.
  // 1) key_a's concrete ID is also broadcast: They are mappable when
  // there is any mappable concrete ID exists in the concrete ID set
  // of id_b.
  // 2) key_a's concrete ID is not broadcast: Since key_a is indeed
  // concrete, it must be mappable with any of concrete ID of id_b,
  // except when a id_b concrete is broadcast.
  const bool key_a_bcast =
      key_a.concreteId() && key_a.concreteId()->isBroadcast();
  const bool key_a_reduction =
      (key_a.concreteId() && key_a.concreteId()->isReduction()) ||
      key_a.id()->isReduction();
  bool mappable_pair_found = false;
  for (const auto& key_b : getConcretizedKeys(td_b, id_b)) {
    const bool mappable = canMap(key_a, key_b);
    mappable_pair_found = mappable_pair_found || mappable;
    // If both concrete IDs are not broadcast, they must be mappable.
    // However, if key_b's concrete ID is a reduction, the concrete ID
    // is a result of a trivial reduction, so it should not prevent
    // any other mapping. Similarly, if key_a is a reduction, it just
    // needs to find any concrete ID of key_b that can be mapped.
    if (!key_a_bcast && !key_b.concreteId()->isBroadcast() &&
        !key_b.concreteId()->isReduction() && !key_a_reduction && !mappable) {
      return false;
    }
  }

  return mappable_pair_found;
}

bool ComputeAtRootDomainMap::canMap(
    const DomainKey& key_a,
    const DomainKey& key_b) const {
  return key_a == key_b || eq_set_.permissiveAreMapped(key_a, key_b);
}

void ComputeAtRootDomainMap::setAlias(
    const TensorDomain* td,
    const TensorDomain* td_alias) {
  auto tmp_bcast_map = bcast_map_;
  for (const auto& kv : bcast_map_) {
    const auto& bcast_map_key = kv.first;
    const auto& bcast_concrete_id_set = kv.second;
    if (bcast_map_key.td() == td) {
      DomainKey alias_key(td_alias, bcast_map_key.id());
      tmp_bcast_map.insert({alias_key, bcast_concrete_id_set});
    }
  }
  bcast_map_ = tmp_bcast_map;

  auto all_elements = eq_set_.getAllElements();
  for (const auto& key : all_elements.vector()) {
    if (key.td() == td) {
      DomainKey alias_key(td_alias, key.id(), key.concreteId());
      eq_set_.mapEntries(key, alias_key);
    }
  }

  auto tmp_new_broadcast_domains = new_broadcast_domains_;
  for (const auto& key : new_broadcast_domains_) {
    if (key.td() == td) {
      DomainKey alias_key(td_alias, key.id());
      tmp_new_broadcast_domains.insert(alias_key);
    }
  }
  new_broadcast_domains_ = tmp_new_broadcast_domains;

  auto tmp_removed_broadcast_domains = removed_broadcast_domains_;
  for (const auto& key : removed_broadcast_domains_) {
    if (key.td() == td) {
      DomainKey alias_key(td_alias, key.id());
      tmp_removed_broadcast_domains.insert(alias_key);
    }
  }
  removed_broadcast_domains_ = tmp_removed_broadcast_domains;
}

std::vector<DomainKey> ComputeAtRootDomainMap::getConcretizedKeys(
    const TensorDomain* td,
    const IterDomain* id) const {
  DomainKey key(td, id);
  auto it = bcast_map_.find(key);
  TORCH_INTERNAL_ASSERT(it != bcast_map_.end(), "Not found: ", key.toString());
  std::vector<DomainKey> domains;
  std::transform(
      it->second.begin(),
      it->second.end(),
      std::back_inserter(domains),
      [&](const IterDomain* concrete_id) {
        return DomainKey(td, id, concrete_id);
      });
  return domains;
}

std::unordered_set<const IterDomain*>& ComputeAtRootDomainMap::
    getConcretizedDomains(const TensorDomain* td, const IterDomain* id) {
  DomainKey key(td, id);
  auto it = bcast_map_.find(key);
  TORCH_INTERNAL_ASSERT(it != bcast_map_.end(), "Not found: ", key.toString());
  return it->second;
}

std::unordered_map<IterDomain*, IterDomain*> ComputeAtRootDomainMap::
    mapBestEffort(
        const TensorDomain* from_td,
        const std::vector<IterDomain*>& from_root,
        const TensorDomain* to_td,
        const std::vector<IterDomain*>& to_root) const {
  std::unordered_map<IterDomain*, IterDomain*> id_map;
  for (auto& from_id : from_root) {
    for (const auto& to_id : to_root) {
      if (canMap(from_td, from_id, to_td, to_id)) {
        TORCH_INTERNAL_ASSERT(
            id_map.insert({from_id, to_id}).second,
            "Multiple matching ID detected for ",
            from_id);
      }
    }
  }
  return id_map;
}

std::unordered_map<IterDomain*, IterDomain*> ComputeAtRootDomainMap::map(
    const TensorDomain* producer,
    const TensorDomain* consumer,
    const std::unordered_set<IterDomain*>& root_dims_to_map,
    bool producer_to_consumer) const {
  const auto& producer_root =
      TensorDomain::noReductions(producer->getMaybeRFactorDomain());
  const auto& consumer_root = consumer->getRootDomain();
  const TensorDomain* from_td = producer_to_consumer ? producer : consumer;
  const TensorDomain* to_td = producer_to_consumer ? consumer : producer;
  const auto& from_ids = producer_to_consumer ? producer_root : consumer_root;
  const auto& to_ids = producer_to_consumer ? consumer_root : producer_root;
  std::unordered_map<IterDomain*, IterDomain*> id_map =
      mapBestEffort(from_td, from_ids, to_td, to_ids);
  for (auto& from_id : from_ids) {
    if (root_dims_to_map.find(from_id) == root_dims_to_map.end()) {
      // Remove mapping if exists
      id_map.erase(from_id);
      continue;
    }
    if (id_map.find(from_id) != id_map.end()) {
      continue;
    }
    // Matching ID not found. It's an error unless the following three cases:
    // 1. from_id is a new broadcast of a consumer domain; or
    // 2. from_id is a removed broadcast of a producer domain; or
    // 3. from_id is a window axis of a consumer domain; or
    // 4. from_id is a ViewAsScalar domain
    // Note that reduction domains are removed from the producer root domain.
    if ((!producer_to_consumer &&
         (new_broadcast_domains_.find(DomainKey(from_td, from_id)) !=
              new_broadcast_domains_.end() ||
          from_id->getIterType() == IterType::VectorComponent ||
          (window_axes_.count(from_id) > 0))) ||
        (producer_to_consumer &&
         removed_broadcast_domains_.find(DomainKey(from_td, from_id)) !=
             removed_broadcast_domains_.end())) {
      continue;
    }
    TORCH_INTERNAL_ASSERT(
        false,
        "Mapping IterDomain ",
        from_id,
        " of ",
        from_td,
        " not possible as it would require recomputing the source tensor.",
        " Producer root: ",
        producer_root,
        ". Consumer root: ",
        consumer_root,
        ". Mapping: ",
        this->toString());
  }
  return id_map;
}

std::unordered_set<IterDomain*> ComputeAtRootDomainMap::getMappableDims(
    const TensorDomain* producer,
    const TensorDomain* consumer) const {
  //! This funciton previously used mapBestEffort but it can fail when
  //! a domain is mapped to multitple domains, which can happen with
  //! views. Since we only need to find mappable domains, just
  //! grab any domain that is mapped in a pairwise way.

  const auto& producer_root = producer->getMaybeRFactorDomain();
  const auto& consumer_root = consumer->getRootDomain();

  std::unordered_set<IterDomain*> mappable_ids;

  for (const auto& p_id : producer_root) {
    for (const auto& c_id : consumer_root) {
      if (canMap(producer, p_id, consumer, c_id)) {
        mappable_ids.emplace(p_id);
        mappable_ids.emplace(c_id);
      }
    }
  }

  return mappable_ids;
}

std::string ComputeAtRootDomainMap::toString() const {
  return eq_set_.toString();
}

ComputeAtRootDomainMapBuilder::ComputeAtRootDomainMapBuilder(
    ComputeAtRootDomainMap& root_map,
    bool map_through_reduction)
    : BackwardVisitor(false),
      root_map_(root_map),
      map_through_reduction_(map_through_reduction) {
  Fusion* fusion = FusionGuard::getCurFusion();
  TORCH_INTERNAL_ASSERT(fusion != nullptr);
  traverseTo(fusion, fusion->outputs(), false);
  if (!pending_map_.empty()) {
    std::stringstream ss;
    ss << "pending map:\n";
    for (auto& kv : pending_map_) {
      ss << "\t" << kv.first.toString() << "\n";
      for (auto& dk : kv.second) {
        ss << "\t\t" << dk.toString() << "\n";
      }
    }
    std::cerr << ss.str();
  }
  TORCH_INTERNAL_ASSERT(pending_map_.empty());
}

// Set concrete domains for broadcast domains that never get joined
// with a concrete domain. Just set its own domain as a concrete
// domain, which is not concrete but is sufficient for this analysis.
void ComputeAtRootDomainMapBuilder::initializeBcastMap(
    const TensorView* tv,
    const IterDomain* id) {
  TORCH_INTERNAL_ASSERT(id->isBroadcast(), "Not a broadcast axis");
  auto key = DomainKey(tv->domain(), id);
  auto it = root_map_.bcast_map_.find(key);
  if (it != root_map_.bcast_map_.end()) {
    // already initialized.
    return;
  }

  // This initialization should be only used for: 1) fusion output tensors, 2)
  // outputs of multi-consumer expressions that are not fusion outputs, 3) view
  // outputs as broadcasts can be merged with non-broadcast domains, resulting
  // in non-broadcast rfactor domains, and 4) squeeze inputs as broadcasts can
  // be removed by squeeze.
  TORCH_INTERNAL_ASSERT(
      tv->isFusionOutput() || ir_utils::isSqueezeInput(tv) ||
          tv->definition()->outputs().size() > 1 ||
          tv->isDefinitionType(ExprType::ViewOp),
      "Invalid tensor to initialize bcast map t",
      tv->name(),
      " in tensor ",
      tv->toString());
  root_map_.bcast_map_.insert({key, {id}});
}

void ComputeAtRootDomainMapBuilder::addToPendingList(
    const DomainKey& producer,
    const DomainKey& consumer) {
  auto it = ensureMapping(pending_map_, producer, {});
  auto& consumer_set = it->second;
  consumer_set.insert(consumer);
}

void ComputeAtRootDomainMapBuilder::setMapped(
    const DomainKey& producer,
    const DomainKey& consumer) {
  root_map_.eq_set_.mapEntries(producer, consumer);
}

void ComputeAtRootDomainMapBuilder::setInvalid(
    const DomainKey& key1,
    const DomainKey& key2) {
  invalid_mappings_.emplace_back(key1, key2);
}

bool ComputeAtRootDomainMapBuilder::isInvalid(
    const DomainKeySet& domains) const {
  // First, collect all invalid mappings for each of the keys in domains
  DomainKeyMap<DomainKeySet> invalid_key_map;
  for (const auto& key : domains) {
    DomainKeySet invalid_keys;
    for (const auto& invalid_pair : invalid_mappings_) {
      if (root_map_.canMap(key, invalid_pair.first)) {
        invalid_keys.insert(invalid_pair.second);
      } else if (root_map_.canMap(key, invalid_pair.second)) {
        invalid_keys.insert(invalid_pair.first);
      }
    }
    invalid_key_map.emplace(key, invalid_keys);
  }

  // Next, check if any pair is invalid to map.
  const auto num_keys = domains.size();
  const std::vector<DomainKey> domains_vec({domains.begin(), domains.end()});
  for (const auto i : c10::irange(num_keys)) {
    const auto& key_i = domains_vec[i];
    // If no invalid keys found for key_i, it can be skipped.
    const auto invalid_key_map_it = invalid_key_map.find(key_i);
    if (invalid_key_map_it == invalid_key_map.end()) {
      continue;
    }

    // Set of keys that are invalid to be mapped with key_i.
    const DomainKeySet& invalid_keys_for_i = invalid_key_map_it->second;

    // If any other key in domains is identified mappable with any of
    // the keys in this set, the mapping with key_i is invalid.
    for (const auto j : c10::irange(i + 1, num_keys)) {
      const auto& key_j = domains_vec[j];
      if (std::any_of(
              invalid_keys_for_i.begin(),
              invalid_keys_for_i.end(),
              [&](const auto& invalid_key_for_i) {
                return root_map_.canMap(key_j, invalid_key_for_i);
              })) {
        return true;
      }
    }
  }
  return false;
}

void ComputeAtRootDomainMapBuilder::setMaybeMapped(
    const TensorDomain* producer_td,
    const IterDomain* producer_id,
    const TensorDomain* consumer_td,
    const IterDomain* consumer_id) {
  const DomainKey producer_key(producer_td, producer_id);
  const DomainKey consumer_key(consumer_td, consumer_id);

  if (producer_id->isBroadcast()) {
    ensureMapping(root_map_.bcast_map_, producer_key, {});
  }

  if (consumer_id->isBroadcast()) {
    TORCH_INTERNAL_ASSERT(producer_id->isBroadcast());
    // Get bcast_map_ entry for consumer_id
    const auto consumer_bcast_domains =
        root_map_.getConcretizedKeys(consumer_td, consumer_id);
    auto& producer_domains =
        root_map_.getConcretizedDomains(producer_td, producer_id);

    // If consumer id is broadcasted, make sure to propagate its concrete_id(s)
    // to producer
    for (const auto& consumer_bcast_key : consumer_bcast_domains) {
      const auto concrete_id = consumer_bcast_key.concreteId();
      const DomainKey producer_bcast_key(producer_td, producer_id, concrete_id);
      producer_domains.insert(concrete_id);
      addToPendingList(producer_bcast_key, consumer_bcast_key);
    }
  } else {
    auto producer_concrete_key = producer_key;
    if (producer_id->isBroadcast()) {
      const auto concrete_id = consumer_id;
      auto& producer_domains =
          root_map_.getConcretizedDomains(producer_td, producer_id);
      producer_concrete_key = DomainKey(producer_td, producer_id, concrete_id);
      producer_domains.insert(concrete_id);
    }
    addToPendingList(producer_concrete_key, consumer_key);
  }
}

void ComputeAtRootDomainMapBuilder::handle(Expr* e) {
  // Avoid visiting expressions multiple times
  if (visited_.find(e) != visited_.end()) {
    return;
  }
  BackwardVisitor::handle(e);
  visited_.insert(e);
}

void ComputeAtRootDomainMapBuilder::mapPointwiseOrReductionOp(Expr* e) {
  if (e->output(0)->getValType() != ValType::TensorView) {
    return;
  }

  // Broadcast is handled separately, so e should never be BroadcastOp.
  TORCH_INTERNAL_ASSERT(e->getExprType() != ExprType::BroadcastOp);
  TORCH_INTERNAL_ASSERT(e->getExprType() != ExprType::SqueezeOp);

  TORCH_INTERNAL_ASSERT(e->outputs().size() >= 1);
  const TensorView* out_tv = e->output(0)->as<TensorView>();
  const TensorDomain* out_td = out_tv->domain();
  const auto& out_root = out_td->getRootDomain();

  // Record equalities from output to all the inputs
  // ignores non-concretizable broadcasts
  for (auto* in_tv : ir_utils::filterByType<TensorView>(e->inputs())) {
    const TensorDomain* in_td = in_tv->domain();
    std::vector<IterDomain*> in_root =
        TensorDomain::noReductions(in_tv->getMaybeRFactorDomain());
    TORCH_INTERNAL_ASSERT(
        in_root.size() == out_root.size(),
        "\nExpression: ",
        e,
        "\nInput root domain: ",
        in_root,
        "\nOutput root domain: ",
        out_root);
    for (const auto it : c10::irange(in_root.size())) {
      if (e->outputs().size() > 1) {
        TORCH_INTERNAL_ASSERT(
            e->isA<WelfordOp>() || e->isA<GroupedReductionOp>() ||
                e->isA<GroupedWelfordOp>(),
            "Unknown multi-output Expr type ",
            e->getExprType().value(),
            " is found");
        for (auto out : e->outputs()) {
          auto out_tv = out->as<TensorView>();
          auto out_td = out_tv->domain();
          auto out_root = out_td->getRootDomain();
          setMaybeMapped(in_td, in_root[it], out_td, out_root[it]);
        }
      } else {
        setMaybeMapped(in_td, in_root[it], out_td, out_root[it]);
      }
    }
  }
}

void ComputeAtRootDomainMapBuilder::handle(BroadcastOp* op) {
  const TensorDomain* in_td = op->in()->as<TensorView>()->domain();
  const TensorDomain* out_td = op->out()->as<TensorView>()->domain();
  const auto in_root =
      TensorDomain::noReductions(in_td->getMaybeRFactorDomain());
  const auto& out_root = out_td->getRootDomain();
  const auto& bcast_dim_flags = op->getBroadcastDimFlags();
  TORCH_INTERNAL_ASSERT(
      out_root.size() == bcast_dim_flags.size(),
      "dim flags: ",
      bcast_dim_flags,
      ", out root: ",
      out_root);
  auto in_it = in_root.begin();
  auto out_it = out_root.begin();
  while (in_it != in_root.end() && out_it != out_root.end()) {
    if (bcast_dim_flags.at(std::distance(out_root.begin(), out_it))) {
      // new broadcast dim. No matching dimension in the input
      // tensor.
      root_map_.new_broadcast_domains_.insert(DomainKey(out_td, *out_it));
      ++out_it;
      continue;
    }
    setMaybeMapped(in_td, *in_it, out_td, *out_it);
    ++in_it;
    ++out_it;
  }
  // At this point, the input domain should have been scanned
  // entirely.
  TORCH_INTERNAL_ASSERT(
      in_it == in_root.end(),
      "Unmatched domain detected: ",
      *in_it,
      " of ",
      in_td);
  // On the other hand, the output may still have some domains left,
  // and they must be new broadcast domains.
  for (; out_it != out_root.end(); ++out_it) {
    TORCH_INTERNAL_ASSERT(
        bcast_dim_flags.at(std::distance(out_root.begin(), out_it)),
        "Unmatched domain detected: ",
        *out_it,
        " of ",
        out_td);
    root_map_.new_broadcast_domains_.insert(DomainKey(out_td, *out_it));
  }
}

void ComputeAtRootDomainMapBuilder::handle(SqueezeOp* op) {
  const TensorDomain* in_td = op->in()->as<TensorView>()->domain();
  const TensorDomain* out_td = op->out()->as<TensorView>()->domain();
  const auto in_root =
      TensorDomain::noReductions(in_td->getMaybeRFactorDomain());
  const auto& out_root = out_td->getRootDomain();
  const auto& squeeze_dim_flags = op->getSqueezeDimFlags();
  TORCH_INTERNAL_ASSERT(
      in_root.size() == squeeze_dim_flags.size(),
      "dim flags: ",
      squeeze_dim_flags,
      ", in root: ",
      in_root);
  auto in_it = in_root.begin();
  auto out_it = out_root.begin();
  while (in_it != in_root.end() && out_it != out_root.end()) {
    if (squeeze_dim_flags.at(std::distance(in_root.begin(), in_it))) {
      // new broadcast dim. No matching dimension in the input
      // tensor.
      root_map_.removed_broadcast_domains_.insert(DomainKey(in_td, *in_it));
      ++in_it;
      continue;
    }
    setMaybeMapped(in_td, *in_it, out_td, *out_it);
    ++in_it;
    ++out_it;
  }
  // At this point, the output domain should have been scanned
  // entirely.
  TORCH_INTERNAL_ASSERT(
      out_it == out_root.end(),
      "Unmatched domain detected: ",
      *out_it,
      " of ",
      out_td);
  // On the other hand, the input may still have some domains left,
  // and they must be removed broadcast domains.
  for (; in_it != in_root.end(); ++in_it) {
    TORCH_INTERNAL_ASSERT(
        squeeze_dim_flags.at(std::distance(in_root.begin(), in_it)),
        "Unmatched domain detected: ",
        *in_it,
        " of ",
        in_td);
    root_map_.removed_broadcast_domains_.insert(DomainKey(in_td, *in_it));
  }
}

void ComputeAtRootDomainMapBuilder::handle(ViewAsScalar* op) {
  const TensorView* out_tv = op->output(0)->as<TensorView>();
  const TensorDomain* out_td = out_tv->domain();
  const auto& out_root = out_td->getRootDomain();

  const TensorView* in_tv = op->input(0)->as<TensorView>();
  const TensorDomain* in_td = in_tv->domain();

  std::vector<IterDomain*> in_root =
      TensorDomain::noReductions(in_tv->getMaybeRFactorDomain());
  TORCH_INTERNAL_ASSERT(
      in_root.size() + 1 == out_root.size(),
      "\nExpression: ",
      op,
      "\nInput root domain: ",
      in_root,
      "\nOutput root domain: ",
      out_root);
  auto in_it = in_root.begin();
  auto out_it = out_root.begin();
  while (in_it != in_root.end() && out_it != out_root.end()) {
    setMaybeMapped(in_td, *in_it, out_td, *out_it);
    ++in_it;
    ++out_it;
  }
  TORCH_INTERNAL_ASSERT(
      (*out_it)->isVectorComponent(),
      "The last dim of ViewDtypeOp's output must be a ViewAsScalar");
}

void ComputeAtRootDomainMapBuilder::handle(TransposeOp* op) {
  const TensorDomain* in_td = op->in()->as<TensorView>()->domain();
  std::vector<IterDomain*> in_root =
      TensorDomain::noReductions(in_td->getMaybeRFactorDomain());

  const TensorDomain* out_td = op->out()->as<TensorView>()->domain();
  const auto& out_root = out_td->getRootDomain();

  TORCH_INTERNAL_ASSERT(in_root.size() == out_root.size());

  const auto& new2old = op->new2old();

  for (const auto it : c10::irange(out_root.size())) {
    setMaybeMapped(in_td, in_root[new2old[it]], out_td, out_root[it]);
  }
}

void ComputeAtRootDomainMapBuilder::handle(GatherOp* op) {
  const TensorDomain* in_td = op->in()->as<TensorView>()->domain();
  const TensorDomain* out_td = op->out()->as<TensorView>()->domain();
  const auto in_root =
      TensorDomain::noReductions(in_td->getMaybeRFactorDomain());
  const auto& out_root = out_td->getRootDomain();

  // Only maps the input root axes. Do not map the new window axes.
  for (const auto it : c10::irange(in_root.size())) {
    setMaybeMapped(in_td, in_root[it], out_td, out_root[it]);
  }

  // Keep track of window axes so that they can be skipped when
  // mapping root domains
  for (const auto it : c10::irange(in_root.size(), out_root.size())) {
    root_map_.window_axes_.insert(out_root[it]);
  }
}

void ComputeAtRootDomainMapBuilder::mapAllPendingMappings(
    const DomainKey& key) {
  auto it = pending_map_.find(key);
  if (it == pending_map_.end()) {
    return;
  }
  const auto& pending_set = it->second;
  // All entries in key_set must be equivalent with each other.
  TORCH_INTERNAL_ASSERT(pending_set.size() > 0);
  bool consistent = safeToMap(pending_set);
  for (const auto pending_key : pending_set) {
    if (consistent) {
      setMapped(key, pending_key);
    } else {
      setInvalid(key, pending_key);
    }
  }
  // This entry should never be used again, so remove it.
  pending_map_.erase(it);
}

void ComputeAtRootDomainMapBuilder::mapAllPendingMappings(
    const TensorDomain* td,
    IterDomain* id) {
  if (id->isBroadcast()) {
    for (const auto& key : root_map_.getConcretizedKeys(td, id)) {
      mapAllPendingMappings(key);
    }
  } else {
    mapAllPendingMappings(DomainKey(td, id));
  }
}

void ComputeAtRootDomainMapBuilder::handle(RNGOp* rop) {
  handle(rop->output(0)->as<TensorView>());
}

void ComputeAtRootDomainMapBuilder::handle(TensorView* tv) {
  const TensorDomain* td = tv->domain();
  const auto rfactor = TensorDomain::noReductions(td->getMaybeRFactorDomain());
  for (auto id : rfactor) {
    if (id->isBroadcast()) {
      initializeBcastMap(tv, id);
    }
    mapAllPendingMappings(td, id);
  }

  // When tv has an rfactor domain, propagate the domain mappings from
  // each of the rfactor axes to the dependent root axes.
  if (td->hasViewLikeRFactor()) {
    std::unordered_set<Val*> root_set(
        {td->getRootDomain().begin(), td->getRootDomain().end()});
    for (auto rf_id : rfactor) {
      if (!rf_id->isRFactorProduct()) {
        continue;
      }
      auto dep = DependencyCheck::getAllValsBetween(root_set, {rf_id});
      for (auto id : ir_utils::filterByType<IterDomain>(dep)) {
        if (root_set.find(id) == root_set.end() || rf_id == id) {
          continue;
        }
        setMaybeMapped(td, id, td, rf_id);
      }
    }
    // Once mappings for rfactor axes are propagated to root axes,
    // aggregates them at each root axis
    for (auto id : tv->getRootDomain()) {
      if (id->isBroadcast()) {
        // There can be broadcast domains that appear at root domains but
        // are removed at rfactor domains as they are merged into
        // non-reduction domains. Initialize the map for those broadcast
        // domains.
        initializeBcastMap(tv, id);
      }
      mapAllPendingMappings(td, id);
    }
  }
}

// Checks whether all consumers of a producer can be joined without
// introducing unsupported mappings, i.e., requiring recomputations.
bool ComputeAtRootDomainMapBuilder::safeToMap(const DomainKeySet& domains) {
  if (domains.size() <= 1) {
    return true;
  }

  // Can't map if reduction output domains would be mapped
  if (incompatible_domains_.isReductionOutputMapped(domains, root_map_) &&
      !map_through_reduction_) {
    return false;
  }
  // Make sure mapping these domains won't cause any invalid mapping
  if (isInvalid(domains)) {
    return false;
  }
  return true;
}

namespace {
class ExactRootDomainMapBuilder : private IterVisitor {
 public:
  ExactRootDomainMapBuilder(
      Fusion* fusion,
      DisjointSets<const IterDomain*>& eq_sets)
      : eq_sets_(eq_sets) {
    traverseTo(fusion, fusion->outputs());
  }

 private:
  using IterVisitor::handle;

  void handle(Expr* expr) final {
    for (auto producer : ir_utils::filterByType<TensorView>(expr->inputs())) {
      for (auto consumer :
           ir_utils::filterByType<TensorView>(expr->outputs())) {
        PairwiseRootDomainMap pwise_map(producer, consumer, true);
        const auto mappings = pwise_map.mapProducerToConsumer(
            producer->domain(), consumer->domain());
        for (const auto& mapping : mappings) {
          eq_sets_.mapEntries(mapping.first, mapping.second);
        }
      }
    }
  }

 private:
  DisjointSets<const IterDomain*>& eq_sets_;
};

} // namespace

ExactRootDomainMap::ExactRootDomainMap(Fusion* fusion) {
  ExactRootDomainMapBuilder builder(fusion, eq_sets_);
}

bool ExactRootDomainMap::areMapped(
    const IterDomain* id_a,
    const IterDomain* id_b) const {
  // With expand going into a view operation there can be an instance where an
  // iteration root domain in the consumer resolves the broadcast from the
  // producer, then immediately rfactors it. In this case the consumer root is
  // not mapped exactly to any other domain, so it might no have an entry in
  // eq_sets_. eq_sets_.strictAreMapped would throw in this case so just return
  // false if a mapping doesn't exist.
  if (!eq_sets_.mappingExists(id_a) || !eq_sets_.mappingExists(id_b)) {
    return false;
  }
  return eq_sets_.strictAreMapped(id_a, id_b);
}

std::unordered_map<IterDomain*, IterDomain*> ExactRootDomainMap::map(
    const TensorDomain* producer,
    const TensorDomain* consumer,
    const std::unordered_set<IterDomain*>& root_dims_to_map,
    bool producer_to_consumer) const {
  const auto& producer_root =
      TensorDomain::noReductions(producer->getMaybeRFactorDomain());
  const auto& consumer_root = consumer->getRootDomain();
  const auto& from_ids = producer_to_consumer ? producer_root : consumer_root;
  const auto& to_ids = producer_to_consumer ? consumer_root : producer_root;

  std::unordered_map<IterDomain*, IterDomain*> id_map;

  for (auto& from_id : from_ids) {
    if (root_dims_to_map.find(from_id) == root_dims_to_map.end()) {
      continue;
    }
    for (const auto& to_id : to_ids) {
      if (areMapped(from_id, to_id)) {
        TORCH_INTERNAL_ASSERT(
            id_map.insert({from_id, to_id}).second,
            "Multiple matching ID detected for ",
            from_id);
      }
    }
  }

  return id_map;
}

std::string ExactRootDomainMap::toString() const {
  return eq_sets_.toString();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
