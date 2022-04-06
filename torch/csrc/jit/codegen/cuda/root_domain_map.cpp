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
    const TensorView* consumer)
    : producer_tv_(producer), consumer_tv_(consumer) {
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

  std::unordered_map<IterDomain*, IterDomain*> dom_map;
  const auto producer_root =
      TensorDomain::noReductions(producer->getMaybeRFactorDomain());
  const auto& consumer_root = consumer->getRootDomain();
  size_t itc = 0, itp = 0;
  while (itc < consumer_root.size() && itp < producer_root.size()) {
    IterDomain* producer_id = producer_root[itp];
    IterDomain* consumer_id = consumer_root[itc];

    // When the consumer ID is a new broadcast domain, there is no
    // mapping for it.
    if (!broadcast_flags.empty() && broadcast_flags.at(itc)) {
      TORCH_INTERNAL_ASSERT(consumer_id->isBroadcast());
      itc++;
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
    IterDomain* map_key_id = producer_root[new2old[i]];
    IterDomain* map_value_id = consumer_root[i];
    if (!producer_to_consumer) {
      std::swap(map_key_id, map_value_id);
    }
    if (root_dims_to_map.find(map_key_id) != root_dims_to_map.end()) {
      dom_map.insert(std::make_pair(map_key_id, map_value_id));
    }
  }
  return dom_map;
}

std::string toString(const PairwiseRootDomainMap& root_map) {
  std::stringstream ss;
  ss << "{producer: " << root_map.producer()
     << ", consumer: " << root_map.consumer() << "}";
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

} // namespace

std::string toString(const DomainKey& key) {
  std::stringstream ss;
  ss << "{";
  if (key.td()) {
    ss << key.td() << " (root: " << key.td()->getRootDomain()
       << ", maybe rfactor: " << key.td()->getMaybeRFactorDomain() << ")";
  } else {
    ss << "null";
  }
  ss << ", ";
  if (key.id()) {
    ss << key.id();
  } else {
    ss << "null";
  }
  if (key.concreteId()) {
    ss << " (" << key.concreteId() << ")";
  }
  ss << "}";
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
    input_keys.insert(DomainKey(tv_->domain(), id));
  }

  DomainKeySet find() {
    traverseFrom(tv_->fusion(), {tv_});
    return input_keys;
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
      if (input_keys.find(out_key) == input_keys.end()) {
        continue;
      }
      auto input_id_it = c2p.find(root_dom);
      if (input_id_it == c2p.end()) {
        continue;
      }
      DomainKey input_key(in_tv->domain(), input_id_it->second);
      input_keys.insert(input_key);
    }
  }

 private:
  TensorView* tv_ = nullptr;
  DomainKeySet input_keys;

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
    const std::vector<DomainKey>& consumer_domains,
    const ComputeAtRootDomainMap& root_map) const {
  for (const auto& kv : reduction_domains_) {
    const DomainKey& reduction_domain = kv.first;
    const DomainKeySet& incompatible_domains = kv.second;
    DomainKey consumer_domain_with_reduction;
    bool reduction_found = false;
    const auto& input_keys = reduction_domain_inputs_.at(reduction_domain);
    for (const DomainKey& consumer_domain : consumer_domains) {
      for (const auto& input_key : input_keys) {
        if (input_key == consumer_domain) {
          consumer_domain_with_reduction = consumer_domain;
          reduction_found = true;
          break;
        }
      }
    }
    if (!reduction_found) {
      continue;
    }
    // Make sure no incompatible domains will be merged with the reduction
    // domain.
    for (const auto& consumer_domain : consumer_domains) {
      if (consumer_domain == consumer_domain_with_reduction) {
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

void ComputeAtRootDomainMap::build(bool map_through_reduction) {
  // Make sure we start from scratch. Throw away previous results.
  eq_set_.clear();
  bcast_map_.clear();
  new_broadcast_domains_.clear();
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
  return key_a == key_b || eq_set_.areEquivalent(key_a, key_b);
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

  for (const auto& key : eq_set_.getAllElements()) {
    if (key.td() == td) {
      DomainKey alias_key(td_alias, key.id(), key.concreteId());
      eq_set_.join(key, alias_key);
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
}

std::vector<DomainKey> ComputeAtRootDomainMap::getConcretizedKeys(
    const TensorDomain* td,
    const IterDomain* id) const {
  DomainKey key(td, id);
  auto it = bcast_map_.find(key);
  TORCH_INTERNAL_ASSERT(it != bcast_map_.end(), "Not found: ", toString(key));
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
  TORCH_INTERNAL_ASSERT(it != bcast_map_.end(), "Not found: ", toString(key));
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
    // Matching ID not found. It's an error unless from_id is a new
    // broadcast of a consumer domain; or from_id is a window axis of
    // a consumer domain. Note that reduction domains are removed from
    // the producer root domain.
    if (!producer_to_consumer &&
        (new_broadcast_domains_.find(DomainKey(from_td, from_id)) !=
             new_broadcast_domains_.end() ||
         (window_axes_.count(from_id) > 0))) {
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
        toString(*this));
  }
  return id_map;
}

std::unordered_set<IterDomain*> ComputeAtRootDomainMap::getMappableDims(
    const TensorDomain* producer,
    const TensorDomain* consumer) const {
  const auto& producer_root = producer->getMaybeRFactorDomain();
  const auto& consumer_root = consumer->getRootDomain();

  std::unordered_map<IterDomain*, IterDomain*> id_map =
      mapBestEffort(producer, producer_root, consumer, consumer_root);

  std::unordered_set<IterDomain*> mappable_ids;

  for (auto& from_id : producer_root) {
    if (id_map.find(from_id) != id_map.end()) {
      mappable_ids.emplace(from_id);
      mappable_ids.emplace(id_map.at(from_id));
    }
  }
  return mappable_ids;
}

std::string toString(const ComputeAtRootDomainMap& root_map) {
  std::stringstream ss;
  root_map.eq_set_.print(ss);
  return ss.str();
}

ComputeAtRootDomainMapBuilder::ComputeAtRootDomainMapBuilder(
    ComputeAtRootDomainMap& root_map,
    bool map_through_reduction)
    : BackwardVisitor(false),
      root_map_(root_map),
      map_through_reduction_(map_through_reduction) {
  Fusion* fusion = FusionGuard::getCurFusion();
  TORCH_INTERNAL_ASSERT(fusion != nullptr);
  traverseFrom(fusion, fusion->outputs(), false);
  if (!pending_map_.empty()) {
    std::stringstream ss;
    ss << "pending map:\n";
    for (auto& kv : pending_map_) {
      ss << "\t" << toString(kv.first) << "\n";
      for (auto& dk : kv.second) {
        ss << "\t\t" << toString(dk) << "\n";
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

  // This initialization should be only used for fusion output tensors and
  // outputs of multi-consumer expressions that are not fusion outputs.
  TORCH_INTERNAL_ASSERT(
      tv->isFusionOutput() || tv->definition()->outputs().size() > 1,
      "Invalid tensor to initialize bcast map: t",
      tv->name());
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
  root_map_.eq_set_.join(producer, consumer);
}

void ComputeAtRootDomainMapBuilder::setInvalid(
    const DomainKey& key1,
    const DomainKey& key2) {
  invalid_mappings_.emplace_back(key1, key2);
}

bool ComputeAtRootDomainMapBuilder::isInvalid(
    const std::vector<DomainKey>& domains) const {
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
  for (const auto i : c10::irange(num_keys)) {
    const auto& key_i = domains[i];
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
      const auto& key_j = domains[j];
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
    TORCH_INTERNAL_ASSERT(
        !consumer_id->isBroadcast(),
        "No concrete domain found for a broadcast domain: ",
        toString(consumer_key));
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

  TORCH_INTERNAL_ASSERT(e->outputs().size() >= 1);
  const TensorView* out_tv = e->output(0)->as<TensorView>();
  const TensorDomain* out_td = out_tv->domain();
  const auto& out_root = out_td->getRootDomain();

  // Record equalities from output to all the inputs
  // ignores un-concretizable broadcasts
  for (auto* i : ir_utils::filterByType<TensorView>(e->inputs())) {
    const TensorDomain* in_td = i->domain();
    std::vector<IterDomain*> in_root =
        TensorDomain::noReductions(i->getMaybeRFactorDomain());
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
            e->isA<WelfordOp>(), "Only supported multioutput op is welford");
        for (auto o : e->outputs()) {
          auto o_tv = o->as<TensorView>();
          auto o_td = o_tv->domain();
          auto o_root = o_td->getRootDomain();
          setMaybeMapped(in_td, in_root[it], o_td, o_root[it]);
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

bool ComputeAtRootDomainMapBuilder::mapAllConsumers(
    const DomainKey& producer_key) {
  auto it = pending_map_.find(producer_key);
  if (it == pending_map_.end()) {
    return false;
  }
  const auto& consumer_set = it->second;
  // All entries in key_set must be equivalent with each other.
  TORCH_INTERNAL_ASSERT(consumer_set.size() > 0);
  bool consistent = safeToMap(consumer_set);
  for (const auto pending_consumer : consumer_set) {
    if (consistent) {
      setMapped(producer_key, pending_consumer);
    } else {
      setInvalid(producer_key, pending_consumer);
    }
  }
  // This entry should never be used again, so remove it.
  pending_map_.erase(it);
  return consistent;
}

void ComputeAtRootDomainMapBuilder::handle(TensorView* tv) {
  const TensorDomain* td = tv->domain();
  const auto root = TensorDomain::noReductions(td->getMaybeRFactorDomain());
  for (auto id : root) {
    if (id->isBroadcast()) {
      initializeBcastMap(tv, id);
      for (const auto& key : root_map_.getConcretizedKeys(td, id)) {
        mapAllConsumers(key);
      }
    } else {
      mapAllConsumers(DomainKey(td, id));
    }
  }
}

// Checks whether all consumers of a producer can be joined without
// introducing unsupported mappings. Specifically, if a domain of a
// consumer has a mapped iteration domain in another consumer that
// does not correspond to the same producer iteration domain, mapping
// the consumer domains would result in the producer iteration domain
// mapped to two different consumer iteration domains, requiring
// recomputations.
bool ComputeAtRootDomainMapBuilder::hasMatchingDomains(
    const std::vector<DomainKey>& unique_domains) {
  for (const auto& key : unique_domains) {
    for (const auto& other_key : unique_domains) {
      if (key == other_key) {
        continue;
      }
      const auto& other_root = other_key.td()->getRootDomain();
      if (std::any_of(
              other_root.begin(), other_root.end(), [&](const IterDomain* id) {
                return root_map_.canMap(key, other_key.td(), id);
              })) {
        return true;
      }
    }
  }
  return false;
}

// Checks whether all consumers of a producer can be joined without
// introducing unsupported mappings, i.e., requiring recomputations.
bool ComputeAtRootDomainMapBuilder::safeToMap(const DomainKeySet& domains) {
  if (domains.size() <= 1) {
    return true;
  }
  // Filter out equivalent domains
  std::vector<DomainKey> unique_domains;
  for (const auto& domain : domains) {
    if (std::none_of(
            unique_domains.begin(),
            unique_domains.end(),
            [&](const auto& unique_dom) {
              return root_map_.canMap(domain, unique_dom);
            })) {
      unique_domains.push_back(domain);
    }
  }
  if (hasMatchingDomains(unique_domains)) {
    return false;
  }
  // Can't map if reduction output domains would be mapped
  if (incompatible_domains_.isReductionOutputMapped(
          unique_domains, root_map_) &&
      !map_through_reduction_) {
    return false;
  }
  // Make sure mapping these domains won't cause any invalid mapping
  if (isInvalid(unique_domains)) {
    return false;
  }
  return true;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
