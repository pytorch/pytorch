#include <maxinfo_propagator.h>
#include <root_domain_map.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

bool MaxInfoSpanningTree::Information::operator>(const Information& r) const {
  return r < *this;
}

bool MaxInfoSpanningTree::Information::operator==(const Information& r) const {
  return !(r < *this) && !(*this < r);
}

// Prim's algorithm
MaxInfoSpanningTree::MaxInfoSpanningTree(
    TensorView* reference,
    std::shared_ptr<Information> reference_info,
    Selector* selector)
    : reference_(reference),
      reference_info_(reference_info),
      selector_(selector) {}

void MaxInfoSpanningTree::compute_spanning_tree() {
  // A set that allows us to quickly tell if a tensor has been replayed. If yes,
  // then we will not bother computing if a new path to this tensor is worth
  // taking (because the answer is always not worth)
  std::unordered_set<TensorView*> replayed;

  // A sorted list of possible next steps. The list is sorted in the order of
  // ascending amount of preserved information about the reference tensor. The
  // back of the list preserves the most amount of information about the
  // reference tensor, and should always be the next step to take. We use
  // std::list instead of std::priority_queue because C++'s
  // std::priority_queue does not support increase-key, and might not be
  // deterministic either.
  std::list<NextHopWithInfo> candidates(1);
  candidates.back().next_hop.from = nullptr;
  candidates.back().next_hop.to = reference_;
  candidates.back().info_to = reference_info_;

  // Insert the given next hop the correct position in `candidates`. If there
  // is an existing next hop that preserves more information, then we will just
  // discard `info`.
  auto insertNextHop = [&](const NextHopWithInfo& info) {
    if (!*(info.info_from)) {
      // When there is no more information about the starting tensor,
      // we are not interested in continuing the path-finding.
      return;
    }
    // Find if there is already a path to the dest tensor
    auto existing = std::find_if(
        candidates.begin(), candidates.end(), [&](const NextHopWithInfo& i) {
          return i.next_hop.to == info.next_hop.to;
        });
    // Only insert if there is no existing path to the dest tensor, or the new
    // path preserves more information about the starting tensor.
    if (existing == candidates.end() || *existing < info) {
      if (existing != candidates.end()) {
        candidates.erase(existing);
      }
      auto pos = std::upper_bound(candidates.begin(), candidates.end(), info);
      candidates.insert(pos, info);
    }
  };

  auto allowC2P = [this](TensorView* from, TensorView* to) {
    if (selector_ == nullptr) {
      return true;
    }
    return selector_->allowC2P(from, to);
  };

  auto allowP2C = [this](TensorView* from, TensorView* to) {
    if (selector_ == nullptr) {
      return true;
    }
    return selector_->allowP2C(from, to);
  };

  auto allowSibling = [this](TensorView* from, TensorView* to) {
    if (selector_ == nullptr) {
      return true;
    }
    return selector_->allowSibling(from, to);
  };

  while (!candidates.empty()) {
    const auto next_hop_info = candidates.back();
    const auto& next_hop = next_hop_info.next_hop;
    candidates.pop_back();

    if (next_hop.from != nullptr) {
      // nullptr used to start from reference
      path_.push_back(next_hop);
    }
    replayed.emplace(next_hop.to);

    for (auto sibling_tv : ir_utils::siblingTvsOf(next_hop.to)) {
      if (replayed.count(sibling_tv) ||
          !allowSibling(next_hop.to, sibling_tv)) {
        continue;
      }
      insertNextHop(NextHopWithInfo(
          NextHop(NextHopType::SIBLING, next_hop.to, sibling_tv),
          next_hop_info.info_to,
          computeInfoSibling(next_hop.to, sibling_tv, next_hop_info.info_to)));
    }

    for (auto consumer_tv : ir_utils::consumerTvsOf(next_hop.to)) {
      if (replayed.count(consumer_tv) || !allowP2C(next_hop.to, consumer_tv)) {
        continue;
      }
      insertNextHop(NextHopWithInfo(
          NextHop(NextHopType::C_AS_P, next_hop.to, consumer_tv),
          next_hop_info.info_to,
          computeInfoCasP(next_hop.to, consumer_tv, next_hop_info.info_to)));
    }

    for (auto producer_tv : ir_utils::producerTvsOf(next_hop.to)) {
      if (replayed.count(producer_tv) || !allowC2P(next_hop.to, producer_tv)) {
        continue;
      }
      insertNextHop(NextHopWithInfo(
          NextHop(NextHopType::P_AS_C, next_hop.to, producer_tv),
          next_hop_info.info_to,
          computeInfoPasC(next_hop.to, producer_tv, next_hop_info.info_to)));
    }
  }
}

void MaxInfoSpanningTree::traverse(Propagator* propagator) {
  if (path_.empty()) {
    compute_spanning_tree();
  }
  propagator->setUp();
  for (const auto& next_hop : path_) {
    switch (next_hop.type) {
      case NextHopType::SIBLING:
        propagator->propagateSibling(next_hop.from, next_hop.to);
        break;
      case NextHopType::C_AS_P:
        propagator->propagateP2C(next_hop.from, next_hop.to);
        break;
      case NextHopType::P_AS_C:
        propagator->propagateC2P(next_hop.from, next_hop.to);
        break;
    }
  }
  propagator->tearDown();
}

MaxRootDomainInfoSpanningTree::RootDomainInfo::operator bool() const {
  return !info.empty();
}

bool MaxRootDomainInfoSpanningTree::RootDomainInfo::operator<(
    const Information& r) const {
  auto rr = dynamic_cast<const RootDomainInfo&>(r);
  if (info.size() != rr.info.size()) {
    return info.size() < rr.info.size();
  }
  size_t l_complete =
      std::count_if(info.begin(), info.end(), [](const RootIDInfo& i) {
        return i.is_complete;
      });
  size_t r_complete =
      std::count_if(rr.info.begin(), rr.info.end(), [](const RootIDInfo& i) {
        return i.is_complete;
      });
  return l_complete < r_complete;
}

namespace {

// Given `root_ids`, a list of IDs in the root domain of `tv`, find their
// corresponding IDs in the rfactor domain of `tv`.
std::unordered_set<IterDomain*> mapRootToRFactor(
    TensorView* tv,
    const std::unordered_set<IterDomain*>& root_ids) {
  std::unordered_set<IterDomain*> mapped_rfactor_ids;
  const auto& rfactor_dom = tv->getMaybeRFactorDomain();
  for (auto id : rfactor_dom) {
    if (root_ids.count(id) > 0) {
      mapped_rfactor_ids.emplace(id);
      continue;
    }
    for (auto root_id : root_ids) {
      if (id == root_id || DependencyCheck::isDependencyOf(root_id, id)) {
        mapped_rfactor_ids.emplace(id);
        break;
      }
    }
  }
  return mapped_rfactor_ids;
}

// Given `rfactor_ids`, a list of IDs in the rfactor domain of `tv`, find their
// corresponding IDs in the root domain of `tv`.
std::unordered_set<IterDomain*> mapRFactorToRoot(
    TensorView* tv,
    const std::unordered_set<IterDomain*>& rfactor_ids) {
  std::unordered_set<IterDomain*> mapped_root_ids;
  for (auto id : tv->getRootDomain()) {
    if (rfactor_ids.count(id) > 0) {
      mapped_root_ids.emplace(id);
      continue;
    }
    for (auto rfactor_id : rfactor_ids) {
      if (DependencyCheck::isDependencyOf(id, rfactor_id)) {
        mapped_root_ids.emplace(id);
        break;
      }
    }
  }
  return mapped_root_ids;
}

} // namespace

// Given the preserved reference root ID info of a producer, compute
// the corresponding info in consumer. The given info may be represented by
// producer's root domain, or rfactor domain, depending on how we reached the
// producer during path-finding. If the given info is already represented with
// producer's rfactor domain, then we directly map it to the consumer's root
// domain. If the given info is represented with producer's root domain, we need
// to first map it to the rfactor domain of the producer, then we can map it to
// the consumer's root domain. The computed info will be represented by root
// domain as root domain contains the raw information.
std::shared_ptr<MaxInfoSpanningTree::Information> MaxRootDomainInfoSpanningTree::
    computeInfoCasP(
        TensorView* from,
        TensorView* to,
        std::shared_ptr<Information> from_info) const {
  RootDomainInfo result;

  TensorView* producer = from;
  TensorView* consumer = to;
  const auto& producer_root_id_info =
      std::dynamic_pointer_cast<RootDomainInfo>(from_info)->info;

  auto pairwise_map = PairwiseRootDomainMap(producer, consumer);
  auto p2c_map = pairwise_map.mapProducerToConsumer(
      producer->domain(), consumer->domain());

  for (auto& info : producer_root_id_info) {
    RootIDInfo consumer_info;
    consumer_info.is_complete = info.is_complete;
    consumer_info.is_rfactor = false;

    // mapped root ids in producer -> mapped rfactor ids in producer
    std::unordered_set<IterDomain*> producer_mapped_rfactor_ids;
    if (producer->hasRFactor() && !info.is_rfactor) {
      producer_mapped_rfactor_ids = mapRootToRFactor(producer, info.mapped_ids);
    } else {
      producer_mapped_rfactor_ids = info.mapped_ids;
    }

    // mapped rfactor ids in producer -> mapped root ids in consumer
    for (auto producer_id : producer_mapped_rfactor_ids) {
      auto it = p2c_map.find(producer_id);
      if (it != p2c_map.end()) {
        consumer_info.mapped_ids.insert(it->second);
      } else {
        consumer_info.is_complete = false;
      }
    }

    // If at least one root id in the consumer contains information
    // of this starting root id, then keep this record
    if (!consumer_info.mapped_ids.empty()) {
      result.info.push_back(consumer_info);
    }
  }
  return std::make_shared<RootDomainInfo>(std::move(result));
}

// Given the preserved reference root ID info of a consumer, compute
// the corresponding info in producer. The given info may be represented by
// consumer's root domain, or rfactor domain, depending on how we reached the
// consumer during path-finding. If the given info is already represented with
// consumer's root domain, then we directly map it to the producer's rfactor
// domain. If the given info is represented with consumer's rfactor domain, we
// need to first map it to the root domain of the consumer, then we can map it
// to the producer's rfactor domain. The computed info will be represented by
// rfactor domain as rfactor domain contains the raw information.
std::shared_ptr<MaxInfoSpanningTree::Information> MaxRootDomainInfoSpanningTree::
    computeInfoPasC(
        TensorView* from,
        TensorView* to,
        std::shared_ptr<Information> from_info) const {
  RootDomainInfo result;

  TensorView* producer = to;
  TensorView* consumer = from;
  const auto& consumer_root_id_info =
      std::dynamic_pointer_cast<RootDomainInfo>(from_info)->info;

  auto pairwise_map = PairwiseRootDomainMap(producer, consumer);
  auto c2p_map = pairwise_map.mapConsumerToProducer(
      consumer->domain(), producer->domain());

  for (auto& info : consumer_root_id_info) {
    RootIDInfo producer_info;
    producer_info.is_complete = info.is_complete;
    producer_info.is_rfactor = true;

    // mapped rfactor ids in consumer -> mapped root ids in consumer
    std::unordered_set<IterDomain*> consumer_mapped_root_ids;
    if (info.is_rfactor && consumer->hasRFactor()) {
      consumer_mapped_root_ids = mapRFactorToRoot(consumer, info.mapped_ids);
    } else {
      consumer_mapped_root_ids = info.mapped_ids;
    }

    // mapped root ids in consumer -> mapped rfactor ids in producer
    for (auto consumer_id : consumer_mapped_root_ids) {
      auto it = c2p_map.find(consumer_id);
      if (it != c2p_map.end()) {
        producer_info.mapped_ids.insert(it->second);
      } else {
        producer_info.is_complete = false;
      }
    }

    // We will stop at the rfactor ids in producer, and will not further map
    // them into root ids in producer. This means, we only keep the unprocessed
    // raw information of a tensor. This behavior is important to make sure that
    // info is as accurate as possible throughout the path-finding.
    //
    // For example, in a C->P->C' path, we want to do
    //   C(root) -> P(rfactor) -> C'(root)
    // instead of
    //   C(root) -> P(rfactor) -> P(root) -> P(rfactor) -> C'(root)
    //
    // and the above two paths do lead to different results:
    //
    // For example if you have a producer tensor
    //   root domain: [I1, I2]
    //   rfactor domain: [I3, I5]
    // where I3, I4 = split(I1), I5 = merge(I4, I2)
    // Then the P(rfactor) -> P(root) -> P(rfactor) could lead to
    // P(rfactor: {I5}) -> P(root: {I1, I2}) -> P(rfactor: {I3, I5})
    // which is not correct

    // If at least one root id in the producer contains information
    // of this starting root id, then keep this record
    if (!producer_info.mapped_ids.empty()) {
      result.info.push_back(producer_info);
    }
  }
  return std::make_shared<RootDomainInfo>(std::move(result));
}

std::shared_ptr<MaxRootDomainInfoSpanningTree::RootDomainInfo>
MaxRootDomainInfoSpanningTree::getReferenceRootIDInfo(TensorView* tv) {
  RootDomainInfo result;
  const auto& root_domain = tv->getRootDomain();
  result.info.reserve(root_domain.size());
  for (auto id : root_domain) {
    result.info.emplace_back(RootIDInfo{{id}, true, false});
  }
  return std::make_shared<RootDomainInfo>(std::move(result));
}

std::shared_ptr<MaxRootDomainInfoSpanningTree::RootDomainInfo>
MaxRootDomainInfoSpanningTree::getReferenceRootIDInfo(
    TensorView* tv,
    int64_t leaf_pos) {
  if (leaf_pos < 0) {
    leaf_pos += int64_t(tv->nDims()) + 1;
  }
  TORCH_CHECK(
      leaf_pos >= 0 && leaf_pos <= int64_t(tv->nDims()),
      "MaxRootDomainInfoSpanningTree called on an leaf_pos outside valid range.");
  RootDomainInfo result;
  const auto& root_domain = tv->getMaybeRFactorDomain();
  const auto& leaf_domain = tv->domain()->domain();
  std::unordered_set<IterDomain*> selected_leaves(
      leaf_domain.begin(), leaf_domain.begin() + leaf_pos);
  for (auto id : root_domain) {
    if (selected_leaves.count(id) > 0) {
      result.info.emplace_back(RootIDInfo{{id}, true, tv->hasRFactor()});
      continue;
    }
    for (auto selected_leaf_id : selected_leaves) {
      if (DependencyCheck::isDependencyOf(id, selected_leaf_id)) {
        result.info.emplace_back(RootIDInfo{{id}, true, tv->hasRFactor()});
        break;
      }
    }
  }
  return std::make_shared<RootDomainInfo>(std::move(result));
}

// Given the preserved reference root ID info of a tensor, compute
// the corresponding info in its sibling. Since info has nothing to do with
// replay state, so sibling info is always identical by definition.
std::shared_ptr<MaxInfoSpanningTree::Information> MaxRootDomainInfoSpanningTree::
    computeInfoSibling(
        TensorView* from,
        TensorView* to,
        std::shared_ptr<Information> from_info) const {
  return from_info;
}

void SpanningTreePrinter::propagateC2P(TensorView* from, TensorView* to) {
  stream_ << "propagateC2P" << std::endl;
  stream_ << "  from: " << from->toString() << std::endl;
  stream_ << "  to: " << to->toString() << std::endl;
}

void SpanningTreePrinter::propagateP2C(TensorView* from, TensorView* to) {
  stream_ << "propagateP2C" << std::endl;
  stream_ << "  from: " << from->toString() << std::endl;
  stream_ << "  to: " << to->toString() << std::endl;
}

void SpanningTreePrinter::propagateSibling(TensorView* from, TensorView* to) {
  stream_ << "propagateSibling" << std::endl;
  stream_ << "  from: " << from->toString() << std::endl;
  stream_ << "  to: " << to->toString() << std::endl;
}

bool SetSelector::allowC2P(TensorView* from, TensorView* to) {
  return selected_.count(to) > 0;
}

bool SetSelector::allowP2C(TensorView* from, TensorView* to) {
  return selected_.count(to) > 0;
}

bool SetSelector::allowSibling(TensorView* from, TensorView* to) {
  return true;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
