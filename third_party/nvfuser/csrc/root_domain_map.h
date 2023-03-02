#pragma once

#include <disjoint_set.h>
#include <ir_all_nodes.h>
#include <iter_visitor.h>
#include <utils.h>

#include <c10/macros/Export.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Generic interface for mapping root domains of a producer-consumer pair.
class TORCH_CUDA_CU_API RootDomainMap : public PolymorphicBase {
 public:
  //! Return a map from a producer TensorDomain to a consumer
  //! TensorDomain
  //!
  //! \param producer A producer TensorDomain
  //! \param consumer A consumer TensorDomain
  //! \param root_dims_to_map Maps only producer root domains in this set
  std::unordered_map<IterDomain*, IterDomain*> mapProducerToConsumer(
      const TensorDomain* producer,
      const TensorDomain* consumer,
      const std::unordered_set<IterDomain*>& root_dims_to_map) const;

  //! Return a map from a producer TensorDomain to a consumer
  //! TensorDomain
  //!
  //! \param producer A producer TensorDomain
  //! \param consumer A consumer TensorDomain
  std::unordered_map<IterDomain*, IterDomain*> mapProducerToConsumer(
      const TensorDomain* producer,
      const TensorDomain* consumer) const;

  //! Return a map from a consumer TensorDomain to a producer
  //! TensorDomain
  //!
  //! \param consumer A consumer TensorDomain
  //! \param producer A producer TensorDomain
  //! \param root_dims_to_map Maps only consumer root domains in this set
  std::unordered_map<IterDomain*, IterDomain*> mapConsumerToProducer(
      const TensorDomain* consumer,
      const TensorDomain* producer,
      const std::unordered_set<IterDomain*>& root_dims_to_map) const;

  //! Return a map from a consumer TensorDomain to a producer
  //! TensorDomain
  //!
  //! \param consumer A consumer TensorDomain
  //! \param producer A producer TensorDomain
  std::unordered_map<IterDomain*, IterDomain*> mapConsumerToProducer(
      const TensorDomain* consumer,
      const TensorDomain* producer) const;

 protected:
  //! Return a map between root IterDomains of a producer-consumer
  //! pair.
  //!
  //! \param producer A producer TensorDomain
  //! \param consumer A consumer TensorDomain
  //! \param root_dims_to_map Maps only from IterDomains in this set
  //! \param producer_to_consumer Maps from producer to consumer if true
  virtual std::unordered_map<IterDomain*, IterDomain*> map(
      const TensorDomain* producer,
      const TensorDomain* consumer,
      const std::unordered_set<IterDomain*>& root_dims_to_map,
      bool producer_to_consumer) const = 0;
};

//! Maps root domains of a producer-consumer pair. This class only
//! looks at the given pair of TensorViews and does not take into
//! consideration the constraints of the computeAt transformation,
//! i.e., unable to compute the same tensors multiple times. This
//! should not be used for transformations implementing computeAt, but
//! should be valid otherwise.
class TORCH_CUDA_CU_API PairwiseRootDomainMap : public RootDomainMap {
 public:
  //! \param producer The producer tensor of a producer-consumer pair.
  //! \param consumer The consumer tensor of a producer-consumer pair.
  explicit PairwiseRootDomainMap(
      const TensorView* producer,
      const TensorView* consumer,
      bool is_exact = false);

  const TensorView* producer() const {
    return producer_tv_;
  }

  const TensorView* consumer() const {
    return consumer_tv_;
  }

  std::string toString() const;

 protected:
  std::unordered_map<IterDomain*, IterDomain*> map(
      const TensorDomain* producer,
      const TensorDomain* consumer,
      const std::unordered_set<IterDomain*>& root_dims_to_map,
      bool producer_to_consumer) const override;

  std::unordered_map<IterDomain*, IterDomain*> mapTranspose(
      const TensorDomain* producer,
      const TensorDomain* consumer,
      const std::unordered_set<IterDomain*>& root_dims_to_map,
      bool producer_to_consumer) const;

 private:
  const TensorView* producer_tv_ = nullptr;
  const TensorView* consumer_tv_ = nullptr;
  //! If true, does not map broadcast IDs with non-broadcast IDs
  const bool is_exact_ = false;
};

//! Represents an iteration domain of a TensorDomain. Only used for
//! root domain mapping.
//!
//! Note that an IterDomain object may be reused
//! across multiple TensorDomains, but an IterDomain in a
//! TensorDomain may not be necessarily mappable to the same
//! IterDomain used in a different TensorDomain. Thus, for the purpose
//! of root domain mapping, an iteration domain needs to be identified
//! with an IterDomain and its TensorDomain.
class DomainKey {
 public:
  DomainKey() = default;
  DomainKey(
      const TensorDomain* td,
      const IterDomain* id,
      const IterDomain* concrete_id = nullptr)
      : td_(td), id_(id), concrete_id_(concrete_id) {}
  const TensorDomain* td() const {
    return td_;
  }
  const IterDomain* id() const {
    return id_;
  }
  const IterDomain* concreteId() const {
    return concrete_id_;
  }
  bool operator==(const DomainKey& other) const {
    return td() == other.td() && id() == other.id() &&
        concreteId() == other.concreteId();
  }
  bool operator!=(const DomainKey& other) const {
    return !(*this == other);
  }

  std::string toString() const;

 private:
  const TensorDomain* td_ = nullptr;
  const IterDomain* id_ = nullptr;
  const IterDomain* concrete_id_ = nullptr;
};

struct DomainKeyHash {
  std::size_t operator()(const DomainKey& key) const {
    return std::hash<const TensorDomain*>{}(key.td()) ^
        std::hash<const IterDomain*>{}(key.id());
  }
};

using DomainKeySet = std::unordered_set<DomainKey, DomainKeyHash>;

template <typename Mapped>
using DomainKeyMap = std::unordered_map<DomainKey, Mapped, DomainKeyHash>;

class ComputeAtRootDomainMap;

//! A helper class to find all DomainKeys that are consumers of
//! reduction outputs. Such consumer IterDomains may not be mapped to
//! the producer reduction domain since the corresponding reduction
//! loop must be closed before any of the consumers can appear.
class TORCH_CUDA_CU_API UnmappableReductionDomains : private IterVisitor {
 public:
  UnmappableReductionDomains();
  ~UnmappableReductionDomains() override = default;

  //! Returns true when mapping consumer domains would cause a
  //! reduction output domain to be mapped with a consumer domain of
  //! the redution. It needs to be avoided as computing consumers of
  //! reduction outputs within the corresponding reduction loop is not
  //! possible. This routine is used to build root domain mappings.
  bool isReductionOutputMapped(
      const DomainKeySet& consumer_domains,
      const ComputeAtRootDomainMap& root_map) const;

  std::string toString() const;

 private:
  using IterVisitor::handle;
  void handle(ReductionOp* op) override;
  void handle(GroupedReductionOp* op) override;
  void handle(WelfordOp* op) override;
  void handle(MmaOp* op) override;

  void handleReductionOutput(TensorView* out_tv);

 private:
  //! Map from Reduction output DomainKeys to consumer DomainKeys
  DomainKeyMap<DomainKeySet> reduction_domains_;
  //! Map from Reduction output DomainKeys to producer DomainKeys
  DomainKeyMap<DomainKeySet> reduction_domain_inputs_;
};

//! Models root-domain mappings for computeAt
//!
//! Two iteration domains are mapped when computeAt of one iteration
//! domain is possible at another iteration domain. Consider a simple
//! example:
//!    T2 [i0,i1] = T1[i2,i3] + T0[i4,i5]
//! This will create mappings between i0, i2 and i4.
//!
//! Note that with views, there can be multiple domains mapped with
//! the same domain. Thus, obtaining one-to-one maps can
//! fail. Currently, the only use of this class is getMappableDims,
//! which just grabs any domain that is mappable, which works no
//! matter view is used or not.
class TORCH_CUDA_CU_API ComputeAtRootDomainMap : public RootDomainMap {
  friend class ComputeAtRootDomainMapBuilder;

 public:
  //! Builds a mapping table by analyzing the current
  //! fusion. Overwrite a previous table if any.
  //!
  //! \param map_through_reduction If set
  //!   true, will disable UnmappableReductionDomains check.
  //!   This is only for re-using logic in detecting
  //!   normalization fusions, which deviates slightly from
  //!   intended use of this class. Should always be true
  //!   in compute_at use cases.
  void build(bool map_through_reduction = false);

  //! Returns if key(td_a, id_a) and key(td_b, id_b) are mapped to eachother
  //! (equivalent), or are the same key.
  //!
  //! \param td_a A TensorDomain
  //! \param id_a An IterDomain in td_a
  //! \param td_b Another TensorDomain
  //! \param id_b An IterDomain in td_b
  //! \returns Boolean representing if they are mapped
  bool canMap(
      const TensorDomain* td_a,
      const IterDomain* id_a,
      const TensorDomain* td_b,
      const IterDomain* id_b) const;

  //! Make a TensorDomain an alias of another TensorDomain
  //!
  //! This is for the computeAt transformation, where TensorViews are
  //! updated with new TensorDomains. Since they keep using the same
  //! root doamins, the root mapping remains valid but needs to
  //! reflect the use of new TensorDomains as aliases of the existing
  //! ones.
  //!
  //! \param td An existing TensorDomain
  //! \param td_alias An alias of td
  void setAlias(const TensorDomain* td, const TensorDomain* td_alias);

  //! Return a map between TensorDomains
  //!
  //! Unlike the other map functions, two TensorDomains do not need to
  //! be a producer-consumer pair. Since they may not be a
  //! producer-consumer pair, this function requires proper root
  //! domains, which may be root or rfactor domains. Also, no error
  //! check is done as we do not assume producer-consumer
  //! relationship.
  //!
  //! Note that an exception is thrown when a domain is found to be
  //! mapped to multiple domains, which can happen with views.
  //!
  //! \param from_td A TensorDomain from which a map is created
  //! \param from_root A root domain of from_td
  //! \param to_td A TensorDomain to which a map is created
  //! \param to_root A root domain of to_td
  std::unordered_map<IterDomain*, IterDomain*> mapBestEffort(
      const TensorDomain* from_td,
      const std::vector<IterDomain*>& from_root,
      const TensorDomain* to_td,
      const std::vector<IterDomain*>& to_root) const;

  // Returns an unordered set of all iter domains in producer and consumer that
  // can map to eachother
  std::unordered_set<IterDomain*> getMappableDims(
      const TensorDomain* producer,
      const TensorDomain* consumer) const;

  std::string toString() const;

 private:
  //! Returns if key_a and key(td_b, id_b) are mapped to eachother (equivalent),
  //! or are the same key.
  //!
  //! \param key_a A DomainKey
  //! \param td_b Another TensorDomain
  //! \param id_b An IterDomain in td_b
  //! \returns Boolean representing if they are mapped
  bool canMap(
      const DomainKey& key_a,
      const TensorDomain* td_b,
      const IterDomain* id_b) const;

  //! Returns if key_a and key_b are mapped to each other (equivalent), or are
  //! the same key. Returns false if two keys are not known to be mapped.
  bool canMap(const DomainKey& key_a, const DomainKey& key_b) const;

  //! Returns the set of (non-broadcast) DomainKeys that id in td is
  //! broadcasted to. Can result in more than one "concrete" DomainKey.
  std::vector<DomainKey> getConcretizedKeys(
      const TensorDomain* td,
      const IterDomain* id) const;

  //! Returns the set of (non-broadcast) iter domains that id in td is
  //! broadcasted to. Can result in more than one "concrete" iter domain.
  std::unordered_set<const IterDomain*>& getConcretizedDomains(
      const TensorDomain* td,
      const IterDomain* id);

  //! Return a map between root IterDomains of a producer-consumer
  //! pair.
  //!
  //! \param producer A producer TensorDomain
  //! \param consumer A consumer TensorDomain
  //! \param root_dims_to_map Maps only from IterDomains in this set
  //! \param producer_to_consumer Maps from producer to consumer if true
  std::unordered_map<IterDomain*, IterDomain*> map(
      const TensorDomain* producer,
      const TensorDomain* consumer,
      const std::unordered_set<IterDomain*>& root_dims_to_map,
      bool producer_to_consumer) const override;

 private:
  //! Disjoint set of all mapped <TD, ID> keys to determine axes equivalency
  DisjointSets<DomainKey, DomainKeyHash> eq_set_;

  //! All IterDomains in the mapping that are a broadcast ID
  DomainKeyMap<std::unordered_set<const IterDomain*>> bcast_map_;

  //! Broadcast iter domain that does not match dimensions in its produer,
  //! meaning it is a brand new domain in its TensorDomain.
  DomainKeySet new_broadcast_domains_;

  //! Keep track of window axes so that the map function can ignore them.
  std::unordered_set<IterDomain*> window_axes_;
};

//! Create a DisjointSets of root IterDomains by traversing the
//! current fusion entirely. IterDomains that can be mapped each
//! other with computeAt are grouped into the same subset in the
//! DisjointSets.
class TORCH_CUDA_CU_API ComputeAtRootDomainMapBuilder
    : private BackwardVisitor {
 public:
  explicit ComputeAtRootDomainMapBuilder(
      ComputeAtRootDomainMap& root_map,
      bool map_through_reduction = false);

 private:
  //! Initialize the bcast map for fusion outputs
  void initializeBcastMap(const TensorView* tv, const IterDomain* id);

  //! Set a pair of producer-consumer domain keys as mappable
  void setMapped(const DomainKey& producer, const DomainKey& consumer);

  //! Records two domains are invalid to map
  void setInvalid(const DomainKey& key1, const DomainKey& key2);

  //! Check if no pair of domains is invalid to map
  bool isInvalid(const DomainKeySet& domains) const;

  //! Track a pair of producer-consumer domains as potentially mappable. Inserts
  //! entries into pending_map_, but does not add anything into the root_map_
  //! (added when handle is called on a TensorView). Maybe mapped will, however,
  //! immediately propagate broadcast iter domains.
  void setMaybeMapped(
      const TensorDomain* producer_td,
      const IterDomain* producer_id,
      const TensorDomain* consumer_td,
      const IterDomain* consumer_id);

  void addToPendingList(const DomainKey& producer, const DomainKey& consumer);

  //! Map pointwise IterDomains from inputs of expressions to outputs.
  //! Do not map reduction IterDomains in inputs.
  void mapPointwiseOrReductionOp(Expr* e);

  using BackwardVisitor::handle;

  void handle(Expr* e) override;

  void handle(UnaryOp* uop) override {
    mapPointwiseOrReductionOp(uop);
  }

  void handle(BinaryOp* bop) override {
    mapPointwiseOrReductionOp(bop);
  }

  void handle(TernaryOp* top) override {
    mapPointwiseOrReductionOp(top);
  }

  void handle(RNGOp* top) override;

  void handle(ReductionOp* op) override {
    mapPointwiseOrReductionOp(op);
  }

  void handle(GroupedReductionOp* op) override {
    mapPointwiseOrReductionOp(op);
  }

  void handle(WelfordOp* wop) override {
    mapPointwiseOrReductionOp(wop);
  }

  void handle(LoadStoreOp* ldst) override {
    mapPointwiseOrReductionOp(ldst);
  }

  void handle(MmaOp* wop) override {
    mapPointwiseOrReductionOp(wop);
  }

  void handle(ShiftOp* op) override {
    mapPointwiseOrReductionOp(op);
  }

  void handle(ViewOp* op) override {
    mapPointwiseOrReductionOp(op);
  }

  void handle(ViewAsScalar* op) override;

  void handle(BroadcastOp* op) override;

  void handle(TransposeOp* op) override;

  void handle(ExpandOp* op) override {
    mapPointwiseOrReductionOp(op);
  }

  void handle(GatherOp* op) override;

  void handle(TensorView* tv) override;

  //! Maps all pending mappings.
  //! This is called for each of TensorViews in a backward traversal,
  //! recursively building mappings from the output tensors to the
  //! input tensors.
  void mapAllPendingMappings(const DomainKey& key);

  //! Maps all pending mappings for id of td. When id is a broadcast,
  //! mapping is done separately for each concrete domain.
  void mapAllPendingMappings(const TensorDomain* td, IterDomain* id);

  bool safeToMap(const DomainKeySet& domains);

 private:
  ComputeAtRootDomainMap& root_map_;
  //! Keep track of what we want to try and map
  DomainKeyMap<DomainKeySet> pending_map_;
  std::unordered_set<Expr*> visited_;
  //! Helper class to find invalid mappings due to reductions
  UnmappableReductionDomains incompatible_domains_;
  //! Running vector of domain pairs that are invalid to map
  std::vector<std::pair<DomainKey, DomainKey>> invalid_mappings_;

  //! Disable UnmappableReductions check, should
  //!  always be false for compute_at use cases
  bool map_through_reduction_ = false;
};

//! Maps root domains of an entire fusion. Does not map broadcast
//! domains with non-broadcast domains.
class TORCH_CUDA_CU_API ExactRootDomainMap : public RootDomainMap {
 public:
  ExactRootDomainMap(Fusion* fusion);

  bool areMapped(const IterDomain* id_a, const IterDomain* id_b) const;

  std::string toString() const;

 protected:
  std::unordered_map<IterDomain*, IterDomain*> map(
      const TensorDomain* producer,
      const TensorDomain* consumer,
      const std::unordered_set<IterDomain*>& root_dims_to_map,
      bool producer_to_consumer) const override;

 private:
  DisjointSets<const IterDomain*> eq_sets_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
