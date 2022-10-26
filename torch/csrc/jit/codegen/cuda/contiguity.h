#pragma once

#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/compute_at_map.h>
#include <torch/csrc/jit/codegen/cuda/disjoint_set.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/lower_shift.h>
#include <torch/csrc/jit/codegen/cuda/lower_trivial_broadcast.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// Goes through the transformations associated with a series of ids and root
// ids. Checks the ordering of the iteration domains through these operations to
// pick out which operations are consistently ordered. For example:
// [i0, i1, i2]
// ->split(0, 4)->merge(1)->merge(1)->merge(0)
// are consistently ordered from largest to smallest extents, but
// ->split(0, 4)->merge(1)->merge(0, 2)->merge(0) is not consistently ordered
// with the roots.
//
// This property is important to understand the contiguity of dimensions through
// complex transformations.
class OrderedIdInformation : public OptInDispatch {
 public:
  OrderedIdInformation() = delete;

  OrderedIdInformation(
      const std::vector<IterDomain*>& ids,
      const std::vector<IterDomain*>& root_domain,
      std::shared_ptr<const ConcretizedBroadcastDomains> concrete_info);

  const std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>&
  idToRootIds() const {
    return id_to_root_ids_;
  }

  bool isConsistentlyOrdered(IterDomain* id) const {
    return consistently_ordered_ids_.find(id) !=
        consistently_ordered_ids_.end();
  }

  bool exclusivelyConsumesRoots(IterDomain* id) const {
    return exclusively_consumes_roots_.find(id) !=
        exclusively_consumes_roots_.end();
  }

 private:
  // Returns if the id in active_ids should be in exclusively_consumes_roots_
  bool checkExclusivelyConsumesRoots(IterDomain* id);

  void handle(Split*) override;

  void handle(Merge* merge) override;

  void handle(Swizzle2D* swizzle) override;

  // Track which root ids were used to generate each iter domain
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
      id_to_root_ids_;

  // Track all IterDomains that have correct ordered transforms for contiguity.
  // i.e. if we have:
  //
  // root = [i0, i1, i2]
  // i3 = merge(i0, i2)
  // would not be consistently ordered transformed
  //
  // root = [i0, i1, i2]
  // i4, i5 = spit(merge(merge(i0, i1), i2), 4)
  // would be consistently ordered transforms
  //
  // root = [i0, i1, i2, i3]
  // i4 = merge(i1, i2) would also be consistently ordered transformed
  std::unordered_set<IterDomain*> consistently_ordered_ids_;

  // Active series of IterDomains that are updated while we're processing the
  // domain. Helps us identify which ids are consistently_ordered_ids_. Used
  // for intermediate storage, not to return.
  std::vector<IterDomain*> active_ids_;

  // IterDomains in this set exclusively consume all the uses of their roots.
  // For example:
  // [i0, i1] split(0, f)->merge(1)
  // [ceilDiv(i0, f), f*i1]
  // neither iter domains exclusively consume the roots. With another:
  // merge(0) -> [ceilDiv(i0, f)*f*i1]
  // The resulting iter domain does exclusively consume the roots.
  //
  // Also:
  // [i0, i1, i2, i3] merge(1)->merge(1)
  // ->[i0, i1*i2*i3]
  // both resulting iter domains do exclusively consume their roots
  std::unordered_set<IterDomain*> exclusively_consumes_roots_;

  // Broadcast domains that are concretized cannot be considered contiguously
  // indexable.
  // TODO: This constraint is more conservative than necessary as it's only if
  // the domain is concretized within the local indexing, not in the entire
  // fusion.
  std::shared_ptr<const ConcretizedBroadcastDomains> concrete_info_;
};

// Based on provided divisible split set, goes through expressions and marks all
// IterDomains that are dependent on a non-divisible split.
class NonDivisibleSplitDependencies : public OptInDispatch {
 public:
  NonDivisibleSplitDependencies() = delete;

  NonDivisibleSplitDependencies(
      const std::vector<IterDomain*>& ids,
      const std::vector<IterDomain*>& root_domain,
      const std::unordered_set<Split*>& divisible_splits);

  bool dependsOnNonDivisibleSplit(IterDomain* id) const {
    return depends_on_non_divisible_split.find(id) !=
        depends_on_non_divisible_split.end();
  }

 private:
  std::unordered_set<IterDomain*> depends_on_non_divisible_split;
};

// A merge is contiguous if:
//   Inputs of outer are to the left in the root domain of the inputs of RHS.
//   All inputs are contiguous in the root domain:
//     - All marked as contiguous
//     - Only gaps between inputs are broadcast or reductoin dims
//   There are no split transformations performed on outer or inner
//   All transformations on outer or inner are contiguous merges
// If this criteria holds, then we can index the input root domains of this
// merge with the indexing provided to the output of the merge in the backward
// index pass

class ContigIDs : public OptInDispatch {
 public:
  //! Check through the history of ids whose inputs map to root_domain with
  //! contiguity root_contiguity. Return unordered_set of all merges that are
  //! contiguous. Ignore root order is primarily used for predicate generation.
  //! In this case we can linearize indexing of any ID that only consists of
  //! merge operations.
  //!
  //! Mapping information from CA Index concrete to reference domains
  //! is used to find if merged output domains can be indexed. If there's
  //! no mapping to a reference domain, there's no corresponding
  //! index, so it isn't marked as conting merge.
  //!
  //! p2c_id_map can be used when replayed producer domains are
  //! analyzed, in which case producer-to-consumer maps should be
  //! passed.
  //!
  //! If ignore_indexability and ignore_halo_constraint are true,
  //! ignore the constraint on indexing and halo, respectively. It is
  //! the caller that is responsible for its correctness.
  //! Not really sure why but clang-tidy only complains about
  //! std::unordered_map if passed as a const reference.
  ContigIDs(
      const std::vector<IterDomain*>& ids,
      const std::vector<IterDomain*>& root_domain,
      const std::vector<bool>& root_contiguity,
      const std::unordered_set<IterDomain*>& final_ids,
      const std::unordered_map<IterDomain*, Val*>& index_map,
      const std::unordered_set<Split*>& divisible_splits,
      std::unordered_map<IterDomain*, IterDomain*> p2c_id_map = {},
      bool ignore_indexability = false,
      bool ignore_consistent_ordering = false);

  //! \param ids IterDomains on the leaves of the domain we're looking for
  //! contiguous indexing into.
  //! \param root_domain the root domain of the domain we're looking for
  //! contiguous indexing into.
  //! \param root_contiguity the contiguity of the root_domain.
  //! \param concrete_to_ref concrete ids of the exact map that the reference
  //! index is using for indexing.
  //! \param divisible_splits a set of all splits in the fusion that are
  //! divisible.
  //! \param ca_map compute at map of the fusion.
  //! \param halo_info halo information of the fusion.
  //! \param concrete_info concretized broadcast information of the fusion.
  //! \param p2c_id_map map from producer to consumer ids used for indexing
  //! producer tensors.
  //! \param ignore_consistent_ordering true for actual indexing into tensors
  //! but false for predicate analysis. Ordering of merges don't matter for
  //! predicate generation as they don't map to a physical address.
  //! \param ignore_indexability can only be true if providing a real
  //! concrete_to_ref map. As what it's checking is if the index is actually
  //! indexable based on the reference.
  ContigIDs(
      const std::vector<IterDomain*>& ids,
      const std::vector<IterDomain*>& root_domain,
      const std::vector<bool>& root_contiguity,
      const std::unordered_set<IterDomain*>& final_ids,
      const std::unordered_map<IterDomain*, Val*>& index_map,
      const std::unordered_set<Split*>& divisible_splits,
      std::shared_ptr<const ComputeAtMap> ca_map,
      std::shared_ptr<const HaloInfo> halo_info,
      std::shared_ptr<const ConcretizedBroadcastDomains> concrete_info,
      std::unordered_map<IterDomain*, IterDomain*> p2c_id_map = {},
      bool ignore_indexability = false,
      bool ignore_consistent_ordering = false);

  //! Return an empty ContigIDs with no contiguous ID
  static ContigIDs getNonContigIDs();

  const std::unordered_set<IterDomain*>& contigIDs() const {
    return contig_ids_;
  }

  const std::unordered_map<IterDomain*, std::unordered_set<IterDomain*>>&
  withinContigIDs() const {
    return within_contig_ids_;
  }

  const std::unordered_map<IterDomain*, IterDomain*>& rootToIndexedID() const {
    return root_to_indexed_id_;
  }

  VectorOfUniqueEntries<IterDomain*> indexedRootIDs(IterDomain* id) const {
    auto root_ids_it = consistent_transform_info_->idToRootIds().find(id);
    if (root_ids_it == consistent_transform_info_->idToRootIds().end()) {
      return {};
    }
    return root_ids_it->second;
  }

 private:
  using OptInDispatch::handle;

  bool inRoot(const std::vector<IterDomain*>& ids) {
    return std::all_of(ids.begin(), ids.end(), [this](IterDomain* id) {
      return is_contig_root_.find(id) != is_contig_root_.end();
    });
  }

  bool isContig(IterDomain* id) {
    return contig_ids_.find(id) != contig_ids_.end();
  }

  // Split outputs are not contiguous, don't need to do anything.
  void handle(Split*) override {}

  void handle(Merge* merge) override;

  // TODO:
  //  Currently not propagating any contiguity information
  // as contiguity is generally not preserved after swizzles.
  // But in follow ups we could gradually add back a few special
  // cases, depending on specific swizzle type and axes.
  void handle(Swizzle2D* swizzle) override {}

  IterDomain* getCAIndexConcreteId(IterDomain* id) const;

  //! True if an ID is indexable.
  //! E.g., a merged domain with broadcast may not be indexable when
  //! its corresponding reference tensor has non-broadcast domains.
  bool isIndexable(IterDomain* id) const;

  //! Return an ID mapped with id_map_ or itself
  IterDomain* getMappedId(IterDomain* id) const;

 private:
  void build(const std::vector<IterDomain*>& ids);

  //! Root domains to analyze contiguity
  const std::vector<IterDomain*>& root_domain_;
  //! Contiguity of root_domain_
  const std::vector<bool>& root_contiguity_;
  //! Domains where indexing/predicates cannot be done with their
  //! consumers domains
  const std::unordered_set<IterDomain*>& final_ids_;
  //! Mapping of concrete domains to indices. Just used to check if
  //! there's an index for an IterDomain.
  const std::unordered_map<IterDomain*, Val*> index_map_;
  // Divisible split information as we can still consider iter domains
  // contiguous through divisible splits.
  const std::unordered_set<Split*>& divisible_splits_;

  std::shared_ptr<const ComputeAtMap> ca_map_;
  std::shared_ptr<const HaloInfo> halo_info_;
  std::shared_ptr<const ConcretizedBroadcastDomains> concrete_info_;

  //! Producer-to-consumer index map in the case of analyzing replayed
  //! producer tensors
  const std::unordered_map<IterDomain*, IterDomain*> p2c_id_map_;

  const bool ignore_indexability_ = false;
  const bool ignore_consistent_ordering_ = false;

  //! Mapping of root domain to bool indicating contiguity
  std::unordered_map<IterDomain*, bool> is_contig_root_;
  // Mark if ids are result of contigous merges
  std::unordered_set<IterDomain*> contig_ids_;
  // Given contiguous domain, return all iter domains within its history.
  std::unordered_map<IterDomain*, std::unordered_set<IterDomain*>>
      within_contig_ids_;
  //! Mapping of root domain to the actual indexed domain, which can
  //! be itself or a contig merged domain if found.
  std::unordered_map<IterDomain*, IterDomain*> root_to_indexed_id_;

  std::unique_ptr<const OrderedIdInformation> consistent_transform_info_;

  NonDivisibleSplitDependencies non_divisible_id_info_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
