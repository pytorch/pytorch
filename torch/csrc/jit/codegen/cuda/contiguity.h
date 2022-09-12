#pragma once

#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

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
  ContigIDs() = delete;

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
  //!
  //! The function interface with many parameters looks ugly, but it
  //! is also important to make ignore_indexability and
  //! ignore_halo_constraint explicit to avoid any surprise.
  //!
  //! Not really sure why but clang-tidy only complains about
  //! std::unordered_map if passed as a const reference.
  ContigIDs(
      const std::vector<IterDomain*>& ids,
      const std::vector<IterDomain*>& root_domain,
      const std::vector<bool>& root_contiguity,
      std::unordered_map<IterDomain*, IterDomain*> concrete_to_ref,
      std::unordered_map<IterDomain*, IterDomain*> p2c_id_map = {},
      bool ignore_indexability = false,
      bool ignore_halo_constraint = false);

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
  //! Root domains to analyze contiguity
  const std::vector<IterDomain*>& root_domain_;
  //! Contiguity of root_domain_
  const std::vector<bool>& root_contiguity_;
  //! Mapping of concrete to reference domains. If a concrete domain
  //! is not mapped, it is not indexable as there's no mapped index.
  const std::unordered_map<IterDomain*, IterDomain*> concrete_to_ref_;
  //! Producer-to-consumer index map in the case of analyzing replayed
  //! producer tensors
  const std::unordered_map<IterDomain*, IterDomain*> p2c_id_map_;
  const bool ignore_indexability_ = false;

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
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
