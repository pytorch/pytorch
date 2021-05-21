#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Auxiliary class to represent information about halo of an axis
class AxisHaloInfo {
 public:
  //! Width of halo.
  //!
  //! pos is either 0 or 1. The width of halo at offset zero is set
  //! when pos is 0.
  unsigned int width(int pos) const {
    TORCH_INTERNAL_ASSERT(pos >= 0 && pos < 2);
    return widths_[pos];
  }

  //! Sum of the widths of both widths
  unsigned int width() const {
    return width(0) + width(1);
  }

  const auto& widths() const {
    return widths_;
  }

  //! Set the halo width of either side.
  //! pos is either 0 or 1. The width of halo at offset zero is set
  //! when pos is 0.
  void setWidth(int pos, unsigned int width) {
    TORCH_INTERNAL_ASSERT(pos >= 0 && pos < 2);
    widths_[pos] = width;
  }

  //! Extend the halo width to account for another axis.
  void merge(int pos, unsigned int other) {
    setWidth(pos, std::max(width(pos), other));
  }

  //! Extend the halo width to account for another axis.
  void merge(const AxisHaloInfo& other) {
    for (size_t i = 0; i < widths_.size(); ++i) {
      merge(i, other.width(i));
    }
  }

  //! True when halo is attached
  bool hasHalo() const {
    return std::any_of(
        widths_.begin(), widths_.end(), [](auto w) { return w != 0; });
  }

  std::string toString() const {
    std::stringstream ss;
    ss << "<" << width(0) << ", " << width(1) << ">";
    return ss.str();
  }

 private:
  //! Sizes of the halo regions of two sides. Both values are zero for
  //! axes with no halo. When an axis has halo at offset zero,
  //! widths_[0] is non-zero and designates the size of the
  //! halo. Similarly, non-zero widths_[1] means the axis has halo at
  //! the other end of the axis.
  std::array<unsigned int, 2> widths_;
};

//! Helper class for lowering tensors with halo. Only valid at the
//! lowering time.
class HaloInfo {
 public:
  //! Scan a fusion and collect all information for lowering
  void build(Fusion* fusion);

  //! Build mappings of extent information of a TensorDomain
  void build(TensorDomain* td);

  //! Set initial AxisHaloInfo of a root axis
  //!
  //! This is only for root or rfactor axes. It is an error to query
  //! with other axes.
  void setRootAxisInfo(IterDomain* id, const AxisHaloInfo& root_axis_info);

  //! Returns the registed AxisHaloInfo of a root axis.
  //!
  //! This is only for root axes. It is an error to query with
  //! non-root axes.
  const AxisHaloInfo& getRootAxisInfo(IterDomain* id) const;
  AxisHaloInfo& getRootAxisInfo(IterDomain* id);
  //! KIR version
  const AxisHaloInfo& getRootAxisInfo(kir::IterDomain* id) const;
  AxisHaloInfo& getRootAxisInfo(kir::IterDomain* id);

  //! Query if an axis has a halo width.
  //!
  //! See the comment at halo_width_map_.
  bool hasHaloWidth(IterDomain* id) const;

  //! Return the halo width of an axis.
  //!
  //! It's an error if queried for an axis with no halo width
  //! information.
  unsigned getHaloWidth(IterDomain* id) const;

  //! Returns an extent if id is extended for halo. Nullptr is
  //! returned otherwise.
  Val* getExtent(IterDomain* id) const;
  kir::Val* getExtent(kir::IterDomain* id) const;

  // True when the extent of id1 is guaranteed to be lesser than or
  // equal to id2. False when it *may* not.
  bool extentLessEqual(IterDomain* id1, IterDomain* id2) const;
  // True when the extent of id1 is guaranteed to be equal to
  // id2. False when it *may* not.
  bool extentEqual(IterDomain* id1, IterDomain* id2) const;

  //! Check if expr must be predicated based on boundary conditions
  //! directly or indirectly induced by shift expressions.
  //!
  //! When yes, the expression needs two predications: one for
  //! interior and another for padding. Predicate insertion is done in
  //! the ShiftPredicateInserter class below.
  bool needsShiftPredicate(Expr* expr);
  bool needsShiftPredicate(kir::Expr* expr);

  std::string toString() const;

 private:
  //! Propagate root axis information from outputs to inputs of an
  //! expression
  void propagateRootAxisInfo(Expr* expr);

  //! Propagate root axis information from consumer to producer
  void propagateRootAxisInfo(
      TensorView* producer,
      TensorView* consumer,
      Expr* expr);

  //! Validate shift usage
  void validate(TensorView* td) const;

 private:
  //! Halo information of root axes
  std::unordered_map<IterDomain*, AxisHaloInfo> root_axis_map_;
  //! KIR version
  std::unordered_map<kir::IterDomain*, AxisHaloInfo> kir_root_axis_map_;

  //! Halo-extended extents. No mapping for axes without halo extension
  std::unordered_map<IterDomain*, Val*> extent_map_;
  //! KIR version of extent_map_ for convenience
  std::unordered_map<kir::IterDomain*, kir::Val*> kir_extent_map_;

  //! The halo width of an axis.
  //!
  //! The mapped value is a sum of two widths of both sizes of an
  //! axis. For root axes, it is equivalent to AxisHaloInfo.widths_[0]
  //! + AxisHaloInfo.widths_[1] (or AxisHaloInfo.width()). For
  //! example, when a root axis is extended by 1 for both sides, it'd
  //! be mapped to 2. For axes with no halo, they are mapped to zero.
  //!
  //! When an axis is split, its halo is only propagated to the inner
  //! output axis, so the value of this map for the inner output is
  //! the same as the input of split, while the outer output is mapped
  //! to zero.
  //!
  //! When an axis is merged, no mapping is created for its
  //! output at this point primarly because it isn't clear what the
  //! "halo width" for a merged axis should mean. Perhaps, a merged
  //! axis of (N+a)*(M+b), where N and M correspond to the original
  //! extens of two axes, and a and b correspond to their halo widths,
  //! it might make sense to set the halo width of this merged axis as
  //! (N+a)*(M+b)-N*M. Currently, however, this isn't necessary, so no
  //! particular mapping is created for merged axes.
  //!
  //! This is currently used only for conservatively comparing the
  //! overall extents of axes. See HaloInfo::extentLessEqual and
  //! HaloInfo::extentEqual.
  //!
  //! Example: Suppose a root axis has {0, 1} of
  //! AxisHaloInfo.widths_. The root axis is mapped to 1. When it is
  //! split, say, by 4, the output axes, [N / 4] and [4], where N is
  //! the extent of the root axis, the outer axis is mapped to 0,
  //! whereas the inner axis is mapped to 1. Further, suppose the
  //! inner axis is merged with another axis of extent M, we know that
  //! the extent of the resulting output axis is 5*M, but we don't
  //! create its mapping.
  std::unordered_map<IterDomain*, unsigned> halo_width_map_;
};

class ShiftPredicateInserter {
 public:
  //! Works mostly the same way as
  //! PredicateCompute::getInlinePredicate but does the insertion of
  //! the generated predicate. The branch structure is different from
  //! the usual predicated expression, so the insertion is also done
  //! here.
  static void insert(
      kir::Expr* expr,
      const std::vector<kir::ForLoop*>& loops,
      kir::Bool* thread_pred);

  //! Returns predicates for the interior and overall domains of a
  //! tensor.
  //!
  //! The isShiftPredicate flag toggles between the predicate for shifted
  //! accesses and padding.
  static kir::Bool* getPredicate(
      const kir::Expr* expr,
      const std::vector<kir::ForLoop*>& loops,
      kir::TensorView* out_tv,
      bool isShiftPredicate);
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
