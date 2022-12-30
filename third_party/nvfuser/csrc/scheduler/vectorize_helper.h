#pragma once

#include <compute_at_map.h>
#include <fusion.h>
#include <ir_all_nodes.h>
#include <lower_divisible_split.h>
#include <maxinfo_propagator.h>
// TODO: Move to cpp file.
#include <ir_builder.h>

#include <sstream>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class SchedulerRuntimeInfo;

namespace vectorize_helper {

// Basic factorization helpers used to simplify factors with only
// multiplications and divisions. i.e. can help simplify: 21/3 but not
// (15-7)/(3+2). This is all that's necessary for current support to partially
// map dimensions through our compute at DAG.
namespace factorization_helpers {

// Returns factors of i
TORCH_CUDA_CU_API std::multiset<int64_t> computeFactors(int64_t i);

TORCH_CUDA_CU_API std::multiset<int64_t> getAllFactors(
    const std::multiset<int64_t>& in_vals);

TORCH_CUDA_CU_API std::pair<std::multiset<int64_t>, std::multiset<int64_t>>
removeCommonFactors(
    const std::multiset<int64_t>& set1,
    const std::multiset<int64_t>& set2);

// Given factors made up of a product of integers over a product of integers,
// simplify the factors. Factorization algorithm here could be improved for
// performance.
TORCH_CUDA_CU_API std::pair<std::vector<int64_t>, std::vector<int64_t>>
removeCommonFactors(
    const std::vector<int64_t>& vec1,
    const std::vector<int64_t>& vec2);

TORCH_CUDA_CU_API std::pair<std::multiset<Val*>, std::multiset<Val*>>
removeSameVals(
    const std::multiset<Val*>& set1,
    const std::multiset<Val*>& set2);

// Remove all Val*s that are in both vec1 and vec2
TORCH_CUDA_CU_API std::pair<std::vector<Val*>, std::vector<Val*>> removeSameVals(
    const std::vector<Val*>& vec1,
    const std::vector<Val*>& vec2);
} // namespace factorization_helpers

// Helper class that to track partial mappings of extents. The stored extent in
// this class is product(numerator values)/product(denominator values) and has
// basic functionality like adding a multiplied factor to either the numerator
// or denominator. This class starts with a value of 0/1, but once a value is
// added it can never be zero again.
class TORCH_CUDA_CU_API ProjectedExtent {
 public:
  // Multiply numerator by provided value, or if currently zero set numerator to
  // provided value.
  void multiplyNumeratorValue(Val* new_numerator_val) {
    TORCH_INTERNAL_ASSERT(
        !new_numerator_val->isZeroInt() &&
            (!new_numerator_val->isConstInt() ||
             new_numerator_val->evaluateInt() > 0),
        "Adding numerator value of zero not supported in ProjectedExtent.");

    zero_ = false;
    if (new_numerator_val->isConstInt()) {
      const_numerator_vals_.push_back(new_numerator_val->evaluateInt());
    } else {
      symb_numerator_vals_.push_back(new_numerator_val);
    }
    valid_extents_ = false;
    is_simplified_ = false;
  }

  // Multiply denominator by provided value
  void multiplyDenominatorValue(Val* new_denominator_val) {
    TORCH_INTERNAL_ASSERT(
        !new_denominator_val->isZeroInt(), "Divide by zero detected.");
    if (new_denominator_val->isConstInt()) {
      TORCH_INTERNAL_ASSERT(
          new_denominator_val->evaluateInt() > 0,
          "Divide by zero or negative value detected, not supported: ",
          new_denominator_val->evaluateInt());
      const_denominator_vals_.push_back(new_denominator_val->evaluateInt());
    } else {
      symb_denominator_vals_.push_back(new_denominator_val);
    }
    valid_extents_ = false;
    is_simplified_ = false;
  }

  // Multiply by other but wrap each value in other with the provided predicate.
  void maybeMul(Val* pred, const ProjectedExtent& other) {
    TORCH_INTERNAL_ASSERT(
        !other.isZero(),
        "Maybe multiplying by zero ProjectedExtent not supported.");

    TORCH_INTERNAL_ASSERT(
        pred != nullptr && pred->isA<Bool>(),
        "Predicate must be a bool value for this function.");
    for (auto other_const_numer : other.const_numerator_vals_) {
      multiplyNumeratorValue(SimplifyingIrBuilder::whereExpr(
          pred,
          IrBuilder::create<Int>(other_const_numer),
          FusionGuard::getCurFusion()->oneVal()));
    }

    for (auto other_symb_numer : other.symb_numerator_vals_) {
      multiplyNumeratorValue(SimplifyingIrBuilder::whereExpr(
          pred, other_symb_numer, FusionGuard::getCurFusion()->oneVal()));
    }

    for (auto other_const_denom : other.const_denominator_vals_) {
      multiplyDenominatorValue(SimplifyingIrBuilder::whereExpr(
          pred,
          IrBuilder::create<Int>(other_const_denom),
          FusionGuard::getCurFusion()->oneVal()));
    }

    for (auto other_symb_numer : other.symb_denominator_vals_) {
      multiplyDenominatorValue(SimplifyingIrBuilder::whereExpr(
          pred, other_symb_numer, FusionGuard::getCurFusion()->oneVal()));
    }

    valid_extents_ = false;
    is_simplified_ = false;
  }

  // If necessary simplify the PartialExtent and return all the numerator values
  // multiplied together as a single Val.
  Val* getNumerator() {
    if (!valid_extents_) {
      computeNumerDenomir();
    }
    return numerator_;
  }

  // If necessary simplify the PartialExtent and return all the denominator
  // values multiplied together as a single Val.
  Val* getDenominator() {
    if (!valid_extents_) {
      computeNumerDenomir();
    }
    return denominator_;
  }

  bool isZero() const {
    return zero_;
  }

  std::string toString() const {
    std::stringstream ss;
    ss << "{";
    bool first = true;
    for (auto numer : const_numerator_vals_) {
      if (!first) {
        ss << " * ";
      }
      first = false;

      ss << numer;
    }

    for (auto numer : symb_numerator_vals_) {
      if (numer == nullptr || numer->isOneInt()) {
        continue;
      }
      if (!first) {
        ss << " * ";
      }
      first = false;
      ss << numer->toInlineString();
    }

    ss << "} / {";
    first = true;
    for (auto denom : const_denominator_vals_) {
      if (!first) {
        ss << " * ";
      }
      first = false;

      ss << denom;
    }

    for (auto denom : symb_denominator_vals_) {
      if (denom == nullptr || denom->isOneInt()) {
        continue;
      }
      if (!first) {
        ss << " * ";
      }
      first = false;

      ss << denom->toInlineString();
    }
    ss << "}";
    return ss.str();
  }

 private:
  // Simplify the partial extent as much as possibly by removing common factors
  // from the numerator and denominator.
  void simplify() {
    if (is_simplified_) {
      return;
    }

    std::tie(const_numerator_vals_, const_denominator_vals_) =
        factorization_helpers::removeCommonFactors(
            const_numerator_vals_, const_denominator_vals_);

    std::tie(symb_numerator_vals_, symb_denominator_vals_) =
        factorization_helpers::removeSameVals(
            symb_numerator_vals_, symb_denominator_vals_);

    valid_extents_ = false;
    is_simplified_ = true;
  }

  // Simplify the partial extent and generate the Val* for numerator and
  // denominator by multiplying all the factors.
  void computeNumerDenomir() {
    if (valid_extents_) {
      return;
    }
    simplify();

    numerator_ = nullptr;

    auto const_numerator_factor = 1;
    for (auto factor : const_numerator_vals_) {
      const_numerator_factor *= factor;
    }

    if (const_numerator_factor > 1) {
      numerator_ = IrBuilder::create<Int>(const_numerator_factor);
    }

    for (auto numerator_val : symb_numerator_vals_) {
      if (numerator_ == nullptr) {
        numerator_ = numerator_val;
      } else {
        numerator_ = SimplifyingIrBuilder::mulExpr(numerator_, numerator_val);
      }
    }

    if (numerator_ == nullptr) {
      numerator_ = isZero() ? FusionGuard::getCurFusion()->zeroVal()
                            : FusionGuard::getCurFusion()->oneVal();
    }

    denominator_ = nullptr;

    auto const_denominator_factor = 1;
    for (auto factor : const_denominator_vals_) {
      const_denominator_factor *= factor;
    }

    if (const_denominator_factor > 1) {
      denominator_ = IrBuilder::create<Int>(const_denominator_factor);
    }

    for (auto denominator_val : symb_denominator_vals_) {
      if (denominator_ == nullptr) {
        denominator_ = denominator_val;
      } else {
        denominator_ =
            SimplifyingIrBuilder::mulExpr(denominator_, denominator_val);
      }
    }

    if (denominator_ == nullptr) {
      denominator_ = FusionGuard::getCurFusion()->oneVal();
    }
    valid_extents_ = true;
  }

  // True if numerator and denominator have been computed and don't need to be
  // updated
  bool valid_extents_ = false;
  // True if no numerator or denominator was added since the last simplify() was
  // called
  bool is_simplified_ = false;
  // Fraction starts at zero, but if a value is ever added in the numerator it
  // can never be zero again, it must be at least 1.
  bool zero_ = true;

  // numerator and denominator values (product of the sets below)
  Val* numerator_ = nullptr;
  Val* denominator_ = nullptr;

  // Const and symbolic numerator and denominator values.
  std::vector<int64_t> const_numerator_vals_;
  std::vector<Val*> symb_numerator_vals_;
  std::vector<int64_t> const_denominator_vals_;
  std::vector<Val*> symb_denominator_vals_;
};

// Projects IterDomains through the fusion starting at provided reference. IDs
// in the reference are expected to be "contiguous", simply means dimensions
// that the iter domains are consecutive and next to eachother in the
// reference. This property is not enforced, but mapping can have some
// unpredictbale results if they are not. The reason we want contiguity here
// is this class is primarily used for vectorization analysis. Domains may be
// inserted or removed while propogating through the fusion and this class has
// to be senstitive to that.
//
// For example:
// Input: T0[i0, i2]
// Reference: T5[i0, i1, i2]
// If we want to base the vectorization size on the reference being contiguous
// in a 1D scheduler, we'd start the proces on the reference with {i0, i1,
// i2}. When we propogate to the input what we would still like is: {i0, i1,
// i2} to signify to us that the root domains in the input that map to the
// reference are not contiguous. So when we think of vector word, if we want
// the input to be included in the vectorized dimensions, we can only check
// multiples based on i2, not i0*i1*i2 like the reference would indicate.
//
// Another example:
// Input:[i1, i0, i2]
// Refrence [i0, i1, i2]
// Similarly as above when we propogate from the reference to the Input we'd
// like {i0, i1, i2}, which is the order of the reference, not the input. This
// is because we can compare that with the input domains to understand it's
// not ordered consistently, so once again we can only take into consideration
// vectorization based on i2.
//
// Another example:
// Input:[i0, i1, i2]
// Intermediate: [i1, i0, i2]
// Refrence [i0, i1, i2]
// Keeping the ordering relative to the reference also allows us to look
// though transpose operations without missing out in a case like this that
// the reference and input are consistently ordered so we can look at i0*i1*i2
// for our vector multiple even though there are transposes in between them.
//
// The tricky part of this class is what happens through combinations of view
// and transpose. IterDomains are projected for example:
//   tv0[2*3, 5*7, 11]
//   tv1[2*3, 5, 7*11] = view(tv0)
// With tv1 and 7*11 as the reference and ids. When we project from tv1 to
// tv0, we'd map the inner most 11, but we also want to map the 5*7 with an
// extent of 7. This can get tricky though as:
//   tv0[2, 3*5*7, 11]
//   tv1[2*3, 5, 7*11] = view(tv0)
// with tv1 and [2*3, 7*11] as the reference and ids. tv0's 2 and 11 dim are
// easily identified as being mapped. The 3*5*7 dimension however, is
// partially mapped on the left and right side. Since this class is  intended to
// line up "inner dimensions" of tensors through out the graph for the purpose
// of unrolling and vectorization, it only tracks partial dimensions as they are
// on the right hand side of iteration domains. For example in the last case we
// would only identify tv0's 3*5*7 dimension as being a mapping with extent 7.
// If we further had:
//   tv0[5*7*11]
//   tv1[5*7, 11] = view(tv0)
//   tv2[5, 7*11] = view(tv1)
// with tv2 and [7*11] as the reference and ids (this could be a valid example
// from the pointwise scheduler).
// (1) tv1 would:
//     map on 5*7 with extent 7
//     map on 11 with extent 11.
// (1) tv0 would:
//     map on 5*7*11 with size 7*11
//
// Finally if we have:
// tv0[3, 5, 7]
// tv1[7, 5, 3] = view(tv0)
// tv2[3, 5, 7] = view(tv1)
// with tv2 mapping on 5, 7
// We use fractional, symbolic, and conditional mappings so tv1:
//   maps on 3 with extent 3
//   maps on 5 with extent 5
//   maps on 7 with extent (5*7)/(5*3)
// Then tv0:
//   maps on 7 with extent 7
//   maps on 5 with extent 5
//
// This class is responsible for both computing the spanning tree and running
// the spanning tree.
//
// In other words this class implements:
//   MaxInfoSpanningTree::computeInfoC2P
//     and
//   MaxInfoSpanningTree::Propagator::propagateC2P
//
// The challenge here is the information we need for
// MaxInfoSpanningTree::computeInfoC2P is the same information we need to
// compute for MaxInfoSpanningTree::Propagator::propagateC2P
//
// We could compute both of these passes at the same time, only saving the
// result produced from processing the edge that's chosen from
// MaxInfoSpanningTree::computeInfoC2P while processing based on
// MaxInfoSpanningTree::Propagator::propagateC2P. However, this would require
// refactoring of MaxInfoSpanningTree so for right now this class just uses
// two passes.
//
// MaxInfoSpanningTree::computeInfoC2P runs first with recording_=false and
// will effectively compute the values of projected_root_ids_ and
// projected_rfactor_ids_. However it will compute these by running all edges
// between expressions. Therefore,
// MaxInfoSpanningTree::Propagator::propagateC2P later simply calls
// MaxInfoSpanningTree::computeInfoC2P with recording_=true where it will
// actually record the computed information since it will be then projected
// through the DAG maximizing saving information.
class TORCH_CUDA_CU_API ContiguousInnerDimensionsMapper
    : public MaxInfoSpanningTree,
      MaxInfoSpanningTree::Propagator {
 public:
  ContiguousInnerDimensionsMapper() = delete;

  static ContiguousInnerDimensionsMapper map(
      TensorView* reference,
      const std::vector<IterDomain*>& ids,
      std::shared_ptr<const ComputeAtMap> ca_map,
      const std::unordered_set<Split*>& divisible_splits);

  static ContiguousInnerDimensionsMapper map(
      TensorView* reference,
      const std::vector<IterDomain*>& ids) {
    auto ca_map = std::make_shared<ComputeAtMap>(reference->fusion());
    auto divisible_splits =
        getAllDivisibleSplits(reference->fusion(), ca_map.get());
    return ContiguousInnerDimensionsMapper::map(
        reference, ids, ca_map, divisible_splits);
  }

  bool hasMappedDims(TensorView* tv) const {
    return tv_infos_.find(tv) != tv_infos_.end();
  }

  const std::vector<IterDomain*>& mappedRootIds(TensorView* tv) const {
    TORCH_INTERNAL_ASSERT(
        tv_infos_.find(tv) != tv_infos_.end(),
        "TensorView not found: ",
        tv->toString());
    return std::dynamic_pointer_cast<const MappedDomain>(tv_infos_.at(tv))
        ->mapped_root_ids_;
  }

  const std::vector<IterDomain*>& mappedRFactorIds(TensorView* tv) const {
    TORCH_INTERNAL_ASSERT(
        tv_infos_.find(tv) != tv_infos_.end(),
        "TensorView not found: ",
        tv->toString());
    return std::dynamic_pointer_cast<const MappedDomain>(tv_infos_.at(tv))
        ->mapped_rfactor_ids_;
  }

  ProjectedExtent& getMappedExtent(IterDomain* id) {
    if (projected_extent_.find(id) != projected_extent_.end()) {
      return projected_extent_.at(id);
    }
    projected_extent_[id] = ProjectedExtent();
    return projected_extent_.at(id);
  }

 private:
  ContiguousInnerDimensionsMapper(
      TensorView* reference,
      const std::vector<IterDomain*>& reference_ids,
      std::shared_ptr<const ComputeAtMap> ca_map,
      const std::unordered_set<Split*>& divisible_splits);

  class MappedDomain : public Information {
   public:
    MappedDomain() = default;

    static std::shared_ptr<MappedDomain> build(
        std::vector<IterDomain*> root_ids,
        std::vector<IterDomain*> rfactor_ids,
        bool is_c2p) {
      auto ptr = std::make_shared<MappedDomain>();
      ptr->mapped_root_ids_ = root_ids;
      ptr->mapped_rfactor_ids_ = rfactor_ids;
      ptr->is_c2p_ = is_c2p;
      return ptr;
    }

    operator bool() const final {
      return !mapped_root_ids_.empty() || !mapped_rfactor_ids_.empty();
    }

    bool operator<(const Information& other_info) const final {
      auto other_mapped_domain = dynamic_cast<const MappedDomain&>(other_info);

      if (is_c2p_) {
        return mapped_rfactor_ids_.size() <
            other_mapped_domain.mapped_rfactor_ids_.size();
      }
      return mapped_root_ids_.size() <
          other_mapped_domain.mapped_root_ids_.size();
    }

    std::vector<IterDomain*> mapped_root_ids_;
    std::vector<IterDomain*> mapped_rfactor_ids_;
    // Information is not symmetric between c2p and p2c, track which direction
    // the computation is in for the < operator
    bool is_c2p_ = true;
  };

  void addProjectedExtent(IterDomain* id, ProjectedExtent pe) {
    if (!recording_) {
      return;
    }
    projected_extent_[id] = pe;
  }

  // MaxInfoSpanningTree functions
  std::shared_ptr<Information> computeInfoC2P(
      TensorView* from,
      TensorView* to,
      std::shared_ptr<Information> from_info) final;

  std::shared_ptr<Information> computeInfoP2C(
      TensorView* from,
      TensorView* to,
      std::shared_ptr<Information> from_info) final;

  std::shared_ptr<Information> computeInfoSibling(
      TensorView* from,
      TensorView* to,
      std::shared_ptr<Information> from_info) final;

  // Projection from root<->rfactor domains
  std::vector<IterDomain*> projectIdToRoot(
      TensorView* ref,
      std::vector<IterDomain*> from_info);

  std::vector<IterDomain*> projectIdToRFactor(
      TensorView* ref,
      std::vector<IterDomain*> from_info);

  // Propagator functions
  void propagateC2P(TensorView* from, TensorView* to) final;
  void propagateP2C(TensorView* from, TensorView* to) final;
  void propagateSibling(TensorView* from, TensorView* to) final;

  // Extent propagation through single transformation operations
  void propagateExtentSplitBackward(Split* split, bool outer_maps);
  void propagateExtentMergeBackward(const Merge* merge);
  void propagateExtentMergeForward(const Merge* merge, bool outer_maps);
  void propagateExtentSplitForward(Split* split);

  // Initialized to false, series of compute... calls will be performed to find
  // the spanning tree. Then propagate... calls will call the compute... calls.
  // recording_ starts as false, and stays that way during the first series of
  // compute... calls. As soon as the first propagate... calls are called,
  // recording_ will perpetually stay on.
  bool recording_ = false;

  std::shared_ptr<const ComputeAtMap> ca_map_;
  std::unordered_set<Split*> divisible_splits_;

  // Mapped root dimensions for each TensorView as we propogate. These
  // mappings are in the order of the reference.

  std::unordered_map<
      TensorView*,
      std::shared_ptr<MaxInfoSpanningTree::Information>>
      tv_infos_;

  std::unordered_map<IterDomain*, ProjectedExtent> projected_extent_;
};

// Returns Mappings of all dims in reference starting from inner most position
// to outer most position.
//
// A tensor like T0[i0, r1, b2] will return 3 Mapper instances associated
// with:
// {{i0, r1, b1}, {r1, b1}, {b1}}
std::vector<ContiguousInnerDimensionsMapper> getAllVectorizedMapsOf(
    TensorView* ref);

// Returns PartialExtent entires associated with each dimension that should be
// evaluated based on contiguity of reference and dimensions mapped to ref in
// mapper.
std::vector<std::pair<ProjectedExtent&, IterDomain*>> getContigVectorSizesOf(
    TensorView* of_tv,
    ContiguousInnerDimensionsMapper& mapper);

// TODO: vectorizable_inputs_outputs should actually be known based on the
// computed mappings. If nothing is mapped for a tensorview it's not
// vectorizable.
size_t getExpandedVectorization(
    const std::vector<ContiguousInnerDimensionsMapper>& reference_maps,
    SchedulerRuntimeInfo& runtime_info,
    const std::vector<TensorView*> vectorizable_inputs_outputs,
    TensorView* reference_tv,
    int break_point,
    size_t default_word_size);

} // namespace vectorize_helper
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
