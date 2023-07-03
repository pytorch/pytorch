#include <transform_view.h>

#include <arith.h>
#include <fusion.h>
#include <instrumentation.h>
#include <ir_builder.h>
#include <ir_internal_nodes.h>
#include <ir_iostream.h>
#include <iter_visitor.h>
#include <transform_iter.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! There's three domains associated with performing a view operation:
//! 1) Original Domain:
//!   This view is the original input to the view operation. It has no
//!   transforms on it, it is however passed in without its reduction domains
//!   (as is expected since we're trying to generate the output of the
//!   operations).
//!
//! Trivially reduced domain:
//!   Predicting which operations are trivial reduced are not trivial. If a
//!   broadcast is between two iter domains in the original domain that must be
//!   merged for the view transform:
//!     - If the broadcast domain lines up with a broadcast domain in the final
//!       tensor domain keep it.
//!     - If the domain is size-1 but not marked as a broadcast domain (runtime
//!       size==1)
//!       Note: This isn't something we generally support consistently
//!     - If the broadcast domain is marked as a compile time broadcast domain,
//!       and doesn't line up with a broadcast domain in the final result.
//!       Trivially reduce it.
//!   The index for these transformations is marked as the index of the original
//!   domain, as that's the input for the trivial reduction. This produces the
//!   trivially reduced domain.
//!
//! Post-view Domain:
//!   This domain is the original domain after the trivial reductions and all
//!   transformations. This domain holds the rfactor domains determined by
//!   merge/split operations of the find transformations pass. It is the final
//!   domain without all the broadcast operations (can have some that were
//!   preserved through the transformations).
//!       For example: {1, 2, 1, 4} -> {1, 2, 1, 2, 2} doesn't have any
//!         conflicts of the view transformation and the broadcast dimensions,
//!         so they won't be trivial reduced, they will simply be propagated
//!         through the view.
//!         {1, 2, 1, 4} -> {1, 8, 1} does have the second 1 dimension in
//!         between the 2 and 8 that have to be merged. The first broadcast axis
//!         will be propagated through the domains unafected, yet the second
//!         braodcast axis will be trivially reduced, then rebroadcasted.
//!  The transformation index marked for the splits/merges to produce this
//!  domain are done based on an "in progress" tensor view (called transform
//!  view index in the find transformation pass). This allows us to simply apply
//!  these transformations serially to produce this domain.
//!
//! Post-broadcast Domain:
//!    This domain finally matches the output of the view operation fully and
//!    can be used in further computations.
//!
//! View process at compute time:
//!   1) View takes in the input TensorView x, original runtime
//!      std::vector<int64_t>, and viewed runtime std::vector<int64_t>.
//!   2) AnalyzeView is called Which will figure out what series of
//!      transformations is required from the input tensor to the output tensor.
//!      These transformations are recorded.
//!   3) Sum operation is called on the trivial reduction axes from the
//!      analysis.
//!   4) applyViewTransforms will generate the output domain of the view
//!      operation.
//!        Calls TensorDomain::view(view_analysis) which returns the rfactored
//!        domain.
//!        Gets forwarded to transformView(TensorDomain, view_analysis)
//!        Gets forwarded to createViewDomain(TensorDomain, view_analysis)
//!        createViewDomain creates the new root domain, and calls
//!        createRfactorDomain on view_analysis.transforms().
//!   5) brooadcast will be called with view_analysis.broadcast_axes
//!
//! TODO: Caching assumes that all size-1 inputs are correctly marked as a
//! broadcast dimension. We should probably remove the runtime size-1 merge
//! support in find transformation.
//!
//! Simple abstract class to record transformation and the indices required to
//! apply it.
class Transform : public PolymorphicBase {
 public:
  virtual std::string toString() const = 0;

  int64_t index() const {
    return index_;
  }

 protected:
  // Relevant location information for the transformation. Stored information is
  // related to when we have to apply that transformation (see long comment at
  // top of this file).
  Transform(int64_t index) : index_(index) {}

  const int64_t index_ = 0;
};

class ViewTransform : public Transform {
 public:
  // Function to apply the transformation. Transformation is applied on
  // current_transformed_domain. root_domain is required here to replace
  // IterDomains so we can flip the rfactor flag on the root domain if it's
  // involved in merge/split trasnforms to produce the rfactor domain.
  virtual void createRfactorDomain(
      std::vector<IterDomain*>& root_domain,
      std::vector<IterDomain*>& current_transformed_domain) = 0;

  // Convenience function to replace id in root_domain with an id that has
  // expand expanded, and rfactor flag turned on.
  static IterDomain* replaceRootIdWithRFactor(
      std::vector<IterDomain*>& root_domain,
      IterDomain* id) {
    auto root_domain_it = std::find(root_domain.begin(), root_domain.end(), id);

    TORCH_INTERNAL_ASSERT(
        root_domain_it != root_domain.end(),
        "Wanted to replace ",
        id->toString(),
        " in root with an rfactor dimension, but IterDomain was not found in root.");

    auto root_domain_pos = std::distance(root_domain.begin(), root_domain_it);

    bool is_expanded_dim = id->hasExpandedExtent();

    auto extent = is_expanded_dim ? id->expandedExtent() : id->extent();

    auto cloned_id =
        IterDomainBuilder(id)
            .iter_type(
                is_expanded_dim ? IterType::Iteration : id->getIterType())
            .extent(extent)
            .expanded_extent(nullptr)
            .is_rfactor_domain(true)
            .build();

    root_domain.erase(root_domain.begin() + root_domain_pos);
    root_domain.insert(root_domain.begin() + root_domain_pos, cloned_id);
    return cloned_id;
  }

  // Debugging utility to convert the transformation into a string.
  virtual std::string toString() const override = 0;

 protected:
  ViewTransform(const int64_t& index) : Transform(index) {}
};

namespace {
//! The merge tranformation either combines two root iterDomains together OR
//! the last rfactor iterDomain with a root iterDomain. Unlike the general
//! TensorView merge there's no merging across axes not placed in consecutive
//! positions for View.
class MergeTransform final : public ViewTransform {
 public:
  MergeTransform(int64_t index) : ViewTransform(index) {}

  virtual std::string toString() const override {
    std::stringstream ss;
    ss << "Merge at index: " << index_;
    return ss.str();
  }

  void createRfactorDomain(
      std::vector<IterDomain*>& root_domain,
      std::vector<IterDomain*>& current_transformed_domain) override {
    TORCH_INTERNAL_ASSERT(
        (index_ + 1) < (int64_t)current_transformed_domain.size(),
        "Tried to apply: ",
        toString(),
        "\t To domain: \t",
        current_transformed_domain);

    // Assumed to never merge over non-contiguous dimensions.
    IterDomain* outer_id = current_transformed_domain[index_];
    if (!outer_id->isRFactorProduct()) {
      outer_id = replaceRootIdWithRFactor(root_domain, outer_id);
    }

    IterDomain* inner_id = current_transformed_domain[index_ + 1];
    if (!inner_id->isRFactorProduct()) {
      inner_id = replaceRootIdWithRFactor(root_domain, inner_id);
    }

    TORCH_INTERNAL_ASSERT(
        outer_id->start()->isZeroInt() && inner_id->start()->isZeroInt(),
        "Didn't expect to apply view transformations on an iter domain",
        " starting at a non-zero position.");

    auto merged_extent = mul(outer_id->extent(), inner_id->extent());

    auto new_merged_id =
        IterDomainBuilder(FusionGuard::getCurFusion()->zeroVal(), merged_extent)
            .is_rfactor_domain(true)
            .build();

    IrBuilder::create<Merge>(new_merged_id, outer_id, inner_id);

    current_transformed_domain.erase(
        current_transformed_domain.begin() + index_);
    current_transformed_domain.erase(
        current_transformed_domain.begin() + index_);
    current_transformed_domain.insert(
        current_transformed_domain.begin() + index_, new_merged_id);
  }
};

//! The split tranformation creates two new iterDomains via an outer split.
class SplitTransform final : public ViewTransform {
 public:
  SplitTransform(const int64_t index, int64_t split_factor)
      : ViewTransform(index), split_factor_(split_factor) {
    TORCH_INTERNAL_ASSERT(
        split_factor > 0,
        "Split factors must be greater than 0, but found ",
        split_factor,
        " during view transformation.");
  }

  virtual std::string toString() const override {
    std::stringstream ss;
    ss << "Split Index at: " << index_ << " by: " << split_factor_ << std::endl;
    return ss.str();
  }

  void createRfactorDomain(
      std::vector<IterDomain*>& root_domain,
      std::vector<IterDomain*>& current_transformed_domain) override {
    TORCH_INTERNAL_ASSERT(
        index_ < (int64_t)current_transformed_domain.size(),
        "Index: \t",
        index_,
        "\t Domain Size:\t",
        current_transformed_domain.size());

    auto factor = IrBuilder::create<Int>(split_factor_);

    IterDomain* id = current_transformed_domain[index_];
    if (!id->isRFactorProduct()) {
      id = replaceRootIdWithRFactor(root_domain, id);
    }

    TORCH_INTERNAL_ASSERT(
        id->start()->isZeroInt(),
        "Didn't expect to apply view transformations on an iter domain",
        " starting at a non-zero position.");

    Val* remainder = ceilDiv(id->extent(), factor);

    // outer loop IterDomain
    IterDomain* factor_id =
        IterDomainBuilder(FusionGuard::getCurFusion()->zeroVal(), factor)
            .parallel_type(id->getParallelType())
            .iter_type(id->getIterType())
            .is_rfactor_domain(true)
            .build();

    // inner loop IterDomain
    IterDomain* remainder_id =
        IterDomainBuilder(
            FusionGuard::getCurFusion()->zeroVal(), remainder->as<Int>())
            .is_rfactor_domain(true)
            .build();

    IrBuilder::create<Split>(factor_id, remainder_id, id, factor, false);

    current_transformed_domain.erase(
        current_transformed_domain.begin() + index_);
    current_transformed_domain.insert(
        current_transformed_domain.begin() + index_, remainder_id);
    current_transformed_domain.insert(
        current_transformed_domain.begin() + index_, factor_id);
  }

  int64_t split_factor() const {
    return split_factor_;
  }

 private:
  const int64_t split_factor_ = 0;
};

//! For any singleton dimensions in the new view, we create an implicit
//! broadcast dimension. We apply these transforms after the trivial reduction
//! and view transformation steps.
class BroadcastTransform final : public Transform {
 public:
  BroadcastTransform(int64_t index) : Transform(index) {}

  virtual std::string toString() const override {
    std::stringstream ss;
    ss << "Broadcast at: " << index_ << std::endl;
    return ss.str();
  }
};

//! For any implicit broadcast dimensions in the original view, we remove
//! them using a trivial reduction.
class TrivialReductionTransform final : public Transform {
 public:
  TrivialReductionTransform(int64_t index) : Transform(index) {}

  virtual std::string toString() const override {
    std::stringstream ss;
    ss << "Trivial reduction at: " << index_ << std::endl;
    return ss.str();
  }
};

//! The primary class that generates the transformations to go from
//! the original view to the new view.
class AnalyzeViewTransformation {
 public:
  AnalyzeViewTransformation(
      const std::vector<int64_t>& original_view,
      const std::vector<int64_t>& new_view,
      std::vector<IterDomain*> root_domain = {})
      : root_domain_not_provided_(root_domain.empty()),
        root_domain_(root_domain),
        root_is_transformed_(original_view.size(), false),
        original_view_(original_view),
        new_view_(new_view) {
    TORCH_INTERNAL_ASSERT(
        root_domain.empty() || original_view.size() == root_domain.size(),
        "Incoming domain must match the original view sizes for view.");
    // Check that the product of original and new view std::vector<int64_t> are
    // equal.
    const int64_t kOriginalNumElements = std::accumulate(
        original_view_.begin(), original_view_.end(), 1, std::multiplies<>());
    const int64_t kNewNumElements = std::accumulate(
        new_view_.begin(), new_view.end(), 1, std::multiplies<>());
    TORCH_INTERNAL_ASSERT(
        kOriginalNumElements == kNewNumElements,
        "Total element counts across view operation must match.");
  }

  AnalyzeViewConstraint constraint() {
    findTransformation();

    AnalyzeViewConstraint constraint;
    constraint.original_constraint =
        std::vector<int64_t>(original_view_.begin(), original_view_.end());
    for (auto i : c10::irange(constraint.original_constraint.size())) {
      if (constraint.original_constraint[i] != 1) {
        constraint.original_constraint[i] = 0;
      }
    }

    constraint.new_constraint =
        std::vector<int64_t>(new_view_.begin(), new_view_.end());
    for (auto i : c10::irange(constraint.new_constraint.size())) {
      if (constraint.new_constraint[i] != 1) {
        constraint.new_constraint[i] = 0;
      }
    }

    for (auto trivial_reduce : trivial_reduction_transforms_) {
      constraint.trivial_reduction_string.push_back(trivial_reduce->index());
    }

    for (auto broadcast : broadcast_transforms_) {
      constraint.broadcast_string.push_back(broadcast->index());
    }

    // Dilimeter for split/merge transforms is -2
    for (auto split_merge : view_transforms_) {
      if (split_merge->isA<SplitTransform>()) {
        constraint.split_merge_string.push_back(split_merge->index());
        constraint.split_merge_string.push_back(
            split_merge->as<SplitTransform>()->split_factor());
        constraint.split_merge_string.push_back(-2);
      } else {
        TORCH_INTERNAL_ASSERT(
            split_merge->isA<MergeTransform>(),
            "Unrecognized transformation found.");
        constraint.split_merge_string.push_back(split_merge->index());
        constraint.split_merge_string.push_back(-2);
      }
    }

    return constraint;
  }

  // Fill out all the information needed in AnalyzeViewResult, this should
  // contain all the information of what's required to perform the view
  // operation.
  AnalyzeViewResult run() {
    // Find all the transformations to go from the original tensor domain to the
    // final output of the view operations.
    findTransformation();

    auto trivial_reduction_axes = generateTrivialReductionAxes();
    auto broadcast_axes = generateBroadcastAxes();

    // Move data to AnalyzeViewResult and return it.
    return {broadcast_axes, trivial_reduction_axes, view_transforms_};
  }

 private:
  // Returns the bool flags that should be used to broadcast the output view
  // tensor
  std::vector<bool> generateBroadcastAxes() {
    std::vector<bool> broadcast_axes(new_view_.size(), false);
    for (auto& bcast : broadcast_transforms_) {
      broadcast_axes.at(bcast->index()) = true;
    }
    return broadcast_axes;
  }

  // Returns the positions for the trivial reductions to be performed before the
  // view operation
  std::vector<int> generateTrivialReductionAxes() {
    std::vector<int> reduction_axes;
    for (auto& tred : trivial_reduction_transforms_) {
      reduction_axes.push_back(tred->index());
    }
    return reduction_axes;
  }

  std::string toString() {
    std::stringstream output;
    output << "===============================" << std::endl;
    output << "old:";
    for (auto s : original_view_) {
      output << " " << s;
    }
    output << std::endl;

    output << "===============================" << std::endl;
    output << "new:";
    for (auto s : new_view_) {
      output << " " << s;
    }
    output << std::endl;

    output << "===============================" << std::endl;
    for (auto& trivial_reduction : trivial_reduction_transforms_) {
      output << trivial_reduction->toString() << "\n";
    }
    for (auto& split_or_merge : view_transforms_) {
      output << split_or_merge->toString() << "\n";
    }
    for (auto& broadcast : broadcast_transforms_) {
      output << broadcast->toString() << "\n";
    }
    output << "===============================" << std::endl;
    return output.str();
  }

  // Validation check after transformations are all found

  bool isImplicitBroadcast(int64_t original_view_index) const {
    if (root_domain_not_provided_) {
      return original_view_[original_view_index] == 1;
    } else {
      TORCH_INTERNAL_ASSERT(original_view_index < (int64_t)root_domain_.size());
      return root_domain_[original_view_index]->isImplicitBroadcast() &&
          !root_domain_[original_view_index]->hasExpandedExtent();
    }
  }

  //! Find the broadcast, merge and split operations necessary
  //! to transform the original view into the new view
  void findTransformation() {
    // There are three particularly important state indices we're working with.
    // There is:
    //   1) original_view_index which is indexing into the original tensor
    //      domain after all reductions are removed. This lines up with the last
    //      domain in original view that we added to current_size.
    //   2) transform_view_index which is the index of the transformations as
    //      we're virtually "developing" the output tensor domain (split/merge
    //      transformations post trivial reductions).
    //   3) The new_view_index which is directly associated with the new_view
    //      and the dimension in new_view we're currently trying to create.

    int64_t original_view_index = 0;
    int64_t transform_view_index = 0;
    int64_t new_view_index = 0;
    int64_t current_size = original_view_[0];

    // Safety counters to make sure we don't end up in an infinite loop.
    int64_t prev_original_view_index = std::numeric_limits<int64_t>::max();
    int64_t prev_new_view_index = std::numeric_limits<int64_t>::max();

    TORCH_INTERNAL_ASSERT(
        view_transforms_.empty(),
        "Already ran find transformation pass for View op, cannot run a second time.");

    // Iterate until original view is completely consumed and new view is
    // completely generated.
    while (original_view_index < (int64_t)original_view_.size() ||
           new_view_index < (int64_t)new_view_.size()) {
      TORCH_INTERNAL_ASSERT(
          !(prev_new_view_index == new_view_index &&
            prev_original_view_index == original_view_index),
          "Infinite loop detected in AnalyzeViewTransformation::findTransformation(). Bailing.");

      prev_new_view_index = new_view_index;
      prev_original_view_index = original_view_index;

      if (new_view_index >= (int64_t)new_view_.size()) {
        TORCH_INTERNAL_ASSERT(
            current_size == 1,
            "View is complete, but there's still some elements to distribute.");
      }

      if ((new_view_index + 1 >= (int64_t)new_view_.size() ||
           (new_view_[new_view_index + 1] != 1)) &&
          original_view_index + 1 < (int64_t)original_view_.size() &&
          original_view_[original_view_index + 1] == 1 &&
          !isImplicitBroadcast(original_view_index + 1)) {
        // Next index in original_view is runtime size 1 and next new view is
        // not, merge the size 1 into the current view before moving on. Even if
        // the current size and new view size match we could have a trailing
        // size 1 dimension on the input that needs to be merged in.
        view_transforms_.push_back(
            std::make_shared<MergeTransform>(transform_view_index));
        ++original_view_index;
        continue;
      }

      if (new_view_index < (int64_t)new_view_.size() &&
          // Still new dimensions to resolve and current size does resolve it.
          current_size == new_view_[new_view_index]) {
        // Keep this dimension, it's good to go, we hit a boundary where there's
        // a multiple of original dims, that matches a multiple of view dims.
        // Increment state and keep going.

        ++transform_view_index;
        ++new_view_index;
        ++original_view_index;

        // Update current_size with the next size in original view
        if (original_view_index < (int64_t)original_view_.size()) {
          current_size = original_view_[original_view_index];
        } else {
          current_size = 0;
        }
        continue;
      }

      // Compile time broadcast in new view, but not a matching one in original
      // view. Insert broadcast and increment new_view. Size 1 dimensions in
      // new_view that don't match up with runtime size 1's in original view are
      // assumed to be broadcast (not a split from a runtime domain).
      if (new_view_index < (int64_t)new_view_.size() &&
          new_view_[new_view_index] == 1) {
        broadcast_transforms_.push_back(
            std::make_shared<BroadcastTransform>(new_view_index));
        ++new_view_index;
        continue;
      }

      // If we run out of original_view dimensions we could still have broadcast
      // dimensions for new_view, but that should be hit before this point.
      TORCH_INTERNAL_ASSERT(
          current_size != 0,
          "View analysis failed, should never process an empty size unless we ",
          "simply need to add broadcasts to the post-view domain.");

      if (current_size == 1 && isImplicitBroadcast(original_view_index)) {
        // Original view has a compile time size 1 dimension, and it's not found
        // in the new_view_ (otherwise would have been caught in a branch
        // above). Do a trivial reduction.
        trivial_reduction_transforms_.push_back(
            std::make_shared<TrivialReductionTransform>(original_view_index));
        ++original_view_index;

        // Update original position and current size.
        if (original_view_index < (int64_t)original_view_.size()) {
          current_size = original_view_[original_view_index];
        } else {
          current_size = 0;
        }

        continue;
      }

      if (original_view_index + 1 < (int64_t)original_view_.size() &&
          isImplicitBroadcast(original_view_index + 1)) {
        // Original view has a compile time size 1 dimension, and it's
        // interfering with necessary transformations. Do a trivial reduction.
        ++original_view_index;
        trivial_reduction_transforms_.push_back(
            std::make_shared<TrivialReductionTransform>(original_view_index));

        continue;
      }

      // We're only left with performing transformations to match a new_view
      // dimension, there must be an activew new_view.
      TORCH_INTERNAL_ASSERT(
          new_view_index < (int64_t)new_view_.size(),
          "Expecting to still have new dimensions to work on in view, but none left.");

      if (new_view_index < (int64_t)new_view_.size() &&
          current_size % new_view_[new_view_index] == 0) {
        // Insert split to generate the next new_view domain.
        view_transforms_.push_back(std::make_shared<SplitTransform>(
            transform_view_index, new_view_[new_view_index]));
        current_size /= new_view_[new_view_index];
        TORCH_INTERNAL_ASSERT(current_size > 1, "This should be unreachable.");
        // Update transform and new since a split doesn't increment from the
        // original domain we're working on.
        ++transform_view_index;
        ++new_view_index;
        continue;
      }

      // Need more of the original_view dimension to resolve the new_view
      // dimension, merge the next dimension in.
      TORCH_INTERNAL_ASSERT(
          original_view_index + 1 < (int64_t)original_view_.size(),
          "Expecting to still have original dimensions to work on in view, but none left.");

      view_transforms_.push_back(
          std::make_shared<MergeTransform>(transform_view_index));
      current_size *= original_view_[++original_view_index];
    }
  }

 private:
  std::vector<std::shared_ptr<ViewTransform>> view_transforms_;
  std::vector<std::shared_ptr<BroadcastTransform>> broadcast_transforms_;
  std::vector<std::shared_ptr<TrivialReductionTransform>>
      trivial_reduction_transforms_;

  // If root domain isn't provided always assume size-1 dimensions are
  // compile-time dimensions. TODO: Remove runtime size-1 dimension support.
  // This should be cached higher in the stack.
  const bool root_domain_not_provided_ = true;

  const std::vector<IterDomain*> root_domain_;
  // Track if the root ID was transformed or kept ()
  std::vector<bool> root_is_transformed_;
  const std::vector<int64_t>& original_view_;
  const std::vector<int64_t>& new_view_;
};

//! Create new TensorDomain with a new root domain and modified rfactor domains
//! using the specified view transformations. Original domain should already be
//! without reduction axes.
TensorDomain* createViewDomain(
    TensorDomain* original_domain,
    const AnalyzeViewResult& view_analysis) {
  FUSER_PERF_SCOPE("createViewDomain");
  TORCH_INTERNAL_ASSERT(!view_analysis.transforms.empty());

  std::vector<IterDomain*> new_root_domain;
  auto orig_root_domain = original_domain->getMaybeRFactorDomain();

  // Apply trivial reductions.
  for (auto id_i : c10::irange(orig_root_domain.size())) {
    auto id = orig_root_domain[id_i];
    if (id->isReduction()) {
      continue;
    }
    if (std::find(
            view_analysis.trivial_reduction_axes.begin(),
            view_analysis.trivial_reduction_axes.end(),
            (int)id_i) != view_analysis.trivial_reduction_axes.end()) {
      continue;
    }

    new_root_domain.push_back(id->cloneWithoutRFactor());
  }

  std::vector<IterDomain*> new_rfactor_domain(
      new_root_domain.begin(), new_root_domain.end());

  // Apply rfactor transformations.
  for (auto& t : view_analysis.transforms) {
    t->createRfactorDomain(new_root_domain, new_rfactor_domain);
  }

  return IrBuilder::create<TensorDomain>(
      new_root_domain,
      new_rfactor_domain,
      new_rfactor_domain,
      std::vector<bool>(new_rfactor_domain.size(), true));
}

} // namespace

std::pair<std::vector<int64_t>, std::vector<int64_t>> inferViewShapes(
    const std::vector<int64_t>& original_sizes,
    const std::vector<int64_t>& new_sizes) {
  bool valid_original_sizes = std::all_of(
      original_sizes.begin(), original_sizes.end(), [](int64_t dim) {
        return dim > 0;
      });
  TORCH_INTERNAL_ASSERT(valid_original_sizes);

  std::vector<int64_t> original_view(
      original_sizes.begin(), original_sizes.end());
  std::vector<int64_t> new_view(new_sizes.size());

  // TODO: refactor
  int64_t dynamic_index = -1;
  int64_t new_size_num_elements = 1;
  for (int64_t idx = 0; idx < (int64_t)new_sizes.size(); ++idx) {
    if (new_sizes[idx] == -1) {
      TORCH_INTERNAL_ASSERT(
          dynamic_index == -1, "Only one dimension can by inferred.")
      dynamic_index = idx;
    } else {
      TORCH_INTERNAL_ASSERT(new_sizes[idx] > 0);
      new_size_num_elements *= new_sizes[idx];
      new_view[idx] = new_sizes[idx];
    }
  }

  const int64_t kNumElements = std::accumulate(
      original_view.begin(), original_view.end(), 1, std::multiplies<>());
  if (dynamic_index != -1) {
    new_view[dynamic_index] = kNumElements / new_size_num_elements;
  }

  return {original_view, new_view};
}

//! Generates the transformations necessary to convert
//! from the original view into the new view.
AnalyzeViewResult analyzeView(
    const TensorView* original_view_tv,
    const std::vector<int64_t>& original_sizes,
    const std::vector<int64_t>& new_sizes) {
  FUSER_PERF_SCOPE("analyzeView");
  TORCH_INTERNAL_ASSERT(
      original_sizes.size() > 0,
      "Empty original size not supported for view operation.");

  TORCH_INTERNAL_ASSERT(
      TensorDomain::noReductions(original_view_tv->getMaybeRFactorDomain())
          .size() == original_sizes.size());

  // Fill -1 dimension in new_std::vector<int64_t> with size infered from all
  // other values
  auto sizes = inferViewShapes(original_sizes, new_sizes);

  // Analysize the transformations required to go from original_sizes to
  // new_sizes
  AnalyzeViewTransformation analyzer(
      sizes.first /* original_view */,
      sizes.second /* new_view */,
      TensorDomain::noReductions(original_view_tv->getMaybeRFactorDomain()));
  return analyzer.run();
}

AnalyzeViewConstraint analyzeViewConstraint(
    const std::vector<int64_t>& original_sizes,
    const std::vector<int64_t>& new_sizes) {
  FUSER_PERF_SCOPE("analyzeViewConstraint");
  auto sizes = inferViewShapes(original_sizes, new_sizes);
  AnalyzeViewTransformation analyzer(
      sizes.first /* original_view */, sizes.second /* new_view */);
  return analyzer.constraint();
}

//! Create new TensorDomain with a modified rfactor domain using the specified
//! view transformations
TensorDomain* transformView(
    TensorDomain* original_domain,
    const AnalyzeViewResult& view_analysis) {
  FUSER_PERF_SCOPE("transformView");
  return createViewDomain(original_domain, view_analysis);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
