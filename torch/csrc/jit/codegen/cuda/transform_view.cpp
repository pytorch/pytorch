#include <torch/csrc/jit/codegen/cuda/transform_view.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/ir_internal_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

struct ViewIndexState {
  // The index into the new view
  size_t new_view_index = 0;

  // The index into the original view
  size_t original_view_index = 0;

  // The index into the transform view
  size_t transform_view_index = 0;

  // The number of broadcast axes before this transformation
  size_t broadcast_offset = 0;

  // The number of trivial reduction axes before this transformation
  size_t trivial_reduction_offset = 0;

  // The number of split transformations
  size_t split_offset = 0;

  // The number of merge transformations
  size_t merge_offset = 0;
};

//! Base class for all tranformations
class Transform : public PolymorphicBase {
 public:
  virtual void toString(std::ostream& output) const = 0;

  size_t index() const {
    return index_;
  }

  size_t originalIndex() const {
    return original_index_;
  }

  size_t newIndex() const {
    return new_index_;
  }

 protected:
  Transform(const ViewIndexState& state, size_t index)
      : index_(index),
        original_index_(state.original_view_index),
        new_index_(Transform::computeNewIndex(state)) {}

  const size_t index_ = 0;
  const size_t original_index_ = 0;
  const size_t new_index_ = 0;

  static size_t computeNewIndex(const ViewIndexState& state) {
    return state.original_view_index - state.trivial_reduction_offset +
        state.split_offset - state.merge_offset + state.broadcast_offset;
  }
};

//! Base class for all view tranformations - Merge, Split, Keep
//! These transforms require updating the rfactor domain of the view TensorView
//! and are applied after removing any unnecessary trivial broadcasts.
class ViewTransform : public Transform {
 public:
  virtual void createRfactorDomain(
      const std::vector<IterDomain*>& new_root_domain,
      std::vector<IterDomain*>& rfactor_domain) = 0;

  virtual bool isOriginalAxisDynamic() const = 0;

 protected:
  ViewTransform(const ViewIndexState& state)
      : Transform(state, ViewTransform::computeIndex(state)) {}

  static size_t computeIndex(const ViewIndexState& state) {
    return state.original_view_index - state.trivial_reduction_offset;
  }
};

namespace {
typedef std::vector<size_t> Sizes;
const size_t kEmptyAxis = 0;
const size_t kSingletonAxis = 1;

//! The merge tranformation either combines two root iterDomains together OR
//! the last rfactor iterDomain with a root iterDomain.
class MergeTransform final : public ViewTransform {
 public:
  MergeTransform(const ViewIndexState& state, bool is_last_axis_rfactor)
      : ViewTransform(state), is_last_axis_rfactor_(is_last_axis_rfactor) {}

  void toString(std::ostream& output) const override {
    output << "Merge Index: " << index_ << " RF: " << is_last_axis_rfactor_
           << std::endl;
  }

  bool isOriginalAxisDynamic() const override {
    return false;
  }

  void createRfactorDomain(
      const std::vector<IterDomain*>& new_root_domain,
      std::vector<IterDomain*>& rfactor_domain) override {
    TORCH_INTERNAL_ASSERT(
        index_ >= 0 && (index_ + 1) < new_root_domain.size(),
        "Index: \t",
        index_,
        "\t Domain Size:\t",
        new_root_domain.size());

    IterDomain* merged_id = nullptr;
    if (is_last_axis_rfactor_) {
      TORCH_INTERNAL_ASSERT(!rfactor_domain.empty());
      merged_id = rfactor_domain.back();
      rfactor_domain.pop_back();
    } else {
      merged_id = new_root_domain[index_];
    }

    auto merged_extent =
        mul(merged_id->extent(), new_root_domain[index_ + 1]->extent());

    auto new_merged_id = IrBuilder::create<IterDomain>(
        FusionGuard::getCurFusion()->zeroVal(),
        merged_extent,
        ParallelType::Serial,
        IterType::Iteration,
        true);

    IrBuilder::create<Merge>(
        new_merged_id, merged_id, new_root_domain[index_ + 1]);

    rfactor_domain.push_back(new_merged_id);
  }

 private:
  const bool is_last_axis_rfactor_ = false;
};

//! The split tranformation creates two new iterDomains via an outer split.
class SplitTransform final : public ViewTransform {
 public:
  SplitTransform(
      const ViewIndexState& state,
      bool is_last_axis_rfactor,
      size_t split_factor)
      : ViewTransform(state),
        is_last_axis_rfactor_(is_last_axis_rfactor),
        split_factor_(split_factor) {}

  void toString(std::ostream& output) const override {
    output << "Split Index: " << index_ << " RF: " << is_last_axis_rfactor_
           << " ARG: " << split_factor_ << std::endl;
  }

  bool isOriginalAxisDynamic() const override {
    return false;
  }

  void createRfactorDomain(
      const std::vector<IterDomain*>& new_root_domain,
      std::vector<IterDomain*>& rfactor_domain) override {
    TORCH_INTERNAL_ASSERT(
        index_ >= 0 && index_ < new_root_domain.size(),
        "Index: \t",
        index_,
        "\t Domain Size:\t",
        new_root_domain.size());

    auto factor = IrBuilder::create<Int>(split_factor_);

    IterDomain* id = nullptr;
    if (is_last_axis_rfactor_) {
      TORCH_INTERNAL_ASSERT(!rfactor_domain.empty());
      id = rfactor_domain.back();
      rfactor_domain.pop_back();
    } else {
      id = new_root_domain[index_];
    }

    Val* remainder = ceilDiv(id->extent(), factor);

    // outer loop IterDomain
    IterDomain* factor_id = IrBuilder::create<IterDomain>(
        FusionGuard::getCurFusion()->zeroVal(),
        factor,
        id->getParallelType(),
        id->getIterType(),
        true);

    // inner loop IterDomain
    IterDomain* remainder_id = IrBuilder::create<IterDomain>(
        FusionGuard::getCurFusion()->zeroVal(),
        remainder->as<Int>(),
        ParallelType::Serial,
        IterType::Iteration,
        true);

    IrBuilder::create<Split>(factor_id, remainder_id, id, factor, false);

    rfactor_domain.push_back(factor_id);
    rfactor_domain.push_back(remainder_id);
  }

 private:
  const bool is_last_axis_rfactor_ = false;
  const size_t split_factor_ = 0;
};

//! The Keep transform moves the root iterDomain to the rfactor domain.
class KeepTransform final : public ViewTransform {
 public:
  KeepTransform(const ViewIndexState& state) : ViewTransform(state) {}

  void toString(std::ostream& output) const override {
    output << "Keep Index: " << index_ << std::endl;
  }

  bool isOriginalAxisDynamic() const override {
    return true;
  }

  void createRfactorDomain(
      const std::vector<IterDomain*>& new_root_domain,
      std::vector<IterDomain*>& rfactor_domain) override {
    TORCH_INTERNAL_ASSERT(
        index_ >= 0 && index_ < new_root_domain.size(),
        "Index: \t",
        index_,
        "\t Domain Size:\t",
        new_root_domain.size());
    rfactor_domain.push_back(new_root_domain[index_]);
  }
};

//! For any singleton dimensions in the new view, we create an implicit
//! broadcast dimension. We apply these transforms after the trivial reduction
//! and view transformation steps.
class BroadcastTransform final : public Transform {
 public:
  BroadcastTransform(const ViewIndexState& state)
      : Transform(state, Transform::computeNewIndex(state)) {}

  void toString(std::ostream& output) const override {
    output << "Bcast Index: " << index_ << std::endl;
  }
};

//! For any implicit broadcast dimensions in the original view, we remove
//! them using a trivial reduction.
class TrivialReductionTransform final : public Transform {
 public:
  TrivialReductionTransform(const ViewIndexState& state)
      : Transform(state, TrivialReductionTransform::computeIndex(state)) {}

  void toString(std::ostream& output) const override {
    output << "1-Red Index: " << index_ << std::endl;
  }

 private:
  static size_t computeIndex(const ViewIndexState& state) {
    return state.original_view_index;
  }
};

//! The primary class that generates the transformations to go from
//! the original view to the new view.
class AnalyzeViewTransformation {
 public:
  AnalyzeViewTransformation(
      const Sizes& original_view,
      const Sizes& new_view,
      std::vector<IterDomain*> root_domain = {})
      : default_implicit_broadcast_(root_domain.empty()),
        root_domain_(root_domain),
        original_view_(original_view),
        new_view_(new_view),
        transform_view_(original_view) {
    // Check that the product of original and new view sizes are equal.
    const size_t kOriginalNumElements = std::accumulate(
        original_view_.begin(), original_view_.end(), 1, std::multiplies<>());
    const size_t kNewNumElements = std::accumulate(
        new_view_.begin(), new_view.end(), 1, std::multiplies<>());
    TORCH_INTERNAL_ASSERT(kOriginalNumElements == kNewNumElements);
  }

  AnalyzeViewConstraint constraint() {
    findTransformation();
    TORCH_INTERNAL_ASSERT(
        validate(),
        "Analyze View Transformation failed to find valid transformation.\n",
        toString());
    std::vector<int64_t> original_constraint(
        original_view_.begin(), original_view_.end());
    std::vector<int64_t> new_constraint(new_view_.begin(), new_view_.end());
    for (auto& vt : view_transforms_) {
      if (vt->isOriginalAxisDynamic()) {
        original_constraint[vt->originalIndex()] = -1;
        new_constraint[vt->newIndex()] = -1;
      }
    }
    return {original_constraint, new_constraint};
  }

  AnalyzeViewResult run() {
    findTransformation();

    TORCH_INTERNAL_ASSERT(
        validate(),
        "Analyze View Transformation failed to find valid transformation.\n",
        toString());

    // Skip view operations if all iterDomains are kept as-is
    bool all_keep_transforms = std::all_of(
        view_transforms_.begin(),
        view_transforms_.end(),
        [](std::shared_ptr<ViewTransform> vt) {
          return vt->isA<KeepTransform>();
        });
    if (all_keep_transforms) {
      view_transforms_.clear();
    }

    return {
        !broadcast_transforms_.empty(),
        generateBroadcastAxes(),
        generateTrivialReductionAxes(),
        view_transforms_};
  }

 private:
  std::vector<bool> generateBroadcastAxes() {
    std::vector<bool> broadcast_axes(new_view_.size(), false);
    for (auto& bcast : broadcast_transforms_) {
      broadcast_axes.at(bcast->index()) = true;
    }
    return broadcast_axes;
  }

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
    for (auto& move : trivial_reduction_transforms_) {
      move->toString(output);
    }
    for (auto& move : view_transforms_) {
      move->toString(output);
    }
    for (auto& move : broadcast_transforms_) {
      move->toString(output);
    }
    output << "===============================" << std::endl;
    return output.str();
  }

  //! is_index_merge_rhs - Does the original_view_index point to the rhs of the
  //! Merge transform
  //! is_last_axis_rfactor - Is the last iterDomain already in the rfactor
  //! domain?
  void addMergeTransform(bool is_index_merge_rhs, bool is_last_axis_rfactor) {
    // The invariant for merge transform is transform index = rhs_position-1
    if (is_index_merge_rhs) {
      // The original_view_index points to the rhs of the Merge transform.
      ViewIndexState clone_state(state_);
      --clone_state.original_view_index;
      view_transforms_.push_back(
          std::make_shared<MergeTransform>(clone_state, is_last_axis_rfactor));
    } else {
      // The original_view_index points to the rhs-1 invariant position.
      view_transforms_.push_back(
          std::make_shared<MergeTransform>(state_, is_last_axis_rfactor));
    }
    ++state_.merge_offset;
  }

  void addSplitTransform(size_t split_factor, bool is_last_axis_rfactor) {
    view_transforms_.push_back(std::make_shared<SplitTransform>(
        state_, is_last_axis_rfactor, split_factor));
    ++state_.split_offset;
    ++state_.new_view_index;
  }

  void addKeepTransform(bool is_last_axis_rfactor) {
    if (!is_last_axis_rfactor) {
      view_transforms_.push_back(std::make_shared<KeepTransform>(state_));
    }
    ++state_.new_view_index;
    ++state_.original_view_index;
    ++state_.transform_view_index;
  }

  void addBroadcastTransform() {
    broadcast_transforms_.push_back(
        std::make_shared<BroadcastTransform>(state_));
    ++state_.broadcast_offset;
    ++state_.new_view_index;
  }

  void addTrivialReductionTransform() {
    trivial_reduction_transforms_.push_back(
        std::make_shared<TrivialReductionTransform>(state_));
    ++state_.trivial_reduction_offset;
  }

  bool validate() const {
    if (state_.new_view_index != new_view_.size() ||
        state_.original_view_index != original_view_.size() ||
        state_.transform_view_index != transform_view_.size()) {
      return false;
    }
    return true;
  }

  bool isImplicitBroadcast(size_t original_view_index) const {
    if (default_implicit_broadcast_) {
      return original_view_[original_view_index] == 1;
    } else {
      TORCH_INTERNAL_ASSERT(!root_domain_.empty());
      return root_domain_[original_view_index]->isImplicitBroadcast();
    }
  }

  //! This utility class merges a fixed set of axes together
  //! according to some invariant. Implicit broadcast axes cannot be
  //! merged with standard iterDomains, so they are handled separately
  //! with the Trivial Reduction transform.
  //!
  //! 1) MergeThenSplitAxes class merges axes until it is evenly divisible
  //!    by the split factor.
  //! 2) MergeAdjacentSingletonAxes class merges or reduces any
  //!    adjacent singleton dimensions.
  class MergeAxesInterface {
   protected:
    // See addMergeTransform for "is_index_merge_rhs" and
    // "is_last_axis_rfactor" descriptions
    void handle(bool is_index_merge_rhs, bool is_last_axis_rfactor) {
      findNumberOfMergeAxes();

      bool any_merge = false;
      for (size_t idx = 0; idx < num_merge_axes_; ++idx) {
        if (avt_->isImplicitBroadcast(state_.original_view_index)) {
          avt_->addTrivialReductionTransform();
        } else {
          avt_->addMergeTransform(
              is_index_merge_rhs, is_last_axis_rfactor || any_merge);
          any_merge = true;
        }
        updateViewIndexState();
      }

      epilogue(is_last_axis_rfactor || any_merge);
    }

    MergeAxesInterface(
        AnalyzeViewTransformation* avt,
        ViewIndexState& state,
        size_t initial_size = 1)
        : avt_(avt), state_(state), merged_axis_size_(initial_size) {}

    // Get the current position in the original view shape
    virtual size_t getCurrentAxisPosition() const = 0;
    virtual bool isMergeInvariantValid() const = 0;
    virtual void updateViewIndexState() = 0;

    // Optional function run after merging all axes together
    virtual void epilogue(bool is_last_axis_rfactor) = 0;

   private:
    bool isStateWithinBounds() const {
      return getCurrentAxisPosition() < avt_->original_view_.size();
    }

    // Get the number of adjacent dimensions for Merge Transform
    void findNumberOfMergeAxes() {
      num_merge_axes_ = 0;
      while (isStateWithinBounds() && isMergeInvariantValid()) {
        merged_axis_size_ *= avt_->original_view_[getCurrentAxisPosition()];
        ++num_merge_axes_;
      }
    }

   protected:
    AnalyzeViewTransformation* avt_;
    ViewIndexState& state_;

    // The number of adjacent axes for merge transform
    size_t num_merge_axes_ = 0;

    // The cumulative product of adjacent axes
    size_t merged_axis_size_ = 0;
  };

  //! We merge axes until the sum of the original sizes is evenly divisible by
  //! the new size. A Split transform is only valid if the axis is divisible
  //! without remainder.
  //!
  //! 1) If the merged axis is larger than new size, then add a Split transform.
  //!
  //! 2) If the merged axis is equal to the new size but neither Split nor Merge
  //! transforms were required, then keep the first non-singleton axis.
  //!
  //! 3) If the merged axis is equal to new size, then apply only the Merge
  //! transforms.
  //!
  class MergeThenSplitAxes : MergeAxesInterface {
   public:
    static void process(
        AnalyzeViewTransformation* avt,
        ViewIndexState& state,
        size_t initial_size,
        size_t split_factor,
        bool is_last_axis_rfactor) {
      MergeThenSplitAxes mtsa(
          avt, state, is_last_axis_rfactor, initial_size, split_factor);
      mtsa.handle(false /* is_index_merge_rhs */, is_last_axis_rfactor);
    }

   private:
    MergeThenSplitAxes(
        AnalyzeViewTransformation* avt,
        ViewIndexState& state,
        bool is_last_axis_rfactor,
        size_t initial_size,
        size_t split_factor)
        : MergeAxesInterface(avt, state, initial_size),
          split_factor_(split_factor) {}

    size_t getCurrentAxisPosition() const override {
      return state_.original_view_index + 1 + num_merge_axes_;
    }

    bool isMergeInvariantValid() const override {
      return merged_axis_size_ % split_factor_ != 0;
    }

    void updateViewIndexState() override {
      avt_->transform_view_[state_.transform_view_index] *=
          avt_->original_view_[state_.original_view_index + 1];
      avt_->transform_view_[state_.original_view_index + 1] = kEmptyAxis;
      ++state_.original_view_index;
    }

    void epilogue(bool is_last_axis_rfactor) override {
      if (merged_axis_size_ > split_factor_) {
        avt_->transform_view_[state_.transform_view_index] /= split_factor_;
        avt_->addSplitTransform(split_factor_, is_last_axis_rfactor);
      } else {
        avt_->addKeepTransform(is_last_axis_rfactor);
      }
    }

   private:
    const size_t split_factor_ = 0;
  };

  //! A utility class to merge any adjacent size-1 dimensions
  class MergeAdjacentSingletonAxes : MergeAxesInterface {
   public:
    static void process(AnalyzeViewTransformation* avt, ViewIndexState& state) {
      MergeAdjacentSingletonAxes masa(avt, state);
      masa.handle(
          true /* is_index_merge_rhs */, true /* is_last_axis_rfactor */);
    }

   private:
    MergeAdjacentSingletonAxes(
        AnalyzeViewTransformation* avt,
        ViewIndexState& state)
        : MergeAxesInterface(avt, state) {}

    size_t getCurrentAxisPosition() const override {
      return state_.original_view_index + num_merge_axes_;
    }

    bool isMergeInvariantValid() const override {
      return avt_->original_view_[getCurrentAxisPosition()] == kSingletonAxis;
    }

    void updateViewIndexState() override {
      ++state_.original_view_index;
      ++state_.transform_view_index;
    }

    void epilogue(bool is_last_axis_rfactor) override {}
  };

  //! Find the broadcast, merge and split operations necessary
  //! to transform the original view into the new view
  void findTransformation() {
    // The original and new view are processed from left to right.
    // old_view_index and new_view_index track the current position in each
    // view respectively.
    // kRfactor - Is the last iterDomain already in the rfactor domain?
    while (state_.new_view_index < new_view_.size() &&
           state_.original_view_index < original_view_.size()) {
      const auto kCurrentSize = transform_view_[state_.transform_view_index];
      auto is_last_axis_rfactor = transform_view_[state_.original_view_index] !=
          original_view_[state_.original_view_index];

      if (kCurrentSize == kEmptyAxis) {
        // If current size in transform view is 0, then it was already handled
        // and should be skipped.
        ++state_.transform_view_index;
      } else if (kCurrentSize == new_view_[state_.new_view_index]) {
        addKeepTransform(is_last_axis_rfactor);
      } else if (new_view_[state_.new_view_index] == kSingletonAxis) {
        addBroadcastTransform();
      } else {
        MergeThenSplitAxes::process(
            this,
            state_,
            kCurrentSize,
            new_view_[state_.new_view_index],
            is_last_axis_rfactor);
      }
    }

    MergeAdjacentSingletonAxes::process(this, state_);

    // Skip any root domains that were merged for any splits with remainder
    // OR any singleton axes
    while (state_.transform_view_index < transform_view_.size() &&
           transform_view_[state_.transform_view_index] <= kSingletonAxis) {
      ++state_.transform_view_index;
    }

    // Add broadcast axes for any remaining size 1 dimensions
    while (state_.original_view_index == original_view_.size() &&
           state_.new_view_index < new_view_.size() &&
           new_view_[state_.new_view_index] == kSingletonAxis) {
      addBroadcastTransform();
    }
  }

 private:
  ViewIndexState state_;

  std::vector<std::shared_ptr<ViewTransform>> view_transforms_;
  std::vector<std::shared_ptr<BroadcastTransform>> broadcast_transforms_;
  std::vector<std::shared_ptr<TrivialReductionTransform>>
      trivial_reduction_transforms_;

  bool default_implicit_broadcast_ = true;
  const std::vector<IterDomain*> root_domain_;
  const Sizes& original_view_;
  const Sizes& new_view_;

  // transform_view is a mutable view and is initialized with the original_view.
  // It is used to track the current state of the original tensor domain.
  //
  // When we merge dimensions in the original_view, we multiply the sizes of
  // the adjacent, merged dimensions. The product size is placed in the current
  // position of the transform_view, while the other dimensions are set to 0.
  //
  // When we add a Split transform, the current size is divided by the outer
  // split factor.
  //
  // Size-0 dimensions are automatically skipped.
  //
  // If transform size != original size for an axis, then the transformation
  // uses the last rfactor domain. Otherwise, it is a root domain
  // transformation.
  Sizes transform_view_;
};

//! Create new TensorDomain with a modified rfactor domain using the specified
//! view transformations
TensorDomain* createViewDomain(
    TensorDomain* original_domain,
    const std::vector<std::shared_ptr<ViewTransform>>& view_transforms) {
  FUSER_PERF_SCOPE("createViewDomain");

  TORCH_INTERNAL_ASSERT(!view_transforms.empty());

  std::vector<IterDomain*> new_root_domain;
  for (auto id : TensorDomain::noReductions(original_domain->getRootDomain())) {
    new_root_domain.push_back(id->clone());
  }

  std::vector<IterDomain*> rfactor_domain;
  for (auto& t : view_transforms) {
    t->createRfactorDomain(new_root_domain, rfactor_domain);
  }

  return IrBuilder::create<TensorDomain>(
      new_root_domain,
      rfactor_domain,
      rfactor_domain,
      std::vector<bool>(rfactor_domain.size(), true));
}

//! Infer -1 value in new view sizes from original view sizes
std::pair<Sizes, Sizes> inferNewViewShape(
    const std::vector<int64_t>& original_sizes,
    const std::vector<int64_t>& new_sizes) {
  bool valid_original_sizes = std::all_of(
      original_sizes.begin(), original_sizes.end(), [](int64_t dim) {
        return dim > 0;
      });
  TORCH_INTERNAL_ASSERT(valid_original_sizes);

  Sizes original_view(original_sizes.begin(), original_sizes.end());
  Sizes new_view(new_sizes.size());

  // TODO: refactor
  int64_t dynamic_index = -1;
  size_t new_size_num_elements = 1;
  for (size_t idx = 0; idx < new_sizes.size(); ++idx) {
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

  const size_t kNumElements = std::accumulate(
      original_view.begin(), original_view.end(), 1, std::multiplies<>());
  if (dynamic_index != -1) {
    new_view[dynamic_index] = kNumElements / new_size_num_elements;
  }

  return {original_view, new_view};
}

} // namespace

//! Generates the transformations necessary to convert
//! from the original view into the new view.
AnalyzeViewResult analyzeView(
    const TensorView* tv,
    const std::vector<int64_t>& original_sizes,
    const std::vector<int64_t>& new_sizes) {
  FUSER_PERF_SCOPE("analyzeView");
  TORCH_INTERNAL_ASSERT(
      TensorDomain::noReductions(tv->getMaybeRFactorDomain()).size() ==
      original_sizes.size());
  auto sizes = inferNewViewShape(original_sizes, new_sizes);
  AnalyzeViewTransformation analyzer(
      sizes.first /* original_view */,
      sizes.second /* new_view */,
      TensorDomain::noReductions(tv->getMaybeRFactorDomain()));
  return analyzer.run();
}

AnalyzeViewConstraint analyzeViewConstraint(
    const std::vector<int64_t>& original_sizes,
    const std::vector<int64_t>& new_sizes) {
  FUSER_PERF_SCOPE("analyzeViewConstraint");
  auto sizes = inferNewViewShape(original_sizes, new_sizes);
  AnalyzeViewTransformation analyzer(
      sizes.first /* original_view */, sizes.second /* new_view */);
  return analyzer.constraint();
}

//! Create new TensorDomain with a modified rfactor domain using the specified
//! view transformations
TensorDomain* transformView(
    TensorDomain* original_domain,
    const std::vector<std::shared_ptr<ViewTransform>>& view_transforms) {
  FUSER_PERF_SCOPE("transformView");
  return createViewDomain(original_domain, view_transforms);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
