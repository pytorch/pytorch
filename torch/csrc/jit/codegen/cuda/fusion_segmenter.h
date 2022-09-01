#pragma once

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_base_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/registry.h>
#include <torch/csrc/jit/codegen/cuda/utils.h>

#include <deque>
#include <list>
#include <unordered_set>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class SegmentedGroup;
class SegmentCandidateFinder;

// A directed edge on DAG,
// Wrapper for values, edges between segmented groups which are made up
// of Exprs. Multiple edges can exist between segmented groups.
struct SegmentedEdge {
  SegmentedEdge(SegmentedGroup* from, SegmentedGroup* to, Val* val)
      : from(from), to(to), val(val) {}

  SegmentedGroup* from;
  SegmentedGroup* to;
  Val* val;

  void print() const;
};

std::ostream& operator<<(std::ostream& os, const SegmentedEdge* edge);

//! Groups together expressions which create a segmented group
//! Can be used to produce fusions
class TORCH_CUDA_CU_API SegmentedGroup {
 public:
  SegmentedGroup(SegmentedFusion* segmented_fusion)
      : segmented_fusion_(segmented_fusion) {}

  SegmentedGroup(Expr* expr, SegmentedFusion* segmented_fusion)
      : segmented_fusion_(segmented_fusion) {
    exprs_.push_back(expr);
  }

  //! Checks if this group takes original fusion's input
  bool isInputGroup() {
    return !input_vals.empty();
  };

  //! Checks if this group is used any where in the segmented fusion
  bool isConnected() const {
    return !producer_edges.empty() || !consumer_edges.empty() ||
        !output_vals.empty();
  }

  //! returns the id assigned by segment pass
  int groupId() const {
    return group_id_;
  }

  //! Returns inputs that this group shares with the original fusion
  const auto& inputs() const {
    return input_vals;
  }

  //! Returns outputs that this group shares with the original fusion
  const auto& outputs() const {
    return output_vals;
  }

  //! Returns the schedule heuristic associated with this group
  ScheduleHeuristic heuristic() const {
    return heuristic_;
  }

  //! Returns the exprs that make up this group
  const auto& exprs() const {
    return exprs_;
  }

  //! Debug print function
  void print() const;

  //! Returns the segmented fusion that this group is in
  SegmentedFusion* segmentedFusion() const {
    return segmented_fusion_;
  }

  //! Utility to re-collect the operators included in this
  //!  segmented group after updating the group boundary.
  void resetExprList();

  //! Try to get a scheduler entry for this group with
  //!  the given runtime info.
  //! Returns a new scheduler with the same heuristics
  //!  for this group if possible.
  //!  Note that the schedule params can be different.
  //! Returns a nullopt if this group cannot be scheduled
  //!  with the same heuristics.
  c10::optional<std::unique_ptr<SchedulerEntry>> getMaybeSchedulerEntry(
      SchedulerRuntimeInfo& runtime_info);

 public:
  //! "Ancestor nodes", towards inputs of segmentedDAG
  std::vector<SegmentedEdge*> producer_edges;

  //! "Descendent nodes", towards outputs of segmentedDAG
  std::vector<SegmentedEdge*> consumer_edges;

  //! Composite Fusion inputs in this group
  std::vector<Val*> input_vals;

  //! Composite Fusion outputs in this group
  std::vector<Val*> output_vals;

 private:
  friend class SegmentCandidateFinder;
  friend class SegmentedFusion;
  friend class FusionKernelRuntime;
  friend class TranslateApplicableWelford;

  //! unique identifier of group in the segmented fusion
  int group_id_ = -1;

  //! The scheduler to use for compiling this group
  ScheduleHeuristic heuristic_ = ScheduleHeuristic::None;

  //! Exprs that make up the group
  std::vector<Expr*> exprs_;

  //! Maximum path distance from an input segmented group required for
  //! Theorem 4.2
  int level_ = -1;

  //! traversal marker, has this node already been processed
  bool visited_ = false;

  //! Did we select another group to merge with
  SegmentedGroup* merge_with_ = nullptr;

  //! if we selected another group to merge, which edge is to be contracted
  SegmentedEdge* merge_through_ = nullptr;

  //! Has this node been merged?
  bool merged_ = false;

 private:
  //! Utility to convert edge vector to value vector
  std::vector<Val*> edgesToVals(const std::vector<SegmentedEdge*>& se_v);

  //! Reset method to call at begining of each
  //!  merge node iteration
  void clearTraversalInfo();

  //! To be called at the very end of segment fusion
  //!  no more segment merging should be done beyond
  void finalize();

  //! Return all segmented groups connected with *this
  std::vector<SegmentedGroup*> getNeighbors();

  //! Utility struct to represent a group connection
  //!  both the group to connect with and the edge
  //!  to connect through
  struct NeighborGroup {
    NeighborGroup(SegmentedGroup* g, SegmentedEdge* e) : group(g), edge(e) {}
    SegmentedGroup* group;
    SegmentedEdge* edge;
  };

  //! TODO: May want to sort this based on size of connections between this and
  //! neighbors as well as if the connection is an output of the fusion (has to
  //! be saved to gmem anyways)
  std::vector<NeighborGroup> getNeighborGroups();

  //! Look at all neighbors of this and return who this could merge with based
  //! on level values of this, neighbors, and merged neighbors of neighbors
  std::vector<NeighborGroup> getMergeCandidates();

  //! Assign schedule heuristic to this group
  void setHeuristic(ScheduleHeuristic sh) {
    heuristic_ = sh;
  }

  //! Assign Id for this group
  void setID(int id) {
    TORCH_INTERNAL_ASSERT(group_id_ == -1);
    group_id_ = id;
  }

  //! SegmentedFusion this group belongs to
  SegmentedFusion* segmented_fusion_;
};

std::ostream& operator<<(std::ostream& os, const SegmentedGroup* group);

//! Auxiliary class for storing heuristics. The managed data is either
//!  a single scheduler entry for complete fusion,
//!  or a vector of schedulers, one for each segment, for segmented fusion.
class TORCH_CUDA_CU_API FusionHeuristics {
  using SchedulerEntryOwningPtr = std::unique_ptr<SchedulerEntry>;

 public:
  //! Constructor for segmented fusion case. Created with empty list and
  //!  uses emplaceBack for inserting heuristics in order
  explicit FusionHeuristics() = default;

  //! Constructor for complete fusion case, generates the scheduler entry
  //!  for the fusion owning the given expression
  explicit FusionHeuristics(
      ScheduleHeuristic schedule_heuristic,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    heuristics_.emplace_back(SchedulerEntry::makeEntry(
        schedule_heuristic, runtime_info.fusion(), runtime_info, data_cache));
    is_segmented_ = false;
  }

  FusionHeuristics(const FusionHeuristics&) = delete;
  FusionHeuristics& operator=(const FusionHeuristics&) = delete;

  //! Place a scheduler entry on the list. Applies to segmented fusion only.
  void emplaceBack(SchedulerEntryOwningPtr&& pt) {
    TORCH_INTERNAL_ASSERT(is_segmented_);
    heuristics_.emplace_back(std::move(pt));
  }

  //! Returns list of schedulers for a segmneted fusion.
  const std::vector<SchedulerEntryOwningPtr>& heuristicsList() const {
    return heuristics_;
  }

  //! Returns the single scheduler for a complete fusion.
  SchedulerEntry* singleKernelHeuristics() {
    TORCH_INTERNAL_ASSERT(!is_segmented_);
    return heuristics_.begin()->get();
  }

 private:
  std::vector<SchedulerEntryOwningPtr> heuristics_;
  bool is_segmented_ = true;
};

//! Exported Interface for representing segmented fusion graph
//!   this class owns the segmented groups
class TORCH_CUDA_CU_API SegmentedFusion {
 public:
  explicit SegmentedFusion(std::unique_ptr<Fusion> fusion);

  //! Factory function for the un-segmented case, directly
  //!  constructs a "SegmentedFusion", with the given Fusion
  //!  as the only group.
  static std::unique_ptr<SegmentedFusion> fromCompleteFusion(
      std::unique_ptr<Fusion> fusion,
      ScheduleHeuristic heuristic);

  //! Is the fusion segmented?
  bool isSegmented() const {
    return !groups_.empty();
  }

  std::vector<SegmentedGroup*>& groups() {
    return groups_;
  }

  std::vector<SegmentedEdge*>& edges() {
    return edges_;
  }

  const std::vector<SegmentedGroup*>& cgroups() const {
    return groups_;
  }

  const std::vector<SegmentedEdge*>& cedges() const {
    return edges_;
  }

  //! Returns the original un-segmented fusion
  Fusion* completeFusion() const {
    return complete_fusion_.get();
  }

  const auto& inputs() const {
    return complete_fusion_->inputs();
  }

  const auto& outputs() const {
    return complete_fusion_->outputs();
  }

  Val* findAlias(Val* val) const {
    auto alias_it = complete_fusion_->ioAlias().find(val);
    if (alias_it != complete_fusion_->ioAlias().end()) {
      return alias_it->second;
    }
    return nullptr;
  }

  //! Make a clone of the group and convert to fusion
  std::unique_ptr<Fusion> makeFusion(SegmentedGroup* sg);

  //! Make heuristics for all groups in this segmented fusion
  std::unique_ptr<FusionHeuristics> makeInitialHeuristics(
      const KernelArgumentHolder& inputs);

  //! Inline Debug print for segmented fusion
  std::string toString(int verbosity) const;

  //! Debug drawing for graphviz
  void draw();

  //! Debug print for segmented fusions
  void print() const;

  //! API for adding groups
  SegmentedGroup* newGroup();

  //! API shortcut for adding a singleton group
  SegmentedGroup* newGroup(Expr* expr);

  //! API for adding edges
  SegmentedEdge* newEdge(SegmentedGroup* from, SegmentedGroup* to, Val* val);

  HeuristicSummary* getCachedHeuristicDataFor(SegmentedGroup* group);

 private:
  //! Unique name for segmented fusion
  int segmented_fusion_name_;

  //! States representing segmentation
  std::vector<SegmentedEdge*> edges_;
  std::vector<SegmentedGroup*> groups_;

  //! Owning object to explicitly manage groups and edges
  class Impl {
   public:
    explicit Impl(SegmentedFusion* sf) : owning_fusion_(sf) {}

    SegmentedGroup* makeGroup();
    SegmentedGroup* makeGroup(Expr*);
    SegmentedEdge* makeEdge(SegmentedGroup* from, SegmentedGroup* to, Val* val);
    void cleanUnused();

   private:
    using GroupPtr = std::unique_ptr<SegmentedGroup>;
    using EdgePtr = std::unique_ptr<SegmentedEdge>;
    std::vector<GroupPtr> groups_;
    std::vector<EdgePtr> edges_;
    SegmentedFusion* owning_fusion_;
  };
  Impl impl_;

  //! A Copy of original full fusion
  std::unique_ptr<Fusion> complete_fusion_;

  //! A set of intermediate tensors that need to be cast to fp16
  std::unordered_set<TensorView*> force_fp16_tv_set_;

  DataType force_half_precision_type_;

  //! Static traversal information to be used for fast heuristics lookup
  std::unordered_map<SegmentedGroup*, std::unique_ptr<HeuristicSummary>>
      heuristic_summary_cache_;

  // TODO: this class needs cleanup
 protected:
  friend class SegmentCandidateFinder;
  //! Make a heuristics entry for a group and parameters
  std::unique_ptr<SchedulerEntry> makeInitialSchedulerEntry(
      SegmentedGroup* sg,
      SchedulerRuntimeInfo& runtime_info);

  //! Cleanup function to be call at the end of fusion
  //!  segment pass
  void finalize();

  //! Collect all the intermediate tensors between segmented
  //!  groups that will cast to fp16
  void annotateFP16IntermediateTensors();

  //! Keep heuristic checking intermediate data
  void setCachedHeuristicDataFor(
      SegmentedGroup* group,
      std::unique_ptr<HeuristicSummary> data);

  //! Utility to give unique name for each segmented fusion
  static size_t segmentedFusionName() {
    static size_t counter = 0;
    return counter++;
  }
};

//! This is a base class for segmenter analysis
//!  provides the minimal implementation on header so that
//!  a unique_ptr can use this base class
//!  actual implementations of analyses are in the .cpp files
//! TODO: In the next refactor PR, should put segment candidate
//!  finder in .cpp file completely since API doesn't require these
//!  details
class SegmenterAnalysis : public PolymorphicBase {};
class GroupDependencyAnalysis;

// Manual node merging passes
class CombineReductions;

//! Options to configure/debug candidate finder
struct TORCH_CUDA_CU_API SegmentCandidateFinderOptions {
  bool run_translate_welford = true;
  bool run_combine_reductions = true;
  bool run_herrmann_merge = true;
  bool run_final_merge = true;
};

//!  SegmentCandidateFinder
//!    Responsible for going through DAG and proposing things we could try to
//!    fuse together, calls "canGenerateCode" on these proposed segments to see
//!    if they are valid and we can generate code for them.
//!  FusionSegment
//!    A group of exprs that are segmented together
//!  FusionSegmentConnections
//!    Holds vals and what they connect. In other words it's a val that is an
//!    output of a FusionSegment "from" and an input of FusionSegment "to".
//!    There's nothing preventing from a val being between segments twice.
//!    TODO: make sure there's nothing wrong with segmentation on nodes that
//!    have the same value input twice. i.e. (B = A*A)
//! Selecting segments to propose is based on the theorem 4.2 in the paper which
//! makes sure when segment the segmented graph will be a DAG (assumes Fusion is
//! already a DAG). The segmentation code relies on assumptions of DAG-ness
//! during segmentation, meaning proposed merging of groups must maintain the
//! DAG property of the graph.
//!
//! Julien Herrmann, Yusuf Özkaya, Bora Uçar, Kamer Kaya, Umit Catalyurek.
//! Multilevel Algorithms for Acyclic Partitioning of Directed Acyclic Graphs.
//! SIAM Journal on Scientific Computing, Society for Industrial and Applied
//! Mathematics, 2019, 41 (4), pp.A2117-A2145. ff10.1137/18M1176865ff.
//! ffhal02306566f
class TORCH_CUDA_CU_API SegmentCandidateFinder {
 public:
  // Perform segmentation on a copy of the given fusion
  static std::unique_ptr<SegmentedFusion> segment(
      const Fusion* fusion,
      const KernelArgumentHolder& inputs,
      SegmentCandidateFinderOptions options = SegmentCandidateFinderOptions()) {
    auto fusion_copy = std::make_unique<Fusion>(*fusion);
    if (isDebugDumpEnabled(DebugDumpOption::FusionSegments)) {
      std::cout << "Segment the fusion (Original Fusion Un-modified): "
                << std::endl;
      fusion_copy->printMath();
    }
    SegmentCandidateFinder scf(std::move(fusion_copy), inputs, options);
    return std::move(scf.segmented_fusion_);
  }

  // Perform segmentation on and take ownership of the given fusion
  static std::unique_ptr<SegmentedFusion> segment(
      std::unique_ptr<Fusion> fusion,
      const KernelArgumentHolder& inputs,
      SegmentCandidateFinderOptions options = SegmentCandidateFinderOptions()) {
    SegmentCandidateFinder scf(std::move(fusion), inputs, options);
    if (isDebugDumpEnabled(DebugDumpOption::FusionSegments)) {
      std::cout << "Segment the fusion (Original Fusion Un-modified): "
                << std::endl;
      scf.completeFusion()->printMath();
    }
    return std::move(scf.segmented_fusion_);
  }

  static bool TranslateWelfordInFusion(
      Fusion* fusion,
      const KernelArgumentHolder& runtime_inputs);

 private:
  // Perform segmentation on and take ownership of the given fusion
  SegmentCandidateFinder(
      std::unique_ptr<Fusion> fusion,
      const KernelArgumentHolder& inputs,
      SegmentCandidateFinderOptions options);

  void resetTraversal();

  void resetLevels();

  SegmentedGroup* mergeNodes();

  bool codeGenSupportedMerge(SegmentedGroup* group1, SegmentedGroup* group2);

  void findSegments();

  std::unordered_set<SegmentedEdge*> disconnectGroup(SegmentedGroup* group);

  std::vector<SegmentedGroup*>& groups() {
    TORCH_INTERNAL_ASSERT(
        segmented_fusion_ != nullptr, "Segment finder not owinging any fusion");
    return segmented_fusion_->groups();
  }

  std::vector<SegmentedEdge*>& edges() {
    TORCH_INTERNAL_ASSERT(
        segmented_fusion_ != nullptr, "Segment finder not owinging any fusion");
    return segmented_fusion_->edges();
  }

  Fusion* completeFusion() {
    TORCH_INTERNAL_ASSERT(
        segmented_fusion_ != nullptr, "Segment finder not owinging any fusion");
    return segmented_fusion_->completeFusion();
  }

  SchedulerRuntimeInfo& runtimeInfo() {
    return runtime_info_;
  }

  ExpressionEvaluator& expressionEvaluator() {
    return runtime_info_.expressionEvaluator();
  }

  //! Additional merging iteration, clean up the rest of
  //!  the merging opportunities
  //!  Herrmann et al. is a fast and safe algorithm for finding merge candidates
  //!  but can become too conservative in our use cases because we place
  //!  additional qualifiers on valid merges other than having to generate DAGs,
  //!  i.e. canSchedule. So we need a bruteforce final merging iteration as a
  //!  clean up pass. Cost isn't expected to be high since the graph at this
  //!  stage is already quite merged. Example cf. test_gpu.cpp:
  //!  FusionDAGMerging_CUDA
  //!
  //!  This merging algorithm is based on Theorem 4.1 of Herrmann et al.,
  //!   to check if a producer-consumer pair can be merged into one group,
  //!   it's enough to check if any other consumer of the producer also
  //!   produces the consumer.
  void finalMerge();

  //! Duplicate and add all exprs producing the used
  //!  scalar values in group
  void resolveScalarsInGroup(SegmentedGroup* group);

  //! Duplicate and add all exprs from "inputs" in the group, to complete
  //! inputs. These expressions are simply unary ops of inputs that we want to
  //! recompute for each segment, instead of computing and producing a segmented
  //! val. For example if we have:
  //! tv1 = tv0 * 2;
  //! tv3 = tv1 + tv2;
  //! tv4 = tv1 + tv4
  //! If we segmented on tv1, we would be producing an output for tv1 for 2
  //! groups that have tv3 or tv4, instead we could easily recompute tv1 from
  //! tv0.
  void resolveInputsInGroup(SegmentedGroup* group);

  //! Remove all scalar edges in group
  //!  (TODO: need structure better so we don't have to do this)
  void removeScalarEdges();

  //! Utility function to merge a vector of groups in one step,
  //!  need to check for DAG condition before using this method
  SegmentedGroup* mergeAllGivenGroups(
      const std::vector<SegmentedGroup*>& groups);

  //! Utility to remove a group and corresponding edges
  //!  TODO: remove inline versions of this as much as possible
  void eraseGroups(std::unordered_set<SegmentedGroup*>& groups_to_erase);

  void finalize();

  //! Return the resulting heuristic corresponding to the merged
  //!  group built by merging the two groups connected by edge
  ScheduleHeuristic deriveHeuristic(SegmentedGroup* edge);

  GroupDependencyAnalysis* getGroupDependency();

 protected:
  //! These are the merge node heuristic passes, should
  //!  eventually should have a dedicated interface
  //!  instead of keeping adding friends
  friend class CombineReductions;

  //! options to configure and debug the segment process
  SegmentCandidateFinderOptions options_;

  std::deque<SegmentedGroup*> to_visit_;
  std::vector<SegmentedGroup*> next_to_visit_;

  std::unordered_set<SegmentedGroup*> clean_up_groups_;
  std::unordered_set<SegmentedEdge*> clean_up_edges_;

  std::vector<SegmentedGroup*> to_merge_;

  std::unique_ptr<SegmentedFusion> segmented_fusion_;

  std::unique_ptr<SegmenterAnalysis> group_dependency_;

  SchedulerRuntimeInfo runtime_info_;

  //! Note:
  //!  Segmenter should eventually rely only on runtime_info_ for
  //!  safe caching. runtime_inputs_ is only used in translateWelford
  //!  to initialize expression evaluators on copies of the original
  //!  fusion, which doesn't use any un-cached info and is safe.
  //!
  //!  Directly using runtime_inputs_ in other cases is in general
  //!   risky.
  //!
  //!  To get rid of runtime_inputs_ we need mechanisms
  //!  to copy expression evaluator values from fusion
  //!  to a copy, or even better to a copy of a
  //!  sub-graph of original fusion.
  //! TODO:
  //!  implement the expression evaluator transfer and
  //!  remove runtime_inputs_ in a follow up.
  const KernelArgumentHolder& runtime_inputs_;
};

// TODO: Make as member functions on classes instead of global scope
TORCH_CUDA_CU_API std::string toString(const SegmentedGroup* group);
TORCH_CUDA_CU_API std::string toString(const SegmentedEdge* edge);
TORCH_CUDA_CU_API std::string toString(const SegmentedFusion* segmented_fusion);
TORCH_CUDA_CU_API std::string toString(
    const SegmentCandidateFinderOptions& segment_options);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
