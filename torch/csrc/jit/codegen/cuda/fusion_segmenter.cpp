#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/fusion_segmenter.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_cloner.h>
#include <torch/csrc/jit/codegen/cuda/ir_graphviz.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>

#include <sstream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

std::vector<SegmentedGroup::NeighborGroup> SegmentedGroup::getNeighborGroups() {
  std::vector<NeighborGroup> neighbors;
  for (auto inp : producer_edges) {
    if (inp->val->isFusionOutput()) {
      // Don't fuse across output nodes, would need to find another path.
      continue;
    }
    neighbors.emplace_back(inp->from, inp);
  }
  for (auto out : consumer_edges) {
    if (out->val->isFusionOutput()) {
      // Don't fuse across output nodes, would need to find another path.
      continue;
    }
    neighbors.emplace_back(out->to, out);
  }
  return neighbors;
}

std::vector<SegmentedGroup*> SegmentedGroup::getNeighbors() {
  std::vector<SegmentedGroup*> neighbors;
  auto neighbors_pair = getNeighborGroups();

  std::transform(
      neighbors_pair.begin(),
      neighbors_pair.end(),
      std::back_inserter(neighbors),
      [](auto& neighbor_group) { return neighbor_group.group; });
  return neighbors;
}

std::vector<SegmentedGroup::NeighborGroup> SegmentedGroup::
    getMergeCandidates() {
  // Don't look for candidates if already merged
  if (merged_) {
    return {};
  }

  std::vector<NeighborGroup> neighbors = getNeighborGroups();

  // Can this node be merged with another? Check if neighbors are merged, if
  // so and merged neighbor is within 1 level or node merged with neighbor is
  // within 1 level, can't merge this node with anything else.
  bool can_merge_this = true;
  for (auto& neighbor : neighbors) {
    if (!neighbor.group->merged_) {
      continue;
    }
    if (std::abs(neighbor.group->level_ - level_) <= 1) {
      can_merge_this = false;
    }
    if (std::abs(neighbor.group->merge_with_->level_ - level_) <= 1) {
      can_merge_this = false;
    }
  }
  if (!can_merge_this) {
    return {};
  }

  std::vector<bool> can_merge(true, neighbors.size());

  // Find neighbors with a level that is only 1 differant than this groups level
  for (size_t i = 0; i < neighbors.size(); i++) {
    if (std::abs(neighbors[i].group->level_ - level_) > 1) {
      can_merge[i] = false;
    }
  }

  // Check neighbor of neighbors we're considering, if any of them are merged
  // with another node, make sure the resulting edge wouldn't have a level
  // difference of 1
  for (size_t i = 0; i < neighbors.size(); i++) {
    if (!can_merge[i]) {
      continue;
    }

    for (auto neighbor_neighbor : neighbors[i].group->getNeighbors()) {
      // Don't check self
      if (neighbor_neighbor == neighbors[i].group) {
        continue;
      }
      if (neighbor_neighbor->merged_) {
        // check neighbor_neighbor level
        if (std::abs(neighbor_neighbor->level_ - level_) <= 1) {
          can_merge[i] = false;
        }
        if (std::abs(neighbor_neighbor->level_ - neighbors[i].group->level_) <=
            1) {
          can_merge[i] = false;
        }

        // check neighbor_neighber->merged_->level_
        if (std::abs(neighbor_neighbor->merge_with_->level_ - level_) <= 1) {
          can_merge[i] = false;
        }
        if (std::abs(
                neighbor_neighbor->merge_with_->level_ -
                neighbors[i].group->level_) <= 1) {
          can_merge[i] = false;
        }
      }
    }
  }

  std::vector<NeighborGroup> merge_candidates;
  for (size_t i = 0; i < neighbors.size(); i++) {
    if (can_merge[i]) {
      merge_candidates.push_back(neighbors[i]);
    }
  }
  return merge_candidates;
}

void SegmentedGroup::clearTraversalInfo() {
  level_ = -1;
  visited_ = false;
  merge_with_ = nullptr;
  merge_through_ = nullptr;
  merged_ = false;
}

std::vector<Val*> SegmentedGroup::edgesToVals(
    const std::vector<SegmentedEdge*>& se_v) {
  std::vector<Val*> ret_v;
  ret_v.reserve(se_v.size());

  std::transform(
      se_v.cbegin(),
      se_v.cend(),
      std::back_inserter(ret_v),
      [](SegmentedEdge* se) { return se->val; });
  return ret_v;
}

template <typename PREDICATE>
void insertUniquePredicated(
    std::vector<Val*>& v,
    const std::vector<SegmentedEdge*>& e,
    PREDICATE pred) {
  std::unordered_set<Val*> to_add;
  std::transform(
      e.cbegin(),
      e.cend(),
      std::inserter(to_add, to_add.end()),
      [](SegmentedEdge* se) { return se->val; });
  std::copy_if(
      to_add.begin(), to_add.end(), std::back_inserter(v), [pred](Val* val) {
        return pred(val);
      });
}

void SegmentedGroup::finalize() {
  // Move all the edges to group input/output
  // Inputs
  insertUniquePredicated(
      input_vals, producer_edges, [](Val* v) { return !v->isFusionInput(); });

  std::unordered_set<Val*> input_set(input_vals.begin(), input_vals.end());

  for (auto expr : exprs_) {
    for (auto i : expr->inputs()) {
      if (i->isAnInt() && i->definition() == nullptr && !i->isConstScalar() &&
          !i->isFusionInput() && !input_set.count(i)) {
        input_set.insert(i);
        input_vals.push_back(i);
      }
    }
  }

  // Outputs
  insertUniquePredicated(
      output_vals, consumer_edges, [](Val* v) { return !v->isFusionOutput(); });
}

std::ostream& operator<<(std::ostream& os, const SegmentedGroup* group) {
  os << "g{";
  for (size_t i = 0; i < group->exprs().size(); i++) {
    os << group->exprs()[i]->name();
    if (i + 1 != group->exprs().size())
      os << ", ";
  }
  os << "}\n";
  return os;
}

void SegmentedGroup::print() const {
  std::cout << this << "\n";
}

std::string toString(const SegmentedGroup* group) {
  std::stringstream ss;
  ss << group;
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, const SegmentedEdge* edge) {
  os << "e{ " << edge->from << " -> " << edge->to << "(";
  IrPrinter irp(os);
  irp.handle(edge->val);
  os << ") }\n";
  return os;
}

void SegmentedEdge::print() const {
  std::cout << this << "\n";
}

std::string toString(const SegmentedEdge* edge) {
  std::stringstream ss;
  ss << edge;
  return ss.str();
}

SegmentedFusion::SegmentedFusion(const Fusion* fusion)
    : fusion_(*fusion), impl_(this) {
  segmented_fusion_name_ = segmentedFusionName();
}

SegmentedGroup* SegmentedFusion::Impl::makeGroup() {
  groups_.emplace_back(std::make_unique<SegmentedGroup>());
  return groups_.back().get();
}

SegmentedGroup* SegmentedFusion::Impl::makeGroup(Expr* expr) {
  groups_.emplace_back(std::make_unique<SegmentedGroup>(expr));
  return groups_.back().get();
}

SegmentedEdge* SegmentedFusion::Impl::makeEdge(
    SegmentedGroup* from,
    SegmentedGroup* to,
    Val* val) {
  edges_.emplace_back(std::make_unique<SegmentedEdge>(from, to, val));
  return edges_.back().get();
}

void SegmentedFusion::Impl::cleanUnused() {
  std::unordered_set<SegmentedGroup*> g_used(
      owning_fusion_->groups().begin(), owning_fusion_->groups().end());
  std::unordered_set<SegmentedEdge*> e_used(
      owning_fusion_->edges().begin(), owning_fusion_->edges().end());

  groups_.erase(
      std::remove_if(
          groups_.begin(),
          groups_.end(),
          [&g_used](auto& g) { return g_used.count(g.get()) == 0; }),
      groups_.end());

  edges_.erase(
      std::remove_if(
          edges_.begin(),
          edges_.end(),
          [&e_used](auto& e) { return e_used.count(e.get()) == 0; }),
      edges_.end());
}

SegmentedGroup* SegmentedFusion::newGroup() {
  SegmentedGroup* g = impl_.makeGroup();
  groups_.push_back(g);
  return g;
}

SegmentedGroup* SegmentedFusion::newGroup(Expr* expr) {
  SegmentedGroup* g = impl_.makeGroup(expr);
  groups_.push_back(g);
  return g;
}

SegmentedEdge* SegmentedFusion::newEdge(
    SegmentedGroup* from,
    SegmentedGroup* to,
    Val* val) {
  SegmentedEdge* e = impl_.makeEdge(from, to, val);
  edges_.push_back(e);
  return e;
}

void SegmentedFusion::finalize() {
  impl_.cleanUnused();
  for (auto g : groups_) {
    g->finalize();
  }
}

void SegmentedFusion::draw() {
  size_t group_index = 0;
  std::unordered_map<const Expr*, size_t> expr_color_map;

  for (auto group : groups()) {
    for (auto expr : group->exprs()) {
      if (ir_utils::isTVOp(expr)) {
        expr_color_map[expr] = group_index;
      }
    }
    group_index++;
  }

  std::stringstream sstream;
  sstream << "segmented_fusion" << segmented_fusion_name_ << ".dot";
  auto filename = sstream.str();

  IrGraphGenerator::print(
      &fusion_,
      filename.c_str(),
      IrGraphGenerator::DetailLevel::ComputeOnly,
      &expr_color_map);
}

namespace {

std::vector<Val*> uniqueValConcat(
    const std::vector<std::vector<Val*>>& val_vecs) {
  std::vector<Val*> unique_vals;
  std::unordered_set<Val*> added;
  for (const auto& vec : val_vecs) {
    for (auto val : vec) {
      if (added.find(val) == added.end()) {
        unique_vals.push_back(val);
        added.emplace(val);
      }
    }
  }
  return unique_vals;
}

// Concat's producer edges of sg1 and sg2, but removes any edges from/to sg1/sg2
std::vector<SegmentedEdge*> getMergedProducerEdges(
    const SegmentedGroup* sg1,
    const SegmentedGroup* sg2) {
  TORCH_INTERNAL_ASSERT(
      sg1 != nullptr && sg2 != nullptr,
      "This function doesn't handle trivial.");

  auto producer_edges = sg1->producer_edges;

  producer_edges.insert(
      producer_edges.end(),
      sg2->producer_edges.begin(),
      sg2->producer_edges.end());

  // Register producers into sg2
  std::unordered_set<Val*> sg2_vals;
  for (auto se : sg2->producer_edges) {
    sg2_vals.emplace(se->val);
  }

  producer_edges.erase(
      std::remove_if(
          producer_edges.begin(),
          producer_edges.end(),
          [&sg1, &sg2, &sg2_vals](SegmentedEdge* se) {
            // remove edges in between the groups and common uses
            return (se->to == sg1 && se->from == sg2) ||
                (se->to == sg2 && se->from == sg1) ||
                (se->to == sg1 && sg2_vals.count(se->val));
          }),
      producer_edges.end());

  // Remove Duplicate Edges

  return producer_edges;
}

// Concat's consumer edges of sg1 and sg2, but removes any edges from/to sg1/sg2
std::vector<SegmentedEdge*> getMergedConsumerEdges(
    const SegmentedGroup* sg1,
    const SegmentedGroup* sg2) {
  TORCH_INTERNAL_ASSERT(
      sg1 != nullptr && sg2 != nullptr,
      "This function doesn't handle trivial.");

  auto consumer_edges = sg1->consumer_edges;
  consumer_edges.insert(
      consumer_edges.end(),
      sg2->consumer_edges.begin(),
      sg2->consumer_edges.end());

  consumer_edges.erase(
      std::remove_if(
          consumer_edges.begin(),
          consumer_edges.end(),
          [&sg1, &sg2](SegmentedEdge* se) {
            return (se->to == sg1 && se->from == sg2) ||
                (se->to == sg2 && se->from == sg1);
          }),
      consumer_edges.end());

  return consumer_edges;
}

// Returns a determinstic, unique set of inputs of the segment group, sg1, or
// the combined group sg1 + sg2
std::vector<Val*> getAllInputs(
    const SegmentedGroup* sg1,
    const SegmentedGroup* sg2 = nullptr) {
  std::vector<SegmentedEdge*> merged_producer_edges;

  if (sg1 != nullptr && sg2 != nullptr) {
    merged_producer_edges = getMergedProducerEdges(sg1, sg2);
  } else if (sg1 != nullptr) {
    merged_producer_edges = sg1->producer_edges;
  } else if (sg2 != nullptr) {
    merged_producer_edges = sg2->producer_edges;
  }

  std::vector<Val*> producer_edge_vals;

  std::transform(
      merged_producer_edges.begin(),
      merged_producer_edges.end(),
      std::back_inserter(producer_edge_vals),
      [](SegmentedEdge* se) { return se->val; });

  return uniqueValConcat(
      {sg1 == nullptr ? std::vector<Val*>() : sg1->input_vals,
       sg2 == nullptr ? std::vector<Val*>() : sg2->input_vals,
       producer_edge_vals});
}

// Returns a determinstic, unique set of outputs of the segment group, sg1, or
// the combined group sg1 + sg2
std::vector<Val*> getAllOutputs(
    const SegmentedGroup* sg1,
    const SegmentedGroup* sg2 = nullptr) {
  std::vector<SegmentedEdge*> merged_consumer_edges;

  if (sg1 != nullptr && sg2 != nullptr) {
    merged_consumer_edges = getMergedConsumerEdges(sg1, sg2);
  } else if (sg1 != nullptr) {
    merged_consumer_edges = sg1->consumer_edges;
  } else if (sg2 != nullptr) {
    merged_consumer_edges = sg2->consumer_edges;
  }

  std::vector<Val*> consumer_edge_vals;

  std::transform(
      merged_consumer_edges.begin(),
      merged_consumer_edges.end(),
      std::back_inserter(consumer_edge_vals),
      [](SegmentedEdge* se) { return se->val; });

  auto output_vals = uniqueValConcat(
      {sg1 == nullptr ? std::vector<Val*>() : sg1->output_vals,
       sg2 == nullptr ? std::vector<Val*>() : sg2->output_vals,
       consumer_edge_vals});

  return output_vals;
}

// Set version of getting merged input or output if segmented_groups were
//  merged
//  outputs respects order in segmented_groups for deterministic
//  merge trace
//  will get input if get_inputs otherwise will get ouputs
//  TODO: merge with the binary counter parts
std::vector<Val*> allInputsIfTrueElseOutputs(
    const std::vector<SegmentedGroup*>& segmented_groups,
    bool get_inputs = true) {
  // Helper to distinguish if we are getting inputs or outputs
  using EdgeVec = std::vector<SegmentedEdge*>;
  using ValVec = std::vector<Val*>;

  // Get producer edges to get inputs, consumer edges to get outputs
  auto edges_to_process_from_or_to_group =
      [get_inputs](SegmentedGroup* group) -> EdgeVec& {
    return get_inputs ? group->producer_edges : group->consumer_edges;
  };

  // Get the group that is connected to current group
  auto global_vals_from_or_to_group =
      [get_inputs](SegmentedGroup* group) -> ValVec& {
    return get_inputs ? group->input_vals : group->output_vals;
  };

  // Get the group that is connected to current group by given edge
  auto opposite_end_of_edge = [get_inputs](SegmentedEdge* edge) {
    return get_inputs ? edge->from : edge->to;
  };

  // Keep track of value and order to ensure deterministic result
  std::vector<Val*> merged_vals;
  std::unordered_set<Val*> merged_vals_set;

  // Put groups in a set for quick look up
  std::unordered_set<SegmentedGroup*> segmented_groups_set(
      segmented_groups.begin(), segmented_groups.end());

  // Collect vals associated with edges
  for (auto group : segmented_groups) {
    for (auto edge : edges_to_process_from_or_to_group(group)) {
      if (
          // Need to de-duplicate values so we don't get multiple of any input
          !merged_vals_set.count(edge->val) &&
          // One side of this edge will be `group`, if the other end is
          //  also in segmented_groups, then this is an internal edge
          //  that we don't want.
          !segmented_groups_set.count(opposite_end_of_edge(edge))) {
        merged_vals.push_back(edge->val);
        merged_vals_set.insert(edge->val);
      }
    }
  }

  // Collect original fusion's inputs/outputs and append at the end
  for (auto group : segmented_groups) {
    for (auto global_val : global_vals_from_or_to_group(group)) {
      // de-duplicate
      if (!merged_vals_set.count(global_val)) {
        merged_vals.push_back(global_val);
        merged_vals_set.insert(global_val);
      }
    }
  }

  return merged_vals;
}

// Utility function to list all expressions in a group
void detailGroupPrint(std::ostream& os, const SegmentedGroup* group) {
  IrPrinter irp(os);
  os << "g{"
     << "(" << toString(group->heuristic()) << ")\n";
  os << "inputs: \n";
  for (auto i : getAllInputs(group)) {
    i->print();
  }
  os << "outputs: \n";
  for (auto o : getAllOutputs(group)) {
    o->print();
  }

  os << "\n\n";

  for (size_t i = 0; i < group->exprs().size(); i++) {
    irp.handle(group->exprs()[i]);
    if (i + 1 != group->exprs().size())
      os << " , ";
  }
  os << "}\n\n";
}

} // namespace

//! An utility class to compute and maintain the "producers of"
//!   relationship in a segmented graph. Space heavy and should
//!   avoid use on very large graphs.
//!
//!  Currently trying to move as far as possible with only a
//!   producer map, without transposing it to make a consumer map.
//!  Making it NonCopyable because we should never need to
//!   copy an instance of this class.
//!  TODO: Space efficiency of this class will be important,
//!        because we need it in the pre-merging of segmentedGroups,
//!        currently O(n^2). O(nlogn) would be a reasonable
//!        goal to achieve.
class GroupDependencyAnalysis : public NonCopyable, public SegmenterAnalysis {
  using GroupSet = std::unordered_set<SegmentedGroup*>;
  using GroupSetOwningPtr = std::unique_ptr<GroupSet>;
  using DependencyMap = std::unordered_map<SegmentedGroup*, GroupSetOwningPtr>;

 public:
  //! Populate producers of all groups in segmented fusion
  explicit GroupDependencyAnalysis(SegmentedFusion* segmented_fusion)
      : segmented_fusion_(segmented_fusion) {
    computeAllProducers();
  }

  //! Checks if group is consumer of any group in groups_to_check
  //!  TODO: refactor this similar to isConsumerOf
  bool isConsumerOfAny(
      SegmentedGroup* group,
      const std::vector<SegmentedGroup*>& groups_to_check) {
    auto& producers_of_group = getAllKnownProducersSet(group);
    for (const auto& potential_producer : groups_to_check) {
      if (producers_of_group->count(potential_producer)) {
        return true;
      }
    }
    return false;
  }

  bool isConsumerOf(SegmentedGroup* a, SegmentedGroup* b) {
    return known_producers_of_.at(a)->count(b);
  }

  bool isProducerOf(SegmentedGroup* a, SegmentedGroup* b) {
    return known_producers_of_.at(b)->count(a);
  }

  //! Finds the common producers of given set of groups
  GroupSet getCommonProducersOf(std::vector<SegmentedGroup*> groups);

  //! Update the map when the given two groups have been merged to create `ab`
  //! this method is for book keeping and query only, doesn't implicitly check
  //!  for DAG
  void mergeGroups(SegmentedGroup* a, SegmentedGroup* b, SegmentedGroup* ab);

  //! Update the map when the given two groups have been merged to create
  //! `merged` this method is for book keeping and query only, doesn't
  //! implicitly check
  //!  for DAG
  void mergeGroups(const GroupSet& groups, SegmentedGroup* merged);

  //! Populate all values that is on a path from producer to consumer
  //!  efficiency can be important here. (TODO)
  GroupSet valuesBetween(SegmentedGroup* producer, SegmentedGroup* consumer) {
    if (producer == consumer) {
      return {};
    }

    GroupSet values_between;
    auto& all_producers_of_consumer = known_producers_of_.at(consumer);
    TORCH_INTERNAL_ASSERT(
        all_producers_of_consumer->count(producer),
        "Fusion segment: Trying to compute path between two nodes that are not producer-consumer pairs");

    std::copy_if(
        all_producers_of_consumer->begin(),
        all_producers_of_consumer->end(),
        std::inserter(values_between, values_between.end()),
        [this, producer](SegmentedGroup* producer_of_consumer) {
          // Checks if producer is on the producer path of this intermediate
          // node
          return known_producers_of_.at(producer_of_consumer)->count(producer);
        });

    return values_between;
  }

  //! Checks if the segmented fusion this class tracks is still a DAG
  //!  used for generating assertions after transforms
  bool isproducerMapDAG() const {
    for (auto& it : known_producers_of_) {
      if (it.second->count(it.first)) {
        return false;
      }
    }
    return true;
  }

 private:
  //! Collect initial producer info using
  //!  a work list algorithm through forward traversal
  //!  a backward DFS would do the same
  void computeAllProducers();

  //! Add all consumers of `producer` to `to_visit`
  void addConsumersToWorkList(SegmentedGroup* producer, GroupSet& to_visit) {
    for (auto e : producer->consumer_edges) {
      // A consumer wouldn't have been worked before any of its producer
      to_visit.insert(e->to);
    }
  }

  //! Propagate all known producers of `from` into `into`, used to keep track
  //! of:
  //!  1. `from` is a producer of `into`
  //!  2. `from` has been merged with other group to create `into`
  void mergeAllKnownProducersIntoFrom(
      SegmentedGroup* into,
      SegmentedGroup* from) {
    auto& producer_set_to_merge = *getAllKnownProducersSet(from);
    for (auto group : producer_set_to_merge) {
      getAllKnownProducersSet(into)->insert(group);
    }
  }

  //! Utility to access known producers of a group so far
  GroupSetOwningPtr& getAllKnownProducersSet(SegmentedGroup* group) {
    auto& producer_set_ptr = known_producers_of_[group];
    if (!producer_set_ptr) {
      producer_set_ptr = std::make_unique<GroupSet>();
    }
    return producer_set_ptr;
  }

  // utility to compute the set intersection of group sets a,b
  GroupSet groupSetIntersection(const GroupSet& a, const GroupSet& b) {
    bool a_is_smaller = a.size() < b.size();
    const auto& smaller_group_set = a_is_smaller ? a : b;
    const auto& bigger_group_set = a_is_smaller ? b : a;

    GroupSet intersection;
    for (auto group : smaller_group_set) {
      if (bigger_group_set.count(group)) {
        intersection.insert(group);
      }
    }
    return intersection;
  }

 private:
  SegmentedFusion* segmented_fusion_;
  DependencyMap known_producers_of_;
};

//! Finds the common producers of given set of groups
GroupDependencyAnalysis::GroupSet GroupDependencyAnalysis::getCommonProducersOf(
    std::vector<SegmentedGroup*> groups) {
  if (groups.empty()) {
    return {};
  }

  // Optimization: start with the smallest producer set
  std::sort(
      groups.begin(),
      groups.end(),
      [this](SegmentedGroup* a, SegmentedGroup* b) {
        return known_producers_of_.at(a)->size() <
            known_producers_of_.at(b)->size();
      });

  // Get intersection of producers
  GroupSet common_producers = *(known_producers_of_.at(groups[0]));
  for (size_t i = 1; i < groups.size(); i++) {
    common_producers = groupSetIntersection(
        common_producers, *(known_producers_of_.at(groups[i])));
  }

  return common_producers;
}

//! Update the map when the given two groups have been merged to create `ab`
//! this method is for book keeping and query only, doesn't implicitly check
//!  for DAG
void GroupDependencyAnalysis::mergeGroups(
    SegmentedGroup* a,
    SegmentedGroup* b,
    SegmentedGroup* ab) {
  // Access/Create the producer set of ab
  auto& ab_set = getAllKnownProducersSet(ab);

  // propagate a's and b's known producers into ab
  mergeAllKnownProducersIntoFrom(ab, a);
  mergeAllKnownProducersIntoFrom(ab, b);

  // a, b are now merged, so no longer exist
  ab_set->erase(a);
  ab_set->erase(b);

  // a, b no longer exist, remove their producer sets
  known_producers_of_.erase(a);
  known_producers_of_.erase(b);

  // update producer maps of other groups
  for (auto& it : known_producers_of_) {
    // for all groups that are produced by either a or b
    if (it.second->count(a) || it.second->count(b)) {
      // insert ab as the new producer
      it.second->insert(ab);
      // all producers of both a and b are now producers of `it`
      mergeAllKnownProducersIntoFrom(it.first, ab);
    }
    // a, b no longer exist, remove them from `it`
    it.second->erase(a);
    it.second->erase(b);
  }
}

//! Update the map when the given two groups have been merged to create
//! `merged` this method is for book keeping and query only, doesn't
//! implicitly check
//!  for DAG
void GroupDependencyAnalysis::mergeGroups(
    const GroupSet& groups,
    SegmentedGroup* merged) {
  // Access/Create the producer set of merged
  auto& merged_set = getAllKnownProducersSet(merged);

  // Populate all producers of groups and
  //  write into producer map of merged
  std::for_each(
      groups.begin(), groups.end(), [this, merged](SegmentedGroup* group) {
        mergeAllKnownProducersIntoFrom(merged, group);
      });

  // Erase all groups that was merged from producer map
  std::for_each(
      groups.begin(), groups.end(), [this, &merged_set](SegmentedGroup* group) {
        // erase inter dependencies
        merged_set->erase(group);
        // erase producer map tracking merged entires
        known_producers_of_.erase(group);
      });

  // Update producer relationships with other groups in producer map
  for (auto& it : known_producers_of_) {
    auto producer_intersection = groupSetIntersection(*(it.second), groups);
    // if current node has any producer that was merged
    if (producer_intersection.size() > 0) {
      for (auto merged_producer : producer_intersection) {
        // delete all disappearing producers
        it.second->erase(merged_producer);
      }
      // insert the new group as producer
      it.second->insert(merged);
    }
  }
}

//! Collect initial producer info using
//!  a work list algorithm through forward traversal
//!  a backward DFS would do the same
void GroupDependencyAnalysis::computeAllProducers() {
  GroupSet visited;
  GroupSet to_visit;

  // Collect source nodes, with no producers we are guaranteed
  //  a source node on a DAG
  std::copy_if(
      segmented_fusion_->groups().begin(),
      segmented_fusion_->groups().end(),
      std::inserter(visited, visited.end()),
      [](SegmentedGroup* group) { return group->producer_edges.empty(); });

  // visited now only contain source nodes
  //  they can go backward to nowhere
  for (auto group : visited) {
    addConsumersToWorkList(group, to_visit);
  }

  while (!to_visit.empty()) {
    SegmentedGroup* to_update = nullptr;
    for (auto visiting_group : to_visit) {
      if (std::all_of(
              visiting_group->producer_edges.begin(),
              visiting_group->producer_edges.end(),
              [&visited](SegmentedEdge* e) {
                return visited.count(e->from);
              })) {
        // filter multi-edges
        GroupSet producers_of_visiting_group;
        for (auto edge : visiting_group->producer_edges) {
          producers_of_visiting_group.insert(edge->from);
        }

        // populate all possible paths
        // from producer backward, including
        // the producer
        for (auto producer : producers_of_visiting_group) {
          getAllKnownProducersSet(visiting_group)->insert(producer);
          mergeAllKnownProducersIntoFrom(visiting_group, producer);
        }
        to_update = visiting_group;
        break;
      }
    }
    if (to_update) {
      addConsumersToWorkList(to_update, to_visit);
      to_visit.erase(to_update);
      visited.insert(to_update);
    } else {
      TORCH_INTERNAL_ASSERT(false, "unreachable, original graph not a DAG");
    }
  }
}

std::ostream& operator<<(
    std::ostream& os,
    const SegmentedFusion* segmented_fusion) {
  os << "Segmented_Fusion{ \n";
  os << "groups: \n";
  for (const auto g : segmented_fusion->cgroups()) {
    os << g << "\n";
  }
  os << "edges: \n";
  for (const auto e : segmented_fusion->cedges()) {
    os << e << "\n";
  }
  os << "group details:\n\n";
  for (const auto g : segmented_fusion->cgroups()) {
    detailGroupPrint(os, g);
  }
  os << "} //Segmented_Fusion\n";
  return os;
}

void SegmentedFusion::print() const {
  std::cout << this << "\n";
}

std::string toString(SegmentedFusion* segmented_fusion) {
  std::stringstream ss;
  ss << segmented_fusion;
  return ss.str();
}

std::unique_ptr<Fusion> SegmentedFusion::makeFusion(SegmentedGroup* sg) {
  std::unique_ptr<Fusion> fusion_segment = std::make_unique<Fusion>();

  auto complete_to_segment_map = Fusion::copy(&fusion_, fusion_segment.get());

  std::vector<Val*> input_list(
      fusion_segment->inputs().begin(), fusion_segment->inputs().end());
  for (auto inp : input_list) {
    fusion_segment->removeInput(inp);
  }

  std::vector<Val*> output_list(
      fusion_segment->outputs().begin(), fusion_segment->outputs().end());
  for (auto out : output_list) {
    fusion_segment->removeOutput(out);
  }

  for (auto inp : getAllInputs(sg)) {
    fusion_segment->addInput(complete_to_segment_map.clone(inp));
  }

  for (auto out : getAllOutputs(sg)) {
    fusion_segment->addOutput(complete_to_segment_map.clone(out));
  }

  return fusion_segment;
}

void SegmentCandidateFinder::resetTraversal() {
  for (auto group : groups()) {
    // Start traversal at input groups
    if (group->producer_edges.empty()) {
      to_visit_.push_back(group);
    }
    group->visited_ = false;
    group->level_ = 0;
  }
}

void SegmentCandidateFinder::resetLevels() {
  while (!to_visit_.empty()) {
    auto visit = to_visit_.front();
    to_visit_.pop_front();

    // All inputs processed?
    bool ready = true;
    if (!visit->producer_edges.empty()) {
      ready = std::all_of(
          visit->producer_edges.begin(),
          visit->producer_edges.end(),
          [&](SegmentedEdge* dep) { return dep->from->visited_; });
    }

    if (!ready) {
      // In case traversal doesn't complete because there's an error in the
      // DAG topology.
      next_to_visit_.push_back(visit);
      continue;
    }

    visit->visited_ = true;

    to_visit_.insert(
        to_visit_.end(), next_to_visit_.begin(), next_to_visit_.end());
    next_to_visit_.clear();

    for (auto out : visit->consumer_edges) {
      to_visit_.push_back(out->to);
    }

    visit->level_ = 0;
    for (auto inp : visit->producer_edges) {
      visit->level_ = std::max(visit->level_, inp->from->level_ + 1);
    }
  }
  TORCH_INTERNAL_ASSERT(
      next_to_visit_.empty(), "Error in graph, is not a DAG.");
}

// Disconect group from neighbors, and return edges that were disconnected
std::unordered_set<SegmentedEdge*> SegmentCandidateFinder::disconnectGroup(
    SegmentedGroup* group) {
  std::unordered_set<SegmentedEdge*> removed_edges(
      group->producer_edges.begin(), group->producer_edges.end());

  for (auto edge : group->producer_edges) {
    auto from = edge->from;
    auto& from_edges = from->consumer_edges;
    auto from_edge_it = std::find(from_edges.begin(), from_edges.end(), edge);
    TORCH_INTERNAL_ASSERT(
        from_edge_it != from_edges.end(), "Could not find edge to remove.");
    from_edges.erase(from_edge_it);
  }

  for (auto edge : group->consumer_edges) {
    removed_edges.insert(edge);
    auto to = edge->to;
    auto& to_edges = to->producer_edges;
    auto to_edge_it = std::find(to_edges.begin(), to_edges.end(), edge);
    TORCH_INTERNAL_ASSERT(
        to_edge_it != to_edges.end(), "Could not find edge to remove.");
    to_edges.erase(to_edge_it);
  }

  group->producer_edges.clear();
  group->consumer_edges.clear();

  return removed_edges;
}

void SegmentCandidateFinder::eraseGroups(
    std::unordered_set<SegmentedGroup*>& groups_to_erase) {
  std::unordered_set<SegmentedEdge*> edges_to_erase;
  for (auto group : groups_to_erase) {
    auto disconnected_edges = disconnectGroup(group);
    edges_to_erase.insert(disconnected_edges.begin(), disconnected_edges.end());
  }

  edges().erase(
      std::remove_if(
          edges().begin(),
          edges().end(),
          [&edges_to_erase](SegmentedEdge* edge) {
            if (edges_to_erase.find(edge) != edges_to_erase.end()) {
              return true;
            };
            return false;
          }),
      edges().end());

  groups().erase(
      std::remove_if(
          groups().begin(),
          groups().end(),
          [&groups_to_erase](SegmentedGroup* group) {
            if (groups_to_erase.find(group) != groups_to_erase.end()) {
              return true;
            };
            return false;
          }),
      groups().end());
}

SegmentedGroup* SegmentCandidateFinder::mergeNodes() {
  SegmentedGroup* last_merged = nullptr;
  auto it = to_merge_.begin();
  TORCH_INTERNAL_ASSERT(to_merge_.size() % 2 == 0);
  while (it != to_merge_.end()) {
    auto group1 = *it++;
    auto group2 = *it++;

    clean_up_groups_.emplace(group1);
    clean_up_groups_.emplace(group2);

    // Make the new joined node
    auto joined_group = segmented_fusion_->newGroup();

    joined_group->input_vals =
        uniqueValConcat({group1->input_vals, group2->input_vals});

    joined_group->output_vals =
        uniqueValConcat({group1->output_vals, group2->output_vals});

    joined_group->exprs_ = group1->exprs_;
    joined_group->exprs_.insert(
        joined_group->exprs_.end(),
        group2->exprs_.begin(),
        group2->exprs_.end());

    auto producer_edges = getMergedProducerEdges(group1, group2);
    // Connect joined group to resulting neighbors
    for (auto edge : producer_edges) {
      auto from = edge->from;
      auto val = edge->val;

      auto new_edge = segmented_fusion_->newEdge(from, joined_group, val);
      joined_group->producer_edges.push_back(new_edge);
      from->consumer_edges.push_back(new_edge);
    }

    auto consumer_edges = getMergedConsumerEdges(group1, group2);

    for (auto edge : consumer_edges) {
      auto to = edge->to;
      auto val = edge->val;

      auto new_edge = segmented_fusion_->newEdge(joined_group, to, val);
      joined_group->consumer_edges.push_back(new_edge);
      edge->to->producer_edges.push_back(new_edge);
    }

    joined_group->setHeuristic(deriveHeuristic(joined_group));
    // Need to maintain the group dependency data if it has been intialized
    //  by previous merging
    if (group_dependency_) {
      group_dependency_->as<GroupDependencyAnalysis>()->mergeGroups(
          group1, group2, joined_group);
    }
    last_merged = joined_group;
  }

  to_merge_.clear();
  for (auto group : clean_up_groups_) {
    auto disconnected_edges = disconnectGroup(group);
    clean_up_edges_.insert(
        disconnected_edges.begin(), disconnected_edges.end());
  }

  edges().erase(
      std::remove_if(
          edges().begin(),
          edges().end(),
          [this](SegmentedEdge* edge) {
            if (this->clean_up_edges_.find(edge) !=
                this->clean_up_edges_.end()) {
              return true;
            };
            return false;
          }),
      edges().end());

  groups().erase(
      std::remove_if(
          groups().begin(),
          groups().end(),
          [this](SegmentedGroup* group) {
            if (this->clean_up_groups_.find(group) !=
                this->clean_up_groups_.end()) {
              return true;
            };
            return false;
          }),
      groups().end());

  clean_up_edges_.clear();
  clean_up_groups_.clear();

  return last_merged;
}

// Logic largely parallels mergeNodes, but they are used
//  in different phases of segmentation. Should consider
//  a clean up and share the implementations.
SegmentedGroup* SegmentCandidateFinder::mergeAllGivenGroups(
    const std::vector<SegmentedGroup*>& groups_to_merge) {
  TORCH_INTERNAL_ASSERT(
      !groups_to_merge.empty(),
      "fusion segment :(mergeAllGivenGroups) tried to merge no groups")

  // Make a set to detect internal edges
  std::unordered_set<SegmentedGroup*> group_set(
      groups_to_merge.begin(), groups_to_merge.end());

  // Sets to de-duplicate multiple uses of
  //  input/edge values and re-computations of exprs
  std::unordered_set<Val*> used_edge_vals_set;
  std::unordered_set<Val*> used_input_vals_set;
  std::unordered_set<Expr*> exprs_set;

  // Create new group
  auto joined_group = segmented_fusion_->newGroup();

  // Populate edges, exprs, global vals
  //  from each of the groups
  for (auto group : groups_to_merge) {
    // Populate complete fusion inputs to the group
    for (auto input_val : group->input_vals) {
      if (!used_input_vals_set.count(input_val)) {
        used_input_vals_set.insert(input_val);
        joined_group->input_vals.push_back(input_val);
      }
    }

    // Populate complete fusion outputs from the group
    for (auto output_val : group->output_vals) {
      joined_group->output_vals.push_back(output_val);
    }

    // Populate producer edges to the group
    for (auto edge : group->producer_edges) {
      if (
          // Check this is not internal edge
          !group_set.count(edge->from) &&
          // Check this val has been added or not
          !used_edge_vals_set.count(edge->val)) {
        used_edge_vals_set.insert(edge->val);
        auto new_producer_edge =
            segmented_fusion_->newEdge(edge->from, joined_group, edge->val);
        joined_group->producer_edges.push_back(new_producer_edge);
        edge->from->consumer_edges.push_back(new_producer_edge);
      }
    }

    // Populate consumer edges from the group
    for (auto edge : group->consumer_edges) {
      if (
          // Check this is not internal edge
          !group_set.count(edge->to)) {
        auto new_consumer_edge =
            segmented_fusion_->newEdge(joined_group, edge->to, edge->val);
        joined_group->consumer_edges.push_back(new_consumer_edge);
        edge->to->producer_edges.push_back(new_consumer_edge);
      }
    }

    // Populate exprs
    for (auto expr : group->exprs_) {
      if (!exprs_set.count(expr)) {
        joined_group->exprs_.push_back(expr);
        exprs_set.insert(expr);
      }
    }
  }

  // Clean up original groups from segmented fusion
  for (auto group : groups_to_merge) {
    auto disconnected_edges = disconnectGroup(group);
    clean_up_edges_.insert(
        disconnected_edges.begin(), disconnected_edges.end());
  }

  edges().erase(
      std::remove_if(
          edges().begin(),
          edges().end(),
          [this](SegmentedEdge* edge) { return clean_up_edges_.count(edge); }),
      edges().end());

  groups().erase(
      std::remove_if(
          groups().begin(),
          groups().end(),
          [&group_set](SegmentedGroup* group) -> bool {
            return group_set.count(group);
          }),
      groups().end());

  clean_up_edges_.clear();

  joined_group->setHeuristic(deriveHeuristic(joined_group));
  return joined_group;
}

namespace {

// Guard to temporarily change the inputs and outputs of a fusion. On
// destruction will return fusion to original state.
// Not used temporarily but will be useful when adding more mergin heuristics
class FusionSegmentGuard : public NonCopyable {
 public:
  FusionSegmentGuard() = delete;

  FusionSegmentGuard(
      Fusion* fusion,
      std::vector<Val*> inputs,
      std::vector<Val*> outputs)
      : fusion_(fusion),
        old_inputs_(fusion->inputs()),
        old_outputs_(fusion->outputs()),
        new_inputs_(std::move(inputs)),
        new_outputs_(std::move(outputs)) {
    TORCH_INTERNAL_ASSERT(fusion_ != nullptr);
    for (auto old_inp : old_inputs_) {
      fusion_->removeInput(old_inp);
    }

    for (auto old_out : old_outputs_) {
      fusion_->removeOutput(old_out);
    }

    for (auto new_inp : new_inputs_) {
      fusion_->addInput(new_inp);
    }

    for (auto new_out : new_outputs_) {
      fusion_->addOutput(new_out);
    }
  }

  ~FusionSegmentGuard() {
    if (fusion_ == nullptr) {
      return;
    }
    for (auto new_inp : new_inputs_) {
      fusion_->removeInput(new_inp);
    }

    for (auto new_out : new_outputs_) {
      fusion_->removeOutput(new_out);
    }

    for (auto old_inp : old_inputs_) {
      fusion_->addInput(old_inp);
    }

    for (auto old_out : old_outputs_) {
      fusion_->addOutput(old_out);
    }
  }

 private:
  Fusion* const fusion_ = nullptr;
  const std::vector<Val*> old_inputs_;
  const std::vector<Val*> old_outputs_;
  const std::vector<Val*> new_inputs_;
  const std::vector<Val*> new_outputs_;
};

c10::optional<ScheduleHeuristic> tryMerge(
    Fusion* fusion,
    SegmentedGroup* a,
    SegmentedGroup* b = nullptr) {
  FusionSegmentGuard fsg(fusion, getAllInputs(a, b), getAllOutputs(a, b));

  return SchedulerEntry::proposeHeuristics(fusion);
}

c10::optional<ScheduleHeuristic> tryMerge(
    Fusion* fusion,
    const std::vector<SegmentedGroup*>& segmented_groups) {
  FusionSegmentGuard fsg(
      fusion,
      allInputsIfTrueElseOutputs(segmented_groups, true),
      allInputsIfTrueElseOutputs(segmented_groups, false));
  return SchedulerEntry::proposeHeuristics(fusion);
}

// This function is for cleanup and
//  easier debugging. It shouldn't affect functionality
//  since segmented fusions are compiled with fusion
//  guard on the edges instead of actually looking
//  at the exprs.
void deDuplicateScalarExprs(std::vector<Expr*>& exprs) {
  // Exprs in SegmentedGroup are not ordered
  // so it is ok to insert them from unordered
  // set
  std::unordered_set<Expr*> scalar_expr_set;

  std::copy_if(
      exprs.begin(),
      exprs.end(),
      std::inserter(scalar_expr_set, scalar_expr_set.end()),
      [](Expr* expr) { return ir_utils::isScalarOp(expr); });

  if (!scalar_expr_set.empty()) {
    exprs.erase(
        std::remove_if(
            exprs.begin(),
            exprs.end(),
            [&scalar_expr_set](Expr* expr) {
              return scalar_expr_set.count(expr);
            }),
        exprs.end());
    exprs.insert(exprs.end(), scalar_expr_set.begin(), scalar_expr_set.end());
  }
}

// Helper function to get a reduction operation from group
ReductionOp* firstReductionFromGroup(SegmentedGroup* group) {
  for (auto expr : group->exprs()) {
    if (auto rop = dynamic_cast<ReductionOp*>(expr)) {
      return rop;
    }
  }
  return nullptr;
}

} // namespace

// Custom merge node passes:
//  These passes are added at the beginning or the end of
//  the node merging process to direct the heuristics of
//  node merging process
//
//  Should consider generalization and make a proper interface
//   if we have more merge node heuristics like this

//! CombineReductions:
//!  This pass works before the main merge node process
//!    It identifies reduction operations that can be combined
//!    together to form a normalization kernel.
//!  Two reductions are considered the same type if they have
//!   the same root domain length, and the reduction axis are the same.
//!   This pass tries to merge nodes with the same reduction type based
//!   on the graph structure.
class CombineReductions {
  using GroupSet = std::unordered_set<SegmentedGroup*>;
  using GroupVec = std::vector<SegmentedGroup*>;
  struct ReductionSignature;

 public:
  static void run(SegmentCandidateFinder* segment_candidate_finder) {
    CombineReductions combine_reductions(segment_candidate_finder);
  }
  static bool shouldRun(SegmentCandidateFinder* segment_candidate_finder);

 private:
  CombineReductions(SegmentCandidateFinder* segment_candidate_finder)
      : segment_candidate_finder_(segment_candidate_finder) {
    // Run pass over the segments

    // Collect segmented groups with reductions in them,
    //  Assuming running before any merge happened, so
    //  should see exactly one non-trivial reduction in each group
    for (auto group : segment_candidate_finder_->groups()) {
      ReductionOp* rop = nullptr;
      for (auto expr : group->exprs()) {
        if (auto rop_in_group = dynamic_cast<ReductionOp*>(expr)) {
          auto rop_signature =
              std::make_unique<ReductionSignature>(rop_in_group);
          // Ignore pure squeeze operations in this analysis
          if (!rop_signature->has_nontrivial_reduction) {
            continue;
          }
          // We should have only one nontrivial reduction in each group since no
          // merging
          //  has happened yet
          TORCH_INTERNAL_ASSERT(
              rop == nullptr,
              "CombineReductions, two reductions found in group some incompatible transform happened before doing this pass");
          rop = rop_in_group;

          groups_with_reductions_.push_back(group);
          // Check if this reduction signature is one that we have seen before
          auto signature_match_it = std::find_if(
              known_reduction_signatures_.begin(),
              known_reduction_signatures_.end(),
              [&rop_signature](auto& know_signature) {
                return know_signature->sameAs(rop_signature.get());
              });
          // Unmatched: Create a new signature entry if not known
          if (signature_match_it == known_reduction_signatures_.end()) {
            group_reduction_signature_map_[group] = rop_signature.get();
            known_reduction_signatures_.emplace_back(std::move(rop_signature));
          } else {
            // Matched known signature: Mark that this groups belongs to know
            // signature
            group_reduction_signature_map_[group] = signature_match_it->get();
          }
        }
      }
    }

    // Keep trying to merge groups with compatible reductions and compatible
    // paths
    //  until no more merge opportunity can be identified
    bool merged_groups = true;
    while (merged_groups) {
      merged_groups = false;

      // Merge one pair of reduction groups at a time, and need
      //  the pass to update dependency info along the way to avoid cycles
      for (size_t first_group_index = 0;
           first_group_index < groups_with_reductions_.size();
           first_group_index++) {
        if (merged_groups) {
          // Need to break and re-enter this loop because
          // groups_with_reductions_ will be updated
          break;
        }

        // Select one of the group to merge and get its reduction signature
        auto first_group = groups_with_reductions_[first_group_index];
        auto first_group_signature =
            group_reduction_signature_map_.at(first_group);

        for (size_t second_group_index = first_group_index + 1;
             second_group_index < groups_with_reductions_.size();
             second_group_index++) {
          if (merged_groups) {
            // Need to break and re-enter this loop because
            // groups_with_reductions_ will be updated
            break;
          }
          auto second_group = groups_with_reductions_[second_group_index];
          auto second_group_signature =
              group_reduction_signature_map_.at(second_group);

          // Cannot merge if their signatures are not the same
          if (!first_group_signature->sameAs(second_group_signature)) {
            continue;
          }

          // first try a vertical merge
          merged_groups =
              verticalReductionMerge(first_group, second_group) != nullptr;
          if (!merged_groups) {
            // vertical merge didn't happen, try a horizontal merge
            merged_groups =
                horizontalReductionMerge(first_group, second_group) != nullptr;
          }
        }
      }
    }
  }

  //! Merge a vertical pair of producers and consumers,
  //!  the resulting group will include all nodes that are
  //!  also consumers of producer and producers of consumer,
  //!  i.e. values between the given producer-consumer pair.
  //!  Can be proven that:
  //!   1. Including all of these nodes will be cycle-free
  //!   2. These nodes are the minimal set of nodes to include if
  //!  for producer-consumer pair to be in the same group cycle-free
  //!
  //!  Returns nullptr if such merge cannot be achieved.
  //!  Reasons for not merging will include:
  //!   1. Given groups do not form producer-consumer pair
  //!   2. Merge will create cycle on the graph
  //!   3. The merged joined group cannot be scheduled
  SegmentedGroup* verticalReductionMerge(
      SegmentedGroup* first_group,
      SegmentedGroup* second_group) {
    // This is part of ReductionCombine pass, and we should only call this
    // function on a pair of
    //  reduction/normalization groups
    TORCH_INTERNAL_ASSERT(
        group_reduction_signature_map_.at(first_group)
            ->sameAs(group_reduction_signature_map_.at(second_group)));
    TORCH_INTERNAL_ASSERT(first_group != second_group);
    // Get the group dependency data from segment finder
    auto dependency_analysis = segment_candidate_finder_->getGroupDependency();

    // Check producer-consumer relationship
    SegmentedGroup* producer = nullptr;
    SegmentedGroup* consumer = nullptr;
    if (dependency_analysis->isConsumerOf(first_group, second_group)) {
      producer = second_group;
      consumer = first_group;
    } else if (dependency_analysis->isProducerOf(first_group, second_group)) {
      producer = first_group;
      consumer = second_group;
    } else {
      // Given groups aren't producer-consumer pair, won't merge
      return nullptr;
    }

    // Collect all groups that we need to merge along with the producer and
    // consumer
    auto all_groups_to_merge =
        getValidMinVerticalMergedGroupSet(producer, consumer);

    if (all_groups_to_merge.empty()) {
      // The vertical paths from producer to consumer have in-compatible
      // reductions
      //   so this vertical merge cannot be done.
      return nullptr;
    }

    // TODO: this step would not be deterministic, because valuesBetween isn't
    //       could fix this by a topological order
    std::vector<SegmentedGroup*> all_groups_to_merge_vec(
        all_groups_to_merge.begin(), all_groups_to_merge.end());

    // Final sanity check: the merged group can actually be scheduled
    Fusion* fusion =
        &segment_candidate_finder_->segmented_fusion_->completeFusion();
    if (!tryMerge(fusion, all_groups_to_merge_vec)) {
      return nullptr;
    }

    // Merge this group
    auto joined_group =
        segment_candidate_finder_->mergeAllGivenGroups(all_groups_to_merge_vec);

    // Update dependency analysis
    dependency_analysis->mergeGroups(all_groups_to_merge, joined_group);

    // Update the reduction groups that are merged
    groups_with_reductions_.push_back(joined_group);
    group_reduction_signature_map_[joined_group] =
        group_reduction_signature_map_.at(first_group);
    groups_with_reductions_.erase(
        std::remove_if(
            groups_with_reductions_.begin(),
            groups_with_reductions_.end(),
            [&all_groups_to_merge](SegmentedGroup* group) {
              return all_groups_to_merge.count(group);
            }),
        groups_with_reductions_.end());

    return joined_group;
  }

  //! Horizontal reduction merging:
  //!  merge two horizontal groups with reduction expressions to make a joined
  //!  normalization group. A pair of horizontal groups are ones that are not
  //!  a producer-consumer pair, and share either a common producer or a common
  //!  consumer.
  //!
  //!  TODO: This implementation looks at common producers only, since common
  //!  consumers
  //!          are not computed easily with current dependency analysis.
  SegmentedGroup* horizontalReductionMerge(
      SegmentedGroup* first_group,
      SegmentedGroup* second_group) {
    // This is part of ReductionCombine pass, and we should only call this
    // function on a pair of
    //  reduction/normalization groups
    TORCH_INTERNAL_ASSERT(
        group_reduction_signature_map_.at(first_group)
            ->sameAs(group_reduction_signature_map_.at(second_group)));
    TORCH_INTERNAL_ASSERT(first_group != second_group);

    auto dependency_analysis = segment_candidate_finder_->getGroupDependency();

    // Check that the two groups are not producer-consumer's
    if (dependency_analysis->isConsumerOf(first_group, second_group) ||
        dependency_analysis->isProducerOf(first_group, second_group)) {
      // This merge pass will not handle producer-consumer pairs
      return nullptr;
    }

    // Get common producers of the two group
    auto common_producers_set =
        dependency_analysis->getCommonProducersOf({first_group, second_group});
    if (common_producers_set.empty()) {
      // The given pair doesn't have a common producer.
      //  Either they have a common consumer, which we don't handle for now,
      //  or maybe the two given groups are not connected.
      return nullptr;
    }

    // We are looking for a very specific patterns here. The cases that this
    //  pattern will not capture are ones that reductions of different
    //  signatures are so interleaved that we cannot find a clear cut as
    //  explained below, without graph rewriting. Some graph re-writing on the
    //  segmented groups level could provide extra merging opportunities for
    //  free, which could be part of next step.
    //
    // The specific pattern we look for contains a common producer P with
    // immediate consumers C1, C2 such that all paths from C1 to first_group and
    // all paths from C2
    //  to second_group won't hit a reduction with a different signature.

    // Topologically sort the common producers and start with the topologically
    // minimal,
    //  i.e. one that are closest to the two groups. This will cut the search
    //  space.
    std::vector<SegmentedGroup*> common_producers(
        common_producers_set.begin(), common_producers_set.end());
    std::sort(
        common_producers.begin(),
        common_producers.end(),
        [&dependency_analysis](SegmentedGroup* a, SegmentedGroup* b) {
          return dependency_analysis->isConsumerOf(a, b);
        });

    // Use a visited filter to prune search space.
    GroupSet visited_common_producers;

    // Visit the common producers found, starting from topologically minimum,
    // i.e. the ones closer to the groups
    for (auto common_producer : common_producers) {
      // Visit this common producer
      // Use a double loop in case the schedulers like some patterns
      //  better than the other
      for (auto first_consumer_edge : common_producer->consumer_edges) {
        auto producer_of_first_group = first_consumer_edge->to;
        if (visited_common_producers.count(producer_of_first_group)) {
          // We have visited this node as common producer before and it
          //  had conflicts. It'd hit the same conflict again if we continued
          //  to pursue this edge.
          continue;
        }
        auto to_merge_with_first_group = getValidMinVerticalMergedGroupSet(
            producer_of_first_group, first_group);
        if (to_merge_with_first_group.empty()) {
          // There's no valid merge path from this consumer of common producer,
          //  either due to a conflicting reduction signature, or simply there's
          //  no path to first group
          continue;
        }
        for (auto second_consumer_edge : common_producer->consumer_edges) {
          auto producer_of_second_group = second_consumer_edge->to;
          if (visited_common_producers.count(producer_of_second_group)) {
            // We have visited this node as common producer before and it
            //  had conflicts. It'd hit the same conflict again if we continued
            //  to pursue this edge.
            continue;
          }
          auto to_merge_with_second_group = getValidMinVerticalMergedGroupSet(
              producer_of_second_group, second_group);
          if (to_merge_with_second_group.empty()) {
            // There's no valid merge path from this consumer of common
            // producer,
            //  either due to a conflicting reduction signature, or simply
            //  there's no path to second group
            continue;
          }

          // At this point we should have a pair of valid candidates,final check
          // is to see if the combined group
          //  can be scheduled by schedulers
          // merge the two paths and de-duplicate,
          //  re-using container here with to_merge_with_second_group
          auto& groups_to_merge_set = to_merge_with_second_group;
          groups_to_merge_set.insert(
              to_merge_with_first_group.begin(),
              to_merge_with_first_group.end());
          std::vector<SegmentedGroup*> groups_to_merge_vec(
              groups_to_merge_set.begin(), groups_to_merge_set.end());
          Fusion* fusion =
              &segment_candidate_finder_->segmented_fusion_->completeFusion();
          if (tryMerge(fusion, groups_to_merge_vec)) {
            // Found a valid horizontal merge, want to proceed with merging here
            auto joined_group = segment_candidate_finder_->mergeAllGivenGroups(
                groups_to_merge_vec);
            dependency_analysis->mergeGroups(groups_to_merge_set, joined_group);

            groups_with_reductions_.push_back(joined_group);
            group_reduction_signature_map_[joined_group] =
                group_reduction_signature_map_.at(first_group);
            groups_with_reductions_.erase(
                std::remove_if(
                    groups_with_reductions_.begin(),
                    groups_with_reductions_.end(),
                    [&groups_to_merge_set](SegmentedGroup* group) {
                      return groups_to_merge_set.count(group);
                    }),
                groups_with_reductions_.end());

            return joined_group;
          }
        }
      }
      // Here we should have searched all consumer edges of this common producer
      // and
      //  found no valid pattern. Should just add it to the visted list.
      visited_common_producers.insert(common_producer);
    }

    // Searched all possibilities and there is no valid horizontal merge pattern
    //  found.
    return nullptr;
  }

  //! This is a utility method that is used in both vertical merging and
  //! horizontal merging.
  //!  It is used to identify the smallest set of groups to merge vertically
  //!  involving the
  //!   two given nodes.
  //!  Given a pair of nodes this utility distinguishes 3 cases:
  //!   1. if maybe_producer is the same as maybe_consumer, then returns
  //!   {maybe_producer}
  //!   2. if maybe_producer is actually a producer of consumer, returns a set
  //!   containing
  //!     the smallest merged group that would contain producer and consumer and
  //!     would not introduce a cycle. Returns empty set if such group has
  //!     a conflicting reduction signature.
  //!   3. returns empty set if neither conditions above apply.
  GroupSet getValidMinVerticalMergedGroupSet(
      SegmentedGroup* maybe_producer,
      SegmentedGroup* maybe_consumer) {
    auto dependency_analysis = segment_candidate_finder_->getGroupDependency();
    if (maybe_consumer == maybe_producer) {
      // maybe producer is the same as maybe_consumer
      return {maybe_consumer};
    } else if (dependency_analysis->isConsumerOf(
                   maybe_consumer, maybe_producer)) {
      auto groups_to_check =
          dependency_analysis->valuesBetween(maybe_producer, maybe_consumer);
      groups_to_check.insert(maybe_producer);
      groups_to_check.insert(maybe_consumer);

      // Check that either no group has a reduction or all groups have the same
      // reduction signature
      ReductionSignature* reduction_signature = nullptr;

      // Iterate through the minimal group set to see if any conflicts
      for (auto group : groups_to_check) {
        // Check that this group does not involve a output edge contraction
        //  This pass is intended to be a pre-merging pass. Since contracting an
        //   output edge does not generate much saving of global memory access
        //   we want to postpone merging these edges till the very final pass
        for (auto producer_edge_of_group : group->producer_edges) {
          if (groups_to_check.count(producer_edge_of_group->from) &&
              producer_edge_of_group->val->isFusionOutput()) {
            return {};
          }
        }
        for (auto consumer_edge_of_group : group->consumer_edges) {
          if (groups_to_check.count(consumer_edge_of_group->to) &&
              consumer_edge_of_group->val->isFusionOutput()) {
            return {};
          }
        }

        // Check that this group does not have a conflicting reduction signature
        if (group_reduction_signature_map_.count(group)) {
          if (reduction_signature != nullptr) {
            if (!group_reduction_signature_map_.at(group)->sameAs(
                    reduction_signature)) {
              // Found a conflict in reduction signature, cannot do a vertical
              // merge
              return {};
            }
          } else {
            reduction_signature = group_reduction_signature_map_.at(group);
          }
        }
      }
      return groups_to_check;
    }
    // maybe producer is not a producer of maybe consumer
    return {};
  }

 private:
  SegmentCandidateFinder* segment_candidate_finder_;

  // Wrapper class for reduction type
  //  Assuming there wouldn't be too many of them
  //  so won't need to create a hash
  // TODO:
  //   Want to reconsider this for transpose operations,
  //   need refactoring to handle reduction fusions across a transpose operation
  struct ReductionSignature {
    size_t root_domain_size = 0;
    std::vector<int> reduction_axes;
    bool has_nontrivial_reduction = false;

    ReductionSignature(ReductionOp* rop) {
      auto out_tv = rop->out()->as<TensorView>();
      has_nontrivial_reduction = out_tv->hasReduction();
      TORCH_INTERNAL_ASSERT(out_tv != nullptr);
      auto& root_domain = out_tv->getRootDomain();
      root_domain_size = root_domain.size();

      // Trivial reduction i.e. squeeze is tricky here:
      //  this pass doesn't want to touch any pure squeeze, i.e.:
      //    T0 [R(1), I(i0), I(i1)]
      //  meanwhile, for two reductions having
      //  squeezes, we do require they have squeeze at the
      //  same position so that they can be easily root domain mapped
      //  So T0 and T1 are the same signature,
      //    T0 [R(1), R(i0), I(i1)]
      //    T1 [R(1), R(i0), I(i1)]
      //  but T2 and T3 below are not
      //    T0 [R(1), R(1), R(i0), I(i1)]
      //    T1 [R(1), R(i0), I(i1)]
      for (size_t i = 0; i < root_domain_size; i++) {
        if (root_domain[i]->isReduction()) {
          reduction_axes.push_back(i);
        }
        if (!root_domain[i]->isTrivialReduction()) {
          has_nontrivial_reduction = true;
        }
      }
    }

    bool sameAs(const ReductionSignature* reduction_signature) {
      if (reduction_signature == this) {
        return true;
      }

      if (root_domain_size != reduction_signature->root_domain_size ||
          has_nontrivial_reduction !=
              reduction_signature->has_nontrivial_reduction ||
          reduction_axes.size() != reduction_signature->reduction_axes.size()) {
        return false;
      }

      for (size_t i = 0; i < reduction_axes.size(); i++) {
        if (reduction_axes[i] != reduction_signature->reduction_axes[i]) {
          return false;
        }
      }

      return true;
    }

    bool sameAs(const ReductionSignature& reduction_signature) {
      return sameAs(&reduction_signature);
    }
  };

  //! Keeps track of groups with reduction expressions,
  //!  using a vector here to maintain a deterministic ordering
  GroupVec groups_with_reductions_;

  //! Maps groups to their corresponding signature type
  std::unordered_map<SegmentedGroup*, ReductionSignature*>
      group_reduction_signature_map_;

  //! Maintains all reduction signatures seen in the segmented fusion
  std::vector<std::unique_ptr<ReductionSignature>> known_reduction_signatures_;
};

//! This is to be checked
bool CombineReductions::shouldRun(
    SegmentCandidateFinder* segment_candidate_finder) {
  std::vector<std::unique_ptr<ReductionSignature>> known_reductions;
  // Iterate over group segments we have before segment candidate finder
  //  tries to merge any groups
  for (auto group : segment_candidate_finder->groups()) {
    if (auto rop = firstReductionFromGroup(group)) {
      auto reduction_signature = std::make_unique<ReductionSignature>(rop);
      if (reduction_signature->has_nontrivial_reduction &&
          std::any_of(
              known_reductions.begin(),
              known_reductions.end(),
              [&reduction_signature](auto& know_signature) {
                return know_signature->sameAs(reduction_signature.get());
              })) {
        // Found two reductions with the same signature, run pass
        return true;
      }
      known_reductions.emplace_back(std::move(reduction_signature));
    }
  }
  return false;
}

bool SegmentCandidateFinder::codeGenSupportedMerge(SegmentedEdge* edge) {
  Fusion* fusion = &segmented_fusion_->completeFusion();
  auto h = tryMerge(fusion, edge->from, edge->to);
  return h.has_value();
}

// TODO: consider caching the heuristics value so tryMerge doesn't have to be
//       called twice
ScheduleHeuristic SegmentCandidateFinder::deriveHeuristic(
    SegmentedGroup* group) {
  Fusion* fusion = &segmented_fusion_->completeFusion();
  auto h = tryMerge(fusion, group);
  TORCH_INTERNAL_ASSERT(h.has_value());
  return h.value();
}

SegmentCandidateFinder::SegmentCandidateFinder(
    const Fusion* fusion,
    SegmentCandidateFinderOptions options)
    : options_(options) {
  segmented_fusion_ = std::make_unique<SegmentedFusion>(fusion);
  findSegments();
  if (isDebugDumpEnabled(DebugDumpOption::FusionSegmentsDrawing)) {
    segmented_fusion_->draw();
  }
}

void SegmentCandidateFinder::findSegments() {
  FUSER_PERF_SCOPE("Finding valid fusion segment solutions");
  // TODO: Make traversal items local to this function.

  // Need this for initialization of the DAG that is process
  std::unordered_map<Expr*, SegmentedGroup*> expr2group;

  // Keep track of complete fusion input use
  std::unordered_map<Val*, SegmentedGroup*> input2group;

  // Initialize DAG, convert each expr to a segment group
  auto exprs = completeFusion().exprs();
  for (auto expr : exprs) {
    if (!ir_utils::isScalarOp(expr)) {
      auto new_group = segmented_fusion_->newGroup(expr);
      expr2group.insert(std::make_pair(expr, new_group));
    }
  }

  // Insert auxiliary groups to use group dependency on inputs as well
  // TODO: these groups should never merged into any other groups, but are
  //       just there to support the dependency analysis. Later re-factor should
  //       avoid introducing them explicitly on the segmented fusion.
  for (auto input : completeFusion().inputs()) {
    // These groups are used to represent input as a common
    //  producer in horizontal merges, and should never be
    //  seen as a candidate for vertical merge
    auto new_group = segmented_fusion_->newGroup();
    input2group.insert({input, new_group});
  }

  // Create edges between the Exprs. Mark inputs and outputs of the fusion.
  for (auto expr : exprs) {
    // No group created for scalar ops
    if (ir_utils::isScalarOp(expr)) {
      continue;
    }

    auto expr_group = expr2group.at(expr);
    for (auto inp : expr->inputs()) {
      if (inp->isFusionInput()) {
        expr_group->input_vals.push_back(inp);
        auto aux_group = input2group.at(inp);
        auto new_edge = segmented_fusion_->newEdge(aux_group, expr_group, inp);
        expr_group->producer_edges.push_back(new_edge);
        aux_group->consumer_edges.push_back(new_edge);
        continue;
      }

      // Could be something like a constant scalar, definition is nullptr, but
      // isn't an "input" to the fusion. At least not one provided by an
      // external source.
      if (inp->definition() == nullptr) {
        continue;
      }

      // No group created for scalar ops since they may need to be duplicated
      //  to avoid scalar edges. They are handled in resolveScalarsInGroup
      if (inp->isScalar()) {
        continue;
      }

      auto def_group = expr2group.at(inp->definition());
      auto new_edge = segmented_fusion_->newEdge(def_group, expr_group, inp);
      expr_group->producer_edges.push_back(new_edge);
      def_group->consumer_edges.push_back(new_edge);
    }
    for (auto out : expr->outputs()) {
      if (out->isFusionOutput()) {
        expr_group->output_vals.push_back(out);
      }
    }
  }

  for (auto group : groups()) {
    // Add all the scalar inputs needed in the group
    resolveScalarsInGroup(group);
    // Set heuristics in case single reduction kernels were left out
    group->setHeuristic(deriveHeuristic(group));
  }

  // Run pre-merge heuristics
  if (options_.run_combine_reductions && CombineReductions::shouldRun(this)) {
    CombineReductions::run(this);
  }

  // All merges will be vertical beyond this point for now, so
  //  we can remove the input auxiliary groups. Should make the vertical
  //  merges avoid auxiliary group once we start general horizontal merges
  std::unordered_set<SegmentedGroup*> input_groups;
  for (auto input : completeFusion().inputs()) {
    input_groups.insert(input2group.at(input));
  }
  eraseGroups(input_groups);

  if (options_.run_herrmann_merge) {
    bool merged_nodes = true;
    // Initial merge iteration
    while (merged_nodes) {
      // Reset stateful traversal details in SegmentedGroups
      resetTraversal();

      resetLevels();

      for (auto& group : groups()) {
        if (group->merged_) {
          continue;
        }
        auto candidates = group->getMergeCandidates();
        if (candidates.empty()) {
          continue;
        }

        auto candidate_it = candidates.begin();
        while (candidate_it != candidates.end() &&
               !codeGenSupportedMerge(candidate_it->edge)) {
          candidate_it++;
        }
        if (candidate_it == candidates.end()) {
          continue;
        }

        to_merge_.emplace_back(group);
        to_merge_.emplace_back(candidate_it->group);

        group->merged_ = true;
        group->merge_with_ = candidate_it->group;
        group->merge_through_ = candidate_it->edge;

        candidate_it->group->merged_ = true;
        candidate_it->group->merge_with_ = group;
        candidate_it->group->merge_through_ = candidate_it->edge;
      }

      if (to_merge_.empty()) {
        merged_nodes = false;
      }

      mergeNodes();
    }
  }

  if (options_.run_final_merge) {
    // TODO: consider interleaving herrmman merge and bruteforce merge, as
    // bruteforce merge can introduce
    //  opportunities for more herrmann merge
    finalMerge();
  }

  finalize();
}

void SegmentCandidateFinder::finalMerge() {
  auto producer_check = getGroupDependency();

  bool merged_nodes = true;
  while (merged_nodes) {
    // Iterate all groups and check if a group
    //  can merge with one of its consumers
    for (auto producer_group : groups()) {
      // Populate consumers and their corresponding consumer edges
      std::unordered_map<SegmentedGroup*, SegmentedEdge*> consumer_edge_map;
      std::vector<SegmentedGroup*> all_consumers_of_producer_group;
      for (auto consumer : producer_group->consumer_edges) {
        // Since this is the last fusion pass, we can enable fusion through
        // outputs. Priority of this was decreased because if the only
        // connection between groups is an output node, best case scenario we
        // can save a single pass in memory. Where if it wasn't an output it
        // would be two passes.
        consumer_edge_map.insert({consumer->to, consumer});
      }
      // Populate all consumers from the map to avoid duplicate
      std::transform(
          consumer_edge_map.begin(),
          consumer_edge_map.end(),
          std::back_inserter(all_consumers_of_producer_group),
          [](auto& it) { return it.first; });

      for (auto consumer : all_consumers_of_producer_group) {
        if (!producer_check->isConsumerOfAny(
                consumer, all_consumers_of_producer_group) &&
            codeGenSupportedMerge(consumer_edge_map.at(consumer))) {
          to_merge_.emplace_back(producer_group);
          to_merge_.emplace_back(consumer);
          producer_group->merged_ = true;
          producer_group->merge_with_ = consumer;
          producer_group->merge_through_ = consumer_edge_map.at(consumer);
          consumer->merged_ = true;
          consumer->merge_with_ = producer_group;
          consumer->merge_through_ = producer_group->merge_through_;
          break;
        }
      }

      // Only want to merge one pair at a time so break if found any
      if (!to_merge_.empty()) {
        break;
      }
    }

    if (to_merge_.empty()) {
      merged_nodes = false;
    } else {
      TORCH_INTERNAL_ASSERT(
          to_merge_.size() == 2, "merging more than 2 nodes in final iter");
      mergeNodes();
    }
  }
}

void SegmentCandidateFinder::resolveScalarsInGroup(SegmentedGroup* group) {
  std::vector<Val*> to_visit;
  std::unordered_set<Val*> visited;

  // Collect all scalar uses in the group
  for (auto expr : group->exprs()) {
    for (auto input : expr->inputs()) {
      if (input->isScalar()) {
        to_visit.push_back(input);
      }
    }
  }

  // Keep track of composite fusion inputs used in this group
  std::unordered_set<Val*> input_set(
      group->input_vals.begin(), group->input_vals.end());

  // Record and append all missing scalar exprs at the end.
  std::vector<Expr*> exprs_to_add;

  // Do a stack based traversal of the scalar ops to avoid
  //  combinatorial duplication of exprs.
  while (!to_visit.empty()) {
    auto stack_top_val = to_visit.back();
    if (visited.count(stack_top_val)) {
      to_visit.pop_back();
    } else if (stack_top_val->definition() == nullptr) {
      // A scalar without def can be a scalar, a tensor dim,
      //  or a composite fusion input
      // The first two cases are handled in finalize(),
      //  the last case needs to add new input_val to this group.
      visited.insert(stack_top_val);
      // If this is a composite fusion scalar input, make sure this group has it
      if (stack_top_val->isFusionInput() && !input_set.count(stack_top_val)) {
        group->input_vals.push_back(stack_top_val);
        input_set.insert(stack_top_val);
      }
      to_visit.pop_back();
    } else {
      // A scalar with an actual definition
      auto definition_expr = stack_top_val->definition();
      bool all_inputs_visited = true;
      // If any of the inputs are not visited, visit them first
      for (auto input : definition_expr->inputs()) {
        if (!visited.count(input)) {
          all_inputs_visited = false;
          to_visit.push_back(input);
        }
      }
      // This node is ready to be visited
      if (all_inputs_visited) {
        // Collect the defining expr to insert into group
        exprs_to_add.push_back(definition_expr);
        visited.insert(stack_top_val);
        to_visit.pop_back();
      }
    }
  }

  // Add all the defining expr to the group
  for (auto expr : exprs_to_add) {
    group->exprs_.push_back(expr);
  }
}

void SegmentCandidateFinder::finalize() {
  // Remove unconnected groups
  groups().erase(
      std::remove_if(
          groups().begin(),
          groups().end(),
          [](SegmentedGroup* sg) { return !sg->isConnected(); }),
      groups().end());

  // Add group labeling
  int i = 0;
  for (auto it = groups().begin(); it != groups().end(); it++, i++) {
    deDuplicateScalarExprs((*it)->exprs_);
    (*it)->setID(i);
  }

  segmented_fusion_->finalize();
}

GroupDependencyAnalysis* SegmentCandidateFinder::getGroupDependency() {
  if (!group_dependency_) {
    group_dependency_ =
        std::make_unique<GroupDependencyAnalysis>(segmented_fusion_.get());
  }
  return group_dependency_->as<GroupDependencyAnalysis>();
}

namespace {
inline void copyValue(
    Val* key,
    ExpressionEvaluator& from,
    ExpressionEvaluator& to) {
  auto concrete_val = from.evaluate(key);
  TORCH_INTERNAL_ASSERT(concrete_val.has_value());
  to.bind(key, concrete_val.value());
}

inline void inferGroupInputs(
    SegmentedGroup* sg,
    ExpressionEvaluator& ee,
    ExpressionEvaluator& local_ee) {
  for (auto v : getAllInputs(sg)) {
    if (auto tv = dynamic_cast<TensorView*>(v)) {
      for (auto id : tv->getRootDomain()) {
        auto extent = id->extent();
        copyValue(extent, ee, local_ee);
      }
    } else if (v != nullptr && v->isAnInt()) {
      copyValue(v, ee, local_ee);
    }
  }
}
} // namespace

FusionKernelRuntime::SchedulerEntryPtr SegmentedFusion::makeSchedulerEntry(
    SegmentedGroup* sg,
    ExpressionEvaluator& ee) {
  ExpressionEvaluator local_ee(&fusion_);
  inferGroupInputs(sg, ee, local_ee);
  FusionSegmentGuard fsg(&fusion_, getAllInputs(sg), getAllOutputs(sg));
  return SchedulerEntry::makeEntry(sg->heuristic(), &fusion_, local_ee);
}

std::unique_ptr<FusionHeuristics> SegmentedFusion::makeHeuristics(
    const at::ArrayRef<IValue>& inputs) {
  auto ret = std::make_unique<FusionHeuristics>();
  auto evaluator = executor_utils::bindFusionInputs(inputs, &fusion_);
  for (auto g : groups()) {
    ret->emplaceBack(makeSchedulerEntry(g, evaluator));
  }
  return ret;
}

TORCH_CUDA_CU_API std::string toString(
    const SegmentCandidateFinderOptions& segment_options) {
  std::stringstream ss;
  ss << "segmentation phases {\n";
  if (segment_options.run_combine_reductions) {
    ss << "combine reductions\n";
  }
  if (segment_options.run_herrmann_merge) {
    ss << "herrmann merging\n";
  }
  if (segment_options.run_final_merge) {
    ss << "final merging\n";
  }
  ss << "\n}\n";
  return ss.str();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch