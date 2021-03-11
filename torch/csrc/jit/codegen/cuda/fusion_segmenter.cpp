#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/fusion_segmenter.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_cloner.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>

#include <sstream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

std::vector<SegmentedGroup::NeighborGroup> SegmentedGroup::getNeighborGroups() {
  std::vector<NeighborGroup> neighbors;
  for (auto inp : producer_edges) {
    neighbors.emplace_back(inp->from, inp);
  }
  for (auto out : consumer_edges) {
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
    : fusion_(*fusion), impl_(this) {}

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

//! An utility class to compute and maintain the "producers of"
//!   relationship in a segmented graph. Space heavy and should
//!   avoid use on very large graphs.
class AllProducerGroups {
  using GroupSet = std::unordered_set<SegmentedGroup*>;
  using GroupSetPtr = std::unique_ptr<GroupSet>;
  using ReachMap = std::unordered_map<SegmentedGroup*, GroupSetPtr>;

 public:
  //! Populate producers of all groups in segmented fusion
  explicit AllProducerGroups(SegmentedFusion* segmented_fusion)
      : segmented_fusion_(segmented_fusion) {
    computeAllProducers();
  }

  //! Checks if group is consumer of any group in groups_to_check
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

  //! Update the map when the given two groups have been merged to create `ab`
  void mergeGroups(SegmentedGroup* a, SegmentedGroup* b, SegmentedGroup* ab) {
    // Access/Create the producer set of ab
    auto& ab_set = getAllKnownProducersSet(ab);

    // propagate a's and b's known producers into ab
    mergeAllKnownProducersIntoFrom(ab, a);
    mergeAllKnownProducersIntoFrom(ab, b);

    // a, b are now merged, so no longer exist
    ab_set->erase(a);
    ab_set->erase(b);

    // a, b no longer exist, remove their producer sets
    producer_map_.erase(a);
    producer_map_.erase(b);

    // update producer maps of other groups
    for (auto& it : producer_map_) {
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

 private:
  //! Collect initial producer info using
  //!  a work list algorithm through forward traversal
  //!  a backward DFS would do the same
  void computeAllProducers() {
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
  GroupSetPtr& getAllKnownProducersSet(SegmentedGroup* group) {
    auto& producer_set_ptr = producer_map_[group];
    if (!producer_set_ptr) {
      producer_set_ptr = std::make_unique<GroupSet>();
    }
    return producer_set_ptr;
  }

 private:
  SegmentedFusion* segmented_fusion_;
  ReachMap producer_map_;
};

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

} // namespace

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

SegmentCandidateFinder::SegmentCandidateFinder(const Fusion* fusion) {
  segmented_fusion_ = std::make_unique<SegmentedFusion>(fusion);
  findSegments();
}

void SegmentCandidateFinder::findSegments() {
  FUSER_PERF_SCOPE("Finding valid fusion segment solutions");
  // TODO: Make traversal items local to this function.

  // Need this for initialization of the DAG that is process
  std::unordered_map<Expr*, SegmentedGroup*> expr2group;

  // Initialize DAG, convert each expr to a segment group
  size_t total_tv_exprs = 0;
  auto exprs = completeFusion().exprs();
  for (auto expr : exprs) {
    if (!ir_utils::isScalarOp(expr)) {
      auto new_group = segmented_fusion_->newGroup(expr);
      expr2group.insert(std::make_pair(expr, new_group));
      total_tv_exprs++;
    }
  }

  segmented_fusion_->total_tv_expr_count_ = total_tv_exprs;

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

  finalMerge();

  finalize();
}

void SegmentCandidateFinder::finalMerge() {
  AllProducerGroups producer_check(segmented_fusion_.get());

  bool merged_nodes = true;
  while (merged_nodes) {
    // Iterate all groups and check if a group
    //  can merge with one of its consumers
    for (auto producer_group : groups()) {
      // Populate consumers and their corresponding consumer edges
      std::unordered_map<SegmentedGroup*, SegmentedEdge*> consumer_edge_map;
      std::vector<SegmentedGroup*> all_consumers_of_producer_group;
      for (auto consumer : producer_group->consumer_edges) {
        consumer_edge_map.insert({consumer->to, consumer});
      }
      // Populate all consumers from the map to avoid duplicate
      std::transform(
          consumer_edge_map.begin(),
          consumer_edge_map.end(),
          std::back_inserter(all_consumers_of_producer_group),
          [](auto& it) { return it.first; });

      for (auto consumer : all_consumers_of_producer_group) {
        if (!producer_check.isConsumerOfAny(
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
      auto merged_a = *to_merge_.begin();
      auto merged_b = merged_a->merge_with_;
      auto merged_ab = mergeNodes();
      producer_check.mergeGroups(merged_a, merged_b, merged_ab);
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
  size_t total_expr = segmented_fusion_->total_tv_expr_count_;
  groups().erase(
      std::remove_if(
          groups().begin(),
          groups().end(),
          [total_expr](SegmentedGroup* sg) {
            // count the number of tensor ops
            const size_t expr_count = std::count_if(
                sg->exprs_.begin(), sg->exprs_.end(), [](Expr* expr) {
                  return !ir_utils::isScalarOp(expr);
                });

            return !sg->isConnected() && expr_count != total_expr;
          }),
      groups().end());

  // Add group labeling
  int i = 0;
  for (auto it = groups().begin(); it != groups().end(); it++, i++) {
    deDuplicateScalarExprs((*it)->exprs_);
    (*it)->setID(i);
  }

  segmented_fusion_->finalize();
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

FusionSegmentRuntime::SchedulerEntryPtr SegmentedFusion::makeSchedulerEntry(
    SegmentedGroup* sg,
    ExpressionEvaluator& ee) {
  ExpressionEvaluator local_ee(&fusion_);
  inferGroupInputs(sg, ee, local_ee);
  FusionSegmentGuard fsg(&fusion_, getAllInputs(sg), getAllOutputs(sg));
  return SchedulerEntry::makeEntry(sg->heuristic(), &fusion_, local_ee);
}

std::unique_ptr<SegmentHeuristics> SegmentedFusion::makeHeuristics(
    const at::ArrayRef<IValue>& inputs) {
  auto ret = std::make_unique<SegmentHeuristics>();
  auto evaluator = executor_utils::bindFusionInputs(inputs, &fusion_);
  for (auto g : groups()) {
    ret->emplace_back(makeSchedulerEntry(g, evaluator));
  }
  return ret;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch