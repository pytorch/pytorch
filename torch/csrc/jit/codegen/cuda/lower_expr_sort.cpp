#include <torch/csrc/jit/codegen/cuda/compute_at_map.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_expr_sort.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

#include <deque>
#include <list>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

// TODO: Review const model, and objects
//  ExprSegmentationSorter
//    Responsible for going through DAG and proposing things we could try to
//    merge together, calls "supportedMerge" on these proposed groups to see
//    if they should be merged together, then merges them if so.
//  ExprGroup
//    A group of exprs that are grouped together based on their loop nest
//    structures.
//  ExprGroupConnections
//    Holds vals and what they connect. In other words it's a val that is an
//    output of a ExprSegmentationSorter "from" and an input of
//    ExprSegmentationSorter "to". There's nothing preventing from a val being
//    between groups twice.
//    TODO: make sure there's nothing wrong with grouping of nodes that
//    have the same value input twice. i.e. (B = A*A)

// Selecting segments to propose is based on the theorem 4.2 in the paper which
// makes sure when segment the segmented graph will be a DAG (assumes Fusion is
// already a DAG). The segmentation code relies on assumptions of DAG-ness
// during segmentation, meaning proposed merging of groups must maintain the DAG
// property of the graph.
//
// Julien Herrmann, Yusuf Özkaya, Bora Uçar, Kamer Kaya, Umit Catalyurek.
// Multilevel Algorithms for Acyclic Partitioning of Directed Acyclic Graphs.
// SIAM Journal on Scientific Computing, Society for Industrial and Applied
// Mathematics, 2019, 41 (4), pp.A2117-A2145. ff10.1137/18M1176865ff.
// ffhal02306566f

class ExprGroup;
struct ExprGroupConnections;
class ExprSegmentationSorter;

// Debug printing disabled due to clang tidy, see below for definitions
// std::ostream& operator<<(std::ostream& os, const ExprGroupConnections* edge);
// std::ostream& operator<<(std::ostream& os, const ExprGroup* group);
// std::ostream& operator<<(std::ostream& os, const ExprSegmentationSorter*
// scf);

// Wrapper for values, these are edges between expr groups. Multiple edges can
// exist between expr groups, and the same Val can show up more than once in
// multiple edges.
struct ExprGroupConnections {
  ExprGroupConnections(
      ExprGroup* group_from,
      ExprGroup* group_to,
      Val* val_to_connect)
      : from(group_from), to(group_to), val(val_to_connect) {}
  ExprGroup* from;
  ExprGroup* to;
  Val* val;
};

struct ExprSortPayload : public PolymorphicBase {
  // Need to track compute at domains as well as produce at domains. Produce at
  // domains will be matched to producers compute at domains. Track the active
  // domains that will be matched from inner most dim to outer most.
  std::vector<IterDomain*> ca_domains_;
  std::vector<IterDomain*> pa_domains_;

  // Maximum path distance from an input expr group required for
  // Theorem 4.2
  int level = -1;

  // Traversal marker, marks if this group has been visited by current pass
  bool visited = false;

  // Marks if this group is already selected to merge with another group, marks
  // which group to merge with
  ExprGroup* merge_with = nullptr;

  // Marks if this group is already selected to merge with another group
  bool merged = false;
};

// Groups together expressions which create a expr group
class ExprGroup {
 public:
  ExprGroup() : payload_(std::make_unique<ExprSortPayload>()) {}

  ExprGroup(Expr* expr) : payload_(std::make_unique<ExprSortPayload>()) {
    exprs_.push_back(expr);
  }

  ExprGroup(const ExprGroup& other)
      : payload_(new ExprSortPayload(*(other.payload_))) {}

  ExprGroup& operator=(const ExprGroup& other) {
    *payload_ = *other.payload_;
    exprs_ = other.exprs_;
    return *this;
  }

  // Clears the traversal information in the payload
  void clearTraversalInfo();

  // Returns all neighbors, producers and consumers
  std::vector<ExprGroup*> getNeighbors();

  // Return neighbors of this proven to be safe nodes to merge with in regards
  // to maining an acyclic graph. This looks at, neighbors  if merged, neighbors
  // level, and merged neighbors of neighbors level. If fallback_mode_enabled
  // will return the inverse set of ExprGroups that are proven to be safe
  // merges.
  std::vector<ExprGroup*> getMergeCandidates(
      bool fallback_mode_enabled = false);

  std::unique_ptr<ExprSortPayload>& payload() {
    return payload_;
  }

  const auto& producerEdges() const {
    return producer_edges_;
  }

  void addProducerEdge(ExprGroupConnections* edge) {
    addEdge(producer_edges_, edge);
  }

  void removeProducerEdge(ExprGroupConnections* edge) {
    removeEdge(producer_edges_, edge);
  }

  void clearProducerEdges() {
    producer_edges_.clear();
  }

  const auto& consumerEdges() const {
    return consumer_edges_;
  }

  void addConsumerEdge(ExprGroupConnections* edge) {
    addEdge(consumer_edges_, edge);
  }

  void removeConsumerEdge(ExprGroupConnections* edge) {
    removeEdge(consumer_edges_, edge);
  }

  void clearConsumerEdges() {
    consumer_edges_.clear();
  }

  auto& exprs() {
    return exprs_;
  }

  const auto& exprs() const {
    return exprs_;
  }

 private:
  static void addEdge(
      std::vector<ExprGroupConnections*>& edges,
      ExprGroupConnections* edge_to_add) {
    edges.push_back(edge_to_add);
  }

  static void removeEdge(
      std::vector<ExprGroupConnections*>& edges,
      ExprGroupConnections* edge_to_remove) {
    auto it = std::find(edges.begin(), edges.end(), edge_to_remove);
    TORCH_INTERNAL_ASSERT(it != edges.end(), "Could not find edge to remove.");
    edges.erase(it);
  }

 private:
  // "Ancestor nodes", towards inputs of segmentedDAG
  std::vector<ExprGroupConnections*> producer_edges_;

  // "Descendent nodes", towards outputs of segmentedDAG
  std::vector<ExprGroupConnections*> consumer_edges_;

  // Exprs that make up the group
  std::vector<Expr*> exprs_;

  // Stateful traversal information
  std::unique_ptr<ExprSortPayload> payload_;
};

// This class sorts expressions guarantees two things, 1) Tensors are produced
// before they're consumed 2) If the production of two tensors are supposed to
// share a for loop, they're in an order where they can. (1) is pretty standard
// of ordering a DAG. (2) is where things get a bit complicated and why we do
// this sorting through segmentation. Consider a section of a DAG: T4 = T3 + T2.
// Where T2 and T3 are not inputs to the fusion, all tensors are 3D, and we want
// the production of T3 to share the inner most loop of T4 and we want the
// production of T2 to share the middle loop with T4. i.e. we're looking for
// For(i:I){
//   For(j: J){
//     For(k: K){
//       T2[i, j, k] = ...
//     }
//     For(k: K){
//       T3[i, j, k] = ...
//       T4[i, j, k] = T2[i, j, k] + T3[i, j, k]
//     }
//   }
// }
// The only valid ordering of expressions is producing T2, then T3, then T4. If
// we swapped T3 and T2, then T3 and T4 couldn't share their inner most loop,
// because T2 has its own inner most loop. If we swapped either tensor with T4,
// then we'd try to be using T2 or T3 without producing them (back to gaurantee
// 1).
class ExprSegmentationSorter {
 public:
  ExprSegmentationSorter(Fusion* fusion) : complete_fusion_(fusion) {}

  void sort();

  std::string toString(int verbosity = 0) const;

  //! Returns a flattened list of sorted exprs
  std::vector<Expr*> getExprs() const;

 private:
  // Allocate an empty expr group and return it
  ExprGroup* makeEmptyGroup();

  // Allocate an expr group with the provided expr and return it
  ExprGroup* makeEmptyGroup(Expr*);

  // Returns if sg1 and sg2 should be merged together, is called if they can
  // based on the current status of the DAG.
  bool supportedMerge(ExprGroup* sg1, ExprGroup* sg2);

  // Returns true if the graph will remain an acyclic graph after merging sg1
  // and sg2
  bool testStillDag(ExprGroup* sg1, ExprGroup* sg2);

  // Merges two ExprGroups and returns the new ExprGroup
  ExprGroup* makeMergedNode(ExprGroup* sg1, ExprGroup* sg2);

  // This is called once no more groups can be merged together. This will lower
  // the compute at position of a segment group if the last dimension of the
  // segment group doesn't map to any of the dimensions of its neighbors.
  bool interIterUpdate();

  // Reset the ExprSortPayload of the groups so we can traverse and identify
  // merge candidates.
  void resetTraversal();

  // Reset the set levels of each group. This is what's used to identify which
  // nodes can be merged together.
  void resetLevels();

  // Go through groups that are marked as to merge and merge them.
  void mergeNodes();

  // Disconnect the edges connecting group to the rest of the graph, and return
  // all the edges that were disconnected
  std::unordered_set<ExprGroupConnections*> disconnectGroup(ExprGroup* group);

 private:
  // Track how many groups we have from iteration to iteration so we can track
  // when we've stopped merging nodes.
  size_t n_groups_ = 0;

  // Lifetime of the graph view of the fusion and segmentation. Use list to not
  // invalidate any entries on insertion/deletion.
  std::list<std::unique_ptr<ExprGroupConnections>> edges_;
  std::list<std::unique_ptr<ExprGroup>> groups_;

  std::deque<ExprGroup*> to_visit_;

  std::unordered_set<ExprGroup*> to_merge_;

  // Maintain my own fusion the state of which is not always the same as the
  // original provided fusion.
  Fusion* complete_fusion_;

  // We use a theorem out of a paper mentioned in other comments. This theorem
  // is good at identifying multiple expr groups to merge during a single
  // iteration without producing a cyclic graph from an acyclic graph. This
  // theorem is not guaranteed to find all possible nodes that can be merged
  // together. We need to be able to group all disjoint groups of exprs or
  // we fail to generate code. Therefore, if we can't find anything to make
  // forward progress based on the theorem we fallback to manually looking if we
  // can segmenet all combinations we haven't previously looked at.
  bool fallback_mode_enabled_ = false;
};

// Debug printing, disabled due to clang-tidy see above for declarations.
// std::ostream& operator<<(std::ostream& os, ExprGroup* group) {
//   os << "g{";
//   for (size_t i = 0; i < group->exprs().size(); i++) {
//     os << group->exprs()[i]->name();
//     if (i + 1 != group->exprs().size())
//       os << ", ";
//   }
//   os << "} (" << group->payload()->ca_domains_.size() << ", "
//      << group->payload()->pa_domains_.size() << ")";
//   os << " ca_ids {";
//   for (size_t i = 0; i < group->payload()->ca_domains_.size(); i++) {
//     os << group->payload()->ca_domains_[i];
//     if (i + 1 != group->payload()->ca_domains_.size())
//       os << ", ";
//   }
//   os << "} pa_ids {";
//   for (size_t i = 0; i < group->payload()->pa_domains_.size(); i++) {
//     os << group->payload()->pa_domains_[i];
//     if (i + 1 != group->payload()->pa_domains_.size())
//       os << ", ";
//   }
//   os << "}";
//   return os;
// }

// std::ostream& operator<<(std::ostream& os, const ExprGroupConnections* edge)
// {
//   os << "e{ " << edge->from << " -> " << edge->to << " }" << std::endl;
//   return os;
// }

// std::ostream& operator<<(std::ostream& os, const ExprSegmentationSorter* scf)
// {
//   return os << scf->toString();
// }

std::vector<ExprGroup*> ExprGroup::getNeighbors() {
  std::vector<ExprGroup*> neighbors;
  for (auto inp : producer_edges_) {
    neighbors.push_back(inp->from);
  }
  for (auto out : consumerEdges()) {
    neighbors.push_back(out->to);
  }
  return neighbors;
}

std::vector<ExprGroup*> ExprGroup::getMergeCandidates(
    bool fallback_mode_enabled) {
  std::vector<ExprGroup*> neighbors = getNeighbors();

  // Don't look for candidates if already merged
  if (payload()->merged) {
    return {};
  }

  // Can this node be merged with another? Check if neighbors are merged, if
  // so and merged neighbor is within 1 level or node merged with neighbor is
  // within 1 level, can't merge this node with anything else.
  bool can_merge_this = true;
  bool neighbor_merged = false;
  for (auto neighbor : neighbors) {
    if (!neighbor->payload()->merged) {
      continue;
    }
    neighbor_merged = true;
    if (std::abs(neighbor->payload()->level - payload()->level) <= 1) {
      can_merge_this = false;
    }
    if (std::abs(
            neighbor->payload()->merge_with->payload()->level -
            payload()->level) <= 1) {
      can_merge_this = false;
    }
  }

  // If something prevents us from merging this node, and we're not in fallback
  // mode, return empty set.
  if (!can_merge_this && !fallback_mode_enabled) {
    return {};
  }

  // If fallback mode already detected a merge somewhere, we shouldn't still be
  // traversing.
  if (fallback_mode_enabled) {
    TORCH_INTERNAL_ASSERT(
        !neighbor_merged,
        "Shouldn't still be traversing in fallback mode if a merge was found.");
  }

  std::vector<bool> can_merge(true, neighbors.size());

  // Find neighbors with a level that is only 1 differant than this groups level
  for (size_t i = 0; i < neighbors.size(); i++) {
    if (std::abs(neighbors[i]->payload()->level - payload()->level) > 1) {
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

    for (auto neighbor_neighbor : neighbors[i]->getNeighbors()) {
      // Don't check self
      if (neighbor_neighbor == neighbors[i]) {
        continue;
      }
      if (neighbor_neighbor->payload()->merged) {
        // check neighbor_neighbor level
        if (std::abs(neighbor_neighbor->payload()->level - payload()->level) <=
            1) {
          can_merge[i] = false;
        }
        if (std::abs(
                neighbor_neighbor->payload()->level -
                neighbors[i]->payload()->level) <= 1) {
          can_merge[i] = false;
        }

        // check neighbor_neighber->merged->level
        if (std::abs(
                neighbor_neighbor->payload()->merge_with->payload()->level -
                payload()->level) <= 1) {
          can_merge[i] = false;
        }
        if (std::abs(
                neighbor_neighbor->payload()->merge_with->payload()->level -
                neighbors[i]->payload()->level) <= 1) {
          can_merge[i] = false;
        }
      }
    }
  }

  std::vector<ExprGroup*> merge_candidates;
  for (size_t i = 0; i < neighbors.size(); i++) {
    if ((can_merge[i] && !fallback_mode_enabled) ||
        (!can_merge[i] && fallback_mode_enabled)) {
      merge_candidates.push_back(neighbors[i]);
    }
  }
  return merge_candidates;
}

void ExprGroup::clearTraversalInfo() {
  payload()->level = -1;
  payload()->visited = false;
  payload()->merge_with = nullptr;
  payload()->merged = false;
}

void ExprSegmentationSorter::resetTraversal() {
  for (auto& group : groups_) {
    // Start traversal at input groups
    if (group->producerEdges().empty()) {
      to_visit_.push_back(group.get());
    }
    group->clearTraversalInfo();
  }
}

// Level is maximum distance from inputs. It's the metric used to select what
// nodes can be merged while maintaining a DAG
void ExprSegmentationSorter::resetLevels() {
  std::vector<ExprGroup*> next_to_visit;

  while (!to_visit_.empty()) {
    auto visit = to_visit_.front();
    to_visit_.pop_front();

    // All inputs processed?
    bool ready = true;
    if (!visit->producerEdges().empty()) {
      ready = std::all_of(
          visit->producerEdges().begin(),
          visit->producerEdges().end(),
          [&](ExprGroupConnections* dep) {
            return dep->from->payload()->visited;
          });
    }

    if (!ready) {
      // In case traversal doesn't complete because there's an error in the
      // DAG topology.
      next_to_visit.push_back(visit);
      continue;
    }

    visit->payload()->visited = true;

    to_visit_.insert(
        to_visit_.end(), next_to_visit.begin(), next_to_visit.end());
    next_to_visit.clear();

    for (auto out : visit->consumerEdges()) {
      to_visit_.push_back(out->to);
    }

    visit->payload()->level = 0;
    for (auto inp : visit->producerEdges()) {
      visit->payload()->level =
          std::max(visit->payload()->level, inp->from->payload()->level + 1);
    }
  }
  TORCH_INTERNAL_ASSERT(next_to_visit.empty(), "Error in graph, is not a DAG.");
}

ExprGroup* ExprSegmentationSorter::makeEmptyGroup() {
  groups_.push_back(std::make_unique<ExprGroup>());
  return groups_.back().get();
}

ExprGroup* ExprSegmentationSorter::makeEmptyGroup(Expr* expr) {
  auto group = makeEmptyGroup();
  group->exprs().push_back(expr);
  if (ir_utils::isTVOp(expr)) {
    auto out_tv = expr->outputs()[0]->as<TensorView>();
    // Grab all id's that are shared with other tensors.
    for (size_t tv_i = 0; tv_i < out_tv->getComputeAtPosition(); tv_i++) {
      group->payload()->ca_domains_.push_back(out_tv->axis(tv_i));
    }
    for (size_t tv_i = 0; tv_i < out_tv->getMaxProducerPosition(); tv_i++) {
      group->payload()->pa_domains_.push_back(out_tv->axis(tv_i));
    }
  }
  return group;
}

// Debug function that prints the current state of the sorter.
std::string ExprSegmentationSorter::toString(int verbosity) const {
  std::stringstream ss;
  ss << "{\n";
  for (auto& group : groups_) {
    ss << "  " << group.get() << "\n";

    if (verbosity > 1) {
      if (group->producerEdges().size() > 0) {
        ss << "    produced by groups: { \n";
        for (auto producer_edge : group->producerEdges()) {
          ss << "      " << producer_edge->from << " via " << producer_edge->val
             << "\n";
        }
        ss << "    }"
           << "\n";
      }
    }

    if (verbosity > 0) {
      if (group->consumerEdges().size() > 0) {
        ss << "    Consumed by groups: { \n";
        for (auto consumer_edge : group->consumerEdges()) {
          ss << "      " << consumer_edge->to << "\n";
        }
        ss << "    }"
           << "\n";
      }
    }

    if (verbosity > 2) {
      ss << "    Exprs{\n";
      for (auto expr : group->exprs()) {
        ss << expr;
      }
      ss << "    }\n";
    }
  }
  ss << "}\n";
  return ss.str();
}

namespace {

// Concat's edges of sg1 and sg2, but removes any edges from/to sg1/sg2
std::vector<ExprGroupConnections*> getMergedEdges(
    const ExprGroup* sg1,
    const std::vector<ExprGroupConnections*>& edges1,
    const ExprGroup* sg2,
    const std::vector<ExprGroupConnections*>& edges2) {
  TORCH_INTERNAL_ASSERT(
      sg1 != nullptr && sg2 != nullptr,
      "This function doesn't handle trivial.");

  auto merged_edges = edges1;
  merged_edges.insert(merged_edges.end(), edges2.begin(), edges2.end());

  // Remove intra edges
  merged_edges.erase(
      std::remove_if(
          merged_edges.begin(),
          merged_edges.end(),
          [&sg1, &sg2](ExprGroupConnections* se) {
            return (se->to == sg1 && se->from == sg2) ||
                (se->to == sg2 && se->from == sg1);
          }),
      merged_edges.end());

  return merged_edges;
}

// Concat's producer edges of sg1 and sg2, but removes any edges from/to sg1/sg2
std::vector<ExprGroupConnections*> getMergedProducerEdges(
    const ExprGroup* sg1,
    const ExprGroup* sg2) {
  return getMergedEdges(sg1, sg1->producerEdges(), sg2, sg2->producerEdges());
}

// Concat's consumer edges of sg1 and sg2, but removes any edges from/to sg1/sg2
std::vector<ExprGroupConnections*> getMergedConsumerEdges(
    const ExprGroup* sg1,
    const ExprGroup* sg2) {
  return getMergedEdges(sg1, sg1->consumerEdges(), sg2, sg2->consumerEdges());
}

// Assuming sg1 and sg2 are connected, figure out which is the consumer
ExprGroup* getProducer(ExprGroup* sg1, ExprGroup* sg2) {
  for (auto producer_edge : sg1->producerEdges()) {
    if (producer_edge->from == sg2) {
      return sg2;
    }
  }

  for (auto consumer_edge : sg1->consumerEdges()) {
    if (consumer_edge->to == sg2) {
      return sg1;
    }
  }

  return nullptr;
}

std::vector<IterDomain*> mergeDomains(
    const std::vector<IterDomain*>& domain1,
    const std::vector<IterDomain*>& domain2) {
  std::vector<IterDomain*> resulting_domain;
  auto it1 = domain1.begin();
  auto it2 = domain2.begin();

  // Need to merge domains together. These domains are representative of what's
  // within all the compute at positions of their respective groups (could be
  // many Exprs). The domains do not necessarily match, and we want to pull in
  // all iteration domains, maintaining relative ordering of both domains, while
  // removing as many duplicate iter domains (iter domains that map to eachother
  // through index map).
  while (it1 != domain1.end() || it2 != domain2.end()) {
    // no lint is for repeated branching, don't lint to avoid running any_of
    // when not necessary.
    if (it1 == domain1.end()) { // NOLINT
      // domain1 has all been pushed, finish pushing domain 2
      resulting_domain.push_back(*it2++);
    } else if (it2 == domain2.end()) { // NOLINT
      // domain2 has all been pushed, finish pushing domain 1
      resulting_domain.push_back(*it1++);
    } else if (GpuLower::current()->caLoopMap().areMapped(
                   *it1, *it2)) { // NOLINT
      resulting_domain.push_back(*it1);
      ++it1;
      ++it2;
    } else if (std::any_of(it1 + 1, domain1.end(), [&](IterDomain* id1) {
                 return GpuLower::current()->caLoopMap().areMapped(id1, *it2);
               })) { // NOLINT
      // Increment it1, as a later iter domain matches the current one in
      // domain2
      resulting_domain.push_back(*it1++);

    } else if (std::any_of(it2 + 1, domain2.end(), [&](IterDomain* id2) {
                 return GpuLower::current()->caLoopMap().areMapped(id2, *it1);
               })) { // NOLINT
      // Increment it2, as a later iter domain matches the current one in
      // domain1
      resulting_domain.push_back(*it2++);
    } else {
      // This should not be reachalble since the axes here only
      // include the shared axes between the two expr groups.
      TORCH_INTERNAL_ASSERT(false, "Should not be reachable.");
      resulting_domain.push_back(*it1++);
      resulting_domain.push_back(*it2++);
    }
  }
  return resulting_domain;
}

} // namespace

// Disconect group from neighbors, and return edges that were disconnected
std::unordered_set<ExprGroupConnections*> ExprSegmentationSorter::
    disconnectGroup(ExprGroup* group) {
  std::unordered_set<ExprGroupConnections*> removed_edges(
      group->producerEdges().begin(), group->producerEdges().end());

  for (auto edge : group->producerEdges()) {
    edge->from->removeConsumerEdge(edge);
  }

  for (auto edge : group->consumerEdges()) {
    edge->to->removeProducerEdge(edge);
  }

  group->clearProducerEdges();
  group->clearConsumerEdges();

  return removed_edges;
}

// TODO: This function may be sub optimial. If we find that an iteration domain
// matches later in the other domain, we will hold all other iteration domains
// until that one matches. There may be cases where duplicating that iteration
// domain, and moving on could be more efficient.
ExprGroup* ExprSegmentationSorter::makeMergedNode(
    ExprGroup* sg1,
    ExprGroup* sg2) {
  // Keep Expr's sorted in topological order.
  const auto producer = getProducer(sg1, sg2);
  const auto consumer = sg1 == producer ? sg2 : sg1;

  // Make the new joined node
  auto joined_groups = makeEmptyGroup();

  TORCH_INTERNAL_ASSERT(
      producer != nullptr,
      "Tried to merge expr's together that aren't neighbors.");

  joined_groups->exprs() = producer->exprs();
  joined_groups->exprs().insert(
      joined_groups->exprs().end(),
      consumer->exprs().begin(),
      consumer->exprs().end());

  auto producer_edges = getMergedProducerEdges(sg1, sg2);
  // Connect joined group to resulting neighbors
  for (auto& edge : producer_edges) {
    auto from = edge->from;
    auto val = edge->val;

    edges_.push_back(
        std::make_unique<ExprGroupConnections>(from, joined_groups, val));

    joined_groups->addProducerEdge(edges_.back().get());
    from->addConsumerEdge(edges_.back().get());
  }

  auto consumer_edges = getMergedConsumerEdges(sg1, sg2);

  for (auto& edge : consumer_edges) {
    auto to = edge->to;
    auto val = edge->val;

    edges_.push_back(
        std::make_unique<ExprGroupConnections>(joined_groups, to, val));
    joined_groups->addConsumerEdge(edges_.back().get());
    edge->to->addProducerEdge(edges_.back().get());
  }

  if (std::all_of(
          producer->consumerEdges().begin(),
          producer->consumerEdges().end(),
          [&consumer](ExprGroupConnections* connection) {
            return connection->to == consumer;
          })) {
    // If all consumers of producer were resolved (i.e. last consumer of
    // producer is the one we're merging with), don't forward the compute at
    // axes of producer
    joined_groups->payload()->ca_domains_ = consumer->payload()->ca_domains_;
  } else {
    // Merge all compute at domains of producer and consumer
    std::vector<IterDomain*> resulting_ca_axes =
        mergeDomains(sg1->payload()->ca_domains_, sg2->payload()->ca_domains_);
    joined_groups->payload()->ca_domains_ = resulting_ca_axes;
  }

  if (std::all_of(
          consumer->producerEdges().begin(),
          consumer->producerEdges().end(),
          [&producer](ExprGroupConnections* connection) {
            return connection->from == producer;
          })) {
    // If all producere edges were resolved (i.e. last producer of consumer is
    // the one we're merging with), don't forward the produce at axes of
    // consumer
    joined_groups->payload()->pa_domains_ = producer->payload()->pa_domains_;
  } else {
    // Merge all produce at domains of producer and consumer
    std::vector<IterDomain*> resulting_pa_axes =
        mergeDomains(sg1->payload()->pa_domains_, sg2->payload()->pa_domains_);

    joined_groups->payload()->pa_domains_ = resulting_pa_axes;
  }

  return joined_groups;
}

bool canReduceCA(ExprGroup* group) {
  IterDomain* g_last_id = nullptr;

  if (group->payload()->ca_domains_.size() > 0) {
    g_last_id = group->payload()->ca_domains_.back();
  }
  if (g_last_id == nullptr) {
    return false;
  }

  // Compute at can sometimes get in a strange position as the update rules are
  // not fool proof. All consumers should have a match to this groups inner most
  // compute at axis, otherwise it should be lowered.
  for (auto consumer_edge : group->consumerEdges()) {
    auto consumer = consumer_edge->to;
    bool has_match = false;
    for (auto c_id : consumer->payload()->pa_domains_) {
      if (GpuLower::current()->caLoopMap().areMapped(c_id, g_last_id)) {
        has_match = true;
        break;
      }
    }
    if (!has_match) {
      return true;
    }
  }

  return false;
}

bool canReducePA(ExprGroup* group) {
  IterDomain* g_last_id = nullptr;

  if (group->payload()->pa_domains_.size() > 0) {
    g_last_id = group->payload()->pa_domains_.back();
  }
  if (g_last_id == nullptr) {
    return false;
  }

  for (auto producer_edge : group->producerEdges()) {
    auto producer = producer_edge->from;
    for (auto p_id : producer->payload()->ca_domains_) {
      if (GpuLower::current()->caLoopMap().areMapped(p_id, g_last_id)) {
        return false;
      }
    }
  }

  return true;
}

// Update in between attempts to segment. This is called once no more groups
// can be merged together. Typically we will want to remove compute at groups
// that have finished being grouped together. However if no groups have been
// merged after we've done this, we may need to stop as we could have multiple
// disjoint groups that won't be merged.
bool ExprSegmentationSorter::interIterUpdate() {
  // Go through groups and lower either pa or ca domain return if anything was
  // lowered
  bool lowered_a_domain = false;
  for (auto& group : groups_) {
    while (canReduceCA(group.get())) {
      group->payload()->ca_domains_.pop_back();
      lowered_a_domain = true;
    }

    if (canReducePA(group.get())) {
      group->payload()->pa_domains_.pop_back();
      lowered_a_domain = true;
    }
  }

  // If we couldn't lower compute at domain any further, and we haven't merged
  // any new groups after fallback_mode_enabled_ has been turned on, make sure
  // we've finished successfully
  if (!lowered_a_domain && n_groups_ == groups_.size()) {
    // Make sure none of the groups are still connected, as that would mean we
    // should have been able to merge them.
    bool successfully_finished = std::all_of(
        groups_.begin(), groups_.end(), [](std::unique_ptr<ExprGroup>& sg) {
          return sg->producerEdges().empty() && sg->consumerEdges().empty();
        });
    if (successfully_finished) {
      return false;
    }
    // If we didn't finish and we tried the fallback, throw.
    TORCH_INTERNAL_ASSERT(
        !fallback_mode_enabled_,
        "Couldn't succcessfully sort out the fusion expressions. ",
        "There are remaining connections of the heirarchical segmentation which should have been ",
        "flattened to a single ordered group, or disjoint ordered groups.");
    // We didn't finish, but we haven't tried the fallback, try again with that.
    fallback_mode_enabled_ = true;
  }

  n_groups_ = groups_.size();
  // Not done, continue.
  return true;
}

void ExprSegmentationSorter::mergeNodes() {
  std::unordered_set<ExprGroup*> clean_up_groups;
  std::unordered_set<ExprGroupConnections*> clean_up_edges;

  while (!to_merge_.empty()) {
    auto group1 = *to_merge_.begin();
    auto group2 = group1->payload()->merge_with;
    to_merge_.erase(group1);
    to_merge_.erase(group2);
    clean_up_groups.emplace(group1);
    clean_up_groups.emplace(group2);
    makeMergedNode(group1, group2);
  }

  for (auto group : clean_up_groups) {
    auto disconnected_edges = disconnectGroup(group);
    clean_up_edges.insert(disconnected_edges.begin(), disconnected_edges.end());
  }

  edges_.remove_if([&](std::unique_ptr<ExprGroupConnections>& edge) {
    return clean_up_edges.find(edge.get()) != clean_up_edges.end();
  });

  groups_.remove_if([&](std::unique_ptr<ExprGroup>& group) {
    return clean_up_groups.find(group.get()) != clean_up_groups.end();
  });
}

bool ExprSegmentationSorter::supportedMerge(ExprGroup* sg1, ExprGroup* sg2) {
  auto producer_group = getProducer(sg1, sg2);
  auto consumer_group = sg1 == producer_group ? sg2 : sg1;

  if (producer_group->payload()->ca_domains_.size() <
      producer_group->payload()->pa_domains_.size()) {
    return false;
  }

  if (consumer_group->payload()->pa_domains_.size() <
      consumer_group->payload()->ca_domains_.size()) {
    return false;
  }

  auto producer_domain = producer_group->payload()->ca_domains_;
  auto consumer_domain = consumer_group->payload()->pa_domains_;

  if (producer_domain.empty() && consumer_domain.empty()) {
    return true;
  }

  if (producer_domain.empty() || consumer_domain.empty()) {
    return false;
  }

  return GpuLower::current()->caLoopMap().areMapped(
      producer_domain.back(), consumer_domain.back());
}

bool ExprSegmentationSorter::testStillDag(ExprGroup* sg1, ExprGroup* sg2) {
  std::deque<ExprGroup*> to_visit;
  std::unordered_set<ExprGroup*> visited;
  // Add consumers of sg1 if not sg2
  for (auto sg1_consumer_edge : sg1->consumerEdges()) {
    if (sg1_consumer_edge->to != sg2) {
      to_visit.emplace_back(sg1_consumer_edge->to);
    }
  }

  // Add consumers of sg2 if not sg1
  for (auto sg2_consumer_edge : sg2->consumerEdges()) {
    if (sg2_consumer_edge->to != sg1) {
      to_visit.emplace_back(sg2_consumer_edge->to);
    }
  }

  while (to_visit.size() > 0) {
    auto group = to_visit.front();
    // Arrived back at one of the original groups, merging these two groups
    // would generate a cycle
    if (group == sg1 || group == sg2) {
      return false;
    }
    to_visit.pop_front();
    if (visited.find(group) != visited.end()) {
      continue;
    }
    visited.emplace(group);
    for (auto consumer_edge : group->consumerEdges()) {
      to_visit.emplace_back(consumer_edge->to);
    }
  }

  // No cycles found, we're good.
  return true;
}

void ExprSegmentationSorter::sort() {
  // Need this for initialization of the DAG that is processed
  std::unordered_map<Expr*, ExprGroup*> expr2group;

  // Initialize DAG, convert each expr to a segment group
  for (auto expr : complete_fusion_->exprs()) {
    auto group = makeEmptyGroup(expr);
    expr2group.insert(std::make_pair(expr, group));
  }

  // Create edges between the Exprs. Mark inputs and outputs of the fusion.
  for (auto expr : complete_fusion_->exprs()) {
    auto expr_group = expr2group.at(expr);
    for (auto inp : expr->inputs()) {
      if (inp->isFusionInput()) {
        continue;
      }

      // Could be something like a constant scalar, definition is nullptr, but
      // isn't an "input" to the fusion. At least not one provided by an
      // external source.
      if (inp->definition() == nullptr) {
        continue;
      }

      auto def_group = expr2group.at(inp->definition());
      edges_.push_back(
          std::make_unique<ExprGroupConnections>(def_group, expr_group, inp));
      expr_group->addProducerEdge(edges_.back().get());
      def_group->addConsumerEdge(edges_.back().get());
    }
  }
  bool inter_iter_update = true;
  while (inter_iter_update) {
    // If we didn't do any update, stop traversal, we're done.
    if (!fallback_mode_enabled_) {
      // Merge expressions in sorted order
      bool merged_nodes = true;
      while (merged_nodes) {
        // Reset stateful traversal details in ExprGroups
        resetTraversal();
        resetLevels();

        for (auto& group : groups_) {
          if (group->payload()->merged) {
            continue;
          }
          auto candidates = group->getMergeCandidates(fallback_mode_enabled_);
          if (candidates.empty()) {
            continue;
          }

          auto candidate_it = candidates.begin();
          while (candidate_it != candidates.end() &&
                 !supportedMerge(group.get(), *candidate_it)) {
            candidate_it++;
          }
          if (candidate_it == candidates.end()) {
            continue;
          }

          to_merge_.emplace(group.get());
          to_merge_.emplace(*candidate_it);

          group->payload()->merged = true;
          group->payload()->merge_with = *candidate_it;

          (*candidate_it)->payload()->merged = true;
          (*candidate_it)->payload()->merge_with = group.get();
        }

        if (to_merge_.empty()) {
          merged_nodes = false;
        }

        mergeNodes();

        // Move compute at axes left
        inter_iter_update = interIterUpdate();
      }
    } else {
      // fallback_mode_enabled = true
      // Reset stateful traversal details in ExprGroups as we'll exclude merge
      // options that were already ruled out and therefore need traversal and
      // levels reset.
      resetTraversal();
      resetLevels();

      for (auto& group : groups_) {
        if (group->payload()->merged) {
          continue;
        }
        // Get merge candidates that weren't proven safe to merge with default
        // algorithm.
        auto candidates = group->getMergeCandidates(fallback_mode_enabled_);
        if (candidates.empty()) {
          continue;
        }

        auto candidate_it = candidates.begin();

        while (candidate_it != candidates.end()) {
          while (candidate_it != candidates.end() &&
                 !supportedMerge(group.get(), *candidate_it)) {
            candidate_it++;
          }

          if (candidate_it == candidates.end()) {
            break;
          }

          if (testStillDag(group.get(), *candidate_it)) {
            // Mark in same style as default algorithm for convenience even
            // though we will only merge once with the fallback
            to_merge_.emplace(group.get());
            to_merge_.emplace(*candidate_it);

            group->payload()->merged = true;
            group->payload()->merge_with = *candidate_it;

            (*candidate_it)->payload()->merged = true;
            (*candidate_it)->payload()->merge_with = group.get();
            break;
          }

          candidate_it++;
        }

        if (to_merge_.size() > 0) {
          break;
        }
      }

      // If we can merge something, merge it, disable fallback, and bail
      if (to_merge_.size() > 0) {
        mergeNodes();
      }

      // Move compute at axes left
      // If fallback didn't work, interIterUpdate will catch that we failed.
      inter_iter_update = interIterUpdate();
      fallback_mode_enabled_ = false;
    }
  }
}

std::vector<Expr*> ExprSegmentationSorter::getExprs() const {
  std::vector<Expr*> exprs;
  for (auto& group : groups_) {
    exprs.insert(exprs.end(), group->exprs().begin(), group->exprs().end());
  }
  return exprs;
}

} // namespace

std::vector<Expr*> reorderExprsForComputeAt() {
  auto fusion = FusionGuard::getCurFusion();
  TORCH_INTERNAL_ASSERT(fusion != nullptr);
  ExprSegmentationSorter sorter(fusion);
  sorter.sort();
  auto sorted_exprs = sorter.getExprs();
  TORCH_INTERNAL_ASSERT(
      sorted_exprs.size() > 0,
      "Error during expression sorting, no expressions produced.");
  return sorted_exprs;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
