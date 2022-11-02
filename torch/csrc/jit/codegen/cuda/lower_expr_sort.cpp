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
// std::ostream& operator<<(std::ostream& os, const ExprGroup* group);

// Wrapper for values, these are edges between expr groups. Multiple edges can
// exist between expr groups, and the same Val can show up more than once in
// multiple edges.
struct ExprGroupConnections {
  ExprGroupConnections(
      ExprGroup* group_from,
      ExprGroup* group_to,
      Val* producer_val,
      Val* consumer_val)
      : from(group_from),
        to(group_to),
        producer_val_(producer_val),
        consumer_val_(consumer_val) {}
  // Producer group from which the edge starts
  ExprGroup* from;

  // Consumer group from which the edge ends
  ExprGroup* to;

  // The value from the producer group connecting the groups
  // This value helps us resolve the compute at position of expr groups

  Val* producer_val_;

  // The value that the producer val gets used to create on this edge
  // This value helps us resolve the produce at position of expr groups
  Val* consumer_val_;
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
  ExprSegmentationSorter(Fusion* fusion) : fusion_(fusion) {}

  void sort();

  std::string toString(int verbosity = 0) const;

  //! Returns a flattened list of sorted exprs
  std::vector<Expr*> getExprs() const;

 private:
  // Allocate an empty expr group and return it
  ExprGroup* makeEmptyGroup();

  // Allocate an expr group with the provided expr and return it. Also requires
  // information on if this expression is a terminating expression (none of its
  // outputs are used in other expressions being sorted).
  ExprGroup* makeEmptyGroup(Expr*, bool terminating_expr);

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

  // Initialize concrete_id_dependencies
  void initializeForLoopDependencies();

  // Checks if the for loop associated with the concrete ID is ready to be
  // resolved in sorting.
  bool loopReady(IterDomain* concrete_id);

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

  std::vector<std::pair<ExprGroup*, ExprGroup*>> to_merge_;

  Fusion* fusion_;

  // We use a theorem out of a paper mentioned in other comments. This theorem
  // is good at identifying multiple expr groups to merge during a single
  // iteration without producing a cyclic graph from an acyclic graph. This
  // theorem is not guaranteed to find all possible nodes that can be merged
  // together. We need to be able to group all disjoint groups of exprs or
  // we fail to generate code. Therefore, if we can't find anything to make
  // forward progress based on the theorem we fallback to manually looking if we
  // can segmenet all combinations we haven't previously looked at.
  bool fallback_mode_enabled_ = false;

  // We need to track ID resolution, see AdvancedLowering6. For loops need
  // to be resolved from inner most to outer most. We therefore track
  // loop dependencies where inner most loops need to be fully resolved before
  // we can resolve the next outer loop. We track this by looking at all tensor
  // views, and each iteration domain. An iter domain in the outer most position
  // has dependencies on all inner dimensions. This tracking is done on concrete
  // id's in the loop map, this is because IDs may exist in some TVs but not
  // others, however, we need a "global" view to track these dependencies.
  std::unordered_map<IterDomain*, std::unordered_set<IterDomain*>>
      concrete_id_dependencies;
};

// // Debug printing, disabled due to clang-tidy see above for declarations.
// std::ostream& operator<<(std::ostream& os, ExprGroup* group) {
//   os << "Group Start{\n  ca, pa ("
//      << group->payload()->ca_domains_.size() << ", "
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
//   os << "\nExprs {\n";
//   for(auto expr : group->exprs()){
//     os << expr;
//   }
//    os << "}Group End\n";
//   return os;
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
  for (const auto i : c10::irange(neighbors.size())) {
    if (std::abs(neighbors[i]->payload()->level - payload()->level) > 1) {
      can_merge[i] = false;
    }
  }

  // Check neighbor of neighbors we're considering, if any of them are merged
  // with another node, make sure the resulting edge wouldn't have a level
  // difference of 1
  for (const auto i : c10::irange(neighbors.size())) {
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
  for (const auto i : c10::irange(neighbors.size())) {
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

ExprGroup* ExprSegmentationSorter::makeEmptyGroup(
    Expr* expr,
    bool terminating_expr) {
  auto group = makeEmptyGroup();
  group->exprs().push_back(expr);
  if (ir_utils::isTvOp(expr)) {
    auto out_tv = expr->outputs()[0]->as<TensorView>();
    // Grab all id's that are shared with other tensors.
    // If not connected to consumers, doesn't mater what compute at is set to
    if (!terminating_expr) {
      for (const auto tv_i : c10::irange(out_tv->getComputeAtPosition())) {
        group->payload()->ca_domains_.push_back(out_tv->axis(tv_i));
      }
    }
    for (const auto tv_i : c10::irange(out_tv->getMaxProducerPosition())) {
      group->payload()->pa_domains_.push_back(out_tv->axis(tv_i));
    }
  }
  return group;
}

// Debug function that prints the current state of the sorter.
//
// Uncomment if needed.
// std::string ExprSegmentationSorter::toString(int verbosity) const {
//   std::stringstream ss;
//   ss << "{\n";
//   for (auto& group : groups_) {
//     ss << "  " << group.get() << "\n";

//     if (verbosity > 1) {
//       if (group->producerEdges().size() > 0) {
//         ss << "Produced by groups with edges: { \n";
//         for (auto producer_edge : group->producerEdges()) {
//           ss << producer_edge->producer_val_ << " -> "
//              << producer_edge->consumer_val_ << "\n";
//         }
//         ss << "    }"
//            << "\n";
//       }
//     }

//     if (verbosity > 1) {
//       if (group->consumerEdges().size() > 0) {
//         ss << "Consumed by groups with edges: { \n";
//         for (auto consumer_edge : group->consumerEdges()) {
//           ss << consumer_edge->producer_val_ << " -> "
//              << consumer_edge->consumer_val_ << "\n";
//         }
//         ss << "    }"
//            << "\n";
//       }
//     }
//   }
//   ss << "}\n";
//   return ss.str();
// }

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

std::vector<IterDomain*> getLocalDomainOrdering(
    const std::vector<Expr*>& exprs,
    const std::unordered_set<IterDomain*> filter,
    const std::unordered_map<IterDomain*, std::unordered_set<IterDomain*>>&
        concrete_id_dependencies) {
  if (exprs.empty()) {
    return std::vector<IterDomain*>();
  }

  const auto& ca_map = GpuLower::current()->caMap();

  std::unordered_set<IterDomain*> domains;

  for (auto expr : exprs) {
    if (!ir_utils::isTvOp(expr)) {
      continue;
    }

    auto tv_inputs = ir_utils::filterByType<TensorView>(expr->inputs());
    for (auto tv_input : tv_inputs) {
      std::vector<IterDomain*> domain;

      std::transform(
          tv_input->domain()->domain().begin(),
          tv_input->domain()->domain().begin() +
              std::max(
                  tv_input->getComputeAtPosition(),
                  tv_input->getMaxProducerPosition()),
          std::back_inserter(domain),
          [&ca_map](IterDomain* id) {
            return ca_map->getConcreteMappedID(id, IdMappingMode::LOOP);
          });

      domain.erase(
          std::remove_if(
              domain.begin(),
              domain.end(),
              [&filter, &ca_map](IterDomain* id) {
                return filter.find(ca_map->getConcreteMappedID(
                           id, IdMappingMode::LOOP)) == filter.end();
              }),
          domain.end());

      domains.insert(domain.begin(), domain.end());
    }
  }

  std::vector<IterDomain*> merged_domain(domains.begin(), domains.end());
  std::sort(
      merged_domain.begin(),
      merged_domain.end(),
      ir_utils::IterDomainDependencySorter(
          concrete_id_dependencies, GpuLower::current()->caMap()));
  return merged_domain;
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
    auto producer_val = edge->producer_val_;
    auto consumer_val = edge->consumer_val_;

    edges_.push_back(std::make_unique<ExprGroupConnections>(
        from, joined_groups, producer_val, consumer_val));

    joined_groups->addProducerEdge(edges_.back().get());
    from->addConsumerEdge(edges_.back().get());
  }

  auto consumer_edges = getMergedConsumerEdges(sg1, sg2);

  for (auto& edge : consumer_edges) {
    auto to = edge->to;
    auto producer_val = edge->producer_val_;
    auto consumer_val = edge->consumer_val_;

    edges_.push_back(std::make_unique<ExprGroupConnections>(
        joined_groups, to, producer_val, consumer_val));
    joined_groups->addConsumerEdge(edges_.back().get());
    edge->to->addProducerEdge(edges_.back().get());
  }

  // Merge the compute at domain of all edges going out from the newly joined
  // group. The val's we're looking for are from our consumer edges, but we want
  // to grab the producer val as that's the one we generate.
  std::unordered_set<IterDomain*> ca_ids;
  for (auto consumer_group_edge : joined_groups->consumerEdges()) {
    auto producer_of_consumer_edge = consumer_group_edge->producer_val_;
    if (producer_of_consumer_edge->isA<TensorView>()) {
      auto tv = producer_of_consumer_edge->as<TensorView>();
      for (const auto tv_i : c10::irange(tv->getComputeAtPosition())) {
        ca_ids.emplace(GpuLower::current()->caMap()->getConcreteMappedID(
            tv->axis(tv_i), IdMappingMode::LOOP));
      }
    }
  }

  // Merge the produce at domain of all edges coming into the newly joined
  // group. The val's we're looking for are from our producer edges, but we want
  // to grab the consumer val as that's the one we generate.
  std::unordered_set<IterDomain*> pa_ids;
  for (auto producer_group_edge : joined_groups->producerEdges()) {
    auto consumer_of_producer_edge = producer_group_edge->consumer_val_;
    if (consumer_of_producer_edge->isA<TensorView>()) {
      auto tv = consumer_of_producer_edge->as<TensorView>();
      for (const auto tv_i : c10::irange(tv->getMaxProducerPosition())) {
        pa_ids.emplace(GpuLower::current()->caMap()->getConcreteMappedID(
            tv->axis(tv_i), IdMappingMode::LOOP));
      }
    }
  }

  auto all_ca_pa_ids = ca_ids;
  all_ca_pa_ids.insert(pa_ids.begin(), pa_ids.end());

  auto ordered_ids = getLocalDomainOrdering(
      joined_groups->exprs(), all_ca_pa_ids, concrete_id_dependencies);

  for (auto id : ordered_ids) {
    if (ca_ids.count(id)) {
      joined_groups->payload()->ca_domains_.emplace_back(id);
    }
    if (pa_ids.count(id)) {
      joined_groups->payload()->pa_domains_.emplace_back(id);
    }
  }

  return joined_groups;
}

bool canReducePA(ExprGroup* group) {
  if (group->payload()->pa_domains_.empty()) {
    return false;
  }

  IterDomain* group_pa_last_id = group->payload()->pa_domains_.back();

  // Look through producer edges to see if we can reduce our produce at domain
  for (auto producer_edge : group->producerEdges()) {
    auto producer_val = producer_edge->producer_val_;
    auto consumer_val = producer_edge->consumer_val_;

    // If producer isn't a tensor view it can't be mapped into a producer dim of
    // this group
    if (!(consumer_val->isA<TensorView>() && producer_val->isA<TensorView>())) {
      continue;
    }

    // If the compute at domains of the producer group is empty, it can't map to
    // the produce at domains of this group
    auto producer_group = producer_edge->from;
    if (producer_group->payload()->ca_domains_.empty()) {
      continue;
    }

    auto producer_tv = producer_val->as<TensorView>();
    auto consumer_tv = consumer_val->as<TensorView>();

    // If this consumer_tv doesn't map to the last producer domain of this group
    // it can't decide if it can be reduced
    bool has_matching_pa = false;
    for (const auto i : c10::irange(consumer_tv->getMaxProducerPosition())) {
      if (GpuLower::current()->caMap()->areMapped(
              consumer_tv->axis(i), group_pa_last_id, IdMappingMode::LOOP)) {
        has_matching_pa = true;
        break;
      }
    }

    if (!has_matching_pa) {
      continue;
    }

    // If any compute at positions of producers directly map to the last produce
    // at position it can't be lowered.
    for (int producer_pos_i =
             static_cast<int>(producer_tv->getComputeAtPosition());
         producer_pos_i > 0;
         producer_pos_i--) {
      if (GpuLower::current()->caMap()->areMapped(
              producer_tv->axis(producer_pos_i - 1),
              group_pa_last_id,
              IdMappingMode::LOOP)) {
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
    ExprGroup *group1 = nullptr, *group2 = nullptr;
    std::tie(group1, group2) = to_merge_.back();
    to_merge_.pop_back();
    TORCH_INTERNAL_ASSERT(
        group2 == group1->payload()->merge_with,
        "Expression Sorter: inconsistent to_merge packing");
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

// Initialize concrete_id_dependencies and concrete_id_to_all_ids
void ExprSegmentationSorter::initializeForLoopDependencies() {
  TORCH_INTERNAL_ASSERT(
      concrete_id_dependencies.empty(),
      "For loop dependencies have already been initialized.");

  for (auto tv : ir_utils::allTvs(fusion_)) {
    std::unordered_set<IterDomain*> dependencies;
    for (size_t tv_id_i =
             std::max(tv->getMaxProducerPosition(), tv->getComputeAtPosition());
         tv_id_i > 0;
         tv_id_i--) {
      auto tv_id = tv->axis((int)(tv_id_i - 1));
      auto concrete_id = GpuLower::current()->caMap()->getConcreteMappedID(
          tv_id, IdMappingMode::LOOP);

      if (concrete_id_dependencies.find(concrete_id) ==
          concrete_id_dependencies.end()) {
        concrete_id_dependencies[concrete_id] = dependencies;
      } else {
        concrete_id_dependencies[concrete_id].insert(
            dependencies.begin(), dependencies.end());
      }

      // Loops after tv_id are dependent on tv_id
      dependencies.emplace(GpuLower::current()->caMap()->getConcreteMappedID(
          tv_id, IdMappingMode::LOOP));
    }
  }

  // Fill out dependencies as IDs will have local dependency information, but
  // it's still not guaranteed to be global.

  // If loop structure is something like:
  // T0 [I0]
  // T1 [I0, I1]
  // T2 [I1, I2]
  //
  // I1 will be marked as a dependency of I0
  // I2 will be marked as a dependency of I1
  //
  // However, I2 will not be marked as a dep of I0, so we need to fill out the
  // dependency analysis. This is done by iterating through IterDomains filling
  // out all the dependencies of dependencies recursively.

  std::deque<IterDomain*> to_visit;
  std::unordered_set<IterDomain*> visited;

  std::transform(
      concrete_id_dependencies.begin(),
      concrete_id_dependencies.end(),
      std::back_inserter(to_visit),
      [](const auto& concrete_dep_entry) { return concrete_dep_entry.first; });

  size_t inf_loop_counter = to_visit.size();
  bool failed = false;

  while (!to_visit.empty()) {
    auto id = to_visit.front();
    to_visit.pop_front();

    if (inf_loop_counter-- == 0) {
      failed = true;
      break;
    }

    auto& dependencies = concrete_id_dependencies.at(id);
    bool ready = dependencies.empty() ||
        std::all_of(dependencies.begin(),
                    dependencies.end(),
                    [&visited](IterDomain* id) { return visited.count(id); });

    if (!ready) {
      to_visit.push_back(id);
      continue;
    }

    inf_loop_counter = to_visit.size();

    for (auto dependency : dependencies) {
      auto dep_of_dep = concrete_id_dependencies.at(dependency);
      dependencies.insert(dep_of_dep.begin(), dep_of_dep.end());
    }
    visited.emplace(id);
  }
  if (failed) {
    std::cerr
        << "ERROR: Iteration domain sorting has failed, infinite loop detected."
        << std::endl;
    std::cerr << "Failed to sort out: " << std::endl;
    for (auto entry : to_visit) {
      std::cerr << entry->toString();
      if (entry != to_visit.back()) {
        std::cerr << ", ";
      }
    }

    std::cerr << "Depdencies: " << std::endl;
    for (const auto& dep_entry : concrete_id_dependencies) {
      std::cerr << "  Deps of " << dep_entry.first->toString() << std::endl
                << "   ";

      for (auto dep : dep_entry.second) {
        std::cerr << dep->toString() << ", ";
      }
      std::cerr << std::endl;
    }

    TORCH_INTERNAL_ASSERT(false);
  }
}

// Checks if the for loop associated with the concrete ID is ready to be
// resolved in sorting. This could be done more efficiently with some
// additional tracking, however we recreate ca_domain_ when we merge groups,
// so it's hard to track what is no longer needed.
bool ExprSegmentationSorter::loopReady(IterDomain* concrete_id) {
  const auto& dependencies = concrete_id_dependencies[concrete_id];
  for (auto& group : groups_) {
    // Only need to check compute at domain here, because if there's an entry in
    // produce at, that has no matching entry in compute at, then that ID can be
    // removed as in canReducePA
    for (auto ca_domain : group->payload()->ca_domains_) {
      if (dependencies.count(ca_domain)) {
        return false;
      }
    }
  }
  return true;
}

// Two expression groups can be merged together if there's a value produced by
// producer group, consumed by consumer group, where the compute at position
// maps to the inner most compute at domain of the producer group and maps to
// the inner most produce at domain of the consumer. If this value doesn't exist
// we can't be certain these domains share the "next" inner most loop.
//
// We're looking for this because we're starting at the inner most loops of all
// expressions, and looking for neighboring expressions that share inner loops.
// Once we've found all the inner most loops that expressions share, we merge
// them together, then look at the next inner most loop of the group and figure
// out which other groups share this next inner most loop.
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

  const auto& producer_ca_domain = producer_group->payload()->ca_domains_;
  const auto& consumer_pa_domain = consumer_group->payload()->pa_domains_;

  if (producer_ca_domain.empty() && consumer_pa_domain.empty()) {
    return true;
  }

  if (producer_ca_domain.empty() || consumer_pa_domain.empty()) {
    return false;
  }

  // If inner loop dependencies have not been resolved, cannot merge.
  if (!loopReady(producer_ca_domain.back()) ||
      !loopReady(consumer_pa_domain.back())) {
    return false;
  }

  for (auto edge : producer_group->consumerEdges()) {
    if (edge->to != consumer_group) {
      continue;
    }
    auto producer_val = edge->producer_val_;
    auto consumer_val = edge->consumer_val_;

    if (!producer_val->isA<TensorView>()) {
      continue;
    }

    TORCH_INTERNAL_ASSERT(
        consumer_val->isA<TensorView>(),
        "Mismatched tensorview to non-tensorview in expression sorting. ",
        producer_val,
        " is consumed by ",
        consumer_val);

    auto producer_tv = producer_val->as<TensorView>();

    auto compute_at_pos = producer_tv->getComputeAtPosition();
    auto compute_at_dim = compute_at_pos > 0
        ? producer_tv->axis((int)producer_tv->getComputeAtPosition() - 1)
        : nullptr;

    if (compute_at_dim == nullptr) {
      continue;
    }

    if (!GpuLower::current()->caMap()->areMapped(
            compute_at_dim, producer_ca_domain.back(), IdMappingMode::LOOP)) {
      continue;
    }

    if (GpuLower::current()->caMap()->areMapped(
            compute_at_dim, consumer_pa_domain.back(), IdMappingMode::LOOP)) {
      return true;
    }
  }
  return false;
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

  auto all_exprs = fusion_->exprs();

  // Figure out all the values used as inputs to the expressions we're sorting
  // (to find terminating expressions). There could be branches of expressions
  // not used to produce outputs, so can't simply check val->uses() to figure
  // out if it's actually used in the expressions we're sorting.
  std::unordered_set<Val*> used_vals;
  for (auto expr : all_exprs) {
    used_vals.insert(expr->inputs().begin(), expr->inputs().end());
  }

  // Initialize DAG, convert each expr to a segment group
  for (auto expr : all_exprs) {
    bool is_terminating_expr = std::none_of(
        expr->outputs().begin(),
        expr->outputs().end(),
        [&used_vals](Val* output) { return used_vals.count(output) > 0; });
    auto group = makeEmptyGroup(expr, is_terminating_expr);
    expr2group.insert(std::make_pair(expr, group));
  }

  // Create edges between the Exprs. Mark inputs and outputs of the fusion.
  for (auto expr : fusion_->exprs()) {
    auto expr_group = expr2group.at(expr);
    auto out = expr->outputs()[0];
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

      auto inp_def_group = expr2group.at(inp->definition());
      edges_.push_back(std::make_unique<ExprGroupConnections>(
          inp_def_group, expr_group, inp, out));
      expr_group->addProducerEdge(edges_.back().get());
      inp_def_group->addConsumerEdge(edges_.back().get());
    }
  }

  // Initialize loop dependency maps
  initializeForLoopDependencies();

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

          to_merge_.emplace_back(std::make_pair(group.get(), *candidate_it));

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
            to_merge_.emplace_back(std::make_pair(group.get(), *candidate_it));

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
  if (fusion->exprs().empty()) {
    return {};
  }
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
