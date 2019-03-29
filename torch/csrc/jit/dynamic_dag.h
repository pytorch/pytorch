#pragma once

#include <algorithm>
#include <iostream>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include <ATen/core/functional.h>
#include <torch/csrc/utils/memory.h>

namespace torch {
namespace jit {
namespace detail {

// DynamicDAG is a simple directed acyclic graph that dynamically maintains a
// topological order as edges/vertices are added and removed.
//
// [Example applications]
// - Let's say you have a DAG where each vertex is black or red. How do we
//   merge black nodes that are directly connected by contracting the
//   edge between them while still maintaining the DAG and a topological order?
//   Use contractEdge().
// - Let's say you have a DAG where each vertex is a Node* and the edges
//   represent data dependencies. We wish to determine if adding a new Node*
//   with certain data dependencies (or moving an existing one to use new
//   dependencies) is valid. Use DynamicDAG::addEdge() to add the new data
//   dependencies to the DAG: it will either find a valid reordering of the
//   DAG's topological order or throw if the resulting DAG is invalid.
//
// The implementation is based off of the PK algorithm in the following paper:
// "A Dynamic Topsort Algorithm for Directed Acyclic Graphs"
// by David Pearce and Paul Kelly
// https://www.doc.ic.ac.uk/~phjk/Publications/DynamicTopoSortAlg-JEA-07.pdf
// It is summarized in [Edge addition] (see DynamicDAG<T>::addEdge)

template <typename T>
struct Vertex;
template <typename T>
struct DynamicDAG;
template <typename T>
using vertex_list = std::vector<Vertex<T>*>;
template <typename T>
using unique_vertex = std::unique_ptr<Vertex<T>>;

enum class DFSDirection { forward, backward };

// Used to represent adjacency lists in DynamicDAG.
// Has set semantics: stores distinct elements.
//
// Because our graphs shouldn't fan out or in very much,
// we use std::vector<Vertex<T>*> to record edges.
// In all of the complexity analysis it is assumed that
// inserting, erasing, and finding take constant time.
template <typename T>
struct vertex_set {
  using iterator = typename vertex_list<T>::iterator;
  using reverse_iterator = typename vertex_list<T>::reverse_iterator;

  // returns if we inserted v into the set.
  bool insert(Vertex<T>* v) {
    if (contains(v)) {
      return false;
    } else {
      data_.push_back(v);
      return true;
    }
  }
  void erase(Vertex<T>* v) {
    data_.erase(std::find(data_.begin(), data_.end(), v));
  }
  bool contains(Vertex<T>* v) const {
    return std::find(data_.begin(), data_.end(), v) != data_.end();
  }
  void sort() {
    std::sort(data_.begin(), data_.end(), [](Vertex<T>* a, Vertex<T>* b) {
      return a->ord < b->ord;
    });
  }
  size_t size() const {
    return data_.size();
  }
  iterator begin() {
    return data_.begin();
  }
  iterator end() {
    return data_.end();
  }
  reverse_iterator rbegin() {
    return data_.rbegin();
  }
  reverse_iterator rend() {
    return data_.rend();
  }

 private:
  std::vector<Vertex<T>*> data_;
};

template <typename T>
struct IOEdges {
  vertex_set<T> in_edges;
  vertex_set<T> out_edges;
};

// Simple RAII wrapper around a vertex_list<T>.
// When adding a vertex to the list, mark it as visited.
// Clears the visited flag of each vertex in the vertex_list on deletion.
template <typename T>
struct visited_list {
  ~visited_list() {
    for (auto* v : data_) {
      v->visited_ = false;
    }
  }

  void push_back(Vertex<T>* elt) {
    AT_ASSERT(!elt->visited_);
    elt->visited_ = true;
    data_.push_back(elt);
  }

  void sort() {
    std::sort(data_.begin(), data_.end(), [](Vertex<T>* a, Vertex<T>* b) {
      return a->ord < b->ord;
    });
  }

  const vertex_list<T>& vector() {
    return data_;
  }

 private:
  vertex_list<T> data_;
};

template <typename T>
struct Vertex {
  Vertex(size_t ord, T datum) : ord(ord), visited_(false) {
    data.push_back(datum);
  }

  std::vector<T> data;
  size_t ord; // unique topological index

  std::string toString();
  vertex_set<T>& in_edges() {
    return edges_.in_edges;
  }
  vertex_set<T>& out_edges() {
    return edges_.out_edges;
  }
  IOEdges<T>&& move_edges() {
    return std::move(edges_);
  }

  bool visited() {
    return visited_;
  }

 private:
  IOEdges<T> edges_;

  friend visited_list<T>;
  bool visited_; // If this vertex has been visited
};

template <typename T>
struct DynamicDAG {
  Vertex<T>* newVertex(T datum);
  IOEdges<T> removeVertex(Vertex<T>* v);

  void addEdge(Vertex<T>* producer, Vertex<T>* consumer);
  void removeEdge(Vertex<T>* producer, Vertex<T>* consumer);
  bool contractEdge(Vertex<T>* producer, Vertex<T>* consumer);

  // max_size() >= the number of live vertices.
  // for all vertices v, v.ord < max_size()
  size_t max_size() const {
    return vertices_.size();
  };
  c10::optional<Vertex<T>*> at(size_t ord) const;

  std::string toString();

  // Use for debugging. Don't call these often.
  size_t debugNumVertices() const;
  void debugCheckInvariants();

 private:
  void mergeProducerIntoConsumer(Vertex<T>* producer, Vertex<T>* consumer);
  void mergeConsumerIntoProducer(Vertex<T>* producer, Vertex<T>* consumer);
  void reorder(visited_list<T> deltaF, visited_list<T> deltaB);
  bool contractionProducesCycle(Vertex<T>* producer, Vertex<T>* consumer);
  bool dfsSearch(
      DFSDirection direction,
      Vertex<T>* start,
      Vertex<T>* end,
      size_t bound,
      visited_list<T>& visited);

  // Store vertices indexed by their topological order.
  // If a vertex v has ord 5, then it can be found at vertices_[5].
  // There may be gaps in vertices_; this is to enable fast deletion.
  std::vector<unique_vertex<T>> vertices_;
};

// O(vertices_.size()). Used for testing, don't call this often.
template <typename T>
size_t DynamicDAG<T>::debugNumVertices() const {
  return std::count_if(
      vertices_.begin(), vertices_.end(), [](const unique_vertex<T>& v) {
        if (v)
          return true;
        return false;
      });
}

template <typename T>
Vertex<T>* DynamicDAG<T>::newVertex(T datum) {
  vertices_.push_back(torch::make_unique<Vertex<T>>(vertices_.size(), datum));
  return vertices_.back().get();
}

template <typename T>
void DynamicDAG<T>::removeEdge(Vertex<T>* producer, Vertex<T>* consumer) {
  AT_ASSERT(producer != consumer);
  AT_ASSERT(producer->out_edges().contains(consumer));
  AT_ASSERT(consumer->in_edges().contains(producer));
  producer->out_edges().erase(consumer);
  consumer->in_edges().erase(producer);
}

template <typename T>
void DynamicDAG<T>::debugCheckInvariants() {
  for (size_t ord = 0; ord < vertices_.size(); ++ord) {
    const auto& vertex = vertices_.at(ord);
    if (!vertex)
      continue;

    AT_ASSERTM(vertex->ord == ord, toString());
    for (auto* v : vertex->in_edges()) {
      AT_ASSERTM(v->ord < ord, toString());
    }
    for (auto* v : vertex->out_edges()) {
      AT_ASSERTM(v->ord > ord, toString());
    }
  }
}

template <typename T>
c10::optional<Vertex<T>*> DynamicDAG<T>::at(size_t ord) const {
  const auto& vertex = vertices_.at(ord);
  if (!vertex) {
    return c10::nullopt;
  } else {
    return vertex.get();
  }
}

template <typename T>
IOEdges<T> DynamicDAG<T>::removeVertex(Vertex<T>* v) {
  for (auto* parent : v->in_edges()) {
    parent->out_edges().erase(v);
  }
  for (auto* child : v->out_edges()) {
    child->in_edges().erase(v);
  }
  auto edges = v->move_edges();
  vertices_[v->ord] = nullptr;
  return edges;
}

/*
 * [Edge addition]
 * When adding an edge x -> y,
 * - if ord(x) < ord(y), don't do anything.
 * - if ord(y) < ord(x), some graph reordering must occur.
 *
 * Assume we are adding an edge x -> y and that ord(x) > ord(y).
 * First, if there is a path y ----> x through some other vertices, then this
 * edge addition would create a cycle. Figure this out via DFS and throw if
 * necessary.
 *
 * Now, consider the set of all vertices v such that ord(x) > ord(v) > ord(y).
 * Call this set the affected region (AR) -- these are the only vertices we
 * need to consider for reordering to make the resulting graph valid.
 *
 * Find all children of y (through DFS) in AR (call this set deltaF and add y to
 * it) Find all parents of x in AR (call this set deltaB and add x to it).
 *
 * Move y and all the children of y to after x and all the parents of x. The
 * result topological ordering is valid.
 *
 * [Visual algorithm reference]
 * Higher nodes come earlier in topological order.
 * We are adding an edge between x -> y.
 * The topological ordering is e, y, c, a, d, b, x, f.
 * The affected region is {y, c, a, d, b, x}. e and f cannot be involved
 * in the reorder.
 *
 *           (e)       <- ord = 0 ->                   (e)
 *            |                                         |
 *            v                                         v
 *           (y)       <- ord = 1 ->        \          (c)
 *            ^ \                       -----\          |
 *        (c) |  v     <- ord = 2 ->    -----/     (d)  v
 *          \ | (a)    <- ord = 3 ->        /       \->(x)
 *           ||  |                                      /\
 *       (d) ||  |     <- ord = 4 ->              (y)<-/  \
 *        |  ||  v                                 \       |
 *        \  v| (b)    <- ord = 5 ->                \->(a) |
 *         ->(x)       <- ord = 6 ->             (b)<--/   v
 *            \->(f)   <- ord = 7 ->                      (f)
 *
 * We find all children of y in the affected region. deltaF = {y, a, b}
 * We find all parents of x via DFS. deltaB = {c, d, x}
 *
 * Now, we reorder all vertices in deltaB to come before deltaF. This is
 * a little involved and happens in four steps:
 *
 * 1) sort all vertices in deltaB, and all vertices in deltaF.
 * deltaB (sorted) = {c(2), d(4), x(6)}. deltaB ords = { 2, 4, 6 }
 * deltaF (sorted) = {y(1), a(3), b(5)}. deltaF ords = { 1, 3, 5 }
 *
 * 2) append the two lists: the result is the order we want these vertices to
 *    have.
 * L = {c(2), d(4), x(6), y(1), a(3), b(5)}.
 *
 * 3) Merge the sorted ords: R = { 1, 2, 3, 4, 5, 6 }.
 *
 * 4) Reassign the vertices in L in order with the sorted ords.
 * We always use the vertices in deltaB, then deltaF, in that order.
 * L = { c(1), d(2), x(3), y(4) a(5), b(6) }
 *
 * This produces th graph shown on the right.
 *
 * [Analysis]
 * This is O(|AR| log |AR|). |AR| is equal to ord(consumer) - ord(producer).
 * AR is the "affected region": { v s.t. ord(v) in [ord(producer),
 * ord(consumer)] } consisting of the only vertices that can possibly be moved
 * around due to this edge addition.
 *
 * NB: Pearce and Kelly give a complexity bound of <<delta>> where
 * delta = union(deltaF, deltaB) and <<S>> on a set S is
 * <<S>> = |S| + |edges out of vertices of S| + |edges into vertices of S|.
 */
template <typename T>
void DynamicDAG<T>::addEdge(Vertex<T>* producer, Vertex<T>* consumer) {
  AT_ASSERT(producer != consumer);

  // NB: DynamicDAG is a simple graph. If an edge exists already, don't do
  // anything.
  bool is_distinct = producer->out_edges().insert(consumer);
  if (!is_distinct)
    return;
  is_distinct = consumer->in_edges().insert(producer);
  AT_ASSERT(is_distinct);

  if (producer->ord <= consumer->ord) {
    // topological ordering is already consistent, no need to update.
    return;
  }

  visited_list<T> deltaF;
  visited_list<T> deltaB;

  // Search for vertices that are reachable from consumer that have a now
  // incorrect topological ordering.
  if (dfsSearch(
          DFSDirection::forward,
          consumer,
          producer,
          /*bound=*/producer->ord,
          deltaF)) {
    // Path found! This means there's a cycle.
    AT_ERROR("Cycle detected while trying to add edge.");
  }

  // Search for vertices that can reach producer that have a now incorrect
  // topological ordering
  AT_ASSERT(!dfsSearch(
      DFSDirection::backward,
      producer,
      consumer,
      /*bound=*/consumer->ord,
      deltaB));

  // Reorder the vertices that are reachable from consumer to occur BEFORE
  // the vertices that can reach producer.
  reorder(std::move(deltaF), std::move(deltaB));
}

// Define the affected region for contractEdge(producer, consumer) as
// { v s.t. ord(v) in [ord(producer), ord(consumer)] }.
// These are the only vertices that can possibly be moved around
// during edge contraction.
//
// contractEdge is O(|AR| log |AR| * min(|out_edges(producer)|,
//                   |in_edges(consumer)|))
template <typename T>
bool DynamicDAG<T>::contractEdge(Vertex<T>* producer, Vertex<T>* consumer) {
  AT_ASSERT(producer != consumer);
  if (contractionProducesCycle(producer, consumer)) {
    return false;
  }

  removeEdge(producer, consumer);

  // Optimization: pick which order to merge depending on potential complexity.
  if (producer->out_edges().size() > consumer->in_edges().size()) {
    mergeConsumerIntoProducer(producer, consumer);
  } else {
    mergeProducerIntoConsumer(producer, consumer);
  }

  return true;
}

template <typename T>
void DynamicDAG<T>::mergeProducerIntoConsumer(
    Vertex<T>* producer,
    Vertex<T>* consumer) {
  // Optimization: we want to concat lists [producer.data, consumer.data].
  // Instead of inserting into the beginning of consumer.data, do a swap.
  producer->data.insert(
      producer->data.end(), consumer->data.begin(), consumer->data.end());
  std::swap(consumer->data, producer->data);

  auto edges = removeVertex(producer);

  // Each of these are constant b/c ord(consumer) > ord(producer) > ord(parent)
  // so the edge addition still preserves the existing topological order.
  for (auto* parent : edges.in_edges) {
    addEdge(parent, consumer);
  }

  // NB: each addEdge call is linear in (ord(consumer) - ord(child)).
  // This makes this function O(|out_edges(producer)| * |AR| log |AR|).
  for (auto* child : edges.out_edges) {
    addEdge(consumer, child);
  }
}

template <typename T>
void DynamicDAG<T>::mergeConsumerIntoProducer(
    Vertex<T>* producer,
    Vertex<T>* consumer) {
  producer->data.insert(
      producer->data.end(), consumer->data.begin(), consumer->data.end());

  auto edges = removeVertex(consumer);

  // Each of these are constant b/c ord(child) > ord(consumer) > ord(producer)
  // so the edge addition still preserves the existing topological order.
  for (auto* child : edges.out_edges) {
    addEdge(producer, child);
  }

  // NB: each addEdge call is linear in (ord(producer) - ord(parent)).
  // This makes this function O(|in_edges(consumer)| * |AR| log |AR|).
  for (auto* parent : edges.in_edges) {
    addEdge(parent, producer);
  }
}

template <typename T>
bool DynamicDAG<T>::contractionProducesCycle(
    Vertex<T>* producer,
    Vertex<T>* consumer) {
  visited_list<T> visited;

  // If there are multiple paths from producer to consumer then contracting
  // (merging) producer and consumer would create a cycle.
  //
  // Search for a path from producer to consumer while ignoring the
  // producer -> consumer edge.
  size_t upper_bound = consumer->ord;
  for (auto* child : producer->out_edges()) {
    if (child == consumer)
      continue;
    if (child->visited())
      continue; // already visited by dfs
    if (dfsSearch(
            DFSDirection::forward, child, consumer, upper_bound, visited)) {
      return true;
    }
  }
  return false;
}

static bool is_within_bound(
    DFSDirection direction,
    size_t value,
    size_t bound) {
  if (direction == DFSDirection::forward) {
    return value < bound; // upper bound
  } else {
    return value > bound; // lower bound
  }
}

// Searches for a path from start to end via a forward or backward dfs.
// Returns if a path exists from start to end.
// In addition, dfsSearch inserts visited vertices into the visited list.
template <typename T>
bool DynamicDAG<T>::dfsSearch(
    DFSDirection direction,
    Vertex<T>* start,
    Vertex<T>* end,
    size_t bound,
    visited_list<T>& visited) {
  vertex_list<T> stack;

  auto visit = [&](Vertex<T>* v) {
    visited.push_back(v);
    stack.push_back(v);
  };

  visit(start);

  while (!stack.empty()) {
    auto* vertex = stack.back();
    stack.pop_back();

    auto& next_edges = (direction == DFSDirection::forward)
        ? vertex->out_edges()
        : vertex->in_edges();

    for (Vertex<T>* next : next_edges) {
      if (next == end) {
        // Path found from start to end.
        visit(next);
        return true;
      }
      if (!next->visited() && is_within_bound(direction, next->ord, bound)) {
        visit(next);
      }
    }
  }
  return false;
}

// Reorder deltaB vertices to occur before deltaF vertices.
template <typename T>
void DynamicDAG<T>::reorder(visited_list<T> deltaF, visited_list<T> deltaB) {
  deltaB.sort();
  deltaF.sort();

  const auto& deltaB_ = deltaB.vector();
  const auto& deltaF_ = deltaF.vector();

  size_t num_affected = deltaB_.size() + deltaF_.size();

  // Gather vertices in the desired order. They don't have correct ords yet.
  std::vector<unique_vertex<T>> desired_vertex_ordering;
  desired_vertex_ordering.reserve(num_affected);
  for (auto it = deltaB_.begin(); it != deltaB_.end(); ++it) {
    desired_vertex_ordering.push_back(std::move(vertices_.at((*it)->ord)));
  }
  for (auto it = deltaF_.begin(); it != deltaF_.end(); ++it) {
    desired_vertex_ordering.push_back(std::move(vertices_.at((*it)->ord)));
  }

  // Sort the ords by merging two already sorted lists into a large sorted list.
  // input (example): deltaB = { v(1), v(4), v(7) } ,
  //                  deltaF = { v(0), v(2), v(5) }.
  // output: { 0, 1, 2, 4, 5, 7 }.
  std::vector<size_t> gathered_ords;
  gathered_ords.reserve(num_affected);
  for (const auto* v : deltaB_) {
    gathered_ords.push_back(v->ord);
  }
  auto middle = gathered_ords.size();
  for (const auto* v : deltaF_) {
    gathered_ords.push_back(v->ord);
  }
  std::inplace_merge(
      gathered_ords.begin(),
      gathered_ords.begin() + middle,
      gathered_ords.end());

  // Return the vertices back into the vertices_ storage.
  for (size_t i = 0; i < num_affected; ++i) {
    desired_vertex_ordering[i]->ord = gathered_ords[i];
    vertices_[gathered_ords[i]] = std::move(desired_vertex_ordering[i]);
  }
}

template <typename T>
std::string DynamicDAG<T>::toString() {
  std::stringstream ss;
  for (auto& v : vertices_) {
    if (v) {
      ss << v->toString() << "\n";
    }
  }
  return ss.str();
}

template <typename T>
std::string Vertex<T>::toString() {
  std::stringstream ss;
  ss << "node(" << ord << ")\n";
  ss << "[";
  for (auto* c : in_edges()) {
    ss << c->ord << " ";
  }
  ss << "] -> {\n";
  for (auto& d : data) {
    if (std::is_pointer<T>::value) {
      ss << "  " << *d;
    } else {
      ss << "  " << d;
    }
  }
  ss << "} (" << ord << ") -> [";
  for (auto* c : out_edges()) {
    ss << c->ord << " ";
  }
  ss << "]\n";
  return ss.str();
}

} // namespace detail
} // namespace jit
} // namespace torch
