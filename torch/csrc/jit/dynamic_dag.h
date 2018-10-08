#pragma once

#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <vector>

#include "torch/csrc/utils/functional.h"

namespace torch { namespace jit { namespace detail {

// DynamicDAG is a DAG that dynamically maintains a topological order as
// edges/vertices are added and removed.
//
// [Example applications]
// - Let's say you have a DAG where each vertex is black or red. How do we
//   merge black nodes that are directly connected by contracting the
//   edge between them while still maintaining the DAG and a topological order?
//   Use contractEdge().
// - Let's say you have a DAG where each vertex is a Node* and the edges represent
//   data dependencies. We wish to determine if adding a new Node* with certain
//   data dependencies (or moving an existing one to use new dependencies) is valid.
//   Use DynamicDAG::addEdge() to add the new data dependencies to the DAG:
//   it will either find a valid reordering of the DAG's topological order or throw
//   if the resulting DAG is invalid.
//
// The implementation is based off of the PK algorithm in the following paper:
// "A Dynamic Topsort Algorithm for Directed Acyclic Graphs"
// by David Pearce and Paul Kelly
// https://www.doc.ic.ac.uk/~phjk/Publications/DynamicTopoSortAlg-JEA-07.pdf
// It is summarized in [Edge addition] (see DynamicDAG<T>::addEdge)

template <typename T> struct Vertex;
template <typename T> struct DynamicDAG;
template <typename T> using vertex_list = std::vector<Vertex<T>*>;
template <typename T> using unique_vertex = std::unique_ptr<Vertex<T>>;

enum DFSDirection {forward, backward};

// Because our graphs shouldn't fan out or in very much,
// we use std::vector<Vertex<T>*> to record edges.
// In all of the complexity analysis it is assumed that
// inserting, erasing, and finding take constant time.
template <typename T>
struct EdgeData {
  // These are both sorted in topological order because iterating
  // in topological order is useful.
  vertex_list<T> in_edges;
  vertex_list<T> out_edges;

  static void insert(vertex_list<T>& lst, Vertex<T>* v) {
    // Keep the list sorted.
    // We can do the same thing (binary search) for erase() and has()
    // but I'm not sure if it will perform better especially since
    // our lists should be small.
    lst.insert(std::upper_bound(lst.begin(), lst.end(), v,
          [](Vertex<T>* x, Vertex<T>* y) { return x->ord < y->ord; }), v);
  }

  static void erase(vertex_list<T>& lst, Vertex<T>* v) {
    lst.erase(std::find(lst.begin(), lst.end(), v));
  }

  static bool has(vertex_list<T>& lst, Vertex<T>* v) {
    return std::find(lst.begin(), lst.end(), v) != lst.end();
  }
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
    JIT_ASSERT(!elt->visited_);
    elt->visited_ = true;
    data_.push_back(elt);
  }

  vertex_list<T>& vector() { return data_; }

 private:
  vertex_list<T> data_;
};

template <typename T>
struct Vertex {
  Vertex(size_t ord, T datum)
  : ord(ord), visited_(false) { data.push_back(datum); }

  // Holds data.
  std::vector<T> data;
  size_t ord; // unique topological index

  std::string toString();
  vertex_list<T>& in_edges() { return edges_.in_edges; }
  vertex_list<T>& out_edges() { return edges_.out_edges; }

  bool visited() { return visited_; }

private:
  friend DynamicDAG<T>;
  EdgeData<T> edges_;

  friend visited_list<T>;
  bool visited_; // If this vertex has been visited
};

template <typename T>
struct DynamicDAG {
  Vertex<T>* newVertex(T datum);
  EdgeData<T> removeVertex(Vertex<T>* v);

  void addEdge(Vertex<T>* producer, Vertex<T>* consumer);
  void removeEdge(Vertex<T>* producer, Vertex<T>* consumer);
  bool contractEdge(Vertex<T>* producer, Vertex<T>* consumer);

  // size() >= the number of live vertices.
  // for all vertices v, v.ord < size()
  size_t size() const { return vertices_.size(); };
  at::optional<Vertex<T>*> at(size_t ord) const;

  void sort(vertex_list<T>& delta);
  std::string toString();

  // Use for debugging. Don't call these often.
  size_t debugNumVertices() const;
  void debugCheckInvariants();

 private:
  void mergeProducerIntoConsumer(Vertex<T>* producer, Vertex<T>* consumer);
  void mergeConsumerIntoProducer(Vertex<T>* producer, Vertex<T>* consumer);
  void reorder(vertex_list<T>& deltaF, vertex_list<T>& deltaB);
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
  return std::count_if(vertices_.begin(), vertices_.end(),
      [](const unique_vertex<T>& v) {
        if (v) return true;
        return false;
      });
}

template <typename T>
Vertex<T>* DynamicDAG<T>::newVertex(T datum) {
  vertices_.emplace_back(new Vertex<T>(vertices_.size(), datum));
  return vertices_.back().get();
}

template <typename T>
void DynamicDAG<T>::sort(vertex_list<T>& delta) {
  std::sort(delta.begin(), delta.end(), [](Vertex<T>* a, Vertex<T>* b) {
    return a->ord < b->ord;
  });
}

template <typename T>
void DynamicDAG<T>::removeEdge(Vertex<T>* producer, Vertex<T>* consumer) {
  JIT_ASSERT(producer != consumer);
  JIT_ASSERT(EdgeData<T>::has(producer->out_edges(), consumer));
  JIT_ASSERT(EdgeData<T>::has(consumer->in_edges(), producer));
  EdgeData<T>::erase(producer->out_edges(), consumer);
  EdgeData<T>::erase(consumer->in_edges(), producer);
}

template <typename T>
void DynamicDAG<T>::debugCheckInvariants() {
  for (size_t ord = 0; ord < vertices_.size(); ++ord) {
    const auto& vertex = vertices_.at(ord);
    if (!vertex) continue;

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
at::optional<Vertex<T>*> DynamicDAG<T>::at(size_t ord) const {
  const auto& vertex = vertices_.at(ord);
  if (!vertex) {
    return at::nullopt;
  } else {
    return vertex.get();
  }
}

template <typename T>
EdgeData<T> DynamicDAG<T>::removeVertex(Vertex<T>* v) {
  for (auto* parent : v->in_edges()) {
    EdgeData<T>::erase(parent->out_edges(), v);
  }
  for (auto* child : v->out_edges()) {
    EdgeData<T>::erase(child->in_edges(), v);
  }
  auto edges = std::move(v->edges_);
  vertices_[v->ord] = std::move(unique_vertex<T>(nullptr));
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
 * edge addition would create a cycle. Figure this out via DFS and throw if necessary.
 *
 * Now, consider the set of all vertices v such that ord(x) > ord(v) > ord(y).
 * Call this set the affected region (AR) -- these are the only vertices we
 * need to consider for reordering to make the resulting graph valid.
 *
 * Find all children of y (through DFS) in AR (call this set deltaF and add y to it)
 * Find all parents of x in AR (call this set deltaB and add x to it).
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
 * Now, we reorder all vertices in deltaB to come before deltaF.
 * result = {c, d, x, y, a, b}, and we assign these {1, 2, 3, 4, 5, 6}.
 *
 * [Analysis]
 * This is O(|AR|) which is the same thing as O(ord(consumer) - ord(producer))
 * AR is the "affected region": { v s.t. ord(v) in [ord(producer), ord(consumer)] }
 * consisting of the only verticies that can possibly be moved around due
 * to this edge addition.
 */
template <typename T>
void DynamicDAG<T>::addEdge(Vertex<T>* producer, Vertex<T>* consumer) {
  JIT_ASSERT(producer != consumer);
  if (EdgeData<T>::has(producer->out_edges(), consumer)) return;
  EdgeData<T>::insert(producer->out_edges(), consumer);
  EdgeData<T>::insert(consumer->in_edges(), producer);

  visited_list<T> deltaF;
  visited_list<T> deltaB;

  if (producer->ord <= consumer->ord) {
    // topological ordering is already consistent, no need to update.
    return;
  }

  // Search for vertices that are reachable from consumer that have a now incorrect
  // topological ordering.
  if (dfsSearch(DFSDirection::forward, consumer, producer,
                /*upper_bound=*/producer->ord, deltaF)) {
    // Path found! This means there's a cycle.
    AT_ERROR("Cycle detected while trying to add edge.");
  }

  // Search for vertices that can reach producer that have a now incorrect
  // topological ordering
  JIT_ASSERT(!dfsSearch(DFSDirection::backward, producer, consumer,
                        /*lower_bound=*/consumer->ord, deltaB));

  // Reorder the vertices that are reachable from consumer to occur BEFORE
  // the vertices that can reach producer.
  reorder(deltaF.vector(), deltaB.vector());
}

// Define the affected region for contractEdge(producer, consumer) as
// { v s.t. ord(v) in [ord(producer), ord(consumer)] }.
// These are the only vertices that can possibly be moved around
// during edge contraction.
//
// contractEdge is O(|AR| * |out_edges(producer)|)
template <typename T>
bool DynamicDAG<T>::contractEdge(Vertex<T>* producer, Vertex<T>* consumer) {
  JIT_ASSERT(producer != consumer);
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
void DynamicDAG<T>::mergeProducerIntoConsumer(Vertex<T>* producer, Vertex<T>* consumer) {
  // Optimization: we want to concat lists [producer.data, consumer.data].
  // Instead of inserting into the beginning of consumer.data, do a swap.
  producer->data.insert(producer->data.end(), consumer->data.begin(), consumer->data.end());
  std::swap(consumer->data, producer->data);

  auto edges = removeVertex(producer);

  // Each of these are constant b/c ord(consumer) > ord(producer) > ord(parent)
  // so the edge addition still preserves the existing topological order.
  for (auto* parent : edges.in_edges) {
    addEdge(parent, consumer);
  }

  // NB: each addEdge call is linear in (ord(consumer) - ord(child)).
  // This makes this function O(|out_edges(producer)| * |AR|).
  for (auto* child : edges.out_edges) {
    addEdge(consumer, child);
  }
}

template <typename T>
void DynamicDAG<T>::mergeConsumerIntoProducer(Vertex<T>* producer, Vertex<T>* consumer) {
  producer->data.insert(producer->data.end(), consumer->data.begin(), consumer->data.end());

  auto edges = removeVertex(consumer);

  // Each of these are constant b/c ord(child) > ord(consumer) > ord(producer)
  // so the edge addition still preserves the existing topological order.
  for (auto* child : edges.out_edges) {
    addEdge(producer, child);
  }

  // NB: each addEdge call is linear in (ord(producer) - ord(parent)).
  // This makes this function O(|in_edges(consumer)| * |AR|).
  for (auto* parent : edges.in_edges) {
    addEdge(parent, producer);
  }

}

template <typename T>
bool DynamicDAG<T>::contractionProducesCycle(Vertex<T>* producer, Vertex<T>* consumer) {
  visited_list<T> visited;

  // If there are multiple paths from producer to consumer then contracting
  // (merging) producer and consumer would create a cycle.
  //
  // Search for a path from producer to consumer while ignoring the
  // producer -> consumer edge.
  size_t upper_bound = consumer->ord;
  for (auto* child : producer->out_edges()) {
    if (child == consumer) continue;
    if (child->visited()) continue; // already visited by dfs
    if (dfsSearch(DFSDirection::forward, child, consumer, upper_bound, visited)) {
      return true;
    }
  }
  return false;
}


static bool is_within_bound(DFSDirection direction, size_t value, size_t bound) {
  if (direction == DFSDirection::forward) {
    return value < bound; // upper bound
  } else {
    return value > bound; // lower bound
  }
}

// Searches for a path from start to end via a forward or backward dfs.
// Returns if a path exists from start to end.
// In addition, dfsSearch marks vertices as visited and inserts them into visited.
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

    vertex_list<T>* next_edges = (direction == DFSDirection::forward) ?
      &vertex->out_edges() :
      &vertex->in_edges();

    for (Vertex<T>* next : *next_edges) {
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
void DynamicDAG<T>::reorder(vertex_list<T>& deltaF, vertex_list<T>& deltaB) {
  sort(deltaB);
  sort(deltaF);

  size_t num_affected = deltaB.size() + deltaF.size();

  // Gather vertices in the desired order. They don't have correct ords yet.
  std::vector<unique_vertex<T>> desired_vertex_ordering;
  desired_vertex_ordering.reserve(num_affected);
  for (auto it = deltaB.begin(); it != deltaB.end(); ++it) {
    desired_vertex_ordering.push_back(std::move(vertices_.at((*it)->ord)));
  }
  for (auto it = deltaF.begin(); it != deltaF.end(); ++it) {
    desired_vertex_ordering.push_back(std::move(vertices_.at((*it)->ord)));
  }

  // Sort the ords.
  // This is done through an O(num_affected) merge operation. For example,
  // deltaB = { 1, 4, 7 } , deltaF = {0, 2, 5 }.
  // These two lists already contain sorted ords; we just need to merge
  // them to create an ordering { 0, 1, 2, 4, 5, 7 } to assign the vertices.
  std::vector<size_t> sorted_ords;
  sorted_ords.reserve(num_affected);
  auto output = sorted_ords.begin();
  auto inputB = deltaB.begin();
  auto inputF = deltaF.begin();
  for (; inputB != deltaB.end(); ++output) {
    if (inputF == deltaF.end()) {
      *output = (*inputB)->ord;
      ++inputB;
      continue;
    }
    if ((*inputB)->ord < (*inputF)->ord) {
      *output = (*inputB)->ord;
      ++inputB;
    } else {
      *output = (*inputF)->ord;
      ++inputF;
    }
  }
  for (; inputF != deltaF.end(); ++inputF) {
    *output = (*inputF)->ord;
    ++output;
  }

  // Return the vertices back into the vertices_ storage.
  for (size_t i = 0; i < num_affected; ++i) {
    desired_vertex_ordering[i]->ord = sorted_ords[i];
    vertices_[sorted_ords[i]] = std::move(desired_vertex_ordering[i]);
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
    ss << "  " << d;
  }
  ss << "} ("<< ord << ") -> [";
  for (auto* c : out_edges()) {
    ss << c->ord << " ";
  }
  ss << "]\n";
  return ss.str();
}

}}}
