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
// The implementation is based off of the following paper:
// "A Dynamic Topsort Algorithm for Directed Acyclic Graphs"
// by David Pearce and Paul Kelly
// https://www.doc.ic.ac.uk/~phjk/Publications/DynamicTopoSortAlg-JEA-07.pdf

template <typename T> struct Vertex;

template <typename T>
using vertex_set = std::unordered_set<Vertex<T>*>;

template <typename T>
using vertex_list = std::vector<Vertex<T>*>;

template <typename T>
using maybe_vertex_list = std::vector<at::optional<Vertex<T>*>>;

template <typename T>
struct Vertex {
  std::vector<T> rdata; // stored in reverse order
  size_t ord; // index in topological ordering. Is unique.
  bool visited; // If this vertex has been visited
  vertex_set<T> in_edges;
  vertex_set<T> out_edges;

  std::string toString();
};

template <typename T>
struct DynamicDAG {
  ~DynamicDAG();

  Vertex<T>* newVertex(T datum);
  std::pair<vertex_set<T>,vertex_set<T>> removeVertex(Vertex<T>* v);

  void addEdge(Vertex<T>* producer, Vertex<T>* consumer);
  void removeEdge(Vertex<T>* producer, Vertex<T>* consumer);
  bool contractEdge(Vertex<T>* producer, Vertex<T>* consumer);

  bool contractionProducesCycle(Vertex<T>* producer, Vertex<T>* consumer);
  bool dfsForward(Vertex<T>* producer, size_t upper_bound, vertex_list<T>& visited);
  void dfsBackward(Vertex<T>* producer, size_t lower_bound, vertex_list<T>& visited);
  void clearVisited(vertex_list<T>& visited);

  void reorder(vertex_list<T>& deltaF, vertex_list<T>& deltaB);
  void sort(vertex_list<T>& delta);

  void checkInvariants();

  Vertex<T>* at(size_t ord) const { return vertices_.at(ord).value(); };
  const maybe_vertex_list<T>& maybe_vertices() const { return vertices_; };

  // Use for debugging. Don't call this often.
  size_t numVertices() const;

  std::string toString();

 private:

  // Store vertices indexed by their topological order.
  // If a vertex v has ord 5, then it can be found at vertices_[5].
  // There may be gaps in vertices_; this is to enable fast deletion.
  maybe_vertex_list<T> vertices_;

  // temp buffers used in the PK algorithm.
  // Their usage makes this data structure NOT thread safe.
  vertex_list<T> deltaF_;  // all vertices visited by dfsForward
  vertex_list<T> deltaB_;  // all vertices visited by dfsBackward
};

template <typename T>
DynamicDAG<T>::~DynamicDAG() {
  for (auto v : vertices_) {
    if (v.has_value()) {
      delete v.value();
    }
  }
}

// O(vertices_.size()). Used for testing, don't call this often.
template <typename T>
size_t DynamicDAG<T>::numVertices() const {
  return std::count_if(vertices_.begin(), vertices_.end(), [](at::optional<Vertex<T>*> v) {
      return v;
  });
}

template <typename T>
Vertex<T>* DynamicDAG<T>::newVertex(T datum) {
  auto* vertex = new Vertex<T>();
  vertex->rdata.push_back(datum);
  vertex->visited = false;
  vertex->ord = vertices_.size();
  vertices_.push_back(vertex);
  return vertex;
}

template <typename T>
void DynamicDAG<T>::removeEdge(Vertex<T>* producer, Vertex<T>* consumer) {
  JIT_ASSERT(producer != consumer);
  JIT_ASSERT(producer->out_edges.count(consumer));
  JIT_ASSERT(consumer->in_edges.count(producer));
  producer->out_edges.erase(consumer);
  consumer->in_edges.erase(producer);
}

template <typename T>
void DynamicDAG<T>::checkInvariants() {
  for (size_t ord = 0; ord < vertices_.size(); ++ord) {
    const auto& maybe_vertex = vertices_.at(ord);
    if (!maybe_vertex.has_value()) continue;
    auto* vertex = maybe_vertex.value();

    JIT_ASSERT(vertex->ord == ord);
    for (auto* v : vertex->in_edges) {
      JIT_ASSERT(v->ord < ord);
    }
    for (auto* v : vertex->out_edges) {
      JIT_ASSERT(v->ord > ord);
    }
  }
}


template <typename T>
std::pair<vertex_set<T>,vertex_set<T>> DynamicDAG<T>::removeVertex(Vertex<T>* v) {
  for (auto* parent : v->in_edges) {
    parent->out_edges.erase(v);
  }
  for (auto* child : v->out_edges) {
    child->in_edges.erase(v);
  }
  auto in_edges = std::move(v->in_edges);
  auto out_edges = std::move(v->out_edges);
  vertices_[v->ord] = at::nullopt;
  delete v;
  return std::make_pair<vertex_set<T>,vertex_set<T>>(std::move(in_edges), std::move(out_edges));
}

// This is O(|AR|) which is the same thing as O(ord(consumer) - ord(producer))
// AR is the "affected region": { v s.t. ord(v) in [ord(producer), ord(consumer)] }
// consisting of the only verticies that can possibly be moved around due
// to this edge addition.
template <typename T>
void DynamicDAG<T>::addEdge(Vertex<T>* producer, Vertex<T>* consumer) {
  JIT_ASSERT(producer != consumer);
  if (producer->out_edges.count(consumer)) return;
  producer->out_edges.insert(consumer);
  consumer->in_edges.insert(producer);

  if (producer->ord <= consumer->ord) {
    // topological ordering is already consistent, no need to update.
    return;
  }

  // Search for vertices that are reachable from consumer that have a now incorrect
  // topological ordering.
  if (!dfsForward(consumer, producer->ord, deltaF_)) {
    // dfsForward returns false if there is a path from consumer to producer.
    throw std::runtime_error("Cycle detected while trying to add edge.");
  }

  // Search for vertices that can reach producer that have a now incorrect
  // topological ordering
  dfsBackward(producer, consumer->ord, deltaB_);

  // Reorder the vertices that are reachable from consumer to occur BEFORE
  // the vertices that can reach producer.
  reorder(deltaF_, deltaB_);
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

  // We can either merge producer into consumer or the other way around.
  // I've chosen to merge producer into consumer for now but this is arbitrary.
  consumer->rdata.insert(consumer->rdata.end(), producer->rdata.begin(), producer->rdata.end());

  vertex_set<T> in_edges;
  vertex_set<T> out_edges;
  std::tie(in_edges, out_edges) = removeVertex(producer);

  // Each of these are constant b/c ord(consumer) > ord(producer) > ord(parent)
  // so the edge addition still preserves the existing topological order.
  for (auto* parent : in_edges) {
    addEdge(parent, consumer);
  }

  // NB: each addEdge call is linear in (ord(consumer) - ord(child)).
  // This makes contractEdges O(|out_edges(producer)| * |AR|).
  for (auto* child : out_edges) {
    addEdge(consumer, child);
  }
  return true;
}

template <typename T>
bool DynamicDAG<T>::contractionProducesCycle(Vertex<T>* producer, Vertex<T>* consumer) {
  // Basically a modified dfsForward
  vertex_list<T> stack;
  vertex_list<T> visited;
  stack.push_back(producer);

  size_t upper_bound = consumer->ord;

  while (!stack.empty()) {
    auto* vertex = stack.back();
    stack.pop_back();

    if (vertex->visited) continue;
    vertex->visited = true;
    visited.push_back(vertex);

    for (auto* next : vertex->out_edges) {
      if (vertex != producer && next->ord == upper_bound) {
        clearVisited(visited);
        return true;
      }
      if (next->ord < upper_bound) {
        stack.push_back(next);
      }
    }
  }
  clearVisited(visited);
  return false;
}


// Performs a forward DFS.
// Marks vertices as 'visited' and inserts them into visited.
// Returns false immediately if upper_bound is reached.
// Returns true otherwise.
template <typename T>
bool DynamicDAG<T>::dfsForward(Vertex<T>* producer, size_t upper_bound,
    vertex_list<T>& visited) {
  vertex_list<T> stack;
  stack.push_back(producer);

  while (!stack.empty()) {
    auto* vertex = stack.back();
    stack.pop_back();

    if (vertex->visited) continue;
    vertex->visited = true;
    visited.push_back(vertex);

    for (auto* next : vertex->out_edges) {
      if (next->ord == upper_bound) {
        return false;
      }
      if (next->ord < upper_bound) {
        stack.push_back(next);
      }
    }
  }
  return true;
}

// Performs a backward DFS.
// Marks vertices as 'visited' and inserts them into visited.
template <typename T>
void DynamicDAG<T>::dfsBackward(Vertex<T>* producer, size_t lower_bound,
    vertex_list<T>& visited) {
  vertex_list<T> stack;
  stack.push_back(producer);

  while (!stack.empty()) {
    auto* vertex = stack.back();
    stack.pop_back();

    if (vertex->visited) continue;

    vertex->visited = true;
    visited.push_back(vertex);

    for (auto* next : vertex->in_edges) {
      if (next->ord > lower_bound) {
        stack.push_back(next);
      }
    }
  }
}

template <typename T>
void DynamicDAG<T>::sort(vertex_list<T>& delta) {
  struct {
    bool operator()(Vertex<T>* a, Vertex<T>* b) const {
      return a->ord < b->ord;
    }
  } custom_less;
  std::sort(delta.begin(), delta.end(), custom_less);
}

template <typename T>
void DynamicDAG<T>::clearVisited(vertex_list<T>& visited) {
  for (auto* v: visited) {
    v->visited = false;
  }
  visited.clear();
}

template <typename T>
void DynamicDAG<T>::reorder(vertex_list<T>& deltaF, vertex_list<T>& deltaB) {
  // Reorder deltaB vertices to occur before deltaF vertices.
  sort(deltaB);
  sort(deltaF);

  vertex_list<T> tmp;

  std::vector<size_t> deltaB_ords = fmap(deltaB, [](Vertex<T>* v) { return v->ord; });
  std::vector<size_t> deltaF_ords = fmap(deltaF, [](Vertex<T>* v) { return v->ord; });

  std::vector<size_t> computed_ords;
  computed_ords.reserve(deltaB_ords.size() + deltaF_ords.size());
  std::merge(
    deltaB_ords.begin(), deltaB_ords.end(),
    deltaF_ords.begin(), deltaF_ords.end(), std::back_inserter(computed_ords));

  tmp.insert(tmp.end(), deltaB.begin(), deltaB.end());
  tmp.insert(tmp.end(), deltaF.begin(), deltaF.end());

  for (size_t i = 0; i < tmp.size(); i++) {
    auto new_ord = computed_ords[i];
    tmp[i]->ord = new_ord;
    vertices_[new_ord] = tmp[i];
  }

  clearVisited(deltaF);
  clearVisited(deltaB);
}

template <typename T>
std::string DynamicDAG<T>::toString() {
  std::stringstream ss;
  for (auto v : maybe_vertices()) {
    if (v.has_value()) {
      ss << v.value()->toString() << "\n";
    }
  }
  return ss.str();
}

template <typename T>
std::string Vertex<T>::toString() {
  std::stringstream ss;
  ss << "node(" << ord << ")\n";
  ss << "[";
  for (auto* c : in_edges) {
    ss << c->ord << " ";
  }
  ss << "] -> {\n";
  for (auto& d : rdata) {
    ss << "  " << d;
  }
  ss << "} ("<< ord << ") -> [";
  for (auto* c : out_edges) {
    ss << c->ord << " ";
  }
  ss << "]\n";
  return ss.str();
}

}}}
