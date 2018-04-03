//===- nomnigraph/Graph/Graph.h - Basic graph implementation ----*- C++ -*-===//
//
// TODO Licensing.
//
//===----------------------------------------------------------------------===//
//
// This file defines a basic graph API for generic and flexible use with
// graph algorithms.
//
//===----------------------------------------------------------------------===//

#ifndef NOM_GRAPH_GRAPH_H
#define NOM_GRAPH_GRAPH_H

#include "nomnigraph/Support/Common.h"

#include <algorithm>
#include <iterator>
#include <list>
#include <unordered_set>
#include <vector>

#include <stdio.h>
#include <assert.h>

#define DEBUG_PRINT(...) ;

namespace nom {

template <typename T, typename U = T> class Graph;

template <typename T, typename U = T> class Node;

template <typename T, typename U = T> class Edge : public StorageType<U> {
public:
  using NodeRef = typename Graph<T, U>::NodeRef;
  using EdgeRef = typename Graph<T, U>::EdgeRef;

  Edge(NodeRef tail, NodeRef head) : Tail(tail), Head(head) {
    DEBUG_PRINT("Creating instance of Edge: %p\n", this);
  }

  Edge(NodeRef tail, NodeRef head, U &&data)
      : StorageType<U>(std::move(data)), Tail(tail), Head(head) {
    DEBUG_PRINT("Creating instance of Edge: %p\n", this);
  }

  const NodeRef &tail() const { return Tail; }
  const NodeRef &head() const { return Head; }

private:
  NodeRef Tail;
  NodeRef Head;
  friend class Graph<T, U>;
};

template <typename T, typename U /* optional */>
class Node : public StorageType<T>, public Notifier<Node<T, U>> {
public:
  /// \brief Create a node with data.
  explicit Node(T &&data) : StorageType<T>(std::move(data)) {
    DEBUG_PRINT("Creating instance of Node: %p\n", this);
  }
  /// \brief Create an empty node.
  explicit Node() : StorageType<T>() {}

  using NodeRef = typename Graph<T, U>::NodeRef;
  using EdgeRef = typename Graph<T, U>::EdgeRef;

  /// \brief Adds an edge by reference to known in-edges.
  /// \p e A reference to an edge that will be added as an in-edge.
  void addInEdge(EdgeRef e) { inEdges.emplace_back(e); }

  /// \brief Adds an edge by reference to known out-edges.
  /// \p e A reference to an edge that will be added as an out-edge.
  void addOutEdge(EdgeRef e) { outEdges.emplace_back(e); }

  /// \brief Removes an edge by reference to known in-edges.
  /// \p e A reference to an edge that will be removed from in-edges.
  void removeInEdge(EdgeRef e) {
    auto iter = std::find(inEdges.begin(), inEdges.end(), e);
    inEdges.erase(iter);
  }

  /// \brief Removes an edge by reference to known out-edges.
  /// \p e A reference to an edge that will be removed from out-edges.
  void removeOutEdge(EdgeRef e) {
    auto iter = std::find(outEdges.begin(), outEdges.end(), e);
    outEdges.erase(iter);
  }

  const std::vector<EdgeRef> &getOutEdges() const { return outEdges; }
  const std::vector<EdgeRef> &getInEdges() const { return inEdges; }

protected:
  std::vector<EdgeRef> inEdges;
  std::vector<EdgeRef> outEdges;
  friend class Graph<T, U>;
};

/// \brief Effectively a constant reference to a graph.
///
/// \note A Subgraph could actually point to an entire Graph.
///
/// Subgraphs can only contain references to nodes/edges in a Graph.
/// They are technically mutable, but this should be viewed as a construction
/// helper rather than a fact to be exploited.  There are no deleters,
/// for example.
///
template <typename T, typename U = T> class Subgraph {
public:
  Subgraph() { DEBUG_PRINT("Creating instance of Subgraph: %p\n", this); }

  using NodeRef = typename Graph<T, U>::NodeRef;
  using EdgeRef = typename Graph<T, U>::EdgeRef;

  void addNode(NodeRef n) { Nodes.insert(n); }
  bool hasNode(NodeRef n) const { return Nodes.count(n) != 0; }
  void removeNode(NodeRef n) { Nodes.erase(n); }

  void addEdge(EdgeRef e) { Edges.insert(e); }
  bool hasEdge(EdgeRef n) const { return Edges.count(n) != 0; }
  void removeEdge(EdgeRef e) { Edges.erase(e); }

  const std::unordered_set<NodeRef> &getNodes() const { return Nodes; }
  const std::unordered_set<EdgeRef> &getEdges() const { return Edges; }

  void printEdges() {
    for (const auto &edge : Edges) {
      printf("Edge: %p (%p -> %p)\n", &edge, edge->tail(), edge->head());
    }
  }

  void printNodes() const {
    for (const auto &node : Nodes) {
      printf("Node: %p\n", node);
    }
  }

  std::unordered_set<NodeRef> Nodes;
  std::unordered_set<EdgeRef> Edges;
};

/// \brief A simple graph implementation
///
/// Everything is owned by the graph to simplify storage concerns.
///
template <typename T, typename U /* optional */> class Graph {
public:
  Graph() { DEBUG_PRINT("Creating instance of Graph: %p\n", this); }
  Graph(const Graph &) = delete;
  Graph(Graph &&) = default;
  Graph& operator=(Graph &&) = default;
  ~Graph() {}

  using NodeRef = Node<T, U> *;
  using EdgeRef = Edge<T, U> *;
  using NodeType = T;
  using EdgeType = U;

  /// \brief Creates a node and retains ownership of it.
  /// \p data An rvalue of the data being held in the node.
  /// \return A reference to the node created.
  NodeRef createNode(T &&data) {
    Nodes.emplace_back(Node<T, U>(std::move(data)));
    DEBUG_PRINT("Creating node (%p)\n", &Nodes.back());
    return &Nodes.back();
  }

  void swapNode(NodeRef node, Graph<T, U> &otherGraph) {
    std::list<Node<T, U>> &otherNodes = otherGraph.Nodes;
    for (auto it = Nodes.begin(); it != Nodes.end(); ++it) {
      if (&(*it) == node) {
        otherNodes.splice(otherNodes.end(), Nodes, it, ++it);
        break;
      }
    }
  }

  void swapEdge(EdgeRef edge, Graph<T, U> &otherGraph) {
    std::list<Edge<T, U>> &otherEdges = otherGraph.Edges;
    for (auto it = Edges.begin(); it != Edges.end(); ++it) {
      if (&(*it) == edge) {
        otherEdges.splice(otherEdges.end(), Edges, it, ++it);
        break;
      }
    }
  }

  NodeRef createNode() {
    Nodes.emplace_back(Node<T, U>());
    DEBUG_PRINT("Creating node (%p)\n", &Nodes.back());
    return &Nodes.back();
  }

  /// \brief Replace a node in the graph with a generic
  /// set of nodes.
  /// \note The node replaced simply has its edges cut, but it not
  /// deleted from the graph.  Call Graph::deleteNode to delete it.
  /// \p old A node to be replaced in the graph.
  /// \p newTail The node that inherit the old node's in-edges
  /// \p newHead (optional) The node that inherit the old node's out-edges
  void replaceNode(const NodeRef &old, const NodeRef &newTail,
                   const NodeRef &newHead_ = nullptr) {
    // If no newHead is specified, make the tail the head as well.
    // We are effectively replacing the node with one node in this case.
    const NodeRef newHead = newHead_ ? newHead_ : newTail;
    const auto inEdges = old->getInEdges();
    const auto outEdges = old->getOutEdges();

    for (const auto &inEdge : inEdges) {
      createEdge(inEdge->tail(), newTail);
    }
    for (const auto &inEdge : inEdges) {
      deleteEdge(inEdge);
    }

    for (const auto &outEdge : outEdges) {
      createEdge(newHead, outEdge->head());
    }
    for (const auto &outEdge : outEdges) {
      deleteEdge(outEdge);
    }
  }

  /// \brief Creates a directed edge and retains ownership of it.
  /// \p tail The node that will have this edge as an out-edge.
  /// \p head The node that will have this edge as an in-edge.
  /// \return A reference to the edge created.
  EdgeRef createEdge(NodeRef tail, NodeRef head) {
    DEBUG_PRINT("Creating edge (%p -> %p)\n", tail, head);
    Edges.emplace_back(Edge<T, U>(tail, head));
    EdgeRef e = &Edges.back();
    head->addInEdge(e);
    tail->addOutEdge(e);
    return e;
  }

  EdgeRef createEdge(NodeRef tail, NodeRef head, U &&data) {
    DEBUG_PRINT("Creating edge (%p -> %p)\n", tail, head);
    Edges.emplace_back(Edge<T, U>(tail, head, std::move(data)));
    EdgeRef e = &Edges.back();
    head->addInEdge(e);
    tail->addOutEdge(e);
    return e;
  }

  /// \brief Get a reference to the edge between two nodes if it exists.
  /// note: will fail assertion if the edge does not exist.
  EdgeRef getEdge(NodeRef tail, NodeRef head) {
    for (auto& inEdge : head->getInEdges()) {
      if (inEdge->tail() == tail) {
        return inEdge;
      }
    }
    assert(0 && "Edge doesn't exist.");
    return nullptr;
  }

  /// \brief Deletes a node from the graph.
  /// \param n A reference to the node.
  /// \param deleteEdges (optional) Whether or not to delete the edges
  /// related to the node.
  void deleteNode(NodeRef n, bool deleteEdges = true) {
    if (deleteEdges) {
      for (auto &edge : n->inEdges) {
        deleteEdge(edge);
      }
      for (auto &edge : n->outEdges) {
        deleteEdge(edge);
      }
    }
    for (auto i = Nodes.begin(); i != Nodes.end(); ++i) {
      if (&*i == n) {
        Nodes.erase(i);
        break;
      }
    }
  }

  /// \brief Deletes a edge from the graph.
  /// \p e A reference to the edge.
  void deleteEdge(EdgeRef e) {
    e->Tail->removeOutEdge(e);
    e->Head->removeInEdge(e);
    for (auto i = Edges.begin(); i != Edges.end(); ++i) {
      if (&*i == e) {
        Edges.erase(i);
        break;
      }
    }
  }

  const std::vector<NodeRef> getMutableNodes() {
    std::vector<NodeRef> v;
    for (auto &n : Nodes) {
      DEBUG_PRINT("Adding node to mutable output (%p)\n", &n);
      v.emplace_back(&n);
    }
    return v;
  }

  const std::vector<EdgeRef> getMutableEdges() {
    std::vector<EdgeRef> v;
    for (auto &e : Edges) {
      DEBUG_PRINT("Adding edge to mutable output (%p)\n", &e);
      v.emplace_back(&e);
    }
    return v;
  }

  void printEdges() {
    for (const auto &edge : Edges) {
      printf("Edge: %p (%p -> %p)\n", &edge, edge.tail(), edge.head());
    }
  }

  void printNodes() const {
    for (const auto &node : Nodes) {
      printf("Node: %p\n", &node);
    }
  }

private:
  std::list<Node<T, U>> Nodes;
  std::list<Edge<T, U>> Edges;
};

} // namespace nom

#endif // NOM_GRAPH_GRAPH_H
