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

#include <assert.h>
#include <stdio.h>

#define DEBUG_PRINT(...)

namespace nom {

template <typename T, typename... U>
class Graph;

template <typename T, typename... U>
class Node;

// Template types:
//   T   : Data stored within a node.
//   U...: Data stored within an edge. When this type is not
//         specified, an empty StorageType is used. If it is
//         specified, only a single type should be given (as supported
//         by the underlying StorageType class).

// \brief Edge within a Graph.
template <typename T, typename... U>
class Edge : public StorageType<U...> {
 public:
  using NodeRef = typename Graph<T, U...>::NodeRef;
  Edge(NodeRef tail, NodeRef head, U... args)
      : StorageType<U...>(std::forward<U...>(args)...),
        tail_(tail),
        head_(head) {
    DEBUG_PRINT("Creating instance of Edge: %p\n", this);
  }

  const NodeRef& tail() const {
    return tail_;
  }
  const NodeRef& head() const {
    return head_;
  }

  void setTail(NodeRef n) {
    tail_ = n;
  }

  void setHead(NodeRef n) {
    head_ = n;
  }

 private:
  NodeRef tail_;
  NodeRef head_;

  friend class Graph<T, U...>;
};

// \brief Node within a Graph.
template <typename T, typename... U>
class Node : public StorageType<T>, public Notifier<Node<T, U...>> {
 public:
  using NodeRef = typename Graph<T, U...>::NodeRef;
  using EdgeRef = typename Graph<T, U...>::EdgeRef;

  /// \brief Create a node with data.
  explicit Node(T&& data) : StorageType<T>(std::move(data)) {
    DEBUG_PRINT("Creating instance of Node: %p\n", this);
  }
  /// \brief Create an empty node.
  explicit Node() : StorageType<T>() {}

  /// \brief Adds an edge by reference to known in-edges.
  /// \p e A reference to an edge that will be added as an in-edge.
  void addInEdge(EdgeRef e) {
    inEdges_.emplace_back(e);
  }

  /// \brief Adds an edge by reference to known out-edges.
  /// \p e A reference to an edge that will be added as an out-edge.
  void addOutEdge(EdgeRef e) {
    outEdges_.emplace_back(e);
  }

  /// \brief Removes an edge by reference to known in-edges.
  /// \p e A reference to an edge that will be removed from in-edges.
  void removeInEdge(EdgeRef e) {
    removeEdgeInternal(inEdges_, e);
  }

  /// \brief Removes an edge by reference to known out-edges.
  /// \p e A reference to an edge that will be removed from out-edges.
  void removeOutEdge(EdgeRef e) {
    removeEdgeInternal(outEdges_, e);
  }

  const std::vector<EdgeRef>& getOutEdges() const {
    return outEdges_;
  }
  const std::vector<EdgeRef>& getInEdges() const {
    return inEdges_;
  }

  void setInEdges(std::vector<EdgeRef> edges) {
    inEdges_ = edges;
  }

  void setOutEdges(std::vector<EdgeRef> edges) {
    outEdges_ = edges;
  }

 private:
  std::vector<EdgeRef> inEdges_;
  std::vector<EdgeRef> outEdges_;

  friend class Graph<T, U...>;

  void removeEdgeInternal(std::vector<EdgeRef>& edges, EdgeRef e) {
    auto iter = std::find(edges.begin(), edges.end(), e);
    assert(
        iter != edges.end() &&
        "Attempted to remove edge that isn't connected to this node");
    edges.erase(iter);
  }
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
template <typename T, typename... U>
class Subgraph {
 public:
  Subgraph() {
    DEBUG_PRINT("Creating instance of Subgraph: %p\n", this);
  }

  using NodeRef = typename Graph<T, U...>::NodeRef;
  using EdgeRef = typename Graph<T, U...>::EdgeRef;

  void addNode(NodeRef n) {
    nodes_.insert(n);
  }

  bool hasNode(NodeRef n) const {
    return nodes_.count(n) != 0;
  }

  void removeNode(NodeRef n) {
    nodes_.erase(n);
  }

  void addEdge(EdgeRef e) {
    edges_.insert(e);
  }

  bool hasEdge(EdgeRef e) const {
    return edges_.count(e) != 0;
  }

  void removeEdge(EdgeRef e) {
    edges_.erase(e);
  }

  const std::unordered_set<NodeRef>& getNodes() const {
    return nodes_;
  }

  size_t getNodesCount() const {
    return (size_t)nodes_.size();
  }

  const std::unordered_set<EdgeRef>& getEdges() const {
    return edges_;
  }

 private:
  std::unordered_set<NodeRef> nodes_;
  std::unordered_set<EdgeRef> edges_;

  void printEdges() {
    for (const auto& edge : edges_) {
      printf("Edge: %p (%p -> %p)\n", &edge, edge->tail(), edge->head());
    }
  }

  void printNodes() const {
    for (const auto& node : nodes_) {
      printf("Node: %p\n", node);
    }
  }
};

/// \brief A simple graph implementation
///
/// Everything is owned by the graph to simplify storage concerns.
///
template <typename T, typename... U>
class Graph {
 public:
  using SubgraphType = Subgraph<T, U...>;
  using NodeRef = Node<T, U...>*;
  using EdgeRef = Edge<T, U...>*;

  Graph() {
    DEBUG_PRINT("Creating instance of Graph: %p\n", this);
  }
  Graph(const Graph&) = delete;
  Graph(Graph&&) = default;
  Graph& operator=(Graph&&) = default;
  ~Graph() {}

  /// \brief Creates a node and retains ownership of it.
  /// \p data An rvalue of the data being held in the node.
  /// \return A reference to the node created.
  NodeRef createNode(T&& data) {
    return createNodeInternal(Node<T, U...>(std::move(data)));
  }

  NodeRef createNode() {
    return createNodeInternal(Node<T, U...>());
  }

  void importNode(NodeRef node, Graph<T, U...>& otherGraph) {
    for (auto it = nodes_.begin(); it != nodes_.end(); ++it) {
      if (&(*it) == node) {
        std::list<Node<T, U...>>& otherNodes = otherGraph.nodes_;
        otherNodes.splice(otherNodes.end(), nodes_, it, ++it);
        otherGraph.nodeRefs_.insert(node);
        break;
      }
    }
  }

  void importEdge(EdgeRef edge, Graph<T, U...>& otherGraph) {
    std::list<Edge<T, U...>>& otherEdges = otherGraph.edges_;
    for (auto it = edges_.begin(); it != edges_.end(); ++it) {
      if (&(*it) == edge) {
        otherEdges.splice(otherEdges.end(), edges_, it, ++it);
        break;
      }
    }
  }

  void swapNodes(NodeRef n1, NodeRef n2) {
    // First rectify the edges
    for (auto& inEdge : n1->getInEdges()) {
      inEdge->setHead(n2);
    }
    for (auto& outEdge : n1->getOutEdges()) {
      outEdge->setTail(n2);
    }
    for (auto& inEdge : n2->getInEdges()) {
      inEdge->setHead(n1);
    }
    for (auto& outEdge : n2->getOutEdges()) {
      outEdge->setTail(n1);
    }
    // Then simply copy the edge vectors around
    auto n1InEdges = n1->getInEdges();
    auto n1OutEdges = n1->getOutEdges();
    auto n2InEdges = n2->getInEdges();
    auto n2OutEdges = n2->getOutEdges();

    n1->setOutEdges(n2OutEdges);
    n1->setInEdges(n2InEdges);
    n2->setOutEdges(n1OutEdges);
    n2->setInEdges(n1InEdges);
  }

  /// \brief Replace a node in the graph with a generic
  /// set of nodes.
  /// \note The node replaced simply has its edges cut, but it not
  /// deleted from the graph.  Call Graph::deleteNode to delete it.
  /// \p old A node to be replaced in the graph.
  /// \p newTail The node that inherit the old node's in-edges
  /// \p newHead (optional) The node that inherit the old node's out-edges
  void replaceNode(
      const NodeRef& old,
      const NodeRef& newTail,
      const NodeRef& newHead_ = nullptr) {
    // If no newHead is specified, make the tail the head as well.
    // We are effectively replacing the node with one node in this case.
    const NodeRef newHead = newHead_ ? newHead_ : newTail;
    const auto inEdges = old->getInEdges();
    const auto outEdges = old->getOutEdges();

    for (const auto& inEdge : inEdges) {
      inEdge->setHead(newTail);
      old->removeInEdge(inEdge);
      newTail->addInEdge(inEdge);
    }

    for (const auto& outEdge : outEdges) {
      outEdge->setTail(newHead);
      old->removeOutEdge(outEdge);
      newTail->addOutEdge(outEdge);
    }
  }

  /// \brief Creates a directed edge and retains ownership of it.
  /// \p tail The node that will have this edge as an out-edge.
  /// \p head The node that will have this edge as an in-edge.
  /// \return A reference to the edge created.
  EdgeRef createEdge(NodeRef tail, NodeRef head, U... data) {
    DEBUG_PRINT("Creating edge (%p -> %p)\n", tail, head);
    this->edges_.emplace_back(
        Edge<T, U...>(tail, head, std::forward<U...>(data)...));
    EdgeRef e = &this->edges_.back();
    head->addInEdge(e);
    tail->addOutEdge(e);
    return e;
  }

  /// \brief Get a reference to the edge between two nodes if it exists.
  /// note: will fail assertion if the edge does not exist.
  EdgeRef getEdge(NodeRef tail, NodeRef head) const {
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
      auto inEdges = n->inEdges_;
      for (auto& edge : inEdges) {
        deleteEdge(edge);
      }
      auto outEdges = n->outEdges_;
      for (auto& edge : outEdges) {
        deleteEdge(edge);
      }
    }
    for (auto i = nodes_.begin(); i != nodes_.end(); ++i) {
      if (&*i == n) {
        nodeRefs_.erase(n);
        nodes_.erase(i);
        break;
      }
    }
  }

  bool hasNode(NodeRef node) const {
    return nodeRefs_.find(node) != nodeRefs_.end();
  }

  /// \brief Deletes a edge from the graph.
  /// \p e A reference to the edge.
  void deleteEdge(EdgeRef e, bool removeRef = true) {
    if (removeRef) {
      e->tail_->removeOutEdge(e);
      e->head_->removeInEdge(e);
    }
    for (auto i = edges_.begin(); i != edges_.end(); ++i) {
      if (&*i == e) {
        edges_.erase(i);
        break;
      }
    }
  }

  const std::vector<NodeRef> getMutableNodes() {
    std::vector<NodeRef> result;
    for (auto& n : nodes_) {
      DEBUG_PRINT("Adding node to mutable output (%p)\n", &n);
      result.emplace_back(&n);
    }
    return result;
  }

  size_t getNodesCount() const {
    return (size_t)nodes_.size();
  }

  const std::vector<EdgeRef> getMutableEdges() {
    std::vector<EdgeRef> result;
    for (auto& e : edges_) {
      DEBUG_PRINT("Adding edge to mutable output (%p)\n", &e);
      result.emplace_back(&e);
    }
    return result;
  }

  size_t getEdgesCount() const {
    return (size_t)edges_.size();
  }

 private:
  std::list<Node<T, U...>> nodes_;
  std::list<Edge<T, U...>> edges_;
  std::unordered_set<NodeRef> nodeRefs_;

  NodeRef createNodeInternal(Node<T, U...>&& node) {
    nodes_.emplace_back(std::move(node));
    NodeRef nodeRef = &nodes_.back();
    DEBUG_PRINT("Creating node (%p)\n", nodeRef);
    nodeRefs_.insert(nodeRef);
    return nodeRef;
  }

  void printEdges() {
    for (const auto& edge : edges_) {
      printf("Edge: %p (%p -> %p)\n", &edge, edge.tail(), edge.head());
    }
  }

  void printNodes() const {
    for (const auto& node : nodes_) {
      printf("Node: %p\n", &node);
    }
  }
};

} // namespace nom

#endif // NOM_GRAPH_GRAPH_H
