//===- nomnigraph/Graph/Graph.h - Basic graph implementation ----*- C++ -*-===//
//
// This file defines a basic graph API for generic and flexible use with
// graph algorithms.
//
//===----------------------------------------------------------------------===//

#ifndef NOM_GRAPH_GRAPH_H
#define NOM_GRAPH_GRAPH_H

#include "caffe2/core/common.h"
#include "nomnigraph/Support/Common.h"

#include <algorithm>
#include <iterator>
#include <list>
#include <unordered_set>
#include <utility>
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
  Node(Node&&) = default;
  Node(const Node&) = delete;
  Node& operator=(const Node&) = delete;

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
    inEdges_ = std::move(edges);
  }

  void setOutEdges(std::vector<EdgeRef> edges) {
    outEdges_ = std::move(edges);
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

  template <class Arg>
  NodeRef createNode(Arg&& arg) {
    return createNode(T(std::forward<Arg>(arg)));
  }

  NodeRef createNode() {
    return createNodeInternal(Node<T, U...>());
  }

  // Note:
  // The move functions below are unsafe.  Use them with caution
  // and be sure to call isValid() after each use.

  // Move a node from this graph to the destGraph
  void moveNode(NodeRef node, Graph<T, U...>* destGraph) {
    assert(hasNode(node));
    for (auto it = nodes_.begin(); it != nodes_.end(); ++it) {
      if (&(*it) == node) {
        std::list<Node<T, U...>>& destNodes = destGraph->nodes_;
        destNodes.splice(destNodes.end(), nodes_, it);
        nodeRefs_.erase(node);
        destGraph->nodeRefs_.insert(node);
        break;
      }
    }
  }

  // Move an edge from this graph to the destGraph
  void moveEdge(EdgeRef edge, Graph<T, U...>* destGraph) {
    assert(hasEdge(edge));
    assert(destGraph->hasNode(edge->tail()));
    assert(destGraph->hasNode(edge->head()));
    std::list<Edge<T, U...>>& destEdges = destGraph->edges_;
    for (auto it = edges_.begin(); it != edges_.end(); ++it) {
      if (&(*it) == edge) {
        destEdges.splice(destEdges.end(), edges_, it);
        break;
      }
    }
  }

  // Move entire subgraph to destGraph.
  // Be sure to delete in/out edges from this graph first.
  void moveSubgraph(
      const Subgraph<T, U...>& subgraph,
      Graph<T, U...>* destGraph) {
    auto sg = subgraph; // Copy to check that all nodes and edges are matched
    std::list<Edge<T, U...>>& destEdges = destGraph->edges_;
    for (auto it = nodes_.begin(); it != nodes_.end(); ++it) {
      auto node = &(*it);
      if (sg.hasNode(node)) {
        std::list<Node<T, U...>>& destNodes = destGraph->nodes_;
        destNodes.splice(destNodes.end(), nodes_, it--);
        nodeRefs_.erase(node);
        destGraph->nodeRefs_.insert(node);
        sg.removeNode(node);
      }
    }
    for (auto it = edges_.begin(); it != edges_.end(); ++it) {
      auto edge = &(*it);
      if (sg.hasEdge(edge)) {
        assert(destGraph->hasNode(edge->tail()));
        assert(destGraph->hasNode(edge->head()));
        destEdges.splice(destEdges.end(), edges_, it--);
        sg.removeEdge(edge);
      }
    }
    assert(sg.getNodes().size() == 0);
    assert(sg.getEdges().size() == 0);
  }

  // Validates the graph.  Returns true if the graph is valid
  // and false if any node or edge referenced in the graph
  // is not actually present in the graph.
  bool isValid() {
    for (auto& node : getMutableNodes()) {
      for (auto& inEdge : node->getInEdges()) {
        if (!hasEdge(inEdge)) {
          DEBUG_PRINT("Invalid inEdge %p on node %p\n", inEdge, node);
          return false;
        }
      }
      for (auto& outEdge : node->getOutEdges()) {
        if (!hasEdge(outEdge)) {
          DEBUG_PRINT("invalid outEdge %p on node %p\n", outEdge, node);
          return false;
        }
      }
      // Check validity of nodeRefs_
      if (!hasNode(node)) {
        DEBUG_PRINT("Invalid node %p\n", node);
        return false;
      }
    }
    for (auto& edge : getMutableEdges()) {
      if (!hasNode(edge->tail())) {
        DEBUG_PRINT("Invalid tail on edge %p\n", edge);
        return false;
      }
      if (!hasNode(edge->head())) {
        DEBUG_PRINT("Invalid head on edge %p\n", edge);
        return false;
      }
    }
    return true;
  }

  // Swap two nodes.
  // Any edge V -> N1 becomes V -> N2, and N1 -> V becomes N2 -> V.
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

  /// \brief Replace a node in the graph with another node.
  /// \note The node replaced simply has its edges cut, but it not
  /// deleted from the graph.  Call Graph::deleteNode to delete it.
  /// \p oldNode A node to be replaced in the graph.
  /// \p newNode The node that inherit the old node's in-edges and out-edges.
  void replaceNode(const NodeRef& oldNode, const NodeRef& newNode) {
    replaceInEdges(oldNode, newNode);
    replaceOutEdges(oldNode, newNode);
  }

  // All out-edges oldNode -> V will be replaced with newNode -> V
  void replaceOutEdges(const NodeRef& oldNode, const NodeRef& newNode) {
    const auto edges = oldNode->getOutEdges();

    for (const auto& edge : edges) {
      edge->setTail(newNode);
      oldNode->removeOutEdge(edge);
      newNode->addOutEdge(edge);
    }
  }

  // All in-edges V -> oldNode  will be replaced with V -> newNode
  void replaceInEdges(const NodeRef& oldNode, const NodeRef& newNode) {
    const auto edges = oldNode->getInEdges();

    for (const auto& edge : edges) {
      edge->setHead(newNode);
      oldNode->removeInEdge(edge);
      newNode->addInEdge(edge);
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

  /// \brief Get a reference to the edge between two nodes if it exists. Returns
  /// nullptr if the edge does not exist.
  EdgeRef getEdgeIfExists(NodeRef tail, NodeRef head) const {
    for (auto& inEdge : head->getInEdges()) {
      if (inEdge->tail() == tail) {
        return inEdge;
      }
    }
    return nullptr;
  }

  /// \brief Returns true if there is an edge between the given two nodes.
  bool hasEdge(NodeRef tail, NodeRef head) const {
    return getEdgeIfExists(tail, head);
  }

  bool hasEdge(EdgeRef e) const {
    for (auto& edge : edges_) {
      if (e == &edge) {
        return true;
      }
    }
    return false;
  }

  /// \brief Get a reference to the edge between two nodes if it exists.
  /// note: will fail assertion if the edge does not exist.
  EdgeRef getEdge(NodeRef tail, NodeRef head) const {
    auto result = getEdgeIfExists(tail, head);
    assert(result && "Edge doesn't exist.");
    return result;
  }

  /// \brief Deletes a node from the graph.
  /// \param n A reference to the node.
  void deleteNode(NodeRef n) {
    if (!hasNode(n)) {
      return;
    }

    auto inEdges = n->inEdges_;
    for (auto& edge : inEdges) {
      deleteEdge(edge);
    }
    auto outEdges = n->outEdges_;
    for (auto& edge : outEdges) {
      deleteEdge(edge);
    }

    for (auto i = nodes_.begin(); i != nodes_.end(); ++i) {
      if (&*i == n) {
        nodeRefs_.erase(n);
        nodes_.erase(i);
        break;
      }
    }
  }

  // Delete all nodes in the set.
  void deleteNodes(const std::unordered_set<NodeRef>& nodes) {
    for (auto node : nodes) {
      deleteNode(node);
    }
  }

  bool hasNode(NodeRef node) const {
    return nodeRefs_.find(node) != nodeRefs_.end();
  }

  /// \brief Deletes a edge from the graph.
  /// \p e A reference to the edge.
  void deleteEdge(EdgeRef e) {
    e->tail_->removeOutEdge(e);
    e->head_->removeInEdge(e);
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
