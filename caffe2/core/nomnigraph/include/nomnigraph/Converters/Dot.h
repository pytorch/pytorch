#ifndef NOM_CONVERTERS_DOT_H
#define NOM_CONVERTERS_DOT_H

#include "nomnigraph/Graph/Graph.h"
#include "nomnigraph/Support/Casting.h"

#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

namespace {

template <typename GraphT>
class DotGenerator {
 public:
  using NodePrinter = std::function<std::map<std::string, std::string>(
      typename GraphT::NodeRef)>;
  using EdgePrinter = std::function<std::map<std::string, std::string>(
      typename GraphT::EdgeRef)>;

  static std::map<std::string, std::string> defaultEdgePrinter(
      typename GraphT::EdgeRef) {
    std::map<std::string, std::string> labelMap;
    return labelMap;
  }

  DotGenerator(NodePrinter nodePrinter, EdgePrinter edgePrinter)
      : nodePrinter_(nodePrinter), edgePrinter_(edgePrinter) {}

  // Convert a graph (with optional subgraphs cluster) to dot.
  std::string convert(
      GraphT* g,
      const std::vector<typename GraphT::SubgraphType*>& subgraphs) {
    std::ostringstream output;
    output << "digraph G {\n\
      ";
    for (const auto& node : g->getMutableNodes()) {
      generateNode(node, nullptr, output);
    }
    for (auto i = 0; i < subgraphs.size(); ++i) {
      const auto& subgraph = subgraphs[i];
      output << "subgraph cluster" << i << " {\n";
      output << "style=dotted;\n";
      for (const auto& node : subgraph->getNodes()) {
        output << node;
        output << ";\n";
      }
      output << "}\n";
    }
    output << "}";
    return output.str();
  }

  // Convert a subgraph to dot.
  std::string convert(typename GraphT::SubgraphType* sg) {
    std::ostringstream output;
    output << "digraph G {\n\
      ";
    for (const auto& node : sg->getNodes()) {
      generateNode(node, sg, output);
    }
    output << "}";
    return output.str();
  }

 private:
  NodePrinter nodePrinter_;
  EdgePrinter edgePrinter_;

  // Generate dot string for a node. Optionally take a subgraph to check if
  // out-edges belong to the subgraph.
  void generateNode(
      typename GraphT::NodeRef node,
      typename GraphT::SubgraphType* sg,
      std::ostringstream& output) {
    output << (uint64_t)node; // dot doesn't like hex
    output << "[";
    for (const auto& attrib : nodePrinter_(node)) {
      output << attrib.first << "=\"" << attrib.second << "\",";
    }
    output << "];\n";
    for (const auto& edge : node->getOutEdges()) {
      if (sg && !sg->hasEdge(edge)) {
        continue;
      }
      output << (uint64_t)edge->tail() << " -> " << (uint64_t)edge->head();
      output << "[";
      for (const auto& attrib : edgePrinter_(edge)) {
        output << attrib.first << "=\"" << attrib.second << "\",";
      }
      output << "];\n";
    }
  }
};

} // namespace

namespace nom {
namespace converters {

// Convert a graph to dot string.
template <typename GraphT>
std::string convertToDotString(
    GraphT* g,
    typename DotGenerator<GraphT>::NodePrinter nodePrinter,
    typename DotGenerator<GraphT>::EdgePrinter edgePrinter =
        DotGenerator<GraphT>::defaultEdgePrinter) {
  auto d = DotGenerator<GraphT>(nodePrinter, edgePrinter);
  return d.convert(g, {});
}

// Convert a graph to dot string and annotate subgraph clusters.
template <typename GraphT>
std::string convertToDotString(
    GraphT* g,
    const std::vector<typename GraphT::SubgraphType*>& subgraphs,
    typename DotGenerator<GraphT>::NodePrinter nodePrinter,
    typename DotGenerator<GraphT>::EdgePrinter edgePrinter =
        DotGenerator<GraphT>::defaultEdgePrinter) {
  auto d = DotGenerator<GraphT>(nodePrinter, edgePrinter);
  return d.convert(g, subgraphs);
}

// Convert a subgraph to dot string.
template <typename GraphT>
std::string convertToDotString(
    typename GraphT::SubgraphType* sg,
    typename DotGenerator<GraphT>::NodePrinter nodePrinter,
    typename DotGenerator<GraphT>::EdgePrinter edgePrinter =
        DotGenerator<GraphT>::defaultEdgePrinter) {
  auto d = DotGenerator<GraphT>(nodePrinter, edgePrinter);
  return d.convert(sg);
}

} // namespace converters
} // namespace nom

#endif // NOM_CONVERTERS_DOT_H
