#ifndef NOM_CONVERTERS_DOT_H
#define NOM_CONVERTERS_DOT_H

#include "nomnigraph/Graph/Graph.h"
#include "nomnigraph/Support/Casting.h"

#include <functional>
#include <iostream>
#include <sstream>

namespace {

template <typename T, typename U = T> class DotGenerator {
public:
  using NodePrinter = std::function<std::map<std::string, std::string>(
      typename nom::Graph<T, U>::NodeRef)>;
  using EdgePrinter = std::function<std::map<std::string, std::string>(
      typename nom::Graph<T, U>::EdgeRef)>;

  static std::map<std::string, std::string>
  defaultEdgePrinter(typename nom::Graph<T, U>::EdgeRef) {
    std::map<std::string, std::string> labelMap;
    return labelMap;
  }

  DotGenerator(typename nom::Graph<T, U> *g) : g_(g) {}

  std::string convert(NodePrinter nodePrinter, EdgePrinter edgePrinter) {
    std::ostringstream output;
    output << "digraph G {\n\
      bgcolor=\"#ffffff00\"\n\
      color=white\n\
      node[fontcolor=white,color=white];\n\
      edge[fontcolor=white,color=white];\n\
      ";
    for (const auto &node : g_->getMutableNodes()) {
      output << (uint64_t)node; // dot doesn't like hex
      output << "[";
      for (const auto &attrib : nodePrinter(node)) {
        output << attrib.first << "=\"" << attrib.second << "\",";
      }
      output << "];\n";
      for (const auto &edge : node->getOutEdges()) {
        output << (uint64_t)edge->tail() << " -> " << (uint64_t)edge->head();
        output << "[";
        for (const auto &attrib : edgePrinter(edge)) {
          output << attrib.first << "=\"" << attrib.second << "\",";
        }
        output << "];\n";
      }
    }
    for (auto i = 0; i < subgraphs_.size(); ++i) {
      const auto &subgraph = subgraphs_[i];
      output << "subgraph cluster" << i << " {\n";
      output << "style=dotted;\n";
      for (const auto &node : subgraph->getNodes()) {
        output << static_cast<uint64_t>(node);
        output << ";\n";
      }
      output << "}\n";
    }
    output << "}";
    return output.str();
  }

  void addSubgraph(const nom::Subgraph<T, U> *s) { subgraphs_.emplace_back(s); }

private:
  typename nom::Graph<T, U> *g_;
  typename std::vector<const nom::Subgraph<T, U> *> subgraphs_;
};

} // namespace

namespace nom {
namespace converters {

template <typename T, typename U = T>
std::string
convertToDotString(nom::Graph<T, U> *g,
                   typename DotGenerator<T, U>::NodePrinter nodePrinter,
                   typename DotGenerator<T, U>::EdgePrinter edgePrinter =
                       DotGenerator<T, U>::defaultEdgePrinter) {
  auto d = DotGenerator<T, U>(g);
  return d.convert(nodePrinter, edgePrinter);
}

template <typename T, typename U = T>
std::string
convertToDotString(nom::Graph<T, U> *g,
                   const std::vector<nom::Subgraph<T, U>> &subgraphs,
                   typename DotGenerator<T, U>::NodePrinter nodePrinter,
                   typename DotGenerator<T, U>::EdgePrinter edgePrinter =
                       DotGenerator<T, U>::defaultEdgePrinter) {
  auto d = DotGenerator<T, U>(g);
  for (const auto &subgraph : subgraphs) {
    d.addSubgraph(&subgraph);
  }
  return d.convert(nodePrinter, edgePrinter);
}

} // namespace converters
} // namespace nom

#endif // NOM_CONVERTERS_DOT_H
