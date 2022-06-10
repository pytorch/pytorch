#ifndef NOM_CONVERTERS_DOT_H
#define NOM_CONVERTERS_DOT_H

#include "c10/util/irange.h"
#include "nomnigraph/Graph/Algorithms.h"
#include "nomnigraph/Graph/Graph.h"
#include "nomnigraph/Support/Casting.h"

#include <functional>
#include <iostream>
#include <map>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace nom {
namespace converters {

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
      const typename GraphT::SubgraphType& sg,
      const std::vector<typename GraphT::SubgraphType*>& subgraphs) const {
    std::ostringstream output;
    output << "digraph G {\nrankdir=LR\n";
    for (const auto& node : sg.getNodes()) {
      generateNode(node, sg, output);
    }
    for (const auto i : c10::irange(subgraphs.size())) {
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
  std::string convert(const typename GraphT::SubgraphType& sg) const {
    std::ostringstream output;
    output << "digraph G {\nrankdir=LR\n";
    for (const auto& node : sg.getNodes()) {
      generateNode(node, sg, output);
    }
    output << "}";
    return output.str();
  }

  /**
   * NOTE No subgraph support
   * Converts given graph into DOT string w/operator input-order preserved
   * Assumes graph is acyclic, nodes are unique_ptr
   * (1) Get & print input nodes (nodes w/o parents)
   *     - Node: <p0>[shape=record, label="{{Data In}|{<p0>*}}"]
   * (2) Find operators w/BFS from input nodes
   * (3) Print operator records & incoming edges
   *     - Node: op_ptr[shape=record, label="{{<i0>*|<i1>*|...}|{op}|{<o0>*}"]
   *     - Edge: <parent_node_ptr>:<ref>:s -> <this_node_ptr>:<ref>:n
   */
  std::string convertStruct(const typename GraphT::SubgraphType& sg) const {
    std::ostringstream output;
    output << "digraph G {\nrankdir=LR\n";

    // Get input nodes (nodes w/o parents)
    std::unordered_map<typename GraphT::NodeRef, int>
        nodeDepthMap; // Touched nodes for BFS
    std::queue<typename GraphT::NodeRef> workList; // Init w/parentless nodes
    for (const auto& node : sg.getNodes()) {
      if (node->getInEdges().size() == 0 && node->getOutEdges().size() > 0) {
        // Add input node to dot string
        output << (uint64_t)node << "[shape=record, label=\"{{Data In}|{<"
               << (uint64_t)node << ">";
        for (const auto& attr : nodePrinter_(node)) {
          output << attr.second;
        }
        output << "}}\"]\n";

        // Track input node
        nodeDepthMap[node] = 0;
        workList.push(node);
      }
    }

    // BFS to get operator nodes
    std::vector<typename GraphT::NodeRef> ops;
    while (workList.size() > 0) {
      const auto& node = workList.front();
      for (const auto& edge : node->getOutEdges()) {
        // Enqueue child iff not touched yet
        const auto& child = edge->head();
        if (!nodeDepthMap.count(child)) {
          nodeDepthMap[child] = nodeDepthMap[node] + 1;
          workList.push(child);
          if (nodeDepthMap[child] % 2 == 1) { // "odd" ==> operator
            ops.emplace_back(child);
          }
        } else {
        }
      }
      workList.pop();
    }

    // Finalize output
    output << getOperatorSubtreeDotString(ops) << "}\n";
    return output.str();
  }

 private:
  NodePrinter nodePrinter_;
  EdgePrinter edgePrinter_;

  /**
   * Get DOT string record of given operator and DOT string of its input edges
   * @param  op          operator to parse
   * @param  nodePrinter node attribute extractor
   * @return             '\n' sep string of operator & input edges
   */
  std::string getOperatorDotString(typename GraphT::NodeRef op) const {
    std::ostringstream output;
    std::ostringstream record; // Operator node record
    record << (uint64_t)op << "[shape=record, label=\"{{";

    // Input refs
    std::string sep = "";
    for (const auto& opInEdge : op->getInEdges()) {
      // Draw edge between prev. op output to cur. op input
      const auto& input = opInEdge->tail();
      int inputInEdgeCt = input->getInEdges().size();
      if (inputInEdgeCt == 0) { // Node @ top of subgraph
        output << (uint64_t)input;
      } else { // Node between operators
        assert(inputInEdgeCt == 1);
        output << (uint64_t)input->getInEdges().at(0)->tail();
      }
      output << ":" << (uint64_t)input << ":s -> " << (uint64_t)op << ":"
             << (uint64_t)input << ":n\n";

      // Add input to operator record
      record << sep << "<" << (uint64_t)input << ">";
      for (const auto& attr : nodePrinter_(input)) {
        record << attr.second;
      }
      sep = "|";
    }

    // Extract operator name
    record << "}|{";
    for (const auto& attr : nodePrinter_(op)) {
      record << attr.second;
    }
    record << "}|{";

    // Output refs
    sep = "";
    for (const auto& edge : op->getOutEdges()) {
      const auto& child = edge->head();
      record << sep << "<" << (uint64_t)child << ">";
      for (const auto& attr : nodePrinter_(child)) {
        record << attr.second;
      }
      sep = "|";
    }

    // Append record to output string
    output << record.str() << "}}\"]\n";
    return output.str();
  }

  /**
   * Prints DOT string of given operator subgraph
   * @param  ops         operators in a given subgraph
   * @param  nodePrinter node attribute extractor
   * @return             DOT string that renders operators subgraph
   */
  std::string getOperatorSubtreeDotString(
      std::vector<typename GraphT::NodeRef> ops) const {
    std::ostringstream output;
    for (const auto& op : ops) {
      output << getOperatorDotString(op);
    }
    return output.str();
  }

  // Generate dot string for a node.
  void generateNode(
      typename GraphT::NodeRef node,
      const typename GraphT::SubgraphType& sg,
      std::ostringstream& output) const {
    output << (uint64_t)node; // dot doesn't like hex
    output << "[";
    for (const auto& attrib : nodePrinter_(node)) {
      output << attrib.first << "=\"" << attrib.second << "\",";
    }
    output << "];\n";
    for (const auto& edge : node->getOutEdges()) {
      if (!sg.hasEdge(edge)) {
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

// Convert a graph to dot string.
template <typename GraphT>
std::string convertToDotString(
    GraphT* g,
    typename DotGenerator<GraphT>::NodePrinter nodePrinter,
    typename DotGenerator<GraphT>::EdgePrinter edgePrinter =
        DotGenerator<GraphT>::defaultEdgePrinter) {
  auto d = DotGenerator<GraphT>(nodePrinter, edgePrinter);
  return d.convert(algorithm::createSubgraph(g), {});
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
  return d.convert(algorithm::createSubgraph(g), subgraphs);
}

// Convert a subgraph to dot string.
template <typename GraphT>
std::string convertToDotString(
    const typename GraphT::SubgraphType& sg,
    typename DotGenerator<GraphT>::NodePrinter nodePrinter,
    typename DotGenerator<GraphT>::EdgePrinter edgePrinter =
        DotGenerator<GraphT>::defaultEdgePrinter) {
  auto d = DotGenerator<GraphT>(nodePrinter, edgePrinter);
  return d.convert(sg);
}

template <typename GraphT>
std::string convertToDotRecordString(
    GraphT* g,
    typename DotGenerator<GraphT>::NodePrinter nodePrinter,
    typename DotGenerator<GraphT>::EdgePrinter edgePrinter =
        DotGenerator<GraphT>::defaultEdgePrinter) {
  auto d = DotGenerator<GraphT>(nodePrinter, edgePrinter);
  return d.convertStruct(algorithm::createSubgraph(g));
}

} // namespace converters
} // namespace nom

#endif // NOM_CONVERTERS_DOT_H
