#ifndef NOM_REPRESENTATIONS_CONTROLFLOW_H
#define NOM_REPRESENTATIONS_CONTROLFLOW_H

#include "nomnigraph/Graph/Graph.h"
#include "nomnigraph/Representations/Compiler.h"

#include <unordered_map>

namespace nom {
namespace repr {

/// \brief A basic block holds a reference to a subgraph
/// of the data flow graph as well as an ordering on instruction
/// execution.  Basic blocks are used for control flow analysis.
template <typename T, typename U> class BasicBlock {
public:
  using NodeRef = typename Subgraph<T, U>::NodeRef;
  BasicBlock() {}
  ~BasicBlock() {
    for (auto pair : callbacks) {
      pair.first->deleteDestructorCallback(pair.second);
    }
  }

  void trackNode(NodeRef node) {
    callbacks[node] = node->registerDestructorCallback([&](NodeRef n){
        assert(hasInstruction(n) && "Destructor callback invoked on untracked node in BasicBlock.");
        deleteInstruction(n);
    });
    Nodes.addNode(node);
  }

  void untrackNode(NodeRef node) {
    callbacks.erase(node);
    Nodes.removeNode(node);
  }

  void pushInstructionNode(NodeRef node) {
    assert(isa<Instruction>(node->data()) &&
           "Cannot push non-instruction node to basic block.");
    Instructions.emplace_back(node);
    trackNode(node);
  }
  const std::vector<NodeRef> &getInstructions() { return Instructions; }

  const bool hasInstruction(NodeRef instr) const {
    return Nodes.hasNode(instr);
  }

  void insertInstructionBefore(NodeRef newInstr, NodeRef instr) {
    auto it = std::find(std::begin(Instructions), std::end(Instructions), instr);
    Instructions.insert(it, newInstr);
    trackNode(newInstr);
  }

  void deleteInstruction(NodeRef instr) {
    assert(hasInstruction(instr) && "Instruction not in basic block.");
    Instructions.erase(std::remove(Instructions.begin(), Instructions.end(), instr),
        Instructions.end());
    untrackNode(instr);
  }

private:
  Subgraph<T, U> Nodes;
  std::vector<NodeRef> Instructions;
  // Because we reference a dataflow graph, we need to register callbacks
  // for when the dataflow graph is modified.
  std::unordered_map<NodeRef, typename Notifier<Node<T, U>>::Callback *> callbacks;
};

using Program = Graph<Value>;

template <typename G> struct ControlFlowGraphImpl {
  // Hack to help debugging in case this class is misused.
  static_assert(sizeof(ControlFlowGraphImpl), "Template parameter G in "
                                              "ControlFlowGraph<G> must be of "
                                              "type Graph<T, U>.");
};

template <typename T, typename U> struct ControlFlowGraphImpl<Graph<T, U>> {
  using type = Graph<std::unique_ptr<BasicBlock<T, U>>, int>;
  using bbType = BasicBlock<T, U>;
};

/// \brief Control flow graph is a graph of basic blocks that
/// can be used as an analysis tool.
///
/// \note G Must be of type Graph<T, U>.
template<typename G>
class ControlFlowGraph : public ControlFlowGraphImpl<G>::type {
public:
  // This is for C++11 compatibility, otherwise we could use "using"
  ControlFlowGraph() {}
  ControlFlowGraph(const ControlFlowGraph &) = delete;
  ControlFlowGraph(ControlFlowGraph &&) = default;
  ControlFlowGraph& operator=(ControlFlowGraph &&) = default;
  ~ControlFlowGraph() {
  }
};

/// \brief Helper for extracting the type of BasicBlocks given
/// a graph (probably a dataflow graph).  TODO: refactor this
/// to come from something like Graph::NodeDataType
template <typename G>
using BasicBlockType = typename ControlFlowGraphImpl<G>::bbType;

/// \brief Converts graph to SSA representation.  Modifies the graph
/// by inserting versions and phi nodes.
template <typename Phi, typename G> void addSSA(G* dfg, ControlFlowGraph<G> *cfg) {
  static_assert(std::is_base_of<Instruction, Phi>::value, "Phi type must be derived from Instruction.");
  auto dfMap = dominanceFrontierMap(cfg);
  for (auto pair : dfMap) {
    for (auto n : pair.second) {
      printf("%llu -> %llu\n", (unsigned long long)pair.first, (unsigned long long)n);
    }
  }
}

/// \brief Deletes a referenced node from the control flow graph.
template <typename G>
void deleteNode(ControlFlowGraph<G>* cfg, typename G::NodeRef node) {
  for (auto bbNode : cfg->getMutableNodes()) {
    auto bb = bbNode->data().get();
    if (bb->hasInstruction(node)) {
      bb->deleteInstruction(node);
    }
  }
}

} // namespace repr
} // namespace nom

#endif // NOM_REPRESENTATIONS_CONTROLFLOW_H
