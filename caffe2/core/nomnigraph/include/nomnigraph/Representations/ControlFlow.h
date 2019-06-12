#ifndef NOM_REPRESENTATIONS_CONTROLFLOW_H
#define NOM_REPRESENTATIONS_CONTROLFLOW_H

#include "caffe2/core/common.h"
#include "nomnigraph/Graph/Graph.h"
#include "nomnigraph/Representations/Compiler.h"
#include "nomnigraph/Support/Pointer.h"

#include <unordered_map>

namespace nom {
namespace repr {

/// \brief A basic block holds a reference to a subgraph
/// of the data flow graph as well as an ordering on instruction
/// execution.  Basic blocks are used for control flow analysis.
template <typename T, typename... U>
class BasicBlock {
 public:
  using NodeRef = typename Subgraph<T, U...>::NodeRef;
  BasicBlock() {}
  BasicBlock(const BasicBlock&) = delete;
  BasicBlock(BasicBlock&&) = default;
  BasicBlock& operator=(const BasicBlock&) = delete;
  ~BasicBlock() {
    for (auto pair : callbacks_) {
      pair.first->deleteDestructorCallback(pair.second);
    }
  }

  void trackNode(NodeRef node) {
    callbacks_[node] = node->registerDestructorCallback([&](NodeRef n) {
      assert(
          hasInstruction(n) &&
          "Destructor callback invoked on untracked node in BasicBlock.");
      deleteInstruction(n);
    });
    nodes_.addNode(node);
  }

  void untrackNode(NodeRef node) {
    callbacks_.erase(node);
    nodes_.removeNode(node);
  }

  void pushInstructionNode(NodeRef node) {
    assert(
        isa<Instruction>(node->data()) &&
        "Cannot push non-instruction node to basic block.");
    instructions_.emplace_back(node);
    trackNode(node);
  }
  const std::vector<NodeRef>& getInstructions() const {
    return instructions_;
  }
  std::vector<NodeRef>* getMutableInstructions() {
    return &instructions_;
  }

  bool hasInstruction(NodeRef instr) const {
    return nodes_.hasNode(instr);
  }

  void insertInstructionBefore(NodeRef newInstr, NodeRef instr) {
    auto it =
        std::find(std::begin(instructions_), std::end(instructions_), instr);
    instructions_.insert(it, newInstr);
    trackNode(newInstr);
  }

  void moveInstructionBefore(NodeRef instr1, NodeRef instr2) {
    assert(hasInstruction(instr1) && "Instruction not in basic block.");
    assert(hasInstruction(instr2) && "Instruction not in basic block.");
    auto it1 =
        std::find(std::begin(instructions_), std::end(instructions_), instr1);
    auto it2 =
        std::find(std::begin(instructions_), std::end(instructions_), instr2);
    auto pos1b = std::distance(instructions_.begin(), it1);
    auto pos2b = std::distance(instructions_.begin(), it2);
    if (pos1b <= pos2b) {
      return;
    }
    instructions_.erase(it1);
    instructions_.insert(it2, instr1);
  }

  void deleteInstruction(NodeRef instr) {
    assert(hasInstruction(instr) && "Instruction not in basic block.");
    instructions_.erase(
        std::remove(instructions_.begin(), instructions_.end(), instr),
        instructions_.end());
    untrackNode(instr);
  }

 private:
  Subgraph<T, U...> nodes_;
  std::vector<NodeRef> instructions_;
  // Because we reference a dataflow graph, we need to register callbacks
  // for when the dataflow graph is modified.
  std::unordered_map<NodeRef, typename Notifier<Node<T, U...>>::Callback*>
      callbacks_;
};

using Program = Graph<Value>;

template <typename G>
struct ControlFlowGraphImpl {
  // Hack to help debugging in case this class is misused.
  static_assert(
      sizeof(ControlFlowGraphImpl),
      "Template parameter G in "
      "ControlFlowGraph<G> must be of "
      "type Graph<T, U...>.");
};

template <typename T, typename... U>
struct ControlFlowGraphImpl<Graph<T, U...>> {
  using type = Graph<BasicBlock<T, U...>, int>;
  using bbType = BasicBlock<T, U...>;
};

/// \brief Helper for extracting the type of BasicBlocks given
/// a graph (probably a dataflow graph).  TODO: refactor this
/// to come from something like Graph::NodeDataType
template <typename G>
using BasicBlockType = typename ControlFlowGraphImpl<G>::bbType;

/// \brief Control flow graph is a graph of basic blocks that
/// can be used as an analysis tool.
///
/// \note G Must be of type Graph<T, U...>.
template <typename G>
class ControlFlowGraph : public ControlFlowGraphImpl<G>::type {
 public:
  // This is for C++11 compatibility, otherwise we could use "using"
  ControlFlowGraph() {}
  ControlFlowGraph(const ControlFlowGraph&) = delete;
  ControlFlowGraph(ControlFlowGraph&&) = default;
  ControlFlowGraph& operator=(ControlFlowGraph&&) = default;
  ~ControlFlowGraph() {}
  std::unordered_map<
      std::string,
      typename ControlFlowGraphImpl<G>::type::SubgraphType>
      functions;
  using BasicBlockRef = typename ControlFlowGraphImpl<G>::type::NodeRef;

  // Named functions are simply basic blocks stored in labeled Subgraphs
  BasicBlockRef createNamedFunction(std::string name) {
    assert(name != "anonymous" && "Reserved token anonymous cannot be used");
    auto bb = this->createNode(BasicBlockType<G>());
    assert(functions.count(name) == 0 && "Name already in use.");
    typename ControlFlowGraphImpl<G>::type::SubgraphType sg;
    sg.addNode(bb);
    functions[name] = sg;
    return bb;
  }

  // Anonymous functions are aggregated into a single Subgraph
  BasicBlockRef createAnonymousFunction() {
    if (!functions.count("anonymous")) {
      functions["anonymous"] =
          typename ControlFlowGraphImpl<G>::type::SubgraphType();
    }

    auto bb = this->createNode(BasicBlockType<G>());
    functions["anonymous"].addNode(bb);
    return bb;
  }
};

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
