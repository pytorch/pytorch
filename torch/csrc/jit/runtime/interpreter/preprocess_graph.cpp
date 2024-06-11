#include <torch/csrc/jit/runtime/interpreter/preprocess_graph.h>

#include <torch/csrc/jit/frontend/schema_matching.h>
#include <torch/csrc/jit/runtime/interpreter/can_emit_inline.h>

namespace torch::jit::interpreter {

namespace {

// Insert explicit prim::MethodCall nodes after prim::Enter nodes
// to actually call __enter__ on the object. All prim::Enter does
// is push the object onto the stack of currently entered objects.
// This is necessary because emitting two instructions for a
// prim::Enter nodes (one ENTER to push onto the entered objects
// stack and one CALL to call __enter__) does not work; the
// accounting that determines when to move a value out of a register
// is based on the number of uses it has in the IR.
void insertEnterMethodCalls(Graph& g) {
  std::vector<Block*> block_queue;
  std::vector<Node*> enter_nodes;
  block_queue.emplace_back(g.block());

  // Traverse the graph while drilling down into blocks belonging to
  // a node and add all encountered prim::Enter nodes to enter_nodes.
  while (!block_queue.empty()) {
    Block* block = block_queue.back();
    block_queue.pop_back();

    for (auto node : block->nodes()) {
      if (node->kind() == prim::Enter) {
        enter_nodes.emplace_back(node);
        continue;
      }

      for (auto& node_block : node->blocks()) {
        block_queue.emplace_back(node_block);
      }
    }
  }

  // For each prim::Enter, emit a prim::MethodCall after it that actually
  // calls __enter__ on the object.
  for (auto& enter : enter_nodes) {
    auto cls = enter->input(0)->type()->expect<ClassType>();

    MatchedSchema enter_matched_schema = matchSchema(
        cls->findMethod("__enter__")->getSchema(),
        enter->input(0)->node()->sourceRange(),
        g,
        {enter->input(0)},
        {});

    Node* call = g.insertMethodCall("__enter__", enter_matched_schema)->node();
    call->moveAfter(enter);
    enter->replaceAllUsesWith(call);
  }
}

// insert Drop nodes to kill references for anything unused:
// this can happen in a few places, e.g. when a node returns
// many values but only one is used
// a, b = foo()
// return a
void dropUnused(Block* b) {
  auto createDropIfUnused = [&](ArrayRef<Value*> values) -> Node* {
    std::vector<Value*> to_drop;
    for (auto v : values) {
      if (v->uses().empty() && v->node()->kind() != prim::Constant) {
        to_drop.push_back(v);
      }
    }
    if (to_drop.empty()) {
      return nullptr;
    }
    return b->owningGraph()->create(prim::Drop, to_drop, 0);
  };

  if (auto d = createDropIfUnused(b->inputs())) {
    b->prependNode(d);
  }
  for (auto n : b->nodes()) {
    if (auto d = createDropIfUnused(n->outputs())) {
      d->insertAfter(n);
    }
    for (auto block : n->blocks()) {
      dropUnused(block);
    }
  }
}

// ensure every value has a final use in the same block where it is defined.
// This already true for most nodes. The exceptions are:
// 1. A value that is unused.
// 2. A value whose last use is nested in some control flow.
// For (1) we simply add a prim::Drop node that uses the value right after
// it is defined. For (2), we insert a prim::Drop right after the control
// flow node where the last use occurs
void insertLastUses(Graph& g) {
  // struct to share common data structures
  struct InsertLastUses {
    Graph& graph;
    // have we seen this value, yet, if not, it is the last use of the value
    std::unordered_set<Value*> seen;

    // A map from an If or Loop node to the optional Drop block that
    // occurs directly after it to release any tensors that go out of scope
    // when the If/Loop exits. These are created and inserted on demand.
    std::unordered_map<Node*, Node*> drop_for_node;

    explicit InsertLastUses(Graph& g) : graph(g) {
      scanBlock(graph.block());
    }
    void scanBlock(Block* b) {
      scanNode(b->return_node());
      for (auto n : b->nodes().reverse()) {
        scanNode(n);
      }
    }
    void scanNode(Node* n) {
      for (auto b : n->blocks()) {
        scanBlock(b);
      }
      // scan backwards so if a value is used twice in the list then it is a
      // move
      for (size_t i = n->inputs().size(); i > 0; --i) {
        scanUse(n, i - 1);
      }
    }
    void scanUse(Node* n, size_t i) {
      auto v = n->inputs()[i];
      auto inserted = seen.insert(v).second;
      if (!inserted) {
        return;
      }

      // the last use of v may be in a nested block of an If or Loop statement
      // find the node 'same_depth_node' at the same depth as the definition of
      // v, and consider that node to be the last use of v. This ensures we do
      // not delete nodes in nested scopes that may be executed multiple times
      // and that nodes used on one side of an if
      // but not the other get deleted regardless of the branch
      // e.g.
      // a = 4
      // while <...>:
      //   y = a + a
      // drop(a)
      // In other words, we find the first program point for v that
      // _reverse_ dominates the definition of v, and add a drop point there.
      Node* same_depth_node = findOwnerInBlock(n, v->node()->owningBlock());
      AT_ASSERT(
          same_depth_node); // failure means v is not in scope for n, use lint!

      // In the case where v and n are in the same block,
      // we have a legit final use already.
      if (same_depth_node == n) {
        return;
      }

      // in the case where the use is nested in a block
      // add a Drop node after that block which will drop 'v'.
      addToDropIfNotExists(
          findOrCreateDropInstructionForNode(same_depth_node), v);
    }

    // finds the node in block 'block' that contains in 'n'
    // or nullptr if no such node exists, e.g.:
    // n0: a = 4
    // n1: if <cond>:
    // n2:    b = a + a
    // findOwnerInBlock(n2, n0.block()) == n1
    Node* findOwnerInBlock(Node* n, Block* block) {
      while (n != nullptr && block != n->owningBlock()) {
        n = n->owningBlock()->owningNode();
      }
      return n;
    }

    Node* findOrCreateDropInstructionForNode(Node* n) {
      auto it = drop_for_node.find(n);
      if (it == drop_for_node.end()) {
        auto drop_node = graph.create(prim::Drop, 0);
        drop_node->insertAfter(n);
        it = drop_for_node.emplace(n, drop_node).first;
      }
      return it->second;
    }

    void addToDropIfNotExists(Node* drop, Value* v) {
      if (v->node()->kind() == prim::Constant) {
        return;
      }
      for (auto i : drop->inputs()) {
        // we already accounted for this use
        if (i == v) {
          return;
        }
      }
      drop->addInput(v);
    }
  };

  InsertLastUses ilu(g);
}

} // namespace

PreprocessGraph::PreprocessGraph(Graph& g) : graph(g.copy()) {
  insertEnterMethodCalls(*graph);
  dropUnused(graph->block());
  // fill in move_flags by scanning blocks;
  insertLastUses(*graph);
  can_emit_inline = std::move(CanEmitInline(*graph.get()).can_emit_inline_);
}
} // namespace torch::jit::interpreter
