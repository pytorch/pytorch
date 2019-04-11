#include <torch/csrc/jit/script/final_returns.h>
#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/ir_views.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/script/error_report.h>

namespace torch {
namespace jit {
namespace script {

/**
 * This pass transforms the Graph so that all ReturnStmts are merged into a
 * single value at the end of the graph.
 *
 * For blocks and control flow nodes that have a return statement that may have
 * been hit, we add an extra output for the return value, and an extra output
 * indicating whether or not the return has been hit (a sentinel value).
 *
 * When we encounter a node that might return, we guard all subsequent nodes
 * in the block with the sentinel value of that node.
 */

// Will a block or node return
enum Return_Status { WONT_RETURN, MIGHT_RETURN, WILL_RETURN };

// The return output of Control Flow Nodes is the second to last output,
// sentinel output is the output
constexpr size_t RETURN_OFFSET = 2;
constexpr size_t SENTINEL_OFFSET = 1;

struct EarlyReturns {
  EarlyReturns(std::shared_ptr<Graph> graph_) : graph(std::move(graph_)){};

  Value* getReturnVal(Node* n) {
    AT_ASSERT(n->kind() == prim::If || n->kind() == prim::Loop);
    return n->outputs().at(n->outputs().size() - RETURN_OFFSET);
  }

  Value* getReturnVal(Block* b) {
    return b->outputs().at(b->outputs().size() - RETURN_OFFSET);
  }

  Value* getSentinelVal(Node* n) {
    AT_ASSERT(n->kind() == prim::If || n->kind() == prim::Loop);
    return n->outputs().at(n->outputs().size() - SENTINEL_OFFSET);
  }

  Value* getSentinelVal(Block* b) {
    return b->outputs().at(b->outputs().size() - SENTINEL_OFFSET);
  }

  void registerReturnAndSentinel(Block* block, Value* ret, Value* sent) {
    block->registerOutput(ret);
    AT_ASSERT(sent->type() == BoolType::get());
    block->registerOutput(sent);
  }

  void addReturnAndSentinel(Block* block) {
    auto b_status = block_status[block];
    if (b_status == WONT_RETURN) {
      registerReturnAndSentinel(block, getBottomVal(), getBoolVal(false));
    } else if (b_status == WILL_RETURN) {
      registerReturnAndSentinel(
          block, block_return_vals[block], getBoolVal(true));
    } else if (b_status == MIGHT_RETURN) {
      registerReturnAndSentinel(
          block, block_return_vals[block], block_sentinel_val[block]);
    } else {
      AT_ASSERT(false);
    }
  }

  // Recurses on the if node and returns its return status
  // If status != WONT_RETURN, sets the block_return_val and sentinel val
  // of its parent block before exit
  Return_Status handleIf(Node* node) {
    auto true_block = node->blocks().at(0);
    auto false_block = node->blocks().at(1);

    // recurse
    makeReturnsFinal(true_block);
    makeReturnsFinal(false_block);

    auto true_status = block_status[true_block];
    auto false_status = block_status[false_block];

    if (true_status == WONT_RETURN && false_status == WONT_RETURN) {
      return WONT_RETURN;
    }

    addReturnAndSentinel(true_block);
    addReturnAndSentinel(false_block);

    auto out_type = unifyTypes(
        getReturnVal(true_block)->type(), getReturnVal(false_block)->type());
    AT_ASSERT(out_type);
    auto out = node->addOutput()->setType(*out_type)->setUniqueName("__return");
    auto sent =
        node->addOutput()->setType(BoolType::get())->setUniqueName("__did_ret");

    block_return_vals[node->owningBlock()] = out;
    block_sentinel_val[node->owningBlock()] = sent;

    if (true_status == WILL_RETURN && false_status == WILL_RETURN) {
      return WILL_RETURN;
    }
    return MIGHT_RETURN;
  }

  // Guards the remaining nodes in the block with an if node that takes
  // sentinel as its conditional
  Return_Status guardBlockNodes(
      Block* block,
      Value* sentinel,
      Value* ret,
      generic_graph_node_list_iterator<Node>& iter) {
    auto new_if = graph->create(prim::If, 0)->insertAfter(sentinel->node());
    new_if->addInput(sentinel);

    auto return_block = new_if->addBlock();
    auto guard_block = new_if->addBlock();

    // NB: need to set return_block status and return value before recursing
    // or an empty block will appear to be a WONT_RETURN block
    block_status[return_block] = WILL_RETURN;
    block_return_vals[return_block] = ret;

    // Move all remaining nodes into the guard block
    WithInsertPoint b(guard_block);
    auto insert_p = graph->insertConstant(false)->node();
    auto prev = insert_p;
    while (iter != block->nodes().end()) {
      auto node = *iter++;
      node->moveAfter(prev);
      prev = node;
    }
    insert_p->destroy();

    // Set the new if to have the same outputs of the original block,
    // then replace the original block outputs with new if's outputs
    for (size_t i = 0; i < block->outputs().size(); ++i) {
      return_block->registerOutput(getBottomVal());
      guard_block->registerOutput(block->outputs().at(i));
      new_if->addOutput()
          ->setType(block->outputs().at(i)->type())
          ->setUniqueName(block->outputs().at(i)->uniqueNameBase());
    }
    while (block->outputs().size() > 0) {
      block->eraseOutput(0);
    }
    for (auto out : new_if->outputs()) {
      block->registerOutput(out);
    }

    return handleIf(new_if);
  }

  void deleteAfterReturnNodes(Block* block, Node* return_node) {
    auto nodes = block->nodes().reverse();
    for (auto it = nodes.begin(); it != nodes.end() && *it != return_node;) {
      auto node = it;
      it++;
      if (*node != block->return_node()) {
        node->destroy();
      }
    }
    if (return_node->kind() == prim::ReturnStmt) {
      return_node->destroy();
    }
  }

  void checkNoLoopReturn(Block* block) {
    for (Node* n : block->nodes()) {
      if (n->kind() == prim::ReturnStmt) {
        throw ErrorReport(n->getSourceLocation())
            << "Return statements within loops are not yet supported\n";
      }
      for (Block* b : n->blocks()) {
        checkNoLoopReturn(b);
      }
    }
  }

  void makeReturnsFinal(Block* block) {
    auto ret_status = WONT_RETURN;
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      Node* node = *it;
      it++;
      switch (node->kind()) {
        case prim::ReturnStmt: {
          ret_status = WILL_RETURN;
          block_return_vals[block] = node->input();
        } break;
        case prim::If: {
          ret_status = handleIf(node);
        } break;
        case prim::Loop:
          checkNoLoopReturn(node->blocks().at(0)); // not yet supported
        default:
          break;
      }
      if (ret_status == WILL_RETURN) {
        deleteAfterReturnNodes(block, node);
        break;
      } else if (ret_status == MIGHT_RETURN) {
        if (it != block->nodes().end()) {
          ret_status = guardBlockNodes(
              block, getSentinelVal(node), getReturnVal(node), it);
        }
        break;
      }
    }
    if (block_status.count(block) == 0) {
      block_status[block] = ret_status;
    } else {
      // Guarded return blocks have their status set prior
      AT_ASSERT(
          block_status[block] == WILL_RETURN &&
          block->nodes().begin() == block->nodes().end());
    }
  }

  Value* getBottomVal() {
    if (bottom_val != nullptr) {
      return bottom_val;
    }
    WithInsertPoint guard(graph->block()->nodes().front());
    bottom_val = graph->insertNode(graph->create(prim::Bottom, {}, 1))
                     ->output()
                     ->setType(BottomType::get());
    return bottom_val;
  }

  Value* getBoolVal(bool val) {
    WithInsertPoint guard(graph->block()->nodes().front());
    if (val) {
      if (true_val != nullptr) {
        return true_val;
      }
      true_val = graph->insertConstant(true);
      return true_val;
    } else {
      if (false_val != nullptr) {
        return false_val;
      }
      false_val = graph->insertConstant(false);
      return false_val;
    }
  }

  void setGraphOutput() {
    // compiler ensures that the graph always returns
    auto block = graph->block();
    AT_ASSERT(block_status[block] == WILL_RETURN);
    block->registerOutput(block_return_vals[block]);
  }

  void run() {
    makeReturnsFinal(graph->block());
    setGraphOutput();
  }

  // After a call to makeReturnsFinal, a block will have set its return status
  std::unordered_map<Block*, Return_Status> block_status;

  // Blocks that might return need a sentinel value to indicate if they
  // returned or not
  std::unordered_map<Block*, Value*> block_sentinel_val;

  // Blocks that might return or will return have a return value
  std::unordered_map<Block*, Value*> block_return_vals;

  Value* bottom_val = nullptr;
  Value* true_val = nullptr;
  Value* false_val = nullptr;

  std::shared_ptr<Graph> graph;
};

void moveAllReturnsToEnd(std::shared_ptr<Graph>& graph) {
  EarlyReturns e(graph);
  e.run();
  // maybe dce ?
}
} // namespace script
} // namespace jit
} // namespace torch
