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
 * indicating whether or not the return has been hit (has_returned).
 *
 * When we encounter a node that might return, we guard all subsequent nodes
 * in the block with the has returned value of that node.
 */

// Will a block or node return
enum ReturnStatus { WONT_RETURN, MIGHT_RETURN, WILL_RETURN };

// If a control flow node contains any nodes might return,
// an output for the return value will be added as the second to last output,
// and a has returned value will be added as the last output
constexpr size_t RETURN_OFFSET = 2;
constexpr size_t HAS_RETURNED_OFFSET = 1;

struct EarlyReturns {
  EarlyReturns(std::shared_ptr<Graph> graph_) : graph(std::move(graph_)) {
    WithInsertPoint guard(graph->block()->nodes().front());
    bottom_val = graph->insertNode(graph->create(prim::Bottom, {}, 1))
                     ->output()
                     ->setType(BottomType::get());
    true_val = graph->insertConstant(true);
    false_val = graph->insertConstant(false);
  };

  Value* getReturnVal(Node* n) {
    AT_ASSERT(n->kind() == prim::If || n->kind() == prim::Loop);
    return n->outputs().at(n->outputs().size() - RETURN_OFFSET);
  }

  Value* getReturnVal(Block* b) {
    return b->outputs().at(b->outputs().size() - RETURN_OFFSET);
  }

  Value* getHasReturned(Node* n) {
    AT_ASSERT(n->kind() == prim::If || n->kind() == prim::Loop);
    return n->outputs().at(n->outputs().size() - HAS_RETURNED_OFFSET);
  }

  Value* getHasReturned(Block* b) {
    return b->outputs().at(b->outputs().size() - HAS_RETURNED_OFFSET);
  }

  void registerReturnAndHasReturned(Block* block, Value* ret, Value* sent) {
    block->registerOutput(ret);
    AT_ASSERT(sent->type() == BoolType::get());
    block->registerOutput(sent);
  }

  void addReturnAndHasReturned(Block* block) {
    switch (block_status_[block]) {
      case WONT_RETURN: {
        return registerReturnAndHasReturned(
            block, bottom_val, getBoolVal(false));
      } break;
      case WILL_RETURN: {
        registerReturnAndHasReturned(
            block, block_return_vals_[block], getBoolVal(true));
      } break;
      case MIGHT_RETURN: {
        registerReturnAndHasReturned(
            block, block_return_vals_[block], block_has_returned_val_[block]);
      } break;
    }
  }

  // Recurses on the if node and returns its return status
  // If status != WONT_RETURN, sets the block_return_val and has returned val
  // of its parent block before exit
  ReturnStatus handleIf(Node* node) {
    auto true_block = node->blocks().at(0);
    auto false_block = node->blocks().at(1);

    // recurse
    auto true_status = makeReturnsFinal(true_block);
    auto false_status = makeReturnsFinal(false_block);

    if (true_status == WONT_RETURN && false_status == WONT_RETURN) {
      return WONT_RETURN;
    }

    addReturnAndHasReturned(true_block);
    addReturnAndHasReturned(false_block);

    auto out_type = unifyTypes(
        getReturnVal(true_block)->type(), getReturnVal(false_block)->type());
    AT_ASSERT(out_type);
    auto out = node->addOutput()->setType(*out_type)->setUniqueName("_return");
    auto sent =
        node->addOutput()->setType(BoolType::get())->setUniqueName("__did_ret");

    block_return_vals_[node->owningBlock()] = out;
    block_has_returned_val_[node->owningBlock()] = sent;

    if (true_status == WILL_RETURN && false_status == WILL_RETURN) {
      return WILL_RETURN;
    }
    return MIGHT_RETURN;
  }

  // Guards the remaining nodes in the block with an if node that takes
  // has_returned as its conditional
  ReturnStatus guardBlockNodes(
      Block* block,
      Value* has_returned,
      Value* ret,
      graph_node_list_iterator& iter) {
    auto new_if = graph->create(prim::If, 0)->insertAfter(has_returned->node());
    new_if->addInput(has_returned);

    auto return_block = new_if->addBlock();
    auto guard_block = new_if->addBlock();

    // NB: need to set return_block status and return value before recursing
    // or an empty block will appear to be a WONT_RETURN block
    block_status_[return_block] = WILL_RETURN;
    block_return_vals_[return_block] = ret;

    // Move all remaining nodes into the guard block
    while (iter != block->nodes().end()) {
      auto node = *iter++;
      node->moveBefore(guard_block->return_node());
    }

    // Set the new if to have the same outputs of the original block,
    // then replace the original block outputs with new if's outputs
    for (size_t i = 0; i < block->outputs().size(); ++i) {
      return_block->registerOutput(bottom_val);
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

  void deleteAfterReturnNodes(Block* block, graph_node_list_iterator& iter) {
    if (iter == block->nodes().end()) {
      return;
    }
    // need to destroy in reverse order so nodes have no uses when destroyed
    for (auto it = block->nodes().reverse().begin(); it != iter;) {
      if (*it == block->return_node()) {
        it++;
      } else {
        it.destroyCurrent();
      }
    }
    iter->destroy();
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

  ReturnStatus makeReturnsFinal(Block* block) {
    auto ret_status = WONT_RETURN;
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      Node* node = *it;
      it++;
      switch (node->kind()) {
        case prim::ReturnStmt: {
          ret_status = WILL_RETURN;
          block_return_vals_[block] = node->input();
          node->destroy();
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
        deleteAfterReturnNodes(block, it);
        break;
      } else if (ret_status == MIGHT_RETURN) {
        if (it != block->nodes().end()) {
          ret_status = guardBlockNodes(
              block, getHasReturned(node), getReturnVal(node), it);
        }
        break;
      }
    }
    if (block_status_.count(block) == 0) {
      block_status_[block] = ret_status;
      return ret_status;
    } else {
      // Guarded return blocks have their status set prior
      AT_ASSERT(
          block_status_[block] == WILL_RETURN &&
          block->nodes().begin() == block->nodes().end());
      return WILL_RETURN;
    }
  }

  Value* getBoolVal(bool val) {
    return val ? true_val : false_val;
  }

  void setGraphOutput() {
    // compiler ensures that the graph always returns
    auto block = graph->block();
    AT_ASSERT(block_status_[block] == WILL_RETURN);
    block->registerOutput(block_return_vals_[block]);
  }

  void run() {
    makeReturnsFinal(graph->block());
    setGraphOutput();
  }

  // After a call to makeReturnsFinal, a block will have set its return status
  std::unordered_map<Block*, ReturnStatus> block_status_;

  // Blocks that might return need a has_returned value to indicate if they
  // returned or not
  std::unordered_map<Block*, Value*> block_has_returned_val_;

  // Blocks that might return or will return have a return value
  std::unordered_map<Block*, Value*> block_return_vals_;

  Value* bottom_val;
  Value* true_val;
  Value* false_val;

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
