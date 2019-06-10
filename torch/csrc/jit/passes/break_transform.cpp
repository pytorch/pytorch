#include <torch/csrc/jit/script/break_transform.h>
#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/ir_views.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/script/error_report.h>
#include <torch/csrc/jit/script/final_returns.h>

namespace torch {
namespace jit {
namespace script {

using NameToValue = std::unordered_map<std::string, Value*>;
using ValueToName = std::unordered_map<Value*, std::string>;

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
enum BreakStatus { WONT_BREAK, MIGHT_BREAK, WILL_BREAK };

// The return output of Control Flow Nodes is the second to last output,
// sentinel output is the output
constexpr size_t HAS_BROKE_OFFSET = 1;

struct BreakTransformer {
  BreakTransformer(std::shared_ptr<Graph> graph_) : graph(std::move(graph_)){};

  // Recurses on the if node and returns its return status
  // If status != WONT_BREAK, sets the block_return_val and sentinel val
  // of its parent block before exit
  BreakStatus handleIf(Node* node) {
    auto true_block = node->blocks().at(0);
    auto false_block = node->blocks().at(1);

    // recurse
    auto true_status = handleBreaks(true_block);
    auto false_status = handleBreaks(false_block);

    if (true_status == WONT_BREAK && false_status == WONT_BREAK) {
      return WONT_BREAK;
    } else if (true_status == WILL_BREAK && false_status == WILL_BREAK) {
      return WILL_BREAK;
    } else {
      return MIGHT_BREAK;
    }
  }
  BreakStatus guardBlockNodes(
      Block* block,
      generic_graph_node_list_iterator<Node>& iter) {
    AT_ASSERT(getBlockStatus(block) == MIGHT_BREAK);
    auto sentinel = block_sentinel_val[block];
    auto new_if = graph->create(prim::If, 0)->insertAfter(sentinel->node());
    new_if->addInput(sentinel);

    auto break_block = new_if->addBlock();
    auto guard_block = new_if->addBlock();

    // Move all remaining nodes into the guard block
    while (iter != block->nodes().end()) {
      auto node = *iter++;
      node->moveBefore(guard_block->return_node());
    }
    std::vector<std::string> block_output_names =
        block->owningNode()->ss(attr::value);
    for (auto name : block_output_names) {
      break_block->registerOutput(environment_stack->findInAnyFrame(name));
    }
    for (size_t i = 0; i < block->outputs().size(); ++i) {
      guard_block->registerOutput(block->outputs().at(i));
    }
    new_if->ss_(attr::value, block_output_names);

    for (size_t i = 0; i < block->outputs().size(); ++i) {
      auto orig_output = block->outputs().at(i);
      new_if->addOutput()
          ->setType(orig_output->type())
          ->setUniqueName(uniqueName(orig_output));
    }
    while (block->outputs().size() > 0) {
      block->eraseOutput(0);
    }
    for (auto out : new_if->outputs()) {
      block->registerOutput(out);
    }
    block_sentinel_val[break_block] = true_val;
    return handleIf(new_if);
  }

  void deleteAfterBreakNodes(Block* block, graph_node_list_iterator& iter) {
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

  void handleLoop(Node* n){

  };

  BreakStatus handleBreaks(Block* block) {
    auto ret_status = WONT_BREAK;
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      Node* node = *it;
      it++;
      switch (node->kind()) {
        case prim::BreakStmt: {
          WithInsertPoint b(block);
          graph->createStore("__did_return", true_val);
          ret_status = WILL_BREAK;
        } break;
        case prim::If: {
          ret_status = handleIf(node);
        } break;
        case prim::Loop:
          handleLoop(node);
          // break statement can only effect the loop node
          ret_status = WONT_BREAK;
        default:
          break;
      }
      if (ret_status == WILL_BREAK) {
        deleteAfterBreakNodes(block, it);
        break;
      } else if (ret_status == MIGHT_BREAK) {
        if (it != block->nodes().end()) {
          // if (block->owningNode()->kind() == prim::Loop) {
          // ret_status = guardLoopBlockNodes(block, getSentinelVal(node), it);
        } else {
          ret_status = guardBlockNodes(block, it);
        }
      }
      break;
    }
  }
  if (block_status.count(block) == 0) {
    block_status[block] = ret_status;
  } else {
    // Guarded return blocks have their status set prior
    AT_ASSERT(
        block_status[block] == WILL_BREAK &&
        block->nodes().begin() == block->nodes().end());
  }
}

Value*
getBottomVal() {
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
// for (Node * n: block->nodes()) {
void associateVarCaptures(Block* block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    Node* n = *it;
    it++;
    if (n->kind() == prim::VarCapture) {
      block_capture[block] = convertVarNameToValueNode(n);
      n->destroy();
    }
    for (Block* b : n->blocks()) {
      associateVarCaptures(b);
    }
  }
}

void run() {
  associateVarCaptures(graph->block());
  handleBreaks(graph->block());
}

// a block may have a value that is used in the loop continue condition
std::unordered_map<Block*, Value*> loop_continue_condition;

// After a call to handleBreaks, a block will have set its break status
std::unordered_map<Block*, BreakStatus> block_status;

// Blocks that might break need a sentinel value to indicate if they
// broke or not
std::unordered_map<Block*, Value*> block_sentinel_val;

std::unordered_map<Block*, NameToValue> block_capture;
// std::unordered_map<Block *, NameToValue> var_capture_blocks;

std::unordered_map<Node*, NameToValue> node_capture;

Value* bottom_val = nullptr;
Value* true_val = nullptr;
Value* false_val = nullptr;

std::shared_ptr<Graph> graph;
}; // namespace script

void transformBreaks(std::shared_ptr<Graph>& graph) {
  ConstantPooling(graph);
  BreakTransformer e(graph);
  e.run();
  // maybe dce ?
}

} // namespace jit
} // namespace torch
} // namespace torch
