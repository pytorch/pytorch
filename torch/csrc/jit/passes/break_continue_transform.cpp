#include <torch/csrc/jit/passes/break_continue_transform.h>
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

void moveBlockBeforeNode(Node* before_node, Block* block);

/**
 * This pass transforms the graph so that break & continue statements are
 * removed. We transform the graph so that ops following a break or continue are
 * not run.
 */

// Will a block or node continue or break
enum LoopStatus { WONT, MIGHT, WILL };

// Are we transforming breaks or continues
enum Transform { BREAKS, CONTINUES };

struct LoopTransformer {
  LoopTransformer(std::shared_ptr<Graph> graph, Transform transform)
      : graph_(std::move(graph)) {
    WithInsertPoint guard(graph_->block()->nodes().front());
    true_val_ = graph_->insertConstant(true);
    false_val_ = graph_->insertConstant(false);
    transform_ = transform;
    incrementCurString();
  };

  const std::string getVarname() {
    return cur_string;
  }

  void incrementCurString() {
    static const std::string& break_name = "$did_break";
    static const std::string& continue_name = "$did_continue";
    const auto& name = transform_ == BREAKS ? break_name : continue_name;
    loop_count++;
    cur_string = name + std::to_string(loop_count);
  }

  void setCurString(const std::string& new_string) {
    cur_string = new_string;
  }

  Symbol transformKind() {
    return transform_ == BREAKS ? prim::BreakStmt : prim::ContinueStmt;
  }

  // Recursively transform both blocks of the if node.
  // If both blocks have hit the transform variable, then the return status,
  // is WILL, if both will not hit the transform variable it is false.
  // Otherwise we may have hit it.
  LoopStatus handleIf(Node* node) {
    auto true_block = node->blocks().at(0);
    auto false_block = node->blocks().at(1);

    auto true_status = handleTransforms(true_block);
    auto false_status = handleTransforms(false_block);

    if (true_status == WONT && false_status == WONT) {
      return WONT;
    } else if (true_status == WILL && false_status == WILL) {
      return WILL;
    } else {
      return MIGHT;
    }
  }

  // if an if node might hit a break or continue statement,
  // we guard all subsequent nodes in the block, and only execute them
  // if the transform is false.
  // The LoopStatus is the result of recursing on the newly created if.
  LoopStatus guardBlockNodes(
      Block* block,
      graph_node_list::iterator& remaining_block_nodes) {
    auto new_if =
        graph_->create(prim::If, 0)->insertBefore(*remaining_block_nodes);
    auto sentinel =
        graph_->createLoad(getVarname(), BoolType::get())->insertBefore(new_if);
    new_if->addInput(sentinel->output());

    auto hit_control_flow_block = new_if->addBlock();
    auto guard_block = new_if->addBlock();

    while (remaining_block_nodes != block->nodes().end()) {
      auto node = *remaining_block_nodes++;
      node->moveBefore(guard_block->return_node());
    }

    {
      WithInsertPoint insert(hit_control_flow_block);
      // NB: insert var scape before transform kind so it is not removed
      // In a graph like:
      // for i in range(3):
      //     if cond == 2:
      //         k : Optional[int] = None
      //         if cond == 2:
      //             m = 2
      //             break
      //         k = 1
      //         j = 2
      //     else:
      //         j = 1
      //         k = 2
      //     m += k
      // We transform the inner cond == 2 block to look like:
      // if cond == 2:
      //     m = 2
      //     $did_break = True
      // else:
      //     $did_break = False
      // if $did_break...
      //    prim::NewVarEscape
      // else:
      //    k = 1
      // For these new if nodes that guard ops after a continue/break may have
      // occurred, the new variables that are defined need to escape scope.
      // Otherwise, in the example above, we would error in the m += k call.
      graph_->insertNode(graph_->create(prim::NewVarEscape, 0));
      graph_->insertNode(graph_->create(transformKind(), 0));
    }
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

  void handleLoop(Node* loop_node) {
    const std::string prev_string = getVarname();
    // Give current loop unique identifier
    incrementCurString();
    // transform the loop
    transformLoop(loop_node);

    // restore previous identifier
    setCurString(prev_string);
  }

  // Create a check for the current transform variable.
  // if transform is true, loop continue condition is false, otherwise
  // run original condition
  void guardConditionBlock(Block* condition_block) {
    WithInsertPoint insert(*condition_block->nodes().begin());
    auto did_break =
        graph_->insertNode(graph_->createLoad(getVarname(), BoolType::get()))
            ->output();
    auto new_loop_condition = graph_->insertNode(graph_->create(prim::If));
    new_loop_condition->addInput(did_break);
    new_loop_condition->output()->setType(BoolType::get());
    new_loop_condition->addBlock()->registerOutput(false_val_);
    auto original_condition = new_loop_condition->addBlock();

    Node* n = new_loop_condition;
    for (n = n->next(); n != condition_block->return_node();) {
      auto cur = n;
      n = n->next();
      cur->moveBefore(original_condition->return_node());
    }
    original_condition->insertOutput(0, condition_block->outputs().at(0));
    condition_block->eraseOutput(0);
    condition_block->registerOutput(new_loop_condition->output());
  }

  void transformLoop(Node* n) {
    Block* body_block = n->blocks().at(0);
    auto ret_status = handleTransforms(body_block);

    // loop header should run even if we have continued
    if (transform_ == CONTINUES || ret_status == WONT) {
      return;
    }

    // because the condition block will get inlined as the start loop condition,
    // we need to make sure that it is defined before the loop executes
    // (and false so original condition is run). Also insert it into the block
    // so it is not an unneccessary loop carried var.
    graph_->createStore(getVarname(), false_val_)->insertBefore(n);
    graph_->createStore(getVarname(), false_val_)
        ->insertAfter(body_block->param_node());
    guardConditionBlock(n->blocks().at(1));
  };

  LoopStatus handleTransforms(Block* block) {
    auto loop_status = WONT;
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      Node* node = *it;
      it++;
      switch (node->kind()) {
        case prim::Function: {
          handleTransforms(node->blocks().at(0));
        } break;
        case prim::ContinueStmt:
        case prim::BreakStmt: {
          if (node->kind() != transformKind()) {
            continue;
          }
          node->destroy();
          loop_status = WILL;
        } break;
        case prim::If: {
          loop_status = handleIf(node);
        } break;
        case prim::Loop: {
          handleLoop(node);
        } break;
      }
      if (loop_status == WILL) {
        deleteAfterBreakNodes(block, it);
        break;
      } else if (loop_status == MIGHT) {
        if (it != block->nodes().end()) {
          loop_status = guardBlockNodes(block, it);
        }
        break;
      }
    }

    {
      // MIGHT value must be an output of an if, so we do not need to set it
      WithInsertPoint insert(block);
      if (loop_status == WILL) {
        graph_->insertNode(graph_->createStore(getVarname(), true_val_));
      } else if (loop_status == WONT) {
        graph_->insertNode(graph_->createStore(getVarname(), false_val_));
      }
    }

    return loop_status;
  }

  void run() {
    handleTransforms(graph_->block());
  }

  size_t loop_count = 0;
  Transform transform_;
  Value* true_val_ = nullptr;
  Value* false_val_ = nullptr;
  std::string cur_string = "";

  std::shared_ptr<Graph> graph_;
};

// These passes are run before SSA, so they need to handle before the
// Loop body and loop condition as a separate block.

void TransformBreaks(std::shared_ptr<Graph>& graph) {
  LoopTransformer breaks(graph, BREAKS);
  breaks.run();
}

void TransformContinues(std::shared_ptr<Graph>& graph) {
  LoopTransformer continues(graph, CONTINUES);
  continues.run();
}

} // namespace script
} // namespace jit
} // namespace torch
