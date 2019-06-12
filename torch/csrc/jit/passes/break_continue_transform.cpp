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
 * This pass transforms the Graph so that break & continue statements are
 * removed. We transform the graph so that ops following a break or continue are
 * not run.
 */

// Will a block or node continue or break
enum LoopStatus { WONT, MIGHT, WILL };

// Are we transforming breaks or continues
enum Transform { BREAKS, CONTINUES };

struct LoopTransformer {
  LoopTransformer(std::shared_ptr<Graph> graph_, Transform transform_)
      : graph(std::move(graph_)) {
    WithInsertPoint guard(graph->block()->nodes().front());
    true_val = graph->insertConstant(true);
    false_val = graph->insertConstant(false);
    transform = transform_;
  };

  const std::string& getVarname() {
    static const std::string& break_name = "$did_break";
    static const std::string& continue_name = "$did_continue";
    return transform == BREAKS ? break_name : continue_name;
  }

  Symbol transformKind() {
    return transform == BREAKS ? prim::BreakStmt : prim::ContinueStmt;
  }

  // Recurses on the if node and returns its return status
  // If status != WONT, sets the block_return_val and sentinel val
  // of its parent block before exit
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

  LoopStatus guardBlockNodes(
      Block* block,
      generic_graph_node_list_iterator<Node>& iter) {
    // if an if node might hit a break or continue statement,
    // we guard all subsequent nodes in the block, and only execute them
    // if we did break / did continue is false.

    auto new_if = graph->create(prim::If, 0)->insertBefore(*iter);
    auto sentinel =
        graph->createLoad(getVarname(), BoolType::get())->insertBefore(new_if);
    new_if->addInput(sentinel->output());

    auto hit_control_flow_block = new_if->addBlock();
    auto guard_block = new_if->addBlock();

    while (iter != block->nodes().end()) {
      auto node = *iter++;
      node->moveBefore(guard_block->return_node());
    }

    {
      WithInsertPoint insert(hit_control_flow_block);
      // NB: insert var scape before transform kind so it is not removed
      // See note in convert_to_ssa for why we need to insert VarEscape
      graph->insertNode(graph->create(prim::VarEscape, 0));
      graph->insertNode(graph->create(transformKind(), 0));
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

  void inlineLoopConditionIntoLoopBody(Node* n) {
    auto body_block = n->blocks().at(0);
    auto pre_header = n->blocks().at(1);
    moveBlockBeforeNode(body_block->return_node(), pre_header);
    body_block->insertOutput(0, pre_header->outputs().at(0));
    n->eraseBlock(1);
  }

  void handleLoop(Node* n) {
    Block* body_block = n->blocks().at(0);
    auto ret_status = handleTransforms(body_block);

    // When we're transforming breaks:
    // the body condition has not yet been inlined. If we we are not breaking
    // we need to inline the condition block into the end of the loop.
    // if we might break, we create an if statement and only execute the loop
    // header if we did not break.
    // Since we run the continue pass before the break pass,
    // we do not need to do any additional work in continues; guardBlock nodes
    // ensures that we do not execute any ops present in the block after a
    // continue, and loop condition is inlined after.

    if (transform == CONTINUES) {
      return;
    }

    if (ret_status == WONT) {
      inlineLoopConditionIntoLoopBody(n);
      return;
    }

    WithInsertPoint insert(body_block);
    auto did_break =
        graph->insertNode(graph->createLoad(getVarname(), BoolType::get()))
            ->output();

    auto new_loop_condition = graph->insertNode(graph->create(prim::If));
    new_loop_condition->addInput(did_break);
    new_loop_condition->output()->setType(BoolType::get());

    // if we did break, we do not continue
    new_loop_condition->addBlock()->registerOutput(false_val);
    auto original_condition = new_loop_condition->addBlock();
    auto pre_header = n->blocks().at(1);
    moveBlockBeforeNode(original_condition->return_node(), pre_header);
    original_condition->insertOutput(0, pre_header->outputs().at(0));
    n->eraseBlock(1);
    body_block->registerOutput(new_loop_condition->output());
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
          WithInsertPoint b(block);
          node->destroy();
          loop_status = WILL;
        } break;
        case prim::If: {
          loop_status = handleIf(node);
        } break;
        case prim::Loop: {
          handleLoop(node);
          // break statement can only effect the loop node
          loop_status = WONT;
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
        graph->insertNode(graph->createStore(getVarname(), true_val));
      } else if (loop_status == WONT) {
        graph->insertNode(graph->createStore(getVarname(), false_val));
      }
    }

    return loop_status;
  }

  void run() {
    handleTransforms(graph->block());
  }

  Transform transform;
  Value* true_val = nullptr;
  Value* false_val = nullptr;

  std::shared_ptr<Graph> graph;
};

void moveBlockBeforeNode(Node* before_node, Block* block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    auto block_node = *it++;
    block_node->moveBefore(before_node);
  }
}

// The loop node is initially emitted as:
// Loop(max_trip_count)
//    block0(loop_counter) {
//      <body>
//    }
//    block1 {
//      <loop condition>
//      -> (condition)
//    }
// Here, we inline the loop condition into:
// Loop(max_trip_count, start_condition)
//    block0(loop_counter) {
//      <body>
//    }
//    block1 {
//      <loop condition>
//      -> (condition)
//    }

void inlineLoopStartCondition(Node* n) {
  auto pre_header = n->blocks().at(1);
  auto header_block = n->addBlock();
  header_block->cloneFrom(pre_header, [](Value* v) { return v; });
  moveBlockBeforeNode(n, header_block);
  n->addInput(header_block->outputs().at(0));
  n->eraseBlock(2);
}

void inlineLoopStartCondition(Block* block) {
  for (Node* n : block->nodes()) {
    switch (n->kind()) {
      case prim::If:
      case prim::Function: {
        for (auto b : n->blocks()) {
          inlineLoopStartCondition(b);
        }
      } break;
      case prim::Loop: {
        inlineLoopStartCondition(n->blocks().at(0));
        inlineLoopStartCondition(n);
      } break;
    }
  }
}

// First we inline the loop input condition.
// Then, we transform the continues. Because the loop body condition
// has not yet been inlined, we can safely ignore it in the continue pass.
// Then, we transform breaks, inlining the loop body condition as part of the
// pass. Because they have not been inlined yet, we can generated nice graphs
// of the form
// if did_break
//    ... loop_continue = False
// else:
//    ... loop_continue = original_condition
void TransformBreaks(std::shared_ptr<Graph>& graph) {
  inlineLoopStartCondition(graph->block());
  LoopTransformer continues(graph, CONTINUES);
  continues.run();
  LoopTransformer breaks(graph, BREAKS);
  breaks.run();
}

} // namespace script
} // namespace jit
} // namespace torch
