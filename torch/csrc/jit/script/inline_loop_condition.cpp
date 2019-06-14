#include <functional>
#include <memory>
#include <string>

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/inline_loop_condition.h>

namespace torch {
namespace jit {
namespace script {

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
// Here, we inline the loop condition and convert the loop to the form:
// Loop(max_trip_count, start_condition)
//    block0(loop_counter, loop_carried_block*) {
//      <body>
//      -> (continue_condition)
//    }
void inlineLoopCondition(Node* n) {
  Block* body_block = n->blocks().at(0);

  auto pre_header = n->blocks().at(1);
  auto header_block = n->addBlock();
  header_block->cloneFrom(pre_header, [](Value* v) { return v; });
  moveBlockBeforeNode(n, header_block);
  n->addInput(header_block->outputs().at(0));
  n->eraseBlock(2);

  moveBlockBeforeNode(body_block->return_node(), pre_header);
  body_block->insertOutput(0, pre_header->outputs().at(0));
  n->eraseBlock(1);
}

void inlineLoopCondition(Block* block) {
  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      inlineLoopCondition(b);
    }
    if (n->kind() == prim::Loop) {
      inlineLoopCondition(n);
    }
  }
}

void InlineLoopCondition(std::shared_ptr<Graph>& graph) {
  inlineLoopCondition(graph->block());
}

} // namespace script
} // namespace jit
} // namespace torch
