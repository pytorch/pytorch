#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// This class facilitates depth-first iteration over all nodes in a graph.
class DepthFirstGraphNodeIterator {
  using BlockIteratorPair = std::pair<Block*, graph_node_list_iterator>;

  // The graph being iterated over.
  std::shared_ptr<Graph> graph_;

  // A stack of all blocks that need to be revisited when the current block has
  // been processed, as well as the corresponding nodes that should be returned
  // when those blocks are revisited. Think of it as the standard DFS stack.
  std::vector<BlockIteratorPair> block_stack_;

  // The {block, node} pair that is currently being processed. current_.first is
  // the block, current_.second is the iterator.
  BlockIteratorPair current_;

 public:
  // Constructor.
  DepthFirstGraphNodeIterator(std::shared_ptr<Graph>& graph)
      : graph_(graph),
        current_({graph->block(), graph->block()->nodes().begin()}) {}

  // Get the next Node in the graph. \returns nullptr if there are no nodes
  // left.
  Node* next() {
    // current_it always points to the next node that should be returned. If it
    // points to the end of the current block, that means there are no nodes
    // left in the graph to visit. This is because the only time an end iterator
    // is pushed to block_stack is if current block is the root block of the
    // graph.
    Node* node = current_.second != (current_.first)->nodes().end()
        ? *(current_.second)
        : nullptr;

    if (node) {
      // Advance current.second because there may be more nodes in
      // current_block.
      ++current_.second;

      // If there are no more nodes, set the current block and iterator to those
      // from the top of the stack; the one that was being iterated over when
      // the current block was encountered.
      if (current_.second == current_.first->nodes().end() &&
          !block_stack_.empty()) {
        current_ = block_stack_.back();
        block_stack_.pop_back();
      }

      // Handle If, Loop and With nodes in special ways because are the only
      // ones that own more blocks.
      if (node->kind() == prim::If) {
        auto* then_block = node->blocks().at(0);
        auto* else_block = node->blocks().at(1);

        bool then_block_empty =
            then_block->nodes().begin() == then_block->nodes().end();
        bool else_block_empty =
            else_block->nodes().begin() == else_block->nodes().end();

        if (!then_block_empty || !else_block_empty) {
          // If either of the then or else blocks have nodes, the current block
          // and iterator position need to be saved on the stack to resume
          // processing later.
          block_stack_.emplace_back(current_.first, current_.second);
        }

        if (!then_block_empty && else_block_empty) {
          // Set current_ to {then_block, then_block.begin()} and push nothing
          // to the stack since the else block is empty.
          current_.first = then_block;
          current_.second = then_block->nodes().begin();
        } else if (then_block_empty && !else_block_empty) {
          // Set current_ to {else_block, else_block.begin()} and push nothing
          // to the stack since the current block is already on the stack.
          current_.first = else_block;
          current_.second = else_block->nodes().begin();
        } else if (!then_block_empty && !else_block_empty) {
          // Set current_ to {then_block, then_block.begin()} and push the
          // else_block to the stack so that it will be processed after.
          block_stack_.emplace_back(else_block, else_block->nodes().begin());
          current_.first = then_block;
          current_.second = then_block->nodes().begin();
        }
      } else if (node->kind() == prim::Loop || node->kind() == prim::With) {
        auto* body_block = node->blocks().at(0);

        bool body_block_empty =
            body_block->nodes().begin() == body_block->nodes().end();

        if (!body_block_empty) {
          // If body_block is not empty, push the current block onto the stack
          // to resume processing it later and set current_ to {body_block,
          // body_block.begin()}.
          block_stack_.emplace_back(current_.first, current_.second);

          current_.first = body_block;
          current_.second = body_block->nodes().begin();
        }
      }
    } else {
      // There are no more nodes in the current block. Resume processing of the
      // block on the top of the stack if there is one.
      if (!block_stack_.empty()) {
        current_ = block_stack_.back();
        block_stack_.pop_back();
      }
    }

    return node;
  }
};

} // namespace jit
} // namespace torch
