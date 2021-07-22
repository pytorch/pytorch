#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// This class facilitates depth-first iteration over all nodes in a graph.
class DepthFirstGraphNodeIterator {
  graph_node_list_iterator current_;

 public:
  // Constructor.
  DepthFirstGraphNodeIterator(std::shared_ptr<Graph>& graph)
      : current_(graph->block()->nodes().begin()) {}

  // Moves up and to the next node (may move up recursively).
  void move_up(Node* from) {
    // Basically we start from the child block (which is current_)
    // and we try to find the block that owns it. Now we need to check
    // if that block is the graph root block, or if it is an If/Loop/etc
    // block.
    //
    // If it's the graph root block we can stop because there is no "up"
    // but if it is a node (e.g. If/Loop/etc) we need to apply logic
    // based on where we are coming from to move to the next block.
    // This might mean that we need to traverse up again (e.g. if we've
    // reached the end of the else clause in an if block we need to go)
    // up to the parent block that contains the if.
    //
    // Similarly if we've reached the end of the parent block containing
    // the else clause we might need to go up again so this is a recursive
    // function.
    //
    //              BlockNode (if/loop/with)
    //                       |
    //            [Block1]  ... [Block2]
    //                |
    //   [ Node1, Node2, Node3, FromNode]
    //
    auto parent_block = from->owningBlock();

    // Check if we've reached the top of the graph.
    if (parent_block == nullptr) {
      current_ = graph_node_list_iterator();
      return;
    }

    // Get the node that owns the parent block.
    // This node has to be an if, loop, or with.
    auto owning_node = parent_block->owningNode();
    if (owning_node == nullptr) {
      // If there's no node that owns this current block then
      // we're at the top of the graph and need to just move to the
      // next node if possible.
      current_ = parent_block->nodes().begin();
      while (*current_ != from) {
        ++current_;
      }

      // Move one past the from node.
      ++current_;
      if (current_ == parent_block->nodes().end()) {
        // If we've reached the end of the root block now set to null
        // since there are no more nodes.
        current_ = graph_node_list_iterator();
      }
      return;
    }


    // Find the owning node of the owning block. We need this
    // because current_ is a pointer to an iterator so we
    // need to find the specific position of the owning node
    // in case we need to go next or up.
    auto owning_node_block = owning_node->owningBlock();
    if (owning_node == nullptr) {
      throw std::runtime_error("Every node should be owned by a block");
    }

    // Find the position of the owning_node in it's parent block.
    auto owning_node_it = owning_node_block->nodes().begin();
    while (*owning_node_it != owning_node) {
      ++owning_node_it;
    }

    // Check the type of node this root is.
    if (owning_node->kind() == prim::If) {
      // Need to check if we came from the `then` branch or the `else` branch.
      auto* then_block = owning_node->blocks().at(0);
      auto* else_block = owning_node->blocks().at(1);

      if (parent_block == else_block) {
        // If else block then we move to the next node in the parent block.
        current_ = owning_node_it;
        ++current_;
        if (current_ == owning_node_block->nodes().end()) {
          move_up(*owning_node_it);
        }
      } else {
        // If then block then move to the else block if it is not empty.
        bool else_block_empty =
          else_block->nodes().begin() == else_block->nodes().end();

        if (! else_block_empty) {
          current_ = else_block->nodes().begin();
        } else {
          // Since it's empty we move to the next node.
          current_ = owning_node_it;
          ++current_;
          if (current_ == owning_node_block->nodes().end()) {
            move_up(*owning_node_it);
          }
        }
      }
    } else if (owning_node->kind() == prim::Loop || owning_node->kind() == prim::With) {
      current_ = owning_node_it;
      ++current_;
      if (current_ == owning_node_block->nodes().end()) {
        move_up(*owning_node_it);
      }
    } else {
      throw std::runtime_error("Node should not have had any child blocks.");
    }
  }

  // Moves to the next adjacent node or up in to the parent if that is not possible.
  void move_next() {
    auto block = current_->owningBlock();
    Node* previous = *current_;

    // Increment to the next node in the current block.
    ++current_;

    // Check if we're at the end of the block. If so we need
    // to move upwards (if it makes sense to).
    if (current_ == block->nodes().end()) {
      move_up(previous);
    }
  }

  // Moves to the next node in the graph into children if it can.
  void move_into() {
    if (*current_ == nullptr) {
      return;
    }

    auto node = *current_;

    // Check if we're currently on a node that contains sub-nodes.
    if (node->kind() == prim::If) {
      // If nodes can have two child blocks. From an If node
      // in `move_into` we only enter the first block which contains
      // elements. After processing of those elements is done, we move
      // to the next block (the else block if it has nodes) from the
      // `move_up` call.
      auto* then_block = node->blocks().at(0);
      auto* else_block = node->blocks().at(1);

      bool then_block_empty =
        then_block->nodes().begin() == then_block->nodes().end();

      bool else_block_empty =
        else_block->nodes().begin() == else_block->nodes().end();

      if (! then_block_empty) {
        current_ = then_block->nodes().begin();
      } else if (! else_block_empty) {
        current_ = else_block->nodes().begin();
      } else {
        // This `if` block does not have any child nodes so we need to continue
        // moving next/over it.
        move_next();
      }
    } else if (node->kind() == prim::Loop || node->kind() == prim::With) {
      auto* body_block = node->blocks().at(0);

      bool body_block_empty =
          body_block->nodes().begin() == body_block->nodes().end();

      if (!body_block_empty) {
        // If we have a body block - we move there next.
        current_ = body_block->nodes().begin();
      } else {
        move_next();
      }
    } else {
      move_next();
    }
  }

  // Get the next Node in the graph. \returns nullptr if there are no nodes
  // left.
  Node* next() {
    auto result = *current_;
    move_into();
    return result;
  }
};

} // namespace jit
} // namespace torch
