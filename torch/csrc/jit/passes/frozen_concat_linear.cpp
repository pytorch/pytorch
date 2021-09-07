#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/frozen_concat_linear.h>
#include <torch/csrc/jit/passes/frozen_conv_folding.h>
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
#include <torch/csrc/jit/passes/remove_dropout.h>
#include <torch/csrc/jit/passes/utils/optimization_utils.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/utils/memory.h>
#include <vector>

namespace torch {
namespace jit {
namespace {

using Tensor = at::Tensor;

class ConcatLinearLayers {
  std::shared_ptr<Graph> graph_;
  bool graph_modified = false;
  std::unique_ptr<AliasDb> aliasDb_ = nullptr;

  AliasDb* getAliasDb() {
    if (!aliasDb_) {
      aliasDb_ = std::make_unique<AliasDb>(graph_);
    }
    return aliasDb_.get();
  }

  void collectConstantLinearLayers(
      Block* b,
      std::unordered_map<Value*, std::list<Node*>>& grouped_linear_layers,
      std::vector<Value*>& ordered_tensor_inputs) {
    // We are using an ordered list so that one knows that
    // one only needs to move arguments forward to be a valid move, not
    // backwards. Otherwise we need to rebuild the aliasDb when we add values.

    for (Node* n : b->nodes()) {
      // Grouping together all linear layers that use the same Tensor for input
      if (n->kind() != aten::linear) {
        continue;
      }

      auto weight = n->namedInput("weight");
      auto bias = n->namedInput("bias");
      if (weight->type() == NoneType::get() ||
          bias->type() == NoneType::get()) {
        continue;
      }

      if (nonConstantParameters(n)) {
        continue;
      }

      Value* op_value = n->inputs().at(0);
      if (grouped_linear_layers.find(op_value) ==
          grouped_linear_layers.cend()) {
        grouped_linear_layers.insert({op_value, std::list<Node*>()});
        ordered_tensor_inputs.push_back(op_value);
      }
      grouped_linear_layers.find(op_value)->second.push_back(n);
    }
  }

  void mergeLinearLayers(
      Node* base_node,
      std::vector<Node*>& compatible_layers) {
    graph_modified = true;

    for (auto n : compatible_layers) {
      if (n == base_node) {
        continue;
      }
      getAliasDb()->moveBeforeTopologicallyValid(base_node, n);
    }

    // Now actually merge the candidate nodes

    // Scope needed to make sure we free the WithInsertPoint guard
    // before we delete `base_node`
    Node* linear_node = nullptr;
    {
      WithInsertPoint guard(base_node);
      auto weight_list = c10::fmap(compatible_layers, [](Node* n) {
        return constant_as<Tensor>(n->namedInput("weight")).value();
      });
      Tensor cat_weight = at::cat(weight_list, /*dim=*/0);
      Value* cat_weight_value = graph_->insertConstant(cat_weight);

      auto bias_list = c10::fmap(compatible_layers, [](Node* n) {
        return constant_as<Tensor>(n->namedInput("bias")).value();
      });
      Tensor cat_bias = at::cat(bias_list, /*dim=*/0);
      Value* cat_bias_value = graph_->insertConstant(cat_bias);

      auto tensor_input = base_node->inputs().at(0);
      std::vector<Value*> linear_in = {
          tensor_input, cat_weight_value, cat_bias_value};
      linear_node = graph_->create(aten::linear, linear_in);
      linear_node->insertBefore(base_node);
    }

    // Update the outputs of the nodes
    WithInsertPoint guard2(linear_node);
    Value* neg1 = graph_->insertConstant(-1);
    Value* one = graph_->insertConstant(1);

    int64_t slice_start = 0;
    Value* slice_start_val = graph_->insertConstant(0);

    for (Node* orig_node : compatible_layers) {
      // for each node in the compatible_layers list,
      // slide the output of the combined linear layer
      // and use it instead of the output of the original node

      Tensor weight_tensor =
          constant_as<Tensor>(orig_node->namedInput("weight")).value();
      int64_t slice_end = slice_start + weight_tensor.size(0);
      Value* slice_end_val = graph_->insertConstant(slice_end);

      Node* slice = graph_->create(
          aten::slice,
          {linear_node->output(), neg1, slice_start_val, slice_end_val, one});
      slice->insertAfter(linear_node);
      orig_node->replaceAllUsesWith(slice);
      orig_node->destroy();

      slice_start = slice_end;
      slice_start_val = slice_end_val;
    }
  }

  bool isNonZeroDimEqual(Tensor& tensor_a, Tensor& tensor_b) {
    if (tensor_a.dim() != tensor_b.dim()) {
      return false;
    }
    for (int64_t i = 1; i < tensor_a.dim(); i++) {
      if (tensor_a.size(i) != tensor_b.size(i)) {
        return false;
      }
    }
    return true;
  }

  // Check the linear_layer_group of a tensor to find ones that can be
  // combined
  void collectAndMergeLinearLayers(std::list<Node*>& linear_layer_group) {
    while (!linear_layer_group.empty()) {
      auto it_first = linear_layer_group.begin();
      Node* base_node = *it_first;

      std::vector<Node*> compatible_layers;
      compatible_layers.push_back(base_node);
      // removing the node from future consideration, as we will only move
      // nodes forward to keep code simple.
      linear_layer_group.erase(it_first);

      auto base_weight =
          constant_as<Tensor>(base_node->namedInput("weight")).value();
      auto base_bias =
          constant_as<Tensor>(base_node->namedInput("bias")).value();

      // Now iterate over the rest of the users of the set to
      // see if there is anything that we can coaleasce `base_node` with.
      for (auto it = linear_layer_group.begin();
           it != linear_layer_group.end();) {
        auto node = *it;
        auto weight = constant_as<Tensor>(node->namedInput("weight")).value();
        auto bias = constant_as<Tensor>(node->namedInput("bias")).value();

        // For now we will just keep it simple and require matching types
        // Type promotion mgiht cause performance to actually decrease.
        if (base_weight.dtype() != weight.dtype() ||
            base_weight.device() != weight.device() ||
            base_bias.dtype() != bias.dtype() ||
            base_bias.device() != bias.device()) {
          it++;
          continue;
        }

        if (!isNonZeroDimEqual(base_weight, weight) ||
            !isNonZeroDimEqual(base_bias, bias)) {
          it++;
          continue;
        }

        bool can_move_before_all = true;
        for (auto n : compatible_layers) {
          can_move_before_all &=
              getAliasDb()->couldMoveBeforeTopologically(node, n);
        }
        if (!can_move_before_all) {
          it++;
          continue;
        }

        // Found a node that is eligible for combination
        auto prev = it;
        it++;
        linear_layer_group.erase(prev);
        compatible_layers.push_back(node);
      }
      if (compatible_layers.size() == 1) {
        continue; // No other layers to merge
      }
      mergeLinearLayers(base_node, compatible_layers);
    }
  }

  void handleBlockAndSubblocks(Block* block) {
    for (auto node : block->nodes()) {
      for (Block* block : node->blocks()) {
        handleBlockAndSubblocks(block);
      }
    }

    // Processing for the block itself
    std::unordered_map<Value*, std::list<Node*>> grouped_linear_layers;
    std::vector<Value*> ordered_tensor_inputs;
    collectConstantLinearLayers(
        block, grouped_linear_layers, ordered_tensor_inputs);

    // Reverse topological ordering is used to prevent the need to
    // update the aliasDB
    for (auto tensor_it = ordered_tensor_inputs.rbegin();
         tensor_it != ordered_tensor_inputs.rend();
         ++tensor_it) {
      collectAndMergeLinearLayers(grouped_linear_layers.at(*tensor_it));
    }
  }

 public:
  bool run(std::shared_ptr<Graph> graph) {
    graph_ = graph;
    handleBlockAndSubblocks(graph_->block());
    return graph_modified;
  }
};
} // namespace

TORCH_API bool FrozenConcatLinear(std::shared_ptr<Graph>& graph) {
  ConcatLinearLayers concatLayers;
  GRAPH_DUMP("Before FrozenConcatLinear", graph);
  bool changed = concatLayers.run(graph);
  if (changed) {
    GRAPH_DUMP("After FrozenConcatLinear", graph);
  }
  return changed;
}

} // namespace jit
} // namespace torch