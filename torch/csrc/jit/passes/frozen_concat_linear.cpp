#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/frozen_concat_linear.h>
#include <torch/csrc/jit/passes/frozen_conv_folding.h>
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
#include <torch/csrc/jit/passes/remove_dropout.h>
#include <torch/csrc/jit/passes/utils/optimization_pass.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/utils/memory.h>
#include <vector>
#include "ATen/core/functional.h"
#include "ATen/core/interned_strings.h"

namespace torch {
namespace jit {
namespace {

using Tensor = at::Tensor;

class ConcatLinearLayers : public torch::jit::OptimizationPass {
  using torch::jit::OptimizationPass::OptimizationPass;
  void handleBlock(Block* b) override {
    // We are using an ordered list so that one knows that
    // one only needs to move arguments forward to be a valid move, not
    // backwards.
    std::unordered_map<Value*, std::list<Node*>> tensorUsers;

    for (Node* n : b->nodes()) {
      // Grouping together all linear layers that use the same Tensor for input
      if (n->kind() != aten::linear) {
        continue;
      }
      
      // Punting on support for non-constant params for now.
      // We really just need to know that the shape of the weights and biases are constant.
      if(nonConstantParameters(n)){
        continue;
      }

      Value* op_value = n->inputs().at(0);
      auto weight = n->namedInput("weight");
      auto bias = n->namedInput("bias");

      if (!constant_as<Tensor>(weight).has_value() ||
          !constant_as<Tensor>(bias).has_value()) {
        continue;
      }
      if (weight->type() == NoneType::get() ||
          bias->type() == NoneType::get()) {
        continue;
      }

      if (tensorUsers.find(op_value) == tensorUsers.cend()) {
        tensorUsers.insert({op_value, std::list<Node*>()});
      }
      tensorUsers.find(op_value)->second.push_back(n);
    }

    // Check the uses of a tensor to find ones that can be combined
    for (auto& kv : tensorUsers) {
      Value* op_value = kv.first;
      auto& uses = kv.second;

      while (!uses.empty()) {
        auto it_first = uses.begin();
        Node* base_node = *it_first;
        uses.erase(it_first);

        auto weight_type = constant_as<Tensor>(base_node->namedInput("weight"))->dtype();
        auto bias_type = constant_as<Tensor>(base_node->namedInput("bias"))->dtype();
        std::vector<Node*> matching_nodes;
        matching_nodes.push_back(base_node);

        // Now iterate over the rest of the users of the set to
        // see if there is anything that we can coaleasce with.
        for (auto it = uses.begin(); it != uses.end();) {
          auto candidate_node = *it;
          auto candidate_weight = constant_as<Tensor>(candidate_node->namedInput("weight"));
          auto candidate_weight_type = candidate_weight->dtype();
          auto candidate_bias = constant_as<Tensor>(candidate_node->namedInput("bias"));
          auto candidate_bias_type = candidate_bias->dtype();

          // For now we will just keep it simple and require matching types
          // Type promotion mgiht cause performance to actually decrease.
          if (weight_type != candidate_weight_type || bias_type != candidate_bias_type) {          
            it++;
            continue;
          }

          bool can_move_before_all = true;
          for (auto n : matching_nodes) {
            can_move_before_all &=
                getAliasDb()->couldMoveBeforeTopologically(candidate_node, n);
          }
          if (!can_move_before_all) {
            it++;
            continue;
          }

          // Found a node that is eligible for combination
          auto prev = it;
          it++;
          uses.erase(prev);
          matching_nodes.push_back(candidate_node);
        }
        if (matching_nodes.size() == 1) {
          // Found no other compatible linear layers, do nothing.
          continue;
        }
        graph_modified = true;

        for (auto n : matching_nodes) {
          if (n == base_node) {
            continue;
          }
          getAliasDb()->moveBeforeTopologicallyValid(base_node, n);
        }

        // Now actually merge the candidate nodes
        graph_->setInsertPoint(base_node);
        Value* zero = graph_->insertConstant(0);

        auto weight_list = c10::fmap(
            matching_nodes, [](Node* n) { return n->namedInput("weight"); });
        auto weight_with_dim(weight_list);
        weight_with_dim.push_back(zero);

        auto bias_with_dim = c10::fmap(
            matching_nodes, [](Node* n) { return n->namedInput("bias"); });
        bias_with_dim.push_back(zero);

        auto weight_cat = graph_->create(prim::VarConcat, weight_with_dim);
        weight_cat->insertBefore(base_node);
        auto bias_cat = graph_->create(prim::VarConcat, bias_with_dim);
        bias_cat->insertBefore(base_node);

        // create the new node
        std::vector<Value*> linear_in = {
            op_value, weight_cat->output(), bias_cat->output()};
        auto linear_node = graph_->create(aten::linear, linear_in);
        linear_node->insertBefore(base_node);

        // Update the outputs of the nodes
        graph_->setInsertPoint(linear_node);
        Value* neg1 = graph_->insertConstant(-1);
        Value* one = graph_->insertConstant(1);

        long long cur_loc = 0;
        Value* cur_val = zero;

        for (auto& orig_node : matching_nodes) {
          Tensor weight_tensor = constant_as<Tensor>(orig_node->namedInput("weight")).value();
          long long next_loc = cur_loc + weight_tensor.size(0);
          Value* next_val = graph_->insertConstant(next_loc);
            
          auto slice = graph_->create(
              aten::slice,
              {linear_node->output(), neg1, cur_val, next_val, one});
          slice->insertAfter(linear_node);
          orig_node->replaceAllUsesWith(slice);
          orig_node->destroy();

          cur_loc = next_loc;
          cur_val = next_val;
        }
      }
    }
  }
};
} // namespace

TORCH_API bool FrozenConcatLinear(std::shared_ptr<Graph>& graph) {
  ConcatLinearLayers concatLayers(graph);
  GRAPH_DUMP("Before FrozenConcatLinear", graph);
  bool changed = concatLayers.run();
  if (changed) {
    GRAPH_DUMP("After FrozenConcatLinear", graph);
  }
  return changed;
}

} // namespace jit
} // namespace torch