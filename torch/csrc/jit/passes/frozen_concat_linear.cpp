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
      // Grouping together all linear layers that use the same
      // Tensor for input, so that we can do a first pass on
      // which linear layers should be compatible for jjoining
      if (n->kind() != aten::linear) {
        continue;
      }
      
      // Punting on support for non-constant params for now.
      if(nonConstantParameters(n)){
        continue;
      }

      Value* op_value = n->inputs().at(0);
      auto weight = n->namedInput("weight");
      auto bias = n->namedInput("bias");

      // make sure that weights can be cast as tensors and are not just opaque
      // pointers
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

    // Find a valid group of nodes to concat together
    for (auto& kv : tensorUsers) {
      Value* op_value = kv.first;
      auto& uses = kv.second;

      while (!uses.empty()) {
        auto it_first = uses.begin();
        Node* base_node = *it_first;
        uses.erase(it_first);

        auto weight_type = base_node->namedInput("weight")->type();
        auto bias_type = base_node->namedInput("bias")->type();
        std::vector<Node*> matching_nodes;
        matching_nodes.push_back(base_node);

        // Now iterate over the rest of the users of the set to
        // see if there is anything that we can coaleasce with.
        for (auto it = uses.begin(); it != uses.end();) {
          auto candidate_node = *it;
          auto candidate_weight_t =
              candidate_node->namedInput("weight")->type();
          auto candidate_bias_t = candidate_node->namedInput("bias")->type();

          // For now we will just keep it simple and require matching types
          // Type promotion mgiht cause performance to actually decrease.
          if (*candidate_weight_t != *weight_type ||
              *candidate_bias_t != *bias_type) {
            it++;
            continue;
          }
          // Now use AliasDB to check that the weights and the biases, and the
          // node can be moved topologically. Need to check against all
          // candidate nodes, because there could be a weird weight of candidate
          // 3 depends on the output of candidate 2
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

        for (auto n : matching_nodes) {
          if (n == base_node) {
            continue;
          }
          getAliasDb()->moveBeforeTopologicallyValid(base_node, n);
        }

        // Now actually merge the candidate nodes
        Value* zero = graph_->insertConstant(0);
        Value* one = graph_->insertConstant(1);
        auto weight_list = c10::fmap(
            matching_nodes, [](Node* n) { return n->namedInput("weight"); });
        auto weight_with_dim(weight_list);
        weight_with_dim.push_back(zero);

        auto bias_with_dim = c10::fmap(
            matching_nodes, [](Node* n) { return n->namedInput("bias"); });
        bias_with_dim.push_back(zero);

        // auto weight_list_n = graph_->createList(weight_type, weight_list);
        // weight_list_n->insertBefore(base_node);
        auto weight_cat = graph_->create(prim::VarConcat, weight_with_dim);
        weight_cat->insertBefore(base_node);
        auto bias_cat = graph_->create(prim::VarConcat, bias_with_dim);
        bias_cat->insertBefore(base_node);

        // create the new node
        std::vector<Value*> linear_in = {
            op_value, weight_cat->output(), bias_cat->output()};
        auto linear_node = graph_->create(aten::linear, linear_in);
        linear_node->insertBefore(base_node);

        // Edit the outputs
        long long cur_loc = 0;
        Value* cur_val = zero;
        for (auto& orig_node : matching_nodes) {
          // Tensor out_tensor = constant_as<Tensor>(orig_node->output()).value();
          // long long next_loc = cur_loc + out_tensor.size(0);
          Tensor weight_tensor = constant_as<Tensor>(orig_node->namedInput("weight")).value();
          long long next_loc = cur_loc + weight_tensor.size(1);

          Value* next_val = graph_->insertConstant(next_loc);

          auto slice = graph_->create(
              aten::slice,
              {linear_node->output(), zero, cur_val, next_val, one});
          slice->insertAfter(linear_node);

          cur_loc = next_loc;
          cur_val = next_val;
          orig_node->replaceAllUsesWith(slice);
          orig_node->destroy();
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