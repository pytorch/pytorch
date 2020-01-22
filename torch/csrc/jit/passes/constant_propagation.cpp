#include <torch/csrc/jit/passes/constant_propagation.h>
#include <ATen/core/functional.h>
#include <ATen/core/ivalue.h>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/constants.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/node_hashing.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/passes/alias_analysis.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/utils/memory.h>

namespace torch {
namespace jit {

c10::optional<Stack> runNodeIfInputsAreConstant(const Node* node) {
  Stack stack;
  for (const Value* input : node->inputs()) {
    if (auto ival = toIValue(input)) {
      stack.push_back(ival);
    } else {
      return c10::nullopt;
    }
  }
  try {
    auto op = getOperation(node);
    op(stack);
    TORCH_INTERNAL_ASSERT(stack.size() == node->outputs().size());
  } catch (...) {
    return c10::nullopt;
  }
  return stack;
}

namespace {

std::unordered_set<Symbol> skip_list = {
    prim::If,
    prim::Loop,
    prim::Function,
    prim::Constant,
    prim::AutogradZero,
    prim::Uninitialized,
    prim::unchecked_unwrap_optional, // TODO remove
    // TODO (zach): we should consider skipping tensor factories in the cases
    // where the constant tensor would be large but cheap to create.
};

struct ConstantPropagator {
  // Runs constant propagation with an aliasing db and checks if inputs or
  // outputs might be mutated in the graph
  static ConstantPropagator WithAliasDb(std::shared_ptr<Graph> graph) {
    return ConstantPropagator(graph, true);
  }

  // Runs constant propagation only on ops that clearly do not have aliased
  // inputs or outputs without computing aliasing information
  static ConstantPropagator NoAliasDb(std::shared_ptr<Graph> graph) {
    return ConstantPropagator(graph, false);
  }

  void run() {
    ConstantPropagation(graph_->block());
  }

 private:
  ConstantPropagator(std::shared_ptr<Graph> graph, bool aliasing_types)
      : graph_(std::move(graph)) {
    if (aliasing_types) {
      aliasDb_ = torch::make_unique<AliasDb>(graph_);
    } else {
      aliasDb_ = nullptr;
    }
  }

  std::vector<IValue> runNode(Node* n) {
    auto op = getOperation(n);
    Stack stack;
    for (auto input : n->inputs()) {
      stack.push_back(*toIValue(input));
    }
    op(stack);
    auto var_outputs = fmap(stack, [&](IValue v) -> IValue {
      if (v.isTensor()) {
        auto t = std::move(v).toTensor();
        if (t.defined()) {
          if (t.requires_grad()) {
            // error gets caught within propagateNode()
            throw c10::Error("Can't insert requires grad as constant", "");
          }
          return IValue(t);
        } else {
          return t;
        }
      } else {
        return v;
      }
    });
    return var_outputs;
  }

  void propagateNode(Node* n) {
    std::vector<IValue> outputs;
    try {
      outputs = runNode(n);
    } catch (...) {
      // Catch exceptions. This op may not be run,
      // so catch the error here & leave the op in the graph
      return;
    }
    auto graph = n->owningGraph();
    WithInsertPoint guard(n);
    for (size_t i = 0; i < outputs.size(); ++i) {
      auto new_output = tryInsertConstant(*graph, outputs[i]);
      if (new_output) {
        GRAPH_UPDATE(
            "Folding %",
            n->outputs()[i]->debugName(),
            " with ",
            getHeader((*new_output)->node()));
        if (outputs[i].isNone()) {
          (*new_output)->setType(n->outputs()[i]->type());
        }
        n->outputs()[i]->replaceAllUsesWith(*new_output);
      }
      // If we cannot insert the IValue as a constant, give up replacing the
      // node and let DCE remove it
    }
  }

  void removeLoopNode(Node* n) {
    auto loop_input_offset = 2; // offset of loop carried deps in input list
    for (size_t i = 0; i < n->outputs().size(); ++i) {
      n->outputs().at(i)->replaceAllUsesWith(
          n->inputs().at(i + loop_input_offset));
    }
    n->destroy();
  }

  bool loopWillNotRun(Node* node) {
    Value* trip_count = node->inputs().at(0);
    int64_t iter_len = constant_as<int64_t>(trip_count).value_or(1);

    Value* start_cond = node->inputs().at(1);
    bool cond_val = constant_as<bool>(start_cond).value_or(true);

    bool loop_might_run = cond_val && iter_len > 0;
    if (!loop_might_run) {
      GRAPH_UPDATE(
          "Removing unexecuted loop: ",
          *node,
          "\ntripcount: ",
          trip_count,
          " and start_cond: ",
          getHeader(start_cond->node()));
    }
    return !loop_might_run;
  }

  void inlineIfBody(Block* body) {
    Node* n = body->owningNode();
    for (auto it = body->nodes().begin(); it != body->nodes().end();) {
      Node* body_node = *it;
      // advance iterator because after body_node is moved its next pointer will
      // be to n
      it++;
      body_node->moveBefore(n);
    }
    for (size_t i = 0; i < n->outputs().size(); ++i) {
      n->outputs().at(i)->replaceAllUsesWith(body->outputs().at(i));
    }
    // NB: destroy the node here, because it might contain side effects, like
    // print
    n->destroy();
  }

  void inlineIf(Node* n) {
    auto input_bool = constant_as<bool>(n->input());
    AT_ASSERT(input_bool);
    GRAPH_UPDATE(
        "Folding if ",
        getHeader(n->input()->node()),
        " where condition = ",
        *input_bool);
    size_t block_index = *input_bool ? 0 : 1;
    ConstantPropagation(n->blocks().at(block_index));
    inlineIfBody(n->blocks().at(block_index));
  }

  void replaceAndRemoveIfOutput(Node* n, size_t i, Value* replacement) {
    n->outputs().at(i)->replaceAllUsesWith(replacement);
    n->eraseOutput(i);
    n->blocks().at(0)->eraseOutput(i);
    n->blocks().at(1)->eraseOutput(i);
  }

  // remove extra outputs from the node
  bool removeExtraIfOutputs(Node* n) {
    TORCH_CHECK(n->kind() == prim::If, "Only supported for If nodes");
    auto true_block = n->blocks()[0];
    auto false_block = n->blocks()[1];
    auto graph = n->owningGraph();
    auto initial_outputs = true_block->outputs().size();
    WithInsertPoint guard(n);
    for (size_t i = 0; i < true_block->outputs().size();) {
      auto t_out = true_block->outputs().at(i);
      auto f_out = false_block->outputs().at(i);

      // neither block changes the output value
      if (true_block->outputs()[i] == false_block->outputs()[i]) {
        replaceAndRemoveIfOutput(n, i, true_block->outputs()[i]);
        continue;
      }

      // true block output is constant and constant matches false block output
      auto maybe_const = toIValue(t_out);
      auto eq = EqualNode();
      if (maybe_const && eq(t_out->node(), f_out->node())) {
        auto new_const = graph->insertConstant(*maybe_const);
        replaceAndRemoveIfOutput(n, i, new_const);
        continue;
      }

      i++; // increment bc we didn't remove current index
    }
    // an output was removed
    return initial_outputs != true_block->outputs().size();
  }

  // remove extra outputs from the node
  void removeExtraLoopOutputs(Node* node) {
    auto loop_body = node->blocks().at(0);
    auto loop_input_offset = 2; // offset of loop carried deps in input list
    auto loop_body_offset =
        1; // offset to the loop carried dependencies in block inputs/outputs
    for (size_t i_1 = node->outputs().size(); i_1 > 0; --i_1) {
      size_t i = i_1 - 1;
      // if the value is no longer changed remove output
      if (loop_body->inputs().at(loop_body_offset + i) ==
          loop_body->outputs().at(loop_body_offset + i)) {
        auto node_input = node->inputs().at(loop_input_offset + i);
        node->outputs().at(i)->replaceAllUsesWith(node_input);
        loop_body->inputs()
            .at(loop_body_offset + i)
            ->replaceAllUsesWith(node_input);
        node->eraseOutput(i);
        node->removeInput(loop_input_offset + i);
        loop_body->eraseInput(loop_body_offset + i);
        loop_body->eraseOutput(loop_body_offset + i);
      }
    }
  }

  // An Op has runnable inputs if:
  // - All inputs are constants.
  // - It is an op that forwards tuples, and all inputs are constants
  // or tuples that we know the ivalue for. We can't use known tuple ivalues
  // for non-forwarding ops because that Tuple could contain an ivalue that is
  // not allowed as a constant, for instance, a Tensor with a gradient.
  bool runnableInputs(Node* n) {
    if (std::all_of(n->inputs().begin(), n->inputs().end(), [&](Value* v) {
          return v->node()->kind() == prim::Constant;
        })) {
      return true;
    }
    return false;
  };

  bool noMutableValues(at::ArrayRef<Value*> values) {
    return std::none_of(values.begin(), values.end(), [](Value* v) {
      return AliasDb::mutableType(v);
    });
  }

  bool supportedNode(Node* n) {
    bool no_mutation;
    if (aliasDb_) {
      no_mutation = !aliasDb_->hasWriters(n);
    } else {
      no_mutation =
          noMutableValues(n->inputs()) && noMutableValues(n->outputs());
    }
    return no_mutation && !n->kind().is_onnx() &&
        skip_list.count(n->kind()) == 0 && !n->isNondeterministic() &&
        !n->hasSideEffects() && n->blocks().size() == 0;
  }

  void ConstantPropagation(at::ArrayRef<Block*> blocks) {
    for (Block* block : blocks) {
      ConstantPropagation(block);
    }
  }

  void ConstantPropagation(Node* n) {
    bool runnable_inputs = runnableInputs(n);
    if (n->kind() == prim::If) {
      // inline node if we can, otherwise check for simplified outputs
      if (runnable_inputs) {
        inlineIf(n);
      } else {
        ConstantPropagation(n->blocks());
        removeExtraIfOutputs(n);
      }
    } else if (n->kind() == prim::Loop) {
      if (loopWillNotRun(n)) {
        removeLoopNode(n);
      } else {
        ConstantPropagation(n->blocks());
        removeExtraLoopOutputs(n);
      }
    } else if (runnable_inputs && supportedNode(n)) {
      propagateNode(n);
    } else {
      ConstantPropagation(n->blocks());
    }
  }

  void ConstantPropagation(Block* block) {
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      Node* n = *it;
      it++; // advance iterator bc the current node may be destroyed
      ConstantPropagation(n);
    }
  }

  std::shared_ptr<Graph> graph_;
  std::unique_ptr<AliasDb> aliasDb_;
};
} // anonymous namespace

void ConstantPropagation(std::shared_ptr<Graph>& graph) {
  ConstantPropagator cp = ConstantPropagator::WithAliasDb(graph);
  cp.run();
  EliminateDeadCode(graph);
  GRAPH_DUMP("After ConstantPropagation: ", graph);
}

void ConstantPropagationImmutableTypes(std::shared_ptr<Graph>& graph) {
  ConstantPropagator cp = ConstantPropagator::NoAliasDb(graph);
  cp.run();
  EliminateDeadCode(graph);
  GRAPH_DUMP("After ConstantPropagation: ", graph);
}

} // namespace jit
} // namespace torch
