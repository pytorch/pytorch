#include <torch/csrc/jit/passes/constant_propagation.h>
#include <ATen/core/functional.h>
#include <ATen/core/ivalue.h>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/constants.h>
#include <torch/csrc/jit/interpreter.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/node_hashing.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/passes/alias_analysis.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

namespace torch {
namespace jit {

namespace {

std::unordered_set<Symbol> skip_list = {
    prim::If,
    prim::Loop,
    prim::Constant,
    prim::AutogradZero,
    prim::unchecked_unwrap_optional, // TODO remove
    // TODO (zach): we should consider skipping tensor factories in the cases
    // where the constant tensor would be large but cheap to create.
};

std::vector<IValue> runNode(Node* n) {
  auto op = getOperation(n);
  Stack stack;
  for (auto input : n->inputs()) {
    stack.push_back(*(toIValue(input)));
  }
  op(stack);
  auto var_outputs = fmap(stack, [&](IValue v) -> IValue {
    if (v.isTensor()) {
      auto t = std::move(v).toTensor();
      if (t.defined()) {
        if (t.requires_grad()) {
          throw c10::Error("Can't insert requires grad as constant", "");
        }
        return IValue(autograd::as_variable_ref(t).data());
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
  } catch (const c10::Error& e) {
    // catch AT_ASSERT errors. This op may not be run reached,
    // so catch the error here & leave the op in the graph
    return;
  }
  auto graph = n->owningGraph();
  WithInsertPoint guard(n);
  for (size_t i = 0; i < outputs.size(); ++i) {
    try {
      auto new_output = graph->insertConstant(outputs[i]);
      if (outputs[i].isNone()) {
        new_output->setType(n->outputs()[i]->type());
      }
      n->outputs()[i]->replaceAllUsesWith(new_output);
    } catch (constant_not_supported_error& err) {
      // we cannot actually represent the IValue as a constant node,
      // so we give up replacing it
    }
    // let dce elimination remove n
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
  return !loop_might_run;
}

void ConstantPropagation(Block* block, const AliasDb& aliasDb);

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

void inlineIf(Node* n, const AliasDb& aliasDb) {
  auto input_bool = constant_as<bool>(n->input());
  AT_ASSERT(input_bool);
  size_t block_index = *input_bool ? 0 : 1;
  ConstantPropagation(n->blocks().at(block_index), aliasDb);
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
  AT_CHECK(n->kind() == prim::If, "Only supported for If nodes");
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
      auto new_const = graph->insertConstant(*maybe_const, t_out->type());
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

void ConstantPropagation(Node* n, const AliasDb& aliasDb) {
  bool constant_inputs =
      std::all_of(n->inputs().begin(), n->inputs().end(), [&](Value* v) {
        return v->node()->kind() == prim::Constant;
      });
  bool supported_node = !n->kind().is_onnx() &&
      skip_list.count(n->kind()) == 0 && !n->isNondeterministic() &&
      !n->hasSideEffects() && !aliasDb.hasWriters(n);
  auto run_blocks = [&]() {
    for (Block* block : n->blocks()) {
      ConstantPropagation(block, aliasDb);
    }
  };
  if (n->kind() == prim::If) {
    // inline node if we can, otherwise check for simplified outputs
    if (constant_inputs) {
      inlineIf(n, aliasDb);
    } else {
      run_blocks();
      removeExtraIfOutputs(n);
    }
  } else if (n->kind() == prim::Loop) {
    if (loopWillNotRun(n)) {
      removeLoopNode(n);
    } else {
      run_blocks();
      removeExtraLoopOutputs(n);
    }
  } else if (constant_inputs && supported_node) {
    propagateNode(n);
  } else {
    run_blocks();
  }
}

void ConstantPropagation(Block* block, const AliasDb& aliasDb) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    Node* n = *it;
    it++; // advance iterator bc the current node may be destroyed
    ConstantPropagation(n, aliasDb);
  }
}
} // anonymous namespace

void ConstantPropagation(std::shared_ptr<Graph>& graph) {
  AliasDb aliasDb(graph);
  ConstantPropagation(graph->block(), aliasDb);
  EliminateDeadCode(graph);
}
} // namespace jit
} // namespace torch
