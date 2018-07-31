#include "torch/csrc/jit/passes/constant_propagation.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/constants.h"
#include "torch/csrc/jit/interpreter.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/ivalue.h"
#include "torch/csrc/jit/operator.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/utils/functional.h"

namespace torch { namespace jit {

namespace {

std::unordered_set<Symbol> skip_list = {
  prim::If,
  prim::Loop, //TODO: handle Loop
  //FIXME Same problem as in DCE - cpp & python PythonOp and CppOp should be
  //FIXME treated as having side effects but ONNX depends on them being removed
  prim::Print,
  //all the rand functions from native_functions.yaml
  aten::permute,
  aten::rand,
  aten::rand_out,
  aten::rand_like,
  aten::randint,
  aten::randint_out,
  aten::randint_like,
  aten::randn,
  aten::randn_out,
  aten::randn_like,
  aten::randperm,
  aten::randperm_out,
 };

std::vector<IValue> runNode(Node* n) {
  auto op = getOperation(n);
  Stack stack;
  for (auto input : n->inputs()) {
    stack.push_back(*(toIValue(input)));
  }
  op(stack);
  auto var_outputs = fmap(stack, [&](IValue v) {
    if (v.isTensor()) {
      return IValue(autograd::as_variable_ref(v.toTensor()).data());
    } else {
      return v;
    }
  });
  return var_outputs;
}

void propagateNode(Node* n) {
  auto outputs = runNode(n);
  auto graph = n->owningGraph();
  WithInsertPoint guard(n);
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto new_output = graph->insertConstant(outputs[i]);
    n->outputs()[i]->replaceAllUsesWith(new_output);
    // let dce elimination remove n
  }
}

void lowerIf(Block *body, Node * n) {
  auto graph = n->owningGraph();
  WithInsertPoint insert_point_guard { n };

  std::unordered_map<Value*, Value*> value_map;
  auto get_value = [&](Value *v) {
    auto it = value_map.find(v);
    if (it != value_map.end())
      return it->second;
    return v;
  };

  for (Node *orig : body->nodes()) {
    Node *clone = graph->insertNode(graph->createClone(orig, get_value));
    for (size_t i = 0; i < orig->outputs().size(); ++i) {
      value_map[orig->outputs()[i]] = clone->outputs()[i];
    }
  }
  for (size_t i = 0; i < n->outputs().size(); ++i) {
    n->outputs().at(i)->replaceAllUsesWith(get_value(body->outputs().at(i)));
  }
  // NB: destroy the node here, because it might contain side effects, like print
  n->destroy();
}

bool isTrueConstant(Value *val) {
  at::optional<bool> maybe_value = constant_as<bool>(val);
  return maybe_value && *maybe_value;
}

void lowerIf(Node *n) {
  if (isTrueConstant(n->input())) {
    lowerIf(n->blocks()[0], n);
  } else {
    lowerIf(n->blocks()[1], n);
  }
}

//returns true if the mutated variables are changed
bool recomputeMutatedVariables(Node *n) {
  JIT_ASSERTM(n->kind() == prim::If, "Only supported for If nodes");
  std::unordered_set<Value*> mutated_variables;
  for (Block * block : n->blocks()) {
    for (Node *n : block->nodes()) {
      for (size_t i = 0; i < n->outputs().size(); ++i) {
        mutated_variables.insert(n->outputs()[i]);
      }
    }
  }
  auto true_block = n->blocks()[0];
  auto false_block = n->blocks()[1];
  auto initial_outputs = true_block->outputs().size();
  for (size_t i = 0; i < true_block->outputs().size();) {
    //neither block mutates output i
    if (!mutated_variables.count(true_block->outputs()[i]) &&
      !mutated_variables.count(false_block->outputs()[i])) {
      n->outputs().at(i)->replaceAllUsesWith(true_block->outputs()[i]);
      n->eraseOutput(i);
      true_block->eraseOutput(i);
      false_block->eraseOutput(i);
    } else {
      i++; //increment bc we didn't remove current index
    }
  }
  //an output was removed
  return initial_outputs != true_block->outputs().size();
}

} // anonymous namespace

//returns whether the node's set of mutated variables changed
bool ConstantPropagation(Node* n, bool recurse) {
  bool constant_inputs = (n->inputs().size() > 0) &&
    std::all_of(n->inputs().begin(), n->inputs().end(), [&](Value* v) {
      return v->node()->kind() == prim::Constant;
    });
  bool supported_node = skip_list.count(n->kind()) == 0;
  auto run_blocks = [&]() {
    bool any_child = false;
    if (recurse) {
      for (Block * block : n->blocks()) {
        auto child = ConstantPropagation(block, recurse);
        any_child = any_child || child;
      }
    }
    return any_child;
  };
  if (n->kind() == prim::If) {
    //did a child node change
    bool changed = run_blocks();
    //inline node if we can, otherwise if a child node changed recompute
    //mutated variables and see if this node changed
    if (constant_inputs) {
      lowerIf(n);
    } else if (changed) {
      changed = recomputeMutatedVariables(n);
    }
    return constant_inputs || changed;
  } else if (constant_inputs && supported_node) {
    propagateNode(n);
  }
  //TODO handle loop nodes. Even if a loop node contains an if that is
  //inlined its mutated variables currently don't get updated
  run_blocks();
  return false;
}

//returns true if the mutated variables in this block's node have changed
bool ConstantPropagation(Block* block, bool recurse) {
  ConstantPropagation(block->param_node(), recurse);
  bool any_child = false;
  for(auto it = block->nodes().begin(); it != block->nodes().end();) {
    Node *n = *it;
    it++; //advance iterator bc the current node may be destroyed
    auto child = ConstantPropagation(n, recurse);
    any_child = any_child || child;
  }
  return any_child;
}

void ConstantPropagation(std::shared_ptr<Graph>& graph) {
  ConstantPropagation(graph->block(), true);
  EliminateDeadCode(graph);
}

}}
