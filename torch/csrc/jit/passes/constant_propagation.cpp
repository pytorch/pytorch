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

namespace torch {
namespace jit {

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

std::unordered_set<Symbol> tuple_ops = {
    prim::TupleSlice,
    prim::TupleIndex,
    prim::TupleUnpack,
    prim::TupleConstruct,
};

struct ConstantPropagator {
  ConstantPropagator(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)), aliasDb(graph_){};

  void run() {
    ConstantPropagation(graph_->block());
  }

 private:
  void pushIValue(Value* v, Stack& stack) {
    if (tuples.count(v)) {
      const auto& ival = tuples[v];
      stack.push_back(ival);
    } else {
      stack.push_back(*toIValue(v));
    }
  }

  std::vector<IValue> runNode(Node* n) {
    auto op = getOperation(n);
    Stack stack;
    for (auto input : n->inputs()) {
      pushIValue(input, stack);
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

  // Tuples are not representable as constants, however
  // we can try to insert each tuple element and then create a TupleConstruct
  // from the elements
  Value* tryInsertTuple(const IValue& tuple, Value* tuple_to_replace) {
    auto type = tuple_to_replace->type();
    TupleTypePtr tup_type;
    if (auto opt = type->cast<OptionalType>()) {
      tup_type = opt->getElementType()->expect<TupleType>();
    } else {
      tup_type = type->expect<TupleType>();
    }
    auto type_elements = tup_type->elements();
    const auto& tuple_elements = tuple.toTuple()->elements();
    std::vector<Value*> inputs;
    for (size_t i = 0; i < type_elements.size(); ++i) {
      auto inp = tryInsertConstant(*graph_, tuple_elements[i]);
      if (inp) {
        inputs.push_back(*inp);
      } else {
        return nullptr;
      }
    }
    auto new_tuple = graph_->insertNode(graph_->createTuple(inputs));
    tuple_to_replace->replaceAllUsesWith(new_tuple->output());
    new_tuple->output()->copyMetadata(tuple_to_replace);
    return new_tuple->output();
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
      } else if (outputs[i].isTuple()) {
        // we save the new Tuple ivalue in case it is used in an op that
        // forwards tuples later in the graph, such as a Tuple index
        auto tuple_val = n->outputs()[i];
        if (auto new_tup = tryInsertTuple(outputs[i], tuple_val)) {
          GRAPH_UPDATE(
              "Folding tuple %",
              n->outputs()[i]->debugName(),
              " with ",
              getHeader(new_tup->node()));
          tuple_val = new_tup;
        }
        tuples[tuple_val] = std::move(outputs[i]);
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
    if (tuple_ops.count(n->kind())) {
      return (
          std::all_of(n->inputs().begin(), n->inputs().end(), [&](Value* v) {
            return v->node()->kind() == prim::Constant || tuples.count(v);
          }));
    }
    return false;
  };

  void ConstantPropagation(Node* n) {
    bool runnable_inputs = runnableInputs(n);
    bool supported_node = !n->kind().is_onnx() &&
        skip_list.count(n->kind()) == 0 && !n->isNondeterministic() &&
        !n->hasSideEffects() && !aliasDb.hasWriters(n);
    auto run_blocks = [&]() {
      for (Block* block : n->blocks()) {
        ConstantPropagation(block);
      }
    };
    if (n->kind() == prim::If) {
      // inline node if we can, otherwise check for simplified outputs
      if (runnable_inputs) {
        inlineIf(n);
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
    } else if (runnable_inputs && supported_node) {
      propagateNode(n);
    } else {
      run_blocks();
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
  AliasDb aliasDb;
  // these are tuples which we know the computed IValue for
  std::unordered_map<Value*, IValue> tuples;
};
} // anonymous namespace

void ConstantPropagation(std::shared_ptr<Graph>& graph) {
  ConstantPropagator cp(graph);
  cp.run();
  GRAPH_DUMP("After ConstantPropagation: ", graph);
  EliminateDeadCode(graph);
}
} // namespace jit
} // namespace torch
