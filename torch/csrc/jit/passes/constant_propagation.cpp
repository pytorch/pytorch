#include <torch/csrc/jit/passes/constant_propagation.h>

#include <ATen/core/ivalue.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/node_hashing.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>

#include <utility>

namespace torch::jit {

std::optional<std::vector<IValue>> runNodeIfInputsAreConstant(
    const Node* n,
    bool ignore_custom_classes,
    AliasDb* db) {
  Stack stack;
  for (auto input : n->inputs()) {
    if (auto ival = toIValue(input)) {
      stack.push_back(*ival);
    } else {
      return std::nullopt;
    }
  }

  switch (n->kind()) {
    case prim::ListUnpack: {
      if (stack.back().toList().size() != n->outputs().size()) {
        return std::nullopt;
      }
      listUnpack(stack, n->outputs().size());
    } break;
    case prim::TupleConstruct: {
      auto tt = n->output()->type()->expect<TupleType>();
      if (tt->name()) {
        namedTupleConstruct(stack, std::move(tt), n->inputs().size());
      } else {
        tupleConstruct(stack, n->inputs().size());
      }
    } break;
    case prim::ListConstruct: {
      listConstruct(
          stack,
          n->output()->type()->expectRef<ListType>(),
          n->inputs().size());
    } break;
    case prim::DictConstruct: {
      dictConstruct(
          stack,
          n->output()->type()->expectRef<DictType>(),
          n->inputs().size());
    } break;
    case prim::CreateObject: {
      createObject(
          stack,
          n->output()->type()->expect<ClassType>(),
          /*use_weak_ref*/ true);
    } break;
    case prim::GetAttr: {
      auto attr = pop(stack).toObject()->getAttr(n->s(attr::name));
      push(stack, attr);
    } break;
    case prim::isinstance: {
      isinstance(stack, n->tys(attr::types));
    } break;
    default: {
      const auto maybe_schema = n->maybeSchema();
      if (maybe_schema && maybe_schema->is_vararg()) {
        // vararg schemas require the number of inputs at the top of the stack
        // but this is broken in other places in constant prop, so disable it
        // for now
        return std::nullopt;
      }

      try {
        auto op = n->getOperation();
        op(stack);
      } catch (...) {
        return std::nullopt;
      }
    } break;
  }

  for (IValue& v : stack) {
    if (v.isTensor()) {
      const at::Tensor& t = v.toTensor();
      if (t.defined() && t.requires_grad()) {
        // requires grad tensors cannot be constants
        return std::nullopt;
      }
    }
    // Weak form of const propagation
    if (ignore_custom_classes) {
      if (v.isCustomClass()) {
        return std::nullopt;
      }
    }
    // see [Constant Object Weak CompilationUnit Reference]
    if (v.isCustomClass()) {
      if (v.toObject()->is_weak_compilation_ref()) {
        continue;
      }
      if (!db) {
        continue;
      }
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      Node* n_non_const = const_cast<Node*>(n);
      if (db->mayContainAlias(
              n_non_const->inputs(), {n_non_const->outputs()})) {
        continue;
      }
      auto obj = v.toObject();
      obj->unsafe_make_weak_compilation_ref();
    }
    if (v.isObject()) {
      if (!v.toObject()->is_weak_compilation_ref()) {
        return std::nullopt;
      }
    }
  }
  return stack;
}

namespace {

std::unordered_set<Symbol> skip_list = {
    prim::If,
    prim::Loop,
    prim::Closure,
    prim::Constant,
    prim::AutogradZero,
    prim::Uninitialized,
    prim::Guard,
    prim::profile,
    prim::profile_ivalue,
    prim::unchecked_unwrap_optional, // TODO remove
    prim::awaitable,
    aten::dequantize,
    // TODO (zach): we should consider skipping tensor factories in the cases
    // where the constant tensor would be large but cheap to create.
};

struct ConstantPropagator {
  // Runs constant propagation with an aliasing db and checks if inputs or
  // outputs might be mutated in the graph
  static ConstantPropagator WithAliasDb(
      std::shared_ptr<Graph> graph,
      bool ignore_custom_classes) {
    return ConstantPropagator(std::move(graph), true, ignore_custom_classes);
  }

  // Runs constant propagation only on ops that clearly do not have aliased
  // inputs or outputs without computing aliasing information
  static ConstantPropagator NoAliasDb(std::shared_ptr<Graph> graph) {
    return ConstantPropagator(std::move(graph), false, false);
  }

  bool run() {
    ConstantPropagation(graph_->block());
    return made_change_;
  }

 private:
  ConstantPropagator(
      std::shared_ptr<Graph> graph,
      bool aliasing_types,
      bool ignore_custom_classes)
      : graph_(std::move(graph)),
        aliasing_types_(aliasing_types),
        ignore_custom_classes_(ignore_custom_classes) {}

  void propagateNode(Node* n) {
    std::vector<IValue> outputs;
    if (auto outputs_opt =
            runNodeIfInputsAreConstant(n, ignore_custom_classes_)) {
      outputs = std::move(outputs_opt.value());
    } else {
      // The op failed to run, so we cannot continue constant-prop for it.
      return;
    }
    auto graph = n->owningGraph();
    WithInsertPoint guard(n);
    for (const auto i : c10::irange(outputs.size())) {
      auto new_output = tryInsertConstant(*graph, outputs[i]);
      if (new_output) {
        made_change_ = true;
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
    made_change_ = true;
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
    made_change_ = true;
  }

  void replaceAndRemoveIfOutput(Node* n, size_t i, Value* replacement) {
    n->outputs().at(i)->replaceAllUsesWith(replacement);
    n->eraseOutput(i);
    n->blocks().at(0)->eraseOutput(i);
    n->blocks().at(1)->eraseOutput(i);
  }

  // remove extra outputs from the node
  void removeExtraIfOutputs(Node* n) {
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
    made_change_ |= initial_outputs != true_block->outputs().size();
  }

  // remove extra outputs from the node
  void removeExtraLoopOutputs(Node* node) {
    auto initial_outputs = node->outputs().size();
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
    made_change_ |= initial_outputs != node->outputs().size();
  }

  bool noMutableValues(at::ArrayRef<Value*> values) {
    return std::none_of(values.begin(), values.end(), [](Value* v) {
      return AliasDb::isMutableType(v);
    });
  }

  AliasDb* getOrCreateAliasDb() {
    if (!aliasDb_) {
      aliasDb_ = std::make_unique<AliasDb>(graph_);
    }
    return aliasDb_.get();
  }

  bool supportedNode(Node* n) {
    bool no_mutation = false;
    if (aliasing_types_) {
      no_mutation = !getOrCreateAliasDb()->hasWriters(n);
    } else {
      no_mutation =
          noMutableValues(n->inputs()) && noMutableValues(n->outputs());
    }
    return no_mutation && !n->kind().is_onnx() &&
        skip_list.count(n->kind()) == 0 && !n->isNondeterministic() &&
        !n->hasSideEffects() && n->blocks().empty();
  }

  void ConstantPropagation(at::ArrayRef<Block*> blocks) {
    for (Block* block : blocks) {
      ConstantPropagation(block);
    }
  }

  void ConstantPropagation(Node* n) {
    bool constant_inputs =
        std::all_of(n->inputs().begin(), n->inputs().end(), [&](Value* v) {
          return v->node()->kind() == prim::Constant;
        });
    if (n->kind() == prim::If) {
      // inline node if we can, otherwise check for simplified outputs
      if (constant_inputs) {
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
    } else if (constant_inputs && supportedNode(n)) {
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
  // lazily initialized if using aliasing_types, otherwise not initialized
  std::unique_ptr<AliasDb> aliasDb_ = nullptr;
  bool aliasing_types_;
  bool made_change_ = false;
  bool ignore_custom_classes_;
};
} // anonymous namespace

bool ConstantPropagation(
    std::shared_ptr<Graph>& graph,
    bool ignore_custom_classes) {
  ConstantPropagator cp =
      ConstantPropagator::WithAliasDb(graph, ignore_custom_classes);
  bool made_change = cp.run();
  if (made_change) {
    EliminateDeadCode(graph);
  }
  GRAPH_DUMP("After ConstantPropagation: ", graph);
  return made_change;
}

bool ConstantPropagationImmutableTypes(std::shared_ptr<Graph>& graph) {
  ConstantPropagator cp = ConstantPropagator::NoAliasDb(graph);
  bool made_change = cp.run();
  if (made_change) {
    EliminateDeadCode(graph);
  }
  GRAPH_DUMP("After ConstantPropagationImmutableTypes: ", graph);
  return made_change;
}

} // namespace torch::jit
