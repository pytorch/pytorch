#include <torch/csrc/jit/tensorexpr/graph_opt.h>

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/symbolic_shape_registry_util.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>

namespace torch::jit::tensorexpr {

// Move the given user of `aten::cat` op to its inputs.
static Node* moveCatAfterUse(
    Node* cat,
    Node* user,
    std::shared_ptr<Graph> subgraph) {
  // Example IR:
  //   %1 = ...
  //   %2 = ...
  //   %3 = prim::ListConstruct(%1, %2)
  //   %4 = aten::cat(%3, ...)
  //   %5 = aten::relu(%4)
  //   return (%5)
  //
  // To be transformed to:
  //   %1 = ...
  //   %2 = ...
  //   %5.1 = aten::relu(%1)
  //   %5.2 = aten::relu(%2)
  //   %3 = prim::ListConstruct(%5.1, %5.2)
  //   %4 = aten::cat(%3, ...)
  //   return (%4)

  TORCH_INTERNAL_ASSERT(
      cat->output()->hasUses(),
      buildErrorMessage("aten::cat output is not used."));
  TORCH_INTERNAL_ASSERT(
      cat->output()->uses().size() == 1,
      buildErrorMessage("aten::cat output is used in multiple places."));
  TORCH_INTERNAL_ASSERT(
      cat->input(0)->node()->kind() == prim::ListConstruct,
      buildErrorMessage("aten::cat inputs are not expected."));
  auto cat_list = cat->input(0)->node();
  auto cat_inputs = cat_list->inputs();

  auto user_tensor_type = user->output()->type()->cast<c10::TensorType>();
  TORCH_INTERNAL_ASSERT(
      user_tensor_type, buildErrorMessage("Unexpected user tensor type"));
  std::unordered_map<Value*, Value*> new_cat_inputs;
  for (auto inp : cat_inputs) {
    auto new_cat_input = subgraph->createClone(
        user, [&](Value* k) { return (k == cat->output()) ? inp : k; });
    // Since we are cloning user, its result should be the same scalar type
    // as the user. But the dims should correspond to that of the input.
    auto input_tensor_type = inp->type()->cast<c10::TensorType>();
    TORCH_INTERNAL_ASSERT(
        input_tensor_type, buildErrorMessage("Unexpected input tensor type"));
    auto new_input_type =
        input_tensor_type->withScalarType(user_tensor_type->scalarType());
    new_cat_input->output()->setType(new_input_type);
    new_cat_input->insertBefore(cat_list);
    new_cat_inputs[inp] = new_cat_input->output();
  }
  auto new_cat_list = subgraph->createClone(
      cat_list, [&](Value* k) { return new_cat_inputs[k]; });
  new_cat_list->insertBefore(cat);
  auto new_cat = subgraph->createClone(cat, [&](Value* k) {
    return (k == cat_list->output()) ? new_cat_list->output() : k;
  });
  new_cat->output()->setType(user_tensor_type);
  new_cat->insertBefore(cat);

  user->output()->replaceAllUsesWith(new_cat->output());
  user->destroy();

  TORCH_INTERNAL_ASSERT(
      !cat->output()->hasUses(),
      buildErrorMessage("aten::cat output is not used."));
  cat->destroy();

  if (!cat_list->output()->hasUses()) {
    cat_list->destroy();
  }

  return new_cat;
}

static int numTensorInputs(Node* node) {
  int count = 0;
  for (auto v : node->inputs()) {
    if (v->type()->cast<c10::TensorType>()) {
      ++count;
    }
  }
  return count;
}

// Returns true if the given `cat` node promotes types.
// If the inputs to `cat` are of different types, then the implementation
// of `cat` is expected to promote type.
static bool doesCatPromoteTypes(Node* node) {
  TORCH_INTERNAL_ASSERT(
      node->kind() == aten::cat,
      buildErrorMessage("Graph node is not aten::cat."));
  TORCH_INTERNAL_ASSERT(
      node->input(0)->node()->kind() == prim::ListConstruct,
      buildErrorMessage("aten::cat inputs are not expected."));
  auto inputs = node->input(0)->node()->inputs();
  TORCH_INTERNAL_ASSERT(
      !inputs.empty(), buildErrorMessage("Empty inputs of ListConstruct"));
  auto scalar_type =
      inputs.front()->type()->cast<c10::TensorType>()->scalarType();
  for (size_t i = 1; i < inputs.size(); ++i) {
    auto inp_scalar_type =
        inputs[i]->type()->cast<c10::TensorType>()->scalarType();
    if (scalar_type != inp_scalar_type) {
      return true;
    }
  }
  return false;
}

// Move the users of the given `aten::cat` op to its inputs.
// The following constraints need to be satisfied on the cat op and its user.
//   * the cat op should have only one use.
//   * the user should be an element-wise op.
//   * the user should have only one tensor input.
//     - If the user has > 1 tensor inputs, that user op cannot be applied on
//       the inputs of cat because the other tensor inputs will not be split,
//       and hence the shape of those tensors would not match that of the
//       inputs of cat.
//       For example:
//           %1 = ...
//           %2 = ...
//           %3 = prim::ListConstruct([%1, %2])
//           %4 = aten::cat(%3, ...)
//           %5 = aten::add(%4, %0)
//       In this example, we cannot move `aten::add` to the inputs of
//       `aten::cat`, %1 and %2, because the shape of %0 will be different.
//    * the cat op does not promote types.
//      - When the cat op promote types, the type of inputs to cat after moving
//        it user needs to reflect the original type. This is currently not
//        handled. TODO
static void moveCatOpToEnd(Node* cat, std::shared_ptr<Graph> subgraph) {
  TORCH_INTERNAL_ASSERT(
      cat->kind() == aten::cat,
      buildErrorMessage("Graph node is not aten::cat."));
  if (cat->output()->uses().size() == 1) {
    auto use = cat->output()->uses().front();
    if (get_tensorexpr_elementwise_set().contains(use.user) &&
        numTensorInputs(use.user) == 1) {
      if (!doesCatPromoteTypes(cat)) {
        TORCH_INTERNAL_ASSERT(
            use.user->output()->owningGraph() == subgraph.get(),
            buildErrorMessage(
                "aten::cat user graph does not math the given subgraph."));
        auto new_cat = moveCatAfterUse(cat, use.user, subgraph);
        moveCatOpToEnd(new_cat, subgraph);
      }
    }
  }
}

// Moves the users of `aten::cat` ops to its inputs whenever possible
// in the given subgraph.
static void moveCatOpsToEnd(std::shared_ptr<Graph> subgraph) {
  std::vector<Node*> cat_nodes;
  for (Node* n : subgraph->nodes()) {
    if (n->kind() == aten::cat) {
      cat_nodes.push_back(n);
    }
  }
  for (auto cat : cat_nodes) {
    moveCatOpToEnd(cat, subgraph);
  }
}

bool OptimizeCat(const std::shared_ptr<Graph>& graph) {
  if (getCatWoConditionals()) {
    moveCatOpsToEnd(graph);
    return true;
  }
  return false;
}

void annotateInputShapes(
    const std::shared_ptr<Graph>& graph,
    const std::vector<c10::optional<at::Tensor>>& example_inputs) {
  TORCH_INTERNAL_ASSERT(
      graph->inputs().size() == example_inputs.size(),
      buildErrorMessage("Given inputs do not match the fuser graph inputs."));
  for (size_t idx = 0; idx < example_inputs.size(); idx++) {
    if (auto t = example_inputs[idx]) {
      auto concrete_tensor_type = tensorTypeInCurrentExecutionContext(*t);
      graph->inputs().at(idx)->setType(concrete_tensor_type);
    }
  }
}

std::shared_ptr<Graph> removeUnusedSelfArgument(
    const std::shared_ptr<Graph>& graph) {
  if (graph->inputs().empty()) {
    return graph;
  }
  jit::Value* self_argument = graph->inputs().at(0);
  if (!self_argument->uses().empty() || !self_argument->type()->is_module()) {
    return graph;
  }
  graph->eraseInput(0);
  return graph;
}

std::vector<int64_t> makeShapesSymbolic(
    std::shared_ptr<Graph>& graph,
    const std::vector<int64_t>& size_vals) {
  std::unordered_set<Value*> values;
  for (auto v : graph->inputs()) {
    values.insert(v);
  }
  for (auto v : graph->outputs()) {
    values.insert(v);
  }
  for (auto n : graph->nodes()) {
    for (auto v : n->inputs()) {
      values.insert(v);
    }
    for (auto v : n->outputs()) {
      values.insert(v);
    }
  }
  std::unordered_map<int64_t, int64_t> shape_to_sym_shape;
  std::vector<int64_t> new_syms;
  for (int64_t size_val : size_vals) {
    auto new_shape_symbol = at::ShapeSymbol::newSymbol().value();
    shape_to_sym_shape[size_val] = new_shape_symbol;
    new_syms.push_back(new_shape_symbol);
    graph->addInput("sym_shape")->setType(IntType::get());
  }

  for (auto v : values) {
    if (!v->type()->cast<TensorType>()) {
      continue;
    }
    auto tt = v->type()->expect<TensorType>();
    if (!tt->symbolic_sizes().sizes()) {
      continue;
    }
    std::vector<at::ShapeSymbol> shape_vec = *tt->symbolic_sizes().sizes();

    auto new_sizes = c10::fmap(shape_vec, [&](const at::ShapeSymbol& shape) {
      auto value = shape.value();
      if (shape_to_sym_shape.count(value)) {
        return shape_to_sym_shape.at(value);
      }
      return value;
    });
    v->setType(tt->withSymbolicShapes(c10::SymbolicShape(new_sizes)));
  }

  return new_syms;
}

bool isGraphCompilable(const std::shared_ptr<Graph>& graph) {
  for (auto input : graph->inputs()) {
    auto const& t = input->type();
    auto const& k = t->kind();
    if (k != TypeKind::TensorType && k != TypeKind::FloatType &&
        k != TypeKind::BoolType && k != TypeKind::IntType) {
      GRAPH_DEBUG("Input %", input->debugName(), " has unsupported type ", *t);
      return false;
    }
  }

  for (auto n : graph->nodes()) {
    for (auto v : n->inputs()) {
      auto const& t = v->type();
      if (t->kind() == TypeKind::TensorType) {
        auto tt = t->cast<TensorType>();
        if (!tt->isComplete()) {
          GRAPH_DEBUG(
              "%",
              v->debugName(),
              " is not a complete tensor! The type is: ",
              *t);
          return false;
        }
      }
    }
    for (auto v : n->outputs()) {
      auto const& t = v->type();
      if (t->kind() == TypeKind::TensorType) {
        auto tt = t->cast<TensorType>();
        if (!tt->isComplete()) {
          GRAPH_DEBUG(
              "%", v->debugName(), " is not a complete! The type is: ", *t);
          return false;
        }
      }
    }
  }

  // TODO: check if all nodes have lowerings
  return true;
}

static void fixupTypeInfoForValue(
    Value* v,
    c10::optional<at::ScalarType> scalar_type,
    c10::optional<at::Device> device) {
  Node* n = v->node();
  auto const& t = v->type();
  if (t->kind() != TypeKind::TensorType) {
    return;
  }

  if (n->kind() == prim::Constant) {
    auto const_tensor = toIValue(v)->toTensor();
    auto concrete_tensor_type =
        tensorTypeInCurrentExecutionContext(const_tensor);
    v->setType(concrete_tensor_type);
    return;
  }

  TensorTypePtr new_tt;
  auto tt = t->cast<TensorType>();
  auto sizes = tt->sizes();
  if (!sizes.concrete_sizes()) {
    GRAPH_DEBUG("No concrete sizes for %", v->debugName());
    return;
  }
  auto strides = tt->strides();
  auto dtype = tt->scalarType() ? tt->scalarType() : scalar_type;
  auto concrete_sizes = *sizes.concrete_sizes();
  auto concrete_strides = strides.concrete_sizes()
      ? *strides.concrete_sizes()
      : TensorType::contiguousStridesOf(concrete_sizes);
  new_tt = TensorType::create(
      dtype, device, concrete_sizes, concrete_strides, false);

  v->setType(new_tt);
}

static c10::optional<at::ScalarType> inferScalarType(Node* n) {
  c10::optional<at::ScalarType> scalar_type;
  for (auto v : n->inputs()) {
    auto const& t = v->type();
    if (t->kind() == TypeKind::TensorType) {
      auto tt = t->cast<TensorType>();
      if (!scalar_type) {
        scalar_type = tt->scalarType();
      }
      if (tt->scalarType() && *tt->scalarType() != scalar_type) {
        GRAPH_DEBUG(
            "Inputs of ", n, " have different scalar types, cannot fixup!");
        return c10::nullopt;
      }
    }
  }
  return scalar_type;
}

static c10::optional<at::Device> inferDevice(Node* n) {
  c10::optional<at::Device> device;
  for (auto v : n->inputs()) {
    auto const& t = v->type();
    if (t->kind() == TypeKind::TensorType) {
      auto tt = t->cast<TensorType>();
      if (!device) {
        device = tt->device();
      }
      if (tt->device() && *tt->device() != device) {
        GRAPH_DEBUG("Inputs of ", n, " have different devices, cannot fixup!");
        return c10::nullopt;
      }
    }
  }
  if (!device) {
    device = at::kCPU;
  }
  return device;
}

void fixupMissingShapeInfo(const std::shared_ptr<Graph>& graph) {
  for (auto input : graph->inputs()) {
    auto const& t = input->type();
    if (t->kind() == TypeKind::TensorType) {
      auto tt = t->cast<TensorType>();
      if (!tt->scalarType()) {
        GRAPH_DEBUG("No dtype for %", input->debugName());
        return;
      }
      fixupTypeInfoForValue(
          input, *tt->scalarType(), tt->device() ? *tt->device() : at::kCPU);
    }
  }

  for (auto n : graph->nodes()) {
    c10::optional<at::ScalarType> scalar_type = inferScalarType(n);
    c10::optional<at::Device> device = inferDevice(n);

    for (auto v : n->outputs()) {
      fixupTypeInfoForValue(v, scalar_type, device);
    }
  }
}

std::shared_ptr<Graph> removeGraphOutput(
    const std::shared_ptr<Graph>& graph,
    size_t idx) {
  graph->eraseOutput(idx);
  return graph;
}

std::shared_ptr<Graph> replaceListOutputWithTuple(
    const std::shared_ptr<Graph>& graph) {
  auto out = graph->outputs()[0];
  auto out_node = out->node();
  if (out_node->kind() != prim::ListConstruct) {
    return graph;
  }
  auto tuple_node = graph->createTuple(out_node->inputs());
  tuple_node->insertAfter(out_node);
  out->replaceAllUsesWith(tuple_node->output());
  return graph;
}

static bool trimGraphOnce(const std::shared_ptr<Graph>& graph) {
  Node* ret = graph->return_node();
  std::unordered_set<Value*> graph_inputs(
      graph->inputs().begin(), graph->inputs().end());
  std::unordered_set<Value*> outputs(
      graph->outputs().begin(), graph->outputs().end());
  bool changed = false;
  for (size_t idx = 0; idx < ret->inputs().size(); idx++) {
    auto v = ret->inputs()[idx];
    if (graph_inputs.count(v)) {
      continue;
    }
    // Delete the graph output IDX and add all inputs of the node producing that
    // value to the graph outputs
    graph->eraseOutput(idx);
    for (auto v_ins : v->node()->inputs()) {
      if (outputs.count(v_ins)) {
        continue;
      }
      if (v_ins->node()->kind() == prim::Constant) {
        continue;
      }

      graph->registerOutput(v_ins);
    }
    changed = true;
    break;
  }
  return changed;
}

static std::shared_ptr<Graph> dequantizeResults(
    const std::shared_ptr<Graph>& graph) {
  for (auto v : graph->outputs()) {
    auto& t = v->type();
    if (t->kind() == TypeKind::TensorType) {
      auto tt = t->cast<TensorType>();
      if (!tt->scalarType() || !c10::isQIntType(*tt->scalarType())) {
        continue;
      }
      Node* deq = graph->create(aten::dequantize, {v});
      graph->appendNode(deq);
      deq->output()->setType(tt->withScalarType(c10::kFloat));
      v->replaceAllUsesAfterNodeWith(deq, deq->output());
    }
  }
  return graph;
}

std::shared_ptr<Graph> trimGraph(
    const std::shared_ptr<Graph>& graph,
    int64_t iters) {
  bool changed = true;
  int64_t iter = 0;
  while (changed && iter++ < iters) {
    changed = trimGraphOnce(graph);
    EliminateDeadCode(graph->block());
  }
  // Avoid letting quantized values to graph outputs.
  // Ideally we should allow quantized outputs as well, but currently the main
  // user of this pass - AOT NNC - does not support it.
  // TODO: remove output dequantization once NNC supports quantized outputs.
  dequantizeResults(graph);
  return graph;
}

} // namespace torch::jit::tensorexpr
