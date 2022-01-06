#include <ATen/core/functional.h>
#include <ATen/core/interned_strings.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/symbolic_shape_runtime_fusion.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/register_ops_utils.h>
#include <sstream>

namespace torch {
namespace jit {

// Inserts the Compute for Each Symbolic Shape in the TensorExpr Graph
// and returns back a map from Symbolic Shape Value to its runtime Value *
std::map<int64_t, Value*> InsertSymbolicShapesCompute(
    const ShapeComputeGraphMapping& shape_mapping,
    Node* tensorexpr_graph) {
  WithInsertPoint guard(tensorexpr_graph);
  auto enclosing_graph = tensorexpr_graph->owningGraph();

  std::map<Value*, Value*> shape_graph_input_to_enclosing_graph_value;
  for (const auto& pair :
       shape_mapping.enclosing_graph_value_to_shape_graph_input_) {
    shape_graph_input_to_enclosing_graph_value[pair.second] = pair.first;
  }
  std::vector<Value*> shape_compute_graph_inputs;
  for (Value* shape_graph_input :
       shape_mapping.partial_eval_shape_graph->inputs()) {
    auto enclosing_graph_input =
        shape_graph_input_to_enclosing_graph_value.find(shape_graph_input);
    TORCH_INTERNAL_ASSERT(
        enclosing_graph_input !=
        shape_graph_input_to_enclosing_graph_value.end());
    if (*enclosing_graph_input->second->type() == *shape_graph_input->type()) {
      shape_compute_graph_inputs.push_back(tensorexpr_graph->inputs().at(
          enclosing_graph_input->second->offset()));
    } else {
      TORCH_INTERNAL_ASSERT(
          enclosing_graph_input->second->type()->cast<TensorType>() &&
          shape_graph_input->type()->isSubtypeOf(ListType::ofInts()));
      shape_compute_graph_inputs.push_back(enclosing_graph->insert(
          aten::size,
          {tensorexpr_graph->inputs().at(
              enclosing_graph_input->second->offset())}));
    }
  }
  auto sym_shape_values = insertGraph(
      *enclosing_graph,
      *shape_mapping.partial_eval_shape_graph,
      shape_compute_graph_inputs);
  std::map<int64_t, Value*> sym_shape_to_enclosing_graph_value;
  for (size_t i = 0;
       i < shape_mapping.partial_eval_shape_graph->outputs().size();
       ++i) {
    Value* output = shape_mapping.partial_eval_shape_graph->outputs().at(i);
    auto sym_shape =
        shape_mapping.graph_output_to_symbolic_shape_dim_.find(output);
    TORCH_INTERNAL_ASSERT(
        sym_shape != shape_mapping.graph_output_to_symbolic_shape_dim_.end());
    sym_shape_to_enclosing_graph_value[sym_shape->second] = sym_shape_values[i];
  }
  return sym_shape_to_enclosing_graph_value;
}

void insertDynamicShapesGuard(
    const ShapeComputeGraphMapping& shape_mapping,
    Node* guarded_node,
    bool add_composed_op);

// Generalize Complete Shapes inputs to Symbolic Shapes.
// Dimensions of value 1 will be preserved, otherwise
// dimensions with the same value will be bucketed to the same
// symbolic shape.
// E.g. Tensor(5, 3), Tensor(3, 1) -> Tensor(SS(-1), SS(-2)), Tensor(SS(-2), 1)
bool TryGeneralizeInputDimensionsToSymbolicShapes(
    std::shared_ptr<Graph> tensorexpr_graph) {
  std::map<size_t, int64_t> shape_to_sym_shape;
  for (Value* v : tensorexpr_graph->inputs()) {
    if (!v->type()->cast<TensorType>()) {
      continue;
    }
    if (!v->type()->expect<TensorType>()->sizes().isComplete()) {
      return false;
    }
    auto tt = v->type()->expect<TensorType>();
    std::vector<at::ShapeSymbol> shape_vec = *tt->symbolic_sizes().sizes();
    auto new_sizes = c10::fmap(shape_vec, [&](const at::ShapeSymbol& shape) {
      auto value = shape.value();
      TORCH_INTERNAL_ASSERT(value >= 0, "Expected complete tensor");
      if (value == 1) {
        return value;
      } else if (shape_to_sym_shape.count(static_cast<size_t>(value))) {
        return shape_to_sym_shape[value];
      } else {
        auto new_shape_symbol = at::ShapeSymbol::newSymbol().value();
        shape_to_sym_shape[static_cast<size_t>(value)] = new_shape_symbol;
        return new_shape_symbol;
      }
    });
    v->setType(tt->withSymbolicShapes(c10::SymbolicShape(new_sizes)));
  }
  return true;
}

void moveConstantTensorsOutOfSubgraph(
    Node* tensorexpr_graph_node,
    std::shared_ptr<Graph> tensorexpr_graph) {
  auto parent = tensorexpr_graph_node->owningGraph();

  auto env = [&](Value* v) {
    TORCH_INTERNAL_ASSERT(
        false,
        "this should never happen since constant nodes do not have any inputs",
        v->debugName());
    return v;
  };

  WithInsertPoint wip(tensorexpr_graph_node);
  std::vector<Node*> to_destroy;
  for (auto node : tensorexpr_graph->nodes()) {
    if (node->kind() == prim::Constant) {
      if (!node->output()->type()->cast<TensorType>()) {
        continue;
      }

      // copy the constant and insert that copy into the parent graph.
      auto copy = parent->createClone(node, env);
      parent->insertNode(copy);

      // add a new input to the te subgraph and replace the uses of the
      // constant with this input.
      auto new_const = tensorexpr_graph->addInput();
      new_const->setType(node->output()->type());
      node->output()->replaceAllUsesWith(new_const);

      // add the copy as input to the te node
      tensorexpr_graph_node->addInput(copy->output());

      to_destroy.push_back(node);
    }
  }

  for (auto n : to_destroy) {
    n->destroy();
  }
}

bool GenerateGuard(Node* tensorexpr_graph_node, bool add_composed_op) {
  auto tensorexpr_graph = SubgraphUtils::getSubgraph(tensorexpr_graph_node);

  // Move constant tensors from the subgraph to the outer scope.
  // This is necessary because symbolic shape analysis does not handle the
  // case of broadcast(constant, symbolic_shape) well and that results in poor
  // performance.
  moveConstantTensorsOutOfSubgraph(tensorexpr_graph_node, tensorexpr_graph);

  // Generalize Inputs
  if (!TryGeneralizeInputDimensionsToSymbolicShapes(tensorexpr_graph)) {
    return false;
  }

  // Try To Propagate Shapes
  auto maybe_shape_compute_mapping =
      PropagateShapesAndBuildLargeShapeComputeGraph(
          tensorexpr_graph,
          *tensorexpr_graph->nodes().begin(),
          *tensorexpr_graph->nodes().end());
  if (!maybe_shape_compute_mapping) {
    return false;
  }

  // Insert Guard
  insertDynamicShapesGuard(
      *maybe_shape_compute_mapping, tensorexpr_graph_node, add_composed_op);
  return true;
}

// TODO: share more logic with tensorexpr_fuser ?
void insertDynamicShapesGuard(
    const ShapeComputeGraphMapping& shape_mapping,
    Node* guarded_node,
    bool add_composed_op) {
  GRAPH_DEBUG(
      "Inserting a prim::TensorExprDynamicGuard guard for a node",
      *guarded_node);
  auto subgraph = SubgraphUtils::getSubgraph(guarded_node);

  // Fixup types of the subgraph inputs
  std::vector<Value*> inputs_to_check;
  std::vector<TypePtr> guard_types;
  for (const auto i : c10::irange(guarded_node->inputs().size())) {
    Value* node_input = guarded_node->inputs().at(i);
    // We only check inputs of the guarded nodes
    if (!node_input->type()->cast<TensorType>()) {
      continue;
    }
    inputs_to_check.push_back(node_input);
    guard_types.push_back(
        subgraph->inputs().at(i)->type()->expect<TensorType>());
  }
  TORCH_INTERNAL_ASSERT(inputs_to_check.size());

  // prim::TensorExprDynamicGuard nodes look like the following:
  //   %types_match : bool = prim::TypeCheck[attr:types](%inp1 : Tensor, %inp2 :
  //   Tensor)
  // The input tensors are checked against the expected types on attr::types
  // Omitting refining the input Tensors for now because they are not actually
  // used within tensorexpr/kernel.cpp (only the inputs to the Graph are, not
  // the inputs to the node) and we would have to redo the mapping to compute
  // symbolic shapes

  Node* typecheck_node =
      guarded_node->owningGraph()
          ->create(Symbol::prim("TensorExprDynamicGuard"), inputs_to_check, 1)
          ->insertBefore(guarded_node);

  typecheck_node->tys_(attr::types, guard_types);
  Value* typecheck_result = typecheck_node->output()->setType(BoolType::get());

  // Insert if
  auto versioning_if =
      guarded_node->owningGraph()
          ->create(prim::If, {typecheck_result}, guarded_node->outputs().size())
          ->insertAfter(typecheck_node);

  for (size_t idx = 0; idx < guarded_node->outputs().size(); ++idx) {
    versioning_if->output(idx)->setType(guarded_node->output(idx)->type());
    guarded_node->output(idx)->replaceAllUsesWith(versioning_if->output(idx));
  }
  auto true_block = versioning_if->addBlock();
  auto false_block = versioning_if->addBlock();

  // Fill in the false block. It should contain the unoptimized
  // copy of the fused subgraph.
  WithInsertPoint guard(false_block->return_node());
  const auto subgraph_outputs = insertGraph(
      *guarded_node->owningGraph(), *subgraph, guarded_node->inputs());
  for (Value* output : subgraph_outputs) {
    false_block->registerOutput(output);
  }

  // types get copied to the fallback graph, so remove specializations before
  // replacing
  removeTensorTypeSpecializations(false_block);
  replaceBlockWithFallbackGraph(false_block, guarded_node->inputs());

  // Fill in the true block. It has all inputs type-checked and its
  // body should be the fusion group node.
  guarded_node->moveBefore(true_block->return_node());

  for (Value* output : guarded_node->outputs()) {
    true_block->registerOutput(output);
  }

  // Insert Symbolic Shapes Compute and add as inputs to TE Node/Graph
  // symbolic_shape_inputs will be a list of each symbolic shape,
  // and the last N inputs to TE Graph/Node will be the N
  // symbolic shape values
  auto map = InsertSymbolicShapesCompute(shape_mapping, guarded_node);
  std::vector<int64_t> symbolic_shape_inputs;
  for (const auto& pair : map) {
    symbolic_shape_inputs.push_back(pair.first);
    guarded_node->addInput(pair.second);
    std::stringstream ss;
    ss << "SS_" << -pair.first;
    subgraph->addInput(ss.str())->setType(IntType::get());
  }
  guarded_node->is_(attr::symbolic_shape_inputs, symbolic_shape_inputs);

  if (add_composed_op) {
    // Create a TensorExprDynamicGroup node
    auto te_dyn_group = SubgraphUtils::createSingletonSubgraph(
        typecheck_node, prim::TensorExprDynamicGroup);
    SubgraphUtils::mergeNodeIntoSubgraph(versioning_if, te_dyn_group);
  }
}

// On each invocation of this guard, we need to check all of the static
// information (dtype/device/requires grad/contiguity/static dims),
// and also the that the symbolic shape dimensions are observed.
// For any symbolic dimension we need to set its value on its first
// use and for all subsequent uses check that the values are equal
RegisterOperators reg_guard({
    Operator(
        "prim::TensorExprDynamicGuard(...) -> bool",
        [](const Node* node) -> Operation {
          const auto& types = node->tys(attr::types);

          // Each inputs expected # of dims
          std::vector<size_t> expected_dims;

          // A flattened vector of all the expected values for all
          // tensor dims. A positive value corresponds to a static
          // shape to check and a negative value corresponds to symbolic
          // dimension index to check
          std::vector<int64_t> flattened_input_dims;

          // Each inputs expected scalar types
          std::vector<c10::ScalarType> expected_scalar_types;

          // Map from symbolic dimension value to its set's index
          std::map<int64_t, size_t> sym_dim_flat_index;
          TORCH_INTERNAL_ASSERT(types.size() >= 1);

          // we should just be fusing fusion groups with a single device
          // and with tensors not requiring grad
          auto maybe_device = types[0]->expect<TensorType>()->device();
          TORCH_INTERNAL_ASSERT(maybe_device);
          auto device = *maybe_device;

          for (auto type : types) {
            auto tt = type->expect<TensorType>();
            auto ss = tt->symbolic_sizes();
            TORCH_INTERNAL_ASSERT(ss.rank());
            expected_dims.push_back(*ss.rank());
            TORCH_INTERNAL_ASSERT(tt->scalarType());
            expected_scalar_types.push_back(*tt->scalarType());
            TORCH_INTERNAL_ASSERT(tt->device() && *tt->device() == device);
            for (size_t i = 0; i < *ss.rank(); ++i) {
              auto sym_dim = ss[i];
              auto value = sym_dim.value();
              if (value >= 0) {
                flattened_input_dims.push_back(value);
              } else {
                // use index for set if it exists, otherwise extend the vector
                // of sym shapes by 1
                int64_t sym_dim_index;
                if (sym_dim_flat_index.count(value)) {
                  sym_dim_index = sym_dim_flat_index[value];
                } else {
                  auto size = sym_dim_flat_index.size();
                  sym_dim_flat_index[value] = (-1) - size;
                  sym_dim_index = sym_dim_flat_index[value];
                }
                // TODO: potential optimization - if there is a Symbolic
                // Sym with only one use we dont need to test anything
                flattened_input_dims.push_back(sym_dim_index);
              }
            }
          }
          const auto num_inputs = types.size();
          const auto num_symbolic_dims = sym_dim_flat_index.size();
          return [num_inputs,
                  expected_dims,
                  device,
                  expected_scalar_types,
                  flattened_input_dims,
                  num_symbolic_dims](Stack& stack) {
            at::ArrayRef<IValue> inputs = last(stack, num_inputs);
            drop(stack, num_inputs);
            // each invocation we need to reset what value of each symbolic
            // symbol is.
            // TODO: could this be a reference and not allocated on
            // each invocation or would that mess up with multithreaded
            // inference since we are writing to it?
            // TODO - smallvector here ?
            std::vector<int64_t> flattened_symbolic_dims(num_symbolic_dims, -1);
            size_t flattened_dim_offset = 0;
            for (const auto i : c10::irange(num_inputs)) {
              at::Tensor tensor = inputs[i].toTensor();
              if (C10_UNLIKELY(
                      tensor.device() != device ||
                      tensor.dtype() != expected_scalar_types[i]) ||
                  tensor.requires_grad()) {
                std::cout << " DEVICE /DTYPE/ REQUIRES GRAD MISMATCH\n";
                std::cout << tensor;
                std::cout << "Expected device : " << device << "\n";
                std::cout << "Expected dtype : " << expected_scalar_types[i]
                          << "\n";
                push(stack, false);
                return;
              }
              // TODO: striding
              if (C10_UNLIKELY(
                      !tensor.is_contiguous(at::MemoryFormat::Contiguous))) {
                std::cout << " CONTIGUITY MISMATCH \n\n";
                std::cout << tensor;
                push(stack, false);
                return;
              }
              const auto& sizes = tensor.sizes();
              const auto num_dims = sizes.size();
              if (C10_UNLIKELY(num_dims != expected_dims[i])) {
                std::cout << " Expected Dims Mismatch \n\n";
                std::cout << tensor;
                std::cout << " Expected num dims: " << expected_dims[i] << "\n";
                push(stack, false);
                return;
              }
              for (const auto dim_index : c10::irange(num_dims)) {
                const int64_t dim_value =
                    flattened_input_dims[dim_index + flattened_dim_offset];
                const int64_t tensor_dim = sizes[dim_index];
                if (dim_value >= 0) {
                  if (C10_UNLIKELY(dim_value != tensor_dim)) {
                    std::cout << " Concrete value dim Mismatch \n\n";
                    std::cout << tensor;
                    std::cout << " Expected dim value: " << tensor_dim << "\n";
                    std::cout << " Dim index: " << dim_index;
                    push(stack, false);
                    return;
                  }
                } else {
                  // flattened sym indices start at -1,
                  // so -1 -> index 0, -2 -> index 1
                  const auto flattened_sym_index = (-dim_value) - 1;
                  const auto flattened_sym_value =
                      flattened_symbolic_dims[flattened_sym_index];
                  // sym symbol already seen, check value
                  if (flattened_symbolic_dims[flattened_sym_index] >= 0) {
                    if (C10_UNLIKELY(flattened_sym_value != tensor_dim)) {
                      std::cout << " symbolic value dim Mismatch \n\n";
                      std::cout << tensor;
                      std::cout << " Expected symbolic value : "
                                << flattened_sym_value << "\n";
                      std::cout << " Dim index: " << dim_index;
                      push(stack, false);
                      return;
                    }
                  } else {
                    // not seen, write value
                    flattened_symbolic_dims[flattened_sym_index] = tensor_dim;
                  }
                }
              }
              flattened_dim_offset += num_dims;
            }

            push(stack, true);
            return;
          };
        },
        aliasAnalysisFromSchema()),
});

void runTensorExprDynamicGroup(std::shared_ptr<Graph> graph, Stack& stack) {
  Code code(graph, "");
  InterpreterState interpreter{code};
  interpreter.run(stack);
}

Operation createTensorExprDynamicGroup(const Node* node) {
  auto graph = node->g(attr::Subgraph);
  // This implementation creates a Code object and InterpreterState on every
  // call to TensorExprDynamicGroup, which affects performance. Ideally, we
  // should be reusing Code and InterpreterState across calls to this op.
  // But that is resulting in a "No frames found" error.
  // TODO: Improve the performance of this by figuring out a better approach.
  return [graph](Stack& stack) {
    runTensorExprDynamicGroup(graph, stack);
    return 0;
  };
}

RegisterOperators TensorExprDynamicOp({
    torch::jit::Operator(
        prim::TensorExprDynamicGroup,
        createTensorExprDynamicGroup,
        AliasAnalysisKind::INTERNAL_SPECIAL_CASE),
});

} // namespace jit
} // namespace torch
