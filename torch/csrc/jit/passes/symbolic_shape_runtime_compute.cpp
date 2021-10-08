#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/symbolic_shape_runtime_compute.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/register_ops_utils.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/jit/jit_log.h>
#include <ATen/core/functional.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <sstream>

namespace torch {
namespace jit {


std::unordered_map<int64_t, Value*> InsertSymbolicShapesCompute(const ShapeComputeGraphMapping& shape_mapping, Node * tensorexpr_graph) {
    WithInsertPoint guard(tensorexpr_graph);
    auto enclosing_graph = tensorexpr_graph->owningGraph();

    std::unordered_map<Value *, Value*> shape_graph_input_to_enclosing_graph_value;
    for (const auto& pair: shape_mapping.enclosing_graph_value_to_shape_graph_input_) {
        shape_graph_input_to_enclosing_graph_value[pair.second] = pair.first;
    }
    std::vector<Value*> inputs;
    for (Value * shape_graph_input: shape_mapping.partial_eval_shape_graph->inputs()) {
        auto enclosing_graph_input = shape_graph_input_to_enclosing_graph_value.find(shape_graph_input);
        TORCH_INTERNAL_ASSERT(enclosing_graph_input != shape_graph_input_to_enclosing_graph_value.end());
        if (*enclosing_graph_input->second->type() == *shape_graph_input->type()) {
            inputs.push_back(tensorexpr_graph->inputs().at(enclosing_graph_input->second->offset()));
        } else {
            TORCH_INTERNAL_ASSERT(enclosing_graph_input->second->type()->cast<TensorType>() && shape_graph_input->type()->isSubtypeOf(ListType::ofInts()));
            inputs.push_back(enclosing_graph->insert(aten::size, {tensorexpr_graph->inputs().at(enclosing_graph_input->second->offset())}));
        }
    }
    auto sym_shape_values = insertGraph(*enclosing_graph, *shape_mapping.partial_eval_shape_graph, inputs);
    std::unordered_map<int64_t, Value*> sym_shape_to_enclosing_graph_value;
    for (size_t i = 0; i < shape_mapping.partial_eval_shape_graph->outputs().size(); ++i) {
        Value * output = shape_mapping.partial_eval_shape_graph->outputs().at(i);
        auto sym_shape = shape_mapping.graph_output_to_symbolic_shape_dim_.find(output);
        TORCH_INTERNAL_ASSERT(sym_shape != shape_mapping.graph_output_to_symbolic_shape_dim_.end());
        sym_shape_to_enclosing_graph_value[sym_shape->second] = sym_shape_values[i];
    }
    return sym_shape_to_enclosing_graph_value;
}

std::unordered_map<int64_t, Value*> insertTypeGuard(const ShapeComputeGraphMapping& shape_mapping, Node* guarded_node);

c10::optional<std::unordered_map<int64_t, Value*>>  GenerateGuard(Node * tensorexpr_graph_node) {
    auto tensorexpr_graph =  SubgraphUtils::getSubgraph(tensorexpr_graph_node);
    std::unordered_map<size_t, int64_t> shape_to_sym_shape;
    for (Value * v: tensorexpr_graph->inputs()) {
        if (!v->type()->cast<TensorType>()) {
            continue;
        }
        if (!v->type()->expect<TensorType>()->sizes().concrete_sizes()) {
            return c10::nullopt;
        }
        auto tt = v->type()->expect<TensorType>();
        std::vector<at::ShapeSymbol> shape_vec = *tt->symbolic_sizes().sizes();
        auto new_sizes =
            c10::fmap(shape_vec, [&](const at::ShapeSymbol& shape) {
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

    auto maybe_shape_compute_mapping = PropagateShapesAndBuildLargeShapeComputeGraph(tensorexpr_graph, *tensorexpr_graph->nodes().begin(), *tensorexpr_graph->nodes().end());
    if (!maybe_shape_compute_mapping) {
        return c10::nullopt;
    }

    return insertTypeGuard(*maybe_shape_compute_mapping, tensorexpr_graph_node);
}

const auto& symbolic_shape_inputsAttr = Symbol::attr("symbolic_shape_inputs");


// TODO: NOT COPY PASTA
std::unordered_map<int64_t, Value*> insertTypeGuard(const ShapeComputeGraphMapping& shape_mapping, Node* guarded_node) {
  GRAPH_DEBUG("Inserting a prim::TensorExprDynamicGuard guard for a node", *guarded_node);
  auto subgraph = SubgraphUtils::getSubgraph(guarded_node);

  // Fixup types of the subgraph inputs
  std::vector<Value*> inputs_to_check;
  std::vector<TypePtr> guard_types;
  for (const auto i : c10::irange(guarded_node->inputs().size())) {
    Value * node_input = guarded_node->inputs().at(i);
    // We only check inputs of the guarded nodes and expect user to infer
    // intermediates and outputs shapes
    if (!node_input->type()->cast<TensorType>()) {
      continue;
    }

    if (node_input->node()->kind() == prim::Constant) {
      continue;
    }
    inputs_to_check.push_back(node_input);
    guard_types.push_back(subgraph->inputs().at(i)->type()->expect<TensorType>());
  }
  TORCH_INTERNAL_ASSERT(inputs_to_check.size());

  // prim::TensorExprDynamicGuard nodes look like the following:
  //   %types_match : bool = prim::TypeCheck[attr:types](%inp1 : Tensor, %inp2 : Tensor)
  // The input tensors are checked against the expected types on attr::types
  // Omitting refining the input Tensors for now because they are not actually
  // used within tensorexpr/kernel.cpp (only the inputs to the Graph are, not the inputs to the node)
  // and we would have to redo the mapping to compute symbolic shapes


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

  auto map = InsertSymbolicShapesCompute(shape_mapping, guarded_node);
  std::vector<int64_t> symbolic_shape_inputs;
  for (const auto& pair: map) {
    symbolic_shape_inputs.push_back(pair.first);
    guarded_node->addInput(pair.second);
    std::stringstream ss;
    ss << "SS_" << -pair.first;
    subgraph->addInput(ss.str())->setType(IntType::get());
  }
  guarded_node->is_(symbolic_shape_inputsAttr, symbolic_shape_inputs);
  return map;
}

RegisterOperators reg_guard({
    Operator(
        "prim::TensorExprDynamicGuard(...) -> bool",
        [](const Node* node) -> Operation {
          const auto& types = node->tys(attr::types);
          at::Tensor t;
          std::vector<size_t> expected_dims;
          // int64_t positive - expected dim value
          // int64_t negative - index into the sym_shapes to check
          std::vector<int64_t> flattened_input_dims;
          std::vector<c10::ScalarType> expected_scalar_types;
          std::unordered_map<int64_t, size_t> sym_dim_flat_index;
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
                int64_t sym_dim_index;
                if (sym_dim_flat_index.count(value)) {
                  sym_dim_index = sym_dim_flat_index[value];
                } else {
                  sym_dim_flat_index[value] = (-1) - sym_dim_flat_index.size();
                }
                // TODO: potential optimization - if there is only one Symbolic
              // Symbol in in input types we dont need to test anything
                flattened_input_dims.push_back(value);
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
            // each invocation we need to reset what value of each symbolic symbol
            // is. TODO: could this be a reference and not allocated on each invocation
            // or would that mess up with multithreaded inference since we are writing to it?
            std::vector<int64_t> flattened_symbolic_dims(num_symbolic_dims, -1);
            size_t flattened_dim_offset = 0;
            for (const auto i : c10::irange(num_inputs)) {
              const at::Tensor& tensor = inputs[i].toTensor();
              if (C10_UNLIKELY(
                      tensor.device() != device ||
                      tensor.dtype() != expected_scalar_types[i]) || tensor.requires_grad()) {
                push(stack, false);
                return;
              }
              // TODO: striding
              if (C10_UNLIKELY(
                      !tensor.is_contiguous(at::MemoryFormat::Contiguous))) {
                push(stack, false);
                return;
              }
              const auto& sizes = tensor.sizes();
              const auto num_dims = sizes.size();
              if (C10_UNLIKELY(num_dims != expected_dims[i])) {
                push(stack, false);
                return;
              }
              for (const auto dim_index : c10::irange(num_dims)) {
                const int64_t dim_value =
                    flattened_input_dims[dim_index + flattened_dim_offset];
                const int64_t tensor_dim = sizes[dim_index];
                if (dim_value >= 0) {
                  if (C10_UNLIKELY(dim_value != tensor_dim)) {
                    push(stack, false);
                    return;
                  }
                } else {
                  // flattened sym indices start at -1,
                  // so -1 -> index 0, -2 -> index 1
                  const auto flattened_sym_index = (-dim_value) - 1;
                  const auto flattened_sym_value =
                      flattened_symbolic_dims[flattened_sym_index];
                  if (flattened_symbolic_dims[flattened_sym_index] >= 0) {
                    if (C10_UNLIKELY(flattened_sym_value != tensor_dim)) {
                      push(stack, false);
                      return;
                    }
                  } else {
                    flattened_symbolic_dims[flattened_sym_index] = tensor_dim;
                  }
                }
              }
              flattened_dim_offset += num_dims;
            }

            push(stack, IValue(true));
            return;
          };
        },
        aliasAnalysisFromSchema()),
});
}
}