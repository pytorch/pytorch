#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/symbolic_shape_runtime_compute.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>


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


void GenerateGuard(Node * tensorexpr_graph_node) {
    auto tensorexpr_graph =  SubgraphUtils::getSubgraph(tensorexpr_graph_node);
    std::unordered_map<size_t, int64_t> shape_to_sym_shape;
    for (Value * v: tensorexpr_graph->inputs()) {
        if (!v->type()->cast<TensorType>()) {
            continue;
        }
        if (!v->type()->expect<TensorType>()->sizes().concrete_sizes()) {
            return;
        }
        auto tt = v->type()->expect<TensorType>();
        std::vector<at::ShapeSymbol> shape_vec = *tt->symbolic_sizes().sizes();
        auto new_sizes =
            c10::fmap(shape_vec, [&](const at::ShapeSymbol& shape) {
              auto value = shape.value();
              TORCH_INTERNAL_ASSERT(value >= 0, "Expected complete tensor");
              if (shape_to_sym_shape.count(static_cast<size_t>(value))) {
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
        return;
    }

    InsertSymbolicShapesCompute(*maybe_shape_compute_mapping, tensorexpr_graph_node);
}


}
}