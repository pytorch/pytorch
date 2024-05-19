#include <ATen/core/functional.h>
#include <ATen/core/interned_strings.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/symbolic_shape_runtime_fusion.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>
#include <torch/csrc/jit/runtime/register_ops_utils.h>
#include <torch/csrc/jit/runtime/static/ops.h>
#include <sstream>
#include <utility>

namespace torch {
namespace jit {

// Inserts the Compute for Each Symbolic Shape in the TensorExpr Graph
// and returns back a map from Symbolic Shape Value to its runtime Value *
static std::map<int64_t, Value*> InsertSymbolicShapesCompute(
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
    bool add_composed_op,
    std::vector<std::vector<StrideInput>>& input_info,
    std::vector<StrideInput>& output_strides);

std::string toString(StrideInput si) {
  switch (si) {
    case StrideInput::TENSOR_CONT:
      return "TENSOR_CONT";
    case StrideInput::TENSOR_CONT_CHANNELS_LAST:
      return "TENSOR_CONT_CHANNELS_LAST";
    case StrideInput::S_ONE:
      return "S_ONE";
    case StrideInput::S_CONT:
      return "S_CONT";
    case StrideInput::S_TRAN_CONT:
      return "S_TRAN_CONT";
    case StrideInput::S_AS_ARG:
      return "S_AS_ARG";
  }
  TORCH_INTERNAL_ASSERT(false);
}

StrideInput strideInputFromString(const std::string& si) {
  if (si == "TENSOR_CONT") {
    return StrideInput::TENSOR_CONT;
  } else if (si == "TENSOR_CONT_CHANNELS_LAST") {
    return StrideInput::TENSOR_CONT_CHANNELS_LAST;
  } else if (si == "S_ONE") {
    return StrideInput::S_ONE;
  } else if (si == "S_CONT") {
    return StrideInput::S_CONT;
  } else if (si == "S_TRAN_CONT") {
    return StrideInput::S_TRAN_CONT;
  } else if (si == "S_AS_ARG") {
    return StrideInput::S_AS_ARG;
  } else {
    TORCH_INTERNAL_ASSERT(false);
  }
}

// in the runtime guard, strides are serialized as one flat
// vector. stride_inputs_offset indexes into that vector
// where the strides of this tensor begin
inline StrideInput summarizeStrideDim(
    const c10::IntArrayRef sizes,
    const c10::IntArrayRef strides,
    size_t dim,
    const std::vector<StrideInput>& stride_inputs,
    size_t stride_inputs_offset) {
  if (strides[dim] == 1) {
    return StrideInput::S_ONE;
  } else if (
      dim + 1 < sizes.size() &&
      strides[dim] == strides[dim + 1] * sizes[dim + 1]) {
    return StrideInput::S_CONT;
    // Transposed Contiguous depends on prior dim and contiguous depends on next
    // dim, so to avoid a mutual dependence check that the next dim is Stride
    // Contiguous
  } else if (
      dim > 0 && strides[dim] == strides[dim - 1] * sizes[dim - 1] &&
      (stride_inputs[dim - 1 + stride_inputs_offset] != StrideInput::S_CONT)) {
    return StrideInput::S_TRAN_CONT;
  } else {
    return StrideInput::S_AS_ARG;
  }
}

static std::vector<StrideInput> summarizeInputStrides(const TensorType& tt) {
  auto strides = *tt.strides().concrete_sizes();
  auto sizes = *tt.sizes().concrete_sizes();
  if (c10::is_contiguous_strides(sizes, strides)) {
    return {StrideInput::TENSOR_CONT};
    // TODO: channels last 3d
  } else if (c10::is_channels_last_strides_2d(sizes, strides)) {
    return {StrideInput::TENSOR_CONT_CHANNELS_LAST};
  }
  std::vector<StrideInput> stride_inputs;
  for (size_t dim = 0; dim < sizes.size(); ++dim) {
    stride_inputs.push_back(
        summarizeStrideDim(sizes, strides, dim, stride_inputs, 0));
  }
  return stride_inputs;
};

// Todo: incorporate in codegen
static StrideInput summarizeOutputStrides(const TensorType& tt) {
  auto strides = *tt.strides().concrete_sizes();
  auto sizes = *tt.sizes().concrete_sizes();
  // We only try to maintain output striding for channels last tensors,
  // otherwise we defer to contiguous
  // TODO: channels last 3d
  if (c10::is_channels_last_strides_2d(sizes, strides)) {
    return StrideInput::TENSOR_CONT_CHANNELS_LAST;
  }
  return StrideInput::TENSOR_CONT;
}

// Generalize Complete Shapes inputs to Symbolic Shapes.
// Dimensions of value 1 will be preserved, otherwise
// dimensions with the same value will be bucketed to the same
// symbolic shape.
// E.g. Tensor(5, 3), Tensor(3, 1) -> Tensor(SS(-1), SS(-2)), Tensor(SS(-2), 1)
// Also summarize input striding behavior. The Size information is stored on the
// type, The striding is returned. See StrideInput for description of stride
// specializations
static c10::optional<std::vector<std::vector<StrideInput>>>
TryGeneralizeInputDimensionsToSymbolicShapes(
    std::shared_ptr<Graph> tensorexpr_graph) {
  std::map<size_t, int64_t> shape_to_sym_shape;
  std::vector<std::vector<StrideInput>> input_striding;

  for (Value* v : tensorexpr_graph->inputs()) {
    if (!v->type()->cast<TensorType>()) {
      continue;
    }
    auto tt = v->type()->expectRef<TensorType>();
    if (!tt.sizes().isComplete() || !tt.strides().isComplete()) {
      return c10::nullopt;
    }
    input_striding.push_back(summarizeInputStrides(tt));
    std::vector<at::ShapeSymbol> shape_vec = *tt.symbolic_sizes().sizes();
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
    v->setType(tt.withSymbolicShapes(c10::SymbolicShape(new_sizes)));
  }
  return input_striding;
}

static void moveConstantTensorsOutOfSubgraph(
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
  auto input_striding =
      TryGeneralizeInputDimensionsToSymbolicShapes(tensorexpr_graph);
  if (!input_striding) {
    return false;
  }

  // Get output striding behavior
  std::vector<StrideInput> output_striding;
  for (Value* v : tensorexpr_graph->outputs()) {
    if (!v->type()->cast<TensorType>()) {
      continue;
    }
    auto tt = v->type()->expectRef<TensorType>();
    if (!tt.sizes().isComplete() || !tt.strides().isComplete()) {
      return false;
    }
    output_striding.push_back(summarizeOutputStrides(tt));
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
      *maybe_shape_compute_mapping,
      tensorexpr_graph_node,
      add_composed_op,
      *input_striding,
      output_striding);
  return true;
}

static void inlineFallbackGraphAndAddSRCopyOutOp(std::shared_ptr<Graph> graph) {
  DepthFirstGraphNodeIterator it(graph);

  Node* n = nullptr;
  while ((n = it.next()) != nullptr) {
    if (n->kind() == prim::FallbackGraph) {
      break;
    }
  }
  TORCH_INTERNAL_ASSERT(n != nullptr, "Expected to find fallback graph");

  auto if_node = n->owningBlock()->owningNode();
  IfView if_v(if_node);
  SubgraphUtils::unmergeSubgraph(n);

  auto false_block = if_v.elseBlock();
  std::vector<Value*> false_block_outputs(
      if_v.elseOutputs().begin(), if_v.elseOutputs().end());
  TORCH_INTERNAL_ASSERT(!false_block_outputs.empty());

  for (auto out : false_block_outputs) {
    TORCH_INTERNAL_ASSERT(out->type()->cast<TensorType>());
  }
  auto copy_node = graph->create(
      prim::StaticRuntimeCopyOuts,
      false_block_outputs,
      false_block_outputs.size());
  false_block->appendNode(copy_node);
  for (size_t i = 0; i < false_block_outputs.size(); ++i) {
    false_block->replaceOutput(i, copy_node->outputs().at(i));
  }
}

// TODO: share more logic with tensorexpr_fuser ?
void insertDynamicShapesGuard(
    const ShapeComputeGraphMapping& shape_mapping,
    Node* guarded_node,
    bool add_composed_op,
    std::vector<std::vector<StrideInput>>& input_info,
    std::vector<StrideInput>& output_strides) {
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
    guard_types.emplace_back(
        subgraph->inputs().at(i)->type()->expect<TensorType>()->withStrides(
            c10::VaryingShape<c10::Stride>()));
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

  typecheck_node->tys_(attr::types, std::move(guard_types));
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
  guarded_node->is_(
      attr::symbolic_shape_inputs, std::move(symbolic_shape_inputs));

  std::vector<std::vector<std::string>> input_striding;
  for (auto& vec : input_info) {
    auto string_info =
        fmap(vec, [&](StrideInput inp) { return toString(inp); });
    input_striding.push_back(string_info);
  }
  auto ival = IValue(input_striding);
  guarded_node->ival_(attr::striding_inputs_desc, ival);
  typecheck_node->ival_(attr::striding_inputs_desc, std::move(ival));

  for (Value* v : subgraph->inputs()) {
    if (auto t = v->type()->cast<TensorType>()) {
      v->setType(t->withStrides(c10::VaryingShape<c10::Stride>()));
    }
  }
  for (Value* v : subgraph->outputs()) {
    if (auto t = v->type()->cast<TensorType>()) {
      v->setType(t->withStrides(c10::VaryingShape<c10::Stride>()));
    }
  }

  std::vector<std::string> output_striding =
      fmap(output_strides, [&](StrideInput inp) { return toString(inp); });
  auto output_ival = IValue(output_striding);
  guarded_node->ival_(attr::striding_outputs_desc, std::move(output_ival));

  if (add_composed_op) {
    // only in SR flow do we check for values on the stack and
    // forward them along as tensor outputs
    // TODO: - refactor and make explicit part of TE Kernel api
    guarded_node->i_(attr::allow_stack_outputs, 1);

    // Create a TensorExprDynamicGroup node
    auto te_dyn_group = SubgraphUtils::createSingletonSubgraph(
        typecheck_node, prim::TensorExprDynamicGroup);
    SubgraphUtils::mergeNodeIntoSubgraph(versioning_if, te_dyn_group);
    inlineFallbackGraphAndAddSRCopyOutOp(
        SubgraphUtils::getSubgraph(te_dyn_group));
  }
}

// This operator is inserted at the end of the fallback block computing outputs
// for the fusion group. We convert block1():
//   %14 : Tensor = aten::mul(%0, %1)
//   %15 : Tensor = aten::mul(%0, %14)
//   -> (%15, %14)
// return (%3, %4)
// to
// block1():
//   %14 : Tensor = aten::mul(%0, %1)
//   %15 : Tensor = aten::mul(%0, %14)
//   %16 : Tensor, %17 : Tensor = prim::StaticRuntimeCopyOuts(%15, %14)
//   -> (%16, %17)
// Every output of the block is added as an input, and for each input there is
// a StaticRuntimeCopyOuts output. SR invokes the composed operator first with
// no tensors on the stack, in which case the Op will just return back the
// inputs. Second it invokes it with pre-allocated tensors, one for each output
// of the Fusion group, which is the same number of outputs of the fallback
// block. In this case we copy over the values of the inputs to pre-allocated
// tensors
// Note: this logic is meant to reflect the invocation of the TE Kernel
// and `runWithAllocatedOutputs` in tensorexpr_fuser.cpp
static Operation StaticRuntimeCopyOuts(const Node* node) {
  auto num_ten_inputs = node->inputs().size();
  return [num_ten_inputs](Stack& stack) {
    std::vector<IValue> inputs = pop(stack, num_ten_inputs);
    // uncommon case - first run
    if (stack.empty()) {
      for (IValue elem : inputs) {
        push(stack, std::move(elem));
      }
    } else {
      at::ArrayRef<IValue> outputs = last(stack, num_ten_inputs);
      for (size_t i = 0; i < inputs.size(); ++i) {
        IValue out = outputs[i];
        at::Tensor& out_t = out.toTensor();
        fastResizeToZero(out_t);
        out_t.resize_as_(inputs[i].toTensor());
        out_t.copy_(inputs[i].toTensor());
      }
    }
    return 0;
  };
}

RegisterOperators SRCopyOuts({
    torch::jit::Operator(
        prim::StaticRuntimeCopyOuts,
        StaticRuntimeCopyOuts,
        AliasAnalysisKind::CONSERVATIVE),
});

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
          TORCH_INTERNAL_ASSERT(!types.empty());

          // we should just be fusing fusion groups with a single device
          // and with tensors not requiring grad
          auto maybe_device = types[0]->expect<TensorType>()->device();
          TORCH_INTERNAL_ASSERT(maybe_device);
          auto device = *maybe_device;

          // flattened vector of each inputs striding behavior
          std::vector<StrideInput> flattened_input_striding;
          const IValue& sym_strides = node->ival(attr::striding_inputs_desc);
          std::vector<std::vector<std::string>> sym_strides_strs =
              sym_strides.to<std::vector<std::vector<std::string>>>();
          for (const auto& vec : sym_strides_strs) {
            std::vector<StrideInput> input_desc;
            for (const std::string& str : vec) {
              flattened_input_striding.push_back(strideInputFromString(str));
            }
          }

          for (const auto& type : types) {
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
                  flattened_input_striding,
                  num_symbolic_dims](Stack& stack) {
            at::ArrayRef<IValue> inputs = last(stack, num_inputs);
            drop(stack, num_inputs);
            // each invocation we need to reset what value of each symbolic
            // symbol is.
            // TODO: could this be a reference and not allocated on
            // each invocation or would that mess up with multithreaded
            // inference since we are writing to it?
            // TODO - smallvector here ?
            bool grad_mode_enabled = at::GradMode::is_enabled();
            std::vector<int64_t> flattened_symbolic_dims(num_symbolic_dims, -1);
            size_t flattened_dim_offset = 0;
            size_t flattened_stride_offset = 0;
            for (const auto i : c10::irange(num_inputs)) {
              at::Tensor tensor = inputs[i].toTensor();
              if (C10_UNLIKELY(
                      tensor.device() != device ||
                      tensor.dtype() != expected_scalar_types[i])) {
                push(stack, false);
                return;
              }
              if (C10_UNLIKELY(grad_mode_enabled && tensor.requires_grad())) {
                push(stack, false);
                return;
              }
              const auto& sizes = tensor.sizes();
              const auto num_dims = sizes.size();
              if (C10_UNLIKELY(num_dims != expected_dims[i])) {
                push(stack, false);
                return;
              }
              auto striding = flattened_input_striding[flattened_stride_offset];
              // Tensors natively store whether they are contiguous
              // in the default memory format or in channels last,
              // so it is more efficient to query whether they follow this
              // property than iterating over dimensions and checking yourself
              if (striding == StrideInput::TENSOR_CONT) {
                if (C10_UNLIKELY(
                        !tensor.is_contiguous(at::MemoryFormat::Contiguous))) {
                  push(stack, false);
                  return;
                }
                flattened_stride_offset += 1;
              } else if (striding == StrideInput::TENSOR_CONT_CHANNELS_LAST) {
                // TODO: 5D channels last
                if (C10_UNLIKELY(!tensor.is_contiguous(
                        at::MemoryFormat::ChannelsLast))) {
                  push(stack, false);
                  return;
                }
                flattened_stride_offset += 1;
              } else {
                auto strides = tensor.strides();
                for (size_t dim = 0; dim < num_dims; ++dim) {
                  auto summarized_dim = summarizeStrideDim(
                      sizes,
                      strides,
                      dim,
                      flattened_input_striding,
                      flattened_stride_offset);
                  if (C10_UNLIKELY(
                          summarized_dim !=
                          flattened_input_striding
                              [dim + flattened_stride_offset])) {
                    push(stack, false);
                    return;
                  }
                }
                flattened_stride_offset += num_dims;
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
                  // sym symbol already seen, check value
                  if (flattened_symbolic_dims[flattened_sym_index] >= 0) {
                    if (C10_UNLIKELY(flattened_sym_value != tensor_dim)) {
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

void runTensorExprDynamicGroup(const Code& code, Stack& stack) {
  InterpreterState interpreter{code};
  interpreter.run(stack);
}

static Operation createTensorExprDynamicGroup(const Node* node) {
  const auto& graph = node->g(attr::Subgraph);
  Code code(graph, "");
  // This implementation creates a Code object and InterpreterState on every
  // call to TensorExprDynamicGroup, which affects performance. Ideally, we
  // should be reusing Code and InterpreterState across calls to this op.
  // But that is resulting in a "No frames found" error.
  // TODO: Improve the performance of this by figuring out a better approach.
  // NB: this is only run in SR, which is single-threaded
  return [code](Stack& stack) {
    runTensorExprDynamicGroup(code, stack);
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
