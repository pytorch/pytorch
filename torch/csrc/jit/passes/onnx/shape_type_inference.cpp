#include <torch/csrc/jit/passes/onnx/shape_type_inference.h>

#include <c10/util/irange.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/onnx/constant_fold.h>
#include <torch/csrc/jit/passes/onnx/constant_map.h>
#include <torch/csrc/jit/passes/onnx/fold_if_node.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/passes/onnx/scalar_type_analysis.h>
#include <torch/csrc/jit/python/python_arg_flatten.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/onnx.h>
#include <torch/csrc/utils/python_strings.h>

#include <onnx/shape_inference/implementation.h>
#include <algorithm>
#include <cmath>

namespace torch {
namespace jit {

// Return a new TypePtr, merging ONNX inferred type with existing type.
// The inferred type will take higher precedence, since it is produced by ONNX
// shape inference, and is more compatible with ONNX. In cases where ONNX shape
// inference fails to produce an inferred type, or produces inferred type that
// is incomplete, refer to existing type and fill in the gap that is missing.
// Currently the following cases are supported.
//  1. existing type: Tensor[], inferred type: Tensor[]
//    For list of tensors, existing type does not store datatype nor shape for
//    inner tensor. Thus inferred type always contain more information, and is
//    returned.
//  2. existing type: Tensor, inferred type: Tensor
//    Fill in missing info (shape, data type) for inferred type from existing
//    type.
//  3. existing type: Scalar[], inferred type: Tensor
//    ONNX represents list of scalars by 1-d Tensor. Return inferred type since
//    it is more compatible with ONNX.
TypePtr MergeInferredType(TypePtr existing_type, TypePtr inferred_type) {
  auto new_list_type = inferred_type->cast<ListType>();
  if (new_list_type) {
    return inferred_type;
  }
  auto new_tensor_type = inferred_type->cast<TensorType>();
  auto old_tensor_type = existing_type->cast<TensorType>();

  if (new_tensor_type && old_tensor_type) {
    if (!old_tensor_type->device()) {
      // device not avaible means this is an invalid tensor type (most likely an
      // empty one) return inferred type directly.
      return new_tensor_type;
    }
    auto type = old_tensor_type;
    if (new_tensor_type->dim()) {
      type = type->withSymbolicShapes(new_tensor_type->symbolic_sizes());
    }
    if (new_tensor_type->scalarType().has_value()) {
      type = type->withScalarType(new_tensor_type->scalarType());
    }
    return type;
  }

  if (old_tensor_type) {
    return existing_type;
  }

  auto old_list_type = existing_type->cast<ListType>();
  if (new_tensor_type && old_list_type) {
    if (new_tensor_type->sizes().isComplete()) {
      return inferred_type;
    }
    return existing_type;
  }

  return inferred_type;
}

namespace {
namespace onnx_torch = ::torch::onnx;
namespace onnx = ::ONNX_NAMESPACE;

TensorTypePtr TorchTensorTypeFromONNX(
    const onnx::TypeProto_Tensor& onnx_tensor_type,
    SymbolDimMap& symbol_map) {
  c10::optional<at::ScalarType> scalar_type;
  if (onnx_tensor_type.has_elem_type()) {
    scalar_type = ONNXTypeToATenType(onnx_tensor_type.elem_type());
  }

  auto v_type = TensorType::create(
      scalar_type,
      at::kCPU,
      c10::SymbolicShape(),
      c10::VaryingShape<c10::Stride>{},
      {});
  if (onnx_tensor_type.has_shape()) {
    std::vector<c10::ShapeSymbol> sizes;
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    auto onnx_shape = onnx_tensor_type.shape();

    for (int i = 0; i < onnx_shape.dim_size(); ++i) {
      auto& dim = onnx_shape.dim(i);
      if (dim.has_dim_value()) {
        sizes.emplace_back(c10::ShapeSymbol::fromStaticSize(dim.dim_value()));
      } else {
        c10::optional<c10::ShapeSymbol> sym = c10::nullopt;
        if (dim.has_dim_param()) {
          // A specific dim param is produced.
          // Search if this is already known,
          // and assign the same Symbol.
          GRAPH_UPDATE("Got dim_param:", dim.dim_param());
          // NOLINTNEXTLINE(performance-for-range-copy)
          for (auto pair : symbol_map) {
            if (pair.second == dim.dim_param()) {
              sym = pair.first;
              break;
            }
          }
          if (!sym) {
            sym = c10::ShapeSymbol::newSymbol();
            symbol_map[sym.value()] = dim.dim_param();
          }
        } else {
          // A None dim param is produced.
          // Assign a new Symbol, no need to keep track
          // of it because there won't be duplicates.
          sym = c10::ShapeSymbol::newSymbol();
        }
        sizes.emplace_back(sym.value());
      }
    }
    v_type = TensorType::create(scalar_type, at::kCPU, sizes.size(), {});
    v_type = v_type->withSymbolicShapes(c10::SymbolicShape(sizes));

    if (v_type->sizes().concrete_sizes().has_value()) {
      // Populate strides based on sizes info, if sizes are all static.
      // Creating strides ensures yielding True for isCompleteTensor.
      v_type = v_type->contiguous();
    }
  }

  return v_type;
}

ListTypePtr TorchListTypeFromONNX(
    const onnx::TypeProto_Sequence& onnx_sequence_type,
    SymbolDimMap& symbol_map) {
  c10::optional<at::ScalarType> scalar_type;
  if (onnx_sequence_type.has_elem_type()) {
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    auto onnx_seq_elem_type = onnx_sequence_type.elem_type();
    if (onnx_seq_elem_type.has_tensor_type()) {
      // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
      auto onnx_tensor_type = onnx_seq_elem_type.tensor_type();
      auto v_tensor_type =
          TorchTensorTypeFromONNX(onnx_tensor_type, symbol_map);
      auto v_type = ListType::create(v_tensor_type);
      return v_type;
    }
  }
  return nullptr;
}

void UpdateTorchValueByOnnxValueInfo(
    Value* v,
    const onnx::ValueInfoProto& p_info,
    SymbolDimMap& symbol_map) {
  if (!p_info.has_type()) {
    return;
  }

  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  auto p_type = p_info.type();
  if (p_type.has_tensor_type()) {
    auto torch_tensor_type =
        TorchTensorTypeFromONNX(p_type.tensor_type(), symbol_map);
    if (torch_tensor_type) {
      v->setType(MergeInferredType(v->type(), torch_tensor_type));
    }
  } else if (p_type.has_sequence_type()) {
    auto torch_list_type =
        TorchListTypeFromONNX(p_type.sequence_type(), symbol_map);
    if (torch_list_type) {
      v->setType(MergeInferredType(v->type(), torch_list_type));
    }
  }
}

bool IsSupportedNode(const Node* n) {
  auto node_kind = n->kind();

  if (!node_kind.is_onnx()) {
    // node kind is not ONNX, skipped.
    return false;
  }

  // Skip when block size is zero. This is when the node is first created,
  // doesn't have subblocks attached yet. Run shape inference for these nodes
  // when the subgraph has already completed shape inferencing.
  if (node_kind == ::c10::onnx::Loop || node_kind == ::c10::onnx::If) {
    if (n->blocks().size() == 0) {
      return false;
    }
    for (auto b : n->blocks()) {
      for (auto b_n : b->nodes()) {
        if (!IsSupportedNode(b_n)) {
          return false;
        }
      }
    }
  }

  return true;
}

Value* CloneValueFromListConstruct(
    Value* v,
    std::shared_ptr<Graph> n_graph,
    int opset_version) {
  auto lc_node = v->node();
  TORCH_INTERNAL_ASSERT(lc_node->kind() == ::c10::prim::ListConstruct);
  // In jit/passes/onnx/peephole.cpp::eraseListConstruct,
  // prim::ListConstruct is converted to onnx::Concat. The conversion should
  // eventually be moved to symbolic. For now, treat this operator as
  // special case, and change from list type to tensor type. The scalar type
  // is preserved. If the elemtype is Int, insert a onnx::Concat node into
  // the graph.
  TypePtr elem = v->type()->castRaw<ListType>()->getElementType();
  c10::optional<at::ScalarType> scalar_type = c10::nullopt;
  if (elem->cast<IntType>()) {
    scalar_type = at::kLong;

    auto lc_node = v->node();
    // ListConstruct Int[] output case, we need to transform to ONNX
    // Concat to ensure the output is a single tensor(dynamic) type in
    // order to be consumed as inputs
    std::vector<Value*> unsqueezed;
    for (auto* input : lc_node->inputs()) {
      auto new_input = n_graph->addInput();
      new_input->copyMetadata(input);
      Node* unsqueezed_node = createONNXUnsqueeze(
          n_graph.get(), n_graph->return_node(), new_input, 0, opset_version);
      unsqueezed.emplace_back(unsqueezed_node->output());
    }
    Node* concat_node =
        n_graph->insertNode(n_graph->create(::c10::onnx::Concat, 1));
    concat_node->i_(attr::axis, 0);
    for (auto v : unsqueezed) {
      concat_node->addInput(v);
    }
    return concat_node->output();
  } else if (elem->cast<FloatType>()) {
    scalar_type = at::kFloat;
  } else if (elem->cast<BoolType>()) {
    scalar_type = at::kBool;
  } else if (auto t_type = elem->cast<TensorType>()) {
    scalar_type = t_type->scalarType();
  }

  auto input = n_graph->addInput();
  if (scalar_type) {
    auto v_type = TensorType::create(
        scalar_type.value(),
        at::kCPU,
        c10::SymbolicShape(),
        c10::VaryingShape<c10::Stride>{},
        {});
    input->setType(v_type);
  }
  return input;
}

// Clone the node n for the new graph.
Node* CloneNodeToGraph(
    Node* n,
    std::shared_ptr<Graph> n_graph,
    const ParamMap& params_dict,
    int opset_version) {
  auto vals_to_params_map =
      buildValueToParamsMap(n->owningGraph()->block(), params_dict);
  auto clone_node = n_graph->createClone(
      n, [&n_graph, &vals_to_params_map, opset_version](Value* v) {
        auto v_n = v->node();
        switch (v_n->kind()) {
          case ::c10::onnx::Constant: {
            // Clone the input if it is constant.
            auto constant_n = n_graph->insertNode(
                n_graph->createClone(v_n, [](Value* v) { return v; }));
            return constant_n->output();
          }
          case ::c10::prim::ListConstruct: {
            return CloneValueFromListConstruct(v, n_graph, opset_version);
          }
          case ::c10::prim::PackPadded: {
            auto input = n_graph->addInput();
            input->copyMetadata(v_n->input(0));
            return input;
          }
          default: {
            if (vals_to_params_map.find(v) != vals_to_params_map.end()) {
              // If the input is a parameter, insert a constant of its value as
              // input.
              auto val = vals_to_params_map.find(v)->second.second.toTensor();
              return n_graph
                  ->insertNode(n_graph->create(::c10::onnx::Constant)
                                   ->t_(attr::value, val))
                  ->output();
            } else {
              // If the input is not constant, we cannot depend on its value
              // in shape inference. Set it to graph input in the new graph,
              // and copy over metadata, such as datatype and shape.
              auto input = n_graph->addInput();
              input->copyMetadata(v);
              return input;
            }
          }
        }
      });
  return clone_node;
}

bool IsGraphValidForInference(std::shared_ptr<Graph> graph) {
  // Verify if every input has type(either Tensor or Sequence) and scalar type.
  // This is a requirement for ONNX graph inputs.
  for (auto in : graph->inputs()) {
    if (auto t_type = in->type()->cast<TensorType>()) {
      if (!t_type->scalarType().has_value()) {
        GRAPH_UPDATE(
            "Input ", in->debugName(), " is tensor type, but miss datatype.");
        return false;
      }
    } else if (auto s_type = in->type()->cast<ListType>()) {
      auto e_type = s_type->getElementType();
      if (auto t_type = e_type->cast<TensorType>()) {
        if (t_type->scalarType().has_value()) {
          continue;
        }
      }
      GRAPH_UPDATE(
          "Input ", in->debugName(), " is sequence type, but miss datatype.");
      return false;
    }
  }
  return true;
}

void ConvertGraphToONNXProto(
    std::shared_ptr<Graph> graph,
    std::shared_ptr<onnx::ModelProto>& model_proto,
    SymbolDimMap& symbol_map,
    int opset_version) {
  RawDataExportMap export_map;
  std::tie(model_proto, export_map, symbol_map) = export_onnx(
      graph,
      {},
      opset_version,
      {},
      false,
      onnx_torch::OperatorExportTypes::ONNX,
      true,
      true,
      {},
      true,
      false,
      std::string());
  for (int i = 0; i < model_proto->graph().output_size(); ++i) {
    model_proto->mutable_graph()->mutable_output(i)->clear_type();
  }
}

// this function checks wheather the blocks of If node have the same return
// type.
bool IsBlockReturnTypeSame(Node* n) {
  TORCH_INTERNAL_ASSERT(n->kind() == ::c10::onnx::If);
  auto then_block = n->blocks()[0];
  auto else_block = n->blocks()[1];
  for (size_t i = 0; i < n->outputs().size(); i++) {
    // check the type
    auto then_block_type = then_block->outputs()[i]->type();
    auto else_block_type = else_block->outputs()[i]->type();
    if (then_block_type->cast<TensorType>() &&
        else_block_type->cast<TensorType>()) {
      if (then_block_type->castRaw<TensorType>()->scalarType() !=
          else_block_type->castRaw<TensorType>()->scalarType()) {
        return false;
      }
    }
  }
  return true;
}

c10::optional<at::Tensor> ComputeConstantFolding(Node* n, int opset_version) {
  if (n->inputs().size() == 0) {
    return c10::nullopt;
  }
  std::vector<at::Tensor> inputTensorValues;
  for (auto i = 0; i < n->inputs().size(); i++) {
    if (TensorTypePtr input_type = n->input(i)->type()->cast<TensorType>()) {
      if (!ConstantValueMap::HasValue(n->input(i)->debugName())) {
        return c10::nullopt;
      }
      auto tensor_value =
          ConstantValueMap::GetValue(n->input(i)->debugName()).value();
      inputTensorValues.emplace_back(tensor_value);
    }
  }
  if (inputTensorValues.size() < n->inputs().size()) {
    return c10::nullopt;
  }
  // The _jit_pass_onnx_fold_if pass is processed after onnx pass,
  // therefore the onnx graph here may contain the if blocks that is never
  // traced. Constant folding on those if blocks may rely on the input shape
  // which does not meet the criteria, so it may get errors. A possible solution
  // is to put _jit_pass_onnx_fold_if pass in an earlier stage.
  try {
    return onnx_constant_fold::runTorchBackendForOnnx(
        n, inputTensorValues, opset_version);
  } catch (const std::exception& ex) {
    TORCH_WARN(
        "Constant folding in symbolic shape inference fails: ", ex.what());
    return c10::nullopt;
  }
}

// When the Reshape node's two inputs are constant, compute the output shape.
// The reshape value 0 and -1 are converted to the real value explicitly.
std::vector<int64_t> ComputeShapeFromReshape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& reshape) {
  TORCH_INTERNAL_ASSERT(
      input_shape.size() > 0 || reshape.size() > 0,
      "Reshape node should have at least one input size > 0 when constant folding.");
  // Case: reshape.size() == 0
  // %22 : int[] = prim::Constant[value=annotate(List[int], [])]()
  // %15 : Float(requires_grad=0, device=cpu) = aten::view(%1, %22)
  if (reshape.size() == 0) {
    return input_shape;
  }
  // Case: input_shape.size() == 0
  // (1) input_shape is not obtained,
  // (2) input_shape is scalar (output is still a tensor, not a scalar),
  // Both cases return reshape
  // TODO: for (1), multiple -1 may conflict each other. Consider use
  // newSymbol() in shapeMap.
  if (input_shape.size() == 0) {
    return reshape;
  }
  auto reshape_size = static_cast<int>(reshape.size());
  auto it_0 = std::find(reshape.begin(), reshape.end(), 0);
  auto reshape_has_zero = it_0 != reshape.end();
  auto input_shape_size = static_cast<int>(input_shape.size());
  auto it_minus_one = std::find(reshape.begin(), reshape.end(), -1);
  int minus_one_pos = it_minus_one == reshape.end()
      ? -1
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      : std::distance(reshape.begin(), it_minus_one);

  if (!reshape_has_zero && minus_one_pos == -1) {
    return reshape;
  }
  std::vector<int64_t> final_shape;
  // shape_ratio is used to calculate the real value of -1.
  // One example with reshape 0 and -1:
  // input_shape = 2 16 5 4
  // reshape = -1 0 4
  // final_shape = 10 16 4
  double shape_ratio = 1.0;
  for (auto i = 0; i < input_shape_size; i++) {
    shape_ratio *= static_cast<double>(input_shape[i]);
  }
  for (auto i = 0; i < reshape_size; i++) {
    if (i != minus_one_pos) {
      if (reshape[i] != 0) {
        shape_ratio /= static_cast<double>(reshape[i]);
      } else {
        shape_ratio /= static_cast<double>(input_shape[i]);
      }
    }
  }

  for (auto i = 0; i < minus_one_pos; i++) {
    int64_t cur_shape = reshape[i] == 0 ? input_shape[i] : reshape[i];
    final_shape.push_back(cur_shape);
  }
  final_shape.push_back(static_cast<int64_t>(std::round(shape_ratio)));
  for (auto i = minus_one_pos + 1; i < reshape_size; i++) {
    int64_t cur_shape = reshape[i] == 0 ? input_shape[i] : reshape[i];
    final_shape.push_back(cur_shape);
  }
  return final_shape;
}

c10::optional<::c10::SymbolicShape> ComputeShapeFromExpand(
    const std::vector<::c10::ShapeSymbol>& input_shape,
    const std::vector<int64_t>& reshape) {
  // NOLINTNEXTLINE(modernize-loop-convert)
  for (auto it = reshape.begin(); it != reshape.end(); ++it) {
    if (*it < 0) {
      return c10::nullopt;
    }
  }
  std::vector<::c10::ShapeSymbol> final_shape;
  if (input_shape.size() >= reshape.size()) {
    final_shape = input_shape;
  } else {
    for (auto v : reshape) {
      final_shape.emplace_back(::c10::ShapeSymbol::fromStaticSize(v));
    }
  }
  auto min_size = std::min(input_shape.size(), reshape.size());
  for (auto i = 0; i < min_size; i++) {
    auto idx = final_shape.size() - i - 1;
    auto input_shape_idx = input_shape.size() - i - 1;
    auto reshape_idx = reshape.size() - i - 1;
    if (input_shape[input_shape_idx].is_static()) {
      auto input_shape_value = input_shape[input_shape_idx].static_size();
      auto reshape_value = reshape[reshape_idx];
      TORCH_INTERNAL_ASSERT(
          input_shape_value == reshape_value || input_shape_value == 1 ||
              reshape_value == 1,
          "ONNX Expand input shape constraint not satisfied.");
      final_shape[idx] = ::c10::ShapeSymbol::fromStaticSize(
          std::max(input_shape_value, reshape_value));

    } else {
      final_shape[idx] = ::c10::ShapeSymbol::newSymbol();
    }
  }
  ::c10::SymbolicShape shape(final_shape);
  return shape;
}

c10::optional<::c10::SymbolicShape> ComputeShapeFromTile(
    const std::vector<::c10::ShapeSymbol>& input_shape,
    const std::vector<int64_t>& reshape) {
  TORCH_INTERNAL_ASSERT(
      input_shape.size() == reshape.size(),
      "ONNX Tile input shapes do not match.");
  // NOLINTNEXTLINE(modernize-loop-convert)
  for (auto it = reshape.begin(); it != reshape.end(); ++it) {
    if (*it < 0) {
      return c10::nullopt;
    }
  }
  std::vector<::c10::ShapeSymbol> final_shape;
  final_shape.reserve(input_shape.size());
  for (auto i = 0; i < input_shape.size(); i++) {
    if (input_shape[i].is_static()) {
      final_shape.emplace_back(::c10::ShapeSymbol::fromStaticSize(
          input_shape[i].static_size() * reshape[i]));
    } else {
      final_shape.emplace_back(::c10::ShapeSymbol::newSymbol());
    }
  }
  ::c10::SymbolicShape shape(final_shape);
  return shape;
}

void UpdateRank(Value* value, size_t rank) {
  ConstantValueMap::SetRank(value->debugName(), rank);
  if (TensorTypePtr value_type = value->type()->cast<TensorType>()) {
    c10::optional<size_t> rank_opt = rank;
    auto shape = ::c10::SymbolicShape(rank_opt);
    value->setType(value_type->withSymbolicShapes(shape));
  }
}

void UpdateShapeFromVector(
    Value* value,
    const std::vector<int64_t>& shape_size) {
  ::c10::SymbolicShape shape(shape_size);
  ConstantValueMap::SetShape(value->debugName(), shape);
  if (shape_size.empty()) {
    UpdateRank(value, 0);
    return;
  }
  ConstantValueMap::SetRank(value->debugName(), shape_size.size());
  if (TensorTypePtr value_type = value->type()->cast<TensorType>()) {
    value->setType(value_type->withSymbolicShapes(shape));
  }
}

void UpdateShape(Value* value, const ::c10::SymbolicShape& shape) {
  ConstantValueMap::SetShape(value->debugName(), shape);
  if (shape.rank().has_value()) {
    auto rank = shape.rank().value();
    if (rank == 0) {
      UpdateRank(value, 0);
      return;
    }
    ConstantValueMap::SetRank(value->debugName(), rank);
    if (TensorTypePtr value_type = value->type()->cast<TensorType>()) {
      value->setType(value_type->withSymbolicShapes(shape));
    }
  }
}

c10::optional<std::vector<int64_t>> GetValueFromListConstructNode(
    Node* lc_node) {
  auto rank = lc_node->inputs().size();
  std::vector<int64_t> shape_size;
  for (size_t i = 0; i < rank; i++) {
    if (TensorTypePtr shape_type =
            lc_node->input(i)->type()->cast<TensorType>()) {
      if (ConstantValueMap::HasValue(lc_node->input(i)->debugName())) {
        auto lc_value =
            ConstantValueMap::GetValue(lc_node->input(i)->debugName()).value();
        if (lc_value.dim() == 0) {
          auto lc_value_0 = lc_value.item<int64_t>();
          shape_size.emplace_back(static_cast<int64_t>(lc_value_0));
        }
      }
    }
  }
  return rank == shape_size.size()
      ? c10::optional<std::vector<int64_t>>(shape_size)
      : c10::nullopt;
}

void ProcessReshapeNode(Node* n) {
  if (ConstantValueMap::HasValue(n->input(1)->debugName())) {
    auto shape_temp =
        ConstantValueMap::GetValueInto1DInt64Vector(n->input(1)->debugName());
    auto shape_vector_0 =
        ConstantValueMap::GetShapeInto1DInt64VectorWithOneUnknown(
            n->input(0)->debugName());
    if (shape_vector_0.has_value()) {
      auto final_shape =
          ComputeShapeFromReshape(shape_vector_0.value(), shape_temp);
      UpdateShapeFromVector(n->output(), final_shape);
      return;
    }
  }

  if (ConstantValueMap::HasShape(n->input(1)->debugName())) {
    auto shape_vector_1 =
        ConstantValueMap::GetShapeInto1DInt64Vector(n->input(1)->debugName());
    if (shape_vector_1.has_value()) {
      TORCH_INTERNAL_ASSERT(shape_vector_1.value().size() == 1);
      UpdateRank(n->output(), shape_vector_1.value()[0]);
      return;
    }
  }

  // ListConstruct is handled at the beginning of ProcessConstantValueMap, no
  // further process here.
  if (TensorTypePtr shape_type = n->input(1)->type()->cast<TensorType>()) {
    // Set rank to Reshape output if possible.
    // From shape inference, we have:
    // %4236 : Float(*, device=cpu) = onnx::Transpose[perm=[0]](%4235)
    // %4237 : Long(2, strides=[1], device=cpu) = onnx::Concat[axis=0](%4232)
    // %4238 : FloatTensor(device=cpu) = onnx::Reshape(%4236, %4237)
    // We can have it as SymbolicShape with known rank:
    // %4238 : Float(*, *, strides=[2480, 1], requires_grad=0, device=cpu) =
    // onnx::Reshape(%4236, %4237)
    auto shape_type_dim = shape_type->dim();
    if (shape_type_dim.has_value()) {
      auto shape_type_size = shape_type->sizes()[0];
      if (shape_type_size.has_value()) {
        size_t rank = shape_type_size.value();
        UpdateRank(n->output(), rank);
      }
    }
  }
}

c10::SymbolicShape ComputeShapeForSlice(
    const std::vector<c10::ShapeSymbol>& input_shape,
    const std::vector<int64_t>& start_vector,
    const std::vector<int64_t>& end_vector,
    const std::vector<int64_t>& axes_vector,
    const std::vector<int64_t>& step_vector) {
  TORCH_INTERNAL_ASSERT(axes_vector.size() <= input_shape.size());
  TORCH_INTERNAL_ASSERT(axes_vector.size() == start_vector.size());
  TORCH_INTERNAL_ASSERT(axes_vector.size() == end_vector.size());
  TORCH_INTERNAL_ASSERT(axes_vector.size() == step_vector.size());
  std::vector<c10::ShapeSymbol> final_shape;
  final_shape = input_shape;
  for (auto idx = 0; idx < axes_vector.size(); ++idx) {
    auto axis = axes_vector[idx];
    if (axis < 0) {
      axis += input_shape.size();
    }
    if (!input_shape[axis].is_static()) {
      final_shape[axis] = c10::ShapeSymbol::newSymbol();
      continue;
    }
    auto input_shape_axis_value = input_shape[axis].static_size();
    auto cur_start = start_vector[idx];
    auto cur_end = end_vector[idx];
    auto cur_step = step_vector[idx];
    if (cur_start < -input_shape_axis_value) {
      cur_start = 0;
    } else if (cur_start < 0) {
      cur_start = input_shape_axis_value + cur_start;
    } else if (cur_start > input_shape_axis_value - 1) {
      cur_start = input_shape_axis_value;
    }
    if (cur_end < -input_shape_axis_value) {
      cur_end = -1;
    } else if (cur_end < 0) {
      cur_end = input_shape_axis_value + cur_end;
    } else if (cur_end > input_shape_axis_value - 1) {
      cur_end = input_shape_axis_value;
    }
    TORCH_INTERNAL_ASSERT(cur_step != 0);
    if (cur_step > 0) {
      final_shape[axis] = c10::ShapeSymbol::fromStaticSize(
          (cur_end - cur_start - 1) / cur_step + 1);
    } else {
      final_shape[axis] = c10::ShapeSymbol::fromStaticSize(
          (cur_start - cur_end - 1) / (-cur_step) + 1);
    }
  }
  return c10::SymbolicShape(final_shape);
}

void ProcessSliceNode(Node* n, int opset_version) {
  if (ConstantValueMap::HasShape(n->input(0)->debugName())) {
    auto shape_size_0 =
        ConstantValueMap::GetShape(n->input(0)->debugName()).value();
    if (shape_size_0.rank().has_value()) {
      auto input0_shape_value = shape_size_0.sizes().value();
      auto valid = true;
      if (opset_version >= 10) {
        valid = ConstantValueMap::HasValue(n->input(1)->debugName()) &&
            ConstantValueMap::HasValue(n->input(2)->debugName());
        for (auto input_idx = 3; input_idx < 5; ++input_idx) {
          if (n->inputs().size() > input_idx) {
            valid = valid &&
                ConstantValueMap::HasValue(n->input(input_idx)->debugName());
          }
        }
      }
      if (!valid) {
        if (ConstantValueMap::HasRank(n->input(0)->debugName())) {
          auto rank =
              ConstantValueMap::GetRank(n->input(0)->debugName()).value();
          UpdateRank(n->output(), rank);
        }
        return;
      }

      std::vector<int64_t> start_vector;
      std::vector<int64_t> end_vector;
      std::vector<int64_t> axes_vector(input0_shape_value.size(), 0);
      for (const auto i : c10::irange(input0_shape_value.size())) {
        axes_vector[i] = i;
      }
      std::vector<int64_t> step_vector;

      if (opset_version < 10) {
        start_vector = n->is(attr::starts);
        end_vector = n->is(attr::ends);
        if (n->hasAttributeS("axes")) {
          axes_vector = n->is(attr::axes);
        }
      } else {
        start_vector = ConstantValueMap::GetValueInto1DInt64Vector(
            n->input(1)->debugName());
        end_vector = ConstantValueMap::GetValueInto1DInt64Vector(
            n->input(2)->debugName());
        if (n->inputs().size() > 3) {
          axes_vector = ConstantValueMap::GetValueInto1DInt64Vector(
              n->input(3)->debugName());
        }
        if (n->inputs().size() > 4) {
          step_vector = ConstantValueMap::GetValueInto1DInt64Vector(
              n->input(4)->debugName());
        }
      }

      if (step_vector.empty()) {
        step_vector = std::vector<int64_t>(axes_vector.size(), 1);
      }

      auto final_shape = ComputeShapeForSlice(
          input0_shape_value,
          start_vector,
          end_vector,
          axes_vector,
          step_vector);
      UpdateShape(n->output(), final_shape);
    }
  }
}

void ProcessTimeSeriesNode(Node* n) {
  auto input0_shape = ConstantValueMap::GetShape(n->input(0)->debugName());
  auto input1_shape = ConstantValueMap::GetShape(n->input(1)->debugName());
  if (!(input0_shape.has_value() && input1_shape.has_value())) {
    return;
  }
  auto input0_shape_value = input0_shape.value().sizes();
  auto input1_shape_value = input1_shape.value().sizes();
  c10::ShapeSymbol seq_length;
  c10::ShapeSymbol num_directions;
  c10::ShapeSymbol batch_size;
  c10::ShapeSymbol hidden_size;
  if (input0_shape_value.has_value()) {
    seq_length = input0_shape_value.value()[0];
    batch_size = input0_shape_value.value()[1];
  }

  if (input1_shape_value.has_value()) {
    num_directions = input1_shape_value.value()[0];
    if (input1_shape_value.value()[1].is_static()) {
      auto input1_value = input1_shape_value.value()[1].static_size();
      switch (n->kind()) {
        case ::c10::onnx::RNN:
          hidden_size = c10::ShapeSymbol::fromStaticSize(input1_value);
          break;
        case ::c10::onnx::LSTM:
          hidden_size = c10::ShapeSymbol::fromStaticSize(input1_value / 4);
          break;
        case ::c10::onnx::GRU:
          hidden_size = c10::ShapeSymbol::fromStaticSize(input1_value / 3);
          break;
        default:
          throw std::runtime_error(
              std::string() + "This is not a valid TimeSeries Node with type " +
              n->kind().toDisplayString());
      }
    } else {
      hidden_size = c10::ShapeSymbol::newSymbol();
    }
  }

  if (n->outputs().size() > 1) {
    std::vector<c10::ShapeSymbol> final_shape = {
        seq_length, num_directions, batch_size, hidden_size};
    UpdateShape(n->output(0), c10::SymbolicShape(final_shape));
  }
  for (size_t idx = 2; idx < 4; ++idx) {
    if (n->outputs().size() > idx) {
      std::vector<c10::ShapeSymbol> final_shape = {
          num_directions, batch_size, hidden_size};
      UpdateShape(n->output(idx - 1), c10::SymbolicShape(final_shape));
    }
  }
}

// As an addition to onnx shape inference, this function leverages constant
// folding and a per-Op based process to update rank/shape for the graph, also
// it update ConstantValueMap accordingly.
void ComputeConstant(Node* n, int opset_version) {
  if (n->kind() == ::c10::onnx::Constant) {
    if (n->kindOf(attr::value) == AttributeKind::t) {
      at::Tensor const_val = n->t(attr::value);
      at::Tensor const_val_copy =
          at::empty(const_val.sizes(), const_val.options());
      const_val_copy.copy_(const_val);
      ConstantValueMap::SetValue(n->output()->debugName(), const_val_copy);
    }
    return;
  }
  auto only_rank_available = false;
  size_t rank = 0;

  // Constant folding.
  auto const_fold_val = ComputeConstantFolding(n, opset_version);
  if (const_fold_val.has_value()) {
    at::Tensor const_fold_val_copy = at::empty(
        const_fold_val.value().sizes(), const_fold_val.value().options());
    const_fold_val_copy.copy_(const_fold_val.value());
    ConstantValueMap::SetValue(n->output()->debugName(), const_fold_val_copy);
    UpdateShapeFromVector(n->output(), const_fold_val_copy.sizes().vec());
    return;
  }

  switch (n->kind()) {
    case ::c10::onnx::Shape: {
      auto input_shape =
          ConstantValueMap::GetShapeInto1DInt64Vector(n->input()->debugName());
      if (input_shape.has_value()) {
        auto shape_value = input_shape.value();
        // TODO: getDevice() ?
        auto options = c10::TensorOptions().dtype(at::kLong).device(at::kCPU);
        auto shape_value_size = static_cast<int64_t>(shape_value.size());
        auto f =
            at::from_blob(shape_value.data(), {shape_value_size}, at::kLong)
                .to(at::kCPU);
        // Need copy here
        at::Tensor f_copy = at::empty({shape_value_size}, options);
        f_copy.copy_(f);
        ConstantValueMap::SetValue(n->output()->debugName(), f_copy);
      }
      break;
    }
    case ::c10::onnx::Reshape: {
      ProcessReshapeNode(n);
      break;
    }
    case ::c10::onnx::Gather: {
      if (ConstantValueMap::HasRank(n->input(0)->debugName()) &&
          ConstantValueMap::HasRank(n->input(1)->debugName())) {
        auto rank_0 =
            ConstantValueMap::GetRank(n->input(0)->debugName()).value();
        auto rank_1 =
            ConstantValueMap::GetRank(n->input(1)->debugName()).value();
        only_rank_available = true;
        rank = rank_0 + rank_1 - 1;
      }
      break;
    }
    case ::c10::onnx::Transpose: {
      if (n->hasAttributeS("perm")) {
        auto perm_v = n->is(attr::perm);
        rank = perm_v.size();
        auto is_default_perm = false;
        if (rank == 2 && perm_v[0] == 1 && perm_v[1] == 0) {
          is_default_perm = true;
        }
        auto shape_updated = false;
        if (ConstantValueMap::HasShape(n->input(0)->debugName())) {
          auto shape_size_0 =
              ConstantValueMap::GetShape(n->input(0)->debugName())
                  .value()
                  .sizes();
          if (shape_size_0.has_value()) {
            auto shape_vector_0 = shape_size_0.value();
            std::vector<::c10::ShapeSymbol> final_shape_vector(
                shape_vector_0.size(), ::c10::ShapeSymbol());
            if (is_default_perm) {
              std::reverse_copy(
                  std::begin(shape_vector_0),
                  std::end(shape_vector_0),
                  std::begin(final_shape_vector));
            } else {
              for (auto i = 0; i < shape_vector_0.size(); i++) {
                final_shape_vector[i] = shape_vector_0[perm_v[i]];
              }
            }
            ::c10::SymbolicShape final_shape(final_shape_vector);
            UpdateShape(n->output(), final_shape);
            shape_updated = true;
          }
        }
        if (!shape_updated) {
          if (!is_default_perm) {
            only_rank_available = true;
          } else if (ConstantValueMap::HasRank(n->input(0)->debugName())) {
            rank = ConstantValueMap::GetRank(n->input(0)->debugName()).value();
            only_rank_available = true;
          }
        }
      }
      break;
    }
    case ::c10::onnx::ConstantOfShape: {
      if (ConstantValueMap::HasValue(n->input()->debugName())) {
        auto shape_temp = ConstantValueMap::GetValueInto1DInt64Vector(
            n->input()->debugName());
        UpdateShapeFromVector(n->output(), shape_temp);
        if (!shape_temp.empty()) {
          if (n->hasAttributeS("value")) {
            auto value = n->t(attr::value).repeat(shape_temp);
            ConstantValueMap::SetValue(n->output()->debugName(), value);
          } else {
            auto options =
                c10::TensorOptions().dtype(at::kFloat).device(at::kCPU);
            auto value = at::full({1}, 0.0, options).repeat(shape_temp);
            ConstantValueMap::SetValue(n->output()->debugName(), value);
          }
        }
      }
      break;
    }
    case ::c10::onnx::Expand: {
      if (ConstantValueMap::HasShape(n->input(0)->debugName())) {
        auto input0_shape_size =
            ConstantValueMap::GetShape(n->input(0)->debugName())
                .value()
                .sizes();
        if (input0_shape_size.has_value()) {
          auto input0_shape_value = input0_shape_size.value();
          if (ConstantValueMap::HasValue(n->input(1)->debugName())) {
            auto shape_temp = ConstantValueMap::GetValueInto1DInt64Vector(
                n->input(1)->debugName());
            auto final_shape =
                ComputeShapeFromExpand(input0_shape_value, shape_temp);
            if (final_shape.has_value()) {
              UpdateShape(n->output(), final_shape.value());
            }
          }
        }
      }
      break;
    }
    case ::c10::onnx::NonZero: {
      if (ConstantValueMap::HasRank(n->input()->debugName())) {
        auto rank = ConstantValueMap::GetRank(n->input()->debugName()).value();
        std::vector<c10::ShapeSymbol> dims;
        dims.emplace_back(
            c10::ShapeSymbol::fromStaticSize(static_cast<int64_t>(rank)));
        auto input_node = n->input()->node();
        if (input_node->kind() == ::c10::onnx::ConstantOfShape) {
          if (input_node->hasAttributeS("value")) {
            auto value =
                input_node->t(attr::value).toType(at::ScalarType::Float);
            auto value_a = value.accessor<float, 1>();
            if (value_a.size(0) == 1 && std::abs(value_a[0]) > 1e-6) {
              if (ConstantValueMap::HasShape(n->input()->debugName())) {
                auto shape_size_0 =
                    ConstantValueMap::GetShape(n->input()->debugName()).value();
                if (shape_size_0.isComplete()) {
                  auto shape_vector_0 = shape_size_0.sizes().value();
                  int64_t num_elements = 1;
                  for (auto cur_dim : shape_vector_0) {
                    num_elements *= cur_dim.static_size();
                  }
                  dims.emplace_back(c10::ShapeSymbol::fromStaticSize(
                      static_cast<int64_t>(num_elements)));
                }
              }
            }
          }
        }
        if (dims.size() == 1) {
          dims.emplace_back(c10::ShapeSymbol::newSymbol());
        }
        c10::SymbolicShape shape_v(dims);
        UpdateShape(n->output(), shape_v);
      }
      break;
    }
    case ::c10::onnx::RNN:
    case ::c10::onnx::LSTM:
    case ::c10::onnx::GRU: {
      ProcessTimeSeriesNode(n);
      break;
    }
    case ::c10::onnx::Slice: {
      ProcessSliceNode(n, opset_version);
      break;
    }
    case ::c10::onnx::Tile: {
      if (ConstantValueMap::HasShape(n->input(0)->debugName())) {
        auto input0_shape_size =
            ConstantValueMap::GetShape(n->input(0)->debugName())
                .value()
                .sizes();
        if (input0_shape_size.has_value()) {
          auto input0_shape_value = input0_shape_size.value();
          if (ConstantValueMap::HasValue(n->input(1)->debugName())) {
            auto shape_temp = ConstantValueMap::GetValueInto1DInt64Vector(
                n->input(1)->debugName());
            auto final_shape =
                ComputeShapeFromTile(input0_shape_value, shape_temp);
            if (final_shape.has_value()) {
              UpdateShape(n->output(), final_shape.value());
            }
          }
        }
      }
      break;
    }
    default: {
      break;
    }
  }
  if (n->outputs().size() > 1 ||
      ConstantValueMap::HasShape(n->output(0)->debugName())) {
    return;
  }
  if (only_rank_available) {
    UpdateRank(n->output(), rank);
  }
}

bool IsListConstructIntType(const Value* v) {
  if (v->node()->kind() == prim::ListConstruct) {
    auto listType = v->node()->output()->type();
    auto containedType = listType->containedTypes().at(0);
    if (containedType == IntType::get()) {
      return true;
    }
  }
  return false;
}

void ProcessConstantValueMap(Node* n, int opset_version) {
  // Update ConstantValueMap on node outputs from onnx shape inference
  // For outputs, only update static shapes. For input, we update symbolic
  // shapes also. ONNX If can have different types on different branches, skip
  // here.
  for (auto i = 0; i < n->outputs().size(); i++) {
    if (TensorTypePtr output_type = n->output(i)->type()->cast<TensorType>()) {
      if (output_type->dim().has_value()) {
        size_t rank = static_cast<size_t>(output_type->dim().value());
        ConstantValueMap::SetRank(n->output(i)->debugName(), rank);
        auto shape = output_type->symbolic_sizes();
        if (shape.isComplete()) {
          UpdateShape(n->output(i), shape);
        }
      }
    }
  }
  // Update ConstantValueMap on node inputs from onnx shape inference.
  // ListConstruct is handled here (we only consider IntType, not TensorType) ,
  // no need to have a per-op based process.
  for (auto i = 0; i < n->inputs().size(); i++) {
    if (TensorTypePtr input_type = n->input(i)->type()->cast<TensorType>()) {
      if (input_type->dim().has_value()) {
        size_t rank = static_cast<size_t>(input_type->dim().value());
        ConstantValueMap::SetRank(n->input(i)->debugName(), rank);
        auto shape = input_type->symbolic_sizes();
        if (!ConstantValueMap::HasShape(n->input(i)->debugName())) {
          UpdateShape(n->input(i), shape);
        }
      }
    } else if (IsListConstructIntType(n->input(i))) {
      auto lc_node = n->input(i)->node();
      auto rank = lc_node->inputs().size();
      auto lc_vector_optional = GetValueFromListConstructNode(lc_node);
      if (lc_vector_optional.has_value()) {
        auto lc_vector = lc_vector_optional.value();
        auto options = c10::TensorOptions().dtype(at::kLong).device(at::kCPU);
        auto lc_vector_size = static_cast<int64_t>(lc_vector.size());
        auto f = at::from_blob(lc_vector.data(), {lc_vector_size}, at::kLong)
                     .to(at::kCPU);
        // Need copy here
        at::Tensor f_copy = at::empty({lc_vector_size}, options);
        f_copy.copy_(f);
        ConstantValueMap::SetValue(n->input(i)->debugName(), f_copy);
        UpdateShapeFromVector(n->input(i), {lc_vector_size});
      } else {
        UpdateShapeFromVector(n->input(i), {static_cast<int64_t>(rank)});
      }
    }
  }
  // Additional logic to update the graph and ConstantValueMap
  ComputeConstant(n, opset_version);
}

// Any additional post process that are specific to individual node kind.
void SpecialPostProcess(Node* n) {
  switch (n->kind()) {
    case ::c10::onnx::If: {
      if (!IsBlockReturnTypeSame(n) && IsStaticConditionONNX(n)) {
        auto cond = ConditionValueONNX(n);
        auto block_idx = cond ? 0 : 1;
        for (size_t i = 0; i < n->outputs().size(); i++) {
          n->outputs()[i]->setType(
              n->blocks()[block_idx]->outputs()[i]->type());
        }
      }
      break;
    }
    case ::c10::onnx::SequenceInsert: {
      // Special case when input sequence to SequenceInsert is empty.
      // onnx Sequence type requires element type to be set.
      // If the list to insert is empty, we set the elem type by
      // looking at the tensor being inserted.
      auto list_node = n->input(0)->node();
      // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
      auto t_node = n->input(1)->node();
      if (!list_node || list_node->kind() != prim::ListConstruct ||
          list_node->inputs().size() != 0) {
        break;
      }

      if (TensorTypePtr t_type = n->input(1)->type()->cast<TensorType>()) {
        if (t_type->scalarType()) {
          n->output()->setType(ListType::create(t_type));
        }
      }
      break;
    }
    case ::c10::onnx::Cast: {
      // ONNX shape inference is not able to assign output tensor shape,
      // when input to onnx::Cast has incomplete tensor shape, for example
      // missing shape, rank, dtype, etc. This postprocess sets the correct
      // dtype for output tensor, since the dtype info is stored in Cast
      // attribute.
      TensorTypePtr t_type = n->output()->type()->cast<TensorType>();
      if (nullptr != t_type && !t_type->scalarType().has_value()) {
        auto onnx_dtype = n->i(attr::to);
        auto aten_dtype = ONNXTypeToATenType(onnx_dtype);
        n->output()->setType(t_type->withScalarType(aten_dtype));
      }
      break;
    }
    case ::c10::onnx::ConstantOfShape: {
      // ONNX shape inference is not able to propagate output tensor shape
      // for onnx::ConstantOfShape if input `shape` is not constant.
      // This is a temporary solution when some partial information is
      // available, for example, knowing rank of output tensor, or knowing
      // symbolic shape. This solution won't be needed once we have proper
      // symbolic propagation.
      auto shape_node = n->input(0)->node();
      if (shape_node->kind() == ::c10::onnx::Shape) {
        // Shape -> ConstantOfShape
        auto orig_type = shape_node->input()->type()->cast<TensorType>();
        auto v_type = n->output()->type()->cast<TensorType>();
        if (v_type && !v_type->sizes().concrete_sizes()) {
          if (orig_type && orig_type->dim()) {
            // Assign symbolic shape of original input of onnx::Shape.
            v_type = v_type->withSymbolicShapes(orig_type->symbolic_sizes());
            n->output()->setType(v_type);
          } else if (
              shape_node->input()->node()->kind() ==
              ::c10::prim::ListConstruct) {
            // Assign rank of original input of onnx::Shape.
            v_type = v_type->withSizes({static_cast<int64_t>(
                shape_node->input()->node()->inputs().size())});
            n->output()->setType(v_type);
          }
        }
      } else if (shape_node->kind() == ::c10::prim::ListConstruct) {
        // ListConstruct -> ConstantOfShape
        auto v_type = n->output()->type()->cast<TensorType>();
        if (v_type && !v_type->sizes().concrete_sizes()) {
          auto value = n->t(attr::value);
          v_type = v_type->withScalarType(value.scalar_type());
          std::vector<c10::ShapeSymbol> sizes(
              shape_node->inputs().size(), c10::ShapeSymbol::newSymbol());
          v_type = v_type->withSymbolicShapes(c10::SymbolicShape(sizes));
          n->output()->setType(v_type);
        }
      }
      break;
    }
  }
}

void UpdateOutputTypeByONNXProto(
    Node* n,
    Node* clone_node,
    const onnx::ModelProto& model_proto,
    SymbolDimMap& symbol_map) {
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  auto graph_proto = model_proto.graph();

  // get data from value_info and updated original graph.
  auto updateNodeOutputsByONNXValueInfo =
      [&](const onnx::ValueInfoProto& v_info) {
        for (size_t i = 0; i < n->outputs().size(); ++i) {
          if (clone_node->output(i)->debugName() == v_info.name()) {
            UpdateTorchValueByOnnxValueInfo(n->output(i), v_info, symbol_map);
          }
        }
      };

  // Check graph outputs for inferred shapes.
  for (size_t i = 0; i < graph_proto.output_size(); ++i) {
    updateNodeOutputsByONNXValueInfo(graph_proto.output(i));
  }

  // Check value_infos for inferred shapes.
  for (size_t i = 0; i < graph_proto.value_info_size(); ++i) {
    updateNodeOutputsByONNXValueInfo(graph_proto.value_info(i));
  }
}

void FetchBlockInputMetadataFromParent(Block* b) {
  auto n = b->owningNode();
  if (nullptr != n && n->kind() == ::c10::onnx::Loop) {
    // Copy node input metadata to subgraph input.
    for (size_t i = 0; i < n->inputs().size(); ++i) {
      b->inputs().at(i)->setType(n->inputs().at(i)->type());
    }
  }
}

void ONNXShapeTypeInference(
    Block* b,
    const ParamMap& params_dict,
    int opset_version) {
  FetchBlockInputMetadataFromParent(b);
  auto valsToParamsMap = buildValueToParamsMap(b, params_dict);
  for (auto const& it : valsToParamsMap) {
    auto key = it.first;
    auto value = it.second;
    if (key->node()->kind() == prim::Param) {
      if (value.second.isTensor()) {
        ConstantValueMap::SetValue(value.first, value.second.toTensor());
      }
    } else if (key->node()->kind() == ::c10::onnx::Constant) {
      at::Tensor const_val = key->node()->t(attr::value);
      at::Tensor const_val_copy =
          at::empty(const_val.sizes(), const_val.options());
      const_val_copy.copy_(const_val);
      ConstantValueMap::SetValue(value.first, const_val_copy);
    } else {
      throw std::runtime_error(
          "ONNXShapeTypeInference - Unsupported kind of constant node found.");
    }
  }
  for (auto n : b->nodes()) {
    for (auto subblock : n->blocks()) {
      ONNXShapeTypeInference(subblock, params_dict, opset_version);
    }
    ONNXShapeTypeInference(n, params_dict, opset_version);
  }
}

} // namespace

void ONNXShapeTypeInference(
    Node* n,
    const ParamMap& params_dict,
    int opset_version) {
  GRAPH_UPDATE(
      "Running ONNX shape inference for node: ", n->kind().toDisplayString());
  if (!IsSupportedNode(n)) {
    return;
  }
  // Create a Graph containing only the single node n.
  // This graph is later converted to ONNX to run shape inference.
  auto n_graph = std::make_shared<Graph>();
  auto clone_node = CloneNodeToGraph(n, n_graph, params_dict, opset_version);
  n_graph->insertNode(clone_node);

  // Register all node outputs as graph outputs.
  for (auto output : clone_node->outputs()) {
    n_graph->registerOutput(output);
  }

  // Use scalar_type_analysis without low precision cast
  ScalarTypeAnalysisForONNX(n_graph, false, opset_version);

  GRAPH_DEBUG("Original torch graph: ", n->owningGraph()->toString());
  GRAPH_DEBUG(
      "Cloned torch graph to run shape inference: ", n_graph->toString());

  if (IsGraphValidForInference(n_graph)) {
    // TODO: Some ops have conversion happen at Peephole pass.
    //       The conversion here is incomplete for these ops.
    //       e.g: ListConstruct, ListUnpack, etc.
    std::shared_ptr<onnx::ModelProto> model_proto;
    SymbolDimMap symbol_map;
    ConvertGraphToONNXProto(n_graph, model_proto, symbol_map, opset_version);
    GRAPH_DEBUG(
        "ONNX graph to run shape inference: ", prettyPrint(*model_proto));

    // infer shape
    try {
      onnx::shape_inference::InferShapes(*model_proto);
      UpdateOutputTypeByONNXProto(n, clone_node, *model_proto, symbol_map);
    } catch (std::runtime_error& ex) {
      // TODO: include this as warning once we have a more consolidated warning
      // system.
      GRAPH_DEBUG(
          "ONNX shape inference fails with: ",
          ex.what(),
          " on graph: ",
          n_graph->toString());
      // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
      const char shape_err[] = "ShapeInferenceError";
      // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
      const char type_err[] = "TypeInferenceError";
      // NOLINTNEXTLINE(modernize-use-nullptr)
      if ((strstr(ex.what(), shape_err) == NULL) &&
          // NOLINTNEXTLINE(modernize-use-nullptr)
          (strstr(ex.what(), type_err) == NULL))
        throw;
    }
    GRAPH_DEBUG(
        "ONNX graph after shape inference: ", prettyPrint(*model_proto));
  }

  SpecialPostProcess(n);
  ProcessConstantValueMap(n, opset_version);
  GRAPH_DEBUG(
      "Torch graph after shape inference:", n->owningGraph()->toString());
}

void ONNXSetDynamicInputShape(
    std::shared_ptr<Graph>& graph,
    const std::unordered_map<
        std::string,
        std::unordered_map<int64_t, std::string>>& dynamic_axes,
    const std::vector<std::string>& input_names) {
  GRAPH_UPDATE("ONNX set dynamic input shape.");
  GRAPH_UPDATE("dynamic axes tensor names:", [&]() {
    std::vector<std::string> res(dynamic_axes.size());
    std::transform(
        dynamic_axes.begin(), dynamic_axes.end(), res.begin(), [](auto pair) {
          return pair.first;
        });
    return res;
  }());

  std::map<std::string, ::c10::ShapeSymbol> name_to_sym;

  for (int i = 0; i < input_names.size(); ++i) {
    auto input_name = input_names[i];
    if (dynamic_axes.find(input_name) != dynamic_axes.end()) {
      auto axes_names = dynamic_axes.find(input_name)->second;
      TORCH_INTERNAL_ASSERT(i < graph->inputs().size());
      auto input_tensor_type = graph->inputs()[i]->type()->cast<TensorType>();
      if (!input_tensor_type) {
        continue;
      }

      auto shape_ref = input_tensor_type->symbolic_sizes().sizes();
      TORCH_CHECK(
          shape_ref.has_value(), "Input tensor shape should have value.");
      auto shape = shape_ref.value();

      // NOLINTNEXTLINE(performance-for-range-copy)
      for (auto pair : axes_names) {
        auto axis = pair.first;
        auto name = pair.second;
        if (name_to_sym.find(name) == name_to_sym.end()) {
          name_to_sym[name] = ::c10::ShapeSymbol::newSymbol();
        }
        TORCH_CHECK(
            axis < shape.size(),
            "Dynamic shape axis should be no more than the shape dimension for ",
            name);
        shape[axis] = name_to_sym[name];
      }

      graph->inputs()[i]->setType(
          input_tensor_type->withSymbolicShapes(::c10::SymbolicShape(shape)));
    }
  }
}

bool HasSequenceTypeOutput(Node* node) {
  if (node->kind() == ::c10::onnx::SplitToSequence ||
      node->kind() == ::c10::onnx::SequenceInsert ||
      node->kind() == ::c10::onnx::SequenceEmpty ||
      node->kind() == ::c10::onnx::SequenceErase ||
      node->kind() == ::c10::onnx::SequenceConstruct ||
      node->kind() == ::c10::onnx::Loop || node->kind() == ::c10::onnx::If)
    return true;
  return false;
}

void ONNXUpdateTypeFromTensor(
    Value* graph_output,
    const at::Tensor& output,
    bool onnx_shape_inference) {
  if (onnx_shape_inference) {
    graph_output->setType(
        MergeInferredType(TensorType::create(output), graph_output->type()));
  } else {
    graph_output->inferTypeFrom(output);
  }
}

// Recursively look into elements in `output_obj`, and assign shape/type info
// into flattened graph outputs. `outputs_index` is passed in to point to the
// current index in flattened graph outputs. The updated `outputs_index` is
// returned at the end of the function.
size_t ONNXAssignOutputShape(
    std::shared_ptr<Graph>& graph,
    size_t outputs_index,
    PyObject* output_obj,
    bool onnx_shape_inference) {
  auto index_check = [&]() {
    TORCH_INTERNAL_ASSERT(
        outputs_index >= 0 && outputs_index <= graph->outputs().size(),
        "Incorrect number of elements provided as example outputs.");
  };

  index_check();

  if (THPVariable_Check(output_obj)) {
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    at::Tensor var = THPVariable_Unpack(output_obj);
    ONNXUpdateTypeFromTensor(
        graph->outputs().at(outputs_index), var, onnx_shape_inference);
    outputs_index++;
  } else if (PyTuple_Check(output_obj)) {
    size_t tuple_len = PyTuple_GET_SIZE(output_obj);
    for (size_t i = 0; i < tuple_len; ++i) {
      outputs_index = ONNXAssignOutputShape(
          graph,
          outputs_index,
          PyTuple_GET_ITEM(output_obj, i),
          onnx_shape_inference);
    }
  } else if (PyList_Check(output_obj)) {
    size_t list_len = PyList_GET_SIZE(output_obj);
    if (HasSequenceTypeOutput(graph->outputs().at(outputs_index)->node())) {
      auto output_type = graph->outputs().at(outputs_index)->type();
      TORCH_CHECK(
          output_type->cast<ListType>(),
          "Expected a sequence type, but received a non-iterable type in graph output index ",
          outputs_index);
      if (list_len > 0) {
        auto list_elem = PyList_GET_ITEM(output_obj, 0);
        TORCH_INTERNAL_ASSERT(THPVariable_Check(list_elem));
        auto& var = THPVariable_Unpack(list_elem);
        for (size_t i = 1; i < list_len; ++i) {
          list_elem = PyList_GET_ITEM(output_obj, i);
          TORCH_INTERNAL_ASSERT(THPVariable_Check(list_elem));
          auto& new_var = THPVariable_Unpack(list_elem);
          TORCH_CHECK(
              var.scalar_type() == new_var.scalar_type(),
              "Unsupported sequence type in model outputs. ONNX supports sequences of elements of the same data type.");
        }
        auto elem_type = graph->outputs()
                             .at(outputs_index)
                             ->type()
                             ->castRaw<ListType>()
                             ->getElementType()
                             ->cast<TensorType>();
        elem_type = elem_type->withScalarType(var.scalar_type());
        graph->outputs()
            .at(outputs_index)
            ->setType(MergeInferredType(
                graph->outputs().at(outputs_index)->type(),
                ListType::create(elem_type)));
      } else {
        graph->outputs()
            .at(outputs_index)
            ->setType(graph->outputs().at(outputs_index)->type());
      }
      outputs_index++;
    } else {
      // When torch output is a list type, but ONNX node is not a
      // sequence type. Like prim::ListConstruct
      for (size_t i = 0; i < list_len; ++i) {
        outputs_index = ONNXAssignOutputShape(
            graph,
            outputs_index,
            PyList_GET_ITEM(output_obj, i),
            onnx_shape_inference);
      }
    }
  } else if (PyDict_Check(output_obj)) {
    // Support for dict data type is limited to fixed size dictionaries in
    // ONNX.
    // Dictionary values are unrolled and keys are not preserved.
    auto unrolled_dict =
        py::reinterpret_borrow<py::list>(PyDict_Items(output_obj));
    TORCH_INTERNAL_ASSERT(PyList_Check(unrolled_dict.ptr()));
    for (size_t i = 0; i < unrolled_dict.size(); ++i) {
      outputs_index = ONNXAssignOutputShape(
          graph,
          outputs_index,
          PyList_GET_ITEM(unrolled_dict.ptr(), i),
          onnx_shape_inference);
    }
  } else if (THPUtils_checkString(output_obj)) {
    // Ignore string, since they are not supported as output in ONNX.
  } else if (strcmp(THPUtils_typename(output_obj), "NoneType") == 0) {
    // For cases with tracing, simply ignore NoneType outputs
    // For cases with scripting, TODO: Add logic to handle NoneType outputs
    // when such output types are supported. For now test cases with NoneType
    // outputs have been disabled.
  } else {
    std::string msg =
        "Only tuples, lists and Variables are supported as JIT inputs/outputs. "
        "Dictionaries and strings are also accepted, but their usage is not "
        "recommended. Here, received an input of unsupported type: ";
    msg += THPUtils_typename(output_obj);
    throw std::runtime_error(msg);
  }

  index_check();

  return outputs_index;
}

void ONNXAssignOutputShape(
    std::shared_ptr<Graph>& graph,
    at::ArrayRef<at::Tensor> outputs,
    const python::IODescriptor& desc,
    bool onnx_shape_inference) {
  size_t outputs_index = 0;
  PyObject* py_obj = unflatten(outputs, desc);
  TORCH_INTERNAL_ASSERT(PyTuple_Check(py_obj));

  outputs_index =
      ONNXAssignOutputShape(graph, outputs_index, py_obj, onnx_shape_inference);

  TORCH_INTERNAL_ASSERT(
      outputs_index == graph->outputs().size(),
      "Incorrect number of elements provided as example outputs.");

  Py_DECREF(py_obj);
}

void ONNXShapeTypeInference(
    std::shared_ptr<Graph>& graph,
    const ParamMap& params_dict,
    int opset_version) {
  ConstantValueMap::ClearMaps();
  ONNXShapeTypeInference(graph->block(), params_dict, opset_version);
}

} // namespace jit
} // namespace torch
