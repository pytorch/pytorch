#include <torch/csrc/jit/passes/onnx/shape_type_inference.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/onnx.h>

#include <onnx/shape_inference/implementation.h>

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
    if (new_tensor_type->sizes().isComplete()) {
      type = type->withSizes(new_tensor_type->sizes().concrete_sizes().value());
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
    const onnx::TypeProto_Tensor& onnx_tensor_type) {
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
    std::vector<int64_t> sizes;
    auto onnx_shape = onnx_tensor_type.shape();

    for (int i = 0; i < onnx_shape.dim_size(); ++i) {
      auto& dim = onnx_shape.dim(i);
      if (dim.has_dim_value()) {
        sizes.push_back(dim.dim_value());
      } else {
        // TODO: handle dim_param?
        return v_type;
      }
    }
    v_type = TensorType::create(scalar_type, at::kCPU, sizes.size(), {});
    v_type = v_type->withSizes(sizes);
  }

  return v_type;
}

ListTypePtr TorchListTypeFromONNX(
    const onnx::TypeProto_Sequence& onnx_sequence_type) {
  c10::optional<at::ScalarType> scalar_type;
  if (onnx_sequence_type.has_elem_type()) {
    auto onnx_seq_elem_type = onnx_sequence_type.elem_type();
    if (onnx_seq_elem_type.has_tensor_type()) {
      auto onnx_tensor_type = onnx_seq_elem_type.tensor_type();
      auto v_tensor_type = TorchTensorTypeFromONNX(onnx_tensor_type);
      auto v_type = ListType::create(v_tensor_type);
      return v_type;
    }
  }
  return nullptr;
}

void UpdateTorchValueByOnnxValueInfo(
    Value* v,
    const onnx::ValueInfoProto& p_info) {
  if (!p_info.has_type()) {
    return;
  }

  auto p_type = p_info.type();
  if (p_type.has_tensor_type()) {
    auto torch_tensor_type = TorchTensorTypeFromONNX(p_type.tensor_type());
    if (torch_tensor_type) {
      v->setType(torch_tensor_type);
    }
  } else if (p_type.has_sequence_type()) {
    auto torch_list_type = TorchListTypeFromONNX(p_type.sequence_type());
    if (torch_list_type) {
      v->setType(torch_list_type);
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
  if ((node_kind == ::c10::onnx::Loop || node_kind == ::c10::onnx::If) &&
      n->blocks().size() == 0) {
    return false;
  }

  return true;
}

// Clone the node n for the new graph.
Node* CloneNodeToGraph(Node* n, std::shared_ptr<Graph> n_graph) {
  auto clone_node = n_graph->createClone(n, [&n_graph](Value* v) {
    auto v_n = v->node();
    if (v_n->kind() == ::c10::onnx::Constant) {
      // Clone the input if it is constant.
      auto constant_n = n_graph->insertNode(
          n_graph->createClone(v_n, [](Value* v) { return v; }));
      return constant_n->output();
    } else if (v_n->kind() == ::c10::prim::ListConstruct) {
      // In jit/passes/onnx/peephole.cpp::eraseListConstruct,
      // prim::ListConstruct is converted to onnx::Concat. The conversion should
      // eventually be moved to symbolic. For now, treat this operator as
      // special case, and change from list type to tensor type. The scalar type
      // is preserved.
      TypePtr elem = v->type()->cast<ListType>()->getElementType();
      c10::optional<at::ScalarType> scalar_type = c10::nullopt;
      if (elem->cast<IntType>()) {
        scalar_type = at::kLong;
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
    } else {
      // If the input is not constant, we cannot depend on its value
      // in shape inference. Set it to graph input in the new graph,
      // and copy over metadata, such as datatype and shape.
      auto input = n_graph->addInput();
      input->copyMetadata(v);
      return input;
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
    onnx::ModelProto& model_proto,
    int opset_version) {
  std::string model_str;
  RawDataExportMap export_map;
  std::tie(model_str, export_map) = export_onnx(
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
  model_proto.ParseFromString(model_str);
  for (int i = 0; i < model_proto.graph().output_size(); ++i) {
    model_proto.mutable_graph()->mutable_output(i)->clear_type();
  }
}

// Any additional post process that are specific to individual node kind.
void SpecialPostProcess(Node* n) {
  switch (n->kind()) {
    case ::c10::onnx::SequenceInsert: {
      // Special case when input sequence to SequenceInsert is empty.
      // onnx Sequence type requires element type to be set.
      // If the list to insert is empty, we set the elem type by
      // looking at the tensor being inserted.
      auto list_node = n->input(0)->node();
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
  }
}

void UpdateOutputTypeByONNXProto(
    Node* n,
    Node* clone_node,
    const onnx::ModelProto& model_proto) {
  auto graph_proto = model_proto.graph();
  // inferred shapes are stored in value_info.
  for (size_t i = 0; i < graph_proto.value_info_size(); ++i) {
    auto v_info = graph_proto.value_info(i);
    // get data from value_info and updated original graph.
    for (size_t j = 0; j < clone_node->outputs().size(); ++j) {
      if (clone_node->output(j)->debugName() == v_info.name()) {
        UpdateTorchValueByOnnxValueInfo(n->output(j), v_info);
      }
    }
  }

  SpecialPostProcess(n);
}

} // namespace

void ONNXShapeTypeInference(Node* n, int opset_version) {
  GRAPH_UPDATE(
      "Running ONNX shape inference for node: ", n->kind().toDisplayString());
  if (!IsSupportedNode(n)) {
    return;
  }
  // Create a Graph containing only the single node n.
  // This graph is later converted to ONNX to run shape inference.
  auto n_graph = std::make_shared<Graph>();
  auto clone_node = CloneNodeToGraph(n, n_graph);
  n_graph->insertNode(clone_node);
  // Register all node outputs as graph outputs.
  for (auto output : clone_node->outputs()) {
    n_graph->registerOutput(output);
  }

  GRAPH_DEBUG("Original torch graph: ", n->owningGraph()->toString());
  GRAPH_DEBUG(
      "Cloned torch graph to run shape inference: ", n_graph->toString());

  if (!IsGraphValidForInference(n_graph)) {
    GRAPH_UPDATE("Skipping ONNX shape inference for this node.");
    return;
  }

  // TODO: Some ops have conversion happen at Peephole pass.
  //       The conversion here is incomplete for these ops.
  //       e.g: ListConstruct, ListUnpack, etc.
  onnx::ModelProto model_proto;
  ConvertGraphToONNXProto(n_graph, model_proto, opset_version);
  GRAPH_DEBUG("ONNX graph to run shape inference: ", prettyPrint(model_proto));

  // infer shape
  onnx::shape_inference::InferShapes(model_proto);
  GRAPH_DEBUG("ONNX graph after shape inference: ", prettyPrint(model_proto));

  UpdateOutputTypeByONNXProto(n, clone_node, model_proto);
  GRAPH_DEBUG(
      "Torch graph after shape inference:", n->owningGraph()->toString());
}

} // namespace jit
} // namespace torch
