#include <torch/csrc/jit/passes/onnx/shape_type_inference.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/onnx.h>

#include <onnx/shape_inference/implementation.h>

namespace torch {
namespace jit {

namespace {
namespace onnx_torch = ::torch::onnx;
namespace onnx = ::ONNX_NAMESPACE;

void UpdateTorchValueByOnnxValueInfo(
    Value* v,
    const onnx::ValueInfoProto& p_info) {
  if (!p_info.has_type()) {
    return;
  }

  auto p_type = p_info.type();
  if (!p_type.has_tensor_type()) {
    // TODO: Support sequence type.
    return;
  }

  auto p_tensor_type = p_type.tensor_type();

  c10::optional<at::ScalarType> scalar_type;
  if (p_tensor_type.has_elem_type()) {
    scalar_type = ONNXTypeToATenType(p_tensor_type.elem_type());
  }

  auto v_type = TensorType::create(
      scalar_type,
      at::kCPU,
      c10::SymbolicShape(),
      c10::VaryingShape<c10::Stride>{},
      {});
  if (p_tensor_type.has_shape()) {
    std::vector<int64_t> sizes;
    auto p_shape = p_tensor_type.shape();

    for (int i = 0; i < p_shape.dim_size(); ++i) {
      auto& dim = p_shape.dim(i);
      if (dim.has_dim_value()) {
        sizes.push_back(dim.dim_value());
      } else {
        // TODO: handle dim_param?
        return;
      }
    }
    v_type = TensorType::create(scalar_type, at::kCPU, sizes.size(), {});
    v_type = v_type->withSizes(sizes);
  }

  v->setType(v_type);
}

bool IsSupportedNode(const Node* n) {
  auto node_kind = n->kind();

  if (!node_kind.is_onnx()) {
    // node kind is not ONNX, skipped.
    return false;
  }

  if (node_kind == ::c10::onnx::SequenceAt ||
      node_kind == ::c10::onnx::SplitToSequence ||
      node_kind == ::c10::onnx::SequenceConstruct ||
      node_kind == ::c10::onnx::SequenceEmpty ||
      node_kind == ::c10::onnx::SequenceInsert ||
      node_kind == ::c10::onnx::ConcatFromSequence ||
      node_kind == ::c10::onnx::SequenceErase) {
    // TODO: ONNX unable to do shape inference for these ops.
    return false;
  }

  if (node_kind == ::c10::onnx::ConstantOfShape) {
    // && n->input()->node()->kind() == ::c10::prim::ListConstruct
    // TODO: ONNX shape inference segfault.
    return false;
  }

  if (node_kind == ::c10::onnx::Loop || node_kind == ::c10::onnx::If) {
    // TODO: Support Loop & If shape inference by propagating input shape to
    // block input.
    return false;
  }

  return true;
}

} // namespace

void ONNXShapeTypeInference(Node* n, int opset_version) {
  if (!IsSupportedNode(n)) {
    return;
  }

  // Create a Graph containing only the single node n.
  // This graph is later converted to ONNX to run shape inference.
  auto n_graph = std::make_shared<Graph>();
  // Clone the node n for the new graph.
  auto clone_node = n_graph->createClone(n, [&n_graph](Value* v) {
    auto v_n = v->node();
    if (v_n->kind() == ::c10::onnx::Constant) {
      // Clone the input if it is constant.
      auto constant_n = n_graph->insertNode(
          n_graph->createClone(v_n, [](Value* v) { return v; }));
      return constant_n->output();
    } else {
      // If the input is not constant, we cannot depend on its value
      // in shape inference. Set it to graph input in the new graph,
      // and copy over metadata, such as datatype and shape.
      auto input = n_graph->addInput();
      input->copyMetadata(v);
      return input;
    }
  });
  n_graph->insertNode(clone_node);
  // Register all node outputs as graph outputs.
  for (auto output : clone_node->outputs()) {
    n_graph->registerOutput(output);
  }

  // TODO: Some ops have conversion happen at Peephole pass.
  //       The conversion here is incomplete for these ops.
  //       e.g: ListConstruct, ListUnpack, etc.
  std::string model_str;
  RawDataExportMap export_map;
  std::tie(model_str, export_map) = export_onnx(
      n_graph,
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
  onnx::ModelProto model_proto;
  model_proto.ParseFromString(model_str);

  // infer shape
  onnx::shape_inference::InferShapes(model_proto);
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
}

} // namespace jit
} // namespace torch
