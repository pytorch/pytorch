#include <torch/csrc/jit/passes/onnx/shape_type_inference.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/python/python_arg_flatten.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/onnx.h>

#include <aten/src/ATen/InitialTensorOptions.h>
#include <onnx/shape_inference/implementation.h>

namespace torch {
namespace jit {

const int ONNX_TYPE_BOOL = 9;

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
    const SymbolDimMap& symbol_map) {
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
    auto onnx_shape = onnx_tensor_type.shape();

    for (int i = 0; i < onnx_shape.dim_size(); ++i) {
      auto& dim = onnx_shape.dim(i);
      if (dim.has_dim_value()) {
        sizes.emplace_back(c10::ShapeSymbol::fromStaticSize(dim.dim_value()));
      } else {
        GRAPH_UPDATE("Got dim_param:", dim.dim_param());
        c10::optional<c10::ShapeSymbol> sym = c10::nullopt;
        for (auto pair : symbol_map) {
          if (pair.second == dim.dim_param()) {
            sym = pair.first;
            break;
          }
        }
        if (!sym) {
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
    SymbolDimMap symbol_map) {
  c10::optional<at::ScalarType> scalar_type;
  if (onnx_sequence_type.has_elem_type()) {
    auto onnx_seq_elem_type = onnx_sequence_type.elem_type();
    if (onnx_seq_elem_type.has_tensor_type()) {
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
    SymbolDimMap symbol_map) {
  if (!p_info.has_type()) {
    return;
  }

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
      // is preserved. If the elemtype is Int, insert a onnx::Concat node into
      // the graph.
      TypePtr elem = v->type()->cast<ListType>()->getElementType();
      c10::optional<at::ScalarType> scalar_type = c10::nullopt;
      if (elem->cast<IntType>()) {
        scalar_type = at::kLong;

        auto lc_node = v->node();
        // ListConstruct Int[] output case, we need to transform to ONNX
        // Concat to ensure the output is a single tensor(dynamic) type in
        // order to be consumed as inputs
        std::vector<Value*> unsqueezed;
        for (auto* input : lc_node->inputs()) {
          Node* unsqueezed_node =
              n_graph->insertNode(n_graph->create(::c10::onnx::Unsqueeze, 1));
          auto new_input = n_graph->addInput();
          new_input->copyMetadata(input);
          unsqueezed_node->addInput(new_input);
          unsqueezed_node->is_(attr::axes, {0});
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
    const onnx::ModelProto& model_proto,
    SymbolDimMap symbol_map) {
  auto graph_proto = model_proto.graph();
  // inferred shapes are stored in value_info.
  for (size_t i = 0; i < graph_proto.value_info_size(); ++i) {
    auto v_info = graph_proto.value_info(i);
    // get data from value_info and updated original graph.
    for (size_t j = 0; j < clone_node->outputs().size(); ++j) {
      if (clone_node->output(j)->debugName() == v_info.name()) {
        UpdateTorchValueByOnnxValueInfo(n->output(j), v_info, symbol_map);
      }
    }
  }
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
    onnx::shape_inference::InferShapes(*model_proto);
    GRAPH_DEBUG(
        "ONNX graph after shape inference: ", prettyPrint(*model_proto));

    UpdateOutputTypeByONNXProto(n, clone_node, *model_proto, symbol_map);
  }

  SpecialPostProcess(n);
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

      auto shape = input_tensor_type->symbolic_sizes().sizes().value();

      for (auto pair : axes_names) {
        auto axis = pair.first;
        auto name = pair.second;
        if (name_to_sym.find(name) == name_to_sym.end()) {
          name_to_sym[name] = ::c10::ShapeSymbol::newSymbol();
        }
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
      node->kind() == ::c10::onnx::SequenceConstruct)
    return true;
  return false;
}

void ONNXUpdateTypeFromTensor(
    Value* graph_output,
    at::Tensor output,
    bool onnx_shape_inference) {
  if (onnx_shape_inference) {
    graph_output->setType(
        MergeInferredType(TensorType::create(output), graph_output->type()));
  } else {
    graph_output->inferTypeFrom(output);
  }
}

void ONNXAssignOutputShape(
    std::shared_ptr<Graph>& graph,
    at::ArrayRef<at::Tensor> outputs,
    const python::IODescriptor& desc,
    bool onnx_shape_inference) {
  size_t outputs_index = 0;

  auto py_obj = unflatten(outputs, desc);

  TORCH_INTERNAL_ASSERT(PyTuple_Check(py_obj));

  for (size_t i = 0; i < PyTuple_GET_SIZE(py_obj); ++i) {
    PyObject* elem = PyTuple_GET_ITEM(py_obj, i);

    if (PyList_Check(elem)) {
      size_t list_len = PyList_GET_SIZE(elem);
      if (HasSequenceTypeOutput(graph->outputs()[i]->node())) {
        if (list_len > 0) {
          auto& var =
              reinterpret_cast<THPVariable*>(PyList_GET_ITEM(elem, 0))->cdata;
          for (size_t j = 1; j < list_len; ++j) {
            PyObject* list_elem = PyList_GET_ITEM(elem, j);
            auto& new_var = reinterpret_cast<THPVariable*>(list_elem)->cdata;
            TORCH_INTERNAL_ASSERT(THPVariable_Check(list_elem));
            TORCH_INTERNAL_ASSERT(var.scalar_type() == new_var.scalar_type());
          }
          outputs_index += list_len;
          graph->outputs()[i]->setType(ListType::create(
              TensorType::create(var.scalar_type(), at::kCPU, {}, {})));
          ONNXUpdateTypeFromTensor(
              graph->outputs()[i], var, onnx_shape_inference);
        }
      } else {
        for (size_t j = 0; j < list_len; ++j) {
          ONNXUpdateTypeFromTensor(
              graph->outputs()[i + j],
              outputs[outputs_index],
              onnx_shape_inference);
          outputs_index++;
        }
      }
    } else if (PyTuple_Check(elem)) {
      size_t tuple_len = PyTuple_GET_SIZE(elem);
      if (tuple_len > 0) {
        at::Tensor var =
            reinterpret_cast<THPVariable*>(PyTuple_GET_ITEM(elem, 0))->cdata;
        ONNXUpdateTypeFromTensor(
            graph->outputs()[i], var, onnx_shape_inference);
        outputs_index += tuple_len;
      }
    } else if (THPVariable_Check(elem)) {
      at::Tensor var = reinterpret_cast<THPVariable*>(elem)->cdata;
      ONNXUpdateTypeFromTensor(graph->outputs()[i], var, onnx_shape_inference);
      outputs_index++;
    } else { // Dict
      TORCH_INTERNAL_ASSERT(PyDict_Check(elem));
      auto dict_items = py::reinterpret_borrow<py::list>(PyDict_Items(elem));
      for (size_t j = 0; j < dict_items.size(); ++j) {
        ONNXUpdateTypeFromTensor(
            graph->outputs()[i + j],
            outputs[outputs_index],
            onnx_shape_inference);
        outputs_index++;
      }
    }
  }

  TORCH_INTERNAL_ASSERT(
      outputs_index == outputs.size(),
      "Incorrect number of elements provided as example outputs.");
}

// Check if node is prim::Uninitialized,
// or output of prim::Uninitialized->onnx::Identity
Node* IsUninitializedNode(Node* n) {
  if (n->kind() == ::c10::onnx::Identity &&
      n->inputs()[0]->node()->kind() == prim::Uninitialized)
    return n->inputs()[0]->node();
  if (n->kind() == prim::Uninitialized)
    return n;
  return nullptr;
}

Node* CreateCastToBoolNode(Value* val, Graph* graph) {
  Node* cast_node = graph->create(::c10::onnx::Cast);
  cast_node->addInput(val);
  cast_node->i_(attr::to, ONNX_TYPE_BOOL);
  cast_node->output()->setType(BoolType::get());
  return cast_node;
}

// Infer shape and type of the uninitialized_output from the corresponding
// output of the other subblock. prim::Uninitialized node is proven to be
// unused. So replace this node with a constant of the inferred shape and type.
void InferShapeTypeForUninitializedOutput(
    Graph* graph,
    Block* block,
    Value* uninitialized_output,
    Value* other_output) {
  auto output_type = other_output->type()->expect<TensorType>();
  auto elem_type = at::initialTensorOptions().dtype(output_type->scalarType());
  Node* const_node = graph->create(::c10::onnx::Constant, 1);

  if (output_type->sizes().concrete_sizes().has_value()) {
    auto size = output_type->sizes().concrete_sizes().value();
    const_node->t_(attr::value, at::zeros(size, elem_type));
    const_node->output()->setType(other_output->type());
    const_node->output()->copyMetadata(other_output);
  } else {
    const_node->t_(attr::value, at::zeros({}, elem_type));
    const_node->output()->setType(
        TensorType::create(*(output_type->scalarType()), at::kCPU, {}, {}));
  }
  const_node->insertBefore(block->return_node());
  uninitialized_output->replaceAllUsesWith(const_node->output());
  uninitialized_output->node()->destroy();
}

// Corresponding outputs for ONNX If then and else subblocks should have
// same shape and type. This pass detects if prim::Uninitialized node
// appears as part of outputs of either of the subblocks, and infers
// shape and type from the corresponding output of the other subblock
// In the example graph below, shape and type of the subblock output %7
// for subblock 1 is inferred from %y.1. Shape and type of Subblock
// output %7 is inferred from %y.5.
//
// graph(%y.1 : Int(3:4, 4:1, requires_grad=0, device=cpu)):
//   ...
//   %7 : Tensor = prim::Uninitialized()
//   %16 : bool, %17 : Tensor, %y.14 : Tensor = prim::If(%15) #
//   test/onnx/test_pytorch_onnx_onnxruntime.py:614:20
//     block0():
//       %y.5 : Tensor = aten::add(%y.1, %3, %6) #
//       test/onnx/test_pytorch_onnx_onnxruntime.py:615:28
//       -> (%2, %7, %y.5)
//     block1():
//       -> (%1, %y.1, %7)
//   ...

void ONNXIfShapeTypeInference(Node* node) {
  for (Block* b : node->blocks()) {
    for (Node* n : b->nodes()) {
      if (n->kind() == ::c10::onnx::If) {
        ONNXIfShapeTypeInference(n);
      }
    }
  }

  if (node->kind() != ::c10::onnx::If) {
    return;
  }

  GRAPH_DUMP("Graph before fixing If shape type: ", node->owningGraph());
  auto* if_node = node;
  auto* graph = if_node->owningGraph();

  // Check if the input to ONNX If node is node Bool, and insert
  // cast to Bool if needed.
  if (!if_node->input()->type()->isSubtypeOf(BoolType::get())) {
    Node* cast_node = CreateCastToBoolNode(if_node->input(), graph);
    cast_node->insertBefore(if_node);
    if_node->replaceInputWith(if_node->input(), cast_node->output());
  }

  Block* then_block = if_node->blocks()[0];
  Block* else_block = if_node->blocks()[1];

  // Infer shape and type for subblock outputs
  TORCH_INTERNAL_ASSERT(
      then_block->outputs().size() == else_block->outputs().size())
  for (size_t i = 0; i < else_block->outputs().size(); i++) {
    Value* then_block_output = then_block->outputs()[i];
    Value* else_block_output = else_block->outputs()[i];

    // If both subblocks have an uninitialized output, shape and type cannot
    // be inferred.
    TORCH_CHECK(
        !(IsUninitializedNode(then_block_output->node()) &&
          IsUninitializedNode(else_block_output->node())),
        "Cannot infer shape and type for ONNX If with uninitialized output in both subblocks. Please check the model graph.");

    if (auto uninitialized_node =
            IsUninitializedNode(then_block_output->node())) {
      InferShapeTypeForUninitializedOutput(
          graph, then_block, then_block_output, else_block_output);
      if_node->outputs()[i]->setType(then_block->outputs()[i]->type());
      if (!uninitialized_node->hasUses())
        uninitialized_node->destroy();
    } else if (
        auto uninitialized_node =
            IsUninitializedNode(else_block_output->node())) {
      InferShapeTypeForUninitializedOutput(
          graph, else_block, else_block_output, then_block_output);
      if_node->outputs()[i]->setType(else_block->outputs()[i]->type());
      if (!uninitialized_node->hasUses())
        uninitialized_node->destroy();
    }
  }
}

void ONNXShapeTypeInference(std::shared_ptr<Graph>& graph, int opset_version) {
  for (auto n : graph->nodes()) {
    ONNXShapeTypeInference(n, opset_version);
    if (n->kind() == ::c10::onnx::If || n->kind() == ::c10::onnx::Loop) {
      ONNXIfShapeTypeInference(n);
    }
  }
}

} // namespace jit
} // namespace torch
