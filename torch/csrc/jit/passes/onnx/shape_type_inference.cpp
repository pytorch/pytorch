#include <torch/csrc/jit/passes/onnx/shape_type_inference.h>

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/onnx/fold_if_node.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/passes/onnx/scalar_type_analysis.h>
#include <torch/csrc/jit/python/python_arg_flatten.h>
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
    SymbolDimMap& symbol_map) {
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
    SymbolDimMap& symbol_map) {
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
      b->inputs().at(i)->copyMetadata(n->inputs().at(i));
    }
  }
}

void ONNXShapeTypeInference(
    Block* b,
    const ParamMap& params_dict,
    int opset_version) {
  FetchBlockInputMetadataFromParent(b);
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

  ScalarTypeAnalysisForONNX(n_graph);

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
      const char shape_err[] = "ShapeInferenceError";
      const char type_err[] = "TypeInferenceError";
      if ((strstr(ex.what(), shape_err) == NULL) &&
          (strstr(ex.what(), type_err) == NULL))
        throw;
    }
    GRAPH_DEBUG(
        "ONNX graph after shape inference: ", prettyPrint(*model_proto));
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

      auto shape_ref = input_tensor_type->symbolic_sizes().sizes();
      TORCH_CHECK(
          shape_ref.has_value(), "Input tensor shape should have value.");
      auto shape = shape_ref.value();

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

void ONNXAssignOutputShape(
    std::shared_ptr<Graph>& graph,
    at::ArrayRef<at::Tensor> outputs,
    const python::IODescriptor& desc,
    bool onnx_shape_inference) {
  size_t outputs_index = 0;

  PyObject* py_obj = unflatten(outputs, desc);
  TORCH_INTERNAL_ASSERT(PyTuple_Check(py_obj));

  for (size_t i = 0; i < PyTuple_GET_SIZE(py_obj); ++i) {
    PyObject* elem = PyTuple_GET_ITEM(py_obj, i);

    if (PyList_Check(elem)) {
      size_t list_len = PyList_GET_SIZE(elem);
      if (HasSequenceTypeOutput(graph->outputs().at(outputs_index)->node())) {
        if (list_len > 0) {
          auto& var =
              reinterpret_cast<THPVariable*>(PyList_GET_ITEM(elem, 0))->cdata;
          for (size_t j = 1; j < list_len; ++j) {
            PyObject* list_elem = PyList_GET_ITEM(elem, j);
            TORCH_INTERNAL_ASSERT(THPVariable_Check(list_elem));
            auto& new_var = reinterpret_cast<THPVariable*>(list_elem)->cdata;
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
          outputs_index++;
          TORCH_INTERNAL_ASSERT(
              outputs_index <= graph->outputs().size(),
              "Incorrect number of elements provided as example outputs.");
        }
      } else { // When torch output is a list type, but ONNX node is not a
               // sequence type. Like prim::ListConstruct
        size_t list_len = PyList_GET_SIZE(elem);
        if (list_len > 0) {
          for (size_t j = 0; j < list_len; ++j) {
            PyObject* list_elem = PyList_GET_ITEM(elem, j);
            TORCH_INTERNAL_ASSERT(THPVariable_Check(list_elem));
            auto& var = reinterpret_cast<THPVariable*>(list_elem)->cdata;
            graph->outputs()
                .at(outputs_index + j)
                ->setType(MergeInferredType(
                    graph->outputs().at(outputs_index + j)->type(),
                    TensorType::create(var)));
          }
          outputs_index += list_len;
          TORCH_INTERNAL_ASSERT(
              outputs_index <= graph->outputs().size(),
              "Incorrect number of elements provided as example outputs.");
        }
      }
    } else if (PyTuple_Check(elem)) {
      size_t tuple_len = PyTuple_GET_SIZE(elem);
      if (tuple_len > 0) {
        for (size_t j = 0; j < tuple_len; ++j) {
          PyObject* tuple_elem = PyTuple_GET_ITEM(elem, j);
          TORCH_INTERNAL_ASSERT(THPVariable_Check(tuple_elem));
          auto& var = reinterpret_cast<THPVariable*>(tuple_elem)->cdata;
          graph->outputs()
              .at(outputs_index + j)
              ->setType(MergeInferredType(
                  graph->outputs().at(outputs_index + j)->type(),
                  TensorType::create(var)));
        }
        outputs_index += tuple_len;
        TORCH_INTERNAL_ASSERT(
            outputs_index <= graph->outputs().size(),
            "Incorrect number of elements provided as example outputs.");
      }
    } else if (THPVariable_Check(elem)) {
      at::Tensor var = reinterpret_cast<THPVariable*>(elem)->cdata;
      ONNXUpdateTypeFromTensor(
          graph->outputs().at(outputs_index), var, onnx_shape_inference);
      outputs_index++;
      TORCH_INTERNAL_ASSERT(
          outputs_index <= graph->outputs().size(),
          "Incorrect number of elements provided as example outputs.");
    } else { // Dict
      // Support for dict data type is limited to fixed size dictionaries in
      // ONNX.
      // Dictionary values are unrolled and keys are not preserved.
      TORCH_INTERNAL_ASSERT(PyDict_Check(elem));
      auto unrolled_dict = py::reinterpret_borrow<py::list>(PyDict_Items(elem));
      TORCH_INTERNAL_ASSERT(PyList_Check(unrolled_dict.ptr()));
      for (size_t j = 0; j < unrolled_dict.size(); ++j) {
        PyObject* tuple_elem = PyList_GET_ITEM(unrolled_dict.ptr(), j);
        TORCH_INTERNAL_ASSERT(PyTuple_Check(tuple_elem));
        TORCH_INTERNAL_ASSERT(PyTuple_GET_SIZE(tuple_elem) == 2);
        auto& var =
            reinterpret_cast<THPVariable*>(PyTuple_GET_ITEM(tuple_elem, 1))
                ->cdata;
        graph->outputs()
            .at(outputs_index + j)
            ->setType(MergeInferredType(
                graph->outputs().at(outputs_index + j)->type(),
                TensorType::create(var)));
      }
      outputs_index += unrolled_dict.size();
      TORCH_INTERNAL_ASSERT(
          outputs_index <= graph->outputs().size(),
          "Incorrect number of elements provided as example outputs.");
    }
  }

  TORCH_INTERNAL_ASSERT(
      outputs_index == graph->outputs().size(),
      "Incorrect number of elements provided as example outputs.");

  Py_DECREF(py_obj);
}

void ONNXShapeTypeInference(
    std::shared_ptr<Graph>& graph,
    const ParamMap& params_dict,
    int opset_version) {
  ONNXShapeTypeInference(graph->block(), params_dict, opset_version);
}

} // namespace jit
} // namespace torch
