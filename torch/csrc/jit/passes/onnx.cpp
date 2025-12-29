#include <torch/csrc/jit/passes/onnx.h>

#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <torch/csrc/autograd/symbolic.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/onnx/constant_map.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/passes/onnx/onnx_log.h>
#include <torch/csrc/jit/passes/onnx/shape_type_inference.h>
#include <torch/csrc/jit/python/python_ir.h>
#include <sstream>

namespace torch::jit {

static void removePrintOps(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto b : it->blocks()) {
      removePrintOps(b);
    }
    if (it->kind() == prim::Print || it->kind() == aten::warn) {
      for (size_t i = 0; i < it->inputs().size();) {
        auto input = it->inputs().at(i);
        // only handling constants bc of potential side effects
        if (input->uses().size() == 1 &&
            input->node()->kind() == prim::Constant) {
          it->removeInput(i);
          input->node()->destroy();
        } else {
          ++i;
        }
      }
      it.destroyCurrent();
    }
  }
}

void RemovePrintOps(std::shared_ptr<Graph>& graph) {
  removePrintOps(graph->block());
  GRAPH_DUMP("After RemovePrintOps: ", graph);
}

static void checkONNXCompatibility(const c10::FunctionSchema& schema) {
  // in ONNX, all inputs are tensors, no support for tensor list
  // so at most one input tensor list is supported
  bool has_tensor_list = false;
  const auto& args = schema.arguments();
  for (const auto& arg : args) {
    if (arg.name() == "_caffe2_preallocated_outputs") {
      continue;
    }
    auto type = arg.type();
    if (type->kind() == TypeKind::OptionalType) {
      type = reinterpret_cast<OptionalType*>(type.get())->getElementType();
      // recursive optional type is not supported
      TORCH_INTERNAL_ASSERT(type->kind() != TypeKind::OptionalType);
    }
    if (type->kind() == TypeKind::ListType) {
      const auto& elem_type =
          reinterpret_cast<ListType*>(type.get())->getElementType();
      if (elem_type->isSubtypeOf(*TensorType::get())) {
        TORCH_INTERNAL_ASSERT(
            !has_tensor_list,
            "ONNX export supports at most one TensorList as input.");
        has_tensor_list = true;
      }
    }
  }
}

static void preprocessCaffe2Ops(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto b : it->blocks()) {
      preprocessCaffe2Ops(b);
    }
    if (it->kind().is_caffe2()) {
      const auto& schema = it->schema();
      checkONNXCompatibility(schema);
      std::vector<Value*> origin_inputs;
      for (Value* v : it->inputs()) {
        origin_inputs.push_back(v);
      }
      it->removeAllInputs();
      const auto& args = schema.arguments();
      size_t origin_inputs_index = 0;
      for (const auto& arg : args) {
        const auto& type = arg.type();
        TORCH_INTERNAL_ASSERT(origin_inputs_index < origin_inputs.size());
        const auto& origin_input = origin_inputs[origin_inputs_index++];
        if (type->kind() == TypeKind::OptionalType &&
            origin_input->mustBeNone()) {
          continue;
        }
        if (type->isSubtypeOf(*TensorType::get())) {
          it->addInput(origin_input);
        } else if (
            type->kind() == TypeKind::BoolType ||
            type->kind() == TypeKind::IntType) {
          const auto* constant_node = origin_input->node();
          TORCH_INTERNAL_ASSERT(constant_node->kind() == prim::Constant);
          it->i_(Symbol::attr(arg.name()), constant_node->i(attr::value));
        } else if (type->kind() == TypeKind::FloatType) {
          const auto* constant_node = origin_input->node();
          TORCH_INTERNAL_ASSERT(constant_node->kind() == prim::Constant);
          it->f_(Symbol::attr(arg.name()), constant_node->f(attr::value));
        } else if (type->kind() == TypeKind::StringType) {
          const auto* constant_node = origin_input->node();
          TORCH_INTERNAL_ASSERT(constant_node->kind() == prim::Constant);
          it->s_(Symbol::attr(arg.name()), constant_node->s(attr::value));
        } else if (type->kind() == TypeKind::ListType) {
          const auto& list_node = origin_input->node();
          const auto& elem_type = type->castRaw<ListType>()->getElementType();
          TORCH_INTERNAL_ASSERT(
              list_node->kind() == prim::ListConstruct ||
              list_node->kind() == prim::Constant);
          if (elem_type->isSubtypeOf(*TensorType::get())) {
            TORCH_INTERNAL_ASSERT(list_node->kind(), prim::ListConstruct);
            const auto& tensor_list = origin_input->node()->inputs();
            for (const auto& t : tensor_list) {
              it->addInput(t);
            }
          } else if (elem_type->kind() == TypeKind::FloatType) {
            std::vector<double> values;
            if (list_node->kind() == prim::ListConstruct) {
              for (const auto* elem_input : list_node->inputs()) {
                const auto* constant_node = elem_input->node();
                TORCH_INTERNAL_ASSERT(constant_node->kind() == prim::Constant);
                values.push_back(constant_node->f(attr::value));
              }
            } else { // is a constant list
              values = list_node->fs(attr::value);
            }
            it->fs_(Symbol::attr(arg.name()), values);
          } else {
            throw std::runtime_error(
                "Unhandled scalar arg: " + arg.name() +
                ", type: " + c10::typeKindToString(elem_type->kind()));
          }
        } else {
          throw std::runtime_error(
              "Unsupported input type of arg " + arg.name() +
              " in Caffe2 operator: " + c10::typeKindToString(type->kind()));
        }
      }
    }
  }
  EliminateDeadCode(
      block, true, DCESideEffectPolicy::ALLOW_DELETING_NODES_WITH_SIDE_EFFECTS);
}

void PreprocessCaffe2Ops(std::shared_ptr<Graph>& graph) {
  preprocessCaffe2Ops(graph->block());
  GRAPH_DUMP("After PreprocessCaffe2Ops: ", graph);
}

// Transform PythonOps into Nodes that match ONNX semantics.
std::shared_ptr<Graph> ToONNX(
    std::shared_ptr<Graph>& graph,
    ::torch::onnx::OperatorExportTypes operator_export_type) {
  ConstantValueMap::ClearMaps();
  auto new_graph = std::make_shared<Graph>(graph->current_scope());
  py::dict env;
  // Kept identical to values in env. Used for constant-time existence check.
  py::set values_in_env;
  try {
    BlockToONNX(
        graph->block(),
        new_graph->block(),
        operator_export_type,
        env,
        values_in_env);
  } catch (std::runtime_error&) {
    ONNX_LOG(
        "ONNX graph being constructed during exception:\n",
        new_graph->toString());
    throw;
  }
  GRAPH_DUMP("after ToONNX: ", new_graph);
  ConstantValueMap::ClearMaps();
  return new_graph;
}

// BlockToONNX.
// is_sub_block = true means the old_block (aten graph) is in the sub block
// (e.g., if sub block), and we want to convert it into its parent block in onnx
// graph. In this case, we don't register the input/output or eliminate the dead
// code.
py::dict BlockToONNX(
    Block* old_block,
    Block* new_block,
    ::torch::onnx::OperatorExportTypes operator_export_type,
    py::dict& env,
    py::set& values_in_env,
    bool is_sub_block) {
  torch::autograd::SymbolicContext ctx{};
  ctx.block = new_block;

  GRAPH_DEBUG(
      "BlockToONNX: graph of old block: ",
      old_block->owningGraph()->toString());

  // Initialize context and environment
  if (!is_sub_block) {
    for (auto input : old_block->inputs()) {
      auto n = ctx.block->addInput()->copyMetadata(input);
      auto py_n = py::cast(n);
      env[py::cast(input)] = py_n;
      values_in_env.add(py_n);
    }
  }

  // Determine if all inputs are static. This is used for each node to
  // determine whether or not to propagate shapes.
  if (!is_sub_block) {
    bool static_input_shape = AllGraphInputsStatic(ctx.block->owningGraph());
    ConstantValueMap::SetAllGraphInputsStatic(static_input_shape);
  }

  // Finally, visit all nodes in the graph
  for (auto node : old_block->nodes()) {
    NodeToONNX(node, ctx.block, operator_export_type, env, values_in_env);
  }

  if (is_sub_block) {
    return env;
  }

  for (auto output : old_block->outputs()) {
    auto py_value = env[py::cast(output)];
    Value* value = py_value.cast<Value*>();
    ctx.block->registerOutput(value);
  }
  // Run dce to clean-up unused functional and inplace ops.
  EliminateDeadCode(
      ctx.block,
      true,
      DCESideEffectPolicy::ALLOW_DELETING_NODES_WITH_SIDE_EFFECTS);

  return py::dict();
}

static bool ConstantFoldCondition(torch::jit::Value* output) {
  auto fold_condition = output->node()->kind() != c10::onnx::Constant &&
      ConstantValueMap::HasValue(output->debugName());
  auto reliable_value =
      ConstantValueMap::GetTypeReliable(output->debugName()).value_or(false);
  return fold_condition && reliable_value;
}

void NodeToONNX(
    Node* old_node,
    Block* new_block,
    ::torch::onnx::OperatorExportTypes operator_export_type,
    py::dict& env,
    py::set& values_in_env) {
  py::object onnx_utils =
      py::module::import("torch.onnx._internal.torchscript_exporter.utils");
  py::object onnx_globals =
      py::module::import("torch.onnx._internal.torchscript_exporter._globals");
  py::object onnx_registration = py::module::import(
      "torch.onnx._internal.torchscript_exporter.registration");

  // Setup all the lambda helper functions.

  // Returns a node that n maps to in the new graph
  auto envFn = [&env](Value* n) -> Value* {
    auto py_n = py::cast(n);
    TORCH_CHECK(env.contains(py_n), "Dangling node reference");
    auto py_value = env[py_n];
    TORCH_CHECK(!py_value.is_none(), "Unused node was subsequently used");
    Value* value = py_value.cast<Value*>();
    return value;
  };

  // Put the new outputs in our environment map, and copy the type from the
  // input graph if they were not set by the symbolic. This is called only
  // with results of symbolic call (not for nodes that are just cloned).
  auto setOutputs = [&](const std::string& op_name,
                        Node* node,
                        const value_list& outputs) {
    auto old_outputs = node->outputs();
    // Count all outputs, excluding Handles
    auto num_old_outputs = old_outputs.size();
    if (outputs.size() != num_old_outputs) {
      std::ostringstream ss;
      ss << "symbolic for " << op_name
         << " produced an incorrect number of outputs (expected ";
      ss << num_old_outputs << ", but got " << outputs.size() << ')';
      throw std::runtime_error(ss.str());
    }
    // For const node, it does not need params_dict info, so set it to {}.
    const ParamMap empty_params_dict = {};
    auto opset_version = py::cast<int>(
        onnx_globals.attr("GLOBALS").attr("export_onnx_opset_version"));
    for (const auto i : c10::irange(num_old_outputs)) {
      auto old = old_outputs[i];
      if (outputs[i]) {
        bool exist_in_env = values_in_env.contains(py::cast(outputs[i]));
        // Update ONNX value debug name with ATen value debug name if existed.
        // Skip if ONNX value already exist in environment.
        // This implies the op is a noop, and the value is owned by
        // other node created elsewhere.
        if (old->hasDebugName() && !exist_in_env) {
          auto old_name = outputs[i]->debugName();
          auto new_name = old->debugNameBase();
          Value* found_value = nullptr;
          bool exists = false;
          // In this scope, we fetch debug_names as a const reference and then
          // construct an iterator exist_name based on it. This iterator will
          // be corrupted if the underlying map of debug_names changes. This
          // will happen as a side-effect of setDebugName. For these reasons,
          // we make an explicit scope for exist_name and make sure that
          // setDebugName is never called with this scope.
          {
            const auto& debug_names = new_block->owningGraph()->debugNames();
            auto exist_name = debug_names.find(new_name);
            exists = exist_name != debug_names.end();
            if (exists) {
              found_value = exist_name->second;
            }
          }
          outputs[i]->setDebugName(new_name);
          if (exists) {
            found_value->setDebugName(new_name);
          }
          ConstantValueMap::UpdateValueName(old_name, outputs[i]->debugName());
        }
        // Allow symbolic() to skip specifying the type of the return node.
        // Unfortunately, they are on the hook for all internal nodes
        // (though in practice, the types are not computed.)
        //
        // If onnx shape inference is turned on, the new outputs will have
        // types inferred, and they will be merged with the old types.
        if (ConstantFoldCondition(outputs[i])) {
          // Create a const node if the node output value is in
          // ConstantValueMap.
          auto value =
              ConstantValueMap::GetValue(outputs[i]->debugName()).value();
          Node* const_node =
              new_block->owningGraph()->create(c10::onnx::Constant);
          const_node->t_(attr::value, value);
          const_node->output()->setType(TensorType::create(value));

          // Copy over source location and scope information to all nodes
          // created by the symbolic
          const_node->copyMetadata(node);
          new_block->appendNode(const_node);
          ONNXShapeTypeInference(const_node, empty_params_dict, opset_version);
          auto py_output = py::cast(const_node->output());
          env[py::cast(old)] = py_output;
          values_in_env.add(py_output);
        } else {
          // An update in ConstantValueMap is also needed here, since
          // the user setType can be only accessed in this step, and it
          // should be reliable.
          MergeInferredTypeAndSetMap(
              outputs[i], old->type(), outputs[i]->type());
          // non ONNX node with no type given will throw out the warnings here.
          UpdateReliable(
              outputs[i],
              AreInputsReliableOrStatic(outputs[i]->node()),
              /*no_type_warning=*/true);
          // For the node type that does not have ComputeConstant logic, it may
          // have reliable shape but its shape is not in ConstantValueMap. So we
          // need to update ConstantValueMap.
          UpdateShapeConstantIfReliable(outputs[i]);

          // Copy over source location and scope information to all nodes
          // created by the symbolic
          // Do not set metadata if outputs[i] is already in env.
          if (!exist_in_env) {
            outputs[i]->node()->copyMetadata(node);
          }
          auto py_output = py::cast(outputs[i]);
          env[py::cast(old)] = py_output;
          values_in_env.add(py_output);
        }
      } else {
        // Null output means that the ONNX op doesn't have outputs corresponding
        // to certain PyTorch outputs
        env[py::cast(old)] = py::none();
        if (!old->uses().empty()) {
          std::ostringstream ss;
          ss << "symbolic for " << op_name << " returned None for the output "
             << i;
          ss << " (indicating conversion for that particular output is not supported), ";
          ss << "but the network uses this output later";
          // TODO: Say what actually used it
          throw std::runtime_error(ss.str());
        }
      }
    }
  };

  // Clone the node and add it to the new graph
  auto cloneNode = [&](Node* node) {
    auto n_ = new_block->appendNode(
        new_block->owningGraph()->createClone(node, envFn));
    for (const auto i : c10::irange(node->outputs().size())) {
      // n_->outputs()[i]->setType(node->outputs()[i]->type());
      auto py_output = py::cast(n_->output(i));
      env[py::cast(node->output(i))] = py_output;
      values_in_env.add(py_output);
    }
  };

  // Inline the prim::PythonOp sub-block nodes and append them to the onnx graph
  auto inlineAutograd = [&](Node* PythonOpNode) {
    for (auto subblock : PythonOpNode->blocks()) {
      for (const auto i : c10::irange(PythonOpNode->inputs().size())) {
        auto py_value = env[py::cast(PythonOpNode->inputs()[i])];
        env[py::cast(subblock->inputs()[i])] = py_value;
        values_in_env.add(py_value);
      }
      for (auto* node : subblock->nodes()) {
        NodeToONNX(node, new_block, operator_export_type, env, values_in_env);
      }
      for (const auto i : c10::irange(PythonOpNode->outputs().size())) {
        auto py_value = env[py::cast(subblock->outputs()[i])];
        env[py::cast(PythonOpNode->outputs()[i])] = py_value;
        values_in_env.add(py_value);
      }
    }
  };

  // Cast output of symbolic() python implementation
  auto processSymbolicOutput = [&](const std::string& op_name,
                                   Node* n,
                                   const py::object& raw_output) {
    if (raw_output.ptr() == Py_None) {
      cloneNode(n);
      return;
    }
    // Cast the outputs back to C++ and put them in the new graph
    std::vector<Value*> outputs;
    try {
      if (py::isinstance<Value>(raw_output)) {
        outputs = value_list{py::cast<Value*>(raw_output)};
      } else {
        outputs = py::cast<std::vector<Value*>>(raw_output);
      }
    } catch (const std::exception&) {
      std::ostringstream ss;
      ss << "Error casting results of symbolic for " << op_name
         << ": expected to return list of op nodes, instead received type ''"
         << py::str(py::type::handle_of(raw_output))
         << "': " << py::str(raw_output);
      throw std::runtime_error(ss.str());
    }

    setOutputs(op_name, n, outputs);
  };

  auto callPySymbolicFunction = [&](Node* n) {
    // The idea is delegate as much of the actual argument massaging to
    // Python as possible

    py::tuple py_inputs(n->inputs().size());
    Py_ssize_t input_nr = 0;
    for (auto* input : n->inputs()) {
      py_inputs[input_nr++] = py::cast(envFn(input));
    }

    Graph* g = new_block->owningGraph();

    WithInsertPoint insert_point_guard(new_block);
    WithCurrentScope scope_guard(*g, n->scope());

    // IMPORTANT: NEVER pass raw pointer of smart pointer managed objects to
    // Python. Check #87343 for details.
    py::list new_nodes = py::list();
    py::object raw_output = onnx_utils.attr("_run_symbolic_function")(
        g->shared_from_this(),
        new_block,
        n,
        py_inputs,
        env,
        values_in_env,
        new_nodes,
        operator_export_type);

    // Find new nodes that have been created by _run_symbolic_function and
    // propagate metadata
    for (py::handle py_node : new_nodes) {
      Node* node = py_node.cast<Node*>();
      node->copyMetadata(n);
    }

    // TODO: Assert it's an ATen identifier???
    // (Sometimes it's not...)
    processSymbolicOutput(n->kind().toUnqualString(), n, raw_output);
    GRAPH_DUMP("after processSymbolicOutput: ", g);
  };

  auto callPySymbolicMethod = [&](ConcretePythonOp* op) {
    // Test if there is a symbolic function; bail if there is not
    auto pyobj = py::handle(op->pyobj.get());
    auto func = op->autogradFunction();
    if (func) {
      pyobj = func->get();
    }

    py::object opset_version =
        onnx_globals.attr("GLOBALS").attr("export_onnx_opset_version");
    // NOTE(justinchuby): Call the internal registry to register the symbolic
    // method defined in the module.
    bool is_registered_op =
        onnx_registration.attr("registry")
            .attr("is_registered_op")("prim::PythonOp", opset_version)
            .cast<bool>();
    py::bool_ is_autograd_inlining_enabled =
        py::cast<bool>(onnx_globals.attr("GLOBALS").attr("autograd_inlining"));
    if (!py::hasattr(pyobj, "symbolic") && !is_registered_op) {
      // Inline the subgraph within the prim::PythonOp unless
      // either of these conditions are satisfied
      // 1. The torch.autograd.Function class of this node object has `symbolic`
      // method defined.
      // 2. Custom export symbolic is registered for prim::PythonOp.
      if ((operator_export_type == ::torch::onnx::OperatorExportTypes::ONNX ||
           operator_export_type ==
               ::torch::onnx::OperatorExportTypes::ONNX_ATEN_FALLBACK) &&
          (py::cast<bool>(is_autograd_inlining_enabled))) {
        try {
          inlineAutograd(op);
        } catch (const std::exception& ex) {
          TORCH_WARN(
              "Unable to inline PythonOp: ",
              op->name(),
              " due to the following exception\n",
              ex.what(),
              "prim::PythonOp will be exported as is and without being inlined\n",
              "Try exporting with the following alternatives: \n",
              "1) Set operator_export_type to ONNX_FALLTHROUGH mode\n",
              "2) Register a symbolic method for the prim::PythonOp ",
              op->name());
          cloneNode(op);
        }
      } else {
        cloneNode(op);
      }
      return;
    }

    // Prepare args for Python. First one is the graph, and is followed
    // by regular args, with Variables replaced by corresponding nodes.
    Py_ssize_t input_nr = 0;
    py::tuple py_symbolic_args(op->cconv.size());
    auto inputs = op->inputs();
    auto node_it = inputs.begin();
    auto scalar_it = op->scalar_args.begin();
    for (auto arg_type : op->cconv) {
      py::object obj;
      if (arg_type == 'c') {
        TORCH_CHECK(
            scalar_it != op->scalar_args.end(),
            "expected too many scalar args");
        obj = py::reinterpret_borrow<py::object>(
            py::handle((scalar_it++)->get()));
      } else if (arg_type == 'd') {
        TORCH_CHECK(node_it != inputs.end(), "expected too many inputs");
        obj = py::cast(envFn(*node_it++));
      } else {
        throw std::runtime_error("unexpected calling convention");
      }
      py_symbolic_args[input_nr++] = obj;
    }

    WithInsertPoint insert_point_guard(new_block);
    WithCurrentScope scope_guard(*new_block->owningGraph(), op->scope());

    if (py::hasattr(pyobj, "symbolic")) {
      // Call the symbolic function
      // Use a little trampoline function so we can give good error messages
      // upon argument mismatch
      // Register as a custom operator
      // TODO: Find a more elegant way to do this without having to touch
      // internal Python modules.
      // TODO(justinchuby): Define a namespace for these Python Ops.
      onnx_registration.attr("registry")
          .attr("register")(
              "::" + op->name(),
              opset_version,
              pyobj.attr("symbolic"),
              /* custom */ true);

      // IMPORTANT: NEVER pass raw pointer of smart pointer managed objects to
      // Python. Check #87343 for details.
      py::object raw_output = onnx_utils.attr("_run_symbolic_method")(
          new_block->owningGraph()->shared_from_this(),
          op->name(),
          pyobj.attr("symbolic"),
          py_symbolic_args);

      processSymbolicOutput(op->name(), op, raw_output);
    } else {
      TORCH_INTERNAL_ASSERT(is_registered_op);
      Node* n = static_cast<Node*>(op);
      n->s_(attr::name, op->name());
      // Call symbolic function
      // IMPORTANT: NEVER pass raw pointer of smart pointer managed objects to
      // Python. Check #87343 for details.
      py::list new_nodes = py::list();
      py::object raw_output = onnx_utils.attr("_run_symbolic_function")(
          new_block->owningGraph()->shared_from_this(),
          new_block,
          n,
          py_symbolic_args,
          env,
          values_in_env,
          new_nodes,
          operator_export_type);

      processSymbolicOutput(op->kind().toUnqualString(), n, raw_output);
    }
  };

  auto k = old_node->kind();
  if (k.is_caffe2()) {
    // Pass on Caffe2 operator, since we already preprocess it
    cloneNode(old_node);
  } else if (k == prim::PythonOp) {
    callPySymbolicMethod(static_cast<ConcretePythonOp*>(old_node));
  } else {
    callPySymbolicFunction(old_node);
  }
}

} // namespace torch::jit
