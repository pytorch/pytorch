#include <torch/csrc/jit/passes/onnx.h>
#include <ATen/core/functional.h>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/symbolic.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/python_ir.h>
#include <torch/csrc/utils/pybind.h>
#include <sstream>
#include <unordered_map>

namespace torch {
namespace jit {

void removePrintOps(Block* block) {
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
}

void checkONNXCompatibility(const c10::FunctionSchema& schema) {
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
      AT_ASSERT(type->kind() != TypeKind::OptionalType);
    }
    if (type->kind() == TypeKind::ListType) {
      const auto& elem_type = reinterpret_cast<ListType*>(type.get())->getElementType();
      if (elem_type->isSubtypeOf(TensorType::get())) {
        AT_ASSERTM(
            !has_tensor_list,
            "ONNX export supports at most one TensorList as input.");
        has_tensor_list = true;
      }
    }
  }
}

void preprocessCaffe2Ops(Block* block) {
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
        auto type = arg.type();
        AT_ASSERT(origin_inputs_index < origin_inputs.size());
        const auto& origin_input = origin_inputs[origin_inputs_index++];
        if (type->kind() == TypeKind::OptionalType) {
          type = reinterpret_cast<OptionalType*>(type.get())->getElementType();
          if (origin_input->mustBeNone()) {
            continue;
          } else {
            // recursive optional type is not supported
            AT_ASSERT(type->kind() != TypeKind::OptionalType);
          }
        }
        if (type->isSubtypeOf(TensorType::get())) {
          it->addInput(origin_input);
        } else if (
            type->kind() == TypeKind::BoolType ||
            type->kind() == TypeKind::IntType) {
          const auto* constant_node = origin_input->node();
          AT_ASSERT(constant_node->kind() == prim::Constant);
          it->i_(Symbol::attr(arg.name()), constant_node->i(attr::value));
        } else if (type->kind() == TypeKind::FloatType) {
          const auto* constant_node = origin_input->node();
          AT_ASSERT(constant_node->kind() == prim::Constant);
          it->f_(Symbol::attr(arg.name()), constant_node->f(attr::value));
        } else if (type->kind() == TypeKind::StringType) {
          const auto* constant_node = origin_input->node();
          AT_ASSERT(constant_node->kind() == prim::Constant);
          it->s_(Symbol::attr(arg.name()), constant_node->s(attr::value));
        } else if (type->kind() == TypeKind::ListType) {
          const auto& list_node = origin_input->node();
          AT_ASSERT(list_node->kind() == prim::ListConstruct);
          const auto& elem_type = reinterpret_cast<ListType*>(type.get())->getElementType();
          if (elem_type->isSubtypeOf(TensorType::get())) {
            const auto& tensor_list = origin_input->node()->inputs();
            for (const auto& t : tensor_list) {
              it->addInput(t);
            }
          } else if (
              elem_type->kind() == TypeKind::IntType ||
              elem_type->kind() == TypeKind::BoolType) {
            // TODO support list of ints and bools, needs c10 op for testing
            throw std::runtime_error("List[int] and List[bool] are not supported yet.");
          } else if (elem_type->kind() == TypeKind::FloatType) {
            std::vector<double> values;
            for (const auto* elem_input : list_node->inputs()) {
              const auto* constant_node = elem_input->node();
              AT_ASSERT(constant_node->kind() == prim::Constant);
              values.push_back(constant_node->f(attr::value));
            }
            it->fs_(Symbol::attr(arg.name()), values);
          } else {
            throw std::runtime_error("Unhandled scalar arg: " + arg.name() +
                ", type: " + c10::typeKindToString(elem_type->kind()));
          }
        } else {
          throw std::runtime_error("Unsupported input type of arg " +
              arg.name() + " in Caffe2 operator: " +
              c10::typeKindToString(type->kind()));
        }
      }
    }
  }
  EliminateDeadCode(block, true, DCESideEffectPolicy::ALLOW_DELETING_NODES_WITH_SIDE_EFFECTS);
}

void PreprocessCaffe2Ops(std::shared_ptr<Graph>& graph) {
  preprocessCaffe2Ops(graph->block());
}

// Transform PythonOps into Nodes that match ONNX semantics.
std::shared_ptr<Graph> ToONNX(
    std::shared_ptr<Graph>& graph,
    ::torch::onnx::OperatorExportTypes operator_export_type) {
  auto new_graph = std::make_shared<Graph>(graph->current_scope());
  std::unordered_map<Value*, Value*> env;
  BlockToONNX(graph->block(), new_graph->block(), operator_export_type, env);
  return new_graph;
}

void BlockToONNX(
    Block* old_block,
    Block* new_block,
    ::torch::onnx::OperatorExportTypes operator_export_type,
    std::unordered_map<Value*, Value*> env) {
  torch::autograd::SymbolicContext ctx{};
  ctx.block = new_block;
  py::object onnx = py::module::import("torch.onnx");
  py::object onnx_symbolic = py::module::import("torch.onnx.symbolic_helper");
  py::object onnx_registry = py::module::import("torch.onnx.symbolic_registry");

  // Returns a node that n maps to in the new graph
  auto envFn = [&env](Value* n) -> Value* {
    auto it = env.find(n);
    TORCH_CHECK(it != env.end(), "Dangling node reference");
    TORCH_CHECK(it->second, "Unused node was subsequently used");
    return it->second;
  };

  // Initialize context and environment
  for (auto input : old_block->inputs()) {
    auto n = ctx.block->addInput()->copyMetadata(input);
    env[input] = n;
  }
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
      ss << num_old_outputs << ", but got " << outputs.size() << ")";
      throw std::runtime_error(ss.str());
    }
    for (size_t i = 0; i < num_old_outputs; ++i) {
      auto old = old_outputs[i];
      if (outputs[i]) {
        // Allow symbolic() to skip specifying the type of the return node.
        // Unfortunately, they are on the hook for all internal nodes
        // (though in practice, the types are not computed.)
        outputs[i]->setType(old->type());
        // Copy over source location and scope information to all nodes
        // created by the symbolic
        outputs[i]->node()->setSourceRange(node->sourceRange());
        outputs[i]->node()->setScope(node->scope());
        env[old] = outputs[i];
      } else {
        // Null output means that the ONNX op doesn't have outputs corresponding
        // to certain PyTorch outputs
        env[old] = nullptr;
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
    auto n_ = ctx.block->appendNode(
        ctx.block->owningGraph()->createClone(node, envFn));
    for (size_t i = 0; i < node->outputs().size(); i++) {
      // n_->outputs()[i]->setType(node->outputs()[i]->type());
      env[node->outputs()[i]] = n_->outputs()[i];
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
    } catch (const std::exception& ex) {
      std::ostringstream ss;
      ss << "Error casting results of symbolic for " << op_name
         << ": expected to return list of op nodes, instead received type ''"
         << py::str(raw_output.get_type()) << "': " << py::str(raw_output);
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

    WithInsertPoint insert_point_guard(ctx.block);
    WithCurrentScope scope_guard(*ctx.block->owningGraph(), n->scope());
    py::object raw_output = onnx.attr("_run_symbolic_function")(
        ctx.block->owningGraph(), n, py_inputs, env, operator_export_type);

    // TODO: Assert it's an ATen identifier???
    // (Sometimes it's not...)
    processSymbolicOutput(n->kind().toUnqualString(), n, raw_output);
  };

  auto callPySymbolicMethod = [&](ConcretePythonOp* op) {
    // Test if there is a symbolic function; bail if there is not
    auto pyobj = py::handle(op->pyobj.get());
    auto func = op->autogradFunction();
    if (func) {
      pyobj = func->get();
    }
    if (!py::hasattr(pyobj, "symbolic")) {
      cloneNode(op);
      return;
    }

    // Prepare args for Python. First one is the graph, and is followed
    // by regular args, with Variables replaced by corresponding nodes.
    Py_ssize_t input_nr = 0;
    py::tuple py_symbolic_args(1 + op->cconv.size());
    py_symbolic_args[input_nr++] = py::cast(ctx.block->owningGraph());
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

    WithInsertPoint insert_point_guard(ctx.block);
    WithCurrentScope scope_guard(*ctx.block->owningGraph(), op->scope());
    // Call the symbolic function
    // Use a little trampoline function so we can give good error messages
    // upon argument mismatch
    py::object opset_version = onnx_symbolic.attr("_export_onnx_opset_version");
    onnx_registry.attr("register_op")(op->name(), pyobj.attr("symbolic"), "", opset_version);
    py::object raw_output = onnx.attr("_run_symbolic_method")(
        op->name(), pyobj.attr("symbolic"), py_symbolic_args);

    processSymbolicOutput(op->name(), op, raw_output);
  };

  // Finally, visit all nodes in the graph
  for (auto node : old_block->nodes()) {
    if (node->kind().is_caffe2()) {
      // Pass on Caffe2 opeartor, since we already preprocess it
      cloneNode(node);
    } else if (node->kind() == prim::PythonOp) {
      callPySymbolicMethod(static_cast<ConcretePythonOp*>(node));
    } else {
      callPySymbolicFunction(node);
    }
  }
  for (auto output : old_block->outputs()) {
    ctx.block->registerOutput(env.at(output));
    env.at(output)->setType(output->type());
  }
  EliminateDeadCode(ctx.block, true, DCESideEffectPolicy::ALLOW_DELETING_NODES_WITH_SIDE_EFFECTS);
}

} // namespace jit
} // namespace torch
