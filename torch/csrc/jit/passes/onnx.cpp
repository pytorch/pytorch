#include "torch/csrc/utils/pybind.h"
#include "torch/csrc/jit/passes/onnx.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/symbolic.h"
#include "torch/csrc/utils/functional.h"
#include <unordered_map>
#include <sstream>

namespace torch { namespace jit {

namespace {

bool hasHandleOutput(Node *node) {
  JIT_ASSERT(node->hasMultipleOutputs());
  Node * last_output = node->outputs().back();
  return last_output->typeOption() && last_output->typeOption()->kind() == TypeKind::HandleType;
}

bool hasUsedHandle(Node *node) {
  if (!hasHandleOutput(node)) return false;
  return node->outputs().back()->uses().size() > 0;
}


} // anonymous namespace

// Transform PythonOps and Cpp Ops into Node's that match ONNX semantics.
void ToONNX(std::shared_ptr<tracer::TracingState>& state) {
  // Check that the tracing state is live (it should be, because
  // you were supposed to request zero derivatives.)
  if (state->is_expired()) {
    throw std::runtime_error("Tracing state is expired!  You should run the tracer with num_derivatives=0");
  }

  auto new_graph = std::make_shared<Graph>();
  std::unordered_map<void*, Node*> new_buffer_map;

  torch::autograd::SymbolicContext ctx;
  ctx.graph = new_graph.get();
  ctx.buffer_map = &new_buffer_map;
  std::unordered_map<Node*, Node*> env;

  // Returns a node that n maps to in the new graph
  auto envFn = [&env](Node * n) -> Node* {
    auto it = env.find(n);
    JIT_ASSERTM(it != env.end(), "Dangling node reference");
    JIT_ASSERTM(it->second, "Unused node was subsequently used");
    return it->second;
  };

  // Initialize context and environment
  for (auto input : state->graph->inputs()) {
    Node* n = ctx.graph->createClone(input, envFn);
    n->setStage(input->stage());
    ctx.graph->addInput(n);
    env[input] = n;
  }
  for (auto kv : state->buffer_map) {
    new_buffer_map[kv.first] = envFn(kv.second);
  }

  // Put the new outputs in our environment map, and copy the type from the
  // input graph if they were not set by the symbolic. This is called only
  // with results of symbolic call (not for nodes that are just cloned).
  auto setOutputs = [&](const std::string& op_name, Node * node, const node_list & outputs) {
    auto old_outputs = node->outputs();
    // Count all outputs, excluding Handles
    bool has_handle = hasHandleOutput(node);
    auto num_old_outputs = old_outputs.size() - (has_handle ? 1 : 0);
    if (outputs.size() != num_old_outputs) {
      std::ostringstream ss;
      ss << "symbolic for " << op_name << " produced an incorrect number of outputs (expected ";
      ss << num_old_outputs << ", but got " << outputs.size() << ")";
      throw std::runtime_error(ss.str());
    }
    for (std::size_t i = 0; i < num_old_outputs; ++i) {
      auto old = old_outputs[i];
      if (outputs[i]) {
        // Allow symbolic() to skip specifying the type of the return node.
        // Unfortunately, they are on the hook for all internal nodes
        // (though in practice, the types are not computed.)
        if (!outputs[i]->hasType()) {
          outputs[i]->setType(old->typeOption());
        }
        env[old] = outputs[i];
      } else {
        // Null output means that the ONNX op doesn't have outputs corresponding
        // to certain PyTorch outputs
        env[old] = nullptr;
        if (!old->uses().empty())
          throw std::runtime_error("ONNX conversion discarded a used output");
      }
    }
    if (has_handle) {
      JIT_ASSERT(old_outputs.back()->uses().empty());
      env[old_outputs.back()] = nullptr;
    }
  };

  // Clone the node (possibly including its Selects) and add it to the new graph
  auto cloneNode = [&](Node * node) {
    auto n_ = ctx.graph->createClone(node, envFn);
    env[node] = n_;
    ctx.graph->appendNode(n_);
    if (node->hasMultipleOutputs()) {
      for (auto s : node->uses()) {
        auto new_node = ctx.graph->createClone(s.user, envFn);
        ctx.graph->appendNode(new_node);
        env[s.user] = new_node;
      }
    }
  };

  auto callPySymbollic = [&](PythonOp* op) {
    // Prepare args for Python. First one is the graph, and is followed
    // by regular args, with Variables replaced by corresponding nodes.
    auto pyobj = py::handle(op->pyobj.get());
    Py_ssize_t input_nr = 0;
    py::tuple py_symbolic_args(1 + op->cconv.size());
    py_symbolic_args[input_nr++] = py::cast(ctx.graph);
    auto node_it = op->inputs().begin();
    auto scalar_it = op->scalar_args.begin();
    for (auto arg_type : op->cconv) {
      py::object obj;
      if (arg_type == 's') {
        JIT_ASSERTM(scalar_it != op->scalar_args.end(), "expected too many scalar args");
        obj = py::reinterpret_borrow<py::object>(py::handle((scalar_it++)->get()));
      } else if (arg_type == 't') {
        JIT_ASSERTM(node_it != op->inputs().end(), "expected too many inputs");
        obj = py::cast(envFn(*node_it++));
      } else {
        throw std::runtime_error("unexpected calling convention");
      }
      py_symbolic_args[input_nr++] = obj;
    }
    // Call the symbolic function
    py::object raw_output = py::reinterpret_steal<py::object>(
        PyObject_CallObject(pyobj.attr("symbolic").ptr(), py_symbolic_args.ptr()));
    if (!raw_output) throw py::error_already_set();
    if (raw_output.ptr() == Py_None)
      throw std::runtime_error("PythonOp's symbolic returned None, indicating conversion not supported " + op->name());

    // Cast the outputs back to C++ and put them in the new graph
    if (py::isinstance<Node>(raw_output)) {
      return node_list{py::cast<Node*>(raw_output)};
    } else {
      return py::cast<std::vector<Node*>>(raw_output);
    }
  };


  // Finally, visit all nodes in the graph
  for (auto node : state->graph->nodes()) {
    if (node->hasMultipleOutputs() && hasUsedHandle(node)) {
      // Nothing we can do here. The handle is used, so we'll need to capture the
      // original state and can't do anything with this op (we don't know what the
      // backward is).
      cloneNode(node);
      continue;
    }
    // Needed so that symbolic calls create nodes with correct stages.
    auto stage_guard = new_graph->setStageTemporary(node->stage());
    IR_IF(node, Select)
      // Selects are translated by multi-return nodes.
      JIT_ASSERT(env.count(value) > 0);
    IR_ELSEIFM(CppOp)
      if (auto fn = std::dynamic_pointer_cast<autograd::HasSymbolic>(value->fn)) {
        auto outputs = fn->symbolic(&ctx, fmap(node->inputs(), envFn));
        setOutputs(value->name(), node, outputs);
      } else {
        cloneNode(node);
      }
    IR_ELSEIFM(PythonOp)
      auto pyobj = py::handle(value->pyobj.get());
      if (py::hasattr(pyobj, "symbolic")) {
        auto outputs = callPySymbollic(value);
        setOutputs(value->name(), node, outputs);
      } else {
        cloneNode(node);
      }
    IR_ELSE()
      cloneNode(node);
    IR_END()
  }
  for (auto output : state->graph->outputs()) {
    ctx.graph->registerOutput(env.at(output));
  }

  // Copy stage from original graph
  new_graph->setStage(state->graph->stage());
  state->graph = std::move(new_graph);
  state->buffer_map = std::move(new_buffer_map);
}

}}
