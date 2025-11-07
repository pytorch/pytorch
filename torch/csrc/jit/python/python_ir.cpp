#include <torch/csrc/jit/python/python_ir.h>

#include <ATen/core/jit_type.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/api/include/torch/python.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/symbolic_shape_analysis.h>
#include <torch/csrc/jit/python/pybind.h>
#include <torch/csrc/jit/python/python_tracer.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/python_print.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_strings.h>
#include <iostream>
#include <sstream>
#include <utility>

namespace torch::jit {

// Controls whether graph source ranges are printed by default
static bool global_print_source_ranges = true;

Symbol ConcretePythonOp::Kind = prim::PythonOp;

using c10::Type;

static std::string getPythonName(const PyObject* obj_) {
  pybind11::gil_scoped_acquire gil;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  PyObject* obj = const_cast<PyObject*>(obj_);
  auto v = py::getattr(obj, "__name__", py::str("<python_value>"));
  // if this was a autograd.Function recover the name of the class
  return py::str(v);
}

static std::ostream& printPyObject(std::ostream& out, const THPObjectPtr& obj) {
  pybind11::gil_scoped_acquire gil;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  auto pyobj = py::handle(const_cast<PyObject*>(obj.get()));
  if (py::isinstance<py::tuple>(pyobj)) {
    // This special-case for printing tuples handles a problem where
    // str((2L, 3L)) outputs "(2L, 3L)" in Python 2 but "(2, 3)"
    // in Python 3.  In order to suppress the L-suffix, we must
    // manually print the string ourselves, calling str() on the
    // sub-elements.
    //
    // This is a fairly fragile fix (What if you have nested tuples
    // in tuples? What if you have dictionaries?) but it seems to hit
    // the cases that are triggered in practice in onnx-pytorch.  Revisit
    // this code if this is not the case.
    //
    // By the way, one non-solution for this problem is to monkeypatch
    // tuple.__str__; this doesn't work because Python doesn't allow
    // monkeypatching methods of built-in types.
    auto pytuple = pyobj.cast<py::tuple>();
    out << "(";
    size_t i = 0;
    for (const auto& o : pytuple) {
      if (i > 0) {
        out << ", ";
      }
      THPObjectPtr str(py::str(o).release().ptr());
      out << THPUtils_unpackString(str.get());
      i++;
    }
    if (i == 1) {
      out << ",";
    }
    out << ")";
    return out;
  } else {
    return out << THPUtils_unpackString(py::str(pyobj).ptr());
  }
}

static Node* findNode(
    c10::ArrayRef<torch::jit::Block*> blocks,
    Symbol kind,
    bool recurse = true) {
  for (Block* block : blocks) {
    for (Node* n : block->nodes()) {
      if (n->kind() == kind) {
        return n;
      }
      if (recurse) {
        auto node = findNode(n->blocks(), kind, recurse);
        if (node != nullptr) {
          return node;
        }
      }
    }
  }
  return nullptr;
}

static Node* findNode(Block* block, Symbol kind, bool recurse = true) {
  std::vector<Block*> blocks = {block};
  return findNode(blocks, kind, recurse);
}

std::string ConcretePythonOp::name() const {
  pybind11::gil_scoped_acquire gil;
  if (auto autograd = autogradFunction()) {
    return getPythonName(autograd->get());
  } else {
    return getPythonName(pyobj.get());
  }
}

void ConcretePythonOp::cloneFrom(Node* other_) {
  // NOLINTNEXTLINE(bugprone-parent-virtual-call)
  Node::cloneFrom(other_);
  auto other = other_->cast<ConcretePythonOp>();
  this->cconv = other->cconv;
  Py_INCREF(other->pyobj.get());
  this->pyobj = THPObjectPtr(other->pyobj.get());
  for (auto& sa : other->scalar_args) {
    Py_INCREF(sa.get());
    this->scalar_args.emplace_back(sa.get());
  }
}

// recover the autograd.Function instance, if this PythonOp's function
// was originally SomeFunction.apply
// used in ONNX for discovering symbolics
std::optional<THPObjectPtr> ConcretePythonOp::autogradFunction() const {
  pybind11::gil_scoped_acquire gil;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  py::handle obj = const_cast<PyObject*>(pyobj.get());

  auto r = py::getattr(obj, "__self__", py::none());
  if (r.is_none())
    return std::nullopt;

  auto apply = py::getattr(r, "apply", py::none());
  if (apply.is_none())
    return std::nullopt;

  auto c = PyObject_RichCompareBool(apply.ptr(), obj.ptr(), Py_NE);
  if (PyErr_Occurred())
    throw py::error_already_set();
  if (c)
    return std::nullopt;

  return THPObjectPtr(r.release().ptr());
}

void ConcretePythonOp::writeScalars(std::ostream& out) const {
  out << "(";
  int i = 0;
  for (auto& scalar : scalar_args) {
    if (i++ > 0)
      out << ", ";
    printPyObject(out, scalar);
  }
  out << ")";
}

void ConcretePythonOp::lint_python() const {
  size_t n_scalars = 0, n_tensors = 0;
  for (auto c : cconv) {
    if (c == 'c') {
      n_scalars++;
    } else if (c == 'd') {
      n_tensors++;
    } else {
      AT_ASSERT(0);
    }
    AT_ASSERT(static_cast<bool>(pyobj));
  }
  AT_ASSERT(n_scalars == scalar_args.size());
  AT_ASSERT(n_tensors == inputs().size());
}

Node* Graph::createPythonOp(
    THPObjectPtr&& pyobj,
    const std::string& cconv,
    pyobj_list&& scalar_args) {
  ConcretePythonOp* op = new ConcretePythonOp(this);
  return op->init(std::move(pyobj), cconv, std::move(scalar_args));
}

void initPythonIRBindings(PyObject* module_) {
  auto m = py::handle(module_).cast<py::module>();

  py::class_<AliasDb, std::shared_ptr<AliasDb>>(m, "AliasDb")
      .def("dump", &AliasDb::dump)
      .def("to_graphviz_str", &AliasDb::toGraphviz)
      .def(
          "may_contain_alias",
          [&](AliasDb& db, Value* v1, Value* v2) {
            return db.mayContainAlias(v1, v2);
          })
      .def(
          "has_writers",
          [&](AliasDb& db, Value* v1) { return db.hasWriters(v1); })
      .def("__str__", &AliasDb::toString)
      .def(
          "move_after_topologically_valid",
          [](AliasDb& db, Node* n, Node* movePoint) {
            return db.moveAfterTopologicallyValid(n, movePoint);
          })
      .def(
          "move_before_topologically_valid",
          [](AliasDb& db, Node* n, Node* movePoint) {
            return db.moveBeforeTopologicallyValid(n, movePoint);
          });
#define GS(name) def(#name, &Graph ::name)
  py::class_<Graph, std::shared_ptr<Graph>>(m, "Graph")
      .def(py::init<>())
      .def(
          "__repr__",
          [&](Graph& g) { return g.toString(global_print_source_ranges); })
      .def("str", &Graph::toString, py::arg("print_source_ranges") = true)
      .def_readonly_static(
          "global_print_source_ranges", &global_print_source_ranges)
      .def_static(
          "set_global_print_source_ranges",
          [&](const bool enabled) { global_print_source_ranges = enabled; },
          py::arg("enabled") = true)
      .def(
          "alias_db",
          [](std::shared_ptr<Graph> g,
             bool isFrozen = false,
             bool descend_function_calls = false) {
            return std::make_shared<AliasDb>(
                std::move(g), isFrozen, descend_function_calls);
          },
          py::arg("isFrozen") = false,
          py::arg("descend_function_calls") = false)
      .def(
          "dump_alias_db",
          [](std::shared_ptr<Graph> g) {
            AliasDb db(std::move(g));
            db.dump();
          })
      .def(
          "_export_onnx",
          [](const std::shared_ptr<Graph>& g,
             const std::map<std::string, at::Tensor>& initializers,
             int64_t onnx_opset_version,
             const std::unordered_map<
                 std::string,
                 std::unordered_map<int64_t, std::string>>& dynamic_axes,
             bool defer_weight_export,
             ::torch::onnx::OperatorExportTypes operator_export_type,
             bool strip_doc_string,
             bool keep_initializers_as_inputs,
             const std::map<std::string, int>& custom_opsets,
             bool add_node_names,
             const std::string& onnx_file_path,
             const NodeAttrNameMap& node_attr_to_name) {
            std::string graph;
            auto
                [model_proto,
                 export_map,
                 symbol_map,
                 val_use_external_data_format,
                 onnx_node_names] =
                    export_onnx(
                        g,
                        initializers,
                        onnx_opset_version,
                        dynamic_axes,
                        defer_weight_export,
                        operator_export_type,
                        strip_doc_string,
                        keep_initializers_as_inputs,
                        custom_opsets,
                        add_node_names,
                        false,
                        onnx_file_path,
                        node_attr_to_name);
            std::unordered_map<std::string, py::bytes>
                python_serialized_export_map;
            for (auto& kv : export_map) {
              auto t = kv.second;
              size_t copy_bytes = t.element_size() * t.numel();
              // TODO: this is an unnecessary copy. In theory we can directly
              // return the map from identifier to Tensor, but we need some API
              // in Python to get raw `bytes` containing the raw tensor data.
              python_serialized_export_map[kv.first] =
                  py::bytes(static_cast<const char*>(t.data_ptr()), copy_bytes);
            }
            graph = serialize_model_proto_to_string(model_proto);
            return std::make_tuple(
                py::bytes(graph),
                python_serialized_export_map,
                val_use_external_data_format,
                onnx_node_names);
          },
          py::arg("initializers"),
          py::arg("onnx_opset_version") = 0,
          py::arg("dynamic_axes"),
          py::arg("defer_weight_export") = false,
          py::arg("operator_export_type") =
              ::torch::onnx::OperatorExportTypes::ONNX,
          py::arg("strip_doc_string") = true,
          py::arg("keep_initializers_as_inputs") = true,
          py::arg("custom_opsets"),
          py::arg("add_node_names") = true,
          py::arg("onnx_file_path") = std::string(),
          py::arg("node_attr_to_name") = NodeAttrNameMap())
      .def(
          "_pretty_print_onnx",
          [](const std::shared_ptr<Graph>& g,
             const std::map<std::string, at::Tensor>& initializers,
             int64_t onnx_opset_version,
             bool defer_weight_export,
             ::torch::onnx::OperatorExportTypes operator_export_type,
             bool google_printer,
             bool keep_initializers_as_inputs,
             const std::map<std::string, int>& custom_opsets,
             bool add_node_names) {
            return pretty_print_onnx(
                g,
                initializers,
                onnx_opset_version,
                defer_weight_export,
                operator_export_type,
                google_printer,
                keep_initializers_as_inputs,
                custom_opsets,
                add_node_names);
          },
          py::arg("initializers"),
          py::arg("onnx_opset_version") = 0,
          py::arg("defer_weight_export") = false,
          py::arg("operator_export_type") =
              ::torch::onnx::OperatorExportTypes::ONNX,
          py::arg("google_printer") = false,
          py::arg("keep_initializers_as_inputs") = true,
          py::arg("custom_opsets"),
          py::arg("add_node_names") = true)
      .def(
          "inputs",
          [](Graph& g) {
            return py::make_iterator(g.inputs().begin(), g.inputs().end());
          },
          py::keep_alive<0, 1>())
      .def(
          "outputs",
          [](Graph& g) {
            return py::make_iterator(g.outputs().begin(), g.outputs().end());
          },
          py::keep_alive<0, 1>())
      // We keep the graph alive while the iterator lives. Destroying
      // nodes might still be hazardous.
      .def(
          "nodes",
          [](Graph& g) {
            return py::make_iterator(g.nodes().begin(), g.nodes().end());
          },
          py::keep_alive<0, 1>())
      .def(
          "findNode",
          [](Graph& g, const std::string& kind, bool recurse) {
            return findNode(g.block(), Symbol::fromQualString(kind), recurse);
          },
          "Find Node",
          py::arg("kind"),
          py::arg("recurse") = true)
      .def(
          "findAllNodes",
          [](Graph& g, const std::string& kind, bool recurse) {
            return findAllNodes(g, Symbol::fromQualString(kind), recurse);
          },
          "Find all nodes",
          py::arg("kind"),
          py::arg("recurse") = true)
      .def(
          "addInput",
          [](Graph& g, const std::string& name) { return g.addInput(name); },
          "Add input to graph with optional name seed",
          py::arg("name") = "")
      .def("copy", [](Graph& g) { return g.copy(); })
      .GS(eraseInput)
      .GS(eraseOutput)
      .GS(registerOutput)
      .def(
          "permuteInputs",
          [](Graph& g, const std::vector<size_t>& new_inputs) {
            g.block()->permuteInputs(new_inputs);
          })
      .def(
          "create",
          [](Graph& g, const char* str) {
            return g.create(Symbol::fromQualString(str));
          })
      .def(
          "create",
          [](Graph& g, const char* str, size_t noutputs) {
            return g.create(Symbol::fromQualString(str), noutputs);
          })
      .def(
          "create",
          [](Graph& g, const char* str, const std::vector<Value*>& inputs) {
            TORCH_CHECK_VALUE(
                std::all_of(
                    inputs.begin(),
                    inputs.end(),
                    [](Value* v) { return (v != nullptr); }),
                "cannot pass None in inputs");
            return g.create(Symbol::fromQualString(str), inputs);
          })
      .def(
          "create",
          [](Graph& g,
             const char* str,
             const std::vector<Value*>& inputs,
             size_t noutputs) {
            TORCH_CHECK_VALUE(
                std::all_of(
                    inputs.begin(),
                    inputs.end(),
                    [](Value* v) { return (v != nullptr); }),
                "cannot pass None in inputs");
            return g.create(Symbol::fromQualString(str), inputs, noutputs);
          })
      .def("param_node", [](Graph& g) { return g.block()->param_node(); })
      .def("return_node", [](Graph& g) { return g.block()->return_node(); })
      .def(
          "createFusionGroup",
          [](Graph& g) { return g.createWithSubgraph(prim::FusionGroup); })
      .def(
          "createCudaFusionGroup",
          [](Graph& g) { return g.createWithSubgraph(prim::CudaFusionGroup); })
      .def(
          "createClone",
          [](Graph& g, Node* n, py::object fn) {
            return g.createClone(
                n, [&](Value* e) { return fn(e).cast<Value*>(); });
          })
      .GS(appendNode)
      .GS(prependNode)
      // NB: insert_point_guard defined over direct modification of insert point
      .def(
          "insert_point_guard",
          [](Graph& g, Node* n) {
            return py::module::import("torch.jit._ir_utils")
                .attr("insert_point_guard")(g, n);
          })
      .def(
          "insert_point_guard",
          [](Graph& g, Block* b) {
            return py::module::import("torch.jit._ir_utils")
                .attr("insert_point_guard")(g, b);
          })
      .GS(insertPoint)
      .def("setInsertPoint", [](Graph& g, Node* n) { g.setInsertPoint(n); })
      .def("setInsertPoint", [](Graph& g, Block* n) { g.setInsertPoint(n); })
      .def(
          "insertGraph",
          [](Graph& g, Graph& callee, const std::vector<Value*>& inputs) {
            return insertGraph(g, callee, inputs);
          })
      .def(
          "insertGraph",
          [](Graph& g,
             Graph& callee,
             const std::vector<Value*>& inputs,
             std::unordered_map<Value*, Value*> value_map) {
            return insertGraph(g, callee, inputs, value_map);
          })
      .def(
          "insert",
          [](Graph& g, Symbol opname, const std::vector<Value*>& args) {
            std::vector<NamedValue> args_named;
            args_named.reserve(args.size());
            for (Value* v : args) {
              args_named.emplace_back(v);
            }
            return g.insert(opname, args_named);
          })
      .def(
          "makeMultiOutputIntoTuple",
          [](Graph& g) {
            auto tup = g.createTuple(g.outputs());
            tup->insertBefore(g.return_node());
            for (int64_t i = static_cast<int64_t>(g.outputs().size()) - 1;
                 i >= 0;
                 i--) {
              g.eraseOutput(0);
            }
            g.registerOutput(tup->output());
          })
      .def(
          "insertConstant",
          [](Graph& g, const IValue& ival) { return g.insertConstant(ival); })
      .GS(lint)
      .def("block", [](Graph& g) { return g.block(); })
      .GS(insertNode);
#undef GS

#define VS(name) def(#name, &Value ::name)
  py::class_<Value, unwrapping_shared_ptr<Value>>(m, "Value")
      .def(
          "__repr__",
          [](Value& n) {
            std::stringstream ss;
            ss << n.debugName() << " defined in (" << *n.node() << ")";
            return ss.str();
          })
      .VS(type)
      .VS(setType)
      .def(
          "inferTypeFrom",
          py::overload_cast<const at::Tensor&>(&Value::inferTypeFrom))
      .def(
          "inferTypeFrom",
          py::overload_cast<const c10::intrusive_ptr<c10::ivalue::Object>&>(
              &Value::inferTypeFrom))
      // skip owningGraph because it returns a raw pointer to a otherwise
      // std::shared_ptr stored graph object, and would cause a double free
      .VS(unique)
      .VS(debugName)
      .VS(setDebugName)
      .VS(offset)
      .VS(uses)
      .VS(replaceAllUsesWith)
      .VS(replaceAllUsesAfterNodeWith)
      .def("node", [](Value& v) { return v.node(); })
      .def(
          "setTypeAs",
          [](Value* node, Value* other) {
            node->setType(other->type());
            return node;
          })
      .VS(copyMetadata)
      .VS(isCompleteTensor)
      .VS(requires_grad)
      .def(
          "requiresGrad",
          [](Value& n) {
            return n.type()->expectRef<TensorType>().requiresGrad();
          })
      .def("toIValue", [](Value& n) { return toIValue(&n); })
      .def("type", [](Value& v) { return v.type(); });
#undef VS

  py::class_<Block, unwrapping_shared_ptr<Block>>(m, "Block")
      .def(
          "nodes",
          [](Block& b) {
            return py::make_iterator(b.nodes().begin(), b.nodes().end());
          })
      .def(
          "findNode",
          [](Block& b, const std::string& kind, bool recurse) {
            return findNode(&b, Symbol::fromQualString(kind), recurse);
          },
          "Find Node",
          py::arg("kind"),
          py::arg("recurse") = true)
      .def(
          "findAllNodes",
          [](Block& b, const std::string& kind, bool recurse) {
            return findAllNodes(b, Symbol::fromQualString(kind), recurse);
          },
          "Find all nodes",
          py::arg("kind"),
          py::arg("recurse") = true)
      .def(
          "inputs",
          [](Block& b) {
            return py::make_iterator(b.inputs().begin(), b.inputs().end());
          })
      .def(
          "outputs",
          [](Block& b) {
            return py::make_iterator(b.outputs().begin(), b.outputs().end());
          })
      .def("returnNode", [](Block& b) { return b.return_node(); })
      .def("paramNode", [](Block& b) { return b.param_node(); })
      .def("owningNode", [](Block& b) { return b.owningNode(); })
      .def(
          "addNode",
          [](Block& b, const char* str, const std::vector<Value*>& inputs) {
            return addNodeToBlock(&b, Symbol::fromQualString(str), inputs);
          })
      .def("addInputToBlock", [](Block& b) { return addInputToBlock(&b); })
      .def("registerOutput", [](Block& b, Value* value) {
        return b.registerOutput(value);
      });

#define NS(name) def(#name, &Node ::name)
  py::class_<Node, unwrapping_shared_ptr<Node>>(m, "Node")
      .def(
          "__repr__",
          [](Node& n) {
            std::stringstream ss;
            ss << n;
            return ss.str();
          })
      .def("sourceRange", [](Node& n) { return n.sourceRange().str(); })
      .def("hasMultipleOutputs", [](Node& n) { return n.outputs().size() > 1; })
      .def("inputsSize", [](Node& n) { return n.inputs().size(); })
      .def("outputsSize", [](Node& n) { return n.outputs().size(); })
      .NS(kind)
      .def("prev", [](Node& n) { return n.prev(); })
      .def("matches", [](Node& n, const char* s) { return n.matches(s); })
      .def("owningBlock", [](Node& n) { return n.owningBlock(); })
      .def("inputsAt", [](Node& n, size_t i) { return n.inputs().at(i); })
      .def(
          "inputs",
          [](Node& n) {
            return py::make_iterator(n.inputs().begin(), n.inputs().end());
          })
      .def(
          "schema",
          [](Node& n) {
            std::stringstream ss;
            if (n.maybeSchema()) {
              ss << n.schema();
            } else {
              ss << "(no schema)";
            }
            return ss.str();
          })
      .def(
          "outputs",
          [](Node& n) {
            return py::make_iterator(n.outputs().begin(), n.outputs().end());
          })
      .def("outputsAt", [](Node& n, size_t i) { return n.outputs().at(i); })
      .def(
          "findNode",
          [](Node& n, const std::string& kind, bool recurse) {
            return findNode(n.blocks(), Symbol::fromQualString(kind), recurse);
          },
          "Find Node",
          py::arg("kind"),
          py::arg("recurse") = true)
      .def(
          "findAllNodes",
          [](Node& n, const std::string& kind, bool recurse) {
            return findAllNodes(
                n.blocks(), Symbol::fromQualString(kind), recurse);
          },
          "Find all nodes",
          py::arg("kind"),
          py::arg("recurse") = true)
      .def("input", [](Node& n) { return n.input(); })
      .def("output", [](Node& n) { return n.output(); })
      .def(
          "getModuleHierarchy",
          [](Node& n) { return torch::jit::utils::getNodesModuleHierarchy(n); })
      .def(
          "namedInput",
          [](Node& n, const std::string& unqualName) {
            return n.namedInput(unqualName);
          })
      .NS(addInput)
      .NS(copyMetadata)
      .NS(replaceInput)
      .NS(replaceInputWith)
      .NS(replaceAllUsesWith)
      .NS(insertBefore)
      .NS(insertAfter)
      .NS(isBefore)
      .NS(isAfter)
      .NS(moveAfter)
      .NS(moveBefore)
      .NS(removeInput)
      .NS(removeAllInputs)
      .NS(destroy)
      .NS(hasUses)
      .NS(eraseOutput)
      .NS(addOutput)
      .NS(scopeName)
      .NS(isNondeterministic)
      .def(
          "blocks",
          [](Node& n) {
            return py::make_iterator(n.blocks().begin(), n.blocks().end());
          })
      .NS(addBlock)
      .NS(mustBeNone)

#define AS(name) def(#name, &Node::name)
      // methods from Attributes
      .AS(copyAttributes)
      .AS(hasAttributes)
#undef AS
#define AS(name) def(#name, &Node::name##S)
      // The default method names take Symbol, but the string conversion for
      // Symbol you to qualify with attr::. This is not very user friendly
      // for attributes, so expose the string variants instead.
      .AS(hasAttribute)
      .AS(kindOf)
      .AS(removeAttribute)
      .AS(attributeNames)
#undef AS
#define CREATE_ACCESSOR(Kind, method)                                       \
  def(#method "_", [](Node& n, const char* name, Kind##Attr::ValueType v) { \
    return n.method##_(Symbol::attr(name), std::move(v));                   \
  }).def(#method, [](Node& n, const char* name) {                           \
    return n.method(Symbol::attr(name));                                    \
  })
      .CREATE_ACCESSOR(Float, f)
      .CREATE_ACCESSOR(Floats, fs)
      .CREATE_ACCESSOR(Complex, c)
      .CREATE_ACCESSOR(String, s)
      .CREATE_ACCESSOR(Strings, ss)
      .CREATE_ACCESSOR(Int, i)
      .CREATE_ACCESSOR(Ints, is)
      .CREATE_ACCESSOR(Graph, g)
      .CREATE_ACCESSOR(Graphs, gs)
      .CREATE_ACCESSOR(IValue, ival)
#undef CREATE_ACCESSOR
      // Tensor (t_) -- manually written to unwrap the variable into a tensor.
      .def(
          "t_",
          [](Node& n, const char* name, const torch::autograd::Variable& v) {
            AT_ASSERT(!v.requires_grad());
            return n.t_(Symbol::attr(name), v);
          })
      .def(
          "t",
          [](Node& n, const char* name) { return n.t(Symbol::attr(name)); })
      // Tensors (ts_) -- manually written to unwrap variables into tensors.
      .def(
          "ts_",
          [](Node& n,
             const char* name,
             const std::vector<torch::autograd::Variable>& vs) {
            std::vector<at::Tensor> tensors;
            tensors.reserve(vs.size());
            for (auto& variable : vs) {
              AT_ASSERT(!variable.requires_grad());
              tensors.push_back(variable);
            }
            return n.ts_(Symbol::attr(name), std::move(tensors));
          })
      .def(
          "ts",
          [](Node& n, const char* name) {
            auto tensors = n.ts(Symbol::attr(name));
            std::vector<torch::autograd::Variable> variables;
            variables.reserve(tensors.size());
            for (auto& tensor : tensors) {
              variables.emplace_back(std::move(tensor));
            }
            return variables;
          })
      .def(
          "z_",
          [](Node& n, const char* name, const at::Tensor& v) {
            return n.t_(
                Symbol::attr(name),
                v.view(std::vector<int64_t>{}).set_requires_grad(false));
          })
      .def(
          "z",
          [](Node& n, const char* name) { return n.t(Symbol::attr(name)); })
      .def(
          "ty_",
          [](Node& n, const char* name, const TypePtr& type) {
            return n.ty_(Symbol::attr(name), type);
          })
      .def(
          "ty",
          [](Node& n, const char* name) { return n.ty(Symbol::attr(name)); })
      .def(
          "tys_",
          [](Node& n, const char* name, const std::vector<TypePtr>& types) {
            return n.tys_(Symbol::attr(name), types);
          })
      .def(
          "tys",
          [](Node& n, const char* name) { return n.tys(Symbol::attr(name)); })
      .def(
          "zs_",
          [](Node& n, const char* name, TensorsAttr::ValueType v) {
            for (auto& i : v) {
              i = i.view(std::vector<int64_t>{}).set_requires_grad(false);
            }
            return n.ts_(Symbol::attr(name), std::move(v));
          })
      .def(
          "zs",
          [](Node& n, const char* name) { return n.ts(Symbol::attr(name)); })
      .def(
          "pyobj",
          [](Node& n) {
            return py::handle(n.expect<ConcretePythonOp>()->pyobj.get())
                .cast<py::object>();
          })
      .def("cconv", [](Node& n) { return n.expect<ConcretePythonOp>()->cconv; })
      .def(
          "pyname",
          [](Node& n) { return n.expect<ConcretePythonOp>()->name(); })
      .def("scalar_args", [](Node& n) {
        auto op = n.expect<ConcretePythonOp>();
        auto scalars = py::list();
        auto append = scalars.attr("append");
        for (auto& arg : op->scalar_args) {
          append(py::handle(arg.get()));
        }
        return scalars;
      });

  using ::c10::Type;
  py::class_<Type, TypePtr>(m, "Type")
      .def("__repr__", [](Type& t) { return t.annotation_str(); })
      .def(
          "str",
          [](Type& t) {
            std::ostringstream s;
            s << t;
            return s.str();
          })
      .def(
          "containedTypes",
          [](Type& self) { return self.containedTypes().vec(); })
      .def("kind", [](const Type& t) { return typeKindToString(t.kind()); })
      .def(
          "dim",
          [](Type& t) {
            auto vshape = t.expectRef<TensorType>().sizes();
            return vshape.size() ? py::cast(*vshape.size())
                                 : py::cast<py::none>(Py_None);
          })
      .def(
          "undefined",
          [](Type& t) {
            auto undef = t.expectRef<TensorType>().undefined();
            return undef.has_value() ? py::cast(*undef)
                                     : py::cast<py::none>(Py_None);
          })
      .def(
          "sizes",
          [](Type& t) -> py::object {
            if (auto ptt = t.expect<TensorType>()) {
              if (auto cs = ptt->sizes().concrete_sizes()) {
                return py::cast(*cs);
              }
            }
            return py::none();
          })
      .def(
          "symbolic_sizes",
          [](Type& t) -> py::object {
            if (auto ptt = t.expect<TensorType>()) {
              auto ss = ptt->symbolic_sizes();
              if (!ss.rank().has_value()) {
                return py::none();
              }

              std::vector<int64_t> ss_vals;
              for (size_t i = 0; i < *ss.rank(); ++i) {
                ss_vals.push_back(ss.at(i).value());
              }
              return py::cast(ss_vals);
            }
            return py::none();
          })
      .def(
          "with_sizes",
          [](Type& t, std::optional<std::vector<std::optional<int64_t>>> sizes)
              -> py::object {
            auto ptt = t.expect<TensorType>();
            if (!ptt) {
              return py::none();
            }
            if (!sizes) {
              return py::cast(ptt->withSymbolicShapes(c10::SymbolicShape()));
            }
            return py::cast(ptt->withSymbolicShapes(*sizes));
          })
      .def(
          "varyingSizes",
          [](Type& t) -> py::object {
            if (auto ptt = t.expect<TensorType>()) {
              if (auto s = ptt->sizes().sizes()) {
                return py::cast(s.value());
              }
            }
            return py::none();
          })
      .def(
          "strides",
          [](Type& t) -> py::object {
            if (auto ptt = t.expect<TensorType>()) {
              if (auto cs = ptt->strides().concrete_sizes()) {
                return py::cast(*cs);
              }
            }
            return py::none();
          })
      .def(
          "contiguous",
          [](Type& t) {
            return std::static_pointer_cast<Type>(
                t.expectRef<TensorType>().contiguous());
          })
      .def(
          "scalarType",
          [](Type& t) {
            auto scalar_type = t.expectRef<TensorType>().scalarType();
            return scalar_type ? toString(*scalar_type) : nullptr;
          })
      .def(
          "device",
          [](Type& t) -> py::object {
            auto device = t.expectRef<TensorType>().device();
            if (!device) {
              return py::none();
            }
            PyObject* thp_device = THPDevice_New(device.value());
            return py::reinterpret_borrow<py::object>(thp_device);
            // return toPyObject(device.value());
          })
      .def(
          "with_device",
          [](Type& t, py::object device) -> py::object {
            at::Device c_device =
                python::detail::py_object_to_device(std::move(device));
            if (auto ptt = t.expect<TensorType>()) {
              return py::cast(ptt->withDevice(c_device));
            }
            return py::none();
          })
      .def(
          "dtype",
          [](Type& t) -> py::object {
            auto scalar_type = t.expectRef<TensorType>().scalarType();
            if (!scalar_type) {
              return py::none();
            }
            THPDtype* thp_dtype = torch::getTHPDtype(*scalar_type);
            py::object dtype =
                py::reinterpret_borrow<py::object>((PyObject*)thp_dtype);
            return dtype;
          })
      .def(
          "with_dtype",
          [](Type& t, py::object dtype) -> py::object {
            at::ScalarType scalar_type =
                python::detail::py_object_to_dtype(std::move(dtype));

            if (auto ptt = t.expect<TensorType>()) {
              // auto scalar_type = dtype->scalar_type;
              return py::cast(ptt->withScalarType(scalar_type));
            }
            return py::none();
          })
      .def(
          "__eq__",
          [](const TypePtr& self, const TypePtr& other) {
            if (!other) {
              return false;
            }
            return *self == *other;
          })
      .def(
          "isSubtypeOf",
          [](const TypePtr& self, const TypePtr& other) {
            if (!other) {
              return false;
            }
            return self->isSubtypeOf(other);
          })
      .def(
          "is_interface_type",
          [](const TypePtr& self) {
            return self->castRaw<InterfaceType>() != nullptr;
          })
      .def(
          "requires_grad",
          [](const TypePtr& self) -> bool { return self->requires_grad(); })
      .def_property_readonly(
          "annotation_str", [](const std::shared_ptr<Type>& self) {
            return self->annotation_str();
          });

  py::class_<AnyType, Type, AnyTypePtr>(m, "AnyType")
      .def_static("get", &AnyType::get);
  py::class_<NumberType, Type, NumberTypePtr>(m, "NumberType")
      .def_static("get", &NumberType::get);
  py::class_<IntType, Type, IntTypePtr>(m, "IntType")
      .def_static("get", &IntType::get);
  py::class_<SymIntType, Type, SymIntTypePtr>(m, "SymIntType")
      .def_static("get", &SymIntType::get);
  py::class_<SymBoolType, Type, SymBoolTypePtr>(m, "SymBoolType")
      .def_static("get", &SymBoolType::get);
  py::class_<FloatType, Type, FloatTypePtr>(m, "FloatType")
      .def_static("get", &FloatType::get);
  py::class_<ComplexType, Type, ComplexTypePtr>(m, "ComplexType")
      .def_static("get", &ComplexType::get);
  py::class_<TensorType, Type, TensorTypePtr>(m, "TensorType")
      .def_static("get", &TensorType::get)
      .def_static("getInferred", &TensorType::getInferred)
      .def_static("create_from_tensor", [](const at::Tensor& t) {
        return TensorType::create(t);
      });
  py::class_<BoolType, Type, BoolTypePtr>(m, "BoolType")
      .def_static("get", &BoolType::get);
  py::class_<StringType, Type, StringTypePtr>(m, "StringType")
      .def_static("get", &StringType::get);
  py::class_<DeviceObjType, Type, DeviceObjTypePtr>(m, "DeviceObjType")
      .def_static("get", &DeviceObjType::get);
  // TODO(antoniojkim): Add GeneratorType to the public API once its been added
  //                    to the public documentation
  py::class_<GeneratorType, Type, GeneratorTypePtr>(m, "_GeneratorType")
      .def_static("get", &GeneratorType::get);
  py::class_<StreamObjType, Type, StreamObjTypePtr>(m, "StreamObjType")
      .def_static("get", &StreamObjType::get);
  py::class_<PyObjectType, Type, PyObjectTypePtr>(m, "PyObjectType")
      .def_static("get", &PyObjectType::get);
  py::class_<NoneType, Type, NoneTypePtr>(m, "NoneType")
      .def_static("get", &NoneType::get);

  py::class_<TupleType, Type, TupleTypePtr>(m, "TupleType")
      .def(py::init([](std::vector<TypePtr> a) {
        return TupleType::create(std::move(a));
      }))
      .def("elements", [](TupleType& self) {
        std::vector<TypePtr> types;
        for (const auto& type : self.elements()) {
          types.push_back(type);
        }
        return types;
      });
  py::class_<UnionType, Type, UnionTypePtr>(m, "UnionType")
      .def(py::init(
          [](const std::vector<TypePtr>& a) { return UnionType::create(a); }));
  py::class_<ListType, Type, ListTypePtr>(m, "ListType")
      .def(py::init([](const TypePtr& a) { return ListType::create(a); }))
      .def_static("ofInts", &ListType::ofInts)
      .def_static("ofTensors", &ListType::ofTensors)
      .def_static("ofFloats", &ListType::ofFloats)
      .def_static("ofComplexDoubles", &ListType::ofComplexDoubles)
      .def_static("ofBools", &ListType::ofBools)
      .def_static("ofStrings", &ListType::ofStrings)
      .def("getElementType", &ListType::getElementType);
  py::class_<DictType, Type, DictTypePtr>(m, "DictType")
      .def(py::init([](TypePtr key, TypePtr value) {
        return DictType::create(std::move(key), std::move(value));
      }))
      .def("getKeyType", &DictType::getKeyType)
      .def("getValueType", &DictType::getValueType);
  py::class_<OptionalType, Type, OptionalTypePtr>(m, "OptionalType")
      .def(py::init([](const TypePtr& a) { return OptionalType::create(a); }))
      .def_static("ofTensor", &OptionalType::ofTensor)
      .def("getElementType", &OptionalType::getElementType);
  py::class_<RRefType, Type, RRefTypePtr>(m, "RRefType")
      .def(py::init([](TypePtr a) { return RRefType::create(std::move(a)); }))
      .def("getElementType", &RRefType::getElementType);

  py::class_<FutureType, Type, FutureTypePtr>(m, "FutureType")
      .def(py::init([](TypePtr a) { return FutureType::create(std::move(a)); }))
      .def("getElementType", &FutureType::getElementType);

  py::class_<AwaitType, Type, AwaitTypePtr>(m, "AwaitType")
      .def(py::init([](TypePtr a) { return AwaitType::create(std::move(a)); }))
      .def("getElementType", &AwaitType::getElementType);

  py::class_<ClassType, Type, ClassTypePtr>(m, "ClassType")
      .def(py::init([](const std::string& qualified_name) {
        return get_python_cu()->get_class(c10::QualifiedName(qualified_name));
      }))
      .def("name", [](ClassType& self) { return self.name()->name(); })
      .def(
          "qualified_name",
          [](ClassType& self) { return self.name()->qualifiedName(); })
      .def("method_names", [](ClassType& self) {
        std::vector<std::string> method_names;
        for (const auto* method : self.methods()) {
          method_names.push_back(method->name());
        }
        return method_names;
      });
  py::class_<EnumType, Type, EnumTypePtr>(m, "EnumType")
      .def(py::init([](const std::string& qualified_name,
                       TypePtr value_type,
                       const std::vector<py::object>& enum_names_values) {
        std::vector<std::pair<std::string, IValue>> names_values;
        names_values.reserve(enum_names_values.size());
        for (const auto& enum_name_value : enum_names_values) {
          auto enum_name = py::cast<std::string>(enum_name_value.attr("name"));
          auto enum_value = toIValue(enum_name_value.attr("value"), value_type);
          names_values.emplace_back(enum_name, enum_value);
        }
        return EnumType::create(
            c10::QualifiedName(qualified_name),
            std::move(value_type),
            std::move(names_values),
            get_python_cu());
      }));
  py::class_<InterfaceType, Type, InterfaceTypePtr>(m, "InterfaceType")
      .def(py::init([](const std::string& qualified_name) {
        return get_python_cu()->get_interface(
            c10::QualifiedName(qualified_name));
      }))
      .def(
          "getMethod",
          [](InterfaceType& self, const std::string& name) {
            return self.getMethod(name);
          },
          py::return_value_policy::reference)
      .def("getMethodNames", [](InterfaceType& self) {
        std::vector<std::string> names;
        for (const FunctionSchema& fn : self.methods()) {
          names.emplace_back(fn.name());
        }
        return names;
      });
  using ::c10::InferredType;
  py::class_<InferredType, std::shared_ptr<InferredType>>(m, "InferredType")
      .def(py::init([](std::shared_ptr<Type> type) {
        return std::make_shared<InferredType>(std::move(type));
      }))
      .def(py::init([](std::string reason) {
        return std::make_shared<InferredType>(std::move(reason));
      }))
      .def(
          "type",
          [](const std::shared_ptr<InferredType>& self) {
            return self->type();
          })
      .def(
          "success",
          [](const std::shared_ptr<InferredType>& self) {
            return self->success();
          })
      .def("reason", [](const std::shared_ptr<InferredType>& self) {
        return self->reason();
      });

  py::class_<Use>(m, "Use")
      .def_readonly("user", &Use::user)
      .def_readonly("offset", &Use::offset)
      .def("isAfter", [](Use& self, Use& other_use) {
        return isBeforeOrAfter(self, other_use, false);
      });

  py::class_<torch::jit::ShapeComputeGraphMapping>(
      m, "_ShapeComputeGraphMapping")
      .def(
          "partial_eval_shape_graph",
          [](ShapeComputeGraphMapping& g) {
            return g.partial_eval_shape_graph;
          })
      .def(
          "graph_output_to_symbolic_shape_dim",
          [](ShapeComputeGraphMapping& g) {
            return g.graph_output_to_symbolic_shape_dim_;
          });
}
} // namespace torch::jit
