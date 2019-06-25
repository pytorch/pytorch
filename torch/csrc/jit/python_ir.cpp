#include <torch/csrc/jit/python_ir.h>

#include <torch/csrc/jit/argument_spec.h>
#include <torch/csrc/jit/export.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/passes/alias_analysis.h>
#include <torch/csrc/jit/passes/python_print.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/pybind.h>
#include <torch/csrc/jit/python_tracer.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/auto_gil.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_strings.h>

#include <iostream>
#include <sstream>

namespace torch {
namespace jit {

Symbol ConcretePythonOp::Kind = prim::PythonOp;

using c10::Type;

std::string getPythonName(const PyObject* obj_) {
  AutoGIL gil;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  PyObject* obj = const_cast<PyObject*>(obj_);
  auto v = py::getattr(obj, "__name__", py::str("<python_value>"));
  // if this was a autograd.Function recover the name of the class
  return py::str(v);
}

std::ostream& printPyObject(std::ostream& out, const THPObjectPtr& obj) {
  AutoGIL gil;
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

std::vector<Node*> findAllNodes(
    c10::ArrayRef<torch::jit::Block*> blocks,
    Symbol kind,
    bool recurse = true) {
  std::vector<Node*> ret;
  for (Block* block : blocks) {
    for (Node* n : block->nodes()) {
      if (n->kind() == kind) {
        ret.push_back(n);
      }
      if (recurse) {
        auto nodes = findAllNodes(n->blocks(), kind, recurse);
        ret.insert(ret.end(), nodes.begin(), nodes.end());
      }
    }
  }
  return ret;
}

std::vector<Node*> findAllNodes(
    Block* block,
    Symbol kind,
    bool recurse = true) {
  std::vector<Block*> blocks = {block};
  return findAllNodes(blocks, kind, recurse);
}

Node* findNode(
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

Node* findNode(Block* block, Symbol kind, bool recurse = true) {
  std::vector<Block*> blocks = {block};
  return findNode(blocks, kind, recurse);
}

std::string ConcretePythonOp::name() const {
  AutoGIL gil;
  if (auto autograd = autogradFunction()) {
    return getPythonName(autograd->get());
  } else {
    return getPythonName(pyobj.get());
  }
}

void ConcretePythonOp::cloneFrom(Node* other_) {
  Node::cloneFrom(other_);
  auto other = other_->cast<ConcretePythonOp>();
  this->cconv = other->cconv;
  Py_INCREF(other->pyobj.get());
  this->pyobj = THPObjectPtr(other->pyobj.get());
  this->ignore_on_export = other->ignore_on_export;
  for (auto& sa : other->scalar_args) {
    Py_INCREF(sa.get());
    this->scalar_args.emplace_back(sa.get());
  }
}

// recover the autograd.Function instance, if this PythonOp's function
// was originally SomeFunction.apply
// used in ONNX for discovering symbolics
c10::optional<THPObjectPtr> ConcretePythonOp::autogradFunction() const {
  AutoGIL gil;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  py::handle obj = const_cast<PyObject*>(pyobj.get());

  auto r = py::getattr(obj, "__self__", py::none());
  if (r.is_none())
    return c10::nullopt;

  auto apply = py::getattr(r, "apply", py::none());
  if (apply.is_none())
    return c10::nullopt;

  auto c = PyObject_RichCompareBool(apply.ptr(), obj.ptr(), Py_NE);
  if (PyErr_Occurred())
    throw py::error_already_set();
  if (c)
    return c10::nullopt;

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
#define GS(name) def(#name, &Graph ::name)
  py::class_<Graph, std::shared_ptr<Graph>>(m, "Graph")
      .def(py::init<>())
      .def(
          "__repr__",
          [](Graph& g) {
            return g.toString();
          })
      .def(
          "str",
          &Graph::toString,
          py::arg("print_source_ranges") = true)
      .def(
          "dump_alias_db",
          [](std::shared_ptr<Graph> g) {
            AliasDb db(g);
            db.dump();
          })
      .def(
          "_export_onnx",
          [](const std::shared_ptr<Graph> g,
             const std::map<std::string, at::Tensor>& initializers,
             int64_t onnx_opset_version,
             const std::unordered_map<std::string, std::unordered_map<int64_t, std::string>>& dynamic_axes,
             bool defer_weight_export,
             ::torch::onnx::OperatorExportTypes operator_export_type,
             bool strip_doc_string) {
            std::string graph;
            RawDataExportMap export_map;
            std::tie(graph, export_map) = export_onnx(
                g,
                initializers,
                onnx_opset_version,
                dynamic_axes,
                defer_weight_export,
                operator_export_type,
                strip_doc_string);
            std::unordered_map<std::string, py::bytes>
                python_serialized_export_map;
            for (auto& kv : export_map) {
              auto t = kv.second;
              size_t copy_bytes = t.element_size() * t.numel();
              // TODO: this is an unecessary copy. In theory we can directly
              // return the map from identifier to Tensor, but we need some API
              // in Python to get raw `bytes` containing the raw tensor data.
              python_serialized_export_map[kv.first] =
                  py::bytes(static_cast<const char*>(t.data_ptr()), copy_bytes);
            }
            return std::make_tuple(
                py::bytes(graph), python_serialized_export_map);
          },
          py::arg("initializers"),
          py::arg("onnx_opset_version") = 0,
          py::arg("dynamic_axes"),
          py::arg("defer_weight_export") = false,
          py::arg("operator_export_type") =
              ::torch::onnx::OperatorExportTypes::ONNX,
          py::arg("strip_doc_string") = true)
      .def(
          "_pretty_print_onnx",
          [](const std::shared_ptr<Graph> g,
             const std::map<std::string, at::Tensor>& initializers,
             int64_t onnx_opset_version,
             bool defer_weight_export,
             ::torch::onnx::OperatorExportTypes operator_export_type,
             bool google_printer) {
            return pretty_print_onnx(
                g,
                initializers,
                onnx_opset_version,
                defer_weight_export,
                operator_export_type,
                google_printer);
          },
          py::arg("initializers"),
          py::arg("onnx_opset_version") = 0,
          py::arg("defer_weight_export") = false,
          py::arg("operator_export_type") =
              ::torch::onnx::OperatorExportTypes::ONNX,
          py::arg("google_printer") = false)
      .def(
          "inputs",
          [](Graph& g) {
            return py::make_iterator(g.inputs().begin(), g.inputs().end());
          })
      .def(
          "outputs",
          [](Graph& g) {
            return py::make_iterator(g.outputs().begin(), g.outputs().end());
          })
      // TODO: Iterator invalidation might make this hazardous
      .def(
          "nodes",
          [](Graph& g) {
            return py::make_iterator(g.nodes().begin(), g.nodes().end());
          })
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
            return findAllNodes(
                g.block(), Symbol::fromQualString(kind), recurse);
          },
          "Find all nodes",
          py::arg("kind"),
          py::arg("recurse") = true)
      .def("addInput", [](Graph& g) { return g.addInput(); })
      .def("copy", [](Graph& g) { return g.copy(); })
      .GS(eraseInput)
      .GS(registerOutput)
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
            return g.create(Symbol::fromQualString(str), inputs);
          })
      .def(
          "create",
          [](Graph& g,
             const char* str,
             const std::vector<Value*>& inputs,
             size_t noutputs) {
            return g.create(Symbol::fromQualString(str), inputs, noutputs);
          })
      .def("param_node", [](Graph& g) { return g.block()->param_node(); })
      .def("return_node", [](Graph& g) { return g.block()->return_node(); })
      .def(
          "createFusionGroup",
          [](Graph& g) { return g.createWithSubgraph(prim::FusionGroup); })
      .def(
          "createClone",
          [](Graph& g, Node* n, py::object fn) {
            return g.createClone(
                n, [&](Value* e) { return fn(e).cast<Value*>(); });
          })
      .GS(appendNode)
      .GS(prependNode)
      .GS(lint)
      .GS(insertNode);
#undef GS

#define VS(name) def(#name, &Value ::name)
  py::class_<Value, std::unique_ptr<Value, py::nodelete>>(m, "Value")
      .def(
          "__repr__",
          [](Value& n) {
            std::stringstream ss;
            ss << n.debugName() << " defined in (" << *n.node() << ")";
            return ss.str();
          })
      .VS(type)
      .VS(setType)
      .VS(inferTypeFrom)
      // skip owningGraph because it returns a raw pointer to a otherwise
      // std::shared_ptr stored graph object, and would cause a double free
      .VS(unique)
      .VS(debugName)
      .VS(setDebugName)
      .VS(offset)
      .VS(uses)
      .VS(replaceAllUsesWith)
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
      .def("toIValue", [](Value& n) { return toIValue(&n); })
      .def("type", [](Value& v) { return v.type(); });
#undef VS

  py::class_<Block, std::unique_ptr<Block, py::nodelete>>(m, "Block")
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
            return findAllNodes(&b, Symbol::fromQualString(kind), recurse);
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
      .def("paramNode", [](Block& b) { return b.param_node(); });

#define NS(name) def(#name, &Node ::name)
  py::class_<Node, std::unique_ptr<Node, py::nodelete>>(m, "Node")
      .def(
          "__repr__",
          [](Node& n) {
            std::stringstream ss;
            ss << n;
            return ss.str();
          })
      .def(
          "sourceRange",
          [](Node& n) {
            return n.sourceRange().str();
          })
      .def("hasMultipleOutputs", [](Node& n) { return n.outputs().size() > 1; })
      .def("outputsSize", [](Node& n) { return n.outputs().size(); })
      .NS(kind)
      .def("inputsAt", [](Node& n, size_t i) { return n.inputs().at(i); })
      .def(
          "inputs",
          [](Node& n) {
            return py::make_iterator(n.inputs().begin(), n.inputs().end());
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
      .NS(addInput)
      .NS(replaceInput)
      .NS(replaceInputWith)
      .NS(replaceAllUsesWith)
      .NS(insertBefore)
      .NS(insertAfter)
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
#define CREATE_ACCESSOR(Kind, method)                          \
  def(#method "_",                                             \
      [](Node& n, const char* name, Kind##Attr::ValueType v) { \
        return n.method##_(Symbol::attr(name), std::move(v));  \
      })                                                       \
      .def(#method, [](Node& n, const char* name) {            \
        return n.method(Symbol::attr(name));                   \
      })
      .CREATE_ACCESSOR(Float, f)
      .CREATE_ACCESSOR(Floats, fs)
      .CREATE_ACCESSOR(String, s)
      .CREATE_ACCESSOR(Strings, ss)
      .CREATE_ACCESSOR(Int, i)
      .CREATE_ACCESSOR(Ints, is)
      .CREATE_ACCESSOR(Graph, g)
      .CREATE_ACCESSOR(Graphs, gs)
#undef CREATE_ACCESSOR
      // Tensor (t_) -- manually written to unwrap the variable into a tensor.
      .def(
          "t_",
          [](Node& n, const char* name, torch::autograd::Variable v) {
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
             std::vector<torch::autograd::Variable> vs) {
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
          [](Node& n, const char* name, at::Tensor v) {
            return n.t_(
                Symbol::attr(name),
                autograd::Variable(v.view({})).set_requires_grad(false));
          })
      .def(
          "z",
          [](Node& n, const char* name) { return n.t(Symbol::attr(name)); })
      .def(
          "zs_",
          [](Node& n, const char* name, TensorsAttr::ValueType v) {
            for (auto& i : v) {
              i = autograd::Variable(i.view({})).set_requires_grad(false);
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
      .def("pyname", [](Node& n) { return n.expect<ConcretePythonOp>()->name(); })
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
  py::class_<Type, std::shared_ptr<Type>>(m, "Type")
      .def("__repr__", [](Type& t) { return t.python_str(); })
      .def(
          "str",
          [](Type& t) {
            std::ostringstream s;
            s << t;
            return s.str();
          })
      .def("kind", [](const Type& t) { return typeKindToString(t.kind()); })
      .def(
          "dim",
          [](const Type& t) {
            return t.expect<DimensionedTensorType>()->dim();
          })
      .def(
          "sizes",
          [](Type& t) { return t.expect<CompleteTensorType>()->sizes(); })
      .def(
          "strides",
          [](Type& t) { return t.expect<CompleteTensorType>()->strides(); })
      .def(
          "contiguous",
          [](Type& t) {
            return std::static_pointer_cast<Type>(
                t.expect<CompleteTensorType>()->contiguous());
          })
      .def(
          "scalarType",
          [](Type& t) {
            return toString(t.expect<DimensionedTensorType>()->scalarType());
          })
      .def(
          "__eq__",
          [](std::shared_ptr<Type>& self, std::shared_ptr<Type>& other) {
            return *self == *other;
          })
      .def(
          "isSubtypeOf",
          [](std::shared_ptr<Type>& self, std::shared_ptr<Type> other) {
            return self->isSubtypeOf(other);
          });

  py::class_<NumberType, Type, std::shared_ptr<NumberType>>(m, "NumberType")
      .def_static("get", &NumberType::get);
  py::class_<IntType, Type, std::shared_ptr<IntType>>(m, "IntType")
      .def_static("get", &IntType::get);
  py::class_<FloatType, Type, std::shared_ptr<FloatType>>(m, "FloatType")
      .def_static("get", &FloatType::get);
  py::class_<TensorType, Type, std::shared_ptr<TensorType>>(m, "TensorType")
      .def_static("get", &TensorType::get);
  py::class_<BoolType, Type, std::shared_ptr<BoolType>>(m, "BoolType")
      .def_static("get", &BoolType::get);
  py::class_<StringType, Type, std::shared_ptr<StringType>>(m, "StringType")
      .def_static("get", &StringType::get);

  py::class_<TupleType, Type, std::shared_ptr<TupleType>>(m, "TupleType")
      .def(
          py::init([](std::vector<TypePtr> a) { return TupleType::create(a); }))
      .def("elements", [](TupleType& self) {
        std::vector<TypePtr> types;
        for (const auto& type : self.elements()) {
          types.push_back(type);
        }
        return types;
      });
  py::class_<ListType, Type, std::shared_ptr<ListType>>(m, "ListType")
      .def(py::init([](TypePtr a) { return ListType::create(a); }))
      .def_static("ofInts", &ListType::ofInts)
      .def_static("ofTensors", &ListType::ofTensors)
      .def("getElementType", &ListType::getElementType);
  py::class_<DictType, Type, std::shared_ptr<DictType>>(m, "DictType")
      .def(py::init([](TypePtr key, TypePtr value) {
        return DictType::create(key, value);
      }))
      .def("getKeyType", &DictType::getKeyType)
      .def("getValueType", &DictType::getValueType);
  py::class_<OptionalType, Type, std::shared_ptr<OptionalType>>(
      m, "OptionalType")
      .def(py::init([](TypePtr a) { return OptionalType::create(a); }))
      .def_static("ofTensor", &OptionalType::ofTensor)
      .def("getElementType", &OptionalType::getElementType);

  py::class_<Use>(m, "Use")
      .def_readonly("user", &Use::user)
      .def_readonly("offset", &Use::offset);
}
} // namespace jit
} // namespace torch
