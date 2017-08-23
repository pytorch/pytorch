#include <pybind11/pybind11.h>
// DO NOT REMOVE, this enables std containers to be recognized
// with pybind11, removing the include disables the support
#include <pybind11/stl.h>

namespace py = pybind11;
#include <iostream>
#include <sstream>
#include "torch/csrc/jit/ir.h"
struct THPGenerator;
#include "torch/csrc/Module.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/jit/python_tracer.h"

// handle Tensor <-> at::Tensor conversions
namespace pybind11 { namespace detail {
    template <> struct type_caster<at::Tensor> {
    public:
        PYBIND11_TYPE_CASTER(at::Tensor, _("at::Tensor"));

        bool load(handle src, bool) {
            /* Extract PyObject from handle */
            PyObject *source = src.ptr();
            if(!THPModule_isTensor(source))
              return false;
            value = torch::createTensor(source);
            return true;
        }
        static handle cast(at::Tensor src, return_value_policy /* policy */, handle /* parent */) {
            return handle(torch::createPyObject(src));
        }
    };
}} // namespace pybind11::detail

namespace torch { namespace jit {

void initPythonIRBindings(PyObject * module_) {
  auto m = py::handle(module_).cast<py::module>();
  #define GS(name) \
    def(#name,&Graph :: name)
  py::class_<Graph,std::shared_ptr<Graph>>(m,"Graph")
    .def(py::init<>())
    .def("__repr__",[](Graph & g) {
      std::stringstream ss;
      ss << g;
      return ss.str();
    })
    .GS(inputs)
    .GS(outputs)
    .def("nodes",[](Graph &g) {
      return py::make_iterator(g.nodes().begin(),g.nodes().end());
    })
    .GS(addInput)
    .GS(advanceStage)
    .GS(stage)
    .GS(eraseInput)
    .GS(registerOutput)
    .def("create",[](Graph & g, const char * str) {
      return g.create(stringToSymbol(str));
    })
    .def("create",[](Graph & g, const char * str, const std::vector<Node*> & inputs) {
      return g.create(stringToSymbol(str),inputs);
    })
    .GS(createSelect)
    .GS(createChunk)
    .GS(createConstant)
    .GS(createFusionGroup)
    .def("createClone",[](Graph & g, Node * n, py::object fn) {
      return g.createClone(n, [&](Node * e) {
        return fn(e).cast<Node*>();
      });
    })
    .GS(appendNode)
    .GS(prependNode)
    .GS(lint)
    ;
    #undef GS

  #define NS(name) \
    def(#name,&Node :: name)
  py::class_<Node,std::unique_ptr<Node, py::nodelete>>(m,"Node")
    .def("__repr__",[](Node & n) {
      std::stringstream ss;
      ss << n;
      return ss.str();
    })
    .def("kind",[](Node & n) {
      return symbolToString(n.kind());
    })
    .NS(stage)
    .NS(type)
    .NS(typeOption)
    .NS(hasMultipleOutputs)
    .NS(hasType)
    .NS(setType)
    .NS(inferTypeFrom)
    // skip owningGraph because it returns a raw pointer to a otherwise
    // std::shared_ptr stored graph object, and would cause a double free
    .NS(debugName)
    .NS(setDebugName)
    .NS(unique)
    .NS(uniqueName)
    .NS(setStage)
    .NS(stage)
    .NS(inputs)
    .NS(input)
    .NS(outputs)
    .NS(offset)
    .NS(uses)
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
    // methods from Attributes
    .def("copyAttributes",[](Node & n, Node & rhs) {
      n.copyAttributes(rhs);
    })
    .def("hasAttribute",[](Node & n, const char * name) {
      return n.hasAttribute(stringToSymbol(name));
    })
    .def("kindOf", [](Node & n, const char * name) {
      return toString(n.kindOf(stringToSymbol(name)));
    })
    .def("removeAttribute",[](Node & n, const char * name) {
      return n.removeAttribute(stringToSymbol(name));
    })
    .NS(hasAttributes)
    .def("attributeNames",[](Node & n) {
      auto names = n.attributeNames();
      std::vector<std::string> names_s;
      for(auto n : names)
        names_s.push_back(symbolToString(n));
      return names_s;
    })
    #define CREATE_ACCESSOR(Kind,method) \
    def(#method "_",[](Node & n, const char * name, Kind##Attr::ValueType v) { \
      return n . method ## _(stringToSymbol(name), std::move(v)); \
    }) \
    .def(#method, [](Node & n, const char * name) { \
      return n.method(stringToSymbol(name)); \
    })
    .CREATE_ACCESSOR(Float,f)
    .CREATE_ACCESSOR(Floats,fs)
    .CREATE_ACCESSOR(String,s)
    .CREATE_ACCESSOR(Strings,ss)
    .CREATE_ACCESSOR(Int,i)
    .CREATE_ACCESSOR(Ints,is)
    .CREATE_ACCESSOR(Tensor,t)
    .CREATE_ACCESSOR(Tensors,ts)
    .CREATE_ACCESSOR(Graph,g)
    .CREATE_ACCESSOR(Graphs,gs)
    .def("pyobj",[](Node & n) {
      return py::handle(n.expect<PythonOp>()->pyobj.get()).cast<py::object>();
    })
    .def("cconv",[](Node & n) {
      return n.expect<PythonOp>()->cconv;
    })
    .def("pyname",[](Node & n) {
      return n.expect<PythonOp>()->name();
    })
    .def("scalar_args",[](Node & n) {
      auto op = n.expect<PythonOp>();
      auto scalars = py::list();
      auto append = scalars.attr("append");
      for(auto & arg : op->scalar_args) {
        append(py::handle(arg.get()));
      }
      return scalars;
    })
    ;

  #define TS(name) \
    def(#name,&Node :: name)
  py::class_<Type,std::shared_ptr<Type>>(m,"Type")
    .def("__repr__",[](Type & t) {
      std::stringstream ss;
      ss << t;
      return ss.str();
    })
    .def("kind",[](Type& t_) {
      Type * t = &t_;
      TYPE_IF(t,MultiType)
        return "MultiType";
      TYPE_ELSEIF(HandleType)
        return "HandleType";
      TYPE_ELSEIF(TensorType)
        return "TensorType";
      TYPE_END()
      jit::barf("unknown type kind");
      return "";
    })
    .def("sizes",[](Type& t) {
      return t.expect<TensorType>()->sizes();
    })
    .def("strides",[](Type& t) {
      return t.expect<TensorType>()->strides();
    })
    .def("contiguous",[](Type& t) {
      return t.expect<TensorType>()->contiguous();
    })
    .def("scalarType",[](Type& t) {
      return at::toString(t.expect<TensorType>()->scalarType());
    })
    ;

  py::class_<Use>(m,"Use")
  .def_readonly("user",&Use::user)
  .def_readonly("offset",&Use::offset);

  m.def("_jit_get_graph", [](py::object h_) {
    auto py_state = h_.ptr();
    JIT_ASSERTM(THPTracingState_Check(py_state), "expected a TracingState instance");
    THPTracingState *state = (THPTracingState*)py_state;
    return state->cdata->graph;
  });
}
}}
