#include "torch/csrc/toffee/export.h"
#include "torch/csrc/utils/python_numbers.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/Exceptions.h"

#include <toffee/toffee.pb.h>
#include <toffee/schema.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "torch/csrc/autograd/functions/convolution.h"
#include "torch/csrc/jit/dead_code_elimination.h"

#include <fstream>
#undef NDEBUG
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#define NDEBUG

namespace py = pybind11;

namespace torch { namespace jit {

std::string node_name(Node* n) {
  return n->uniqueName();
}

template<typename R, typename T>
static std::vector<R> mapv(const std::vector<T> & inputs, std::function<R(const T &)> fn) {
  std::vector<R> r;
  r.reserve(inputs.size());
  for(auto & input : inputs)
    r.push_back(fn(input));
  return r;
}
// transform PythonOps and Cpp Ops into Node's that match ToffeeIR
// semantics.
// Eventually this should just be part of init_pass but we should avoid
// tight coupling of the JIT and Toffee IR exporter until ready.
std::shared_ptr<Graph> ToToffeeIR(std::shared_ptr<Graph>& g) {
  torch::autograd::PrimSpecContext ctx;
  std::unordered_map<Node*, Node*> env;
  ctx.graph = std::make_shared<Graph>();
  for (auto input : g->inputs())
    env[input] = ctx.graph->addInput()->setType(input->typeOption());
  auto envFn = [&](Node * n) {
    return env.at(n);
  };
  // put the new outputs in our environment map, and
  // copy the type from the input graph if they were not set by the
  // primspec
  auto setOutputs = [&](Node * node, const std::vector<Node*> & outputs) {
    auto old_outputs = node->outputs();
    JIT_ASSERTM(outputs.size() <= old_outputs.size(), "primspec produced too many outputs");
    size_t i = 0;
    for(auto & old : old_outputs) {
      if(i >= outputs.size()) {
        // primspecs do not deal with Handles at the moment
        // so we map handles to Unused nodes
        auto typ = old->typeOption();
        JIT_ASSERTM(typ && typ->kind() == jit::TypeKind::HandleType,
          "primspec produced too few outputs");
        env[old] = ctx.graph->create(jit::kUnused);
      } else {
        if(!outputs[i]->hasType()) {
          outputs[i]->setType(old->typeOption());
          env[old] = outputs[i];
        }
      }
      i++;
    }
  };
  for (auto node : g->nodes()) {
    IR_IF(node, Select)
      //selects are translated by multi-return nodes.
      JIT_ASSERT(env.count(value) > 0);
    IR_ELSEIFM(CppOp)
      if (auto fn = std::dynamic_pointer_cast<autograd::HasPrimSpec>(value->fn)) {
        auto outputs = fn->primspec(&ctx, mapv<Node*,Node*>(node->inputs(),envFn));
        setOutputs(node,outputs);
      } else {
        throw std::runtime_error("CppOp doesn't define primspec " + value->name());
      }
    IR_ELSEIFM(PythonOp)
      auto pyobj = py::handle(value->pyobj.get());
      if(!py::hasattr(pyobj, "primspec"))
        throw std::runtime_error("PythonOp doesn't define primspec " + value->name());

      py::object primspec_fn = pyobj.attr("primspec");

      py::tuple py_primspec_args(1+value->cconv.size());

      auto node_it = node->inputs().begin();
      auto scalar_it = value->scalar_args.begin();
      Py_ssize_t input_nr = 0;
      py_primspec_args[input_nr++] = py::cast(ctx.graph);

      for (auto arg_type : value->cconv) {
        py::object obj;
        if (arg_type == 's') {
          JIT_ASSERTM(scalar_it != value->scalar_args.end(), "expected too many scalar args");
          obj = py::reinterpret_borrow<py::object>(py::handle((scalar_it++)->get()));
        } else if (arg_type == 't') {
          JIT_ASSERTM(node_it != node->inputs().end(),
            "expected too many inputs");
          Node * n_i = envFn(*node_it++);
          obj = py::cast(n_i);
          Node * back = py::cast<Node*>(obj);
          JIT_ASSERT(back == n_i);
        } else {
          throw std::runtime_error("unexpected calling convention");
        }
        py_primspec_args[input_nr++] = obj;
      }
      py::object raw_output = py::reinterpret_borrow<py::object>(PyObject_CallObject(primspec_fn.ptr(), py_primspec_args.ptr()));
      if(!raw_output)
        throw python_error();
      if(raw_output.ptr() == Py_None)
        throw std::runtime_error("PythonOp's primspec returned None, indicating conversion not supported " + value->name());
      std::vector<Node*> outputs;
      if(py::isinstance<Node>(raw_output)) {
        outputs.push_back(py::cast<Node*>(raw_output));
      } else {
        outputs = py::cast<std::vector<Node*>>(raw_output);
      }
      setOutputs(node,outputs);
    IR_ELSE()
      if(node->kind() == kConstant && node->t(kValue).defined()) {
        throw std::runtime_error("Constant not supported yet");
      }
      auto n_ = ctx.graph->createClone(node, envFn);
      if(node->hasMultipleOutputs()) {
        int i = 0;
        for(auto s : node->uses()) {
          env[s.user] = ctx.graph->createSelect(n_,i++);
        }
      } else {
        env[node] = n_;
      }
    IR_END()
  }
  for (auto output : g->outputs()) {
    ctx.graph->registerOutput(env.at(output));
  }
  return std::move(ctx.graph);
}

static void encodeTensor(toffee::TensorProto * p, const at::Tensor & tensor) {
    for(auto d : tensor.sizes()) {
      p->add_dims(d);
    }
    p->set_data_type(toffee::TensorProto_DataType_FLOAT);
    //TODO: other types, we force conversion here
    at::Tensor cont = tensor.toType(at::CPU(at::kFloat));
    float * data = cont.data<float>();
    int64_t N = cont.numel();
    for(int64_t i = 0; i < N; ++i) {
      p->add_float_data(data[i]);
    }
}
static void encodeGraph(toffee::GraphProto * p_g, std::shared_ptr<Graph> & g, const std::vector<at::Tensor> & initializers);
static void addAttribute(toffee::NodeProto * n_p, jit::Node * n, jit::Symbol name) {
  auto attr = n_p->add_attribute();
  attr->set_name(jit::symbolToString(name));
  switch(n->kindOf(name)) {
    case AttributeKind::f:
      attr->set_f(n->f(name));
      break;
    case AttributeKind::fs:
      for(auto & v : n->fs(name))
        attr->add_floats(v);
      break;
    case AttributeKind::i:
      attr->set_i(n->i(name));
      break;
    case AttributeKind::is:
      for(auto & v : n->is(name))
        attr->add_ints(v);
      break;
    case AttributeKind::s:
      attr->set_s(n->s(name));
      break;
    case AttributeKind::ss:
      for(auto & v : n->ss(name))
        attr->add_strings(v);
      break;
    case AttributeKind::t: {
      //TODO: tensors but no tensor?
      auto t = attr->add_tensors();
      encodeTensor(t, n->t(name));
    } break;
    case AttributeKind::ts:
      for(auto & v : n->ts(name)) {
        auto t = attr->add_tensors();
        encodeTensor(t, v);
      }
      break;
    case AttributeKind::g: {
      //TODO: graphs but no graph?
      auto g = attr->add_graphs();
      encodeGraph(g, n->g(name), {});
    } break;
    case AttributeKind::gs:
      for(auto & v : n->gs(name)) {
        auto g = attr->add_graphs();
        encodeGraph(g, v, {});
      }
      break;
  }
}

static void encodeGraph(toffee::GraphProto * p_g, std::shared_ptr<Graph> & g, const std::vector<at::Tensor> & initializers) {
  for (auto input : g->inputs()) {
    p_g->add_input(node_name(input));
  }
  for (auto output : g->outputs()) {
    p_g->add_output(node_name(output));
  }
  for (auto node : g->nodes()) {
    if (node->kind() == kSelect) {
      // No select nodes in ToffeeIR: instead we make use
      // of the select invariant
      continue;
    }
    auto p_n = p_g->add_node();
    for(auto input : node->inputs()) {
      p_n->add_input(node_name(input));
    }
    // jit::Node and toffee protobuf don't agree on how to represent
    // so called 'inplace' operators that mutate inputs
    // 'InPlaceOutputs' has an entry for each output of this node
    // if the entry is >= 0, it specifies the index of the input
    // which is mutated and returned as an output
    // we use that to translate to ToffeeIR's replicated naming scheme
    // where inputs/outputs will have the same name
    std::vector<int64_t> * inplace_outputs = nullptr;
    if(node->hasAttribute(jit::kInPlaceOutputs))
      inplace_outputs = &node->is(jit::kInPlaceOutputs);
    int i = 0;
    for(auto output : node->outputs()) {
      if(!inplace_outputs || inplace_outputs->at(i) < 0) {
        p_n->add_output(node_name(output));
      } else {
        Node * input = node->inputs().at(inplace_outputs->at(i));
        p_n->add_output(node_name(input));
      }
      i++;
    }
    p_n->set_op_type(symbolToString(node->kind()));
    for(auto attr_name : node->attributeNames()) {
      if(attr_name == kInPlaceOutputs)
        continue;
      addAttribute(p_n, node, attr_name);
    }
  }
  int inputs_count = 0;
  for (auto & tensor : initializers) {
    // TODO: stop using positions to determine which initializers
    // match to which inputs
    std::string name = p_g->input(inputs_count++);
    auto p = p_g->add_initializer();
    p->set_name(name);
    encodeTensor(p, tensor);
  }
}

// Exports a graph to ToffeeIR
std::string ExportGraph(std::shared_ptr<Graph>& g_, const std::vector<at::Tensor> & initializers) {
  auto g = ToToffeeIR(g_);
  toffee::GraphProto p_g;
  p_g.set_name("torch-jit-export");
  encodeGraph(&p_g, g, initializers);
  std::string s;
  google::protobuf::TextFormat::PrintToString(p_g, &s);
  return s; // RVO
}

}}
