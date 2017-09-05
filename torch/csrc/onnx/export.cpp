#include "torch/csrc/onnx/export.h"
#include "torch/csrc/autograd/symbolic.h"
#include "torch/csrc/utils/python_numbers.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/onnx.h"

#include "torch/csrc/autograd/functions/convolution.h"
#include "torch/csrc/jit/dead_code_elimination.h"
#include "torch/csrc/utils/functional.h"
#include <ATen/ATen.h>

#include <fstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace torch { namespace jit {

std::string node_name(Node* n) {
  return n->uniqueName();
}

// transform PythonOps and Cpp Ops into Node's that match ONNX
// semantics.
// Eventually this should just be part of init_pass but we should avoid
// tight coupling of the JIT and ONNX IR exporter until ready.
std::shared_ptr<Graph> ToONNX(std::shared_ptr<Graph>& g,
                                  const std::unordered_map<void*, Node*>& old_buffer_map,
                                  bool verbose) {
  torch::autograd::SymbolicContext ctx;
  std::unordered_map<Node*, Node*> env;
  std::shared_ptr<Graph> out_graph = std::make_shared<Graph>();
  ctx.graph = out_graph.get();
  for (auto input : g->inputs())
    env[input] = ctx.graph
      ->addInput()
      ->setType(input->typeOption())
      ->setDebugName(input->debugName());
  auto envFn = [&env](Node * n) {
    auto it = env.find(n);
    JIT_ASSERTM(it != env.end(), "Dangling node reference");
    JIT_ASSERTM(it->second, "Unused node was subsequently used");
    return it->second;
  };
  std::unordered_map<void*, Node*> buffer_map;
  for (auto kv : old_buffer_map) {
    buffer_map[kv.first] = envFn(kv.second);
  }
  ctx.buffer_map = &buffer_map;
  // put the new outputs in our environment map, and
  // copy the type from the input graph if they were not set by the
  // symbolic
  auto setOutputs = [&](Node * node, const node_list & outputs) {
    auto old_outputs = node->outputs();
    // The symbolic can produce less outputs than the actual IR node,
    // because many IR nodes have an implicit extra trailing output
    // of type Handle, which is irrelevant for the purposes of export.
    // It's bad design to ask the symbolic() implementers to actually
    // handle this!
    JIT_ASSERTM(outputs.size() <= old_outputs.size(), "symbolic produced too many outputs");
    size_t i = 0;
    for(auto & old : old_outputs) {
      // NB: There is at most one handle, and if it exists, it is the last input
      if(i >= outputs.size()) {
        // symbolics do not deal with Handles at the moment, so we just
        // assert the handle isn't actually used.
        auto typ = old->typeOption();
        JIT_ASSERTM(typ && typ->kind() == jit::TypeKind::HandleType,
          "symbolic produced too few outputs");
        env[old] = nullptr;
        if (!old->uses().empty()) {
          throw std::runtime_error("In ONNX export, handles should be unused");
        }
      } else {
        if (outputs[i]) {
          if (!outputs[i]->hasType()) {
            outputs[i]->setType(old->typeOption());
            env[old] = outputs[i];
          } else {
            // This case occurs if an input was just passed through
            env[old] = outputs[i];
          }
        } else {
          env[old] = nullptr;
          if (!old->uses().empty()) {
            throw std::runtime_error("In ONNX export, non-exported PyTorch return not supported " + std::to_string(i));
          }
        }
      }
      i++;
    }
  };
  auto visitNode = [&](Node * node) {
    IR_IF(node, Select)
      // Selects are translated by multi-return nodes.
      JIT_ASSERT(env.count(value) > 0);
    IR_ELSEIFM(CppOp)
      if (auto fn = std::dynamic_pointer_cast<autograd::HasSymbolic>(value->fn)) {
        auto outputs = fn->symbolic(&ctx, fmap(node->inputs(), envFn));
        setOutputs(node, outputs);
      } else {
        throw std::runtime_error("CppOp doesn't define symbolic " + value->name());
      }
    IR_ELSEIFM(PythonOp)
      auto pyobj = py::handle(value->pyobj.get());
      if(!py::hasattr(pyobj, "symbolic"))
        throw std::runtime_error("PythonOp doesn't define symbolic " + value->name());

      py::object symbolic_fn = pyobj.attr("symbolic");

      py::tuple py_symbolic_args(1+value->cconv.size());

      auto node_it = node->inputs().begin();
      auto scalar_it = value->scalar_args.begin();
      Py_ssize_t input_nr = 0;
      py_symbolic_args[input_nr++] = py::cast(ctx.graph);

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
        py_symbolic_args[input_nr++] = obj;
      }
      py::object raw_output = py::reinterpret_steal<py::object>(PyObject_CallObject(symbolic_fn.ptr(), py_symbolic_args.ptr()));
      if(!raw_output)
        throw py::error_already_set();
      if(raw_output.ptr() == Py_None)
        throw std::runtime_error("PythonOp's symbolic returned None, indicating conversion not supported " + value->name());
      node_list outputs;
      if(py::isinstance<Node>(raw_output)) {
        outputs.push_back(py::cast<Node*>(raw_output));
      } else {
        outputs = py::cast<std::vector<Node*>>(raw_output);
      }
      setOutputs(node, outputs);
    IR_ELSE()
      auto n_ = ctx.graph->createClone(node, envFn);
      ctx.graph->appendNode(n_); // will be ignored by ONNX
      if(node->hasMultipleOutputs()) {
        int i = 0;
        for(auto s : node->uses()) {
          auto new_node = ctx.graph->createSelect(n_,i++);
          ctx.graph->appendNode(new_node);
          new_node->setType(s.user->typeOption());
          env[s.user] = new_node;
        }
      } else {
        env[node] = n_;
      }
    IR_END()
  };
  for(auto n : g->nodes()) {
    try {
      visitNode(n);
    } catch(...) {
      if(verbose) {
        std::cerr << "Error occured while handling:\n" << *n << "\n";
        std::cerr << "Exported graph so far:\n" << *out_graph << "\n";
      }
      throw;
    }
  }

  for (auto output : g->outputs()) {
    ctx.graph->registerOutput(env.at(output));
  }
  return out_graph; // RVO
}

static void encodeTensor(onnx::TensorProto * p, const at::Tensor & tensor) {
    for(auto d : tensor.sizes()) {
      p->add_dims(d);
    }
    at::ScalarType at_type;
    onnx::DataType onnx_type;
    switch(tensor.type().scalarType()) {
      case at::kDouble:
      case at::kFloat:
      case at::kHalf:
        onnx_type = onnx::kFLOAT;
        at_type = at::kFloat;
        break;
      case at::kByte:
      case at::kChar:
        onnx_type = onnx::kINT8;
        at_type = at::kByte;
        break;
      case at::kShort:
        onnx_type = onnx::kINT16;
        at_type = at::kShort;
        break;
      case at::kInt:
        onnx_type = onnx::kINT32;
        at_type = at::kInt;
        break;
      case at::kLong:
        onnx_type = onnx::kINT64;
        at_type = at::kLong;
        break;
      default:
        jit::barf("unexpected tensor scalar type");
        break;
    }
    p->set_data_type(onnx_type);
    at::Tensor cont = tensor.toType(at::CPU(at_type)).contiguous();
    p->set_raw_data(cont);
}
static void encodeGraph(onnx::GraphProto * p_g, std::shared_ptr<Graph> & g, const std::vector<at::Tensor> & initializers);
static void addAttribute(onnx::NodeProto * n_p, jit::Node * n, jit::Symbol name) {
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
      auto t = attr->mutable_t();
      encodeTensor(t, n->t(name));
    } break;
    case AttributeKind::ts:
      for(auto & v : n->ts(name)) {
        auto t = attr->add_tensors();
        encodeTensor(t, v);
      }
      break;
    case AttributeKind::g: {
      auto g = attr->mutable_g();
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

static void encodeGraph(onnx::GraphProto * p_g, std::shared_ptr<Graph> & g, const std::vector<at::Tensor> & initializers) {
  for (auto input : g->inputs()) {
    p_g->add_input(node_name(input));
  }
  for (auto output : g->outputs()) {
    p_g->add_output(node_name(output));
  }
  for (auto node : g->nodes()) {
    if (node->kind() == kSelect) {
      // No select nodes in ONNX: instead we make use
      // of the select invariant
      continue;
    }
    if (node->kind() == kUndefined && node->uses().empty()) {
      // Undefined nodes never show up in ONNX; they're just a tool
      // to help symbolics do the right thing.
      continue;
    }
    auto p_n = p_g->add_node();
    for(auto input : node->inputs()) {
      p_n->add_input(node_name(input));
    }
    for(auto output : node->outputs()) {
      p_n->add_output(node_name(output));
    }
    p_n->set_op_type(symbolToString(node->kind()));
    for(auto attr_name : node->attributeNames()) {
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

// Exports a graph to ONNX
std::string ExportGraph(std::shared_ptr<Graph>& g_,
                        const std::unordered_map<void*, Node*>& buffer_map,
                        const std::vector<at::Tensor> & initializers,
                        bool verbose) {
  auto g = ToONNX(g_, buffer_map, verbose);
  if(verbose) {
    std::cout << *g << "\n";
  }
  g->lint();
  onnx::GraphProto p_g;
  p_g.set_name("torch-jit-export");
  encodeGraph(&p_g, g, initializers);
  size_t out_size;
  pb_get_encoded_size(&out_size, onnx_GraphProto_fields, &p_g.proto);
  std::string out(out_size, '\0');
  pb_ostream_t ostream = pb_ostream_from_buffer(reinterpret_cast<pb_byte_t *>(&out[0]), out_size);
  pb_encode(&ostream, onnx_GraphProto_fields, &p_g.proto);
  return out; // RVO
}

}}
