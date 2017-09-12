#include "torch/csrc/jit/export.h"
#include "torch/csrc/onnx/onnx.h"
#include "torch/csrc/autograd/symbolic.h"
#include "torch/csrc/utils/python_numbers.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/Exceptions.h"

#include "torch/csrc/autograd/functions/convolution.h"
#include "torch/csrc/utils/functional.h"
#include <ATen/ATen.h>

#include <fstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace torch { namespace jit {

namespace {

std::string node_name(Node* n) {
  return n->uniqueName();
}

void encodeGraph(onnx::GraphProto * p_g, const std::shared_ptr<Graph> & g, const std::vector<at::Tensor> & initializers);

void encodeTensor(onnx::TensorProto * p, const at::Tensor & tensor) {
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

void addAttribute(onnx::NodeProto * n_p, jit::Node * n, jit::Symbol name) {
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

void encodeGraph(onnx::GraphProto * p_g, const std::shared_ptr<Graph> & g, const std::vector<at::Tensor> & initializers) {
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

void checkGraph(const std::shared_ptr<Graph>& graph) {
  for (auto node : graph->nodes()) {
    if (node->kind() == kPythonOp || node->kind() == kCppOp) {
#define GET_NAME(T) dynamic_cast<T*>(node)->name()
      auto name = node->kind() == kPythonOp ? GET_NAME(PythonOp) : GET_NAME(CppOp);
      throw std::runtime_error(std::string("Couldn't export ") + name + " function - "
              "maybe it doesn't implement a symbolic definition?");
#undef GET_NAME
    }
    if (node->kind() == kAddConstant) {
      throw std::runtime_error("can't serialize PyTorch-only node AddConstant (not implemented yet)");
    }
  }
}

}

std::string ExportGraph(const std::shared_ptr<Graph>& graph,
                        const std::vector<at::Tensor> & initializers) {
  checkGraph(graph);

  onnx::GraphProto graph_proto;
  graph_proto.set_name("torch-jit-export");

  // Set up nanopb callbacks and compute the amount of space needed to store
  // the resulting protobuf
  encodeGraph(&graph_proto, graph, initializers);
  size_t out_size;
  pb_get_encoded_size(&out_size, onnx_GraphProto_fields, &graph_proto.proto);

  // Allocate storage and export the graph
  std::string out(out_size, '\0');
  pb_ostream_t ostream = pb_ostream_from_buffer(reinterpret_cast<pb_byte_t *>(&out[0]), out_size);
  pb_encode(&ostream, onnx_GraphProto_fields, &graph_proto.proto);

  return out;
}

}}
