#include <Python.h>

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

std::string value_name(Value* n) {
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
      onnx_type = onnx::kDOUBLE;
      at_type = at::kDouble;
      break;
    case at::kFloat:
      onnx_type = onnx::kFLOAT;
      at_type = at::kFloat;
      break;
    case at::kHalf:
      onnx_type = onnx::kFLOAT16;
      at_type = at::kHalf;
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
      torch::barf("unexpected tensor scalar type");
      break;
  }
  p->set_data_type(onnx_type);
  // CPU's HalfTensor doesn't have contiguous(), so first calling contiguous()
  at::Tensor cont = tensor.contiguous().toType(at::CPU(at_type));
  p->set_raw_data(cont);
}

void addAttribute(onnx::NodeProto * n_p, jit::Node * n, jit::Symbol name) {
  auto attr = n_p->add_attribute();
  attr->set_name(jit::symbolToString(name));
  switch(n->kindOf(name)) {
    case AttributeKind::f:
      attr->set_f(n->f(name));
      attr->set_type(onnx::aFLOAT);
      break;
    case AttributeKind::fs:
      attr->set_type(onnx::aFLOATS);
      for(auto & v : n->fs(name))
        attr->add_floats(v);
      break;
    case AttributeKind::i:
      attr->set_type(onnx::aINT);
      attr->set_i(n->i(name));
      break;
    case AttributeKind::is:
      attr->set_type(onnx::aINTS);
      for(auto & v : n->is(name))
        attr->add_ints(v);
      break;
    case AttributeKind::s:
      attr->set_type(onnx::aSTRING);
      attr->set_s(n->s(name));
      break;
    case AttributeKind::ss:
      attr->set_type(onnx::aSTRINGS);
      for(auto & v : n->ss(name))
        attr->add_strings(v);
      break;
    case AttributeKind::t: {
      attr->set_type(onnx::aTENSOR);
      auto t = attr->mutable_t();
      encodeTensor(t, n->t(name));
    } break;
    case AttributeKind::ts:
      attr->set_type(onnx::aTENSORS);
      for(auto & v : n->ts(name)) {
        auto t = attr->add_tensors();
        encodeTensor(t, v);
      }
      break;
    case AttributeKind::g: {
      attr->set_type(onnx::aGRAPH);
      auto g = attr->mutable_g();
      encodeGraph(g, n->g(name), {});
    } break;
    case AttributeKind::gs:
      attr->set_type(onnx::aGRAPHS);
      for(auto & v : n->gs(name)) {
        auto g = attr->add_graphs();
        encodeGraph(g, v, {});
      }
      break;
  }
}

void encodeTypeProtoTensorType(onnx::TypeProtoTensorTypeProto* tensor_type, Value* n) {
  onnx::TypeProtoTensorShapeProto* shape = tensor_type->mutable_shape();
  JIT_ASSERT(n->hasType());
  TensorType* node_type = n->type()->expect<TensorType>();
  const std::vector<std::int64_t>& sizes = node_type->sizes();
  for (std::int64_t s : sizes) {
    shape->add_dim(s);
  }
  onnx::DataType onnx_type;
  switch(node_type->scalarType()) {
    case at::kDouble:
      onnx_type = onnx::kDOUBLE;
      break;
    case at::kFloat:
      onnx_type = onnx::kFLOAT;
      break;
    case at::kHalf:
      onnx_type = onnx::kFLOAT16;
      break;
    case at::kByte:
    case at::kChar:
      onnx_type = onnx::kINT8;
      break;
    case at::kShort:
      onnx_type = onnx::kINT16;
      break;
    case at::kInt:
      onnx_type = onnx::kINT32;
      break;
    case at::kLong:
      onnx_type = onnx::kINT64;
      break;
    default:
      torch::barf("unexpected tensor scalar type");
      break;
  }
  tensor_type->set_data_type(onnx_type);
}

void encodeValueInfo(onnx::ValueInfoProto* v, Value* n) {
  v->set_name(value_name(n));
  onnx::TypeProto* t = v->mutable_type();
  onnx::TypeProtoTensorTypeProto* tensor_type = t->mutable_tensor_type();
  encodeTypeProtoTensorType(tensor_type, n);
}

void encodeGraph(onnx::GraphProto * p_g, const std::shared_ptr<Graph> & g, const std::vector<at::Tensor> & initializers) {
  JIT_ASSERT(p_g != nullptr);
  p_g->set_name("torch-jit-export");

  for (auto input : g->inputs()) {
    onnx::ValueInfoProto* v = p_g->add_input();
    encodeValueInfo(v, input);
  }
  for (auto output : g->outputs()) {
    onnx::ValueInfoProto* v = p_g->add_output();
    encodeValueInfo(v, output);
  }
  for (auto node : g->nodes()) {
    if (node->kind() == kUndefined && !node->hasUses()) {
      // Undefined nodes never show up in ONNX; they're just a tool
      // to help symbolics do the right thing.
      continue;
    }
    auto p_n = p_g->add_node();
    if (node->getSourceLocation()) {
      p_n->set_doc_string(node->getSourceLocation()->python_traceback);
    }
    for(auto input : node->inputs()) {
      p_n->add_input(value_name(input));
    }
    for(auto output : node->outputs()) {
      p_n->add_output(value_name(output));
    }
    p_n->set_op_type(symbolToString(node->kind()));
    for(auto attr_name : node->attributeNames()) {
      addAttribute(p_n, node, attr_name);
    }
  }
  auto num_initializers = initializers.size();
  int inputs_count = g->inputs().size() - num_initializers;
  for (auto & tensor : initializers) {
    // TODO: stop using positions to determine which initializers
    // match to which inputs
    std::string name = p_g->get_input_name(inputs_count++);
    auto p = p_g->add_initializer();
    p->set_name(name);
    encodeTensor(p, tensor);
  }
}

void encodeModel(onnx::ModelProto* p_m, const std::shared_ptr<Graph>& g,
                 const std::vector<at::Tensor>& initializers) {
  onnx::GraphProto* p_g = p_m->mutable_graph();
  encodeGraph(p_g, g, initializers);
}

void validateGraph(const std::shared_ptr<Graph>& graph) {
  for (auto it = graph->begin(); it != graph->end(); ++it) {
      // Macro'ed so we get a marginally better line number on failed export
#define FAIL_EXPORT(name) \
      throw std::runtime_error(std::string("ONNX export failed: ") + name + "\n\nGraph we tried to export:\n" + graph->toString());
    IR_IF(*it, CppOp)
      auto cpp_node = static_cast<torch::jit::CppOp*>(value);
      FAIL_EXPORT("Couldn't export C++ operator " + cpp_node->name())
    IR_ELSEIF(PythonOp)
      auto py_node = static_cast<torch::jit::PythonOp*>(value);
      FAIL_EXPORT("Couldn't export Python operator " + py_node->name())
    IR_ELSE()
      // Expand is not a real ONNX operator yet, reject it
      if (it->kind() == kExpand) {
        FAIL_EXPORT("Couldn't export operator expand; this usually means you used a form of broadcasting that ONNX does not currently support");
      }
      if (it->kind() == kUndefined) {
        FAIL_EXPORT("Couldn't export undefined constant tensor (please file an issue)")
      }
      std::string n = symbolToString(it->kind());
      if (n.size() == 0) {
        FAIL_EXPORT("Operator to export had empty name (please file an issue)")
      }
      // NB: Upper-case is ONNX, lower-case is ATen.  If we want to be more
      // robust, need to explicitly flag operators as ONNX or ATen
      if (!isupper(n[0])) {
        FAIL_EXPORT("Couldn't export operator " + n);
      }
    IR_END()
#undef FAIL_EXPORT
  }
}

}

std::string ExportGraph(const std::shared_ptr<Graph>& graph,
                        const std::vector<at::Tensor> & initializers,
                        int64_t onnx_opset_version) {

  validateGraph(graph);

  onnx::ModelProto model_proto;
  model_proto.set_producer_name("pytorch");
  model_proto.set_producer_version("0.3");
  auto* imp = model_proto.add_opset_import();
  // This is the version of ONNX operator set we are targeting
  imp->set_version(onnx_opset_version);

  // Set up nanopb callbacks and compute the amount of space needed to store
  // the resulting protobuf
  encodeModel(&model_proto, graph, initializers);

  size_t out_size;
  pb_get_encoded_size(&out_size, onnx_ModelProto_fields, &model_proto.proto);

  // Allocate storage and export the graph
  std::string out(out_size, '\0');
  pb_ostream_t ostream = pb_ostream_from_buffer(reinterpret_cast<pb_byte_t *>(&out[0]), out_size);
  pb_encode(&ostream, onnx_ModelProto_fields, &model_proto.proto);

  return out;
}

}}
