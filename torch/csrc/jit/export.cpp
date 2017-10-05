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

void encodeTypeProtoTensorType(onnx::TypeProtoTensorTypeProto* tensor_type, Node* n) {
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
    case at::kFloat:
    case at::kHalf:
      onnx_type = onnx::kFLOAT;
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
      jit::barf("unexpected tensor scalar type");
      break;
  }
  tensor_type->set_data_type(onnx_type);
}

void encodeValueInfo(onnx::ValueInfoProto* v, Node* n) {
  v->set_name(node_name(n));
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

void standardizeGraph(const std::shared_ptr<Graph>& graph) {
  for (auto it = graph->nodes().begin(); it != graph->nodes().end(); ++it) {
#define FAIL_EXPORT(name) \
      throw std::runtime_error(std::string("Couldn't export ") + name + " function - " \
              "maybe it doesn't implement a symbolic definition?");
    IR_IF(*it, AddConstant)
      throw std::runtime_error("can't serialize PyTorch-only node AddConstant (not implemented yet)");
    IR_ELSEIF(Concat)
      JIT_ASSERT(!value->hasMultipleOutputs());
      Node *real_output = value->makeMultireturn();
      Node *fake_output = graph->createSelect(value, 1);
      fake_output->insertAfter(real_output);
    IR_ELSEIF(CppOp)
      auto cpp_node = static_cast<torch::jit::CppOp*>(value);
      FAIL_EXPORT(cpp_node->name())
    IR_ELSEIF(PythonOp)
      auto py_node = static_cast<torch::jit::PythonOp*>(value);
      if (py_node->name() == "Index") {
        if (py_node->scalar_args.size() != 1 ||
            !THPUtils_checkLong(py_node->scalar_args[0].get())) {
          throw std::runtime_error("ONNX export only support indexing with a single int");
        }
        auto index = THPUtils_unpackLong(py_node->scalar_args[0].get());
        JIT_ASSERT(py_node->inputs().size() == 1);
        auto input = py_node->inputs()[0];
        auto input_type = input->type()->expect<TensorType>();
        int64_t ndim = input_type->sizes().size();

        // Create starts and ends
        auto starts = at::CPU(at::kInt).zeros({ndim});
        auto starts_data = starts.toIntData();
        auto ends = at::CPU(at::kInt).tensor({ndim});
        auto ends_data = ends.toIntData();

        // Fill them to select out a single slice along first dim
        starts_data[0] = index;
        std::copy(input_type->sizes().begin(), input_type->sizes().end(), ends_data);
        ends_data[0] = index + 1;

        Node *starts_constant = graph->create(kConstant)->t_(kvalue, starts)->insertBefore(py_node);
        Node *ends_constant = graph->create(kConstant)->t_(kvalue, ends)->insertBefore(py_node);

        Node *slice = graph->create(kSlice, {input, starts_constant, ends_constant})
                           ->insertBefore(py_node);
        Node *squeeze = graph->create(kSqueeze, {slice})->is_(kaxes, {0})
                             ->insertBefore(py_node);
        auto first_select = py_node->uses()[0].user;
        first_select->replaceAllUsesWith(squeeze);
        for (auto use : py_node->uses())
          use.user->destroy();
        it.destroyCurrent();
      } else {
        FAIL_EXPORT(py_node->name())
      }
    IR_ELSE()
      // Do nothing.
    IR_END()
#undef FAIL_EXPORT
  }
}

}

std::string ExportGraph(const std::shared_ptr<Graph>& graph,
                        const std::vector<at::Tensor> & initializers) {

  standardizeGraph(graph);

  onnx::ModelProto model_proto;
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
