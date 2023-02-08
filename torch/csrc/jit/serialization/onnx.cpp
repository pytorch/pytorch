#include <c10/util/irange.h>
#include <torch/csrc/jit/serialization/onnx.h>
#include <torch/csrc/onnx/onnx.h>

#include <sstream>
#include <string>

namespace torch::jit {

namespace {
namespace onnx_torch = ::torch::onnx;
namespace onnx = ::ONNX_NAMESPACE;

// Pretty printing for ONNX
constexpr char indent_char = ' ';
constexpr size_t indent_multiplier = 2;

std::string idt(size_t indent) {
  return std::string(indent * indent_multiplier, indent_char);
}

std::string nlidt(size_t indent) {
  return std::string("\n") + idt(indent);
}

void dump(const onnx::TensorProto& tensor, std::ostream& stream) {
  stream << "TensorProto shape: [";
  for (const auto i : c10::irange(tensor.dims_size())) {
    stream << tensor.dims(i) << (i == tensor.dims_size() - 1 ? "" : " ");
  }
  stream << "]";
}

void dump(const onnx::TensorShapeProto& shape, std::ostream& stream) {
  for (const auto i : c10::irange(shape.dim_size())) {
    auto& dim = shape.dim(i);
    if (dim.has_dim_value()) {
      stream << dim.dim_value();
    } else {
      stream << "?";
    }
    stream << (i == shape.dim_size() - 1 ? "" : " ");
  }
}

void dump(const onnx::TypeProto_Tensor& tensor_type, std::ostream& stream) {
  stream << "Tensor dtype: ";
  if (tensor_type.has_elem_type()) {
    stream << tensor_type.elem_type();
  } else {
    stream << "None.";
  }
  stream << ", ";
  stream << "Tensor dims: ";
  if (tensor_type.has_shape()) {
    dump(tensor_type.shape(), stream);
  } else {
    stream << "None.";
  }
}

void dump(const onnx::TypeProto& type, std::ostream& stream);

void dump(const onnx::TypeProto_Optional& optional_type, std::ostream& stream) {
  stream << "Optional<";
  if (optional_type.has_elem_type()) {
    dump(optional_type.elem_type(), stream);
  } else {
    stream << "None";
  }
  stream << ">";
}

void dump(const onnx::TypeProto_Sequence& sequence_type, std::ostream& stream) {
  stream << "Sequence<";
  if (sequence_type.has_elem_type()) {
    dump(sequence_type.elem_type(), stream);
  } else {
    stream << "None";
  }
  stream << ">";
}

void dump(const onnx::TypeProto& type, std::ostream& stream) {
  if (type.has_tensor_type()) {
    dump(type.tensor_type(), stream);
  } else if (type.has_sequence_type()) {
    dump(type.sequence_type(), stream);
  } else if (type.has_optional_type()) {
    dump(type.optional_type(), stream);
  } else {
    stream << "None";
  }
}

void dump(const onnx::ValueInfoProto& value_info, std::ostream& stream) {
  stream << "{name: \"" << value_info.name() << "\", type:";
  dump(value_info.type(), stream);
  stream << "}";
}

void dump(const onnx::GraphProto& graph, std::ostream& stream, size_t indent);

void dump(
    const onnx::AttributeProto& attr,
    std::ostream& stream,
    size_t indent) {
  stream << "{ name: '" << attr.name() << "', type: ";
  if (attr.has_f()) {
    stream << "float, value: " << attr.f();
  } else if (attr.has_i()) {
    stream << "int, value: " << attr.i();
  } else if (attr.has_s()) {
    stream << "string, value: '" << attr.s() << "'";
  } else if (attr.has_g()) {
    stream << "graph, value:\n";
    dump(attr.g(), stream, indent + 1);
    stream << nlidt(indent);
  } else if (attr.has_t()) {
    stream << "tensor, value:";
    dump(attr.t(), stream);
  } else if (attr.floats_size()) {
    stream << "floats, values: [";
    for (const auto i : c10::irange(attr.floats_size())) {
      stream << attr.floats(i) << (i == attr.floats_size() - 1 ? "" : " ");
    }
    stream << "]";
  } else if (attr.ints_size()) {
    stream << "ints, values: [";
    for (const auto i : c10::irange(attr.ints_size())) {
      stream << attr.ints(i) << (i == attr.ints_size() - 1 ? "" : " ");
    }
    stream << "]";
  } else if (attr.strings_size()) {
    stream << "strings, values: [";
    for (const auto i : c10::irange(attr.strings_size())) {
      stream << "'" << attr.strings(i) << "'"
             << (i == attr.strings_size() - 1 ? "" : " ");
    }
    stream << "]";
  } else if (attr.tensors_size()) {
    stream << "tensors, values: [";
    for (auto& t : attr.tensors()) {
      dump(t, stream);
    }
    stream << "]";
  } else if (attr.graphs_size()) {
    stream << "graphs, values: [";
    for (auto& g : attr.graphs()) {
      dump(g, stream, indent + 1);
    }
    stream << "]";
  } else {
    stream << "UNKNOWN";
  }
  stream << "}";
}

void dump(const onnx::NodeProto& node, std::ostream& stream, size_t indent) {
  stream << "Node {type: \"" << node.op_type() << "\", inputs: [";
  for (const auto i : c10::irange(node.input_size())) {
    stream << node.input(i) << (i == node.input_size() - 1 ? "" : ",");
  }
  stream << "], outputs: [";
  for (const auto i : c10::irange(node.output_size())) {
    stream << node.output(i) << (i == node.output_size() - 1 ? "" : ",");
  }
  stream << "], attributes: [";
  for (const auto i : c10::irange(node.attribute_size())) {
    dump(node.attribute(i), stream, indent + 1);
    stream << (i == node.attribute_size() - 1 ? "" : ",");
  }
  stream << "]}";
}

void dump(const onnx::GraphProto& graph, std::ostream& stream, size_t indent) {
  stream << idt(indent) << "GraphProto {" << nlidt(indent + 1) << "name: \""
         << graph.name() << "\"" << nlidt(indent + 1) << "inputs: [";
  for (const auto i : c10::irange(graph.input_size())) {
    dump(graph.input(i), stream);
    stream << (i == graph.input_size() - 1 ? "" : ",");
  }
  stream << "]" << nlidt(indent + 1) << "outputs: [";
  for (const auto i : c10::irange(graph.output_size())) {
    dump(graph.output(i), stream);
    stream << (i == graph.output_size() - 1 ? "" : ",");
  }
  stream << "]" << nlidt(indent + 1) << "value_infos: [";
  for (const auto i : c10::irange(graph.value_info_size())) {
    dump(graph.value_info(i), stream);
    stream << (i == graph.value_info_size() - 1 ? "" : ",");
  }
  stream << "]" << nlidt(indent + 1) << "initializers: [";
  for (const auto i : c10::irange(graph.initializer_size())) {
    dump(graph.initializer(i), stream);
    stream << (i == graph.initializer_size() - 1 ? "" : ",");
  }
  stream << "]" << nlidt(indent + 1) << "nodes: [" << nlidt(indent + 2);
  for (const auto i : c10::irange(graph.node_size())) {
    dump(graph.node(i), stream, indent + 2);
    if (i != graph.node_size() - 1) {
      stream << "," << nlidt(indent + 2);
    }
  }
  stream << nlidt(indent + 1) << "]\n" << idt(indent) << "}\n";
}

void dump(
    const onnx::OperatorSetIdProto& operator_set_id,
    std::ostream& stream) {
  stream << "OperatorSetIdProto { domain: " << operator_set_id.domain()
         << ", version: " << operator_set_id.version() << "}";
}

void dump(const onnx::ModelProto& model, std::ostream& stream, size_t indent) {
  stream << idt(indent) << "ModelProto {" << nlidt(indent + 1)
         << "producer_name: \"" << model.producer_name() << "\""
         << nlidt(indent + 1) << "domain: \"" << model.domain() << "\""
         << nlidt(indent + 1) << "doc_string: \"" << model.doc_string() << "\"";
  if (model.has_graph()) {
    stream << nlidt(indent + 1) << "graph:\n";
    dump(model.graph(), stream, indent + 2);
  }
  if (model.opset_import_size()) {
    stream << idt(indent + 1) << "opset_import: [";
    for (auto& opset_imp : model.opset_import()) {
      dump(opset_imp, stream);
    }
    stream << "],\n";
  }
  stream << idt(indent) << "}\n";
}

} // namespace

std::string prettyPrint(const ::ONNX_NAMESPACE::ModelProto& model) {
  std::ostringstream ss;
  dump(model, ss, 0);
  return ss.str();
}

} // namespace torch::jit
