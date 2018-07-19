#include "torch/csrc/onnx/onnx.h"

namespace torch { namespace onnx {

template <>
bool micropb_encode<std::string, nullptr>(pb_ostream_t *stream, std::string* arg) {
  return pb_encode_string(stream, reinterpret_cast<const pb_byte_t *>(arg->c_str()), arg->size());
}
// NB: Overloads don't work so great for signed variables.  Hope this doesn't
// come up!
template <>
bool micropb_encode<int64_t, nullptr>(pb_ostream_t *stream, int64_t* arg) {
  // Yes, this looks dodgy, and yes, this is what the docs say to do:
  // https://jpa.kapsi.fi/nanopb/docs/reference.html#pb-encode-varint
  return pb_encode_varint(stream, *reinterpret_cast<uint64_t*>(arg));
}
template <>
bool micropb_encode<float, nullptr>(pb_ostream_t *stream, float* arg) {
  return pb_encode_fixed32(stream, static_cast<void*>(arg));
}
template <>
bool micropb_encode<double, nullptr>(pb_ostream_t *stream, double* arg) {
  return pb_encode_fixed64(stream, static_cast<void*>(arg));
}

template <>
bool micropb_encode<Dimension, nullptr>(pb_ostream_t *stream, Dimension* arg) {
  return pb_encode_submessage(stream, onnx_TensorShapeProto_Dimension_fields,
                              static_cast<void*>(arg));
}

// TODO: I'm not entirely sure why this can't be in the header...
bool micropb_callback_string_from_tensor(pb_ostream_t *stream, const pb_field_t *field, void * const *arg) {
  at::Tensor* t = static_cast<at::Tensor*>(*arg);
  JIT_ASSERT(t->is_contiguous());
  // Packed array format!
  pb_encode_tag_for_field(stream, field);
  pb_encode_string(stream, (pb_byte_t*)(t->data_ptr()),  t->type().elementSizeInBytes()*t->numel());

  return true;
}

GraphProto* AttributeProto::add_graphs() {
  auto ptr = new GraphProto();
  graphs.emplace_back(ptr);
  return ptr;
}

constexpr char indent_char = ' ';
constexpr size_t indent_multiplier = 2;

std::string idt(size_t indent) {
  return std::string(indent * indent_multiplier, indent_char);
}

std::string nlidt(size_t indent) {
  return std::string("\n") + idt(indent);
}

void TensorProto::dump(std::ostream& stream, size_t indent) {
  stream << "TensorProto shape: [";
  for (size_t i = 0; i < dims.size(); ++i) {
    stream << *dims[i] << (i == dims.size() - 1 ? "" : " ");
  }
  stream << "]";
}

void TensorShapeProto::dump(std::ostream& stream, size_t indent) {
  for (size_t i=0; i < dims.size(); ++i) {
    auto &dim = dims[i];
    if (dim->has_dim_value) {
      stream << dim->dim_value;
    } else {
      stream << "?";
    }
    stream << (i == dims.size() - 1 ? "" : " ");
  }
}

void TypeProtoTensor::dump(std::ostream& stream, size_t indent) {
  stream << "Tensor dims: ";
  shape->dump(stream);
}

void TypeProto::dump(std::ostream& stream, size_t indent) {
  tensor_type->dump(stream);
}

void ValueInfoProto::dump(std::ostream& stream, size_t indent) {
  stream << "{name: \"" << name
         << "\", type:";
  type->dump(stream);
  stream << "}";
}

void AttributeProto::dump(std::ostream& stream, size_t indent) {
  stream << "{ name: '" << name << "', type: ";
  if (proto.has_f) {
    stream << "float, value: " << proto.f;
  } else if (proto.has_i) {
    stream << "int, value: " << proto.i;
  } else if (s.length()) {
    stream << "string, value: '" << s << "'";
  } else if (g) {
    stream << "graph, value:\n";
    g->dump(stream, indent+1);
    stream << nlidt(indent);
  } else if (t) {
    stream << "tensor, value:";
    t->dump(stream, indent+1);
  } else if (floats.size()) {
    stream << "floats, values: [";
    for (size_t i=0; i < floats.size(); ++i)
      stream << *floats[i] << (i == floats.size() - 1 ? "" : " ");
    stream << "]";
  } else if (ints.size()) {
    stream << "ints, values: [";
    for (size_t i=0; i < ints.size(); ++i)
      stream << *ints[i] << (i == ints.size() - 1 ? "" : " ");
    stream << "]";
  } else if (strings.size()) {
    stream << "strings, values: [";
    for (size_t i=0; i < strings.size(); ++i)
      stream << "'" << *strings[i] << "'" << (i == strings.size() - 1 ? "" : " ");
    stream << "]";
  } else if (tensors.size()) {
    stream << "tensors, values: [";
    for (auto& t : tensors) {
      t->dump(stream, indent+1);
    }
    stream << "]";
  } else if (graphs.size()) {
    stream << "graphs, values: [";
    for (auto& g : graphs) {
      g->dump(stream, indent+1);
    }
    stream << "]";
  } else {
    stream << "UNKNOWN";
  }
  stream << "}";
}

void NodeProto::dump(std::ostream& stream, size_t indent) {
  stream << "Node {type: \"" << op_type << "\", inputs: [";
  for (size_t i=0; i < inputs.size(); ++i) {
    stream << *inputs[i] << (i == inputs.size() - 1 ? "" : ",");
  }
  stream << "], outputs: [";
  for (size_t i=0; i < outputs.size(); ++i) {
    stream << *outputs[i] << (i == outputs.size() - 1 ? "" : ",");
  }
  stream << "], attributes: [";
  for (size_t i=0; i < attributes.size(); ++i) {
    attributes[i]->dump(stream, indent+1);
    stream << (i == attributes.size() - 1 ? "" : ",");
  }
  stream << "]}";
}

void GraphProto::dump(std::ostream& stream, size_t indent) {
  stream << idt(indent) << "GraphProto {" << nlidt(indent+1)
         << "name: \"" << name << "\"" << nlidt(indent+1)
         << "inputs: [";
  for (size_t i=0; i < inputs.size(); ++i) {
    inputs[i]->dump(stream, indent+2);
    stream << (i == inputs.size() - 1 ? "" : ",");
  }
  stream << "]" << nlidt(indent+1)
         << "outputs: [";
  for (size_t i=0; i < outputs.size(); ++i) {
    outputs[i]->dump(stream, indent+2);
    stream << (i == outputs.size() - 1 ? "" : ",");
  }
  stream << "]" << nlidt(indent+1)
         << "initializers: [";
  for (size_t i=0; i < initializers.size(); ++i) {
    initializers[i]->dump(stream, indent+2);
    stream << (i == initializers.size() - 1 ? "" : ",");
  }
  stream << "]" << nlidt(indent+1)
         << "nodes: [" << nlidt(indent+2);
  for (size_t i=0; i < nodes.size(); ++i) {
    nodes[i]->dump(stream, indent+2);
    if (i != nodes.size() - 1) stream << "," << nlidt(indent+2);
  }
  stream << nlidt(indent+1) << "]\n" << idt(indent) << "}\n";
}

void OperatorSetIdProto::dump(std::ostream& stream, size_t indent) {
  stream << "OperatorSetIdProto { domain: " << domain << "}";
}

void ModelProto::dump(std::ostream& stream, size_t indent) {
  stream << idt(indent)
         << "ModelProto {" << nlidt(indent+1)
         << "producer_name: \"" << producer_name << "\"" << nlidt(indent+1)
         << "domain: \"" << domain << "\"" << nlidt(indent+1)
         << "doc_string: \"" << doc_string << "\"";
  if (graph) {
    stream << nlidt(indent+1) << "graph:\n";
    graph->dump(stream, indent+2);
  }
  if (opset_import.size()) {
    stream << idt(indent+1) << "opset_import: [";
    for (auto &opset_imp : opset_import) {
      opset_imp->dump(stream, indent+2);
    }
    stream << "],\n";
  }
  stream << idt(indent) << "}\n";
}

}} // namespace onnx
