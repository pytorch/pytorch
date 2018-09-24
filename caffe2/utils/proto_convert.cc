#include "caffe2/utils/proto_convert.h"
#include "caffe2/core/logging.h"

namespace caffe2 {

CAFFE2_EXPORT void ArgumentToAttributeProto(
    const Argument& arg,
    ::torch::AttributeProto* attr) {
  CAFFE_ENFORCE(arg.has_name());
  attr->set_name(arg.name());
  if (arg.has_f()) {
    attr->set_f(arg.f());
  } else if (arg.has_i()) {
    attr->set_i(arg.i());
  } else if (arg.has_s()) {
    attr->set_s(arg.s());
  } else if (arg.has_n()) {
    // TODO
    CAFFE_THROW("NetDef conversion is not implemented yet.");
  } else if (arg.floats_size() > 0) {
    attr->mutable_floats()->CopyFrom(arg.floats());
  } else if (arg.ints_size() > 0) {
    attr->mutable_ints()->CopyFrom(arg.ints());
  } else if (arg.strings_size() > 0) {
    attr->mutable_strings()->CopyFrom(arg.strings());
  } else if (arg.nets_size() > 0) {
    // TODO
    CAFFE_THROW("NetDefs conversion is not implemented yet.");
  }
}

CAFFE2_EXPORT void AttributeProtoToArgument(
    const ::torch::AttributeProto& attr,
    Argument* arg) {
  CAFFE_ENFORCE(attr.has_name());
  arg->set_name(attr.name());
  CAFFE_ENFORCE(attr.has_type());
  const auto type = attr.type();
  if (type ==
      ::torch::AttributeProto_AttributeType::
          AttributeProto_AttributeType_FLOAT) {
    CAFFE_ENFORCE(attr.has_f());
    arg->set_f(attr.f());
  } else if (
      type ==
      ::torch::AttributeProto_AttributeType::AttributeProto_AttributeType_INT) {
    CAFFE_ENFORCE(attr.has_i());
    arg->set_i(attr.i());
  } else if (
      type ==
      ::torch::AttributeProto_AttributeType::
          AttributeProto_AttributeType_STRING) {
    CAFFE_ENFORCE(attr.has_s());
    arg->set_s(attr.s());
  } else if (
      type ==
      ::torch::AttributeProto_AttributeType::
          AttributeProto_AttributeType_TENSOR) {
    CAFFE_THROW("Caffe2's Argument does not support tensor as attribute.");
  } else if (
      type ==
      ::torch::AttributeProto_AttributeType::
          AttributeProto_AttributeType_GRAPH) {
    // TODO
    CAFFE_THROW("GraphProto conversion is not implemented yet.");
  } else if (
      type ==
      ::torch::AttributeProto_AttributeType::
          AttributeProto_AttributeType_FLOATS) {
    arg->mutable_floats()->CopyFrom(attr.floats());
  } else if (
      type ==
      ::torch::AttributeProto_AttributeType::
          AttributeProto_AttributeType_INTS) {
    arg->mutable_ints()->CopyFrom(attr.ints());
  } else if (
      type ==
      ::torch::AttributeProto_AttributeType::
          AttributeProto_AttributeType_STRINGS) {
    arg->mutable_strings()->CopyFrom(attr.strings());
  } else if (
      type ==
      ::torch::AttributeProto_AttributeType::
          AttributeProto_AttributeType_TENSORS) {
    CAFFE_THROW("Caffe2's Argument does not support tensors as attribute.");
  } else if (
      type ==
      ::torch::AttributeProto_AttributeType::
          AttributeProto_AttributeType_GRAPHS) {
    // TODO
    CAFFE_THROW("GraphProtos conversion is not implemented yet.");
  } else {
    CAFFE_THROW("Unknow Attribute type.");
  }
}

CAFFE2_EXPORT void OperatorDefToNodeProto(
    const OperatorDef& def,
    ::torch::NodeProto* node) {
  node->mutable_input()->CopyFrom(def.input());
  node->mutable_output()->CopyFrom(def.output());
  if (def.has_name()) {
    node->set_name(def.name());
  }
  CAFFE_ENFORCE(def.has_type());
  node->set_op_type(def.type());
  for (int i = 0; i < def.arg_size(); ++i) {
    auto attr = node->add_attribute();
    ArgumentToAttributeProto(def.arg(i), attr);
  }
  if (def.has_device_option()) {
    node->mutable_device_option()->CopyFrom(def.device_option());
  }
  if (def.has_engine()) {
    auto attr = node->add_annotations();
    attr->set_name("engine");
    attr->set_type(::torch::AttributeProto_AttributeType::
                       AttributeProto_AttributeType_STRING);
    attr->set_s(def.engine());
  }
  if (def.control_input_size() > 0) {
    auto attr = node->add_annotations();
    attr->set_name("control_input");
    attr->set_type(::torch::AttributeProto_AttributeType::
                       AttributeProto_AttributeType_STRINGS);
    attr->mutable_strings()->CopyFrom(def.control_input());
  }
  if (def.has_is_gradient_op()) {
    auto attr = node->add_annotations();
    attr->set_name("is_gradient_op");
    attr->set_type(::torch::AttributeProto_AttributeType::
                       AttributeProto_AttributeType_INT);
    if (def.is_gradient_op()) {
      attr->set_i(1);
    } else {
      attr->set_i(0);
    }
  }
  if (def.has_debug_info()) {
    node->set_doc_string(def.debug_info());
  }
}

CAFFE2_EXPORT void NodeProtoToOperatorDef(
    const ::torch::NodeProto& node,
    OperatorDef* def) {
  def->mutable_input()->CopyFrom(node.input());
  def->mutable_output()->CopyFrom(node.output());
  if (node.has_name()) {
    def->set_name(node.name());
  }

  CAFFE_ENFORCE(node.has_op_type());
  def->set_type(node.op_type());
  for (int i = 0; i < node.attribute_size(); ++i) {
    auto arg = def->add_arg();
    AttributeProtoToArgument(node.attribute(i), arg);
  }
  if (node.has_doc_string()) {
    def->set_debug_info(node.doc_string());
  }
  for (int i = 0; i < node.annotations_size(); ++i) {
    const auto& attr = node.annotations(i);
    CAFFE_ENFORCE(attr.has_name());
    if (attr.name() == "engine") {
      CAFFE_ENFORCE(attr.has_s());
      def->set_engine(attr.s());
    } else if (attr.name() == "control_input") {
      def->mutable_control_input()->CopyFrom(attr.strings());
    } else if (attr.name() == "is_gradient_op") {
      CAFFE_ENFORCE(attr.has_i());
      if (i == 0) {
        def->set_is_gradient_op(false);
      } else {
        def->set_is_gradient_op(true);
      }
    }
    auto arg = def->add_arg();
    AttributeProtoToArgument(node.annotations(i), arg);
  }
  if (node.has_device_option()) {
    def->mutable_device_option()->CopyFrom(node.device_option());
  }
}

} // namespace caffe2
