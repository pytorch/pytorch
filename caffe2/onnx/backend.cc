#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/onnx/backend.h"
#include "caffe2/onnx/device.h"
#include "caffe2/onnx/helper.h"
#include "caffe2/utils/map_utils.h"
#include "caffe2/utils/proto_utils.h"

#if !CAFFE2_MOBILE
#include "onnx/checker.h"
#include "onnx/optimizer/optimize.h"
#endif

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"

#include <cmath>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace caffe2 {
namespace onnx {

namespace {

constexpr static int kKnownOpsetVersion = 6;

bool AlmostEqual(double a, double b) {
  constexpr static double kEps = 1e-15;
  return (fabs(a - b) < kEps);
}

template <class T>
bool TryConvertingTensorRawValues(
    const TensorProto& onnx_tensor,
    ::google::protobuf::RepeatedField<T>* field) {
  if (!onnx_tensor.has_raw_data()) {
    return false;
  }

  size_t raw_size = onnx_tensor.raw_data().size();
  CAFFE_ENFORCE_EQ(raw_size % sizeof(T), 0);

  size_t num_elements = raw_size / sizeof(T);
  const void* src_ptr = static_cast<const void*>(onnx_tensor.raw_data().data());
  field->Resize(num_elements, 0);
  void* target_ptr = static_cast<void*>(field->mutable_data());
  memcpy(target_ptr, src_ptr, raw_size);

  return true;
}

bool IsOperator(const std::string& op_type) {
  // pull in all the operators upon first invocation
  // Intentional leaky
  static std::set<std::string>* ops_ =
      new std::set<std::string>(caffe2::GetRegisteredOperators());
  return ops_->count(caffe2::OpRegistryKey(op_type, "DEFAULT"));
}

caffe2::DeviceOption GetDeviceOption(const Device& onnx_device) {
  static const std::unordered_map<DeviceType, caffe2::DeviceType> m = {
      {DeviceType::CPU, caffe2::DeviceType::CPU},
      {DeviceType::CUDA, caffe2::DeviceType::CUDA}};
  caffe2::DeviceOption d;
  d.set_device_type(static_cast<int32_t>(m.at(onnx_device.type)));
  d.set_cuda_gpu_id(onnx_device.device_id);
  return d;
}

#if !CAFFE2_MOBILE
ModelProto OptimizeOnnx(const ModelProto& input, bool init) {
  std::vector<std::string> passes{"fuse_consecutive_transposes",
                                  "eliminate_nop_transpose",
                                  "fuse_transpose_into_gemm"};

  if (init) {
    passes.emplace_back("split_init");
  } else {
    passes.emplace_back("split_predict");
  }
  return ::ONNX_NAMESPACE::optimization::Optimize(input, passes);
}
#endif

template <class T, class U>
U LookUpWithDefault(
    const std::unordered_map<T, U>& map,
    const T& key,
    const U& default_value) {
  const auto it = map.find(key);
  if (it == map.end()) {
    return default_value;
  } else {
    return it->second;
  }
}

void UpdateNames(const caffe2::OperatorDef& op) {
  for (const auto& n : op.input()) {
    DummyName::AddName(n);
  }
  for (const auto& n : op.output()) {
    DummyName::AddName(n);
  }
}

void BuildOperator(
    caffe2::OperatorDef* c2_op,
    const std::string& op_type,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::vector<caffe2::Argument>& args) {
  c2_op->set_name("");
  c2_op->set_type(op_type);
  for (const auto& input : inputs) {
    c2_op->add_input(input);
  }
  for (const auto& output : outputs) {
    c2_op->add_output(output);
  }
  for (const auto& arg : args) {
    auto* tmp = c2_op->add_arg();
    tmp->CopyFrom(arg);
  }
}

void BuildOperator(
    caffe2::OperatorDef* c2_op,
    const std::string& op_type,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs) {
  std::vector<caffe2::Argument> empty;
  BuildOperator(c2_op, op_type, inputs, outputs, empty);
}

void CopyOnnxAttrValueToCaffe2Arg(
    caffe2::Argument* arg,
    const AttributeProto& attr) {
  if (attr.has_f()) {
    arg->set_f(attr.f());
  } else if (attr.has_i()) {
    arg->set_i(attr.i());
  } else if (attr.has_s()) {
    arg->set_s(attr.s());
  } else if (attr.has_t()) {
    // For proto, we convert it to serialized string
    std::string buffer;
    attr.t().SerializeToString(&buffer);
    arg->set_s(buffer);
  } else if (attr.floats_size()) {
    arg->mutable_floats()->CopyFrom(attr.floats());
  } else if (attr.ints_size()) {
    arg->mutable_ints()->CopyFrom(attr.ints());
  } else if (attr.strings_size()) {
    arg->mutable_strings()->CopyFrom(attr.strings());
  } else {
    CAFFE_THROW("Unsupported ONNX attribute: ", attr.name());
  }
}
} // namespace

OnnxAttributes::OnnxAttributes(const NodeProto& node) {
  for (const auto& attr : node.attribute()) {
    onnx_attrs_.emplace(attr.name(), &attr);
  }
}

template <>
int64_t OnnxAttributes::get(const std::string& key) const {
  int64_t value = 0;
  const auto it = onnx_attrs_.find(key);
  if (it != onnx_attrs_.end()) {
    const AttributeProto& attr = *it->second;
    value = attr.i();
  }
  return value;
}

template <>
float OnnxAttributes::get(const std::string& key) const {
  float value = 0.0;
  const auto it = onnx_attrs_.find(key);
  if (it != onnx_attrs_.end()) {
    const AttributeProto& attr = *it->second;
    value = attr.f();
  }
  return value;
}

template <>
::google::protobuf::RepeatedPtrField<std::string> OnnxAttributes::get(
    const std::string& key) const {
  ::google::protobuf::RepeatedPtrField<std::string> value;
  const auto it = onnx_attrs_.find(key);
  if (it != onnx_attrs_.end()) {
    const AttributeProto& attr = *it->second;
    value.CopyFrom(attr.strings());
  }
  return value;
}

template <>
::google::protobuf::RepeatedField<::google::protobuf::int64>
OnnxAttributes::get(const std::string& key) const {
  ::google::protobuf::RepeatedField<::google::protobuf::int64> value;
  const auto it = onnx_attrs_.find(key);
  if (it != onnx_attrs_.end()) {
    const AttributeProto& attr = *it->second;
    value.CopyFrom(attr.ints());
  }
  return value;
}

template <>
const TensorProto* OnnxAttributes::get(const std::string& key) const {
  const TensorProto* value = nullptr;
  const auto it = onnx_attrs_.find(key);
  if (it != onnx_attrs_.end()) {
    const AttributeProto& attr = *it->second;
    value = &attr.t();
  }
  return value;
}

::google::protobuf::RepeatedPtrField<caffe2::Argument>
OnnxAttributes::OnnxAttrToCaffe2Arg(
    std::function<std::string(const std::string&)> mapper) const {
  ::google::protobuf::RepeatedPtrField<caffe2::Argument> args;
  for (const auto& kv : onnx_attrs_) {
    // If the attribute was rewritten, we use it instead. Note that the
    // rewritten attibute still has the unmapped name
    const auto& attr = rewritten_onnx_attrs_.count(kv.first)
        ? rewritten_onnx_attrs_.at(kv.first)
        : (*kv.second);
    auto* arg = args.Add();
    arg->set_name(mapper(attr.name()));
    CopyOnnxAttrValueToCaffe2Arg(arg, attr);
  }
  for (const auto& kv : rewritten_onnx_attrs_) {
    // If rewritten attribute doesn't appear in the original attributes, this is
    // a newlly added one and we need to add this to argument too
    if (!onnx_attrs_.count(kv.first)) {
      const auto& attr = kv.second;
      auto* arg = args.Add();
      arg->set_name(mapper(attr.name()));
      CopyOnnxAttrValueToCaffe2Arg(arg, attr);
    }
  }

  return args;
}

const std::unordered_map<std::string, int>&
Caffe2Backend::get_broken_operators() const {
  const static std::unordered_map<std::string, int> kBrokenOperators{};
  return kBrokenOperators;
}

// Temporary hack for RNN related operators, as we don't have C++ interface in
// C2 to build those operators yet
const std::unordered_set<std::string>& Caffe2Backend::get_rnn_operators()
    const {
  const static std::unordered_set<std::string> kRNNOperators{
      "LSTM", "GRU", "RNN"};
  return kRNNOperators;
}

// Operators that are different between Caffe2 and
// ONNX but only in their name.
// In most cases, this should be empty - as the effort of ONNX is
// to unify the operator definitions.
const std::unordered_map<std::string, std::string>&
Caffe2Backend::get_renamed_operators() const {
  const static std::unordered_map<std::string, std::string> kRenamedOperators{
      {"Caffe2ConvTranspose", "ConvTranspose"},
      {"GlobalMaxPool", "MaxPool"},
      {"GlobalAveragePool", "AveragePool"},
      {"Pad", "PadImage"},
      {"Neg", "Negative"},
      {"BatchNormalization", "SpatialBN"},
      {"InstanceNormalization", "InstanceNorm"},
      {"MatMul", "BatchMatMul"},
      {"Upsample", "ResizeNearest"},
      {"Identity", "Copy"},
      {"InstanceNormalization", "InstanceNorm"},
      {"Equal", "EQ"},
      {"Less", "LT"},
      {"Greater", "GT"},
      {"Unsqueeze", "ExpandDims"}};
  return kRenamedOperators;
}

const std::unordered_map<std::string, std::string>&
Caffe2Backend::get_renamed_attrs() const {
  const static std::unordered_map<std::string, std::string> kRenamedAttrs{
      {"kernel_shape", "kernels"}};
  return kRenamedAttrs;
}

const std::
    unordered_map<std::string, std::unordered_map<std::string, std::string>>&
    Caffe2Backend::get_per_op_renamed_attrs() const {
  const static std::
      unordered_map<std::string, std::unordered_map<std::string, std::string>>
          kPerOpRenamedAttrs = {{"Squeeze", {{"axes", "dims"}}},
                                {"Unsqueeze", {{"axes", "dims"}}},
                                {"Transpose", {{"perm", "axes"}}},
                                {"Upsample", {{"mode", ""}}},
                                {"ConvTranspose", {{"output_padding", "adjs"}}},
                                {"Selu", {{"gamma", "scale"}}}};

  return kPerOpRenamedAttrs;
}

// operators whose behavior is different beyond renaming
// the value is an attribute of this class that is a
// function from ToffeIR node_def to caffe2 op_def
const std::unordered_map<std::string, Caffe2Backend::SpecialOpConverter>&
Caffe2Backend::get_special_operators() const {
  const static std::
      unordered_map<std::string, Caffe2Backend::SpecialOpConverter>
          kSpecialOperators = {
              {"Constant", &Caffe2Backend::CreateConstant},
              {"Conv", &Caffe2Backend::CreateConvPoolOpBase},
              {"AveragePool", &Caffe2Backend::CreateConvPoolOpBase},
              {"GlobalAveragePool", &Caffe2Backend::CreateConvPoolOpBase},
              {"GlobalMaxPool", &Caffe2Backend::CreateConvPoolOpBase},
              {"MaxPool", &Caffe2Backend::CreateConvPoolOpBase},
              {"Reshape", &Caffe2Backend::CreateReshape},
              {"Gather", &Caffe2Backend::CreateGather},
              {"Gemm", &Caffe2Backend::CreateGemm},
              {"Pad", &Caffe2Backend::CreatePad},
              {"Concat", &Caffe2Backend::CreateConcat},
              {"LogSoftmax", &Caffe2Backend::CreateLogSoftmax},
              {"Slice", &Caffe2Backend::CreateSlice},
              {"Reciprocal", &Caffe2Backend::CreateReciprocal},
              {"BatchNormalization", &Caffe2Backend::CreateBatchNormalization},
              {"MatMul", &Caffe2Backend::CreateMatMul}};
  return kSpecialOperators;
}

//============================
// Special Operator Converters
//============================

Caffe2Ops Caffe2Backend::CreateConstant(
    OnnxNode* onnx_node,
    int opset_version) {
  CAFFE_ENFORCE_EQ(onnx_node->node.output_size(), 1);

  Caffe2Ops ret;
  auto* c2_op = ret.ops.Add();
  const auto* value = onnx_node->attributes.get<const TensorProto*>("value");
  BuildTensorFillingOp(c2_op, *value, onnx_node->node.output(0));

  return ret;
}

//  Note [Caffe2 ConvPoolOpBase]
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//  To understand what is going on here, we have to talk a little bit about
//  Caffe2's internals.
//
//  First, it's important to know that all of Caffe2's pooling and convolution
//  operators inherit from "ConvPoolOpBase", which is an abstract class that
//  defines all of the attributes (kernels, dilations, strides, etc) which one
//  sees on these operators.  Unfortunately, Caffe2's documentation generator
//  doesn't know how to handle cases like this, so for example, if you look at
//  the docs for MaxPool at
//  <https://caffe2.ai/docs/operators-catalogue.html#maxpool> you won't see any
//  of the attributes.  You have to go source diving to find the information; in
//  particular, you want to look at:
//  https://github.com/caffe2/caffe2/blob/master/caffe2/operators/conv_pool_op_base.h
//  This class handles *global* pooling as well.
//
//  Second, it's important to know what Caffe2 expects for padding, which can
//  be somewhat difficult to understand from the code because Caffe2 handles
//  both singular/pluralized spellings of padding, and there is also legacy
//  padding business.  The short version of the story is that, for NON-legacy
//  padding (which is what we want to output), padding is expected to be
//  *twice* the size of kernels.  So if you have a 2D convolution, Caffe2
//  will accept two values in 'kernels', but FOUR values in 'pads';
//  furthermore, this is *mandatory.*
//
//  Finally, ConvPoolOpBase is not the only class of it's kind; there is
//  be tricked by the fact that Conv and ConvTranspose have similar
//  parameters; they exercise different codepaths and need to be handled
//  differently.
Caffe2Ops Caffe2Backend::CreateConvPoolOpBase(
    OnnxNode* onnx_node,
    int opset_version) {
  const auto& node = onnx_node->node;
  auto& attributes = onnx_node->attributes;
  if (node.op_type().find("Global") == 0) {
    auto* attr = attributes.AddRewrittenAttibute("global_pooling");
    attr->set_i(1);
  }

  if (attributes.HasAttribute("kernel_shape") &&
      attributes.HasAttribute("pads")) {
    auto kernel_shape =
        attributes
            .get<::google::protobuf::RepeatedField<::google::protobuf::int64>>(
                "kernel_shape");
    auto pads =
        attributes
            .get<::google::protobuf::RepeatedField<::google::protobuf::int64>>(
                "pads");
    if (kernel_shape.size() == pads.size()) {
      // Caffe2 requires pads to be twice the size of kernels.
      auto* attr = attributes.AddRewrittenAttibute("pads");
      attr->mutable_ints()->CopyFrom(pads);
      attr->mutable_ints()->MergeFrom(pads);
    }
  }

  return CommonOnnxNodeToCaffe2Ops(onnx_node, opset_version);
}

Caffe2Ops Caffe2Backend::CreateReshape(OnnxNode* onnx_node, int opset_version) {
  auto c2_op = CommonOnnxNodeToCaffe2Ops(onnx_node, opset_version);
  CAFFE_ENFORCE_EQ(c2_op.ops.size(), 1);
  auto* op = c2_op.ops.Mutable(0);
  op->add_output(DummyName::NewDummyName());

  return c2_op;
}

Caffe2Ops Caffe2Backend::CreateReciprocal(
    OnnxNode* onnx_node,
    int opset_version) {
  const auto& node = onnx_node->node;
  if (node.input_size() != 1 || node.output_size() != 1) {
    CAFFE_THROW("Caffe2 Reciprocal should have 1 input and 1 output");
  }

  Caffe2Ops ret;
  auto* c2_op = ret.ops.Add();

  caffe2::Argument exponent;
  exponent.set_name("exponent");
  exponent.set_f(-1.0);
  BuildOperator(c2_op, "Pow", {node.input(0)}, {node.output(0)}, {exponent});
  return ret;
}

Caffe2Ops Caffe2Backend::CreateGather(OnnxNode* onnx_node, int opset_version) {
  const auto& node = onnx_node->node;
  if (node.input_size() < 2 || node.output_size() < 1) {
    CAFFE_THROW("Caffe2 Gather should have 2 inputs and 1 output");
  }

  Caffe2Ops ret;
  auto* c2_op = ret.ops.Add();

  std::vector<std::string> inputs;
  inputs.emplace_back(node.input(0));
  inputs.emplace_back(node.input(1));
  std::vector<std::string> outputs;
  outputs.emplace_back(node.output(0));

  auto axis = onnx_node->attributes.get<int64_t>("axis", 0L);
  if (axis == 0) {
    BuildOperator(c2_op, "Gather", inputs, outputs);
  } else if (axis == 1) {
    BuildOperator(c2_op, "BatchGather", inputs, outputs);
  } else {
    CAFFE_THROW(
        "Caffe2 only supports Gather with axis being 1 or 2, ",
        "whereas axis is ",
        axis);
  }

  return ret;
}

Caffe2Ops Caffe2Backend::CreateGemm(OnnxNode* onnx_node, int opset_version) {
  const auto& node = onnx_node->node;
  if (node.input_size() < 3 || node.output_size() < 1) {
    CAFFE_THROW("Caffe2 Gemm should have 3 inputs and 1 output");
  }

  Caffe2Ops ret;
  auto input_a = node.input(0);
  auto input_b = node.input(1);
  auto input_c = node.input(2);
  auto output = node.output(0);

  auto alpha = onnx_node->attributes.get<float>("alpha", 1.0);
  auto beta = onnx_node->attributes.get<float>("beta", 1.0);
  if (!AlmostEqual(alpha, 1)) {
    auto scaled_a = DummyName::NewDummyName();
    caffe2::Argument scale;
    scale.set_name("scale");
    scale.set_f(alpha);

    auto* c2_op = ret.ops.Add();
    BuildOperator(c2_op, "Scale", {input_a}, {scaled_a}, {scale});
    input_a = scaled_a;
  }
  if (!AlmostEqual(beta, 1)) {
    auto scaled_c = DummyName::NewDummyName();
    caffe2::Argument scale;
    scale.set_name("scale");
    scale.set_f(beta);

    auto* c2_op = ret.ops.Add();
    BuildOperator(c2_op, "Scale", {input_c}, {scaled_c}, {scale});
    input_c = scaled_c;
  }

  auto trans_a = onnx_node->attributes.get<int64_t>("transA", 0L);
  auto trans_b = onnx_node->attributes.get<int64_t>("transB", 0L);
  auto broadcast = onnx_node->attributes.get<int64_t>("broadcast", 0L);
  if ((!trans_a) && trans_b && broadcast) {
    auto* c2_op = ret.ops.Add();
    BuildOperator(c2_op, "FC", {input_a, input_b, input_c}, {output});
  } else {
    auto ab = DummyName::NewDummyName();
    caffe2::Argument arg_trans_a;
    arg_trans_a.set_name("trans_a");
    arg_trans_a.set_i(trans_a);
    caffe2::Argument arg_trans_b;
    arg_trans_b.set_name("trans_b");
    arg_trans_b.set_i(trans_b);
    caffe2::Argument arg_broadcast;
    arg_broadcast.set_name("broadcast");
    arg_broadcast.set_i(broadcast);

    auto* c2_op = ret.ops.Add();
    BuildOperator(
        c2_op, "MatMul", {input_a, input_b}, {ab}, {arg_trans_a, arg_trans_b});
    c2_op = ret.ops.Add();
    BuildOperator(c2_op, "Add", {ab, input_c}, {output}, {arg_broadcast});
  }

  return ret;
}

Caffe2Ops Caffe2Backend::CreatePad(OnnxNode* onnx_node, int opset_version) {
  const auto& node = onnx_node->node;
  auto& attributes = onnx_node->attributes;
  ::google::protobuf::RepeatedField<::google::protobuf::int64> pads;
  std::string pad_name = opset_version < 2 ? "paddings" : "pads";
  pads = attributes
             .get<::google::protobuf::RepeatedField<::google::protobuf::int64>>(
                 pad_name);
  std::string str;
  std::stringstream ss;
  ss << "[";
  for (const auto& i : pads) {
    ss << i << ", ";
  }
  ss << "]";
  str = ss.str();

  // Guard the invalid (negative) pads attribute.
  for (const auto i : pads) {
    if (i < 0) {
      CAFFE_THROW("ONNX does not support negative pads in Pad, but get ", str);
    }
  }

  // first two dim is for batch and channel. Note that now all the values are
  // non-negative
  if (!(pads.size() == 8 &&
        (pads.Get(0) + pads.Get(1) + pads.Get(4) + pads.Get(5) == 0))) {
    CAFFE_THROW(
        "Caffe2 only supports padding 2D Tensor, whereas padding is ", str);
  }

  // rewrite the padding info
  auto* attr = attributes.AddRewrittenAttibute(pad_name);
  attr->add_ints(pads.Get(2));
  attr->add_ints(pads.Get(3));
  attr->add_ints(pads.Get(6));
  attr->add_ints(pads.Get(7));

  return CommonOnnxNodeToCaffe2Ops(onnx_node, opset_version);
}

// TODO: Caffe2 Concat has an extra output. It should be only
// used when doing training, so we should change Caffe2 to allow
// 1 output.
Caffe2Ops Caffe2Backend::CreateConcat(OnnxNode* onnx_node, int opset_version) {
  auto c2_op = CommonOnnxNodeToCaffe2Ops(onnx_node, opset_version);
  CAFFE_ENFORCE_EQ(c2_op.ops.size(), 1);
  auto* op = c2_op.ops.Mutable(0);
  op->add_output(DummyName::NewDummyName());

  return c2_op;
}

Caffe2Ops Caffe2Backend::CreateLogSoftmax(
    OnnxNode* onnx_node,
    int opset_version) {
  const auto& node = onnx_node->node;
  if (node.input_size() < 1 || node.output_size() < 1) {
    CAFFE_THROW("LogSoftmax should have 1 input and 1 output");
  }
  auto axis = onnx_node->attributes.get<int64_t>("axis", 1L);
  caffe2::Argument arg_axis;
  arg_axis.set_name("axis");
  arg_axis.set_i(axis);
  auto softmax_a = DummyName::NewDummyName();

  Caffe2Ops ret;
  auto* c2_op = ret.ops.Add();
  BuildOperator(c2_op, "Softmax", {node.input(0)}, {softmax_a}, {arg_axis});
  c2_op = ret.ops.Add();
  BuildOperator(c2_op, "Log", {softmax_a}, {node.output(0)});

  return ret;
}

Caffe2Ops Caffe2Backend::CreateSlice(OnnxNode* onnx_node, int opset_version) {
  auto op_tmp = CommonOnnxNodeToCaffe2Ops(onnx_node, opset_version);
  CAFFE_ENFORCE_EQ(op_tmp.ops.size(), 1);
  auto* op = op_tmp.ops.Mutable(0);
  std::unordered_map<std::string, caffe2::Argument*> args;
  for (auto& arg : *op->mutable_arg()) {
    args.emplace(arg.name(), &arg);
  }

  caffe2::Argument starts_vals;
  starts_vals.set_name("values");
  auto pos = args.find("starts");
  if (pos != args.end()) {
    for (auto i : pos->second->ints()) {
      starts_vals.add_ints(i);
    }
    args.erase(pos);
  }

  caffe2::Argument ends_vals;
  ends_vals.set_name("values");
  pos = args.find("ends");
  if (pos != args.end()) {
    for (auto i : pos->second->ints()) {
      ends_vals.add_ints(i < 0 ? i - 1 : i);
    }
    args.erase(pos);
  }

  caffe2::Argument axes_vals;
  axes_vals.set_name("values");
  pos = args.find("axes");
  if (pos != args.end()) {
    for (auto i : pos->second->ints()) {
      axes_vals.add_ints(i);
    }
    args.erase(pos);
  } else {
    auto ndim = starts_vals.ints_size();
    for (int64_t i = 0; i < ndim; ++i) {
      axes_vals.add_ints(i);
    }
  }

  CAFFE_ENFORCE_GE(op->input_size(), 1);
  auto data = op->input(0);
  auto shape_tensor = DummyName::NewDummyName();
  Caffe2Ops ret;

  auto* c2_op = ret.ops.Add();
  BuildOperator(c2_op, "Shape", {data}, {shape_tensor});

  auto axes_tensor = DummyName::NewDummyName();
  c2_op = ret.ops.Add();
  {
    caffe2::Argument shape;
    shape.set_name("shape");
    shape.add_ints(axes_vals.ints_size());
    BuildOperator(
        c2_op, "GivenTensorIntFill", {}, {axes_tensor}, {shape, axes_vals});
  }

  auto starts_vals_tensor = DummyName::NewDummyName();
  auto starts_tensor = DummyName::NewDummyName();
  auto casted_starts_tensor = DummyName::NewDummyName();
  c2_op = ret.ops.Add();
  {
    caffe2::Argument shape_starts;
    shape_starts.set_name("shape");
    shape_starts.add_ints(starts_vals.ints_size());
    BuildOperator(
        c2_op,
        "GivenTensorInt64Fill",
        {},
        {starts_vals_tensor},
        {shape_starts, starts_vals});
  }

  caffe2::Argument dtype;
  dtype.set_name("dtype");
  dtype.set_i(static_cast<int64_t>(caffe2::TensorProto::INT64));
  caffe2::Argument constant;
  constant.set_name("value");
  constant.set_i(0);
  c2_op = ret.ops.Add();
  BuildOperator(
      c2_op,
      "ConstantFill",
      {shape_tensor},
      {starts_tensor},
      {dtype, constant});
  c2_op = ret.ops.Add();
  BuildOperator(
      c2_op,
      "ScatterAssign",
      {starts_tensor, axes_tensor, starts_vals_tensor},
      {starts_tensor});
  // Slice only accepts starts as int
  caffe2::Argument to;
  to.set_name("to");
  to.set_i(static_cast<int64_t>(caffe2::TensorProto::INT32));
  c2_op = ret.ops.Add();
  BuildOperator(c2_op, "Cast", {starts_tensor}, {casted_starts_tensor}, {to});

  auto ends_vals_tensor = DummyName::NewDummyName();
  auto ends_tensor = DummyName::NewDummyName();
  auto casted_ends_tensor = DummyName::NewDummyName();
  c2_op = ret.ops.Add();
  {
    caffe2::Argument shape_ends;
    shape_ends.set_name("shape");
    shape_ends.add_ints(ends_vals.ints_size());
    BuildOperator(
        c2_op,
        "GivenTensorInt64Fill",
        {},
        {ends_vals_tensor},
        {shape_ends, ends_vals});
  }

  constant.set_i(-1);
  c2_op = ret.ops.Add();
  BuildOperator(
      c2_op, "ConstantFill", {shape_tensor}, {ends_tensor}, {dtype, constant});
  c2_op = ret.ops.Add();
  BuildOperator(
      c2_op,
      "ScatterAssign",
      {ends_tensor, axes_tensor, ends_vals_tensor},
      {ends_tensor});
  // Slice only accepts ends as int
  c2_op = ret.ops.Add();
  BuildOperator(c2_op, "Cast", {ends_tensor}, {casted_ends_tensor}, {to});

  // attach the original op at the end
  c2_op = ret.ops.Add();
  c2_op->CopyFrom(*op);
  c2_op->mutable_input()->Clear();
  c2_op->add_input(data);
  c2_op->add_input(casted_starts_tensor);
  c2_op->add_input(casted_ends_tensor);
  c2_op->mutable_arg()->Clear();
  for (const auto& kv : args) {
    c2_op->add_arg()->CopyFrom(*kv.second);
  }

  return ret;
}

Caffe2Ops Caffe2Backend::CreateBatchNormalization(
    OnnxNode* onnx_node,
    int opset_version) {
  const auto& node = onnx_node->node;
  if (opset_version < 6) {
    auto& attributes = onnx_node->attributes;
    attributes.remove("consumed_inputs");
  }

  return CommonOnnxNodeToCaffe2Ops(onnx_node, opset_version);
}

Caffe2Ops Caffe2Backend::CreateMatMul(OnnxNode* onnx_node, int opset_version) {
  const auto& node = onnx_node->node;
  if (node.input_size() != 2) {
    CAFFE_THROW("MatMul should have 2 inputs");
  }

  auto c2_op = CommonOnnxNodeToCaffe2Ops(onnx_node, opset_version);
  CAFFE_ENFORCE_EQ(c2_op.ops.size(), 1);
  auto* op = c2_op.ops.Mutable(0);
  auto* broadcast_arg = op->add_arg();
  broadcast_arg->set_name("broadcast");
  broadcast_arg->set_i(1);

  return c2_op;
}

//==============================================
// Rest of the member funtions for Caffe2Backend
//==============================================
std::unordered_set<std::string> Caffe2Backend::AllNamesInGraph(
    const GraphProto& graph) {
  std::unordered_set<std::string> names;

  for (const auto& input : graph.input()) {
    names.emplace(input.name());
  }
  for (const auto& output : graph.output()) {
    names.emplace(output.name());
  }
  for (const auto& node : graph.node()) {
    for (const auto& n : node.input()) {
      names.emplace(n);
    }
    for (const auto& n : node.output()) {
      names.emplace(n);
    }
  }

  return names;
}

//  This translator performs the basic translation of ONNX nodes into
//  Caffe2 operators.  Besides doing a straightforward marshalling from
//  one format to another, it also does these extra things:
//
//    - Renames operators based on 'renamed_operators'
//    - Renames attributes based on 'renamed_attrs' and
//      'get_per_op_renamed_attrs'
//
//  If you're writing a custom translator, consider calling this first,
//  and then fixing things up further.
Caffe2Ops Caffe2Backend::CommonOnnxNodeToCaffe2Ops(
    OnnxNode* onnx_node,
    int opset_version) {
  Caffe2Ops ret;
  auto* c2_op = ret.ops.Add();

  const auto& node = onnx_node->node;
  c2_op->mutable_input()->MergeFrom(node.input());
  c2_op->mutable_output()->MergeFrom(node.output());
  c2_op->set_name(node.name());

  const auto onnx_op_type = node.op_type();
  auto broken_version = caffe2::get_default(
      get_broken_operators(), onnx_op_type, std::numeric_limits<int>::max());
  if (broken_version <= opset_version) {
    CAFFE_THROW(
        "Don't know how to translate op ",
        onnx_op_type,
        " in ONNX operator set v",
        opset_version,
        " (I only support prior to v",
        broken_version);
  }
  c2_op->set_type(
      caffe2::get_default(get_renamed_operators(), onnx_op_type, onnx_op_type));
  if (!IsOperator(c2_op->type())) {
    CAFFE_THROW(
        "Don't know how to translate op ", onnx_op_type);
  }

  auto mapper = [&, this](const std::string& k) {
    const auto it = get_per_op_renamed_attrs().find(onnx_op_type);
    if (it != get_per_op_renamed_attrs().end()) {
      const auto it_op = it->second.find(k);
      if (it_op != it->second.end()) {
        return it_op->second;
      }
    }
    const auto it_global = get_renamed_attrs().find(k);
    if (it_global != get_renamed_attrs().end()) {
      return it_global->second;
    }
    return k;
  };
  c2_op->mutable_arg()->MergeFrom(
      onnx_node->attributes.OnnxAttrToCaffe2Arg(mapper));

  return ret;
}

Caffe2Ops Caffe2Backend::ConvertNode(
    const std::string& node_str,
    int opset_version) {
  ::google::protobuf::RepeatedPtrField<NodeProto> nodes;
  auto* n = nodes.Add();
  ParseProtoFromLargeString(node_str, n);
  ModelProto init_model;
  ModelProto pred_model;
  OnnxNode onnx_node = OnnxNode(nodes.Get(0));
  return OnnxNodeToCaffe2Ops(init_model, pred_model, &onnx_node, opset_version);
}

Caffe2Ops Caffe2Backend::OnnxNodeToCaffe2Ops(
    const ModelProto& init_model,
    const ModelProto& pred_model,
    OnnxNode* onnx_node,
    int opset_version) {
  if (get_special_operators().count(onnx_node->node.op_type())) {
    return (this->*get_special_operators().at(onnx_node->node.op_type()))(
        onnx_node, opset_version);
  } else {
    return CommonOnnxNodeToCaffe2Ops(onnx_node, opset_version);
  }
}

void Caffe2Backend::OnnxToCaffe2(
    caffe2::NetDef* init_net,
    caffe2::NetDef* pred_net,
    const ModelProto& onnx_model,
    const std::string& device,
    int opset_version,
    bool include_initializers,
    const std::vector<Caffe2Ops>& extras) {
  auto device_option = GetDeviceOption(Device(device));

#if !CAFFE2_MOBILE
  ModelProto init_model = OptimizeOnnx(onnx_model, true);
  ModelProto pred_model = OptimizeOnnx(onnx_model, false);
#else
  ModelProto init_model = ModelProto();
  ModelProto pred_model = onnx_model;
  pred_model.mutable_graph()->mutable_initializer()->Clear();
#endif

  init_net->set_name(onnx_model.graph().name() + "_init");
  pred_net->set_name(onnx_model.graph().name() + "_predict");

  // Convert initializer if necessary
  if (include_initializers) {
    for (const auto& tp : onnx_model.graph().initializer()) {
      auto* c2_op = init_net->add_op();
      BuildTensorFillingOp(c2_op, tp);
    }
  }

  auto name_set = AllNamesInGraph(init_model.graph());
  auto name_set_pred = AllNamesInGraph(pred_model.graph());
  name_set.insert(name_set_pred.begin(), name_set_pred.end());
  DummyName::Reset(name_set);

  size_t idx_extra = 0;
  auto converter = [&](const ModelProto& model, caffe2::NetDef* net) mutable {
    net->mutable_device_option()->CopyFrom(device_option);
    for (const auto& node : model.graph().node()) {
      auto* init_net_tmp = include_initializers ? init_net : net;
      // For RNN operators, we rely on Python code to convert them for us, and
      // we simply deserilize the string. This is hack and eventually we want to
      // get rid of this to have one flow. Note that we need to update the dummy
      // name generator to avoid having duplicated names between Python and C++
      // generated dummies
      if (get_rnn_operators().count(node.op_type())) {
        if (idx_extra < extras.size()) {
          const auto& c2ops = extras[idx_extra++];
          for (const auto& op : c2ops.init_ops) {
            UpdateNames(op);
          }
          init_net_tmp->mutable_op()->MergeFrom(c2ops.init_ops);
          for (const auto& op : c2ops.ops) {
            UpdateNames(op);
          }
          net->mutable_op()->MergeFrom(c2ops.ops);
          for (const auto& input : c2ops.interface_blobs) {
            DummyName::AddName(input);
          }
          net->mutable_external_input()->MergeFrom(c2ops.interface_blobs);
        } else {
          CAFFE_THROW(
              "Don't know how to convert ",
              node.op_type(),
              " without enough extra preconverted string");
        }
      } else {
        auto onnx_node = OnnxNode(node);
        auto c2ops = OnnxNodeToCaffe2Ops(
            init_model, pred_model, &onnx_node, opset_version);
        init_net_tmp->mutable_op()->MergeFrom(c2ops.init_ops);
        net->mutable_op()->MergeFrom(c2ops.ops);
        net->mutable_external_input()->MergeFrom(c2ops.interface_blobs);
      }
    }

    for (const auto& value : model.graph().output()) {
      net->add_external_output(value.name());
    }
    for (const auto& value : model.graph().input()) {
      net->add_external_input(value.name());
    }
  };

  converter(init_model, init_net);
  converter(pred_model, pred_net);
}

Caffe2BackendRep* Caffe2Backend::Prepare(
    const std::string& onnx_model_str,
    const std::string& device,
    const std::vector<Caffe2Ops>& extras) {
  Caffe2BackendRep* rep = new Caffe2BackendRep();
  ModelProto onnx_model;
  ParseProtoFromLargeString(onnx_model_str, &onnx_model);

#if !CAFFE2_MOBILE
  ::ONNX_NAMESPACE::checker::check_model(onnx_model);
#endif

  int opset_version = -1;
  for (const auto& imp : onnx_model.opset_import()) {
    if ((!imp.has_domain()) || imp.domain().empty()) {
      opset_version = imp.version();
      if (opset_version > kKnownOpsetVersion) {
        std::cout
            << "This version of onnx-caffe2 targets ONNX operator set version "
            << kKnownOpsetVersion
            << ", but the model we are trying to import uses version "
            << opset_version << ".  We will try to import it anyway, "
            << "but if the model uses operators which had BC-breaking changes "
               "in the intervening versions, import will fail."
            << std::endl;
      }
    } else {
      std::cout << "Unrecognized operator set " << opset_version << std::endl;
    }
  }
  if (opset_version < 0) {
    if (onnx_model.ir_version() >= 0x00000003) {
      CAFFE_THROW(
          "Model with IR version >= 3 did not specify ONNX operator set "
          "version (onnx-caffe2 requires it)");
    } else {
      opset_version = 1;
    }
  }

  // TODO: avoid extra copy by directly feed initialiers to backend blobs
  OnnxToCaffe2(
      &rep->init_net(),
      &rep->pred_net(),
      onnx_model,
      device,
      opset_version,
      true,
      extras);

  // Get a list of uninitialized inputs to help with the inference setup
  auto& uninitialized_inputs = rep->uninitialized_inputs();
  std::unordered_set<std::string> initialized_inputs;
  for (const auto& tp : onnx_model.graph().initializer()) {
    initialized_inputs.emplace(tp.name());
  }
  for (const auto& input : onnx_model.graph().input()) {
    if (!initialized_inputs.count(input.name())) {
      uninitialized_inputs.emplace_back(input.name());
    }
  }

  return rep;
}

void Caffe2Backend::BuildTensorFillingOp(
    caffe2::OperatorDef* c2_op,
    const TensorProto& onnx_tensor,
    const std::string& name) {
  auto fill_name = name.empty() ? onnx_tensor.name() : name;
  CAFFE_ENFORCE(!fill_name.empty());

  if (onnx_tensor.has_segment()) {
    CAFFE_THROW("Currently not supporting loading segments.");
  }

  auto* c2_values = c2_op->add_arg();
  c2_values->set_name("values");

  if (onnx_tensor.data_type() == TensorProto::FLOAT) {
    c2_op->set_type("GivenTensorFill");
    auto* floats = c2_values->mutable_floats();
    if (!TryConvertingTensorRawValues<float>(onnx_tensor, floats)) {
      floats->CopyFrom(onnx_tensor.float_data());
    }
  } else if (onnx_tensor.data_type() == TensorProto::DOUBLE) {
    c2_op->set_type("GivenTensorDoubleFill");
    ::google::protobuf::RepeatedField<double> tmp;
    const ::google::protobuf::RepeatedField<double>* src = &tmp;
    if (!TryConvertingTensorRawValues<double>(onnx_tensor, &tmp)) {
      src = &onnx_tensor.double_data();
    } else {
      for (const auto i : *src) {
        c2_values->add_floats(i);
      }
    }
  } else if (onnx_tensor.data_type() == TensorProto::INT64) {
    c2_op->set_type("GivenTensorInt64Fill");
    auto* ints = c2_values->mutable_ints();
    if (!TryConvertingTensorRawValues<::google::protobuf::int64>(
            onnx_tensor, ints)) {
      ints->CopyFrom(onnx_tensor.int64_data());
    }
  } else if (onnx_tensor.data_type() == TensorProto::UINT32) {
    c2_op->set_type("GivenTensorInt64Fill");
    ::google::protobuf::RepeatedField<::google::protobuf::uint64> tmp;
    const ::google::protobuf::RepeatedField<::google::protobuf::uint64>* src =
        &tmp;
    if (!TryConvertingTensorRawValues<::google::protobuf::uint64>(
            onnx_tensor, &tmp)) {
      src = &onnx_tensor.uint64_data();
    } else {
      for (const auto i : *src) {
        c2_values->add_ints(i);
      }
    }
  } else if (
      onnx_tensor.data_type() == TensorProto::BOOL ||
      onnx_tensor.data_type() == TensorProto::UINT8 ||
      onnx_tensor.data_type() == TensorProto::INT8 ||
      onnx_tensor.data_type() == TensorProto::UINT16 ||
      onnx_tensor.data_type() == TensorProto::INT16 ||
      onnx_tensor.data_type() == TensorProto::INT32) {
    c2_op->set_type(
        onnx_tensor.data_type() == TensorProto::BOOL ? "GivenTensorBoolFill"
                                                     : "GivenTensorIntFill");
    ::google::protobuf::RepeatedField<::google::protobuf::int32> tmp;
    const ::google::protobuf::RepeatedField<::google::protobuf::int32>* src =
        &tmp;
    if (!TryConvertingTensorRawValues<::google::protobuf::int32>(
            onnx_tensor, &tmp)) {
      src = &onnx_tensor.int32_data();
    } else {
      for (const auto i : *src) {
        c2_values->add_ints(i);
      }
    }
  } else if (onnx_tensor.data_type() == TensorProto::STRING) {
    c2_op->set_type("GivenTensorStringFill");
    auto* strings = c2_values->mutable_strings();
    strings->CopyFrom(onnx_tensor.string_data());
  } else {
    CAFFE_THROW(
        "unrecognized tensor type: ",
        TensorProto::DataType_Name(onnx_tensor.data_type()));
  }

  auto* c2_shape = c2_op->add_arg();
  c2_shape->set_name("shape");
  for (const auto d : onnx_tensor.dims()) {
    c2_shape->add_ints(d);
  }
  c2_op->add_output(fill_name);
}

bool Caffe2Backend::SupportOp(const std::string type) const {
  return get_special_operators().count(type);
}

} // namespace onnx
} // namespace caffe2
