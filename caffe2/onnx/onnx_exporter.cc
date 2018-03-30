#include "caffe2/core/logging.h"
#include "caffe2/onnx/onnx_exporter.h"
#include "caffe2/onnx/helper.h"
#include "caffe2/proto/caffe2_legacy.pb.h"
#include "caffe2/utils/map_utils.h"

#include <unordered_set>

namespace caffe2 {
namespace onnx {

namespace {
// rewrite padding attributes
void ApplyTrans(
    std::unordered_map<std::string, AttributeProto>* attrs,
    bool global,
    const std::string& k,
    int dim = 2,
    const std::string& ks = "") {
  std::string ks2 = ks.empty() ? (k + "s") : ks;
  std::string k_h, k_w, k_t, k_l, k_b, k_r;
  if (dim == 2) {
    k_h = k + "_h";
    k_w = k + "_w";
  } else {
    k_t = k + "_t";
    k_l = k + "_l";
    k_b = k + "_b";
    k_r = k + "_r";
  }

  std::vector<int64_t> vals;
  if (dim == 2 && attrs->count(k_h) && attrs->count(k_w)) {
    auto it = attrs->find(k_h);
    vals.push_back(it->second.i());
    attrs->erase(it);
    it = attrs->find(k_w);
    vals.push_back(it->second.i());
    attrs->erase(it);
  } else if (
      dim == 4 && attrs->count(k_t) && attrs->count(k_b) && attrs->count(k_l) &&
      attrs->count(k_r)) {
    auto it = attrs->find(k_t);
    vals.push_back(it->second.i());
    attrs->erase(it);
    it = attrs->find(k_l);
    vals.push_back(it->second.i());
    attrs->erase(it);
    it = attrs->find(k_b);
    vals.push_back(it->second.i());
    attrs->erase(it);
    it = attrs->find(k_r);
    vals.push_back(it->second.i());
    attrs->erase(it);
  } else if (attrs->count(k)) {
    auto it = attrs->find(k);
    auto tmp = it->second.i();
    for (int i = 0; i < dim; ++i) {
      vals.push_back(tmp);
    }
    attrs->erase(it);
  }

  if (!vals.empty() && !global) {
    attrs->emplace(ks2, MakeAttribute(ks2, vals));
  }
}

int64_t DimProd(const caffe2::TensorShape& shape, int start, int end) {
  int64_t acc = 1;
  for (int i = start; i < end; ++i) {
    acc *= shape.dims(i);
  }
  return acc;
}

TensorProto CreateOnnxShapeTensor(const std::vector<int64_t>& shape) {
  TensorProto tensor;
  tensor.set_name(DummyName::NewDummyName());
  tensor.set_data_type(TensorProto::INT64);
  tensor.add_dims(shape.size());
  tensor.mutable_raw_data()->assign(
      reinterpret_cast<const char*>(shape.data()), sizeof(int64_t) * shape.size());
  return tensor;
}
} // namespace

const std::unordered_map<std::string, std::string>&
OnnxExporter::get_renamed_operators() const {
  const static std::unordered_map<std::string, std::string> kRenamedOperators{
      {"SpatialBN", "BatchNormalization"},
      {"Conv1D", "Conv"},
      {"Conv2D", "Conv"},
      {"Conv3D", "Conv"},
      {"ConvTranspose1D", "ConvTranspose"},
      {"ConvTranspose2D", "ConvTranspose"},
      {"ConvTranspose3D", "ConvTranspose"},
      {"MaxPool1D", "MaxPool"},
      {"MaxPool2D", "MaxPool"},
      {"MaxPool3D", "MaxPool"},
      {"AveragePool1D", "AveragePool"},
      {"AveragePool2D", "AveragePool"},
      {"AveragePool3D", "AveragePool"}};
  return kRenamedOperators;
}

const std::unordered_map<std::string, std::string>&
OnnxExporter::get_renamed_attrs() const {
  const static std::unordered_map<std::string, std::string> kRenamedAttrs{
      {"kernels", "kernel_shape"}};
  return kRenamedAttrs;
}

const std::
    unordered_map<std::string, std::unordered_map<std::string, std::string>>&
    OnnxExporter::get_per_op_renamed_attrs() const {
  const static std::
      unordered_map<std::string, std::unordered_map<std::string, std::string>>
          kPerOpRenamedAttrs = {{"Squeeze", {{"dims", "axes"}}},
                                {"Unsqueeze", {{"dims", "axes"}}},
                                {"Transpose", {{"axes", "perm"}}},
                                {"ConvTranspose", {{"adjs", "output_padding"}}},
                                {"Selu", {{"scale", "gamma"}}}};

  return kPerOpRenamedAttrs;
}

const std::unordered_map<std::string, OnnxExporter::SpecialOpConverter>&
OnnxExporter::get_special_operators() const {
  const static std::unordered_map<std::string, OnnxExporter::SpecialOpConverter>
      kSpecialOperators = {
        {"Conv", &OnnxExporter::CreateConvPoolNodes},
        {"ConvTranspose", &OnnxExporter::CreateConvPoolNodes},
        {"MaxPool", &OnnxExporter::CreateConvPoolNodes},
        {"AveragePool", &OnnxExporter::CreateConvPoolNodes},
        {"FC", &OnnxExporter::CreateGemmNodes},
        {"Concat", &OnnxExporter::CreateConcatNodes},
        {"LRN", &OnnxExporter::CreateLrnNodes},
        {"Reshape", &OnnxExporter::CreateReshapeNodes},
        {"Slice", &OnnxExporter::CreateSliceNodes},
        {"ChannelShuffle",  &OnnxExporter::CreateChannelShuffleNodes}
      };
  return kSpecialOperators;
}

void OnnxExporter::CopyCaffe2ArgToOnnxAttr(
    AttributeProto* attr,
    const std::string& op_type,
    const caffe2::Argument& arg) {
  std::string name;
  const auto& per_op_renamed_attr_lut = get_per_op_renamed_attrs();
  const auto it = per_op_renamed_attr_lut.find(op_type);
  if (it != per_op_renamed_attr_lut.end()) {
    name = caffe2::get_default(it->second, arg.name(), arg.name());
  } else {
    name = caffe2::get_default(get_renamed_attrs(), arg.name(), arg.name());
  }
  attr->set_name(name);

  if (arg.has_f()) {
    attr->set_f(arg.f());
    attr->set_type(AttributeProto::FLOAT);
  } else if (arg.has_i()) {
    attr->set_i(arg.i());
    attr->set_type(AttributeProto::INT);
  } else if (arg.has_s()) {
    attr->set_s(arg.s());
    attr->set_type(AttributeProto::STRING);
  } else if (arg.floats_size()) {
    attr->mutable_floats()->CopyFrom(arg.floats());
    attr->set_type(AttributeProto::STRINGS);
  } else if (arg.ints_size()) {
    attr->mutable_ints()->CopyFrom(arg.ints());
    attr->set_type(AttributeProto::INTS);
  } else if (arg.strings_size()) {
    attr->mutable_strings()->CopyFrom(arg.strings());
    attr->set_type(AttributeProto::STRINGS);
  } else {
    CAFFE_THROW(
        caffe2::MakeString("Unsupported Caffe2 argument: ", arg.name()));
  }
}

bool OnnxExporter::IsBlackListed(const caffe2::Argument& arg) {
  const static std::unordered_map<std::string, std::unordered_set<std::string>>
      kBlackListString = {{"order", {"NCHW"}}};
  const static std::unordered_map<std::string, std::unordered_set<int64_t>>
      kBlackListInt = {{"cudnn_exhaustive_search", {0, 1}},
                       {"use_cudnn", {0, 1}}};

  if (arg.has_i()) {
    const auto it = kBlackListInt.find(arg.name());
    if (it != kBlackListInt.end()) {
      return it->second.count(arg.i());
    }
  } else if (arg.has_s()) {
    const auto it = kBlackListString.find(arg.name());
    if (it != kBlackListString.end()) {
      return it->second.count(arg.s());
    }
  }

  return false;
}

ConvertedResult OnnxExporter::Caffe2OpToOnnxNodes(
    const caffe2::OperatorDef& def,
    const std::unordered_map<std::string, caffe2::TensorShape>& shapes) {
  std::string type = def.type();
  const auto& renamed_op_lut = get_renamed_operators();
  const auto it = renamed_op_lut.find(type);
  if (it != renamed_op_lut.end()) {
    type = it->second;
  }
  const auto& special_op_lut = get_special_operators();
  const auto it_op = get_special_operators().find(type);
  if (it_op != special_op_lut.end()) {
    return (this->*(it_op->second))(def, shapes);
  } else {
    return CommonCaffe2OpToOnnxNodes(def);
  }
}

ConvertedResult OnnxExporter::CommonCaffe2OpToOnnxNodes(
    const caffe2::OperatorDef& def) {
  ConvertedResult result;
  auto& nodes = result.first;
  nodes.emplace_back();
  NodeProto& node = nodes.back();
  node.set_name(def.name());
  node.set_op_type(
      caffe2::get_default(get_renamed_operators(), def.type(), def.type()));
  for (const auto& i : def.input()) {
    node.add_input(i);
  }
  for (const auto& o : def.output()) {
    node.add_output(o);
  }
  for (const auto& a : def.arg()) {
    if (!IsBlackListed(a)) {
      auto* attr = node.add_attribute();
      CopyCaffe2ArgToOnnxAttr(attr, def.type(), a);
    }
  }
  return result;
}

ConvertedResult OnnxExporter::CreateConvPoolNodes(
    const caffe2::OperatorDef& def,
    const std::unordered_map<std::string, caffe2::TensorShape>& shapes) {
  auto result = CommonCaffe2OpToOnnxNodes(def);
  auto& nodes = result.first;
  auto& node = nodes.back();

  std::unordered_map<std::string, AttributeProto> attrs;
  for (const auto& attr : node.attribute()) {
    attrs.emplace(attr.name(), attr);
  }

  // Handle global pooling
  bool global = false;
  if (node.op_type() == "MaxPool" || node.op_type() == "AveragePool") {
    auto it = attrs.find("global_pooling");
    if (it != attrs.end() && it->second.has_i() && it->second.i()) {
      node.set_op_type("Global" + node.op_type());
      global = true;
      attrs.erase(it);
    }
  }

  ApplyTrans(&attrs, global, "kernel", 2, "kernel_shape");
  ApplyTrans(&attrs, global, "stride");
  ApplyTrans(&attrs, global, "dilation");
  ApplyTrans(&attrs, global, "adj");
  ApplyTrans(&attrs, global, "pad", 4);

  // Fix legacy pad attr
  auto it = attrs.find("legacy_pad");
  if (it != attrs.end()) {
    auto legacy_pad_attr = it->second;
    attrs.erase(it);
    CAFFE_ENFORCE(
        node.op_type().size() >= 4 &&
        (node.op_type().rfind("Pool") == node.op_type().size() - 4));
    CAFFE_ENFORCE(!global);
    const auto& input_size = shapes.at(node.input(0));
    const auto& output_size = shapes.at(node.output(0));
    CAFFE_ENFORCE(output_size.dims().size() == 4);
    if (legacy_pad_attr.i() ==
        static_cast<int64_t>(caffe2::LegacyPadding::VALID)) {
      CAFFE_ENFORCE(!attrs.count("pads"));
      attrs.emplace("auto_pad", MakeAttribute("auto_pad", "VALID"));
    } else if (
        legacy_pad_attr.i() ==
        static_cast<int64_t>(caffe2::LegacyPadding::SAME)) {
      CAFFE_ENFORCE(!attrs.count("pads"));
      // default behavior in Caffe2 is SAME_UPPER
      // https://github.com/caffe2/caffe2/blob/master/caffe2/operators/conv_pool_op_base.h#L39
      attrs.emplace("auto_pad", MakeAttribute("auto_pad", "SAME_UPPER"));
    } else if (
        legacy_pad_attr.i() ==
        static_cast<int64_t>(caffe2::LegacyPadding::CAFFE_LEGACY_POOLING)) {
      // The problem here is that, Pool op in Caffe may add an additional pixel,
      // if the last part is smaller than stride. So we use the explicit padding
      // to replace legacy_pad. pad[end] = output_size[start + 2] *
      // stride[start] - pad[start] - 1 + kernel[start] - input[start + 2] end =
      // start + len(pad) / 2
      LOG(WARNING) << "Converting legacy padding to explicit padding.";
      auto* pads_attr = attrs.at("pads").mutable_ints();
      auto& strides_attr = attrs.at("strides").ints();
      auto& kernel_shape_attr = attrs.at("kernel_shape").ints();
      for (int i = 0; i < 2; ++i) {
        int64_t tmp_pad = output_size.dims(i + 2) * strides_attr.Get(i) -
            pads_attr->Get(i) - 1 + kernel_shape_attr.Get(i) -
            input_size.dims(i + 2);
        pads_attr->Set(i + 2, tmp_pad);
      }
    } else if (
        legacy_pad_attr.i() !=
        static_cast<int64_t>(caffe2::LegacyPadding::NOTSET)) {
      CAFFE_THROW(caffe2::MakeString(
          "Don't know how to handle the legacy_pad, while processing operator: ",
          def.type()));
    }
  }

  node.clear_attribute();
  for (const auto& kv : attrs) {
    auto* attr = node.add_attribute();
    attr->CopyFrom(kv.second);
  }

  return result;
}

ConvertedResult OnnxExporter::CreateLrnNodes(
    const caffe2::OperatorDef& def,
    const std::unordered_map<std::string, caffe2::TensorShape>& shapes) {
  auto result = CommonCaffe2OpToOnnxNodes(def);
  auto& nodes = result.first;

  CAFFE_ENFORCE_EQ(nodes.size(), 1);
  auto& node = nodes.back();
  if (node.output_size() == 2) {
    node.mutable_output()->RemoveLast();
  }

  return result;
}

ConvertedResult OnnxExporter::CreateConcatNodes(
    const caffe2::OperatorDef& def,
    const std::unordered_map<std::string, caffe2::TensorShape>& shapes) {
  auto result = CommonCaffe2OpToOnnxNodes(def);
  auto& nodes = result.first;

  CAFFE_ENFORCE_EQ(nodes.size(), 1);
  auto& node = nodes.back();
  if (node.output_size() == 2) {
    node.mutable_output()->RemoveLast();
  }

  bool explicit_axis = false;
  for (const auto& a: def.arg()) {
    if (a.name() == "axis") {
      explicit_axis = true;
      break;
    }
  }
  if (!explicit_axis) {
    node.add_attribute()->CopyFrom(MakeAttribute("axis", 1L));
  }

  return result;
}

ConvertedResult OnnxExporter::CreateChannelShuffleNodes(
    const caffe2::OperatorDef& def,
    const std::unordered_map<std::string, caffe2::TensorShape>& shapes) {
  const auto& x = def.input(0);
  const auto& y = def.output(0);
  const auto& x_shape = shapes.at(x);
  CAFFE_ENFORCE_EQ(
      x_shape.dims().size(),
      4,
      "Input shape of ChannelShuffle needs to be in NCHW format");
  auto n = x_shape.dims(0);
  auto c = x_shape.dims(1);
  auto h = x_shape.dims(2);
  auto w = x_shape.dims(3);
  int64_t g = 0;
  for (const auto& arg: def.arg()) {
    if (arg.name() == "group") {
      g = arg.i();
      break;
    }
  }
  CAFFE_ENFORCE(g && c % g == 0);
  ConvertedResult result;
  auto& nodes = result.first;
  auto& const_tensors = result.second;

  const auto reshape_output = DummyName::NewDummyName();
  std::vector<int64_t> dims = {n, g, c / g, h, w};
  const_tensors.emplace_back(CreateOnnxShapeTensor(dims));
  nodes.emplace_back(
      MakeNode("Reshape", {x, const_tensors.back().name()}, {reshape_output}));

  const auto transpose_output = DummyName::NewDummyName();
  dims = {0, 2, 1, 3, 4};
  nodes.emplace_back(MakeNode(
      "Transpose",
      {reshape_output},
      {transpose_output},
      {MakeAttribute("perm", dims)}));

  dims = {n, c, h, w};
  const_tensors.emplace_back(CreateOnnxShapeTensor(dims));
  nodes.emplace_back(MakeNode(
      "Reshape", {transpose_output, const_tensors.back().name()}, {y}));

  return result;
}

ConvertedResult OnnxExporter::CreateSliceNodes(
    const caffe2::OperatorDef& def,
    const std::unordered_map<std::string, caffe2::TensorShape>& shapes) {
  CAFFE_ENFORCE_EQ(
      def.input_size(),
      1,
      "ONNX Slice operator does not support dynamic slice.");
  auto result = CommonCaffe2OpToOnnxNodes(def);
  auto& nodes = result.first;
  CAFFE_ENFORCE_EQ(nodes.size(), 1);
  auto& node = nodes.back();
  const auto& shape = shapes.at(node.input(0));

  std::vector<int64_t> dims;
  for (auto& attr: *node.mutable_attribute()) {
    if (attr.name() == "starts") {
      auto len = attr.ints_size();
      if (len) {
        dims.resize(len);
        std::iota(dims.begin(), dims.end(), 0);
      }
    } else if (attr.name() == "ends") {
      for (int i = 0; i < attr.ints_size(); ++i) {
        auto end = attr.ints(i);
        if (end >=0) {
          continue;
        }
        if (end == -1) {
          end = shape.dims(i);
        } else {
          ++end;
        }
        attr.set_ints(i, end);
      }
    }
  }
  if (!dims.empty()) {
    node.add_attribute()->CopyFrom(MakeAttribute("axes", dims));
  }

  return result;
}

ConvertedResult OnnxExporter::CreateReshapeNodes(
    const caffe2::OperatorDef& def,
    const std::unordered_map<std::string, caffe2::TensorShape>& shapes) {
  auto result = CommonCaffe2OpToOnnxNodes(def);
  auto& nodes = result.first;
  auto& const_tensors = result.second;
  CAFFE_ENFORCE_EQ(nodes.size(), 1);
  auto& node = nodes.back();

  int i = 0;
  int attr_size = node.attribute_size();
  for (; i < attr_size; ++i) {
    const auto& attr = node.attribute(i);
    if (attr.name() == "shape") {
      std::vector<int64_t> shape;
      for (const auto k: attr.ints()) {
        shape.push_back(k);
      }
      const_tensors.emplace_back(CreateOnnxShapeTensor(shape));
      node.add_input(const_tensors.back().name());
      break;
    }
  }
  if (i != attr_size) {
    if (i != attr_size - 1) {
      node.mutable_attribute()->SwapElements(i, attr_size - 1);
    }
    node.mutable_attribute()->RemoveLast();
  }

  if (node.output_size() == 2) {
    node.mutable_output()->RemoveLast();
  }

  return result;
}

ConvertedResult OnnxExporter::CreateGemmNodes(
    const caffe2::OperatorDef& def,
    const std::unordered_map<std::string, caffe2::TensorShape>& shapes) {
  CAFFE_ENFORCE_EQ(def.input_size(), 3);
  CAFFE_ENFORCE_GE(def.output_size(), 1);
  auto x = def.input(0);
  auto w = def.input(1);
  const auto& b = def.input(2);
  const auto& y = def.output(0);
  const auto& x_shape = shapes.at(x);

  ConvertedResult result;
  auto& nodes = result.first;
  auto& const_tensors = result.second;
  std::unordered_map<std::string, const caffe2::Argument*> args;
  for (const auto& a : def.arg()) {
    args.emplace(a.name(), &a);
  }

  auto it = args.find("axis");
  bool has_axis = (it != args.end());
  int64_t axis = 0;
  if (has_axis) {
    axis = it->second->i();
    auto outer = DimProd(x_shape, 0, axis);
    auto inner = DimProd(x_shape, axis, x_shape.dims().size());
    std::vector<int64_t> dims = {outer, inner};
    auto reshaped_x = DummyName::NewDummyName();
    const_tensors.emplace_back(CreateOnnxShapeTensor(dims));
    nodes.emplace_back(
        MakeNode("Reshape", {x, const_tensors.back().name()}, {reshaped_x}));
    x = reshaped_x;
  }

  it = args.find("axis_w");
  if (it != args.end()) {
    auto axis_w = it->second->i();
    const auto& w_shape = shapes.at(w);
    auto outer = DimProd(w_shape, 0, axis_w);
    auto inner = DimProd(w_shape, axis_w, w_shape.dims().size());
    std::vector<int64_t> dims = {outer, inner};
    auto reshaped_w = DummyName::NewDummyName();
    const_tensors.emplace_back(CreateOnnxShapeTensor(dims));
    nodes.emplace_back(
        MakeNode("Reshape", {w, const_tensors.back().name()}, {reshaped_w}));
    w = reshaped_w;
  }

  auto gemm_y_output = (has_axis) ? DummyName::NewDummyName() : y;
  nodes.emplace_back(MakeNode(
      "Gemm",
      {x, w, b},
      {gemm_y_output},
      {MakeAttribute("transB", 1L), MakeAttribute("broadcast", 1)},
      def.name()));

  if (has_axis) {
    std::vector<int64_t> dims;
    for (int i = 0; i < axis; ++i) {
      dims.push_back(x_shape.dims(i));
    }
    dims.push_back(-1);
    const_tensors.emplace_back(CreateOnnxShapeTensor(dims));
    nodes.emplace_back(
        MakeNode("Reshape", {gemm_y_output, const_tensors.back().name()}, {y}));
  }

  return result;
}
} // namespace onnx
} // namespace caffe2

