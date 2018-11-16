#include "caffe2/onnx/onnx_exporter.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/tensor_impl.h"
#include "caffe2/onnx/helper.h"
#include "caffe2/proto/caffe2_legacy.pb.h"
#include "caffe2/utils/map_utils.h"
#include "caffe2/utils/proto_utils.h"

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

TensorProto CreateOnnxShapeTensor(
    std::shared_ptr<DummyName> dummy,
    const std::vector<int64_t>& shape) {
  TensorProto tensor;
  tensor.set_name(dummy->NewDummyName());
  tensor.set_data_type(TensorProto::INT64);
  tensor.add_dims(shape.size());
  tensor.mutable_raw_data()->assign(
      reinterpret_cast<const char*>(shape.data()),
      sizeof(int64_t) * shape.size());
  return tensor;
}

std::string SsaName(const std::string& n, int version) {
  return c10::str(n, "_", version);
}

NodeProto AddShapeNode(const std::string& input, const std::string& output) {
  NodeProto shape_node;
  shape_node.set_op_type("Shape");
  shape_node.add_input(input);
  shape_node.add_output(output);
  return shape_node;
}

} // namespace

::ONNX_NAMESPACE::TensorProto::DataType Caffe2TypeToOnnxType(
    caffe2::TensorProto::DataType t) {
#define CAFFE2_TO_ONNX_TYPE(x)   \
  case (caffe2::TensorProto::x): \
    return ::ONNX_NAMESPACE::TensorProto::x
  switch (t) {
    CAFFE2_TO_ONNX_TYPE(FLOAT);
    CAFFE2_TO_ONNX_TYPE(BOOL);
    CAFFE2_TO_ONNX_TYPE(INT8);
    CAFFE2_TO_ONNX_TYPE(UINT8);
    CAFFE2_TO_ONNX_TYPE(UINT16);
    CAFFE2_TO_ONNX_TYPE(INT16);
    CAFFE2_TO_ONNX_TYPE(INT32);
    CAFFE2_TO_ONNX_TYPE(INT64);
    CAFFE2_TO_ONNX_TYPE(FLOAT16);
    default:
      LOG(WARNING) << "Unsupported Caffe2 tensor type: " << t
                   << ", fallback to FLOAT";
      return ::ONNX_NAMESPACE::TensorProto::FLOAT;
  }
#undef CAFFE2_TO_ONNX_TYPE
}

std::unordered_map<std::string, std::string> SsaRewrite(
    caffe2::NetDef* init_net,
    caffe2::NetDef* pred_net) {
  std::unordered_map<std::string, std::string> input_mapping;
  std::unordered_map<std::string, int> blob_versions;

#define REWRITE_EXTERNAL_IO(net, name)                 \
  for (auto& name : *net->mutable_external_##name()) { \
    auto version = blob_versions.at(name);             \
    auto new_##name = SsaName(name, version);          \
    name##_mapping.emplace(new_##name, name);          \
    name = new_##name;                                 \
  }

  if (init_net) {
    for (auto& op : *init_net->mutable_op()) {
      CAFFE_ENFORCE_EQ(op.type().find("GivenTensor"), 0);
      CAFFE_ENFORCE_EQ(op.type().rfind("Fill"), op.type().size() - 4);
      CAFFE_ENFORCE_EQ(op.output_size(), 1);
      const auto& output = op.output(0);
      op.set_output(0, SsaName(output, 0));
    }
    for (const auto& input : init_net->external_input()) {
      blob_versions.emplace(input, 0);
    }
    for (const auto& output : init_net->external_output()) {
      blob_versions.emplace(output, 0);
    }
    REWRITE_EXTERNAL_IO(init_net, input);
    blob_versions.clear();
  }

  if (pred_net) {
    for (const auto& input : pred_net->external_input()) {
      blob_versions.emplace(input, 0);
    }
    REWRITE_EXTERNAL_IO(pred_net, input);
    for (auto& op : *pred_net->mutable_op()) {
      for (auto& input : *op.mutable_input()) {
        const auto it = blob_versions.find(input);
        if (it != blob_versions.end()) {
          input = SsaName(input, it->second);
        } else {
          blob_versions.emplace(input, 0);
          input = SsaName(input, 0);
        }
      }
      for (auto& output : *op.mutable_output()) {
        auto it = blob_versions.find(output);
        if (it != blob_versions.end()) {
          it->second += 1;
          output = SsaName(output, it->second);
        } else {
          blob_versions.emplace(output, 0);
          output = SsaName(output, 0);
        }
      }
    }

    // Fix the external output name back to original
    std::unordered_set<std::string> external_outputs;
    for (const auto& output : pred_net->external_output()) {
      external_outputs.emplace(output);
    }
    for (auto& op : *pred_net->mutable_op()) {
      for (auto& output : *op.mutable_output()) {
        auto pos = output.find_last_of('_');
        CAFFE_ENFORCE_NE(pos, 0);
        auto basename = output.substr(0, pos);
        if (!external_outputs.count(basename)) {
          continue;
        }
        auto it = blob_versions.find(basename);
        if (it != blob_versions.end() &&
            SsaName(basename, it->second) == output) {
          output = basename;
        }
      }
    }
  }
#undef REWRITE_EXTERNAL_IO

  return input_mapping;
}

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
          {"ArgMax", &OnnxExporter::CreateArgMaxMinOpNodes},
          {"ArgMin", &OnnxExporter::CreateArgMaxMinOpNodes},
          {"Add", &OnnxExporter::CreateBinaryElementwiseOpNodes},
          {"Sub", &OnnxExporter::CreateBinaryElementwiseOpNodes},
          {"Mul", &OnnxExporter::CreateBinaryElementwiseOpNodes},
          {"Div", &OnnxExporter::CreateBinaryElementwiseOpNodes},
          {"Pow", &OnnxExporter::CreateBinaryElementwiseOpNodes},
          {"And", &OnnxExporter::CreateBinaryElementwiseOpNodes},
          {"Or", &OnnxExporter::CreateBinaryElementwiseOpNodes},
          {"Xor", &OnnxExporter::CreateBinaryElementwiseOpNodes},
          {"Equal", &OnnxExporter::CreateBinaryElementwiseOpNodes},
          {"Greater", &OnnxExporter::CreateBinaryElementwiseOpNodes},
          {"Less", &OnnxExporter::CreateBinaryElementwiseOpNodes},
          {"Cast", &OnnxExporter::CreateCastNodes},
          {"Conv", &OnnxExporter::CreateConvPoolNodes},
          {"ConvTranspose", &OnnxExporter::CreateConvPoolNodes},
          {"MaxPool", &OnnxExporter::CreateConvPoolNodes},
          {"AveragePool", &OnnxExporter::CreateConvPoolNodes},
          {"FC", &OnnxExporter::CreateGemmNodes},
          {"Concat", &OnnxExporter::CreateConcatNodes},
          {"LRN", &OnnxExporter::CreateLrnNodes},
          {"Reshape", &OnnxExporter::CreateReshapeNodes},
          {"Slice", &OnnxExporter::CreateSliceNodes},
          {"ChannelShuffle", &OnnxExporter::CreateChannelShuffleNodes},
          {"ResizeNearest", &OnnxExporter::CreateUpsampleNodes}};
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
    CAFFE_THROW(c10::str("Unsupported Caffe2 argument: ", arg.name()));
  }
}

bool OnnxExporter::IsBlackListed(const caffe2::Argument& arg) {
  const static std::unordered_map<std::string, std::unordered_set<std::string>>
      kBlackListString = {{"order", {"NCHW"}}};
  const static std::unordered_map<std::string, std::unordered_set<int64_t>>
      kBlackListInt = {{"cudnn_exhaustive_search", {0, 1}},
                       {"use_cudnn", {0, 1}},
                       {"exhaustive_search", {0, 1}},
                       {"is_test", {0, 1}},
                       {"broadcast", {0, 1}}};

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

ConvertedResult OnnxExporter::CreateArgMaxMinOpNodes(
    const caffe2::OperatorDef& def,
    const std::unordered_map<std::string, caffe2::TensorShape>& shapes) {
  auto result = CommonCaffe2OpToOnnxNodes(def);
  auto& nodes = result.first;

  CAFFE_ENFORCE_EQ(nodes.size(), 1);
  auto& node = nodes.back();

  if (!ArgumentHelper::HasArgument(def, "axis")) {
    const auto& x = def.input(0);
    const auto& x_shape = shapes.at(x);
    node.add_attribute()->CopyFrom(
        MakeAttribute("axis", x_shape.dims().size() - 1));
  }

  return result;
}

ConvertedResult OnnxExporter::CreateBinaryElementwiseOpNodes(
    const caffe2::OperatorDef& def,
    const std::unordered_map<std::string, caffe2::TensorShape>& shapes) {
  caffe2::OperatorDef mdef(def); // The modified def without broadcast and axis
  const auto& x = mdef.input(0);
  const auto& y = def.input(1); // Refer to the old def, later won't change it.
  const auto& x_shape = shapes.at(x);
  const auto& y_shape = shapes.at(y);
  for (int i = 0; i < mdef.arg_size(); ++i) {
    const auto& arg = mdef.arg(i);
    if (arg.name() == "broadcast") {
      ArgumentHelper::RemoveArgument(mdef, i);
      break;
    }
  }
  std::vector<int64_t> axes;
  for (int i = 0; i < mdef.arg_size(); ++i) {
    const auto& arg = mdef.arg(i);
    if (arg.name() == "axis") {
      int64_t axis = arg.i();
      if (x_shape.dims().size() - axis != y_shape.dims().size()) {
        // The upper bound (excluded) of expanded y.
        int64_t end_dim =
            y_shape.dims().size() - 1 - axis + x_shape.dims().size();
        axes.resize(end_dim - y_shape.dims().size());
        std::iota(axes.begin(), axes.end(), y_shape.dims().size());
        mdef.set_input(1, dummy_->NewDummyName());
      }
      ArgumentHelper::RemoveArgument(mdef, i);
      break;
    }
  }

  auto result = CommonCaffe2OpToOnnxNodes(mdef);
  if (axes.size() != 0) {
    result.first.insert(
        result.first.begin(),
        MakeNode(
            "Unsqueeze", {y}, {mdef.input(1)}, {MakeAttribute("axes", axes)}));
  }
  return result;
}

ConvertedResult OnnxExporter::CreateCastNodes(
    const caffe2::OperatorDef& def,
    const std::unordered_map<std::string, caffe2::TensorShape>& shapes) {
  auto result = CommonCaffe2OpToOnnxNodes(def);
  auto* attr = result.first[0].mutable_attribute(0);
  auto onnx_dtype = ::ONNX_NAMESPACE::TensorProto::UNDEFINED;
  const auto& arg = def.arg(0);
  if (arg.has_s()) {
    auto c2_dtype = arg.s();
    std::transform(
        c2_dtype.begin(), c2_dtype.end(), c2_dtype.begin(), ::toupper);
    if (c2_dtype == "FLOAT") {
      onnx_dtype = ::ONNX_NAMESPACE::TensorProto::FLOAT;
    } else if (c2_dtype == "INT32") {
      onnx_dtype = ::ONNX_NAMESPACE::TensorProto::INT32;
    } else if (c2_dtype == "BOOL") {
      onnx_dtype = ::ONNX_NAMESPACE::TensorProto::BOOL;
    } else if (c2_dtype == "UINT8") {
      onnx_dtype = ::ONNX_NAMESPACE::TensorProto::UINT8;
    } else if (c2_dtype == "INT8") {
      onnx_dtype = ::ONNX_NAMESPACE::TensorProto::INT8;
    } else if (c2_dtype == "UINT16") {
      onnx_dtype = ::ONNX_NAMESPACE::TensorProto::UINT16;
    } else if (c2_dtype == "INT16") {
      onnx_dtype = ::ONNX_NAMESPACE::TensorProto::INT16;
    } else if (c2_dtype == "INT64") {
      onnx_dtype = ::ONNX_NAMESPACE::TensorProto::INT64;
    } else if (c2_dtype == "FLOAT16") {
      onnx_dtype = ::ONNX_NAMESPACE::TensorProto::FLOAT16;
    } else if (c2_dtype == "DOUBLE") {
      onnx_dtype = ::ONNX_NAMESPACE::TensorProto::DOUBLE;
    } else {
      onnx_dtype = ::ONNX_NAMESPACE::TensorProto::UNDEFINED;
    }
    CAFFE_ENFORCE_NE(
        onnx_dtype,
        ::ONNX_NAMESPACE::TensorProto::UNDEFINED,
        "Casting to '",
        c2_dtype,
        "' dtype is not supported");
    attr->clear_s();
    attr->set_type(AttributeProto::INT);
  } else if (arg.has_i()) {
    const auto& c2_dtype = arg.i();
    switch (c2_dtype) {
      case caffe2::TensorProto::FLOAT:
        onnx_dtype = ::ONNX_NAMESPACE::TensorProto::FLOAT;
        break;
      case caffe2::TensorProto::INT32:
        onnx_dtype = ::ONNX_NAMESPACE::TensorProto::INT32;
        break;
      case caffe2::TensorProto::BOOL:
        onnx_dtype = ::ONNX_NAMESPACE::TensorProto::BOOL;
        break;
      case caffe2::TensorProto::UINT8:
        onnx_dtype = ::ONNX_NAMESPACE::TensorProto::UINT8;
        break;
      case caffe2::TensorProto::INT8:
        onnx_dtype = ::ONNX_NAMESPACE::TensorProto::INT8;
        break;
      case caffe2::TensorProto::UINT16:
        onnx_dtype = ::ONNX_NAMESPACE::TensorProto::UINT16;
        break;
      case caffe2::TensorProto::INT16:
        onnx_dtype = ::ONNX_NAMESPACE::TensorProto::INT16;
        break;
      case caffe2::TensorProto::INT64:
        onnx_dtype = ::ONNX_NAMESPACE::TensorProto::INT64;
        break;
      case caffe2::TensorProto::FLOAT16:
        onnx_dtype = ::ONNX_NAMESPACE::TensorProto::FLOAT16;
        break;
      case caffe2::TensorProto::DOUBLE:
        onnx_dtype = ::ONNX_NAMESPACE::TensorProto::DOUBLE;
        break;

      case caffe2::TensorProto::STRING:
      case caffe2::TensorProto::BYTE:
      case caffe2::TensorProto::UNDEFINED:
        onnx_dtype = ::ONNX_NAMESPACE::TensorProto::UNDEFINED;
        break;
    }
    CAFFE_ENFORCE_NE(
        onnx_dtype,
        ::ONNX_NAMESPACE::TensorProto::UNDEFINED,
        "Casting to '",
        c2_dtype,
        "' dtype is not supported");
  }
  attr->set_i(onnx_dtype);
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
    const auto& input_size = shapes.at(node.input(0));
    const auto& output_size = shapes.at(node.output(0));
    CAFFE_ENFORCE_EQ(output_size.dims().size(), 4);
    if (!global &&  // global pool does not care about legacy pad
        legacy_pad_attr.i() != static_cast<int64_t>(caffe2::LegacyPadding::NOTSET)) {
      if (legacy_pad_attr.i() ==
          static_cast<int64_t>(caffe2::LegacyPadding::VALID)) {
        CAFFE_ENFORCE(!attrs.count("pads"));
        attrs.emplace("auto_pad", MakeAttribute("auto_pad", "VALID"));
      } else if (legacy_pad_attr.i() ==
          static_cast<int64_t>(caffe2::LegacyPadding::SAME)) {
        CAFFE_ENFORCE(!attrs.count("pads"));
        // default behavior in Caffe2 is SAME_UPPER
        // https://github.com/caffe2/caffe2/blob/master/caffe2/operators/conv_pool_op_base.h#L39
        attrs.emplace("auto_pad", MakeAttribute("auto_pad", "SAME_UPPER"));
      } else if (legacy_pad_attr.i() ==
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
      } else {
        LOG(ERROR) << "Don't know how to handle the legacy_pad:" << legacy_pad_attr.i();
        CAFFE_THROW("Failed to handle legacy padding in pool operator!");
      }
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
  caffe2::OperatorDef mdef(def); // The modified def without add_axis
  // In caffe2, we can optionally add an axis specified by `add_axis`
  int add_axis = 0;
  for (int i = 0; i < mdef.arg_size(); ++i) {
    const auto& arg = mdef.arg(i);
    if (arg.name() == "add_axis") {
      add_axis = arg.i();
      ArgumentHelper::RemoveArgument(mdef, i);
      break;
    }
  }

  auto result = CommonCaffe2OpToOnnxNodes(mdef);
  auto& nodes = result.first;
  nodes.reserve(nodes.size() + 3);
  auto& const_tensors = result.second;

  CAFFE_ENFORCE_EQ(nodes.size(), 1);
  auto& node = nodes.back();
  bool explicit_axis = false;
  int axis = -1;
  if (ArgumentHelper::HasArgument(mdef, "axis")) {
    axis = ArgumentHelper::GetSingleArgument(mdef, "axis", -1);
    explicit_axis = true;
  }
  if (!explicit_axis) {
    node.add_attribute()->CopyFrom(MakeAttribute("axis", 1));
  }

  // If we have add_axis, we need to add a reshape node
  auto final_output = node.output(0);
  if (add_axis > 0) {
    CAFFE_ENFORCE_GE(axis, 0);
    std::vector<int64_t> dims;
    const auto& shape0 = shapes.at(mdef.input(0));
    for (int i = 1; i < mdef.input_size(); ++i) {
      const auto& shape = shapes.at(mdef.input(i));
      CAFFE_ENFORCE_EQ(shape.dims(axis), shape0.dims(axis));
    }
    for (const auto d : shape0.dims()) {
      dims.push_back(d);
    }
    dims.insert(dims.begin() + axis, mdef.input_size());

    auto concat_output = dummy_->NewDummyName();
    *node.mutable_output(0) = concat_output;
    const_tensors.emplace_back(CreateOnnxShapeTensor(dummy_, dims));
    nodes.emplace_back(MakeNode(
        "Reshape",
        {concat_output, const_tensors.back().name()},
        {final_output}));
  }

  // If we have two output, we need to output the split_info, which can be
  // statically inferred from the input shapes
  if (node.output_size() == 2) {
    std::string second_output = node.output(1);
    node.mutable_output()->RemoveLast();
    std::vector<int32_t> split_info;
    int adj_size = shapes.at(mdef.input(0)).dims_size() + (add_axis ? 1 : 0);
    int canonical_axis = canonical_axis_index_(axis, adj_size);
    CAFFE_ENFORCE_LT(canonical_axis, adj_size, "Axis not in input ndim range.");
    for (int i = 0; i < mdef.input_size(); ++i) {
      split_info.push_back(
          add_axis ? 1 : shapes.at(mdef.input(i)).dims(canonical_axis));
    }
    auto split_info_tensor =
        MakeTensor("split_info", split_info, TensorProto::INT32);
    auto cnode = MakeNode("Constant", {}, {second_output});
    cnode.add_attribute()->CopyFrom(MakeAttribute("value", split_info_tensor));
    nodes.emplace_back(std::move(cnode));
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
  for (const auto& arg : def.arg()) {
    if (arg.name() == "group") {
      g = arg.i();
      break;
    }
  }
  CAFFE_ENFORCE(g && c % g == 0);
  ConvertedResult result;
  auto& nodes = result.first;
  auto& const_tensors = result.second;

  const auto reshape_output = dummy_->NewDummyName();
  std::vector<int64_t> dims = {n, g, c / g, h, w};
  const_tensors.emplace_back(CreateOnnxShapeTensor(dummy_, dims));
  nodes.emplace_back(
      MakeNode("Reshape", {x, const_tensors.back().name()}, {reshape_output}));

  const auto transpose_output = dummy_->NewDummyName();
  dims = {0, 2, 1, 3, 4};
  nodes.emplace_back(MakeNode(
      "Transpose",
      {reshape_output},
      {transpose_output},
      {MakeAttribute("perm", dims)}));

  dims = {n, c, h, w};
  const_tensors.emplace_back(CreateOnnxShapeTensor(dummy_, dims));
  nodes.emplace_back(MakeNode(
      "Reshape", {transpose_output, const_tensors.back().name()}, {y}));

  return result;
}

ConvertedResult OnnxExporter::CreateUpsampleNodes(
    const caffe2::OperatorDef& def,
    const std::unordered_map<std::string, caffe2::TensorShape>& shapes) {
  ConvertedResult result;
  //{H, W} => {1, 1, H, W}
  auto& nodes = result.first;
  auto resolved_scale = dummy_->NewDummyName();
  if (def.input_size() == 1) {
    float width_scale = 1.0;
    float height_scale = 1.0;
    for (const auto& a : def.arg()) {
      if (a.name() == "width_scale") {
        width_scale = a.f();
      } else if (a.name() == "height_scale") {
        height_scale = a.f();
      }
    }
    CAFFE_ENFORCE_GT(width_scale, 0);
    CAFFE_ENFORCE_GT(height_scale, 0);
    std::vector<float> tmp_vector = {1, 1, height_scale, width_scale};
    auto resolved_scale_tensor =
        MakeTensor("resolved scale tensor", tmp_vector, TensorProto::FLOAT);

    auto node = MakeNode("Constant", {}, {resolved_scale});
    MakeAttribute("value", resolved_scale_tensor);
    node.add_attribute()->CopyFrom(
        MakeAttribute("value", resolved_scale_tensor));
    nodes.emplace_back(node);

  } else {
    CAFFE_ENFORCE_EQ(def.input_size(), 2);
    std::vector<float> tmp_vector = {1, 1};
    auto scale_pads_tensor =
        MakeTensor("scale pads", tmp_vector, TensorProto::FLOAT);
    auto unresolved_scale_pads = dummy_->NewDummyName();

    auto node = MakeNode("Constant", {}, {unresolved_scale_pads});
    node.add_attribute()->CopyFrom(MakeAttribute("value", scale_pads_tensor));
    nodes.emplace_back(node);

    node = MakeNode(
        "Concat", {unresolved_scale_pads, def.input(1)}, {resolved_scale});
    node.add_attribute()->CopyFrom(MakeAttribute("axis", 0));
    nodes.emplace_back(node);
  }
  std::vector<std::string> inputs = {def.input(0), resolved_scale};
  std::vector<std::string> outputs(def.output().begin(), def.output().end());
  auto node = MakeNode("Upsample", inputs, outputs, def.name());
  node.add_attribute()->CopyFrom(MakeAttribute("mode", "nearest"));
  nodes.emplace_back(node);
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
  for (auto& attr : *node.mutable_attribute()) {
    if (attr.name() == "starts") {
      auto len = attr.ints_size();
      if (len) {
        dims.resize(len);
        std::iota(dims.begin(), dims.end(), 0);
      }
    } else if (attr.name() == "ends") {
      for (int i = 0; i < attr.ints_size(); ++i) {
        auto end = attr.ints(i);
        if (end >= 0) {
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
      for (const auto k : attr.ints()) {
        shape.push_back(k);
      }
      const_tensors.emplace_back(CreateOnnxShapeTensor(dummy_, shape));
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
    std::string shape_input = node.output(0);
    std::string shape_output = node.output(1);
    node.mutable_output()->RemoveLast();
    nodes.emplace_back(AddShapeNode(shape_input, shape_output));
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
  const auto& w_shape = shapes.at(w);
  CAFFE_ENFORCE_GE(x_shape.dims().size(), 2);
  CAFFE_ENFORCE_GE(w_shape.dims().size(), 2);

  ConvertedResult result;
  auto& nodes = result.first;
  auto& const_tensors = result.second;
  std::unordered_map<std::string, const caffe2::Argument*> args;
  for (const auto& a : def.arg()) {
    args.emplace(a.name(), &a);
  }

  auto it = args.find("axis");
  int64_t axis = 1;
  bool has_axis = (it != args.end());
  if (has_axis) {
    axis = it->second->i();
  }
  if (x_shape.dims().size() > 2) {
    // we need to reshape only when dimension is higher than 2
    auto outer = DimProd(x_shape, 0, axis);
    auto inner = DimProd(x_shape, axis, x_shape.dims().size());
    std::vector<int64_t> dims = {outer, inner};
    auto reshaped_x = dummy_->NewDummyName();
    const_tensors.emplace_back(CreateOnnxShapeTensor(dummy_, dims));
    nodes.emplace_back(
        MakeNode("Reshape", {x, const_tensors.back().name()}, {reshaped_x}));
    x = reshaped_x;
  }

  it = args.find("axis_w");
  int64_t axis_w = 1;
  if (it != args.end()) {
    axis_w = it->second->i();
  }
  if (w_shape.dims().size() > 2) {
    // we need to reshape only when dimension is higher than 2
    auto outer = DimProd(w_shape, 0, axis_w);
    auto inner = DimProd(w_shape, axis_w, w_shape.dims().size());
    std::vector<int64_t> dims = {outer, inner};
    auto reshaped_w = dummy_->NewDummyName();
    const_tensors.emplace_back(CreateOnnxShapeTensor(dummy_, dims));
    nodes.emplace_back(
        MakeNode("Reshape", {w, const_tensors.back().name()}, {reshaped_w}));
    w = reshaped_w;
  }

  auto gemm_y_output = (has_axis) ? dummy_->NewDummyName() : y;
  std::vector<AttributeProto> attrs = {MakeAttribute("transB", 1L)};
  nodes.emplace_back(MakeNode(
      "Gemm",
      {x, w, b},
      {gemm_y_output},
      attrs,
      def.name()));

  if (has_axis) {
    std::vector<int64_t> dims;
    for (int i = 0; i < axis; ++i) {
      dims.push_back(x_shape.dims(i));
    }
    dims.push_back(-1);
    const_tensors.emplace_back(CreateOnnxShapeTensor(dummy_, dims));
    nodes.emplace_back(
        MakeNode("Reshape", {gemm_y_output, const_tensors.back().name()}, {y}));
  }

  return result;
}

void OnnxExporter::InitOpToTensorProto(
    const caffe2::OperatorDef& op,
    TensorProto* tensor) {
  CAFFE_ENFORCE_EQ(op.input_size(), 0);
  CAFFE_ENFORCE_EQ(op.output_size(), 1);

  // Set name
  tensor->set_name(op.output(0));

  const Argument* values = nullptr;
  const Argument* shape = nullptr;
  for (const auto& arg: op.arg()) {
    if (arg.name() == "values") {
      values = &arg;
    } else if (arg.name() == "shape") {
      shape = &arg;
    }
  }

  CAFFE_ENFORCE(values);
  CAFFE_ENFORCE(shape);

  // Set dims
  for (const auto i: shape->ints()) {
    tensor->add_dims(i);
  }

  // Set value
  if (op.type() == "GivenTensorFill") {
    tensor->set_data_type(TensorProto::FLOAT);
    for (const auto i : values->floats()) {
      tensor->add_float_data(i);
    }
  } else if (op.type() == "GivenTensorInt64Fill") {
    tensor->set_data_type(TensorProto::INT64);
    for (const auto i : values->ints()) {
      tensor->add_int64_data(i);
    }
  } else if (op.type() == "GivenTensorIntFill") {
    tensor->set_data_type(TensorProto::INT32);
    for (const auto i : values->ints()) {
      tensor->add_int32_data(i);
    }
  } else if (op.type() == "GivenTensorBoolFill") {
    tensor->set_data_type(TensorProto::INT32);
    for (const auto i : values->ints()) {
      tensor->add_int32_data(i);
    }
  } else if (op.type() == "GivenTensorStringFill") {
    tensor->set_data_type(TensorProto::STRING);
    // TODO: we might need to do two pass to avoid adverse memory allocations
    for (const auto& s : values->strings()) {
      tensor->mutable_raw_data()->append(s);
    }
  } else {
    CAFFE_THROW(
        c10::str("Cannot convert C2 op ", op.type(), "to ONNX TensorProto"));
  }
}

} // namespace onnx
} // namespace caffe2

