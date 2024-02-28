#include "caffe2/onnx/onnx_exporter.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/memonger.h"
#include "caffe2/core/tensor_impl.h"
#include "caffe2/onnx/helper.h"
#include "caffe2/proto/caffe2_legacy.pb.h"
#include "caffe2/utils/map_utils.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/utils/string_utils.h"

#include <numeric>
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

void collectExternalsFromIfOpSubnet(
    const NetDef* net,
    std::vector<std::string>* input,
    std::vector<std::string>* output) {
  std::set<std::string> in_input, in_output;
  for (const auto& op : net->op()) {
    for (const auto& blob : op.input()) {
      in_input.emplace(blob);
    }
    for (const auto& blob : op.output()) {
      in_output.emplace(blob);
    }
  }

  for (const auto& blob : in_input) {
    if (!in_output.count(blob)) {
      input->push_back(blob);
    }
  }
  for (const auto& blob : in_output) {
    if (!in_input.count(blob)) {
      output->push_back(blob);
    }
  }
}

void ssaRewriteForIfOp(
    OperatorDef* op,
    std::unordered_map<std::string, int>* blob_versions,
    std::set<std::string>* is_initialized_tensor) {
  // Get all the "external" inputs and outputs of the subnet
  // Since then_net and else_net has same external input/output, we only collect
  // external input/output from one of its subnet And perform the rewrite to
  // both then_net and else_net
  std::vector<std::string> if_external_input;
  std::vector<std::string> if_external_output;

  std::unordered_set<std::string> if_inputs, if_outputs;
  for (const auto& input : op->input()) {
    if_inputs.insert(input);
  }
  for (const auto& output : op->output()) {
    if_outputs.insert(output);
  }

  ArgumentHelper helper(*op);
  Argument *then_arg = nullptr, *else_arg = nullptr;
  NetDef* target_net = nullptr;
  bool has_then = false, has_else = false;

  if (helper.HasSingleArgumentOfType<NetDef>("then_net")) {
    then_arg = GetMutableArgument("then_net", false, op);
    target_net = then_arg->mutable_n();
    has_then = true;
  }
  if (helper.HasSingleArgumentOfType<NetDef>("else_net")) {
    else_arg = GetMutableArgument("else_net", false, op);
    if (!has_then) {
      target_net = else_arg->mutable_n();
    }
    has_else = true;
  }

  if (has_then || has_else) {
    collectExternalsFromIfOpSubnet(
        target_net, &if_external_input, &if_external_output);

    // Add inputs/outputs of the sub_net to the inputs/outputs of the op
    for (const auto& input : if_external_input) {
      if (if_inputs.count(input) == 0) {
        op->add_input(input);
      }
    }
    for (const auto& output : if_external_output) {
      if (if_outputs.count(output) == 0) {
        op->add_output(output);
      }
    }
    std::map<string, string> oldname_to_newname;

    // Build oldname_to_newname map
    for (auto& input : if_external_input) {
      const auto it = blob_versions->find(input);
      if (it != blob_versions->end()) {
        oldname_to_newname[input] = SsaName(input, it->second);
      }
    }
    for (auto& output : if_external_output) {
      auto it = blob_versions->find(output);
      if (it != blob_versions->end()) {
        if (is_initialized_tensor->count(output) == 0) {
          it->second += 1;
        } else {
          is_initialized_tensor->erase(output);
        }
        oldname_to_newname[output] = SsaName(output, it->second);
      } else {
        blob_versions->emplace(output, 0);
        oldname_to_newname[output] = SsaName(output, 0);
      }
    }

    if (has_then) {
      rewriteSubnet(then_arg, oldname_to_newname);
    }
    if (has_else) {
      rewriteSubnet(else_arg, oldname_to_newname);
    }
  }
}

void revertRenamedExternalOutput(
    OperatorDef* op,
    const std::unordered_map<std::string, std::string>&
        renamed_external_outputs) {
  for (auto& input : *(op->mutable_input())) {
    const auto it = renamed_external_outputs.find(input);
    if (it != renamed_external_outputs.end()) {
      input = it->second;
    }
  }
  for (auto& output : *(op->mutable_output())) {
    const auto it = renamed_external_outputs.find(output);
    if (it != renamed_external_outputs.end()) {
      output = it->second;
    }
  }
}

void revertRenamedExternalOutputForIfOp(
    OperatorDef* if_op,
    const std::unordered_map<std::string, std::string>&
        renamed_external_outputs) {
  ArgumentHelper helper(*if_op);
  Argument *then_arg = nullptr, *else_arg = nullptr;

  revertRenamedExternalOutput(if_op, renamed_external_outputs);

  if (helper.HasSingleArgumentOfType<NetDef>("then_net")) {
    then_arg = GetMutableArgument("then_net", false, if_op);
    NetDef* net = then_arg->mutable_n();
    for (auto& op : *(net->mutable_op())) {
      revertRenamedExternalOutput(&op, renamed_external_outputs);
    }
  }
  if (helper.HasSingleArgumentOfType<NetDef>("else_net")) {
    else_arg = GetMutableArgument("else_net", false, if_op);
    NetDef* net = else_arg->mutable_n();
    for (auto& op : *(net->mutable_op())) {
      revertRenamedExternalOutput(&op, renamed_external_outputs);
    }
  }
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

void rewriteSubnet(
    Argument* arg,
    std::map<std::string, std::string> oldname_to_newname) {
  NetDef* net = arg->mutable_n();
  // clear external inputs and outputs since they're no longer valid
  net->mutable_external_input()->Clear();
  net->mutable_external_output()->Clear();
  for (auto& op : *(net->mutable_op())) {
    for (auto& input : *(op.mutable_input())) {
      if (oldname_to_newname.find(input) != oldname_to_newname.end()) {
        input = oldname_to_newname[input];
      }
    }
    for (auto& output : *(op.mutable_output())) {
      if (oldname_to_newname.find(output) != oldname_to_newname.end()) {
        output = oldname_to_newname[output];
      }
    }
  }
}

std::unordered_map<std::string, std::string> SsaRewrite(
    caffe2::NetDef* init_net,
    caffe2::NetDef* pred_net,
    bool PreserveInPlaceOps) {
  std::unordered_map<std::string, std::string> input_mapping;
  std::unordered_map<std::string, int> blob_versions;

  if (init_net) {
    // No ssa rewrite is done for init net. The reason being that the output
    // blobs of init net are what becomes the input blobs of pred_net. Since
    // inputs of pred_net are not renamed we are not renaming the output of
    // init_net. Furthermore, the assumption made is that init_net is simple net
    // with each operator producing the one output and thus not renaming
    // translates to not renaming the outputs of the init_net. Create identical
    // mapping for now. This shall be removed eventually.
    for (const auto& name : init_net->external_input()) {
      input_mapping.emplace(name, name);
    }
    blob_versions.clear();
  }

  std::set<std::string> is_initialized_tensor;
  if (pred_net) {
    // Ssa rewriting modifies the net, check if the net passes schema check
    run_schema_check(*pred_net);

    std::unordered_set<std::string> external_outputs;
    for (const auto& input : pred_net->external_input()) {
      // Create identical mapping for now. This shall be removed eventually.
      input_mapping.emplace(input, input);
    }
    for (const auto& output : pred_net->external_output()) {
      external_outputs.emplace(output);
    }
    for (auto& op : *pred_net->mutable_op()) {
      // Special SSA Rewrite for subnet of If Operator
      // This needs to happen first because the inputs/outputs of If/AsyncIf
      // may get modified inside ssaRewriteForIfOp
      if (op.type() == "If" || op.type() == "AsyncIf") {
        ssaRewriteForIfOp(&op, &blob_versions, &is_initialized_tensor);
      }

      for (auto& input : *op.mutable_input()) {
        const auto it = blob_versions.find(input);
        if (it != blob_versions.end()) {
          input = SsaName(input, it->second);
        } else {
          // Input blob is not versioned yet.
          // If it is not versioned yet, it is assumed to be primary input,
          // Thus skip renaming it.
          continue;
        }
      }

      for (int out_idx = 0; out_idx < op.output_size(); out_idx++) {
        auto& output = *op.mutable_output(out_idx);

        // restore in-place settings
        bool is_inplace = false;
        if (PreserveInPlaceOps) {
          for (int in_idx = 0; in_idx < op.input_size(); in_idx++) {
            auto* schema = OpSchemaRegistry::Schema(op.type());
            if (schema && schema->inplace_enforced(in_idx, out_idx)) {
              output = op.input(in_idx);
              is_inplace = true;
              break;
            }
          }
        }
        if (is_inplace) {
          continue;
        }

        auto it = blob_versions.find(output);
        if (it != blob_versions.end()) {
          if (op.type() != "If" && op.type() != "AsyncIf") {
            if (is_initialized_tensor.count(output) == 0) {
              it->second += 1;
            } else {
              is_initialized_tensor.erase(output);
            }
          }
          output = SsaName(output, it->second);

        } else {
          blob_versions.emplace(output, 0);
          // These filling ops are designed for a by-default value for the
          // tensors generated by ops like If. For example, if an If op's
          // condition is not satisfied, and it does not have else_net, then it
          // will not generate any output blob, which may cause some error in
          // the future. Here we would like to ensure these tensors only been
          // ssa re-write once but not twice. (One in the filling operator, one
          // in If op)
          if ((caffe2::StartsWith(op.type(), "GivenTensor") &&
               caffe2::EndsWith(op.type(), "Fill")) ||
              op.type() == "ConstantFill" ||
              op.type() == "Int8GivenTensorFill" ||
              op.type() == "Int8GivenIntTensorFill") {
            is_initialized_tensor.insert(output);
          }
          output = SsaName(output, 0);
        }
      }
    }

    // For all the renamed blobs find if the blob is one of the external
    // output. If so add a mapping from it's latest renamed version to its
    // original name.
    std::unordered_map<std::string, std::string> renamed_external_outputs;
    for (const auto& it : blob_versions) {
      if (external_outputs.count(it.first)) {
        renamed_external_outputs.emplace(
            SsaName(it.first, it.second), it.first);
      }
    }

    // Use the mapping to find if the input or output of an op was a renamed
    // external output. If so replace it with its original name.
    for (auto& op : *pred_net->mutable_op()) {
      // If/AsyncIf needs special handling
      if (op.type() == "If" || op.type() == "AsyncIf") {
        revertRenamedExternalOutputForIfOp(&op, renamed_external_outputs);
      } else {
        revertRenamedExternalOutput(&op, renamed_external_outputs);
      }
    }
  }
  // run schema check again
  // NOLINTNEXTLINE(clang-analyzer-core.NonNullParamChecker)
  run_schema_check(*pred_net);

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
      {"AveragePool3D", "AveragePool"},
      {"Copy", "Identity"}};
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
          kPerOpRenamedAttrs = {
              {"Squeeze", {{"dims", "axes"}}},
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
          {"ElementwiseLinear", &OnnxExporter::CreateElementwiseLinearNodes},
          {"Conv", &OnnxExporter::CreateConvPoolNodes},
          {"ConvTranspose", &OnnxExporter::CreateConvPoolNodes},
          {"MaxPool", &OnnxExporter::CreateConvPoolNodes},
          {"AveragePool", &OnnxExporter::CreateConvPoolNodes},
          {"FC", &OnnxExporter::CreateGemmNodes},
          {"Concat", &OnnxExporter::CreateConcatNodes},
          {"MergeDim", &OnnxExporter::CreateMergeDimNodes},
          {"LRN", &OnnxExporter::CreateLrnNodes},
          {"Reshape", &OnnxExporter::CreateReshapeNodes},
          {"Slice", &OnnxExporter::CreateSliceNodes},
          {"ChannelShuffle", &OnnxExporter::CreateChannelShuffleNodes},
          {"ReduceMean", &OnnxExporter::CreateReduceMeanNodes},
          {"ReduceFrontMean", &OnnxExporter::CreateReduceMeanNodes},
          {"ReduceBackMean", &OnnxExporter::CreateReduceMeanNodes},
          {"ResizeNearest", &OnnxExporter::CreateUpsampleNodes}};
  return kSpecialOperators;
}

void OnnxExporter::CopyCaffe2ArgToOnnxAttr(
    AttributeProto* attr,
    const std::string& op_type,
    const caffe2::Argument& arg) {
  std::string name =
      caffe2::get_default(get_renamed_attrs(), arg.name(), arg.name());
  const auto& per_op_renamed_attr_lut = get_per_op_renamed_attrs();
  const auto it = per_op_renamed_attr_lut.find(op_type);
  if (it != per_op_renamed_attr_lut.end()) {
    // Per-op attribute renames override the global attribute renames
    name = caffe2::get_default(it->second, arg.name(), name);
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

bool OnnxExporter::IsBlockListed(const caffe2::Argument& arg) {
  const static std::unordered_map<std::string, std::unordered_set<std::string>>
      kBlockListString = {{"order", {"NCHW"}}};
  const static std::unordered_map<std::string, std::unordered_set<int64_t>>
      kBlockListInt = {
          {"cudnn_exhaustive_search", {0, 1}},
          {"use_cudnn", {0, 1}},
          {"exhaustive_search", {0, 1}},
          {"is_test", {0, 1}},
          {"broadcast", {0, 1}}};

  if (arg.has_i()) {
    const auto it = kBlockListInt.find(arg.name());
    if (it != kBlockListInt.end()) {
      return it->second.count(arg.i());
    }
  } else if (arg.has_s()) {
    const auto it = kBlockListString.find(arg.name());
    if (it != kBlockListString.end()) {
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
    if (!IsBlockListed(a)) {
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

ConvertedResult OnnxExporter::CreateElementwiseLinearNodes(
    const caffe2::OperatorDef& def,
    const std::unordered_map<std::string, caffe2::TensorShape>& shapes) {
  CAFFE_ENFORCE_EQ(def.input_size(), 3);
  CAFFE_ENFORCE_GE(def.output_size(), 1);
  const auto& x = def.input(0);
  const auto& w = def.input(1);
  const auto& b = def.input(2);
  const auto& y = def.output(0);
  CAFFE_ENFORCE_EQ(shapes.at(w).dims().size(), 1);
  CAFFE_ENFORCE_EQ(shapes.at(b).dims().size(), 1);

  ConvertedResult result;
  auto& nodes = result.first;
  auto& const_tensors = result.second;
  std::unordered_map<std::string, const caffe2::Argument*> args;
  for (const auto& a : def.arg()) {
    args.emplace(a.name(), &a);
  }

  const auto& x_shape = shapes.at(x);
  const auto it = args.find("axis");
  const int64_t axis = it == args.end() ? 1 : it->second->i();
  const bool need_reshape = axis + 1 != x_shape.dims().size();

  auto fma_x_input = x;
  if (need_reshape) {
    const auto inner = DimProd(x_shape, axis, x_shape.dims().size());
    CAFFE_ENFORCE_EQ(shapes.at(w).dims(0), inner);
    CAFFE_ENFORCE_EQ(shapes.at(b).dims(0), inner);

    fma_x_input = dummy_->NewDummyName();
    const_tensors.emplace_back(CreateOnnxShapeTensor(
        dummy_, std::vector<int64_t>{-1, shapes.at(w).dims(0)}));
    nodes.emplace_back(
        MakeNode("Reshape", {x, const_tensors.back().name()}, {fma_x_input}));
  }

  const auto& mul_output = dummy_->NewDummyName();
  nodes.emplace_back(
      MakeNode("Mul", {fma_x_input, w}, {mul_output}, def.name()));

  const auto& fma_y_output = need_reshape ? dummy_->NewDummyName() : y;
  nodes.emplace_back(
      MakeNode("Add", {mul_output, b}, {fma_y_output}, def.name()));

  if (need_reshape) {
    const auto shape = dummy_->NewDummyName();
    nodes.emplace_back(MakeNode("Shape", {x}, {shape}));
    nodes.emplace_back(MakeNode("Reshape", {fma_y_output, shape}, {y}));
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
    const auto& input_size = shapes.at(node.input(0));
    const auto& output_size = shapes.at(node.output(0));
    CAFFE_ENFORCE_EQ(output_size.dims().size(), 4);
    if (!global && // global pool does not care about legacy pad
        legacy_pad_attr.i() !=
            static_cast<int64_t>(caffe2::LegacyPadding::NOTSET)) {
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
        // The problem here is that, Pool op in Caffe may add an additional
        // pixel, if the last part is smaller than stride. So we use the
        // explicit padding to replace legacy_pad. pad[end] = output_size[start
        // + 2] * stride[start] - pad[start] - 1 + kernel[start] - input[start +
        // 2]; end = start + len(pad) / 2
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
        LOG(ERROR) << "Don't know how to handle the legacy_pad:"
                   << legacy_pad_attr.i();
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
      // NOLINTNEXTLINE(performance-inefficient-vector-operation)
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

ConvertedResult OnnxExporter::CreateMergeDimNodes(
    const caffe2::OperatorDef& def,
    const std::unordered_map<std::string, caffe2::TensorShape>& shapes) {
  const auto& x = def.input(0);
  const auto& y = def.output(0);

  ConvertedResult result;
  auto& nodes = result.first;
  auto& const_tensors = result.second;

  {
    const auto ndim = shapes.at(x).dims().size();
    CAFFE_ENFORCE_GE(ndim, 2, "No enough dims to merge.");
    std::vector<int64_t> dims(ndim);
    dims[0] = 1;
    dims[1] = -1;
    const_tensors.emplace_back(CreateOnnxShapeTensor(dummy_, dims));
  }

  const auto reshaped = dummy_->NewDummyName();
  nodes.emplace_back(
      MakeNode("Reshape", {x, const_tensors.back().name()}, {reshaped}));

  nodes.emplace_back(MakeNode(
      "Squeeze",
      {reshaped},
      {y},
      std::vector<AttributeProto>{
          MakeAttribute("axes", std::vector<int64_t>{0}),
      }));

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

ConvertedResult OnnxExporter::CreateReduceMeanNodes(
    const caffe2::OperatorDef& def,
    const std::unordered_map<std::string, caffe2::TensorShape>& shapes) {
  CAFFE_ENFORCE_GE(def.input_size(), 1);
  CAFFE_ENFORCE_LE(def.input_size(), 2);
  CAFFE_ENFORCE_EQ(def.input_size(), 1, "Input \"lengths\" is not supported.");
  CAFFE_ENFORCE_GE(def.output_size(), 1);
  const auto& x = def.input(0);
  const auto& y = def.output(0);
  const auto& dims = shapes.at(x).dims();

  ConvertedResult result;
  auto& nodes = result.first;
  std::unordered_map<std::string, const caffe2::Argument*> args;
  for (const auto& a : def.arg()) {
    args.emplace(a.name(), &a);
  }

  std::vector<int64_t> axes;
  int64_t keepdims = 1;

  if (def.type() == "ReduceMean") {
    // axes
    auto it = args.find("axes");
    if (it == args.end()) {
      axes.resize(dims.size());
      std::iota(axes.begin(), axes.end(), 0);
    } else {
      axes.assign(it->second->ints().begin(), it->second->ints().end());
    }

    // keepdims
    it = args.find("keepdims");
    if (it != args.end()) {
      keepdims = it->second->i();
    }
  } else {
    // num_reduce_dim
    auto it = args.find("num_reduce_dim");
    const int64_t num_reduce_dim = it == args.end() ? 1 : it->second->i();
    CAFFE_ENFORCE_LE(num_reduce_dim, dims.size());
    axes.resize(num_reduce_dim);

    int64_t start_dim = 0;
    if (def.type() == "ReduceFrontMean") {
      start_dim = 0;
    } else if (def.type() == "ReduceBackMean") {
      start_dim = dims.size() - axes.size();
    }
    std::iota(axes.begin(), axes.end(), start_dim);

    keepdims = 0;
  }

  nodes.emplace_back(MakeNode(
      "ReduceMean",
      {x},
      {y},
      {
          MakeAttribute("axes", axes),
          MakeAttribute("keepdims", keepdims),
      },
      def.name()));

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
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
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

  auto gemm_x_input = x;
  if (x_shape.dims().size() > 2) {
    // we need to reshape only when dimension is higher than 2
    const auto inner = DimProd(x_shape, axis, x_shape.dims().size());

    gemm_x_input = dummy_->NewDummyName();
    const_tensors.emplace_back(
        CreateOnnxShapeTensor(dummy_, std::vector<int64_t>{-1, inner}));
    nodes.emplace_back(
        MakeNode("Reshape", {x, const_tensors.back().name()}, {gemm_x_input}));
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
    auto reshaped_w = dummy_->NewDummyName();
    const_tensors.emplace_back(
        CreateOnnxShapeTensor(dummy_, std::vector<int64_t>{outer, inner}));
    nodes.emplace_back(
        MakeNode("Reshape", {w, const_tensors.back().name()}, {reshaped_w}));
    w = reshaped_w;
  }

  auto gemm_y_output = axis > 1 ? dummy_->NewDummyName() : y;
  nodes.emplace_back(MakeNode(
      "Gemm",
      {gemm_x_input, w, b},
      {gemm_y_output},
      {MakeAttribute("transB", 1L)},
      def.name()));

  // capture the outer shape if needed.
  if (axis > 1) {
    const auto x_shape_2 = dummy_->NewDummyName();
    nodes.emplace_back(MakeNode("Shape", {x}, {x_shape_2}));

    const auto x_shape_outer = dummy_->NewDummyName();
    nodes.emplace_back(MakeNode(
        "Slice",
        {x_shape_2},
        {x_shape_outer},
        std::vector<AttributeProto>{
            MakeAttribute("starts", std::vector<int64_t>{0}),
            MakeAttribute("ends", std::vector<int64_t>{axis}),
        }));

    const auto y_shape = dummy_->NewDummyName();
    const_tensors.emplace_back(CreateOnnxShapeTensor(dummy_, {-1}));
    nodes.emplace_back(MakeNode(
        "Concat",
        {x_shape_outer, const_tensors.back().name()},
        {y_shape},
        std::vector<AttributeProto>{
            MakeAttribute("axis", static_cast<int64_t>(0)),
        }));

    nodes.emplace_back(MakeNode("Reshape", {gemm_y_output, y_shape}, {y}));
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
  for (const auto& arg : op.arg()) {
    if (arg.name() == "values") {
      values = &arg;
    } else if (arg.name() == "shape") {
      shape = &arg;
    }
  }

  CAFFE_ENFORCE(values);
  CAFFE_ENFORCE(shape);

  // Set dims
  for (const auto i : shape->ints()) {
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
