#include "caffe2/opt/onnxifi_transformer.h"

#include <iostream>
#include <unordered_set>

#include "onnx/proto_utils.h"

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/onnx/onnx_exporter.h"
#include "caffe2/opt/backend_cutting.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/utils/string_utils.h"

namespace caffe2 {

namespace {
const std::string kRealBatchSizeBlob = "real_batch_size";
const std::string kInitializers = "initializers";
constexpr size_t kBufferSize = 64;

// Convert ShapeInfo map to TensorShape map
std::unordered_map<std::string, TensorShape> stripShapeInfoMap(
    const ShapeInfoMap& info_map) {
  std::unordered_map<std::string, TensorShape> shape_map;
  for (const auto& kv : info_map) {
    shape_map.emplace(kv.first, kv.second.shape);
  }
  return shape_map;
}

uint64_t onnxifiDataType(caffe2::TensorProto::DataType t) {
#define CAFFE2_TO_ONNXIFI_TYPE(x, y) \
  case (caffe2::TensorProto::x):     \
    return y
  switch (t) {
    CAFFE2_TO_ONNXIFI_TYPE(FLOAT, ONNXIFI_DATATYPE_FLOAT32);
    CAFFE2_TO_ONNXIFI_TYPE(INT8, ONNXIFI_DATATYPE_INT8);
    CAFFE2_TO_ONNXIFI_TYPE(UINT8, ONNXIFI_DATATYPE_UINT8);
    CAFFE2_TO_ONNXIFI_TYPE(INT16, ONNXIFI_DATATYPE_INT16);
    CAFFE2_TO_ONNXIFI_TYPE(UINT16, ONNXIFI_DATATYPE_UINT16);
    CAFFE2_TO_ONNXIFI_TYPE(INT32, ONNXIFI_DATATYPE_INT32);
    CAFFE2_TO_ONNXIFI_TYPE(INT64, ONNXIFI_DATATYPE_INT64);
    CAFFE2_TO_ONNXIFI_TYPE(FLOAT16, ONNXIFI_DATATYPE_FLOAT16);
    default:
      LOG(WARNING) << "Unsupported Caffe2 tensor type: " << t
                   << ", fallback to FLOAT";
      return ONNXIFI_DATATYPE_FLOAT32;
  }
#undef CAFFE2_TO_ONNXIFI_TYPE
}

std::vector<::ONNX_NAMESPACE::ValueInfoProto> convertToValueInfo(
    const std::vector<std::string>& names,
    const std::unordered_map<std::string, TensorShape>& shape_hints,
    const std::unordered_map<std::string, ::ONNX_NAMESPACE::TypeProto>&
        extra_shape_hints) {
  std::vector<::ONNX_NAMESPACE::ValueInfoProto> r;
  for (const auto& s : names) {
    r.emplace_back();
    auto& value_info = r.back();
    value_info.set_name(s);
    const auto it = shape_hints.find(s);
    if (it == shape_hints.end()) {
      const auto eit = extra_shape_hints.find(s);
      if (eit == extra_shape_hints.end()) {
        LOG(WARNING) << "Cannot get shape of " << s;
      } else {
        value_info.mutable_type()->CopyFrom(eit->second);
      }
    } else {
      auto* tensor_type = value_info.mutable_type()->mutable_tensor_type();
      tensor_type->set_elem_type(
          onnx::Caffe2TypeToOnnxType(it->second.data_type()));
      auto* shape = tensor_type->mutable_shape();
      for (int i = 0; i < it->second.dims().size(); ++i) {
        shape->add_dim()->set_dim_value(it->second.dims(i));
      }
    }
  }
  return r;
}

// Given a net, with primiary inputs and outputs defined in its
// external_inputs/outputs, and given the set of weights and extra weights
// (created during conversion to ONNX if exists), we check whether some of the
// weights are used in the net, and if so, we put it in the initialize_list and
// add it to the external_inputs too.
// \param net [in] c2 net (cutoff from a bigger net)
// \param weights_in_ws [in] all the weights in the workspace
// \param extra_weights [in] extra weights possibly generated during ONNX
// conversion \param initialization_list [out] weights that needs to be offload
// to backend \param total_inputs_vec [out] total #inputs of the net that
// doesn't have a producer
void getWeightsAndInputs(
    const NetDef& net,
    const std::unordered_set<std::string>& weights_in_ws,
    const std::vector<std::string>& extra_weights,
    std::unordered_set<std::string>* initialization_list,
    std::vector<std::string>* total_inputs_vec) {
  std::unordered_set<std::string> total_inputs;

  // extra weights is definitely extra weights/inputs
  for (const auto& extra_weight : extra_weights) {
    if (total_inputs.emplace(extra_weight).second) {
      total_inputs_vec->emplace_back(extra_weight);
    }
    initialization_list->emplace(extra_weight);
  }

  // Boundary inputs that should not be weights
  std::unordered_set<std::string> boundary_inputs;
  for (const auto& i : net.external_input()) {
    boundary_inputs.emplace(i);
  }

  for (const auto& op : net.op()) {
    for (const auto& input : op.input()) {
      bool not_seen = total_inputs.emplace(input).second;
      if (!not_seen) {
        continue;
      }
      if (weights_in_ws.count(input)) {
        // We add weights as inputs too
        total_inputs_vec->emplace_back(input);
        initialization_list->emplace(input);
        VLOG(2) << "Add weights: " << input;
      } else if (boundary_inputs.count(input)) {
        VLOG(2) << "Adding boundary input: " << input;
        total_inputs_vec->emplace_back(input);
      }
    }
  }
}

void collectInputsAndOutputs(
    const OperatorDef& op,
    std::set<std::string>* inputs,
    std::set<std::string>* outputs) {
  for (const auto& blob : op.input()) {
    inputs->emplace(blob);
  }
  for (const auto& blob : op.output()) {
    outputs->emplace(blob);
  }
}

void fetchInputsToIfOpsSubnet(NetDef* net) {
  NetDef clone(*net);
  clone.clear_op();
  for (auto& op : net->op()) {
    if (op.type() == "If" || op.type() == "AsyncIf") {
      OperatorDef new_op(op);
      ArgumentHelper helper(op);
      std::set<std::string> subnet_inputs, subnet_outputs;
      if (helper.HasSingleArgumentOfType<NetDef>("then_net")) {
        auto then_net = helper.GetSingleArgument<NetDef>("then_net", NetDef());
        for (const auto& nested_op : then_net.op()) {
          collectInputsAndOutputs(nested_op, &subnet_inputs, &subnet_outputs);
        }
      }
      if (helper.HasSingleArgumentOfType<NetDef>("else_net")) {
        auto else_net = helper.GetSingleArgument<NetDef>("else_net", NetDef());
        for (const auto& nested_op : else_net.op()) {
          collectInputsAndOutputs(nested_op, &subnet_inputs, &subnet_outputs);
        }
      }
      for (const std::string& blob : subnet_inputs) {
        if (subnet_outputs.count(blob) == 0) {
          new_op.add_input(blob);
        }
      }
      clone.add_op()->CopyFrom(new_op);
    } else {
      clone.add_op()->CopyFrom(op);
    }
  }
  net->Swap(&clone);
}

void fillModelInfo(::ONNX_NAMESPACE::ModelProto* model) {
  model->set_ir_version(::ONNX_NAMESPACE::Version::IR_VERSION);
  model->set_producer_name("caffe2");
  auto* opset_id = model->add_opset_import();
  opset_id->set_domain("");
  opset_id->set_version(7);
}

int64_t getBlob1stDimSize(const ShapeInfo& shape_info) {
  if (shape_info.shape.dims_size() == 0) {
    return 0;
  } else {
    return shape_info.shape.dims(0);
  }
}

NetDef composeResultNet(const OperatorDef& onnxifi_op) {
  NetDef net_opt;
  net_opt.add_op()->CopyFrom(onnxifi_op);
  return net_opt;
}

void enforceFp32InputsToFp16(
    const std::unordered_set<std::string>& weights,
    NetDef* pred_net,
    ShapeInfoMap* shape_hints) {
  std::unordered_map<std::string, ShapeInfo> user_input_map;
  for (const auto& i : pred_net->external_input()) {
    if (weights.count(i)) {
      continue;
    }
    auto it = shape_hints->find(i);
    if (it == shape_hints->end() ||
        it->second.shape.data_type() != TensorProto_DataType_FLOAT) {
      continue;
    }
    auto& shape_info = it->second;
    user_input_map[i] = shape_info;
    shape_info.shape.set_data_type(TensorProto_DataType_FLOAT16);
  }

  if (user_input_map.empty()) {
    return;
  }

  std::vector<OperatorDef> ops;
  for (const auto& op : pred_net->op()) {
    ops.emplace_back(op);
  }
  pred_net->clear_op();
  int current_pos = ops.size();

  const char kBridgeTensorSuffix[] = "_to_float_bridge";
  std::vector<OperatorDef> converts;
  for (const auto& elem : user_input_map) {
    const auto& name = elem.first;
    const auto& shape_info = elem.second;
    std::string new_name = name + kBridgeTensorSuffix;
    shape_hints->emplace(new_name, shape_info);
    converts.emplace_back(CreateOperatorDef(
        "HalfToFloat",
        "",
        {name},
        {new_name},
        {MakeArgument<int>(kNetPos, current_pos++)}));
  }
  for (const auto& op : converts) {
    pred_net->add_op()->CopyFrom(op);
  }

  for (auto& op : ops) {
    for (auto& input : *op.mutable_input()) {
      if (user_input_map.count(input)) {
        input += kBridgeTensorSuffix;
      }
    }
  }

  for (const auto& op : ops) {
    pred_net->add_op()->CopyFrom(op);
  }
}

void mergeFp32InputsAndConvertToFp16(
    size_t batch_size,
    const std::unordered_set<std::string>& weights,
    NetDef* pred_net,
    ShapeInfoMap* shape_hints) {
  std::unordered_map<std::string, ShapeInfo> user_input_map;
  for (const auto& i : pred_net->external_input()) {
    if (weights.count(i)) {
      continue;
    }
    const auto it = shape_hints->find(i);
    // Heuristic: the input has to be of float type, 2-dimensional and the first
    // dimension has to be of batch size
    if (it == shape_hints->end() ||
        it->second.shape.data_type() != TensorProto_DataType_FLOAT) {
      continue;
    }
    auto shape_info = it->second;
    if (shape_info.shape.dims_size() != 2 ||
        shape_info.shape.dims(0) != batch_size) {
      continue;
    }
    shape_info.shape.set_data_type(TensorProto_DataType_FLOAT16);

    user_input_map[i] = shape_info;
  }

  if (user_input_map.empty()) {
    return;
  }
  std::unordered_map<std::string, std::vector<std::string>>
      user_inputs_by_partition;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      user_input_set_by_partition;
  for (const auto& op : pred_net->op()) {
    for (const auto& i : op.input()) {
      if (user_input_map.find(i) != user_input_map.end()) {
        const auto& partition = op.device_option().node_name().empty()
            ? "default"
            : op.device_option().node_name();
        if (user_input_set_by_partition[partition].find(i) ==
            user_input_set_by_partition[partition].end()) {
          user_inputs_by_partition[partition].emplace_back(i);
          user_input_set_by_partition[partition].insert(i);
        }
      }
    }
  }

  std::vector<OperatorDef> ops;
  for (const auto& op : pred_net->op()) {
    ops.emplace_back(op);
  }
  pred_net->clear_op();
  int current_pos = ops.size();

  for (const auto& elem : user_inputs_by_partition) {
    const auto& partition = elem.first;
    const auto& user_inputs = elem.second;
    const auto& user_input_set = user_input_set_by_partition[partition];

    OperatorDef op1;
    op1.set_type("Concat");
    for (const auto& i : user_inputs) {
      op1.add_input(i);
    }
    op1.add_output(partition + "_fp32_input_concated");
    op1.add_output(partition + "_fp32_input_concated_split_info");
    auto shape_info = user_input_map[user_inputs.front()];
    int total = 0;
    for (const auto& u : user_inputs) {
      total += user_input_map[u].shape.dims(1);
    }
    shape_info.shape.set_dims(1, total);
    AddArgument("axis", 1, &op1);
    AddArgument(kNetPos, current_pos++, &op1);
    pred_net->add_op()->CopyFrom(op1);

    // TODO: a possible optimization is to fuse the fp16 conversion into Concat
    OperatorDef op2;
    op2.set_type("FloatToHalf");
    op2.add_input(partition + "_fp32_input_concated");
    op2.add_output(partition + "_fp16_input_concated");
    AddArgument("clip", 1, &op2);
    AddArgument(kNetPos, current_pos++, &op2);
    shape_hints->emplace(partition + "_fp16_input_concated", shape_info);
    pred_net->add_op()->CopyFrom(op2);

    OperatorDef op3;
    op3.set_type("Split");
    op3.add_input(partition + "_fp16_input_concated");
    op3.mutable_device_option()->set_node_name(partition);

    std::vector<OperatorDef> converts;
    for (const auto& i : user_inputs) {
      std::string new_name = partition + "_" + i + "_split_fp16";
      op3.add_output(new_name);
      shape_hints->emplace(new_name, user_input_map[i]);
      converts.emplace_back(CreateOperatorDef(
          "HalfToFloat",
          "",
          {partition + "_" + i + "_split_fp16"},
          {partition + "_" + i + "_split"},
          {MakeArgument<int>(kNetPos, current_pos++)}));
      converts.back().mutable_device_option()->set_node_name(partition);

      auto converted_shape = user_input_map[i];
      converted_shape.shape.set_data_type(TensorProto_DataType_FLOAT);
      shape_hints->emplace(partition + "_" + i + "_split", converted_shape);
    }
    AddArgument("axis", 1, &op3);
    AddArgument(kNetPos, current_pos++, &op3);
    auto* arg = op3.add_arg();
    arg->set_name("split");
    for (const auto& u : user_inputs) {
      arg->add_ints(user_input_map[u].shape.dims(1));
    }
    pred_net->add_op()->CopyFrom(op3);
    for (const auto& op : converts) {
      pred_net->add_op()->CopyFrom(op);
    }

    for (auto& op : ops) {
      if ((!op.device_option().node_name().empty() &&
           op.device_option().node_name() == partition) ||
          (op.device_option().node_name().empty() && partition == "default")) {
        for (auto& i : *op.mutable_input()) {
          if (user_input_set.count(i)) {
            i = partition + "_" + i + "_split";
          }
        }
      }
    }
  }

  for (const auto& op : ops) {
    pred_net->add_op()->CopyFrom(op);
  }
}

} // namespace

void splitSparseLengthsSumSparse(NetDef* net, const Workspace& ws) {
  const static std::unordered_map<string, string> slss = {
      {"SparseLengthsSum4BitRowwiseSparse", "SparseLengthsSumFused4BitRowwise"},
      {"SparseLengthsWeightedSum4BitRowwiseSparse",
       "SparseLengthsWeightedSumFused4BitRowwise"},
      {"SparseLengthsSum8BitRowwiseSparse", "SparseLengthsSumFused8BitRowwise"},
      {"SparseLengthsWeightedSum8BitRowwiseSparse",
       "SparseLengthsWeightedSumFused8BitRowwise"},
      {"SparseLengthsSum2BitRowwiseSparse", "SparseLengthsSumFused2BitRowwise"},
      {"SparseLengthsWeightedSum2BitRowwiseSparse",
       "SparseLengthsWeightedSumFused2BitRowwise"}};
  NetDef new_net;
  new_net.CopyFrom(*net);
  new_net.mutable_op()->Clear();
  for (const auto& op : net->op()) {
    const auto it = slss.find(op.type());
    if (it == slss.end()) {
      new_net.add_op()->CopyFrom(op);
    } else {
      const bool is_weighted =
          (op.type().find("Weighted") != std::string::npos);
      const auto& compressed_mapping = op.input(is_weighted ? 4 : 3);
      const auto* b = ws.GetBlob(compressed_mapping);
      bool fallback = false;
      if (b && b->IsType<Tensor>()) {
        const auto& t = BlobGetTensor(*b, CPU);
        fallback = ((t.numel() == 1) && (t.template data<int32_t>()[0] == 0));
      }

      if (fallback) {
        // If fallback, we just replace the original slss op with a normal sls
        // op
        OperatorDef new_op;
        new_op.CopyFrom(op);
        new_op.set_type(it->second);
        new_op.mutable_input()->RemoveLast();
        new_net.add_op()->CopyFrom(new_op);
      } else {
        // Otherwise, we replace slss with slss_lookup followed by a normal sls
        OperatorDef new_op;
        new_op.CopyFrom(op);
        new_op.set_type("SparseLengthsSumSparseLookup");
        new_op.clear_input();
        const auto& indices_in = is_weighted ? op.input(2) : op.input(1);
        const auto& lengths_in = is_weighted ? op.input(3) : op.input(2);
        const auto& compress_mapping = is_weighted ? op.input(4) : op.input(3);
        const auto& weights_in = is_weighted ? op.input(1) : "";
        new_op.add_input(indices_in);
        new_op.add_input(lengths_in);
        new_op.add_input(compress_mapping);
        const auto indices_out = indices_in + "_decomp";
        const auto lengths_out = lengths_in + "_decomp";
        const auto weights_out = weights_in + "_decomp";
        new_op.clear_output();
        new_op.add_output(indices_out);
        new_op.add_output(lengths_out);
        if (is_weighted) {
          new_op.add_input(weights_in);
          new_op.add_output(weights_out);
        }
        new_net.add_op()->CopyFrom(new_op);

        new_op.CopyFrom(op);
        new_op.set_type(it->second);
        new_op.mutable_input()->RemoveLast();
        *new_op.mutable_input()->Mutable(is_weighted ? 2 : 1) = indices_out;
        *new_op.mutable_input()->Mutable(is_weighted ? 3 : 2) = lengths_out;
        if (is_weighted) {
          *new_op.mutable_input()->Mutable(1) = weights_out;
        }
        new_net.add_op()->CopyFrom(new_op);
      }
    }
  }

  new_net.Swap(net);
}

OnnxifiOptionHelper::OnnxifiOptionHelper() {
  lib_ = onnx::initOnnxifiLibrary();
  CAFFE_ENFORCE(lib_, "Cannot initialize ONNXIFI library");
}

bool OnnxifiOptionHelper::setOnnxifiOption(
    const std::string& option,
    const std::string& value) {
#ifdef ONNXIFI_ENABLE_EXT
  onnxStatus (*onnxSetOptionFunctionPointer)(
      const char* optionName, const char* optionValue) = nullptr;
  union {
    onnxExtensionFunctionPointer p;
    decltype(onnxSetOptionFunctionPointer) set;
  } u{};
  onnxBackendID backend_id = nullptr;
  if (lib_->onnxGetExtensionFunctionAddress(
          backend_id, "onnxSetOptionFunction", &u.p) !=
      ONNXIFI_STATUS_SUCCESS) {
    LOG(ERROR) << "Cannot find onnxSetOptionFunction";
    return false;
  } else {
    onnxSetOptionFunctionPointer = u.set;
  }
  if (onnxSetOptionFunctionPointer != nullptr &&
      (*onnxSetOptionFunctionPointer)(option.c_str(), value.c_str()) ==
          ONNXIFI_STATUS_SUCCESS) {
    return true;
  }
#endif
  return false;
}

std::string OnnxifiOptionHelper::getOnnxifiOption(const std::string& option) {
#ifdef ONNXIFI_ENABLE_EXT
  onnxStatus (*onnxGetOptionFunctionPointer)(
      const char* optionName, char* optionValue, size_t* optionValueLength) =
      nullptr;
  union {
    onnxExtensionFunctionPointer p;
    decltype(onnxGetOptionFunctionPointer) get;
  } u{};
  onnxBackendID backend_id = nullptr;
  if (lib_->onnxGetExtensionFunctionAddress(
          backend_id, "onnxGetOptionFunction", &u.p) !=
      ONNXIFI_STATUS_SUCCESS) {
    LOG(ERROR) << "Cannot find onnxGetOptionFunction";
    return "";
  } else {
    onnxGetOptionFunctionPointer = u.get;
  }

  constexpr size_t ll = 1024;
  char buf[ll];
  size_t len = ll;
  if (onnxGetOptionFunctionPointer != nullptr &&
      (*onnxGetOptionFunctionPointer)(option.c_str(), buf, &len) ==
          ONNXIFI_STATUS_SUCCESS) {
    return std::string(buf, len);
  }
#endif

  return "";
}

OnnxifiTransformer::OnnxifiTransformer(const OnnxifiTransformerOptions& opts)
    : BackendTransformerBase(), opts_(opts) {
  lib_ = onnx::initOnnxifiLibrary();
  CAFFE_ENFORCE(lib_, "Cannot initialize ONNXIFI library");
  CAFFE_ENFORCE_EQ(
      lib_->onnxGetBackendIDs(nullptr, &num_backends_),
      ONNXIFI_STATUS_FALLBACK);
  CAFFE_ENFORCE_GT(
      num_backends_, 0, "At least 1 onnxifi backend should be available");
  backend_ids_.resize(num_backends_);
  CAFFE_ENFORCE_EQ(
      lib_->onnxGetBackendIDs(backend_ids_.data(), &num_backends_),
      ONNXIFI_STATUS_SUCCESS);
}

OnnxifiTransformer::~OnnxifiTransformer() {
  for (unsigned i = 0; i < num_backends_; ++i) {
    if (lib_->onnxReleaseBackendID(backend_ids_[i]) != ONNXIFI_STATUS_SUCCESS) {
      LOG(ERROR) << "Error when calling onnxReleaseBackendID";
    }
  }
}

bool OnnxifiTransformer::canPassOutputShapeHintsPerBs(
    const OperatorDef& op,
    const std::unordered_map<int, ShapeInfoMap>& shape_hints_per_bs) const {
  if (shape_hints_per_bs.empty()) {
    return false;
  }

  for (int bs = 1; bs < opts_.bound_shape_spec.max_batch_size; ++bs) {
    auto shape_hints_search = shape_hints_per_bs.find(bs);
    if (shape_hints_search == shape_hints_per_bs.end()) {
      return false;
    }
    const auto& shape_hints = shape_hints_search->second;

    for (int output_idx = 0; output_idx < op.output_size(); ++output_idx) {
      auto shape_hint_search = shape_hints.find(op.output(output_idx));
      if (shape_hint_search == shape_hints.end()) {
        return false;
      }
    }
  }

  return true;
}

OperatorDef OnnxifiTransformer::buildOnnxifiOp(
    const std::string& onnx_model_str,
    const std::unordered_set<std::string>& initialization_list,
    const std::vector<std::string>& external_inputs,
    const std::vector<std::string>& external_outputs,
    const ShapeInfoMap& shape_hints_max_bs,
    const std::unordered_map<int, ShapeInfoMap>& shape_hints_per_bs) {
  OperatorDef op;
  op.set_type("Onnxifi");
  auto* onnx_model_arg = op.add_arg();
  onnx_model_arg->set_name("onnx_model");
  onnx_model_arg->set_s(onnx_model_str);

  // Add the names of the initializer blobs that we want to fetch from the
  // workspace later
  auto* initializers_arg = op.add_arg();
  initializers_arg->set_name(kInitializers);
  for (const auto& s : initialization_list) {
    initializers_arg->add_strings(s);
  }

  // Add the input/output
  int idx = 0;
  auto* input_names = op.add_arg();
  input_names->set_name("input_names");
  for (const auto& input : external_inputs) {
    if (!initialization_list.count(input)) {
      op.add_input(input);
      input_names->add_strings(input);
    }
  }
  auto* output_names = op.add_arg();
  output_names->set_name("output_names");
  for (const auto& output : external_outputs) {
    op.add_output(output);
    output_names->add_strings(output);
  }

  // Find out the index of input that has a nominal batch size
  const auto max_batch_size = opts_.bound_shape_spec.max_batch_size;
  idx = 0;
  int nominal_batch_idx{0};
  for (const auto& input : external_inputs) {
    if (!initialization_list.count(input)) {
      const auto it = shape_hints_max_bs.find(input);
      CAFFE_ENFORCE(
          it != shape_hints_max_bs.end(),
          "Input shape for ",
          input,
          " not found");
      const auto& info = it->second;
      if (info.getDimType(0) == TensorBoundShape_DimType_BATCH &&
          getBlob1stDimSize(info) == max_batch_size) {
        nominal_batch_idx = idx;
        break;
      }
      ++idx;
    }
  }

  // Add output size hints for max batch size
  auto* output_shape_info_arg = op.add_arg();
  output_shape_info_arg->set_name("output_shape_info");
  auto* output_qshape_info_arg = op.add_arg();
  output_qshape_info_arg->set_name("output_qshape_info");
  for (int i = 0; i < op.output_size(); ++i) {
    const auto& o = op.output(i);
    const auto it = shape_hints_max_bs.find(o);
    if (it != shape_hints_max_bs.end()) {
      if (!it->second.is_quantized) {
        output_shape_info_arg->mutable_tensors()->Add()->CopyFrom(
            wrapShapeInfoIntoTensorProto(o, it->second));
      } else {
        output_qshape_info_arg->mutable_qtensors()->Add()->CopyFrom(
            wrapShapeInfoIntoQTensorProto(o, it->second));
      }
      VLOG(2) << "Adding output hint: " << o;
    }
  }

  // Add output size hints per batch size
  if (canPassOutputShapeHintsPerBs(op, shape_hints_per_bs)) {
    VLOG(2) << "Passing in output shape hints for batch sizes in [1, "
            << opts_.bound_shape_spec.max_batch_size << ")";
    AddArgument("use_passed_output_shapes", 1, &op);

    for (int bs = 1; bs < opts_.bound_shape_spec.max_batch_size; ++bs) {
      auto* output_shape_arg = op.add_arg();
      output_shape_arg->set_name("output_shapes_bs_" + caffe2::to_string(bs));
      auto* output_qshape_arg = op.add_arg();
      output_qshape_arg->set_name("output_qshapes_bs_" + caffe2::to_string(bs));

      const auto& shape_hints = shape_hints_per_bs.find(bs)->second;

      for (int output_idx = 0; output_idx < op.output_size(); ++output_idx) {
        const auto& output_name = op.output(output_idx);
        const auto& shape_hint = shape_hints.find(output_name)->second;
        if (!shape_hint.is_quantized) {
          output_shape_arg->mutable_tensors()->Add()->CopyFrom(
              wrapShapeInfoIntoTensorProto(output_name, shape_hint));
        } else {
          output_shape_arg->mutable_qtensors()->Add()->CopyFrom(
              wrapShapeInfoIntoQTensorProto(output_name, shape_hint));
        }
      }
    }
  } else {
    AddArgument("use_passed_output_shapes", 0, &op);
  }

  // Tell Onnxifi op that the model is in onnx or c2 proto format
  AddArgument("use_onnx", opts_.use_onnx ? 1 : 0, &op);

  // Tell Onnxifi op which backend id to use
  AddArgument("backend_id", idx_, &op);

  // Add model_id and net_pos to the onnxifi model
  AddArgument(kModelId, model_id_, &op);
  AddArgument(kNetPos, c10::to_string(onnxifi_op_id_++), &op);

  // Add output resizing hints
  if (opts_.adjust_batch) {
    AddArgument("adjust_output_batch", 1, &op);
  } else {
    AddArgument("adjust_output_batch", 0, &op);
  }
  AddArgument("max_batch_size", opts_.bound_shape_spec.max_batch_size, &op);
  AddArgument("max_seq_size", opts_.bound_shape_spec.max_seq_size, &op);
  AddArgument("timeout", opts_.timeout, &op);
  AddArgument("nominal_batch_idx", nominal_batch_idx, &op);

  return op;
}

NetDef OnnxifiTransformer::SubnetToOnnxifiOpViaC2(
    const caffe2::NetDef& net,
    const std::unordered_set<std::string>& weights_in_ws,
    const ShapeInfoMap& shape_hints_max_bs,
    const std::unordered_map<int, ShapeInfoMap>& shape_hints_per_bs) {
  int onnxifi_op_id = onnxifi_op_id_;
  if (opts_.debug) {
    WriteProtoToTextFile(
        net,
        "debug_original_net_" + c10::to_string(onnxifi_op_id) + ".pb_txt",
        false);
  }
  if (opts_.min_ops > net.op_size()) {
    return net;
  }
  // We already have all the ops and external inputs and outputs!
  NetDef onnxifi_net(net);

  // Remove the second output of Concat/Reshape from external_output. Remove
  // rest of the outputs of LayerNorm too. In addition, we remove those outputs
  // from the Onnxifi op too.
  // TODO: This approach is a bit hacky as we assume that the second output is
  // never used. A more appropriate approach can be learned from the ONNX path,
  // where we statically computes the split_info given input shape and insert a
  // GivenTensorIntFill op
  std::unordered_set<std::string> split_infos;
  for (auto& op : *onnxifi_net.mutable_op()) {
    if ((op.type() == "Concat" || op.type() == "Reshape") &&
        op.output_size() == 2) {
      split_infos.emplace(op.output(1));
    } else if (
        op.type() == "SparseLengthsSum" ||
        op.type() == "SparseLengthsSumFused8BitRowwise" ||
        op.type() == "SparseLengthsWeightedSum" ||
        op.type() == "SparseLengthsWeightedSumFused8BitRowwise" ||
        op.type() == "SparseLengthsSumFused4BitRowwise" ||
        op.type() == "SparseLengthsWeightedSumFused4BitRowwise") {
      int weighted = (op.type() == "SparseLengthsWeightedSum" ||
                      op.type() == "SparseLengthsWeightedSumFused8BitRowwise" ||
                      op.type() == "SparseLengthsWeightedSumFused4BitRowwise")
          ? 1
          : 0;
      const auto& indices_hint = shape_hints_max_bs.at(op.input(1 + weighted));
      const auto& lengths_hint = shape_hints_max_bs.at(op.input(2 + weighted));
      const auto& indices_shape = indices_hint.shape;
      const auto& lengths_shape = lengths_hint.shape;
      if ((indices_hint.getDimType(0) ==
               TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX ||
           indices_hint.getDimType(0) ==
               TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX_DEFAULT) &&
          indices_shape.dims_size() == 1 && lengths_shape.dims_size() == 1 &&
          indices_shape.dims(0) == lengths_shape.dims(0)) {
        op.add_arg()->CopyFrom(MakeArgument<int>("length1", 1));
      }
    } else if (op.type() == "LayerNorm" && op.output_size() > 1) {
      for (int i = 1; i < op.output_size(); ++i) {
        split_infos.emplace(op.output(i));
      }
    }
  }
  onnxifi_net.clear_external_output();
  for (const auto& o : net.external_output()) {
    if (!split_infos.count(o)) {
      onnxifi_net.add_external_output(o);
    }
  }

  // Figure out weights and add it to external_inputs too
  std::unordered_set<std::string> initialization_list;
  std::vector<std::string> total_inputs_vec;
  getWeightsAndInputs(
      net,
      weights_in_ws,
      std::vector<std::string>(),
      &initialization_list,
      &total_inputs_vec);
  auto* shape_arg = onnxifi_net.add_arg();
  auto* qshape_arg = onnxifi_net.add_arg();
  shape_arg->set_name("input_shape_info");
  qshape_arg->set_name("input_qshape_info");
  std::sort(total_inputs_vec.begin(), total_inputs_vec.end());
  onnxifi_net.clear_external_input();
  for (const auto& i : total_inputs_vec) {
    onnxifi_net.add_external_input(i);
    auto info = shape_hints_max_bs.at(i);
    if (!info.is_quantized) {
      shape_arg->mutable_tensors()->Add()->CopyFrom(
          wrapShapeInfoIntoTensorProto(i, shape_hints_max_bs.at(i)));
    } else {
      qshape_arg->mutable_qtensors()->Add()->CopyFrom(
          wrapShapeInfoIntoQTensorProto(i, shape_hints_max_bs.at(i)));
    }
  }

  // Add partition info
  for (const auto& p : partition_infos_) {
    onnxifi_net.add_partition_info()->CopyFrom(p);
  }

  // Add initializers (weights) list to the net as an arg
  auto* w_arg = onnxifi_net.add_arg();
  w_arg->set_name(kInitializers);
  for (const auto& i : initialization_list) {
    w_arg->add_strings(i);
  }

  // Build ONNXIFI Op
  std::string model_str;
  onnxifi_net.SerializeToString(&model_str);
  std::vector<std::string> onnxifi_net_inputs(
      onnxifi_net.external_input().begin(), onnxifi_net.external_input().end());
  std::vector<std::string> onnxifi_net_outputs(
      onnxifi_net.external_output().begin(),
      onnxifi_net.external_output().end());
  auto onnxifi_op = buildOnnxifiOp(
      model_str,
      initialization_list,
      onnxifi_net_inputs,
      onnxifi_net_outputs,
      shape_hints_max_bs,
      shape_hints_per_bs);
  NetDef net_opt = composeResultNet(onnxifi_op);

  // Debugging stuff
  if (opts_.debug) {
    WriteProtoToTextFile(
        onnxifi_net,
        "debug_onnxifi_net_" + c10::to_string(onnxifi_op_id) + ".pb_txt",
        false);
    WriteProtoToTextFile(
        net_opt,
        "debug_optimized_net_" + c10::to_string(onnxifi_op_id) + ".pb_txt",
        false);
  }
  return net_opt;
}

NetDef OnnxifiTransformer::SubnetToOnnxifiOpViaOnnx(
    const caffe2::NetDef& net,
    const std::unordered_set<std::string>& weights_in_ws,
    Workspace* ws,
    onnx::OnnxExporter* exporter,
    ShapeInfoMap* shape_hints_max_bs,
    const std::unordered_map<int, ShapeInfoMap>& shape_hints_per_bs) {
  if (opts_.min_ops > net.op_size()) {
    return net;
  }
  ::ONNX_NAMESPACE::ModelProto onnx_model;
  fillModelInfo(&onnx_model);

  caffe2::NetDef onnxifi_net(net);

  // Convert c2 ops to onnx ops, add const weights if there are any
  DeviceOption option;
  CPUContext context(option);
  context.SwitchToDevice();
  std::vector<std::string> extra_weights;
  for (const auto& op : onnxifi_net.op()) {
    const auto results = exporter->Caffe2OpToOnnxNodes(op, shape_hints_onnx_);
    for (const auto& n : results.first) {
      onnx_model.mutable_graph()->add_node()->CopyFrom(n);
    }
    for (const auto& t : results.second) {
      VLOG(2) << "Adding extra init tensor: " << t.name();
      TensorShape shape;
      shape.mutable_dims()->CopyFrom(t.dims());
      auto ret = shape_hints_onnx_.emplace(t.name(), std::move(shape));
      shape_hints_max_bs->emplace(
          std::piecewise_construct,
          std::forward_as_tuple(ret.first->first),
          std::forward_as_tuple(
              std::vector<TensorBoundShape::DimType>(
                  shape.dims_size(), TensorBoundShape_DimType_CONSTANT),
              ret.first->second));

      // Feed into workspace as CPU Tensors
      auto* blob = ws->CreateBlob(t.name());
      auto* cpu_tensor = BlobGetMutableTensor(blob, CPU);
      std::vector<int64_t> dims;
      for (const auto& d : t.dims()) {
        dims.push_back(d);
      }
      cpu_tensor->Resize(dims);
      if (t.data_type() == ::ONNX_NAMESPACE::TensorProto::FLOAT) {
        context.CopyBytesSameDevice(
            cpu_tensor->numel() * sizeof(float),
            static_cast<const void*>(t.raw_data().data()),
            cpu_tensor->raw_mutable_data(TypeMeta::Make<float>()));
      } else if (t.data_type() == ::ONNX_NAMESPACE::TensorProto::INT64) {
        context.CopyBytesSameDevice(
            cpu_tensor->numel() * sizeof(int64_t),
            static_cast<const void*>(t.raw_data().data()),
            cpu_tensor->raw_mutable_data(TypeMeta::Make<int64_t>()));
      } else {
        CAFFE_THROW(
            "Unsupported tensor data type for conversion: ", t.data_type());
      }
      context.FinishDeviceComputation();

      // Add mappings
      extra_weights.emplace_back(t.name());
    }
  }

  // Convert outputs and compute output shape hints
  std::vector<std::string> onnxifi_net_outputs;
  for (const auto& o : net.external_output()) {
    onnxifi_net_outputs.emplace_back(o);
  }
  auto io_vec = convertToValueInfo(
      onnxifi_net_outputs,
      shape_hints_onnx_,
      std::unordered_map<std::string, ::ONNX_NAMESPACE::TypeProto>());
  for (const auto& i : io_vec) {
    onnx_model.mutable_graph()->add_output()->CopyFrom(i);
  }

  // Convert inputs and figure out weights
  std::unordered_set<std::string> initialization_list;
  std::vector<std::string> onnxifi_net_inputs;
  getWeightsAndInputs(
      net,
      weights_in_ws,
      extra_weights,
      &initialization_list,
      &onnxifi_net_inputs);
  io_vec = convertToValueInfo(
      onnxifi_net_inputs,
      shape_hints_onnx_,
      std::unordered_map<std::string, ::ONNX_NAMESPACE::TypeProto>());
  for (const auto& i : io_vec) {
    onnx_model.mutable_graph()->add_input()->CopyFrom(i);
  }

  // Onnx model is ready. Build ONNXIFI Op
  std::string model_str;
  onnx_model.SerializeToString(&model_str);
  auto onnxifi_op = buildOnnxifiOp(
      model_str,
      initialization_list,
      onnxifi_net_inputs,
      onnxifi_net_outputs,
      *shape_hints_max_bs,
      shape_hints_per_bs);
  NetDef net_opt = composeResultNet(onnxifi_op);

  // Debugging stuff
  if (opts_.debug) {
    WriteProtoToTextFile(onnx_model, "debug_onnxifi_net.onnx_txt", false);
    WriteProtoToTextFile(net_opt, "debug_optimized_net.pb_txt", false);
  }
  return net_opt;
}

bool OnnxifiTransformer::supportOpOnnx(
    const caffe2::OperatorDef& op,
    onnx::OnnxExporter* exporter,
    const std::unordered_set<int>& blocklisted_ops,
    onnxBackendID backend_id) const {
  try {
    int pos =
        ArgumentHelper::GetSingleArgument<OperatorDef, int>(op, kNetPos, -1);
    if (blocklisted_ops.count(pos)) {
      LOG(INFO) << "Skipping blocklisted op " << op.type() << " at pos " << pos;
      return false;
    }
    const OpSchema* schema = OpSchemaRegistry::Schema(op.type());
    // NB: this might not be a hard constraint as we can just export C2
    // domain specific ops to ONNX
    if (!schema || schema->onnx_schema().empty()) {
      LOG(INFO) << "Cannot export c2 op " << op.type()
                << " to onnx as there is no corresponding ONNX schema.";
      return false;
    }

    ::ONNX_NAMESPACE::ModelProto onnx_model;
    fillModelInfo(&onnx_model);
    auto results = exporter->Caffe2OpToOnnxNodes(op, shape_hints_onnx_);
    std::unordered_set<std::string> used_inputs;
    std::unordered_set<std::string> used_outputs;
    std::vector<std::string> boundary_inputs;
    std::vector<std::string> boundary_outputs;
    std::unordered_set<std::string> reshape_info;
    // nodes are in topological order, so we just need to iterate
    for (const auto& n : results.first) {
      onnx_model.mutable_graph()->add_node()->CopyFrom(n);
      for (const auto& i : n.input()) {
        bool is_new = used_inputs.emplace(i).second;
        // The input is not seen and it's not referred by any nodes before as
        // output, we count it as an boundary input
        if (is_new && !used_outputs.count(i)) {
          boundary_inputs.emplace_back(i);
        }
      }
      for (const auto& o : n.output()) {
        used_outputs.emplace(o);
      }

      // For reshape node, if it has more than 1 inputs, we need to feed the
      // second input which contains the shape info
      if (n.op_type() == "Reshape" && n.input_size() > 1) {
        reshape_info.emplace(n.input(1));
      }
    }
    // Second iteration to account all the boundary outputs, which is a newly
    // seen output and is not referred as input before
    used_outputs.clear();
    for (const auto& n : results.first) {
      for (const auto& o : n.output()) {
        bool is_new = used_outputs.emplace(o).second;
        if (is_new && !used_inputs.count(o)) {
          boundary_outputs.emplace_back(o);
        }
      }
    }
    std::unordered_map<std::string, ::ONNX_NAMESPACE::TypeProto>
        extra_shape_hints;
    for (const auto& t : results.second) {
      extra_shape_hints.emplace(t.name(), onnx::ExtraTypeProto(t));
      if (reshape_info.count(t.name())) {
        onnx_model.mutable_graph()->add_initializer()->CopyFrom(t);
      }
    }

    // Add input/output shape info
    auto io_vec = convertToValueInfo(
        boundary_inputs, shape_hints_onnx_, extra_shape_hints);
    for (const auto& i : io_vec) {
      onnx_model.mutable_graph()->add_input()->CopyFrom(i);
    }
    io_vec = convertToValueInfo(
        boundary_outputs, shape_hints_onnx_, extra_shape_hints);
    for (const auto& i : io_vec) {
      onnx_model.mutable_graph()->add_output()->CopyFrom(i);
    }

    std::string onnx_model_str;
    onnx_model.SerializeToString(&onnx_model_str);
    auto ret = lib_->onnxGetBackendCompatibility(
        backend_id, onnx_model_str.size(), onnx_model_str.c_str());
    if (ret != ONNXIFI_STATUS_SUCCESS) {
      LOG(INFO) << "Don't support onnx for " << op.type() << " c2 op (" << ret
                << ")";
      return false;
    } else {
      return true;
    }
  } catch (const std::exception& ex) {
    LOG(ERROR) << "Caught exception when converting op " << op.type()
               << ", what: " << ex.what();
    return false;
  }
}

bool OnnxifiTransformer::supportOpC2(
    const caffe2::OperatorDef& op,
    const ShapeInfoMap& shape_hints,
    const std::unordered_set<std::string>& weights,
    const std::unordered_set<int>& blocklisted_ops,
    onnxBackendID backend_id) const {
  try {
    int pos =
        ArgumentHelper::GetSingleArgument<OperatorDef, int>(op, kNetPos, -1);
    if (blocklisted_ops.count(pos)) {
      LOG(INFO) << "Skipping blocklisted op " << op.type() << " at pos " << pos;
      return false;
    }

    // Build a c2 net with one op
    NetDef net;
    net.add_op()->CopyFrom(op);
    std::unordered_set<std::string> seenExternalInputs;
    for (const auto& i : op.input()) {
      if (seenExternalInputs.count(i)) {
        continue;
      }
      seenExternalInputs.insert(i);
      net.add_external_input(i);
    }
    for (const auto& o : op.output()) {
      net.add_external_output(o);
    }
    // Remove the second output of Concat/Reshape from the external_output
    if ((op.type() == "Concat" || op.type() == "Reshape") &&
        op.output_size() == 2) {
      net.mutable_external_output()->RemoveLast();
    } else if (op.type() == "LayerNorm" && op.output_size() > 1) {
      int remove = op.output_size() - 1;
      for (int i = 0; i < remove; ++i) {
        net.mutable_external_output()->RemoveLast();
      }
    }

    // Encode the input/output shapes to an argument
    auto* shape_arg = net.add_arg();
    auto* qshape_arg = net.add_arg();
    shape_arg->set_name("input_shape_info");
    qshape_arg->set_name("input_qshape_info");
    std::unordered_set<std::string> seenInputsForShapeArgs;
    for (const auto& i : op.input()) {
      if (seenInputsForShapeArgs.count(i)) {
        continue;
      }
      seenInputsForShapeArgs.insert(i);
      const auto it = shape_hints.find(i);
      if (it == shape_hints.end()) {
        VLOG(1) << "Skipping " << op.type() << " (" << pos
                << ") due to missing shape info for input " << i;
        return false;
      }
      if ((it->second).is_quantized == false) {
        shape_arg->mutable_tensors()->Add()->CopyFrom(
            wrapShapeInfoIntoTensorProto(i, it->second));
      } else {
        qshape_arg->mutable_qtensors()->Add()->CopyFrom(
            wrapShapeInfoIntoQTensorProto(i, it->second));
      }
    }

    qshape_arg = net.add_arg();
    shape_arg = net.add_arg();
    shape_arg->set_name("output_shape_info");
    qshape_arg->set_name("output_qshape_info");
    for (const auto& i : op.output()) {
      const auto it = shape_hints.find(i);
      if (it == shape_hints.end()) {
        VLOG(1) << "Skipping " << op.type() << " (" << pos
                << ") due to missing shape info for output " << i;
        return false;
      }
      if ((it->second).is_quantized == false) {
        shape_arg->mutable_tensors()->Add()->CopyFrom(
            wrapShapeInfoIntoTensorProto(i, it->second));
      } else {
        qshape_arg->mutable_qtensors()->Add()->CopyFrom(
            wrapShapeInfoIntoQTensorProto(i, it->second));
      }
    }

    // Annnote the inputs that are weights
    auto w_arg = net.add_arg();
    w_arg->set_name(kInitializers);
    for (const auto& i : op.input()) {
      if (weights.count(i)) {
        w_arg->add_strings(i);
      }
    }

    std::string c2_model_str;
    net.SerializeToString(&c2_model_str);
    auto ret = lib_->onnxGetBackendCompatibility(
        backend_id, c2_model_str.size(), c2_model_str.c_str());
    if (ret != ONNXIFI_STATUS_SUCCESS) {
      LOG(INFO) << "Don't support c2 op " << op.type() << " at pos " << pos
                << " (" << ret << ")";
      return false;
    } else {
      return true;
    }
  } catch (const std::exception& ex) {
    LOG(ERROR) << "Caught exception when converting op " << op.type()
               << ", what: " << ex.what();
    return false;
  }
}

void OnnxifiTransformer::tieGatherAndSparseLengthsWeightedSumOps(
    const NetDef& net,
    const ShapeInfoMap& shape_hints,
    const std::unordered_set<std::string>& weights,
    std::unordered_set<int>* blocklisted_ops) const {
  std::unordered_map<std::string, int> output_pos;
  onnx::OnnxExporter exporter(nullptr);
  onnxBackendID backend_id = backend_ids_[idx_];

  for (const auto& op : net.op()) {
    std::string check;
    if (op.type() == "Gather") {
      int pos =
          ArgumentHelper::GetSingleArgument<OperatorDef, int>(op, kNetPos, -1);
      for (const auto& output : op.output()) {
        output_pos.emplace(output, pos);
      }
    } else if (StartsWith(op.type(), "SparseLengthsWeighted")) {
      auto supported = opts_.use_onnx
          ? supportOpOnnx(op, &exporter, *blocklisted_ops, backend_id)
          : supportOpC2(op, shape_hints, weights, *blocklisted_ops, backend_id);
      if (!supported && op.input_size() > 1) {
        check = op.input(1);
      }
    } else if (
        op.type() == "SparseLengthsSumSparseLookup" && op.input_size() > 3) {
      check = op.input(3);
    }
    if (!check.empty()) {
      const auto it = output_pos.find(check);
      if (it == output_pos.end()) {
        continue;
      }
      blocklisted_ops->emplace(it->second);
      // We know that current op is not going to be supported. Might as well
      // blocklist it too
      blocklisted_ops->emplace(
          ArgumentHelper::GetSingleArgument<OperatorDef, int>(op, kNetPos, -1));
    }
  }
}

void OnnxifiTransformer::blocklistCpuPartition(
    const NetDef& net,
    std::unordered_set<int>* blocklisted_ops) const {
  std::unordered_set<std::string> cpu_partitions;
  for (const auto& p : partition_infos_) {
    if (p.device_id_size() == 0) {
      cpu_partitions.emplace(p.name());
    }
  }
  for (const auto& op : net.op()) {
    const auto& pname = op.device_option().node_name();
    if (cpu_partitions.count(pname)) {
      blocklisted_ops->emplace(
          ArgumentHelper::GetSingleArgument<OperatorDef, int>(op, kNetPos, -1));
    }
  }
}

void OnnxifiTransformer::applyFilteringRules(
    const NetDef& net,
    const ShapeInfoMap& shape_hints,
    const std::unordered_set<std::string>& weights,
    std::unordered_set<int>* blocklisted_ops) const {
  tieGatherAndSparseLengthsWeightedSumOps(
      net, shape_hints, weights, blocklisted_ops);
  blocklistCpuPartition(net, blocklisted_ops);
}

std::vector<onnxBackendID> OnnxifiTransformer::getBackendId() {
  idx_ = 0;

  if (opts_.use_onnx) {
    return backend_ids_;
  }
  // Try to find a backend that support Caffe2 proto. Note that this is quite
  // opportunistic as we don't officially support Caffe2 proto.
  char buf[kBufferSize];
  for (int i = 0; i < backend_ids_.size(); ++i) {
    size_t len = kBufferSize;
    auto ret = lib_->onnxGetBackendInfo(
        backend_ids_[i], ONNXIFI_BACKEND_DEVICE, buf, &len);
    if (ret == ONNXIFI_STATUS_SUCCESS && strstr(buf, "Caffe2")) {
      LOG(INFO) << "Using backend with Caffe2 Proto, ID: " << i;
      idx_ = i;
      break;
    }
  }
  return backend_ids_;
}

NetDef OnnxifiTransformer::TransformViaC2(
    NetDef* pred_net,
    const std::unordered_set<std::string>& weights,
    const std::unordered_set<int>& blocklisted_ops,
    const ShapeInfoMap& shape_hints_max_bs,
    const std::unordered_map<int, ShapeInfoMap>& shape_hints_per_bs) {
  onnxBackendID backend_id = backend_ids_[idx_];

  auto c2_supports =
      [this, &shape_hints_max_bs, &blocklisted_ops, backend_id, &weights](
          const caffe2::OperatorDef& op) {
        return supportOpC2(
            op, shape_hints_max_bs, weights, blocklisted_ops, backend_id);
      };

  auto c2_converter = [this,
                       &weights,
                       &shape_hints_max_bs,
                       &shape_hints_per_bs](const caffe2::NetDef& net) {
    return SubnetToOnnxifiOpViaC2(
        net, weights, shape_hints_max_bs, shape_hints_per_bs);
  };

  return opt::OptimizeForBackend(
      *pred_net, c2_supports, c2_converter, opts_.debug);
}

NetDef OnnxifiTransformer::TransformViaOnnx(
    Workspace* ws,
    NetDef* pred_net,
    const std::unordered_set<std::string>& weights,
    const std::unordered_set<int>& blocklisted_ops,
    ShapeInfoMap* shape_hints_max_bs,
    const std::unordered_map<int, ShapeInfoMap>& shape_hints_per_bs) {
  onnxBackendID backend_id = backend_ids_[idx_];

  // function to tell whether the ONNXIFI backend supports a given C2 op or not
  onnx::OnnxExporter exporter(nullptr);
  auto onnx_supports = [this, &exporter, &blocklisted_ops, backend_id](
                           const caffe2::OperatorDef& op) {
    return supportOpOnnx(op, &exporter, blocklisted_ops, backend_id);
  };

  // function to convert runnable subgraph into an onnxifi op. We need to keep
  // the same exporter throughout the process to avoid duplicated dummy name
  // generation
  onnx::OnnxExporter exporter2(nullptr);
  auto onnx_converter = [this,
                         ws,
                         &weights,
                         shape_hints_max_bs,
                         &exporter2,
                         &shape_hints_per_bs](
                            const caffe2::NetDef& net) mutable {
    return SubnetToOnnxifiOpViaOnnx(
        net, weights, ws, &exporter2, shape_hints_max_bs, shape_hints_per_bs);
  };

  return opt::OptimizeForBackend(
      *pred_net, onnx_supports, onnx_converter, opts_.debug);
}

void OnnxifiTransformer::extractPartitionInfo(const NetDef& net) {
  partition_infos_.clear();
  for (const auto& p : net.partition_info()) {
    partition_infos_.emplace_back(p);
  }
}

// Cutting off the runnable part and replace with ONNXIFI ops. Asssume the nets
// were topologically sorted
void OnnxifiTransformer::transform(
    Workspace* ws,
    NetDef* pred_net,
    const std::vector<std::string>& weight_names,
    const ShapeInfoMap& input_shape_hints,
    const std::unordered_set<int>& blocklisted_ops) {
  CAFFE_ENFORCE(ws);
  CAFFE_ENFORCE(pred_net, "Predict net cannot be nullptr");

  if (opts_.debug) {
    WriteProtoToTextFile(*pred_net, "debug_pre_ssa_net.pb_txt", false);
  }

  // Get model id and reset Onnxifi op id to 0
  model_id_ = getModelId(*pred_net);
  onnxifi_op_id_ = 0;

  // Unroll If ops
  fetchInputsToIfOpsSubnet(pred_net);

  std::unordered_set<std::string> weights(
      weight_names.begin(), weight_names.end());

  // SSA Rewrite the net if it has not been rewritten
  ShapeInfoMap shape_hints_mapped;
  if (opts_.predictor_net_ssa_rewritten) {
    LOG(INFO) << "predictor net has been ssaRewritten, skip rewritting here";
    annotateOpIndex(pred_net);
    shape_hints_mapped = input_shape_hints;
    for (const auto& w : weights) {
      input_mapping_.emplace(w, w);
    }
  } else {
    shape_hints_mapped = ssaRewriteAndMapNames(ws, pred_net, input_shape_hints);
  }

  // Populate shape info
  // TODO(yingz): We should not need to create mapped_ws since we did not change
  // any input mappings during ssarewrite. However this is here for the
  // following reason: BlackBoxPredictor calls RunNetOnce before onnxifi to
  // populate dimension info. However during this, it was observed, that new
  // blob for output is created. This causes problem if inferShape uses original
  // ws since it does not expect the output blob to be present.
  Workspace mapped_ws(ws, input_mapping_);
  ShapeInfoMap shape_hints_max_bs = inferShapes(
      &mapped_ws, pred_net, shape_hints_mapped, opts_.bound_shape_spec);
  if (opts_.use_onnx) {
    shape_hints_onnx_ = stripShapeInfoMap(shape_hints_max_bs);
  }
  if (opts_.enforce_fp32_inputs_into_fp16) {
    enforceFp32InputsToFp16(weights, pred_net, &shape_hints_max_bs);
  }
  if (opts_.merge_fp32_inputs_into_fp16) {
    mergeFp32InputsAndConvertToFp16(
        opts_.bound_shape_spec.max_batch_size,
        weights,
        pred_net,
        &shape_hints_max_bs);
  }

  if (opts_.debug) {
    caffe2::NetDef ssa_net;
    ssa_net.CopyFrom(*pred_net);
    auto* w_arg = ssa_net.add_arg();
    w_arg->set_name(kInitializers);
    for (const auto& w : weights) {
      w_arg->add_strings(w);
    }
    dumpNet(ssa_net, shape_hints_max_bs, "debug_ssa_net.pb_txt");
  }
  extractPartitionInfo(*pred_net);

  // Get backend id
  getBackendId();

  // Apply some filtering rules
  std::unordered_set<int> new_blocklisted_ops(
      blocklisted_ops.begin(), blocklisted_ops.end());
  applyFilteringRules(
      *pred_net, shape_hints_max_bs, weights, &new_blocklisted_ops);

  // Transform the net
  NetDef net_opt = opts_.use_onnx ? TransformViaOnnx(
                                        ws,
                                        pred_net,
                                        weights,
                                        new_blocklisted_ops,
                                        &shape_hints_max_bs,
                                        opts_.shape_hints_per_bs)
                                  : TransformViaC2(
                                        pred_net,
                                        weights,
                                        new_blocklisted_ops,
                                        shape_hints_max_bs,
                                        opts_.shape_hints_per_bs);

  // Need to figure out a proper place to handle device option
  net_opt.mutable_device_option()->CopyFrom(pred_net->device_option());
  net_opt.set_type(pred_net->type());

  pred_net->Swap(&net_opt);

  addShapeToNet(*pred_net, shape_hints_max_bs);
  if (opts_.debug) {
    WriteProtoToTextFile(*pred_net, "debug_full_opt_net.pb_txt", false);
  }
}

} // namespace caffe2
