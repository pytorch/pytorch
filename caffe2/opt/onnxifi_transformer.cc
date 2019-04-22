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
const std::string kRealBatchSizeBlob("real_batch_size");
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

void unrollIfOps(NetDef* net) {
  NetDef clone(*net);
  clone.clear_op();
  for (const auto& op : net->op()) {
    if (op.type() == "If") {
      ArgumentHelper helper(op);
      if (helper.HasSingleArgumentOfType<NetDef>("then_net")) {
        auto then_net = helper.GetSingleArgument<NetDef>("then_net", NetDef());
        for (const auto& nested_op : then_net.op()) {
          clone.add_op()->CopyFrom(nested_op);
        }
      }
      if (helper.HasSingleArgumentOfType<NetDef>("else_net")) {
        auto else_net = helper.GetSingleArgument<NetDef>("else_net", NetDef());
        for (const auto& nested_op : else_net.op()) {
          clone.add_op()->CopyFrom(nested_op);
        }
      }
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

std::unordered_set<string> toHashSet(
    const ::google::protobuf::RepeatedPtrField<string>& strs) {
  return std::unordered_set<string>(strs.begin(), strs.end());
}

int64_t getBlob1stDimSize(
    const ShapeInfo& shape_info,
    const string& blob_name) {
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

} // namespace

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

std::unordered_map<int, std::string>
OnnxifiTransformer::generateBatchPaddingHints(
    const NetDef& onnxifi_net,
    const ShapeInfoMap& shape_hints) {
  std::unordered_map<int, std::string> batch_pos_map;
  const auto external_inputs = toHashSet(onnxifi_net.external_input());
  const auto external_outputs = toHashSet(onnxifi_net.external_output());
  for (const auto& op : onnxifi_net.op()) {
    for (auto i = 0; i < op.input_size(); ++i) {
      const auto& input_blob = op.input(i);
      if (external_inputs.count(input_blob)) {
        auto shape_info_it = shape_hints.find(input_blob);
        if (shape_info_it == shape_hints.end()) {
          LOG(WARNING) << "Cannot find shape_info for external input blob: "
                       << input_blob;
          continue;
        }
        if (shape_info_it->second.dim_type == ShapeInfo::DimType::BATCH ||
            shape_info_it->second.dim_type == ShapeInfo::DimType::SEQ) {
          batch_pos_map.emplace(
              getBlob1stDimSize(shape_info_it->second, input_blob), input_blob);
        }
      }
    }

    // Correctness check on the output
    for (const auto& output_blob : op.output()) {
      if (external_outputs.count(output_blob)) {
        auto shape_info_it = shape_hints.find(output_blob);
        CAFFE_ENFORCE(
            shape_info_it != shape_hints.end(),
            "Cannot find shape info for ",
            output_blob,
            " to adjust output batch size");
        if (shape_info_it->second.dim_type == ShapeInfo::DimType::BATCH) {
          auto max_batch_size =
              getBlob1stDimSize(shape_info_it->second, output_blob);
          CAFFE_ENFORCE(
              batch_pos_map.count(max_batch_size),
              "Cannot find input with max batch size",
              max_batch_size);
        } else if (shape_info_it->second.dim_type == ShapeInfo::DimType::SEQ) {
          LOG(WARNING) << "It's unusual that output tensor " << output_blob
                       << " is of dim_type SEQ. "
                       << "AdjustBatchOp won't attached "
                       << "and it might degrade the performance";
        }
      }
    }
  }
  return batch_pos_map;
}

OperatorDef OnnxifiTransformer::BuildOnnxifiOp(
    const std::string& onnx_model_str,
    const std::unordered_map<std::string, TensorShape>& output_shape_hints,
    const std::unordered_set<std::string>& initialization_list,
    const std::vector<std::string>& external_inputs,
    const std::vector<std::string>& external_outputs,
    const std::unordered_map<int, std::string>& batch_pos_map) {
  OperatorDef op;
  op.set_type("Onnxifi");
  auto* onnx_model_arg = op.add_arg();
  onnx_model_arg->set_name("onnx_model");
  onnx_model_arg->set_s(onnx_model_str);

  // Add the names of the initializer blobs that we want to fetch from the
  // workspace later
  auto* initializers_arg = op.add_arg();
  initializers_arg->set_name("initializers");
  for (const auto& s : initialization_list) {
    initializers_arg->add_strings(s);
  }

  // Add the input/output
  std::unordered_map<std::string, int> input_pos_map;
  int idx = 0;
  auto* input_names = op.add_arg();
  input_names->set_name("input_names");
  for (const auto& input : external_inputs) {
    if (!initialization_list.count(input)) {
      op.add_input(input);
      input_names->add_strings(input);
      input_pos_map.emplace(input, idx++);
    }
  }
  auto* output_names = op.add_arg();
  output_names->set_name("output_names");
  for (const auto& output : external_outputs) {
    op.add_output(output);
    output_names->add_strings(output);
  }

  // Add output size hints
  for (int i = 0; i < op.output_size(); ++i) {
    const auto& o = op.output(i);
    const auto it = output_shape_hints.find(o);
    if (it != output_shape_hints.end()) {
      const auto& shape = it->second;
      auto* output_shape_hint_arg = op.add_arg();
      output_shape_hint_arg->set_name(c10::str("output_shape_hint_", i));
      output_shape_hint_arg->add_ints(onnxifiDataType(shape.data_type()));
      for (const auto& d : shape.dims()) {
        output_shape_hint_arg->add_ints(d);
      }

      VLOG(2) << "Adding output hint: " << o;
    }
  }

  // Tell Onnxifi op that the model is in onnx or c2 proto format
  AddArgument("use_onnx", opts_.use_onnx ? 1 : 0, &op);

  // Tell Onnxifi op which backend id to use
  AddArgument("backend_id", idx_, &op);

  // Add model_id and net_pos to the onnxifi model
  AddArgument(kModelId, model_id_, &op);
  AddArgument(kNetPos, c10::to_string(onnxifi_op_id_++), &op);

  // Add output resizing hints
  auto* resize_arg = op.add_arg();
  resize_arg->set_name("output_resize_hints");
  for (const auto kv : batch_pos_map) {
    const auto it = input_pos_map.find(kv.second);
    CAFFE_ENFORCE(
        it != input_pos_map.end(),
        "Cannot find input in OnnxifiOp: ",
        kv.second);
    resize_arg->add_ints(kv.first);
    resize_arg->add_ints(it->second);
  }

  return op;
}

NetDef OnnxifiTransformer::SubnetToOnnxifiOpViaC2(
    const caffe2::NetDef& net,
    const std::unordered_set<std::string>& weights_in_ws,
    const ShapeInfoMap& shape_hints) {
  int onnxifi_op_id = onnxifi_op_id_;
  if (opts_.debug) {
    WriteProtoToTextFile(
        net, "debug_original_net_" + c10::to_string(onnxifi_op_id) + ".pb_txt");
  }
  if (opts_.min_ops > net.op_size()) {
    return net;
  }
  // We already have all the ops and external inputs and outputs!
  NetDef onnxifi_net(net);

  // Remove the second output of Concat/Reshape from external_output. In
  // addition, we remove those outputs from the Onnxifi op too.
  // TODO: This approach is a bit hacky as we assume that the second output is
  // never used. A more appropriate approach can be learned from the ONNX path,
  // where we statically computes the split_info given input shape and insert a
  // GivenTensorIntFill op
  std::unordered_set<std::string> split_infos;
  for (auto& op : *onnxifi_net.mutable_op()) {
    if ((op.type() == "Concat" || op.type() == "Reshape") &&
        op.output_size() == 2) {
      split_infos.emplace(op.output(1));
    }
  }
  onnxifi_net.clear_external_output();
  for (const auto& o : net.external_output()) {
    if (!split_infos.count(o)) {
      onnxifi_net.add_external_output(o);
    }
  }

  // Add batch padding hints
  auto batch_pos_map = generateBatchPaddingHints(onnxifi_net, shape_hints);

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
  onnxifi_net.clear_external_input();
  for (const auto& i : total_inputs_vec) {
    onnxifi_net.add_external_input(i);
    auto info = shape_hints.at(i);
    if (!info.is_quantized) {
      shape_arg->mutable_tensors()->Add()->CopyFrom(
          wrapShapeInfoIntoTensorProto(i, shape_hints.at(i)));
    } else {
      qshape_arg->mutable_qtensors()->Add()->CopyFrom(
          wrapShapeInfoIntoQTensorProto(i, shape_hints.at(i)));
    }
  }

  // Compute output shape hints
  std::unordered_map<std::string, TensorShape> output_shape_hints;
  for (const auto& o : onnxifi_net.external_output()) {
    const auto it = shape_hints.find(o);
    CAFFE_ENFORCE(
        it != shape_hints.end(), "Cannot find shape info for output ", o);
    const auto& shape = it->second.shape;
    output_shape_hints.emplace(o, shape);
  }

  // Build ONNXIFI Op
  std::vector<std::string> onnxifi_net_inputs(
      onnxifi_net.external_input().begin(), onnxifi_net.external_input().end());
  std::vector<std::string> onnxifi_net_outputs(
      onnxifi_net.external_output().begin(),
      onnxifi_net.external_output().end());
  std::string model_str;
  onnxifi_net.SerializeToString(&model_str);
  auto onnxifi_op = BuildOnnxifiOp(
      model_str,
      output_shape_hints,
      initialization_list,
      onnxifi_net_inputs,
      onnxifi_net_outputs,
      batch_pos_map);
  NetDef net_opt = composeResultNet(onnxifi_op);

  // Debugging stuff
  if (opts_.debug) {
    WriteProtoToTextFile(
        onnxifi_net,
        "debug_onnxifi_net_" + c10::to_string(onnxifi_op_id) + ".pb_txt");
    WriteProtoToTextFile(
        net_opt,
        "debug_optimized_net_" + c10::to_string(onnxifi_op_id) + ".pb_txt");
  }
  return net_opt;
}

NetDef OnnxifiTransformer::SubnetToOnnxifiOpViaOnnx(
    const caffe2::NetDef& net,
    const std::unordered_set<std::string>& weights_in_ws,
    Workspace* ws,
    onnx::OnnxExporter* exporter,
    ShapeInfoMap* shape_hints) {
  if (opts_.min_ops > net.op_size()) {
    return net;
  }
  ::ONNX_NAMESPACE::ModelProto onnx_model;
  fillModelInfo(&onnx_model);

  caffe2::NetDef onnxifi_net(net);
  auto batch_pos_map = generateBatchPaddingHints(onnxifi_net, *shape_hints);

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
      shape_hints->emplace(
          std::piecewise_construct,
          std::forward_as_tuple(ret.first->first),
          std::forward_as_tuple(
              ShapeInfo::DimType::CONSTANT, ret.first->second));

      // Feed into workspace as CPU Tensors
      auto* blob = ws->CreateBlob(t.name());
      auto* cpu_tensor = BlobGetMutableTensor(blob, CPU);
      std::vector<int64_t> dims;
      for(const auto& d : t.dims()) {
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
  std::unordered_map<std::string, TensorShape> output_shape_hints;
  for (const auto& i : io_vec) {
    onnx_model.mutable_graph()->add_output()->CopyFrom(i);
    const auto it = shape_hints_onnx_.find(i.name());
    CAFFE_ENFORCE(
        it != shape_hints_onnx_.end(),
        "Cannot find shape info for output ",
        i.name());
    const auto& shape = it->second;
    output_shape_hints.emplace(i.name(), shape);
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
  auto onnxifi_op = BuildOnnxifiOp(
      model_str,
      output_shape_hints,
      initialization_list,
      onnxifi_net_inputs,
      onnxifi_net_outputs,
      batch_pos_map);
  NetDef net_opt = composeResultNet(onnxifi_op);

  // Debugging stuff
  if (opts_.debug) {
    WriteProtoToTextFile(onnx_model, "debug_onnxifi_net.onnx_txt");
    WriteProtoToTextFile(net_opt, "debug_optimized_net.pb_txt");
  }
  return net_opt;
}

bool OnnxifiTransformer::supportOpOnnx(
    const caffe2::OperatorDef& op,
    onnx::OnnxExporter* exporter,
    const std::unordered_set<int>& blacklisted_ops,
    onnxBackendID backend_id) const {
  try {
    int pos =
        ArgumentHelper::GetSingleArgument<OperatorDef, int>(op, kNetPos, -1);
    if (blacklisted_ops.count(pos)) {
      LOG(INFO) << "Skipping blacklisted op " << op.type() << " at pos " << pos;
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
        // output, we count it as an boudary input
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
    const std::unordered_set<int>& blacklisted_ops,
    onnxBackendID backend_id) const {
  try {
    int pos =
        ArgumentHelper::GetSingleArgument<OperatorDef, int>(op, kNetPos, -1);
    if (blacklisted_ops.count(pos)) {
      LOG(INFO) << "Skipping blacklisted op " << op.type() << " at pos " << pos;
      return false;
    }

    // Build a c2 net with one op
    NetDef net;
    net.add_op()->CopyFrom(op);
    for (const auto& i : op.input()) {
      net.add_external_input(i);
    }
    for (const auto& o : op.output()) {
      net.add_external_output(o);
    }
    // Remove the second output of Concat/Reshape from the external_output
    if ((op.type() == "Concat" || op.type() == "Reshape") &&
        op.output_size() == 2) {
      net.mutable_external_output()->RemoveLast();
    }

    // Encode the input/output shapes to an argument
    auto* shape_arg = net.add_arg();
    auto* qshape_arg = net.add_arg();
    shape_arg->set_name("input_shape_info");
    qshape_arg->set_name("input_qshape_info");
    for (const auto& i : op.input()) {
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

    std::string c2_model_str;
    net.SerializeToString(&c2_model_str);
    auto ret = lib_->onnxGetBackendCompatibility(
        backend_id, c2_model_str.size(), c2_model_str.c_str());
    if (ret != ONNXIFI_STATUS_SUCCESS) {
      LOG(INFO) << "Don't support c2 op " << op.type() << " (" << ret << ")";
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
    std::unordered_set<int>* blacklisted_ops) const {
  std::unordered_map<std::string, int> output_pos;
  onnx::OnnxExporter exporter(nullptr);
  onnxBackendID backend_id = backend_ids_[idx_];

  for (const auto& op : net.op()) {
    if (op.type() == "Gather") {
      int pos =
          ArgumentHelper::GetSingleArgument<OperatorDef, int>(op, kNetPos, -1);
      for (const auto& output : op.output()) {
        output_pos.emplace(output, pos);
      }
    } else if (StartsWith(op.type(), "SparseLengthsWeighted")) {
      auto supported = opts_.use_onnx
          ? supportOpOnnx(op, &exporter, *blacklisted_ops, backend_id)
          : supportOpC2(op, shape_hints, *blacklisted_ops, backend_id);
      if (!supported && op.input_size() > 1) {
        const auto it = output_pos.find(op.input(1));
        if (it == output_pos.end()) {
          continue;
        }
        blacklisted_ops->emplace(it->second);
        // We know that current op is not going to be supported. Might as well
        // blacklist it too
        blacklisted_ops->emplace(
            ArgumentHelper::GetSingleArgument<OperatorDef, int>(
                op, kNetPos, -1));
      }
    }
  }
}

void OnnxifiTransformer::applyFilteringRules(
    const NetDef& net,
    const ShapeInfoMap& shape_hints,
    std::unordered_set<int>* blacklisted_ops) const {
  tieGatherAndSparseLengthsWeightedSumOps(net, shape_hints, blacklisted_ops);
}

void OnnxifiTransformer::getBackendId() {
  idx_ = 0;

  if (opts_.use_onnx) {
    return;
  }
  // Try to find a backend that support Caffe2 proto. Note that this is quite
  // opportunistic as we don't offcially support Caffe2 proto.
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
}

NetDef OnnxifiTransformer::TransformViaC2(
    NetDef* pred_net,
    const std::unordered_set<std::string>& weights,
    const std::unordered_set<int>& blacklisted_ops,
    const ShapeInfoMap& shape_hints) {
  onnxBackendID backend_id = backend_ids_[idx_];

  auto c2_supports = [this, &shape_hints, &blacklisted_ops, backend_id](
                         const caffe2::OperatorDef& op) {
    return supportOpC2(op, shape_hints, blacklisted_ops, backend_id);
  };

  auto c2_converter =
      [this, &weights, &shape_hints](const caffe2::NetDef& net) {
        return SubnetToOnnxifiOpViaC2(net, weights, shape_hints);
      };

  return opt::OptimizeForBackend(
      *pred_net, c2_supports, c2_converter, opts_.debug);
}

NetDef OnnxifiTransformer::TransformViaOnnx(
    Workspace* ws,
    NetDef* pred_net,
    const std::unordered_set<std::string>& weights,
    const std::unordered_set<int>& blacklisted_ops,
    ShapeInfoMap* shape_hints) {
  onnxBackendID backend_id = backend_ids_[idx_];

  // function to tell whether the ONNXIFI backend supports a given C2 op or not
  onnx::OnnxExporter exporter(nullptr);
  auto onnx_supports = [this, &exporter, &blacklisted_ops, backend_id](
                           const caffe2::OperatorDef& op) {
    return supportOpOnnx(op, &exporter, blacklisted_ops, backend_id);
  };

  // function to convert runnable subgraph into an onnxifi op. We need to keep
  // the same exporter throughout the process to avoid duplicated dummy name
  // generation
  onnx::OnnxExporter exporter2(nullptr);
  auto onnx_converter = [this, ws, &weights, shape_hints, &exporter2](
                            const caffe2::NetDef& net) mutable {
    return SubnetToOnnxifiOpViaOnnx(net, weights, ws, &exporter2, shape_hints);
  };

  return opt::OptimizeForBackend(
      *pred_net, onnx_supports, onnx_converter, opts_.debug);
}

// Cutting off the runnable part and replace with ONNXIFI ops. Asssume the nets
// were topologically sorted
void OnnxifiTransformer::transform(
    Workspace* ws,
    NetDef* pred_net,
    const std::vector<std::string>& weight_names,
    const std::unordered_map<std::string, TensorShape>& input_shape_hints,
    const std::unordered_set<int>& blacklisted_ops) {
  CAFFE_ENFORCE(ws);
  CAFFE_ENFORCE(pred_net, "Predict net cannot be nullptr");

  // Get model id and reset Onnxifi op id to 0
  model_id_ = getModelId(*pred_net);
  onnxifi_op_id_ = 0;

  // Unroll If ops
  unrollIfOps(pred_net);

  std::unordered_set<std::string> weights(
      weight_names.begin(), weight_names.end());

  // SSA Rewrite the net
  auto shape_hints_mapped =
      ssaRewriteAndMapNames(ws, pred_net, input_shape_hints);

  // Populate shape info
  // TODO(yingz): We should not need to create mapped_ws since we did not change
  // any input mappings during ssarewrite. However this is here for the
  // following reason: BlackBoxPredictor calls RunNetOnce before onnxifi to
  // populate dimension info. However during this, it was observed, that new
  // blob for output is created. This causes problem if inferShape uses original
  // ws since it does not expect the output blob to be present.
  Workspace mapped_ws(ws, input_mapping_);
  ShapeInfoMap shape_hints = inferShapes(
      &mapped_ws, pred_net, shape_hints_mapped, opts_.bound_shape_spec);
  if (opts_.use_onnx) {
    shape_hints_onnx_ = stripShapeInfoMap(shape_hints);
  }

  if (opts_.debug) {
    dumpNet(*pred_net, shape_hints, "debug_ssa_net.pb_txt");
  }

  // Get backend id
  getBackendId();

  // Apply some filtering rules
  std::unordered_set<int> new_blacklisted_ops(
      blacklisted_ops.begin(), blacklisted_ops.end());
  applyFilteringRules(*pred_net, shape_hints, &new_blacklisted_ops);

  // Transform the net
  NetDef net_opt = opts_.use_onnx
      ? TransformViaOnnx(
            ws, pred_net, weights, new_blacklisted_ops, &shape_hints)
      : TransformViaC2(pred_net, weights, new_blacklisted_ops, shape_hints);

  // Need to figure out a proper place to handle device option
  net_opt.mutable_device_option()->CopyFrom(pred_net->device_option());

  if (opts_.debug) {
    dumpNet(*pred_net, shape_hints, "debug_full_opt_net.pb_txt");
  }
  pred_net->Swap(&net_opt);
}

} // namespace caffe2
