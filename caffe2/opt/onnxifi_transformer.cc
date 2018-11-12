#include "caffe2/opt/onnxifi_transformer.h"

#include <iostream>
#include <unordered_set>

#include <google/protobuf/text_format.h>

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/onnx/onnx_exporter.h"
#include "caffe2/opt/backend_cutting.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

namespace {

const std::string kNetPos("net_pos");

// TODO: We probably don't want use protobuf as annotation in the future.
void AnnotateOpIndex(NetDef* net) {
  int i = 0;
  for (auto& op : *(net->mutable_op())) {
    AddArgument(kNetPos, i++, &op);
  }
}

uint64_t OnnxifiDataType(caffe2::TensorProto::DataType t) {
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

std::unordered_map<std::string, TensorShape> InferShapes(
    Workspace* ws,
    NetDef* pred_net,
    CaffeMap<std::string, TensorShape>* shape_hints_ordered,
    bool infer_shapes) {
  std::unordered_map<std::string, TensorShape> shape_hints;
  if (infer_shapes) {
    // Populate shapes from workplace
    const std::vector<std::string> ws_blobs = ws->Blobs();
    for (const auto& s : ws_blobs) {
      auto shape = GetTensorShapeOfBlob(ws->GetBlob(s));
      if (!shape.unknown_shape()) {
        shape_hints_ordered->emplace(s, std::move(shape));
      }
    }

    std::vector<NetDef*> nets;
    nets.emplace_back(pred_net);
    InferBlobShapesAndTypes(*shape_hints_ordered, nets);
    for (const auto& kv : *shape_hints_ordered) {
      shape_hints.emplace(kv.first, kv.second);
    }
  } else {
    Workspace ws_local(ws);
    ws_local.RunNetOnce(*pred_net);
    const std::vector<std::string> ws_blobs = ws_local.Blobs();
    for (const auto& s : ws_blobs) {
      const Blob* b = ws_local.GetBlob(s);
      auto shape = GetTensorShapeOfBlob(b);
      if (!shape.unknown_shape()) {
        shape_hints.emplace(s, std::move(shape));
      }
    }
  }

  return shape_hints;
}

void DumpModel(
    const ::ONNX_NAMESPACE::ModelProto& model,
    const std::string& fname) {
  std::ofstream ff(fname);
  std::string body;
  ::google::protobuf::TextFormat::PrintToString(model.graph(), &body);
  ff << body << std::endl;
  ff.close();
}

std::vector<::ONNX_NAMESPACE::ValueInfoProto> ConvertToValueInfo(
    const std::vector<std::string>& names,
    const std::unordered_map<std::string, TensorShape>& shape_hints) {
  std::vector<::ONNX_NAMESPACE::ValueInfoProto> r;
  for (const auto& s : names) {
    r.emplace_back();
    auto& value_info = r.back();
    value_info.set_name(s);
    const auto it = shape_hints.find(s);
    if (it == shape_hints.end()) {
      LOG(WARNING) << "Cannot get shape of " << s;
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

void FillModelInfo(::ONNX_NAMESPACE::ModelProto* model) {
  model->set_ir_version(::ONNX_NAMESPACE::Version::IR_VERSION);
  model->set_producer_name("caffe2");
  auto* opset_id = model->add_opset_import();
  opset_id->set_domain("");
  opset_id->set_version(7);
}
} // namespace

OnnxifiTransformer::OnnxifiTransformer(bool infer_shapes, bool debug)
    : infer_shapes_(infer_shapes), debug_(debug) {
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

OperatorDef OnnxifiTransformer::BuildOnnxifiOp(
    const std::string& onnx_model_str,
    const std::unordered_map<std::string, TensorShape>& output_shape_hints,
    const std::unordered_set<std::string>& initialization_list,
    const caffe2::NetDef& net) {
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
    initializers_arg->add_strings(input_mapping_.at(s));
  }

  // Add the input/output
  for (const auto& input : net.external_input()) {
    if (!initialization_list.count(input)) {
      op.add_input(input);
    }
  }
  for (const auto& output : net.external_output()) {
    op.add_output(output);
  }

  // Add output size hints
  for (int i = 0; i < op.output_size(); ++i) {
    const auto& o = op.output(i);
    const auto it = output_shape_hints.find(o);
    if (it != output_shape_hints.end()) {
      const auto& shape = it->second;
      auto* output_shape_hint_arg = op.add_arg();
      output_shape_hint_arg->set_name(c10::str("output_shape_hint_", i));
      output_shape_hint_arg->add_ints(OnnxifiDataType(shape.data_type()));
      for (const auto& d : shape.dims()) {
        output_shape_hint_arg->add_ints(d);
      }

      VLOG(2) << "Adding output hint: " << o;
    }
  }
  return op;
}

NetDef OnnxifiTransformer::SubnetToOnnxifiOp(
    const caffe2::NetDef& net,
    const std::unordered_set<std::string>& weights_in_ws,
    Workspace* ws,
    onnx::OnnxExporter* exporter,
    std::unordered_map<std::string, TensorShape>* shape_hints) {
  ::ONNX_NAMESPACE::ModelProto onnx_model;
  FillModelInfo(&onnx_model);

  // Convert c2 ops to onnx ops, add const weights if there are any
  DeviceOption option;
  CPUContext context(option);
  context.SwitchToDevice();
  std::vector<std::string> extra_weights;
  for (const auto& op : net.op()) {
    const auto results = exporter->Caffe2OpToOnnxNodes(op, *shape_hints);
    for (const auto& n : results.first) {
      onnx_model.mutable_graph()->add_node()->CopyFrom(n);
    }
    for (const auto& t : results.second) {
      VLOG(2) << "Adding extra init tensor: " << t.name();
      TensorShape shape;
      shape.mutable_dims()->CopyFrom(t.dims());
      shape_hints->emplace(t.name(), std::move(shape));

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
      CAFFE_ENFORCE(
          input_mapping_.emplace(t.name(), t.name()).second,
          c10::str("Tensor ", t.name(), " already exists in the workspace"));
    }
  }

  // Convert outputs and compute output shape hints
  std::vector<std::string> io_names;
  for (const auto& output : net.external_output()) {
    io_names.emplace_back(output);
  }
  auto io_vec = ConvertToValueInfo(io_names, *shape_hints);
  std::unordered_map<std::string, TensorShape> output_shape_hints;
  for (const auto& i : io_vec) {
    onnx_model.mutable_graph()->add_output()->CopyFrom(i);
    const auto it = shape_hints->find(i.name());
    CAFFE_ENFORCE(
        it != shape_hints->end(),
        "Cannot find shape info for output ",
        i.name());
    const auto& shape = it->second;
    output_shape_hints.emplace(i.name(), shape);
  }

  // Convert inputs and figure out weights
  std::unordered_set<std::string> total_inputs;
  std::unordered_set<std::string> initialization_list;
  std::vector<std::string> total_inputs_vec;

  // Extra intermediate weights created during conversion
  for (const auto& extra_weight : extra_weights) {
    if (total_inputs.emplace(extra_weight).second) {
      total_inputs_vec.emplace_back(extra_weight);
    }
    initialization_list.emplace(extra_weight);
  }
  // Boundary inputs, should not be weights
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
        total_inputs_vec.emplace_back(input);
        initialization_list.emplace(input);
        VLOG(2) << "Add weights: " << input;
      } else if (boundary_inputs.count(input)) {
        VLOG(2) << "Adding boundary input: " << input;
        total_inputs_vec.emplace_back(input);
      }
    }
  }
  io_vec = ConvertToValueInfo(total_inputs_vec, *shape_hints);
  for (const auto& i : io_vec) {
    onnx_model.mutable_graph()->add_input()->CopyFrom(i);
  }

  // Debugging stuff
  if (debug_) {
    DumpModel(onnx_model, "debug.onnxtxt");
  }

  // Onnx model is ready. Build ONNXIFI Op
  std::string model_str;
  onnx_model.SerializeToString(&model_str);
  NetDef net_opt;
  auto* op = net_opt.add_op();
  *op = BuildOnnxifiOp(model_str, output_shape_hints, initialization_list, net);
  for (const auto& i : op->input()) {
    net_opt.add_external_input(i);
  }
  for (const auto& i : op->output()) {
    net_opt.add_external_output(i);
  }

  return net_opt;
}

CaffeMap<std::string, TensorShape> OnnxifiTransformer::SsaRewriteAndMapNames(
    Workspace* ws,
    NetDef* pred_net,
    const std::unordered_map<std::string, TensorShape>& input_shape_hints) {
  input_mapping_ = onnx::SsaRewrite(nullptr, pred_net);
  // Annote the ops with net position
  AnnotateOpIndex(pred_net);
  std::vector<std::string> external_inputs;
  for (const auto kv : input_mapping_) {
    reverse_input_mapping_.emplace(kv.second, kv.first);
    if (!ws->HasBlob(kv.second)) {
      external_inputs.emplace_back(kv.first);
    }
  }
  for (const auto& i : external_inputs) {
    input_mapping_.erase(i);
  }
  CaffeMap<std::string, TensorShape> shape_hints_ordered;
  for (const auto& kv : input_shape_hints) {
    const auto it = reverse_input_mapping_.find(kv.first);
    if (it != reverse_input_mapping_.end()) {
      shape_hints_ordered.emplace(it->second, kv.second);
    } else {
      shape_hints_ordered.emplace(kv.first, kv.second);
    }
  }
  return shape_hints_ordered;
}

// Cutting off the runnable part and replace with ONNXIFI ops. Asssume the nets
// were topologically sorted
void OnnxifiTransformer::Transform(
    Workspace* ws,
    NetDef* pred_net,
    const std::vector<std::string>& external_inputs,
    const std::unordered_map<std::string, TensorShape>& input_shape_hints,
    const std::unordered_set<int>& blacklisted_ops) {
  CAFFE_ENFORCE(ws);
  auto shape_hints_ordered =
      SsaRewriteAndMapNames(ws, pred_net, input_shape_hints);
  Workspace mapped_ws(ws, input_mapping_);
  std::unordered_map<std::string, TensorShape> shape_hints =
      InferShapes(&mapped_ws, pred_net, &shape_hints_ordered, infer_shapes_);

  CAFFE_ENFORCE(pred_net, "Predict net cannot be nullptr");
  onnx::OnnxExporter exporter(nullptr);

  // function to tell whether the ONNXIFI backend supports a given C2 op or not
  // TODO: choose backend id
  onnxifi_library* backend = lib_;
  onnxBackendID backend_id = backend_ids_[0];
  auto supports = [&exporter,
                   &shape_hints,
                   &blacklisted_ops,
                   backend,
                   backend_id](const caffe2::OperatorDef& op) {
    try {
      int pos =
          ArgumentHelper::GetSingleArgument<OperatorDef, int>(op, kNetPos, -1);
      if (blacklisted_ops.count(pos)) {
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
      FillModelInfo(&onnx_model);
      auto results = exporter.Caffe2OpToOnnxNodes(op, shape_hints);
      for (const auto& n : results.first) {
        onnx_model.mutable_graph()->add_node()->CopyFrom(n);
      }

      // Add input shape info
      std::vector<std::string> input_tmp;
      for (const auto& op_input : op.input()) {
        input_tmp.emplace_back(op_input);
      }
      auto io_vec = ConvertToValueInfo(input_tmp, shape_hints);
      for (const auto& i : io_vec) {
        onnx_model.mutable_graph()->add_input()->CopyFrom(i);
      }

      std::string onnx_model_str;
      onnx_model.SerializeToString(&onnx_model_str);
      auto ret = backend->onnxGetBackendCompatibility(
          backend_id, onnx_model_str.size(), onnx_model_str.c_str());
      if (ret != ONNXIFI_STATUS_SUCCESS) {
        LOG(INFO) << "Don't support onnx for " << op.type() << " c2 op (" << ret
                  << ")";
        return false;
      } else {
        return true;
      }
    } catch (const std::exception& ex) {
      LOG(ERROR) << "Gaught exception when converting op " << op.type()
                 << ", what: " << ex.what();
      return false;
    }
  };

  // function to convert runnable subgraph into a trt op. Note that to keep the
  // interface clean, we do the double conversion from C2 op to Onnx ops here
  // but it should be OK as the cost is really small. We also need to keep the
  // same exporter throughout the process to avoid duplicated dummy name
  // generation
  onnx::OnnxExporter exporter2(nullptr);
  std::unordered_set<std::string> weights;
  std::unordered_set<std::string> input_set;
  for (const auto& i : external_inputs) {
    const auto it = reverse_input_mapping_.find(i);
    if (it != reverse_input_mapping_.end()) {
      input_set.emplace(it->second);
    }
  }
  const std::vector<string>& ws_blobs = mapped_ws.Blobs();
  for (const auto& s : ws_blobs) {
    if (!input_set.count(s)) {
      weights.emplace(s);
    }
  }
  auto trt_converter = [this, ws, &weights, &shape_hints, &exporter2](
                           const caffe2::NetDef& net) mutable {
    return SubnetToOnnxifiOp(net, weights, ws, &exporter2, &shape_hints);
  };

  NetDef net_opt = opt::OptimizeForBackend(*pred_net, supports, trt_converter);

  // Need to figure out a proper place to handle device option
  net_opt.mutable_device_option()->CopyFrom(pred_net->device_option());
  pred_net->Swap(&net_opt);
}

} // namespace caffe2
