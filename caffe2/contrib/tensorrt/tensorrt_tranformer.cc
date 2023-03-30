#include "caffe2/contrib/tensorrt/tensorrt_tranformer.h"

#include <iostream>
#include <unordered_set>

#include <NvInfer.h>
#include <onnx2trt.hpp>

#include "onnx/proto_utils.h"

#include "caffe2/contrib/tensorrt/trt_utils.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/onnx/onnx_exporter.h"
#include "caffe2/opt/backend_cutting.h"

namespace caffe2 {

namespace {

// TODO(yinghai): Remove the awkward conversion between unordered_map and map
std::unordered_map<std::string, TensorShape> InferShapes(
    Workspace* ws,
    NetDef* pred_net,
    CaffeMap<std::string, TensorShape>* shape_hints_ordered) {
  // Populate shapes from workplace
  const std::vector<string>& ws_blobs = ws->Blobs();
  for (const auto& s : ws_blobs) {
    shape_hints_ordered->emplace(s, GetTensorShapeOfBlob(ws->GetBlob(s)));
  }

  std::vector<NetDef*> nets;
  nets.emplace_back(pred_net);
  InferBlobShapesAndTypes(*shape_hints_ordered, nets);
  std::unordered_map<std::string, TensorShape> shape_hints;
  for (const auto& kv : *shape_hints_ordered) {
    shape_hints.emplace(kv.first, kv.second);
  }

  return shape_hints;
}

void DumpModel(const ::ONNX_NAMESPACE::ModelProto& model, const std::string& fname) {
  std::ofstream ff(fname);
  ff << ::ONNX_NAMESPACE::ProtoDebugString(model) << std::endl;
  ff.close();
}

void CPUTensorToTensorProto(
    const TensorCPU& cpu_tensor,
    ::ONNX_NAMESPACE::TensorProto* t) {
  const auto len = cpu_tensor.size();
  if (cpu_tensor.template IsType<float>()) {
    t->set_data_type(::ONNX_NAMESPACE::TensorProto::FLOAT);
    const float* data = cpu_tensor.template data<float>();
    for (auto i = 0; i < len; ++i) {
      t->add_float_data(*data++);
    }
  } else if (cpu_tensor.template IsType<int64_t>()) {
    t->set_data_type(::ONNX_NAMESPACE::TensorProto::INT64);
    const int64_t* data = cpu_tensor.template data<int64_t>();
    for (auto i = 0; i < len; ++i) {
      t->add_int64_data(*data++);
    }
  } else if (cpu_tensor.template IsType<int32_t>()) {
    t->set_data_type(::ONNX_NAMESPACE::TensorProto::INT32);
    const int32_t* data = cpu_tensor.template data<int32_t>();
    for (auto i = 0; i < len; ++i) {
      t->add_int32_data(*data++);
    }
  } else {
    CAFFE_THROW(
        "Don't know how to convert workspace tensor type ",
        cpu_tensor.meta().name(),
        " to ONNX TensorProto");
  }
}

void BlobToTensorProto(
    const std::string& name,
    Workspace* ws,
    CUDAContext* context,
    ::ONNX_NAMESPACE::TensorProto* t) {
  // Set name
  t->set_name(name);
  const Blob* blob = ws->GetBlob(name);
  CAFFE_ENFORCE(blob, "Blob ", name, " doesn't exist");

  // Set dims
  const auto shape = GetTensorShapeOfBlob(blob);
  for (const auto i : shape.dims()) {
    t->add_dims(i);
  }

  // Set values
  if (BlobIsTensorType(*blob, CPU)) {
    const auto& cpu_tensor = blob->template Get<TensorCPU>();
    CPUTensorToTensorProto(cpu_tensor, t);
  } else if (BlobIsTensorType(*blob, CUDA)) {
    const auto& cuda_tensor = blob->template Get<TensorCUDA>();
    const auto cpu_tensor = TensorCPU(cuda_tensor, CPU);
    context->FinishDeviceComputation();
    CPUTensorToTensorProto(cpu_tensor, t);
  } else {
    CAFFE_THROW(
        "Initialization blob ",
        name,
        " needs to be either TensorCPU or TensorCUDA");
  }
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
          ::ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT);
      auto* shape = tensor_type->mutable_shape();
      for (int i = 0; i < it->second.dims().size(); ++i) {
        auto* dim = shape->add_dim();
        dim->set_dim_value(it->second.dims(i));
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

void BuildInitializationList(
    Workspace* ws,
    ::ONNX_NAMESPACE::GraphProto* g,
    std::unordered_set<std::string>* initialization_list) {
  const std::vector<string>& ws_blobs = ws->Blobs();

  // Create a CUDA context and reuse it for potential tensor copies across
  // devices
  CUDAContext context;

  for (const auto& s : ws_blobs) {
    auto it = initialization_list->find(s);
    if (it != initialization_list->end()) {
      auto* init_tensor = g->add_initializer();
      BlobToTensorProto(s, ws, &context, init_tensor);
      initialization_list->erase(it);
    }
  }
  CAFFE_ENFORCE(
      initialization_list->empty(), "Unfulfilled initialization list");
  for (const auto& t : g->initializer()) {
    VLOG(2) << "Initializer: " << t.name();
  }
}

void TensorRTTransformer::AddTrtOptions(
    OperatorDef* op,
    const std::unordered_map<std::string, std::vector<int>>&
        output_size_hints) {
  auto* max_batch_size_arg = op->add_arg();
  max_batch_size_arg->set_name("max_batch_size");
  max_batch_size_arg->set_i(max_batch_size_);

  auto* verbosity_arg = op->add_arg();
  verbosity_arg->set_name("log_verbosity");
  verbosity_arg->set_i(verbosity_);

  for (int i = 0; i < op->output_size(); ++i) {
    const auto& o = op->output(i);
    const auto it = output_size_hints.find(o);
    if (it != output_size_hints.end()) {
      const auto& dims = it->second;
      auto* output_size_hint_arg = op->add_arg();
      output_size_hint_arg->set_name(c10::str("output_size_hint_", i));
      for (const auto& d : dims) {
        output_size_hint_arg->add_ints(d);
      }

      LOG(INFO) << "Adding output hint: " << o;
    }
  }
}

OperatorDef TensorRTTransformer::BuildTrtOpLazy(
    const std::string& onnx_model_str,
    const std::unordered_map<std::string, std::vector<int>>& output_size_hints,
    const std::unordered_set<std::string>& initialization_list,
    const caffe2::NetDef& net) {
  OperatorDef op;
  op.set_type("TensorRT");
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

  // Additional arguments for TRT builder
  auto* debug_builder_arg = op.add_arg();
  debug_builder_arg->set_name("debug_builder");
  debug_builder_arg->set_i(debug_builder_);
  auto* max_workspace_size_arg = op.add_arg();
  max_workspace_size_arg->set_name("max_workspace_size");
  max_workspace_size_arg->set_i(max_workspace_size_);
  AddTrtOptions(&op, output_size_hints);
  return op;
}

OperatorDef TensorRTTransformer::BuildTrtOp(
    const std::string& onnx_model_str,
    const std::unordered_map<std::string, std::vector<int>>& output_size_hints) {
  OperatorDef op;
  op.set_type("TensorRT");

  tensorrt::TrtLogger logger;
  auto trt_engine = tensorrt::BuildTrtEngine(
      onnx_model_str,
      &logger,
      max_batch_size_,
      max_workspace_size_,
      debug_builder_);

  // Set up inputs/outputs in the order of they appearnce in getNbBindings
  int num_bindings = trt_engine->getNbBindings();
  for (int b = 0; b < num_bindings; ++b) {
    const auto& name = trt_engine->getBindingName(b);
    if (trt_engine->bindingIsInput(b)) {
      op.add_input(name);
    } else {
      op.add_output(name);
    }
  }

  auto engine_plan = tensorrt::TrtObject(trt_engine->serialize());
  auto* serialized_engine_arg = op.add_arg();
  serialized_engine_arg->set_s("");
  serialized_engine_arg->set_name("backend_buffer");
  auto* s = serialized_engine_arg->mutable_s();
  s->assign((char*)engine_plan->data(), engine_plan->size());

  AddTrtOptions(&op, output_size_hints);

  return op;
}

NetDef TensorRTTransformer::SubnetToTrtOp(
    const caffe2::NetDef& net,
    Workspace* ws,
    onnx::OnnxExporter* exporter,
    std::unordered_map<std::string, TensorShape>* shape_hints) {
  ::ONNX_NAMESPACE::ModelProto onnx_model;
  FillModelInfo(&onnx_model);

  // Convert c2 ops to onnx ops, add const weights if there are any
  for (const auto& op : net.op()) {
    const auto results = exporter->Caffe2OpToOnnxNodes(op, *shape_hints);
    const auto& node_protos = results.first;
    for (const auto& n : node_protos) {
      onnx_model.mutable_graph()->add_node()->CopyFrom(n);
    }
    for (const auto& t : results.second) {
      VLOG(2) << "Adding extra init tensor: " << t.name();
      TensorShape shape;
      shape.mutable_dims()->CopyFrom(t.dims());
      shape_hints->emplace(t.name(), std::move(shape));
      ::ONNX_NAMESPACE::TensorProto tf;
      tf.set_name(t.name());
      tf.mutable_dims()->CopyFrom(t.dims());

      if (t.data_type() == ::ONNX_NAMESPACE::TensorProto::FLOAT) {
        tf.set_data_type(::ONNX_NAMESPACE::TensorProto::FLOAT);
        std::vector<int64_t> v;
        v.resize(t.raw_data().size() / sizeof(int64_t));
        memcpy(v.data(), t.raw_data().data(), t.raw_data().size());
        std::vector<float> vf;
        for (auto i : v) {
          vf.push_back(static_cast<float>(i));
        }
        tf.mutable_raw_data()->assign(
            reinterpret_cast<const char *>(vf.data()), sizeof(float) * vf.size());
      } else if (t.data_type() == ::ONNX_NAMESPACE::TensorProto::INT64) {
        tf.set_data_type(::ONNX_NAMESPACE::TensorProto::INT64);
        tf.mutable_raw_data()->assign(t.raw_data().data(), t.raw_data().size());
      } else {
        CAFFE_THROW("Unsupported tensor data type for conversion: ",
            t.data_type());
      }
      onnx_model.mutable_graph()->add_initializer()->CopyFrom(tf);
    }
  }

  // Convert outputs and compute output shape hints
  std::vector<std::string> io_names;
  for (const auto& output : net.external_output()) {
    io_names.emplace_back(output);
  }
  auto io_vec = ConvertToValueInfo(io_names, *shape_hints);
  std::unordered_map<std::string, std::vector<int>> output_shape_hints;
  for (const auto& i : io_vec) {
    onnx_model.mutable_graph()->add_output()->CopyFrom(i);
    auto ret = output_shape_hints.emplace(i.name(), std::vector<int>());
    auto& vec = ret.first->second;
    const auto it = shape_hints->find(i.name());
    CAFFE_ENFORCE(
        it != shape_hints->end(),
        "Cannot find shape info for output ",
        i.name());
    const auto& shape = it->second;
    for (int k = 0; k < shape.dims().size(); ++k) {
      vec.push_back(shape.dims(k));
    }
  }

  // Convert inputs and figure out weights
  std::unordered_set<std::string> weights;
  const std::vector<string>& ws_blobs = ws->Blobs();
  for (const auto& s : ws_blobs) {
    VLOG(2) << "Add weights: " << s;
    weights.emplace(s);
  }

  std::unordered_set<std::string> total_inputs;
  std::unordered_set<std::string> initialization_list;
  std::vector<std::string> total_inputs_vec;

  // Extra intermediate weights created during conversion
  for (const auto& extra_weight : onnx_model.graph().initializer()) {
    if (total_inputs.emplace(extra_weight.name()).second) {
      total_inputs_vec.emplace_back(extra_weight.name());
    }
  }
  // Boundary inputs, should not be weights
  std::unordered_set<std::string> boundary_inputs;
  for (const auto& i : net.external_input()) {
    boundary_inputs.emplace(i);
  }

  for (const auto& op : net.op()) {
    for (const auto& input : op.input()) {
      if (total_inputs.emplace(input).second && weights.count(input)) {
        // We add weights as inputs too
        total_inputs_vec.emplace_back(input);
        initialization_list.emplace(input);
        VLOG(2) << "Add input weights: " << input;
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

  // Debug stuff
  if (debug_builder_) {
    DumpModel(onnx_model, "debug.onnxtxt");
  }

  // Convert weights to initializing tensors if we are building serializable trt
  // op or defer it to construction time of trt op
  if (build_serializable_op_) {
    BuildInitializationList(
        ws, onnx_model.mutable_graph(), &initialization_list);
  }

  // Onnx model is ready. Call onnx-trt to convert to one trt c2 op
  std::string model_str;
  onnx_model.SerializeToString(&model_str);
  NetDef net_opt;
  auto* op = net_opt.add_op();
  if (build_serializable_op_) {
    *op = BuildTrtOp(model_str, output_shape_hints);
  } else {
    *op =
        BuildTrtOpLazy(model_str, output_shape_hints, initialization_list, net);
  }
  for (const auto& i : op->input()) {
    net_opt.add_external_input(i);
  }
  for (const auto& i : op->output()) {
    net_opt.add_external_output(i);
  }

  return net_opt;
}

CaffeMap<std::string, TensorShape> TensorRTTransformer::SsaRewriteAndMapNames(
    Workspace* ws,
    NetDef* pred_net,
    const std::unordered_map<std::string, TensorShape>& input_shape_hints) {
  input_mapping_ = onnx::SsaRewrite(nullptr, pred_net);
  std::unordered_map<std::string, std::string> input_reverse_mapping;
  std::vector<std::string> external_inputs;
  for (const auto kv : input_mapping_) {
    input_reverse_mapping.emplace(kv.second, kv.first);
    if (!ws->HasBlob(kv.second)) {
      external_inputs.emplace_back(kv.first);
    }
  }
  for (const auto& i : external_inputs) {
    input_mapping_.erase(i);
  }
  CaffeMap<std::string, TensorShape> shape_hints_ordered;
  for (const auto& kv : input_shape_hints) {
    const auto it = input_reverse_mapping.find(kv.first);
    if (it != input_reverse_mapping.end()) {
      LOG(INFO) << "Adding input hint: " << it->second;
      shape_hints_ordered.emplace(it->second, kv.second);
    } else {
      shape_hints_ordered.emplace(kv.first, kv.second);
    }
  }
  return shape_hints_ordered;
}

void TensorRTTransformer::PruneUnusedWeights(
    Workspace* ws,
    const NetDef& pred_net) {
  std::unordered_set<std::string> used_weights;
  for (const auto& op : pred_net.op()) {
    for (const auto& i : op.input()) {
      used_weights.emplace(i);
    }
  }

  for (const auto kv : input_mapping_) {
    // for weights that are not referenced anywhere, we remove it from the
    // original workspace
    if (!used_weights.count(kv.first)) {
      VLOG(2) << "Removing unused weight blob: " << kv.second << " ("
              << kv.first << ")";
      ws->RemoveBlob(kv.second);
    }
  }
}

// Cutting off the runnable part and replace with tensor ops. Asssume the nets
// were topologically sorted
void TensorRTTransformer::Transform(
    Workspace* ws,
    NetDef* pred_net,
    const std::unordered_map<std::string, TensorShape>& input_shape_hints) {
  CAFFE_ENFORCE(ws);
  auto shape_hints_ordered =
      SsaRewriteAndMapNames(ws, pred_net, input_shape_hints);
  Workspace mapped_ws(ws, input_mapping_);
  auto shape_hints = InferShapes(&mapped_ws, pred_net, &shape_hints_ordered);

  CAFFE_ENFORCE(pred_net, "Predict net cannot be nullptr");
  onnx::OnnxExporter exporter(nullptr);
  tensorrt::TrtLogger logger;
  auto trt_builder = tensorrt::TrtObject(nvinfer1::createInferBuilder(logger));
  auto trt_network = tensorrt::TrtObject(trt_builder->createNetwork());
  auto importer =
      tensorrt::TrtObject(nvonnxparser::createParser(*trt_network, logger));

  // function to tell whether TensorRT supports a given C2 op or not
  auto supports =
      [&exporter, &shape_hints, importer](const caffe2::OperatorDef& op) {
        const OpSchema* schema = OpSchemaRegistry::Schema(op.type());
        if (!schema || schema->onnx_schema().empty()) {
          LOG(INFO) << "Cannot export c2 op " << op.type() << " to onnx";
          return false;
        }

        auto results = exporter.Caffe2OpToOnnxNodes(op, shape_hints);
        for (const auto& n : results.first) {
          if (!importer->supportsOperator(n.op_type().c_str())) {
            LOG(INFO) << "TRT does not support ONNX node " << n.op_type();
            return false;
          }
        }
        return true;
      };

  // function to convert runnable subgraph into a trt op. Note that to keep the
  // interface clean, we do the double conversion from C2 op to Onnx ops here
  // but it should be OK as the cost is really small. We also need to keep the
  // same exporter throughout the process to avoid duplicated dummy name
  // generation
  onnx::OnnxExporter exporter2(nullptr);
  auto trt_converter = [this, &mapped_ws, &shape_hints, &exporter2](
                           const caffe2::NetDef& net) mutable {
    return SubnetToTrtOp(net, &mapped_ws, &exporter2, &shape_hints);
  };

  auto cutResult = opt::OptimizeForBackend(*pred_net, supports, trt_converter);
  NetDef net_opt = std::move(cutResult.net);

  // Need to figure out a proper place to handle device option
  net_opt.mutable_device_option()->CopyFrom(pred_net->device_option());
  pred_net->Swap(&net_opt);

  if (build_serializable_op_) {
    PruneUnusedWeights(ws, *pred_net);
  }
}

} // namespace caffe2
