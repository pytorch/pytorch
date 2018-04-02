#include "caffe2/contrib/tensorrt/tensorrt_tranformer.h"
#include "caffe2/contrib/tensorrt/trt_utils.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/onnx/onnx_exporter.h"
#include <onnx2trt.hpp>
#include <NvInfer.h>

#include <google/protobuf/text_format.h>
#include <iostream>
#include <unordered_set>

namespace caffe2 {

namespace {

// TODO(yinghai): Remove the awkward conversion between unordered_map and map
std::unordered_map<std::string, TensorShape> InferShapes(
    NetDef* init_net,
    NetDef* pred_net,
    const std::unordered_map<std::string, TensorShape>& input_shape_hints) {
  CaffeMap<std::string, TensorShape> shape_hints_ordered;
  for (const auto& kv : input_shape_hints) {
    shape_hints_ordered.emplace(kv.first, kv.second);
  }
  std::vector<std::unique_ptr<NetDef>> nets;
  nets.emplace_back(init_net);
  nets.emplace_back(pred_net);
  InferBlobShapesAndTypes(shape_hints_ordered, nets);
  for (auto& net : nets) {
    net.release();
  }
  std::unordered_map<std::string, TensorShape> shape_hints;
  for (const auto& kv : shape_hints_ordered) {
    shape_hints.emplace(kv.first, kv.second);
  }

  return shape_hints;
}

std::vector<std::string> FigureInputs(
    const NetDef& pred_net,
    int start,
    int end,
    const std::vector<OperatorDef>& new_ops,
    const std::unordered_set<std::string>& weights,
    std::unordered_set<std::string>* initialization_list) {
  // TODO: cache this
  std::unordered_set<std::string> boundary_inputs;
  for (const auto& i : pred_net.external_input()) {
    boundary_inputs.emplace(i);
  }

  //
  for (const auto& op : new_ops) {
    for (const auto& output: op.output()) {
      boundary_inputs.emplace(output);
    }
  }

  std::unordered_set<std::string> total_inputs;
  std::vector<std::string> total_inputs_vec;
  for(int i = start; i < end; ++i) {
    const auto& op = pred_net.op(i);
    for (const auto& input: op.input()) {
      auto rt = total_inputs.emplace(input);
      if (rt.second) {
        if (weights.count(input)) {
          // We add weights as inputs too
          total_inputs_vec.emplace_back(input);
          initialization_list->emplace(input);
        } else if (boundary_inputs.count(input)) {
          LOG(INFO) << "Adding boundary input: " << input;
          total_inputs_vec.emplace_back(input);
        }
      }
    }
  }
  return total_inputs_vec;
}

std::vector<std::string>
FigureOutputs(const NetDef& pred_net, int start, int end) {
  std::unordered_set<std::string> all_outputs;
  std::vector<std::string> all_outputs_vec;
  std::unordered_set<std::string> ext_outputs;
  for (const auto& e: pred_net.external_output()) {
    ext_outputs.emplace(e);
  }
  for (int i = start; i < end; ++i) {
    const auto& op = pred_net.op(i);
    for (const auto& output: op.output()) {
      all_outputs.emplace(output);
      all_outputs_vec.emplace_back(output);
    }
  }
  std::unordered_set<std::string> referred_inputs;
  for (int i = end; i < pred_net.op_size(); ++i) {
    const auto& op = pred_net.op(i);
    for (const auto& input: op.input()) {
      referred_inputs.emplace(input);
    }
  }
  // Remove the output that is
  // 1. Not referred by the subsequent ops
  // 2. Not in the external ouput of the net
  all_outputs_vec.erase(
      std::remove_if(
          all_outputs_vec.begin(),
          all_outputs_vec.end(),
          [&ext_outputs, &referred_inputs](const std::string& output) {
            return (
                !referred_inputs.count(output) && !ext_outputs.count(output));
          }),
      all_outputs_vec.end());
  return all_outputs_vec;
}

std::vector<::ONNX_NAMESPACE::ValueInfoProto> ConvertToValueInfo(
    const std::vector<std::string>& names,
    const std::unordered_map<std::string, TensorShape>& shape_hints) {
  std::vector<::ONNX_NAMESPACE::ValueInfoProto> r;
  for (const auto& s: names) {
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
      for (int i = 0 ; i < it->second.dims().size(); ++i) {
        auto* dim = shape->add_dim();
        dim->set_dim_value(it->second.dims(i));
      }
    }
  }
  return r;
}

void PruneWeights(const NetDef& pred_net, NetDef* init_net) {
  std::unordered_set<std::string> used_weights;
  for (const auto& op: pred_net.op()) {
    for (const auto& i: op.input()) {
      used_weights.emplace(i);
    }
  }

  int last = init_net->op_size();
  for (int i = 0; i < last;) {
    if (!used_weights.count(init_net->op(i).output(0))) {
      if (i != last - 1) {
        init_net->mutable_op()->SwapElements(i, last - 1);
      } else {
        ++i;
      }
      --last;
    } else {
      ++i;
    }
  }

  if (last < init_net->op_size()) {
    init_net->mutable_op()->DeleteSubrange(last, init_net->op_size() - last);
  }

  LOG(INFO) << "New init_net op size: " << init_net->op_size();
}

void FillModelInfo(::ONNX_NAMESPACE::ModelProto* model) {
  model->set_ir_version(::ONNX_NAMESPACE::Version::IR_VERSION);
  model->set_producer_name("caffe2");
  auto* opset_id = model->add_opset_import();
  opset_id->set_domain("");
  opset_id->set_version(3);
}
} // namespace

OperatorDef TensorRTTransformer::BuildTrtOp(
    const std::string& onnx_model_str,
    const std::unordered_map<std::string, std::vector<int>>&
        output_size_hints) {
  OperatorDef op;
  op.set_type("TensorRT");

  TrtLogger logger;
  auto trt_builder = InferObject(nvinfer1::createInferBuilder(logger));
  auto trt_network = InferObject(trt_builder->createNetwork());
  auto importer = InferObject(onnx2trt::createImporter(trt_network.get()));
  auto status = importer->import(onnx_model_str.data(), onnx_model_str.size(), false);
  if (status.is_error()) {
    CAFFE_THROW(MakeString(
        "TensorRTTransformer ERROR: ",
        status.file(),
        ":",
        status.line(),
        " In function ",
        status.func(),
        ":\n",
        "[",
        status.code(),
        "] ",
        status.desc()));
  }
  trt_builder->setMaxBatchSize(max_batch_size_);
  trt_builder->setMaxWorkspaceSize(max_workspace_size_);
  trt_builder->setDebugSync(debug_builder_);
  auto trt_engine =
      InferObject(trt_builder->buildCudaEngine(*trt_network.get()));

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

  auto engine_plan = InferObject(trt_engine->serialize());

  auto* serialized_engine_arg = op.add_arg();
  serialized_engine_arg->set_s("");
  serialized_engine_arg->set_name("serialized_engine");
  auto* s = serialized_engine_arg->mutable_s();
  s->assign((char*)engine_plan->data(), engine_plan->size());

  auto* max_batch_size_arg = op.add_arg();
  max_batch_size_arg->set_name("max_batch_size");
  max_batch_size_arg->set_i(max_batch_size_);

  auto* verbosity_arg = op.add_arg();
  verbosity_arg->set_name("log_verbosity");
  verbosity_arg->set_i(verbosity_);

  auto* output_size_hints_arg = op.add_arg();
  auto* output_size_names_arg = op.add_arg();
  output_size_hints_arg->set_name("output_size_hints");
  output_size_names_arg->set_name("output_size_names");
  for (int i = 0; i < op.output_size(); ++i) {
    const auto& o = op.output(i);
    const auto it = output_size_hints.find(o);
    if (it != output_size_hints.end()) {
      const auto& dims = it->second;
      auto* output_size_hint_arg = op.add_arg();
      output_size_hint_arg->set_name(MakeString("output_size_hint_", i));
      for (const auto& d : dims) {
        output_size_hint_arg->add_ints(d);
      }

      LOG(INFO) << "Adding output hint: " << o;
    }
  }

  return op;
}

void TensorRTTransformer::ClusterToTrtOp(
    const NetDef& init_net,
    const NetDef& pred_net,
    int start,
    int end,
    const std::unordered_set<std::string>& weights,
    const std::unordered_map<std::string, TensorShape>& shape_hints,
    ::ONNX_NAMESPACE::ModelProto* model,
    std::vector<OperatorDef>* new_ops) {
  if (model->graph().node_size() == 0) {
    return;
  }
  model->mutable_graph()->clear_input();
  model->mutable_graph()->clear_output();
  model->mutable_graph()->clear_initializer();

  // Figure out the boundary outputs
  auto outputs = ConvertToValueInfo(FigureOutputs(pred_net, start, end), shape_hints);
  std::unordered_map<std::string, std::vector<int>> output_shape_hints;
  for (const auto& i: outputs) {
    model->mutable_graph()->add_output()->CopyFrom(i);
    auto ret = output_shape_hints.emplace(i.name(), std::vector<int>());
    auto& vec = ret.first->second;
    const auto it = shape_hints.find(i.name());
    CAFFE_ENFORCE(it != shape_hints.end(), "Cannot find shape info for output ", i.name());
    const auto& shape = it->second;
    for (int k = 0; k < shape.dims().size(); ++k) {
      vec.push_back(shape.dims(k));
    }
  }

  // Figure out the boundary inputs
  std::unordered_set<std::string> initialization_list;
  auto total_inputs_vec = FigureInputs(pred_net, start, end, *new_ops, weights, &initialization_list);
  auto inputs = ConvertToValueInfo(total_inputs_vec, shape_hints);
  for (const auto& i: inputs) {
    LOG(INFO) << "Added input: " << i.name();
    model->mutable_graph()->add_input()->CopyFrom(i);
  }

  // Convert weights to initializing tensors
  onnx::OnnxExporter exporter;
  for (const auto& op: init_net.op()) {
    CAFFE_ENFORCE(op.output_size() == 1);
    auto it = initialization_list.find(op.output(0));
    if (it != initialization_list.end()) {
      auto* init_tensor = model->mutable_graph()->add_initializer();
      exporter.InitOpToTensorProto(op, init_tensor);
      initialization_list.erase(it);
    }
  }
  CAFFE_ENFORCE(initialization_list.empty(), "Unfulfilled initialization list");
  for (const auto& t: model->graph().initializer()) {
    LOG(INFO) << "Initializer: " << t.name();
  }

  // Onnx model is ready. Call onnx-trt to convert to one trt c2 op
  std::string model_str;
  model->SerializeToString(&model_str);
  new_ops->emplace_back(BuildTrtOp(model_str, output_shape_hints));

  model->mutable_graph()->clear_node();
}

// Cutting off the runnable part and replace with tensor ops. Asssume the nets
// were topologically sorted
void TensorRTTransformer::Transform(
    NetDef* init_net,
    NetDef* pred_net,
    const std::unordered_map<std::string, TensorShape>& input_shape_hints) {
  const auto shape_hints = InferShapes(init_net, pred_net, input_shape_hints);

  std::unordered_set<std::string> weights;
  if (init_net) {
    for (const auto& op : init_net->op()) {
      CAFFE_ENFORCE(op.type().find("GivenTensor") == 0);
      CAFFE_ENFORCE(op.type().rfind("Fill") == op.type().size() - 4);
      CAFFE_ENFORCE(op.output_size() == 1);
      for (const auto& op_output : op.output()) {
        weights.emplace(op_output);
      }
    }
  }

  CAFFE_ENFORCE(pred_net, "pred_net cannot be nullptr");
  ::ONNX_NAMESPACE::ModelProto onnx_model;
  FillModelInfo(&onnx_model);

  std::vector<OperatorDef> new_ops;
  bool trt_group = false;
  auto importer = InferObject(onnx2trt::createImporter(nullptr));
  int op_idx = 0;
  int start = 0;
  int end = 0;
  for (const OperatorDef& op : pred_net->op()) {
    bool support_trt = true;
    const OpSchema* schema = OpSchemaRegistry::Schema(op.type());
    caffe2::onnx::ConvertedResult results;
    if (!schema or schema->onnx_schema().empty()) {
      LOG(INFO) << "Cannot export c2 op " << op.type() << " to onnx";
      support_trt = false;
    } else {
      // One c2 op can be converted into multiple onnx nodes. For simplicity, we
      // enforce all or nothing here
      results = onnx::OnnxExporter().Caffe2OpToOnnxNodes(op, shape_hints);
      for (const auto& n : results.first) {
        if (!importer->supports(n)) {
          LOG(INFO) << "TRT does not support ONNX node " << n.op_type();
          support_trt = false;
          break;
        }
      }
    }

    if (support_trt) {
      const auto& node_protos = results.first;
      if (!trt_group) {
        trt_group = true;
        start = op_idx;
      }
      for (const auto& n : node_protos) {
        auto* node = onnx_model.mutable_graph()->add_node();
        node->CopyFrom(n);
      }
    } else {
      end = op_idx;
      ClusterToTrtOp(
          *init_net,
          *pred_net,
          start,
          end,
          weights,
          shape_hints,
          &onnx_model,
          &new_ops);
      trt_group = false;
      new_ops.emplace_back(op);
    }
    ++op_idx;
  }
  if (trt_group) {
    end = op_idx;
    ClusterToTrtOp(
        *init_net,
        *pred_net,
        start,
        end,
        weights,
        shape_hints,
        &onnx_model,
        &new_ops);
    trt_group = false;
  }

  pred_net->clear_op();
  for (const auto& op : new_ops) {
    pred_net->add_op()->CopyFrom(op);
  }
  PruneWeights(*pred_net, init_net);
}

}
