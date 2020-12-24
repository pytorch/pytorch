#include "caffe2/opt/backend_transformer_base.h"
#include "caffe2/onnx/onnx_exporter.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

// Populate 'net_pos' argument for any ops that don't already have it. 'net_pos'
// we populate here starts after the max 'net_pos' value we encountered.
void BackendTransformerBase::annotateOpIndex(NetDef* net) {
  // find the max net_pos that we have so far.
  int i = -1;
  for (const auto& op : net->op()) {
    ArgumentHelper helper(op);
    int old_index = helper.GetSingleArgument(op, kNetPos, -1);
    i = std::max(i, old_index);
  }

  // populate net_pos for any op that doesn't already have it.
  for (auto& op : *(net->mutable_op())) {
    if (!ArgumentHelper::HasArgument(op, kNetPos)) {
      AddArgument(kNetPos, ++i, &op);
    }
  }
}

std::string BackendTransformerBase::getModelId(const NetDef& net) {
  static std::atomic<size_t> seq_id{0};
  std::string model_id;
  for (const auto& arg : net.arg()) {
    if (arg.name() == kModelId) {
      if (arg.has_s()) {
        model_id = arg.s();
      } else if (arg.has_i()) {
        model_id = c10::to_string(arg.i());
      }
      break;
    }
  }

  if (model_id.empty()) {
    model_id = "unnamed_" + c10::to_string(seq_id++);
  }
  return model_id;
}

TensorProto wrapShapeInfoIntoTensorProto(
    const std::string& name,
    const ShapeInfo& shape_info) {
  TensorProto t;
  t.set_name(name);
  t.set_data_type(shape_info.shape.data_type());
  for (const auto i : shape_info.shape.dims()) {
    t.add_dims(i);
  }
  for (const auto& dimType : shape_info.getDimType()) {
    t.add_int32_data(static_cast<int32_t>(dimType));
  }
  return t;
}

QTensorProto wrapShapeInfoIntoQTensorProto(
    const std::string& name,
    const ShapeInfo& shape_info) {
  QTensorProto t;
  CAFFE_ENFORCE(
      shape_info.is_quantized == true,
      "Only quantized shapeinfo can be extracted into QTensor!");
  t.set_name(name);
  t.set_data_type(shape_info.shape.data_type());
  t.set_axis(shape_info.q_info.axis);
  t.set_is_multiparam(true);
  for (const auto i : shape_info.q_info.scale) {
    t.add_scales(i);
  }
  t.set_scale(1.0);
  for (const auto i : shape_info.q_info.offset) {
    t.add_biases(i);
  }
  t.set_bias(0.0);
  // precision and is_signed is not used in onnxifi workflow, but it is required
  // field
  t.set_precision(0);
  t.set_is_signed(0);
  for (const auto i : shape_info.shape.dims()) {
    t.add_dims(i);
  }
  for (const auto& dimType : shape_info.getDimType()) {
    t.add_data(static_cast<int32_t>(dimType));
  }
  return t;
}

ShapeInfoMap BackendTransformerBase::ssaRewriteAndMapNames(
    Workspace* ws,
    NetDef* pred_net,
    const ShapeInfoMap& input_shape_hints) {
  input_mapping_ = onnx::SsaRewrite(nullptr, pred_net);
  // Annote the ops with net position
  annotateOpIndex(pred_net);

  // Since we are going to create a mapped workspace, we need to make sure that
  // the parent workspace has the mapped blob names. If the blobs don't exist
  // (usually such blobs are input tensor names), we exclude them from mapping.
  std::vector<std::string> exclude_mapping;
  for (const auto kv : input_mapping_) {
    if (!ws->HasBlob(kv.second)) {
      exclude_mapping.emplace_back(kv.first);
    }
  }
  for (const auto& i : exclude_mapping) {
    input_mapping_.erase(i);
  }

  ShapeInfoMap shape_hints_mapped;
  for (const auto& kv : input_shape_hints) {
    shape_hints_mapped.emplace(kv.first, kv.second);
  }
  return shape_hints_mapped;
}

ShapeInfoMap BackendTransformerBase::inferShapes(
    Workspace* ws,
    NetDef* pred_net,
    const ShapeInfoMap& shape_hints_mapped,
    const BoundShapeSpec& spec) {
  ShapeInfoMap shape_map;

  // Populate shapes from workplace
  const std::vector<std::string> ws_blobs = ws->Blobs();
  for (const auto& s : ws_blobs) {
    auto shape_info = getShapeInfoFromBlob(ws->GetBlob(s));
    if (shape_info.dimTypeIsSet()) {
      shape_map.emplace(s, shape_info);
    }
  }
  for (const auto& s : shape_hints_mapped) {
    shape_map.insert(s);
  }
  auto eng = BoundShapeInferencerRegistry()->Create("C10", spec);
  eng->InferBoundShapeAndType(*pred_net, shape_map, ws);
  const auto& out_map = eng->shape_info();
  shape_map.clear();
  for (const auto& kv : out_map) {
    shape_map.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(kv.first),
        std::forward_as_tuple(
            kv.second.getDimType(),
            kv.second.shape,
            kv.second.is_quantized,
            kv.second.q_info));
  }
  return shape_map;
}

void BackendTransformerBase::addShapeToNet(
    NetDef& shape_net,
    const ShapeInfoMap& shape_hints) const {
  auto* shape_arg = shape_net.add_arg();
  auto* qshape_arg = shape_net.add_arg();
  shape_arg->set_name("shape_info");
  qshape_arg->set_name("qshape_info");
  for (const auto& kv : shape_hints) {
    if (!kv.second.is_quantized) {
      auto t = wrapShapeInfoIntoTensorProto(kv.first, kv.second);
      shape_arg->mutable_tensors()->Add()->CopyFrom(t);
    } else {
      auto t = wrapShapeInfoIntoQTensorProto(kv.first, kv.second);
      qshape_arg->mutable_qtensors()->Add()->CopyFrom(t);
    }
  }
}

void BackendTransformerBase::dumpNet(
    const NetDef& pred_net,
    const ShapeInfoMap& shape_hints,
    const std::string& fname) const {
  NetDef shape_net(pred_net);
  addShapeToNet(shape_net, shape_hints);
  WriteProtoToTextFile(shape_net, fname, false);
}
} // namespace caffe2
