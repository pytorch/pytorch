#include "caffe2/opt/backend_transformer_base.h"
#include "caffe2/onnx/onnx_exporter.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

namespace {
void annotateOpIndex(NetDef* net) {
  int i = 0;
  for (auto& op : *(net->mutable_op())) {
    AddArgument(kNetPos, i++, &op);
  }
}
} // namespace

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

TensorProto BackendTransformerBase::wrapShapeInfoIntoTensorProto(
    const std::string& name,
    const ShapeInfo& shape_info) const {
  TensorProto t;
  t.set_name(name);
  t.set_data_type(shape_info.shape.data_type());
  for (const auto i : shape_info.shape.dims()) {
    t.add_dims(i);
  }
  return t;
}

QTensorProto BackendTransformerBase::wrapShapeInfoIntoQTensorProto(
    const std::string& name,
    const ShapeInfo& shape_info) const {
  QTensorProto t;
  CAFFE_ENFORCE(
      shape_info.is_quantized == true,
      "Only quantized shapeinfo can be extracted into QTensor!");
  t.set_name(name);
  t.set_data_type(shape_info.shape.data_type());
  t.set_scale(shape_info.q_info.scale);
  t.set_bias(shape_info.q_info.offset);
  // precision and is_signed is not used in onnxifi workflow, but it is required
  // field
  t.set_precision(0);
  t.set_is_signed(0);
  for (const auto i : shape_info.shape.dims()) {
    t.add_dims(i);
  }
  return t;
}

std::unordered_map<std::string, TensorShape>
BackendTransformerBase::ssaRewriteAndMapNames(
    Workspace* ws,
    NetDef* pred_net,
    const std::unordered_map<std::string, TensorShape>& input_shape_hints) {
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

  std::unordered_map<std::string, TensorShape> shape_hints_mapped;
  for (const auto& kv : input_shape_hints) {
    shape_hints_mapped.emplace(kv.first, kv.second);
  }
  return shape_hints_mapped;
}

ShapeInfoMap BackendTransformerBase::inferShapes(
    Workspace* ws,
    NetDef* pred_net,
    const std::unordered_map<std::string, TensorShape>& shape_hints_mapped,
    const BoundShapeSpec& spec) {
  ShapeInfoMap shape_map;
  // We treat hinted shapes as BATCH. If there are shape hints on blobs in the
  // workspace, since they are already inserted as CONSTANT, it will take effect
  // here. For SEQ typed tensors, there are only a few of them and they will be
  // handled by BoundShapeInferencer.
  for (const auto& kv : shape_hints_mapped) {
    shape_map.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(kv.first),
        std::forward_as_tuple(ShapeInfo::DimType::BATCH, kv.second));
  }
  // Populate shapes from workplace
  const std::vector<std::string> ws_blobs = ws->Blobs();
  for (const auto& s : ws_blobs) {
    auto shape_info = getShapeInfoFromBlob(ws->GetBlob(s));
    if (shape_info.dim_type != ShapeInfo::DimType::UNKNOWN) {
      shape_map.emplace(s, shape_info);
    }
  }
  auto eng = BoundShapeInferencerRegistry()->Create("C10", spec);
  eng->InferBoundShapeAndType(*pred_net, shape_map);
  const auto& out_map = eng->shape_info();
  shape_map.clear();
  for (const auto& kv : out_map) {
    shape_map.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(kv.first),
        std::forward_as_tuple(
            kv.second.dim_type,
            kv.second.shape,
            kv.second.is_quantized,
            kv.second.q_info));
  }
  return shape_map;
}

void BackendTransformerBase::dumpNet(
    const NetDef& pred_net,
    const ShapeInfoMap& shape_hints,
    const std::string& fname) const {
  NetDef shape_net(pred_net);
  auto* shape_arg = shape_net.add_arg();
  auto* qshape_arg = shape_net.add_arg();
  shape_arg->set_name("shape_info");
  qshape_arg->set_name("qshape_info");
  for (const auto& kv : shape_hints) {
    if (!kv.second.is_quantized) {
      auto t = wrapShapeInfoIntoTensorProto(kv.first, kv.second);
      t.add_int32_data(static_cast<int32_t>(kv.second.dim_type));
      shape_arg->mutable_tensors()->Add()->CopyFrom(t);
    } else {
      auto t = wrapShapeInfoIntoQTensorProto(kv.first, kv.second);
      t.add_data(static_cast<int32_t>(kv.second.dim_type));
      qshape_arg->mutable_qtensors()->Add()->CopyFrom(t);
    }
  }
  WriteProtoToTextFile(shape_net, fname);
}
} // namespace caffe2
