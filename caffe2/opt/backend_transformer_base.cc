#include "caffe2/opt/backend_transformer_base.h"
#include "caffe2/onnx/onnx_exporter.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

namespace {
void AnnotateOpIndex(NetDef* net) {
  int i = 0;
  for (auto& op : *(net->mutable_op())) {
    AddArgument(kNetPos, i++, &op);
  }
}
} // namespace

std::string BackendTransformerBase::getModelId(const NetDef& net) {
  static std::atomic<size_t> seq_id{0};
  auto model_id =
      ArgumentHelper(net).GetSingleArgument<std::string>(kModelId, "");
  if (model_id.empty()) {
    model_id = "unnamed_" + c10::to_string(seq_id++);
  }
  return model_id;
}

TensorProto BackendTransformerBase::wrapShapeInfoIntoTensorProto(
    const std::string& name,
    const ShapeInfo& shape_info) {
  TensorProto t;
  t.set_name(name);
  t.set_data_type(shape_info.shape.data_type());
  for (const auto i : shape_info.shape.dims()) {
    t.add_dims(i);
  }
  return t;
}

std::unordered_map<std::string, TensorShape>
BackendTransformerBase::ssaRewriteAndMapNames(
    Workspace* ws,
    NetDef* pred_net,
    const std::unordered_set<std::string>& weights,
    const std::unordered_map<std::string, TensorShape>& input_shape_hints) {
  // Make sure weights do not contain output of any op.
  for (const auto& op : pred_net->op()) {
    for (const auto& output : op.output()) {
      CAFFE_ENFORCE_EQ(
          weights.count(output),
          0,
          "Weight ",
          output,
          " shouldn't appear in the output");
    }
  }
  input_mapping_ = onnx::SsaRewrite(nullptr, pred_net, weights);
  // Annote the ops with net position
  AnnotateOpIndex(pred_net);

  // Need to add mapping for weights. This will be used to create new workspace
  // with mapped weights.
  for (const auto& w : weights) {
    input_mapping_.emplace(w, w);
  }

  // Since we are going to create a mapped workspace, we need to make sure that
  // the parent workspace has the mapped blob names. If the blobs don't exist
  // (usually such blobs are input tensor names), we exclude them from mapping.
  std::vector<std::string> exclude_mapping;
  for (const auto kv : input_mapping_) {
    reverse_input_mapping_.emplace(kv.second, kv.first);
    if (!ws->HasBlob(kv.second)) {
      exclude_mapping.emplace_back(kv.first);
    }
  }
  for (const auto& i : exclude_mapping) {
    input_mapping_.erase(i);
  }
  std::unordered_map<std::string, TensorShape> shape_hints_mapped;
  for (const auto& kv : input_shape_hints) {
    const auto it = reverse_input_mapping_.find(kv.first);
    if (it != reverse_input_mapping_.end()) {
      shape_hints_mapped.emplace(it->second, kv.second);
    } else {
      shape_hints_mapped.emplace(kv.first, kv.second);
    }
  }
  return shape_hints_mapped;
}

ShapeInfoMap BackendTransformerBase::inferShapes(
    Workspace* ws,
    NetDef* pred_net,
    const std::unordered_map<std::string, TensorShape>& shape_hints_mapped,
    const BoundShapeSpec& spec) {
  ShapeInfoMap shape_map;
  // Populate shapes from workplace
  const std::vector<std::string> ws_blobs = ws->Blobs();
  for (const auto& s : ws_blobs) {
    auto shape_info = getShapeInfoFromBlob(ws->GetBlob(s));
    if (shape_info.dim_type != ShapeInfo::DimType::UNKNOWN) {
      shape_map[s] = shape_info;
    }
  }
  for (const auto& kv : shape_hints_mapped) {
    shape_map.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(kv.first),
        std::forward_as_tuple(ShapeInfo::DimType::CONSTANT, kv.second));
  }
  BoundShapeInferencer eng(spec);
  eng.InferBoundShapeAndType(*pred_net, shape_map);
  const auto& out_map = eng.shape_info();

  for (const auto& kv : out_map) {
    shape_map.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(kv.first),
        std::forward_as_tuple(kv.second.dim_type, kv.second.shape));
  }
  return shape_map;
}
} // namespace caffe2
