#include "caffe2/opt/custom/in_batch_broadcast.h"

#include "caffe2/utils/proto_utils.h"

namespace caffe2 {
namespace opt {

const std::string kFP16_SUFFIX = "_fp16";
const std::string kFP32_SUFFIX = "_fp32";
const std::string kTILE_SUFFIX = "_tile";

void inBatchBroadcast(
    NetDef* net,
    const std::unordered_set<std::string>& to_broadcast_blobs,
    int32_t batch_size,
    ShapeInfoMap& shape_hints) {
  int current_pos = net->op_size();
  caffe2::NetDef broadcast_net;
  broadcast_net.CopyFrom(*net);
  broadcast_net.clear_op();
  std::vector<OperatorDef> pre_ops;
  std::vector<OperatorDef> post_ops;

  // Heuristic: if any of to_broadcast_blobs is connected to
  // Fused8BitRowwiseQuantizedToFloat only, we move Tile after
  // Fused8BitRowwiseQuantizedToFloat to save some compute.
  std::unordered_map<std::string, int> consumers;
  for (const auto& op : net->op()) {
    for (const auto& i : op.input()) {
      if (to_broadcast_blobs.count(i)) {
        consumers[i] += 1;
      }
    }
  }
  std::unordered_map<std::string, std::string> to_broadcast_replace;
  for (const auto& op : net->op()) {
    bool match = false;
    if (op.type() == "Fused8BitRowwiseQuantizedToFloat") {
      CAFFE_ENFORCE_EQ(
          op.input_size(),
          1,
          "Fused8BitRowwiseQuantizedToFloat can only have 1 input");
      CAFFE_ENFORCE_EQ(
          op.output_size(),
          1,
          "Fused8BitRowwiseQuantizedToFloat can only have 1 output");
      const auto it = consumers.find(op.input(0));
      if (it != consumers.end() && it->second == 1) {
        match = true;
      }
    }
    if (match) {
      to_broadcast_replace.emplace(op.input(0), op.output(0));
      pre_ops.emplace_back(op);
    } else {
      post_ops.emplace_back(op);
    }
  }
  // Build a reverse mapping. Not that such mapping is bijective, because if it
  // is not, some key will have multiple consumers, which violates the single
  // consumer condition above.
  std::unordered_map<std::string, std::string> reversed;
  for (const auto& kv : to_broadcast_replace) {
    reversed.emplace(kv.second, kv.first);
  }

  std::unordered_set<std::string> to_broadcast_copy;
  for (const auto& b : to_broadcast_blobs) {
    const auto it = to_broadcast_replace.find(b);
    if (it != to_broadcast_replace.end()) {
      to_broadcast_copy.emplace(it->second);
    } else {
      to_broadcast_copy.emplace(b);
    }
  }
  for (const auto& op : pre_ops) {
    broadcast_net.add_op()->CopyFrom(op);
  }

  auto setShape = [&shape_hints, batch_size](
                      const std::string& blob,
                      const std::string& new_blob) mutable {
    auto it = shape_hints.find(blob);
    CAFFE_ENFORCE(it != shape_hints.end(), "Cannot find shape info for ", blob);
    auto& shape = it->second;
    CAFFE_ENFORCE(shape.shape.dims_size(), "Dim size for ", blob, " is 0");
    if (!new_blob.empty()) {
      shape_hints.emplace(new_blob, shape);
    }
    CAFFE_ENFORCE_EQ(
        shape.shape.dims(0) % batch_size,
        0,
        "Dims(0) for ",
        blob,
        ": ",
        shape.shape.dims(0),
        " cannot be divided by batch_size ",
        batch_size);
    shape.shape.set_dims(0, shape.shape.dims(0) / batch_size);
    shape.setDimType(0, TensorBoundShape_DimType_CONSTANT);
  };

  for (const auto& blob : to_broadcast_copy) {
    auto it = shape_hints.find(blob);
    CAFFE_ENFORCE(it != shape_hints.end(), "Cannot find shape info for ", blob);
    const auto& shape = it->second;
    CAFFE_ENFORCE_GT(shape.shape.dims_size(), 0, "Dim size for ", blob, " is 0");

    // If an op like Fused8BitRowwiseQuantizedToFloat ends up on CPU and
    // Tile ends up on an accelerator and only FP16 is supported, then we want
    // to make sure conversion from FP32 to FP16 is done on CPU to save cycles
    // on accelerator.
    const std::string blob_fp16 = blob + kFP16_SUFFIX;
    const std::string blob_fp32 = blob + kFP32_SUFFIX;
    const bool isFp32Optimization =
        (shape.shape.data_type() == TensorProto_DataType_FLOAT);
    if (isFp32Optimization) {
      auto* op_fp16 = broadcast_net.add_op();
      op_fp16->CopyFrom(CreateOperatorDef(
          "FloatToHalf",
          "",
          {blob},
          {blob_fp16},
          {MakeArgument<int>("net_pos", current_pos++)}));
      auto* op_fp32 = broadcast_net.add_op();
      op_fp32->CopyFrom(CreateOperatorDef(
          "HalfToFloat",
          "",
          {blob_fp16},
          {blob_fp32},
          {MakeArgument<int>("net_pos", current_pos++)}));
    }

    std::string blob_tile = blob + kTILE_SUFFIX;
    auto* op_tile = broadcast_net.add_op();
    op_tile->CopyFrom(CreateOperatorDef(
        "Tile",
        "",
        {isFp32Optimization ? blob_fp32 : blob},
        {blob_tile},
        {MakeArgument<int>("tiles", batch_size),
         MakeArgument<int>("axis", 0),
         // Indicating that we are tiling to max_batch_size
         MakeArgument<int>("dynamic", 1),
         MakeArgument<int>("net_pos", current_pos++)}));

    setShape(blob, blob_tile);
    if (isFp32Optimization) {
      const auto adjusted_shape = shape_hints[blob];

      auto shape_fp16 = adjusted_shape;
      shape_fp16.shape.set_data_type(TensorProto_DataType_FLOAT16);
      shape_hints.emplace(blob_fp16, shape_fp16);

      shape_hints.emplace(blob_fp32, adjusted_shape);
    }

    const auto rit = reversed.find(blob);
    if (rit != reversed.end()) {
      const auto& original_input = rit->second;
      setShape(original_input, "");
    }
  }

  for (auto& op : post_ops) {
    for (int j = 0; j < op.input_size(); j++) {
      if (to_broadcast_copy.count(op.input(j))) {
        *op.mutable_input(j) = op.input(j) + kTILE_SUFFIX;
      }
    }
    broadcast_net.add_op()->CopyFrom(op);
  }
  net->Swap(&broadcast_net);
}

} // namespace opt
} // namespace caffe2
