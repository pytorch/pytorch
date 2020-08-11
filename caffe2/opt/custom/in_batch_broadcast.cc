#include "caffe2/opt/custom/in_batch_broadcast.h"

#include "caffe2/utils/proto_utils.h"

namespace caffe2 {
namespace opt {

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

  // We are done with the Fused8BitRowwiseQuantizedToFloat swapping heuristics.
  // Now we focus on creating Tile ops based on `to_broadcast_copy`. Here is a
  // convenient function to change the blob shape from batch size to 1, while
  // returning the old batch size.
  auto unbatchShape = [&shape_hints, batch_size](
                          const std::string& blob) mutable {
    auto it = shape_hints.find(blob);
    CAFFE_ENFORCE(it != shape_hints.end(), "Cannot find shape info for ", blob);
    auto& shape = it->second;
    CAFFE_ENFORCE(shape.shape.dims_size(), "Dim size for ", blob, " is 0");
    auto old_shape = shape;
    CAFFE_ENFORCE_EQ(
        shape.shape.dims(0) % batch_size,
        0,
        "Dims(0) for ",
        blob,
        ": ",
        shape.shape.dims(0),
        " cannot be divided by batch_size ");
    shape.shape.set_dims(0, shape.shape.dims(0) / batch_size);
    shape.setDimType(0, TensorBoundShape_DimType_CONSTANT);
    return old_shape;
  };

  // Build the tile ops. For inputs that are 2D, we have this Concat/Tile/Split
  // optimization. We hardcode it as 2D as this is our expected input shape to
  // apply optimization. If certain to-broadcast inputs are not 2D, we end up
  // lose some performance by adding one Tile for each such input. We can also
  // improve this by covering the 1D, 3D cases but we can keep it simple for
  // now.
  int total_dim1 = 0;
  std::vector<std::string> concatInputs;
  std::vector<int> split;
  std::vector<std::string> splitOutputs;
  for (const auto& blob : to_broadcast_copy) {
    const auto new_blob = blob + kTILE_SUFFIX;
    auto old_shape = unbatchShape(blob);
    if (old_shape.shape.dims_size() == 2) {
      // For 2D input we prepare info to create one Concat/Tile/Split chain
      const auto& new_shape = shape_hints.at(blob);
      total_dim1 += new_shape.shape.dims(1);
      concatInputs.emplace_back(blob);
      split.emplace_back(new_shape.shape.dims(1));
      splitOutputs.emplace_back(new_blob);
    } else {
      // Otherwise we create one Tile for each input
      broadcast_net.add_op()->CopyFrom(CreateOperatorDef(
          "Tile",
          "",
          {blob},
          {new_blob},
          {MakeArgument<int>("tiles", batch_size),
           MakeArgument<int>("axis", 0),
           // Indicating that we are tiling to max_batch_size
           MakeArgument<int>("dynamic", 1),
           MakeArgument<int>("net_pos", current_pos++)}));
    }
    shape_hints.emplace(new_blob, old_shape);

    // If this blob is the output of a Fused8BitRowwiseQuantizedToFloat op, we
    // need to further unbatch the input of that
    // Fused8BitRowwiseQuantizedToFloat op.
    const auto rit = reversed.find(blob);
    if (rit != reversed.end()) {
      const auto& orignal_input = rit->second;
      unbatchShape(orignal_input);
    }
  }

  // Create one Concat/Tile/Split chain for all the 2D inputs
  if (!concatInputs.empty()) {
    broadcast_net.add_op()->CopyFrom(CreateOperatorDef(
        "Concat",
        "",
        concatInputs,
        {"inbatch_concat", "inbatch_concat_splitinfo"},
        {MakeArgument<int>("axis", 1),
         MakeArgument<int>("net_pos", current_pos++)}));
    auto shape_info = shape_hints.at(concatInputs.front());
    shape_info.shape.set_dims(1, total_dim1);
    shape_hints.emplace("inbatch_concat", shape_info);
    broadcast_net.add_op()->CopyFrom(CreateOperatorDef(
        "Tile",
        "",
        {"inbatch_concat"},
        {"inbatch_concat_tile"},
        {MakeArgument<int>("tiles", batch_size),
         MakeArgument<int>("axis", 0),
         MakeArgument<int>("dynamic", 1),
         MakeArgument<int>("net_pos", current_pos++)}));
    shape_info.shape.set_dims(0, shape_info.shape.dims(0) * batch_size);
    shape_info.setDimType(0, TensorBoundShape_DimType_BATCH);
    shape_hints.emplace("inbatch_concat_tile", shape_info);
    broadcast_net.add_op()->CopyFrom(CreateOperatorDef(
        "Split",
        "",
        {"inbatch_concat_tile"},
        splitOutputs,
        {MakeArgument<int>("axis", 1),
         MakeArgument<vector<int>>("split", split),
         MakeArgument<int>("net_pos", current_pos++)}));
  }

  // Add rest of the ops and reroute them to consume the broadcasted blobs
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
