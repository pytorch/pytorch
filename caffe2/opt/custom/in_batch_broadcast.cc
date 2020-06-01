#include "caffe2/opt/custom/in_batch_broadcast.h"

#include "caffe2/utils/proto_utils.h"

namespace caffe2 {
namespace opt {

const std::string kTILE_SUFFIX = "_tile";

void inBatchBroadcast(
    NetDef* net,
    std::unordered_set<std::string>& to_broadcast_blobs,
    int32_t batch_size,
    ShapeInfoMap& shape_hints) {
  caffe2::NetDef broadcast_net;
  broadcast_net.CopyFrom(*net);
  broadcast_net.clear_op();
  for (auto& blob : to_broadcast_blobs) {
    auto new_blob = blob + kTILE_SUFFIX;
    auto* op = broadcast_net.add_op();
    op->CopyFrom(CreateOperatorDef(
        "Tile",
        "",
        {blob},
        {new_blob},
        {MakeArgument<int>("tiles", batch_size),
         MakeArgument<int>("axis", 0),
         MakeArgument<int>("dynamic", 1)}));
    auto it = shape_hints.find(blob);
    CAFFE_ENFORCE(it != shape_hints.end(), "Cannot find shape info for ", blob);
    auto& shape = it->second;
    CAFFE_ENFORCE(shape.shape.dims_size(), "Dim size for ", blob, " is 0");
    shape_hints.emplace(new_blob, shape);
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
  }

  for (auto op : net->op()) {
    for (int j = 0; j < op.input_size(); j++) {
      if (to_broadcast_blobs.count(op.input(j))) {
        *op.mutable_input(j) = op.input(j) + kTILE_SUFFIX;
      }
    }
    broadcast_net.add_op()->CopyFrom(op);
  }
  net->Swap(&broadcast_net);
}

} // namespace opt
} // namespace caffe2
