#ifndef CAFFE2_UTILS_PROTO_CONVERT_H_
#define CAFFE2_UTILS_PROTO_CONVERT_H_

#include "caffe2/core/common.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/proto/torch_pb.h"

namespace caffe2 {

CAFFE2_API void ArgumentToAttributeProto(
    const Argument& arg,
    ::torch::AttributeProto* attr);
CAFFE2_API void AttributeProtoToArgument(
    const ::torch::AttributeProto& attr,
    Argument* arg);
CAFFE2_API void OperatorDefToNodeProto(
    const OperatorDef& def,
    ::torch::NodeProto* node);
CAFFE2_API void NodeProtoToOperatorDef(
    const ::torch::NodeProto& node,
    OperatorDef* def);

} // namespace caffe2

#endif // CAFFE2_UTILS_PROTO_CONVERT_H_
