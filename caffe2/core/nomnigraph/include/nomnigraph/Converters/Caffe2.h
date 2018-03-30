#ifndef NOM_CONVERTERS_CAFFE2_H
#define NOM_CONVERTERS_CAFFE2_H

#include "nomnigraph/Graph/Graph.h"
#include "nomnigraph/Representations/ControlFlow.h"
#include "nomnigraph/Representations/NeuralNet.h"
#include "caffe2/core/common.h"
#include "caffe2/proto/caffe2.pb.h"

#include <unordered_map>

namespace nom {
namespace converters {

repr::NNModule convertFromCaffe2Proto(const caffe2::NetDef &net, std::unordered_map<std::string, repr::NNGraph::NodeRef>* blobMapOut = nullptr);
caffe2::NetDef convertToCaffe2Proto(repr::NNModule&);

} // namespace converters
} // namespace nom

#endif // NOM_CONVERTERS_CAFFE2_H
