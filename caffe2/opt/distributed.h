#pragma once

#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/proto/caffe2_pb.h"
#include "nomnigraph/Representations/NeuralNet.h"

namespace caffe2 {

/// \brief Convert to an NNModule and apply a mapping of
/// tensor names to DeviceOptions to it.
///
/// This *only* applies the map to Declare/Export
/// nodes, which are representationally equivalent to
/// external_input/external_output in caffe2 NetDefs.
///
/// Throws an exception if the passed in blobMap contains
/// blobs that are not present in the NNModule.
TORCH_API nom::repr::NNModule convertToNNModule(
    caffe2::NetDef&,
    std::map<std::string, caffe2::DeviceOption>);

/// Helpers for the convertToNNModule for use
/// if you already have an NNModule.
/// You probably don't want to use these
/// if you can use convertToNNModule instead.
TORCH_API void addBlobDeviceOptions(
    std::map<std::string, caffe2::DeviceOption> blobMap,
    nom::repr::NNModule* nn);
TORCH_API void injectDataEdgeIndicators(nom::repr::NNModule* nn);
TORCH_API void removeDataEdgeIndicators(nom::repr::NNModule* nn);

} // namespace caffe2
