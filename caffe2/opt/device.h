#include "caffe2/core/common.h"
#include "nomnigraph/Representations/NeuralNet.h"

namespace caffe2 {
namespace opt {

TORCH_API void insertCopies(
    nom::repr::NNModule* nn,
    std::function<bool(nom::repr::NNGraph::NodeRef)> supported,
    std::function<nom::repr::NNGraph::NodeRef(nom::repr::NNGraph&)> copyToFn,
    std::function<nom::repr::NNGraph::NodeRef(nom::repr::NNGraph&)> copyFromFn);

} // namespace opt
} // namespace caffe2
