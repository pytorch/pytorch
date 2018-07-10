#include "nomnigraph/Representations/NeuralNet.h"

namespace caffe2 {
namespace opt {

void insertCopies(
    nom::repr::NNModule* nn,
    std::function<bool(nom::repr::NNGraph::NodeRef)> supported,
    std::function<nom::repr::NNGraph::NodeRef(nom::repr::NNGraph&)> copyToFn,
    std::function<nom::repr::NNGraph::NodeRef(nom::repr::NNGraph&)> copyFromFn);

} // namespace opt
} // namespace caffe2
