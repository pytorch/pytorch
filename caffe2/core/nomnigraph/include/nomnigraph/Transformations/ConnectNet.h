#ifndef NOM_TRANSFORMATIONS_CONNECTNET_H
#define NOM_TRANSFORMATIONS_CONNECTNET_H

#include "nomnigraph/Graph/Graph.h"
#include "nomnigraph/Representations/NeuralNet.h"

namespace nom {
namespace transformations {

bool connectNet(repr::NNGraph *);

} // namespace transformations
} // namespace nom

#endif // NOM_TRANSFORMATIONS_CONNECTNET_H
