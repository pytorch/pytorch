#ifndef NOM_TRANSFORMATIONS_OPERATORFUSION_H
#define NOM_TRANSFORMATIONS_OPERATORFUSION_H

#include "nomnigraph/Graph/Graph.h"
#include "nomnigraph/Representations/NeuralNet.h"

namespace nom {
namespace transformations {

bool fuseConvRelu(Graph<std::unique_ptr<repr::Value>, int> *);

} // namespace transformations
} // namespace nom

#endif // NOM_TRANSFORMATIONS_OPERATORFUSION_H
