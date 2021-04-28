#include "caffe2/core/logging.h"
#include "caffe2/opt/converter.h"
#include "caffe2/opt/passes.h"

namespace caffe2 {
namespace opt {

using namespace nom;
using namespace nom::repr;

void deadCodeElim(NNModule* nn) {
  // Iteratively remove unconsumed non-external outputs.
  bool changed = false;
  do {
    changed = false;
    for (const auto& node : nn->dataFlow.getMutableNodes()) {
      NOM_REQUIRE_OR_CONT(nn::is<repr::NeuralNetOperator>(node));

      bool isUsed = false;
      for (const auto& output : nn::getOutputs(node)) {
        if (nn::hasConsumer(output) || nn->outputs.count(output)) {
          isUsed = true;
          break;
        }
      }

      NOM_REQUIRE_OR_CONT(!isUsed);

      // No outputs are used, delete them and the node itself.
      for (const auto& output : nn::getOutputs(node)) {
        nn->dataFlow.deleteNode(output);
      }
      nn->dataFlow.deleteNode(node);
      changed = true;
      break;
    }
  } while (changed);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_OPT_PASS_FROM_FUNC(DeadCodeElim, deadCodeElim);

} // namespace opt
} // namespace caffe2
