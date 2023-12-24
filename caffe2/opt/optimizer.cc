#include "caffe2/opt/optimizer.h"

#include "caffe2/opt/converter.h"
#include "caffe2/opt/mobile.h"
#include "caffe2/opt/fusion.h"

namespace caffe2 {
namespace opt {

void workspaceOptimizations(nom::repr::NNModule* nn, Workspace* ws, int level) {
  switch (level) {
    case 1:
      opt::fuseConvBN(nn, ws);
      break;
    case 0:
    default:
      break;
  }
}

void graphOptimzations(nom::repr::NNModule* nn, int level) {
  switch (level) {
    case 1:
#ifdef USE_NNPACK
      opt::addNNPACK(nn, false);
      opt::fuseNNPACKConvRelu(nn);
#endif
    case 0:
    default:
      break;
  }
}

NetDef optimize(NetDef net, Workspace* ws, int level) {
  auto nn = convertToNNModule(net);
  graphOptimzations(&nn, level);
  workspaceOptimizations(&nn, ws, level);
  return convertToCaffe2Proto(nn, net);
}

NetDef optimize(NetDef net, int level) {
  auto nn = convertToNNModule(net);
  graphOptimzations(&nn, level);
  return convertToCaffe2Proto(nn, net);
}

} // namespace opt
} // namespace caffe2
