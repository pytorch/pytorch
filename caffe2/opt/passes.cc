#include "caffe2/opt/passes.h"

namespace caffe2 {

C10_DEFINE_REGISTRY(
    WorkspaceOptimizationPassRegistry,
    WorkspaceOptimizationPass,
    nom::repr::NNModule*,
    Workspace*);
C10_DEFINE_REGISTRY(OptimizationPassRegistry, OptimizationPass, nom::repr::NNModule*);

} // namespace caffe2
