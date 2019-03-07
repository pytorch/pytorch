#include "caffe2/opt/passes.h"

namespace caffe2 {

C10_DEFINE_REGISTRY(
    WorkspaceOptimizationPassRegistry,
    WorkspaceOptimizationPass,
    NNModule*,
    Workspace*);
C10_DEFINE_REGISTRY(OptimizationPassRegistry, OptimizationPass, NNModule*);

} // namespace caffe2
