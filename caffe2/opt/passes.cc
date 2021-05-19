#include "caffe2/opt/passes.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_REGISTRY(
    WorkspaceOptimizationPassRegistry,
    WorkspaceOptimizationPass,
    NNModule*,
    Workspace*);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_REGISTRY(OptimizationPassRegistry, OptimizationPass, NNModule*);

} // namespace caffe2
