#include "caffe2/opt/passes.h"

namespace caffe2 {

CAFFE_DEFINE_REGISTRY(WorkspaceOptimizationPassRegistry, WorkspaceOptimizationPass, NNModule*, Workspace*);
CAFFE_DEFINE_REGISTRY(OptimizationPassRegistry, OptimizationPass, NNModule*);

} // namespace caffe2
