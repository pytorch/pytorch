#include "caffe2/opt/passes.h"

C10_DEFINE_REGISTRY(WorkspaceOptimizationPassRegistry, caffe2::WorkspaceOptimizationPass, caffe2::NNModule*, caffe2::Workspace*);
C10_DEFINE_REGISTRY(OptimizationPassRegistry, caffe2::OptimizationPass, caffe2::NNModule*);
