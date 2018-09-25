#include "caffe2/opt/passes.h"

C10_DEFINE_REGISTRY(WorkspaceOptimizationPassRegistry, caffe2::WorkspaceOptimizationPass, nom::repr::NNModule*, caffe2::Workspace*);
C10_DEFINE_REGISTRY(OptimizationPassRegistry, caffe2::OptimizationPass, nom::repr::NNModule*);
