#ifndef CAFFE2_OPT_OPT_PASSS_H
#define CAFFE2_OPT_OPT_PASSS_H

#include "caffe2/core/common.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2_pb.h"

#include "nomnigraph/Representations/NeuralNet.h"

namespace caffe2 {

/* This file sets up the optimization pass registry.
 *
 * You'll want to either create a class that inherits from OptimizationPass
 * and implements run or use the REGISTER_OPT_PASS_FROM_FUNC(name, func)
 * to register a function that takes in an NNModule*.
 *
 * If you need access to the workspace in the optimization you'll need to
 * use a different registry and inherit from WorkspaceOptimizationPass.
 */

class TORCH_API OptimizationPass {
 public:
  OptimizationPass(nom::repr::NNModule* nn) : nn_(nn) {}
  virtual void run() = 0;
  virtual ~OptimizationPass() = default;

 protected:
  nom::repr::NNModule* nn_;
};

class TORCH_API WorkspaceOptimizationPass : public OptimizationPass {
 public:
  WorkspaceOptimizationPass(nom::repr::NNModule* nn, Workspace* ws) : OptimizationPass(nn), ws_(ws) {}
  virtual ~WorkspaceOptimizationPass() = default;

 protected:
  Workspace* ws_;
};

C10_DECLARE_REGISTRY(
    WorkspaceOptimizationPassRegistry,
    WorkspaceOptimizationPass,
    nom::repr::NNModule*,
    Workspace*);
#define REGISTER_WS_OPT_PASS(clsname) \
  C10_REGISTER_CLASS(WorkspaceOptimizationPassRegistry, clsname, clsname)
#define REGISTER_WS_OPT_PASS_FROM_FUNC(passname, funcname)      \
  class passname : public WorkspaceOptimizationPass {           \
   public:                                                      \
    using WorkspaceOptimizationPass::WorkspaceOptimizationPass; \
    void run() override {                                       \
      funcname(nn_, ws_);                                       \
    }                                                           \
  };                                                            \
  REGISTER_WS_OPT_PASS(passname);

C10_DECLARE_REGISTRY(OptimizationPassRegistry, OptimizationPass, nom::repr::NNModule*);
#define REGISTER_OPT_PASS(clsname) \
  C10_REGISTER_CLASS(OptimizationPassRegistry, clsname, clsname)
#define REGISTER_OPT_PASS_FROM_FUNC(passname, funcname) \
  class passname : public OptimizationPass {            \
   public:                                              \
    using OptimizationPass::OptimizationPass;           \
    void run() override {                               \
      funcname(nn_);                                    \
    }                                                   \
  };                                                    \
  REGISTER_OPT_PASS(passname);

} // namespace caffe2

#endif // CAFFE2_OPT_OPT_PASSS_H
