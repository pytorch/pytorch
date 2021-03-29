#pragma once
#include <memory>

#include "caffe2/core/tensor.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/metanet.pb.h"
#include "caffe2/proto/predictor_consts.pb.h"

namespace caffe2 {

/*
 * Parameters for a Predictor provided by name.
 * They are stored as shared_ptr to accommodate parameter sharing
 */
using PredictorParameters = std::map<std::string, std::shared_ptr<Blob>>;

/**
 * Stores parameters nessasary for creating a PredictorInterface object.
 */
struct TORCH_API PredictorConfig {
  // A map of parameter name to Tensor object. Predictor is supposed to
  // guarantee constness of all these Tensor objects.
  std::shared_ptr<PredictorParameters> parameters;

  std::shared_ptr<NetDef> predict_net;

  // Input names of a model. User will have to provide all of the inputs
  // for inference
  std::vector<std::string> input_names;
  // Output names of a model. All outputs will be returned as results of
  // inference
  std::vector<std::string> output_names;
  // Parameter names of a model. Should be a subset of parameters map passed in.
  // We provide a separate set of parameter names here as whole parameter set
  // passed in by a user might contain extra tensors used by other models
  std::vector<std::string> parameter_names;

  // TODO We still save ws is because of the current design of workspace and
  // tensor. Once tensor support intrusive_ptr, we'll get rid of this and use
  // parameters to construct Workspace
  std::shared_ptr<Workspace> ws;
};

TORCH_API Workspace makeWorkspace(std::shared_ptr<PredictorParameters> parameters);

TORCH_API PredictorConfig makePredictorConfig(
    const MetaNetDef& net,
    Workspace* parent = nullptr,
    bool run_init = true);

TORCH_API PredictorConfig makePredictorConfig(
    const NetDef& init_net,
    const NetDef& run_net,
    Workspace* parent = nullptr,
    bool run_init = true,
    int optimization = 1);

} // namespace caffe2
