#pragma once

#include "caffe2/core/workspace.h"

#include "caffe2/predictor/InferenceGraph.h"
#include "caffe2/predictor/ThreadLocalPtr.h"

CAFFE2_DECLARE_bool(caffe2_predictor_cleanup_activations);

namespace caffe2 {

class PredictorState {
 public:
  explicit PredictorState(Workspace* parameter_workspace)
      : workspace(parameter_workspace) {}

 public:
  Workspace workspace;

  // We store pointers to anything we are going to access during inference time
  // This way we avoid map lookups in the workspace
  vector<Blob*> input_blobs;
  vector<Blob*> output_blobs;
  NetBase* predict_net;
  std::vector<Blob*> internal_blobs;

  // These could be global. But storing them per thread won't give much memory
  // overhead while frees us from holding a global mutex in the end of each
  // iteration
  std::vector<size_t> internal_blob_max_sizes;
  size_t internal_blob_max_sizes_sum{0};

  // This buffer is allocated each time we call Run() based on an information
  // from previous blob sizes
  Tensor buffer{CPU};
};

/**
 * ThreadSafeActivationCleaningPredictor doesn't provide any guarantess
 * regarding its memory to a user. Input tensors become in ownership of the
 * predictor and output tensors are given up to a caller's ownership.
 *
 * This design allows to implement arbitrary optimizations under the hood.
 *
 * Currently ThreadSafeActivationCleaningPredictor performs a memory
 * optimization where after each Run() call we remember sizes of each of the
 * internal blobs. Then they are cleared out. Next time Run() is called it does
 * just one memory allocation and maps all internal tensors to parts of this
 * buffer using maximum of the capacities for each of the tensors. This strategy
 * is useful when you have a lot of of models. When model is not used, its
 * temporary activations doesn't take up memory. But when its used allocation
 * happen faster than if tensors allocated one by one.
 */
class ThreadSafeActivationCleaningPredictor {
 public:
  /**
   * @brief Creates a predictor given already populated with parameters
   * workspace, initialization and prediction nets, list of inputs, outputs and
   * parameter names.
   *
   * Provided by a shared pointer parameter_workspace constness is enforced via
   * init and main nets inspection as framework doesn't have support for
   * constant workspaces when running operators on their children workspaces.
   */
  ThreadSafeActivationCleaningPredictor(
      std::shared_ptr<Workspace> parameter_workspace,
      const NetDef& predict_init_net,
      std::shared_ptr<NetDef> predict_net,
      const std::vector<std::string>& input_names,
      const std::vector<std::string>& output_names,
      const std::vector<std::string>& parameter_names);

  /**
   * @brief Similar to the constructor taking a shared_ptr to the parameter
   * workspace, but the caller has to guarantee its lifetime.
   */
  ThreadSafeActivationCleaningPredictor(
      Workspace* parameter_workspace,
      const NetDef& predict_init_net,
      std::shared_ptr<NetDef> predict_net,
      const std::vector<std::string>& input_names,
      const std::vector<std::string>& output_names,
      const std::vector<std::string>& parameter_names);

  virtual ~ThreadSafeActivationCleaningPredictor() {}

  using TensorList = std::vector<TensorCPU>;
  using TensorMap = std::unordered_map<std::string, TensorCPU>;

  TensorList operator()(const TensorList& args);
  TensorList operator()(const TensorMap& kwargs);
  TensorList operator()(const TensorList& args, const TensorMap& kwargs);

  const TensorList& operator()(const TensorList& args, TensorList* outputs);
  const TensorList& operator()(const TensorMap& kwargs, TensorList* outputs);
  const TensorList& operator()(
      const TensorList& args,
      const TensorMap& kwargs,
      TensorList* outputs);

  const std::vector<std::string>& InputNames();
  const std::vector<std::string>& OutputNames();
  // TODO: this function is for mapping a map to a list for performance
  int InputIndex(const std::string& name) const;
  const InferenceGraph& GetInferenceGraph();

  void CleanUpMemory(PredictorState* state);
  void AllocateMemory(PredictorState* state);
  void PrepareThreadLocalState();
  // Override this method in order to add non generic optimizations
  // Overriding method is responsible for calling the base method
  // in order to get general optimizations

  // We will populate inference network with an argument with this name
  // It will contain the list of all parameters for logging purposes
  static std::string kPredictorParamName;

 protected:
  virtual void OptimizeNetwork();

 protected:
  InferenceGraph graph_;
  std::shared_ptr<Workspace> parameter_workspace_;

 private:
  // Model parameters
  caffe2::ThreadLocalPtr<PredictorState> thread_local_state_;

  std::vector<std::string> internal_blob_names_;
  std::unordered_map<std::string, int> input_idx_;
};

} // namespace caffe2
