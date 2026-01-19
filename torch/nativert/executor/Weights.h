#pragma once

#include <c10/util/FbcodeMaps.h>
#include <c10/util/Logging.h>
#include <caffe2/serialize/inline_container.h>

#include <torch/nativert/graph/Graph.h>

namespace torch::nativert {

using WeightVersion = int;
/**
 * @brief A class that manages the weights of a graph, providing functionality
 * to load, access, and manipulate them.
 *
 * It is responsible for handling the parameters, buffers, and constants
 * associated with a graph It provides mechanisms to load weights from
 * serialized data, access and modify them, and performs necessary validation
 * checks.
 */
class Weights {
 public:
  Weights(
      const Graph* graph,
      const std::optional<std::unordered_map<std::string, c10::IValue>>&
          stateDict = std::nullopt,
      const std::optional<std::unordered_map<std::string, c10::IValue>>&
          constants = std::nullopt);

  // Arguments
  // - pytorchStreamReader: the reader for the model archive
  // - stateDictPath: a map from parameter/buffer/constant name to file path in
  // the archive
  // - stateDictPathPrefix: a prefix that will be prepended to paths in
  // stateDictPathPrefix
  // - constantPaths: a map from constant name to file path in the archive
  // - constantPathPrefix: a prefix that will be prepended to paths in
  // constantPathPrefix
  explicit Weights(
      const Graph* graph,
      std::shared_ptr<caffe2::serialize::PyTorchStreamReader>
          pytorchStreamReader,
      const std::unordered_map<std::string, std::string>& stateDictPaths,
      std::string_view stateDictPathPrefix,
      const std::unordered_map<std::string, std::string>& constantPaths,
      std::string_view constantPathPrefix,
      std::function<bool(const std::string&)> skipSizeCheck = {},
      std::function<bool(const std::string&)> skipDtypeCheck = {},
      std::shared_ptr<std::unordered_map<
          std::string,
          std::shared_ptr<torch::nativert::TensorMeta>>> maybeNewWeightsMeta =
          nullptr);

  at::Tensor at(const std::string& name) const;
  at::Tensor& at(const std::string& name);
  bool contains(const std::string& name) const;
  c10::IValue getCustomObj(const std::string& name) const;
  c10::IValue getCustomObjByFileName(const std::string& name) const;

  std::unordered_map<std::string, at::Tensor> parameters() const;

  std::unordered_map<std::string, at::Tensor> buffers() const;

  std::unordered_map<std::string, at::Tensor> attributes() const;

  void loadStateDict(
      const std::unordered_map<std::string, c10::IValue>& stateDict);

  /*
   * Replace the value stored at the weight with name "name".
   */
  void setValue(const std::string& name, const at::Tensor& newValue);
  void setValue(
      const std::string& name,
      const at::Tensor& newValue,
      bool skipDeviceCheck);

  /*
   * Update the value stored at the weight with name "name".
   * This is done in-place.
   */
  void updateValue(const std::string& name, const at::Tensor& newValue);

  void updateValues(
      const std::unordered_map<std::string, at::Tensor>& newValues);

  void validateValue(const std::string& name, const at::Tensor& newValue) const;
  void validateValue(
      const std::string& name,
      const at::Tensor& newValue,
      bool skipDeviceCheck) const;

  void validateAllWeightsLoaded();

  void updateFoldedConst(std::string_view name, c10::IValue tensor);

  const std::unordered_map<std::string, c10::IValue>& getFoldedConsts() const;

  C10_ALWAYS_INLINE const c10::FastMap<std::string, c10::IValue>&
  getConstFoldedValues() const {
    return constFoldedValues_;
  }

  C10_ALWAYS_INLINE void setConstFoldedValue(
      const std::string& n,
      c10::IValue iv) {
    constFoldedValues_.insert_or_assign(n, std::move(iv));
  }

  std::string toString() const;

  WeightVersion version() const {
    return version_;
  }

 private:
  const Graph* graph_;
  const std::unordered_map<std::string, TensorMeta>& weightsMeta_;

  // keys are parameter/buffer/constant names, not graph input names!
  std::unordered_map<std::string, at::Tensor> allValues_;

  std::unordered_map<std::string, c10::IValue> customObjs_;

  // contains CustomClassHolder map from a file name to an arbitrary
  // key in customObjs_ that hold the loaded content of the file.
  // This is used in AOTIDelegateExecutor.
  std::unordered_map<std::string, std::string> customObjsPaths_;

  // The liftcycle of folded consts should be tied with the weights from which
  // it was derived. The ordering of the constant should be consistent with
  // the output order of const graph.
  std::vector<c10::IValue> foldedConsts_;
  std::unordered_map<std::string, c10::IValue> foldedConstsMap_;

  c10::FastMap<std::string, c10::IValue> constFoldedValues_;

  // unique version number for this instance of weight
  const WeightVersion version_;

  // every instance of Weight has a unique version number
  static WeightVersion globalVersion_;

  std::function<bool(const std::string&)> skipSizeCheck_;
  std::function<bool(const std::string&)> skipDtypeCheck_;

  // save the names of unused weights
  std::unordered_set<std::string> unusedWeights_;
};

} // namespace torch::nativert
