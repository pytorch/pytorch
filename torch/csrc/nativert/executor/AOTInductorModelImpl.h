#pragma once

#include <dlfcn.h>
#include <torch/csrc/inductor/aoti_runtime/interface.h> // @manual
#include <torch/csrc/inductor/aoti_runtime/model.h> // @manual
#include <torch/csrc/inductor/aoti_torch/proxy_executor.h> // @manual
#include <torch/torch.h> // @manual=//caffe2:torch-cpp
#include <memory>
#include <optional>

#include "c10/util/FbcodeMaps.h"

namespace torch::aot_inductor {

struct ConstantInfo {
  at::ScalarType dtype;
  std::string originalFqn;
  ConstantType type;
};

class AOTInductorModelImpl {
 public:
  explicit AOTInductorModelImpl(
      const std::string& model_path,
      std::optional<std::string> cubin_dir,
      std::vector<std::string> input_names,
      std::vector<std::string> output_names,
      std::optional<at::ScalarType> input_dtype,
      std::optional<at::ScalarType> output_dtype,
      std::optional<std::string> extern_kernel_nodes_path,
      const std::string& device_str,
      int64_t num_runtimes = 2,
      std::optional<std::unordered_map<std::string, c10::IValue>> custom_objs =
          std::nullopt);

  ~AOTInductorModelImpl() {
    if (containerHandle_) {
      deleteFunc_(containerHandle_);
    }
  }

  std::vector<torch::Tensor> forward(std::vector<torch::Tensor>& inputs);

  std::vector<const char*> get_call_spec();

  std::unordered_map<std::string, ConstantInfo> getConstantInfos() const;

  void updateConstantBuffer(
      std::unordered_map<std::string, torch::Tensor*>&& constants,
      bool use_inactive,
      bool validate_full_update);

  void updateInactiveConstantBuffer(
      std::unordered_map<std::string, torch::Tensor*>&& constants);

  void runConstantFolding(bool use_inactive);

  void swapConstantBuffers();

  void profile(
      std::vector<torch::Tensor>& inputs,
      const std::string& filename,
      size_t num_iters);

  // If we need to move or copy this object, then we should just
  // define a unique_ptr with deleter for the handle.
  AOTInductorModelImpl(const AOTInductorModelImpl&) = delete;
  AOTInductorModelImpl& operator=(const AOTInductorModelImpl&) = delete;

  static void registerLibraryNameToPathMap(
      std::unordered_map<std::string, std::string> map);

  static std::string getFullPathForLibraryName(const std::string& name);

  static void setCubinDir(std::optional<std::string> cubin_dir);

  static std::optional<std::string> getCubinDir();

  static void registerExternKernelsSpecNameToPathMap(
      std::unordered_map<std::string, std::string> mapping);

  static std::optional<std::string> getFullPathForExternKernelsSpecName(
      const std::string& name);

  static bool getDeserializePickledModel();

  static void setDeserializePickledModel(bool deserializePickledModel);

  /*
   * Returns a path to .so file (either relative or absolute).
   */
  const std::string& libraryPath() const {
    return libraryPath_;
  }

  const std::string& libraryBasename() const {
    return libraryBasename_;
  }

  const std::vector<std::string>& inputNames() const {
    return inputNames_;
  }

  const std::vector<std::string>& outputNames() const {
    return outputNames_;
  }

  const std::optional<at::ScalarType> floatingPointInputDtype() const {
    return floatingPointInputDtype_;
  }

  const std::optional<at::ScalarType> floatingPointOutputDtype() const {
    return floatingPointOutputDtype_;
  }

  static void add_library_search_path(const std::string& path) {
    library_search_paths_.push_back(path);
  }

  bool is_cpu() const {
    return deviceStr_ == "cpu";
  }

 private:
  // @lint-ignore CLANGTIDY facebook-hte-NonPodStaticDeclaration
  static std::vector<std::string> library_search_paths_;
  // @lint-ignore CLANGTIDY facebook-hte-NonPodStaticDeclaration
  static thread_local std::unordered_map<std::string, std::string>
      lib_name_to_path_;
  // @lint-ignore CLANGTIDY facebook-hte-NonPodStaticDeclaration
  static thread_local std::optional<std::string> cubin_dir_;

  /*
   * Example:
   * {
   *   "aaa.json": "/tmp/abcdef/aaa.json",
   *   "bbb.json": "/tmp/abcdef/bbb.json",
   * }
   */
  // @lint-ignore CLANGTIDY facebook-hte-NonPodStaticDeclaration
  static thread_local std::unordered_map<std::string, std::string>
      extern_kernels_spec_name_to_path_;

  static thread_local bool deserialize_pickled_model_;

  struct DlcloseDeleter {
    void operator()(void* p) const {
      if (p) {
        dlclose(p);
      }
    }
  };

  std::vector<torch::Tensor> processInputs(
      std::vector<torch::Tensor>& python_inputs);

  std::vector<torch::Tensor> processOutputs(
      std::vector<torch::Tensor>&& outputs);

  std::unique_ptr<void, DlcloseDeleter> handle_ = nullptr;
  AOTInductorModelContainerHandle containerHandle_;

  decltype(&AOTInductorModelContainerDelete) deleteFunc_ = nullptr;
  decltype(&AOTInductorModelContainerRun) runFunc_ = nullptr;
  decltype(&AOTInductorModelContainerGetOutputName) getOutputNameFunc_ =
      nullptr;
  decltype(&AOTInductorModelContainerGetCallSpec) getCallSpecFunc_ = nullptr;
  decltype(&AOTInductorModelContainerGetNumConstants) getNumConstantsFunc_{
      nullptr};
  decltype(&AOTInductorModelContainerGetConstantName) getConstantNameFunc_{
      nullptr};
  decltype(&AOTInductorModelContainerGetConstantOriginalFQN)
      getConstantOriginalFQNFunc_{nullptr};
  decltype(&AOTInductorModelContainerGetConstantFromFolded)
      getConstantFromFoldedFunc_{nullptr};
  decltype(&AOTInductorModelContainerGetConstantType) getConstantTypeFunc_{
      nullptr};
  decltype(&AOTInductorModelContainerGetConstantDtype) getConstantDtypeFunc_{
      nullptr};
  decltype(&AOTInductorModelContainerRunConstantFolding)
      runConstantFoldingFunc_{nullptr};
  decltype(&AOTInductorModelContainerUpdateConstantBuffer)
      updateConstantBufferFunc_{nullptr};
  decltype(&AOTInductorModelContainerUpdateInactiveConstantBuffer)
      updateInactiveConstantBufferFunc_{nullptr};
  decltype(&AOTInductorModelContainerSwapConstantBuffer)
      swapConstantBufferFunc_{nullptr};

  const std::string libraryBasename_;
  const std::string libraryPath_;
  const std::vector<std::string> inputNames_;
  const std::vector<std::string> outputNames_;
  const std::optional<at::ScalarType> floatingPointInputDtype_;
  const std::optional<at::ScalarType> floatingPointOutputDtype_;
  c10::FastMap<const char*, size_t> inputNameToIndex_;
  c10::FastMap<const char*, size_t> outputNameToIndex_;

  std::unique_ptr<ProxyExecutor> proxyExecutor_;
  std::string deviceStr_;
};
} // namespace torch::aot_inductor
