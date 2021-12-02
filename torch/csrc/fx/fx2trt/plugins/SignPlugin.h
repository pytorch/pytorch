#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
// CHECK-MESSAGES: [[@LINE+1]]:10: error: 'NvInfer.h' file not found [clang-diagnostic-error]
#include <NvInfer.h>

namespace {

const char* PLUGIN_NAMESPACE{"fx2trt"};

} // namespace

namespace fx2trt {

using namespace nvinfer1;

class SignPlugin : public IPluginV2IOExt {
 private:
  int32_t size_;
  DataType dtype_ = DataType::kFLOAT;
  at::TensorOptions tensor_options_;

  std::string serializeToString() const;

 public:
  SignPlugin() = default;
  SignPlugin(int32_t size, DataType dtype);

  // IPluginV2IOExt Methods
  void configurePlugin(
      PluginTensorDesc const* in,
      int32_t nbInput,
      PluginTensorDesc const* out,
      int32_t nbOutput) noexcept override;
  bool supportsFormatCombination(
      int32_t pos,
      PluginTensorDesc const* inOut,
      int32_t nbInputs,
      int32_t nbOutputs) const noexcept override;

  // IPluginV2Ext Methods
  DataType getOutputDataType(
      int32_t index,
      DataType const* inputTypes,
      int32_t nbInputs) const noexcept override;
  bool isOutputBroadcastAcrossBatch(
      int32_t outputIndex,
      bool const* inputIsBroadcasted,
      int32_t nbInputs) const noexcept override;
  bool canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept override;
  void attachToContext(cudnnContext*, cublasContext*, IGpuAllocator*) noexcept
      override {}
  void detachFromContext() noexcept override {}
  IPluginV2Ext* clone() const noexcept override;

  // IPluginV2 Methods
  const char* getPluginType() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  int getNbOutputs() const noexcept override;
  Dims getOutputDimensions(
      int32_t index,
      const Dims* inputs,
      int32_t nbInputDims) noexcept override;
  int initialize() noexcept override;
  void terminate() noexcept override {}
  size_t getWorkspaceSize(int32_t maxBatchSize) const noexcept override;
  int32_t enqueue(
      int32_t batchSize,
      void const* const* inputs,
      void* const* outputs,
      void* workspace,
      cudaStream_t stream) noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void* buffer) const noexcept override;
  void destroy() noexcept override {}
  void setPluginNamespace(const char* pluginNamespace) noexcept override {}
  const char* getPluginNamespace() const noexcept override {
    return PLUGIN_NAMESPACE;
  }
};

class SignPluginCreator : public IPluginCreator {
 public:
  SignPluginCreator() = default;
  ~SignPluginCreator() override {};
  const char* getPluginNamespace() const noexcept override {
    return PLUGIN_NAMESPACE;
  }
  void setPluginNamespace(const char* N) noexcept override {}

  const char* getPluginName() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  IPluginV2* deserializePlugin(
      const char* name,
      const void* data,
      size_t length) noexcept override;
  const PluginFieldCollection* getFieldNames() noexcept override;
  IPluginV2* createPlugin(
      const char* name,
      const PluginFieldCollection* fc) noexcept override;
};

class SignPluginCreator2 : public IPluginCreator {
 public:
  SignPluginCreator2() = default;
  ~SignPluginCreator2() override {};
  const char* getPluginNamespace() const noexcept override {
    return PLUGIN_NAMESPACE;
  }
  void setPluginNamespace(const char* N) noexcept override {}

  const char* getPluginName() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  IPluginV2* deserializePlugin(
      const char* name,
      const void* data,
      size_t length) noexcept override;
  const PluginFieldCollection* getFieldNames() noexcept override;
  IPluginV2* createPlugin(
      const char* name,
      const PluginFieldCollection* fc) noexcept override;
};
} // namespace fx2trt
