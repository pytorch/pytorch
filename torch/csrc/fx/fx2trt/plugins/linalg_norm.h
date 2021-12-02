#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
// CHECK-MESSAGES: [[@LINE+1]]:10: error: 'NvInfer.h' file not found [clang-diagnostic-error]
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <sstream>
#include <string>
#include <vector>

namespace fx2trt {

using namespace nvinfer1;

class LinalgNormPlugin : public IPluginV2IOExt {
 private:
  std::optional<int32_t> order_int_  = std::nullopt;
  std::optional<std::string> order_inf_  = std::nullopt;
  std::optional<std::string> order_str_  = std::nullopt;
  std::vector<int32_t> axes_ = {};
  int32_t keep_dims_ = 0;
  at::TensorOptions tensor_options_ = at::TensorOptions();
  std::vector<int64_t> input_sizes_ = {};
  std::vector<int64_t> output_sizes_ = {};
  DataType dtype_ = DataType::kHALF;

 public:
  LinalgNormPlugin(std::optional<int32_t> order_int_, std::optional<std::string> order_inf_, std::optional<std::string> order_str_, std::vector<int32_t> axes_, int32_t keep_dims_, at::TensorOptions tensor_options_, std::vector<int64_t> input_sizes_, std::vector<int64_t> output_sizes_);

  LinalgNormPlugin(const char* data, size_t length);

  void configurePlugin(
      nvinfer1::PluginTensorDesc const* in,
      int32_t nbInput,
      nvinfer1::PluginTensorDesc const* out,
      int32_t nbOutput) noexcept override;

  bool supportsFormatCombination(
      int pos,
      PluginTensorDesc const* inOut,
      int nbInputs,
      int nbOutputs) const noexcept override;

  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
      noexcept override;

  bool isOutputBroadcastAcrossBatch(
      int32_t outputIndex,
      bool const* inputIsBroadcasted,
      int32_t nbInputs) const noexcept override;

  bool canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept override;

  IPluginV2Ext* clone() const noexcept override;

  const char* getPluginType() const noexcept override;

  const char* getPluginVersion() const noexcept override;

  int getNbOutputs() const noexcept override;

  nvinfer1::Dims getOutputDimensions(
      int32_t index,
      Dims const* inputs,
      int32_t nbInputDims) noexcept override;

  int initialize() noexcept override;

  void terminate() noexcept override;

  size_t getWorkspaceSize(int32_t maxBatchSize) const noexcept override;

  int enqueue(
      int batchSize,
      const void* const* inputs,
      void* const* outputs,
      void* workspace,
      cudaStream_t stream) noexcept override;
  std::string serializeToString() const;
  size_t getSerializationSize() const noexcept override;
  void serialize(void* buffer) const noexcept override;
  void destroy() noexcept override;
  void setPluginNamespace(const char* pluginNamespace) noexcept override {}
  const char* getPluginNamespace() const noexcept override {
    return "fx2trt";
  }
};

class LinalgNormPluginCreator : public IPluginCreator {
 public:
  LinalgNormPluginCreator();
  ~LinalgNormPluginCreator() override {}
  const char* getPluginNamespace() const noexcept override;
  void setPluginNamespace(const char* N) noexcept override {}

  const char* getPluginName() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  IPluginV2* deserializePlugin(
      const char* name,
      const void* data,
      size_t length) noexcept override;
  // We can return nullptr here if we are lazy.
  const PluginFieldCollection* getFieldNames() noexcept override;
  IPluginV2* createPlugin(
      const char* name,
      const PluginFieldCollection* fc) noexcept override;
};

class LinalgNormPluginCreator2 : public IPluginCreator {
 public:
  LinalgNormPluginCreator2();
  ~LinalgNormPluginCreator2() override {}
  const char* getPluginNamespace() const noexcept override;
  void setPluginNamespace(const char* N) noexcept override {}

  const char* getPluginName() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  IPluginV2* deserializePlugin(
      const char* name,
      const void* data,
      size_t length) noexcept override;
  // We can return nullptr here if we are lazy.
  const PluginFieldCollection* getFieldNames() noexcept override;
  IPluginV2* createPlugin(
      const char* name,
      const PluginFieldCollection* fc) noexcept override;
};

} // namespace fx2trt
