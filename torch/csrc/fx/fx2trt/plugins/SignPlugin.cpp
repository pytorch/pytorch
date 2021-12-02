// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#include <torch/csrc/fx/fx2trt/plugins/SignPlugin.h>
#include <torch/torch.h>
#include <iostream>
#include <numeric>
#include <ATen/Functions.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/Exception.h>

namespace {

const char* PLUGIN_VERSION{"1"};
const char* PLUGIN_NAME{"SignPlugin"};

} // namespace

namespace fx2trt {

std::string SignPlugin::serializeToString() const {
  torch::serialize::OutputArchive output_archive;
  output_archive.write("size", torch::IValue(size_));
  output_archive.write("dtype", torch::IValue((int32_t)dtype_));
  std::ostringstream data_str;
  output_archive.save_to(data_str);
  return data_str.str();
}

SignPlugin::SignPlugin(int32_t size, DataType dtype)
    : size_(size), dtype_(dtype) {
  tensor_options_ = tensor_options_.device(c10::kCUDA);
  if (dtype_ == DataType::kFLOAT) {
    tensor_options_ = tensor_options_.dtype(c10::kFloat);
  } else {
    tensor_options_ = tensor_options_.dtype(c10::kHalf);
  }
}

void SignPlugin::configurePlugin(
    PluginTensorDesc const* in,
    int32_t nbInput,
    PluginTensorDesc const* out,
    int32_t nbOutput) noexcept {
  TORCH_CHECK(in[0].type == DataType::kFLOAT || in[0].type == DataType::kHALF);
  TORCH_CHECK(in[0].format == PluginFormat::kLINEAR);

  tensor_options_ = tensor_options_.device(c10::kCUDA);
  if (in[0].type == DataType::kFLOAT) {
    dtype_ = DataType::kFLOAT;
    tensor_options_ = tensor_options_.dtype(c10::kFloat);
  } else {
    dtype_ = DataType::kHALF;
    tensor_options_ = tensor_options_.dtype(c10::kHalf);
  }

  size_ = std::accumulate(
      in[0].dims.d,
      in[0].dims.d + in[0].dims.nbDims,
      1,
      std::multiplies<int32_t>());
}

bool SignPlugin::supportsFormatCombination(
    int32_t pos,
    PluginTensorDesc const* inOut,
    int32_t nbInputs,
    int32_t nbOutputs) const noexcept {
  return inOut[pos].format == PluginFormat::kLINEAR &&
      inOut[pos].type == inOut[0].type &&
      (inOut[0].type == DataType::kFLOAT || inOut[0].type == DataType::kHALF);
}

DataType SignPlugin::getOutputDataType(
    int32_t index,
    DataType const* inputTypes,
    int32_t nbInputs) const noexcept {
  return inputTypes[0];
}

bool SignPlugin::isOutputBroadcastAcrossBatch(
    int32_t outputIndex,
    bool const* inputIsBroadcasted,
    int32_t nbInputs) const noexcept {
  return false;
}

bool SignPlugin::canBroadcastInputAcrossBatch(
    int32_t inputIndex) const noexcept {
  return false;
}

IPluginV2Ext* SignPlugin::clone() const noexcept {
  return new SignPlugin(size_, dtype_);
}

const char* SignPlugin::getPluginType() const noexcept {
  return PLUGIN_NAME;
}

const char* SignPlugin::getPluginVersion() const noexcept {
  return PLUGIN_VERSION;
}

int SignPlugin::getNbOutputs() const noexcept {
  return 1;
}

Dims SignPlugin::getOutputDimensions(
    int32_t index,
    const Dims* inputs,
    int32_t nbInputDims) noexcept {
  TORCH_CHECK(index == 0);
  TORCH_CHECK(nbInputDims == 1);
  //NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  Dims *dims = new Dims{};
  dims->nbDims = inputs[0].nbDims;
  for (auto i = 0; i < dims->nbDims; i++) {
    dims->d[i] = inputs[0].d[i];
  }
  return *dims;
}

int SignPlugin::initialize() noexcept {
  return 0;
}

size_t SignPlugin::getWorkspaceSize(int32_t maxBatchSize) const noexcept {
  return 0;
}

int32_t SignPlugin::enqueue(
    int32_t batchSize,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
  at::Tensor input = at::from_blob(
      (void*)inputs[0], {batchSize, size_}, [](void*) {}, tensor_options_);
  at::Tensor output = at::from_blob(
      (void*)outputs[0], {batchSize, size_}, [](void*) {}, tensor_options_);

  at::cuda::CUDAStreamGuard guard(
      c10::cuda::getStreamFromExternal(stream, c10::cuda::current_device()));
  at::sign_out(output, input);

  return 0;
}

size_t SignPlugin::getSerializationSize() const noexcept {
  return serializeToString().size();
}

void SignPlugin::serialize(void* buffer) const noexcept {
  std::string data = serializeToString();
  size_t size = getSerializationSize();
  data.copy((char*)buffer, size);
}

const char* SignPluginCreator::getPluginName() const noexcept {
  return PLUGIN_NAME;
}

const char* SignPluginCreator::getPluginVersion() const noexcept {
  return PLUGIN_VERSION;
}

IPluginV2* SignPluginCreator::deserializePlugin(
    const char* name,
    const void* data,
    size_t length) noexcept {
  std::istringstream data_stream(std::string((const char*)data, length));
  torch::serialize::InputArchive input_archive;
  input_archive.load_from(data_stream);
  int size = 0;
  //NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  DataType dtype = nvinfer1::DataType::kINT8;
  {
    torch::IValue value;
    input_archive.read("size", value);
    size = value.toInt();
  }
  {
    torch::IValue value;
    input_archive.read("dtype", value);
    dtype = (DataType)value.toInt();
  }
  SignPlugin* obj = nullptr;
  obj = new SignPlugin(size, dtype);
  return obj;
}

const PluginFieldCollection* SignPluginCreator::getFieldNames() noexcept {
  return nullptr;
}

IPluginV2* SignPluginCreator::createPlugin(
    const char* name,
    const PluginFieldCollection* fc) noexcept {
  return new SignPlugin();
}

const char* SignPluginCreator2::getPluginName() const noexcept {
  return "SignPluginfx2trt";
}

const char* SignPluginCreator2::getPluginVersion() const noexcept {
  return PLUGIN_VERSION;
}

IPluginV2* SignPluginCreator2::deserializePlugin(
    const char* name,
    const void* data,
    size_t length) noexcept {
  std::istringstream data_stream(std::string((const char*)data, length));
  torch::serialize::InputArchive input_archive;
  input_archive.load_from(data_stream);
  int size = 0;
  //NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  DataType dtype = nvinfer1::DataType::kINT8;
  {
    torch::IValue value;
    input_archive.read("size", value);
    size = value.toInt();
  }
  {
    torch::IValue value;
    input_archive.read("dtype", value);
    dtype = (DataType)value.toInt();
  }
  SignPlugin* obj = nullptr;
  obj = new SignPlugin(size, dtype);
  return obj;
}

const PluginFieldCollection* SignPluginCreator2::getFieldNames() noexcept {
  return nullptr;
}

IPluginV2* SignPluginCreator2::createPlugin(
    const char* name,
    const PluginFieldCollection* fc) noexcept {
  return new SignPlugin();
}
} // namespace fx2trt
