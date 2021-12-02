#include <torch/csrc/fx/fx2trt/plugins/linalg_norm.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Optional.h>

#include <torch/torch.h>
#include <iostream>

namespace fx2trt {


/*
 * LinalgNormPlugin class implementations
 */
LinalgNormPlugin::LinalgNormPlugin(
  std::optional<int32_t> order_int,
  std::optional<std::string> order_inf,
  std::optional<std::string> order_str,
  std::vector<int32_t> axes,
  int32_t keep_dims,
  at::TensorOptions tensor_options,
  std::vector<int64_t> input_sizes,
  std::vector<int64_t> output_sizes)
    : order_int_(order_int), order_inf_(order_inf), order_str_(order_str), axes_(axes), keep_dims_(keep_dims), tensor_options_(tensor_options), input_sizes_(input_sizes), output_sizes_(output_sizes) {
    }

LinalgNormPlugin::LinalgNormPlugin(const char* data, size_t length) {
  std::istringstream data_stream(std::string(data, length));

  torch::serialize::InputArchive input_archive;
  input_archive.load_from(data_stream);
  {
    torch::IValue value;
    order_int_ = std::optional<int32_t>(value.isNone() ? std::optional<int32_t>() : value.toInt());
  }
  {
    torch::IValue value;
    input_archive.read("order_inf", value);
    order_inf_ = std::optional<std::string>(value.isNone() ? std::optional<std::string>() : value.toStringRef());
  }
  {
    torch::IValue value;
    input_archive.read("order_str", value);
    order_str_ = std::optional<std::string>(value.isNone() ? std::optional<std::string>() : value.toStringRef());
  }
  {
    torch::IValue value;
    input_archive.read("axes", value);
    auto values = value.toIntVector();
    //NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<int32_t> doubleVec(values.begin(), values.end());
    axes_.assign(doubleVec.begin(), doubleVec.end());
  }
  {
    torch::IValue value;
    input_archive.read("keep_dims", value);
    keep_dims_ = (int32_t)value.toInt();
  }
  {
    torch::IValue value;
    input_archive.read("input_sizes", value);
#ifdef USE_DEPRECATED_INTLIST
    input_sizes_ = value.toIntListRef().vec();
#else
    input_sizes_ = value.toIntVector();
#endif
  }
  {
    torch::IValue value;
    input_archive.read("output_sizes", value);
#ifdef USE_DEPRECATED_INTLIST
    output_sizes_ = value.toIntListRef().vec();
#else
    output_sizes_ = value.toIntVector();
#endif
  }
  {
    torch::IValue value;
    input_archive.read("is_half", value);
    bool is_half = value.toBool();
    if (is_half) {
      tensor_options_ = tensor_options_.dtype(c10::kHalf);
      dtype_ = DataType::kHALF;
    } else {
      tensor_options_ = tensor_options_.dtype(c10::kFloat);
      dtype_ = DataType::kFLOAT;
    }
  }
}

nvinfer1::DataType LinalgNormPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs)
    const noexcept {
  return inputTypes[0];
}

int LinalgNormPlugin::getNbOutputs() const noexcept {
  return 1;
}

const char* LinalgNormPlugin::getPluginType() const noexcept {
  return "LinalgNormPlugin";
}

const char* LinalgNormPlugin::getPluginVersion() const noexcept {
  return "1";
}


bool LinalgNormPlugin::isOutputBroadcastAcrossBatch(
      int32_t outputIndex,
      bool const* inputIsBroadcasted,
      int32_t nbInputs) const noexcept {
  return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool LinalgNormPlugin::canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept {
  return false;
}

nvinfer1::IPluginV2Ext*  LinalgNormPlugin::clone() const noexcept {
  return new LinalgNormPlugin(order_int_, order_inf_, order_str_, axes_, keep_dims_, tensor_options_, input_sizes_, output_sizes_);
}

nvinfer1::Dims LinalgNormPlugin::getOutputDimensions(
    int index,
    const Dims* inputs,
    int nbInputDims) noexcept {
  //NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  Dims *dims = new Dims{};
  dims->nbDims = keep_dims_? inputs[0].nbDims : inputs[0].nbDims - axes_.size();
  int idx = 0;
  for (int i = 0; i <= inputs->nbDims; i++) {
    if (std::find(axes_.begin(), axes_.end(), i+1) != axes_.end()) {
      if (keep_dims_) {
        dims->d[idx] = 1;
        idx++;
      }
    } else {
      dims->d[idx++] = inputs->d[i];
    }
  }

  return *dims;
}

void LinalgNormPlugin::configurePlugin(
    nvinfer1::PluginTensorDesc const* in,
    int32_t nbInput,
    nvinfer1::PluginTensorDesc const* out,
    int32_t nbOutput) noexcept {
  // set device
  tensor_options_ = tensor_options_.device(c10::kCUDA);

  auto& type = in[0].type;
  // set data type
  if (type == DataType::kFLOAT) {
    tensor_options_ = tensor_options_.dtype(c10::kFloat);
    dtype_ = type;
  } else if (type == DataType::kHALF) {
    tensor_options_ = tensor_options_.dtype(c10::kHalf);
    dtype_ = type;
  }

  // set input sizes
  input_sizes_.resize(in[0].dims.nbDims);
  for (int i = 0; i < in[0].dims.nbDims; i++) {
    input_sizes_[i] = in[0].dims.d[i];
  }

  // set output sizes
  output_sizes_.resize(out[0].dims.nbDims);
  for (int i = 0; i < out[0].dims.nbDims; i++) {
    output_sizes_[i] = out[0].dims.d[i];
  }
}

int LinalgNormPlugin::initialize() noexcept {
  // set device
  tensor_options_ = tensor_options_.device(c10::kCUDA);

  // set data type
  if (dtype_ == DataType::kFLOAT) {
    tensor_options_ = tensor_options_.dtype(c10::kFloat);
  } else if (dtype_ == DataType::kHALF) {
    tensor_options_ = tensor_options_.dtype(c10::kHalf);
  }

  return 0;
}

void LinalgNormPlugin::serialize(void* buffer) const noexcept {
  std::string data = serializeToString();
  size_t size = getSerializationSize();
  data.copy((char*)buffer, size);
}

void LinalgNormPlugin::destroy() noexcept {}

std::string LinalgNormPlugin::serializeToString() const {
  torch::serialize::OutputArchive output_archive;
  //NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<int64_t> axesVec(axes_.begin(), axes_.end());
  c10::optional<int32_t> order_int_opt = order_int_.has_value()?(c10::optional<int32_t>(order_int_.value())):(c10::optional<int32_t>());
  output_archive.write("order_int", torch::IValue(order_int_opt));
  c10::optional<std::string> order_inf_opt = order_inf_.has_value()?(c10::optional<std::string>(order_inf_.value())):(c10::optional<std::string>());
  output_archive.write("order_inf", torch::IValue(order_inf_opt));
  c10::optional<std::string> order_str_opt = order_str_.has_value()?(c10::optional<std::string>(order_str_.value())):(c10::optional<std::string>());
  output_archive.write("order_str", torch::IValue(order_str_opt));
  output_archive.write("axes", torch::IValue(axesVec));
  output_archive.write("keep_dims", torch::IValue((int64_t)keep_dims_));
  output_archive.write("input_sizes", torch::IValue(input_sizes_));
  output_archive.write("output_sizes", torch::IValue(output_sizes_));
  output_archive.write("is_half", torch::IValue(tensor_options_.dtype() == c10::kHalf ? true : false));
  std::ostringstream data_str;
  output_archive.save_to(data_str);

  return data_str.str();
}

size_t LinalgNormPlugin::getSerializationSize() const noexcept {
  return serializeToString().size();
}

bool LinalgNormPlugin::supportsFormatCombination(
    int pos,
    PluginTensorDesc const* inOut,
    int nbInputs,
    int nbOutputs) const noexcept {
  TORCH_CHECK(pos >= 0 && pos <= 1, "There should be exactly 2 connections to the plugin - 1 input, 1 output");
  TORCH_CHECK(nbInputs == 1, "Expected a single tensor as input to linalg norm plugin");
  TORCH_CHECK(nbOutputs == 1, "Expected a single tensor as output to linalg norm plugin");

  const nvinfer1::PluginTensorDesc& in = inOut[0];

  if (pos == 0) {
    return (in.type == nvinfer1::DataType::kFLOAT || in.type == nvinfer1::DataType::kHALF) &&
           (in.format == nvinfer1::TensorFormat::kLINEAR);
  }

  // pos == 1, accessing information about output tensor
  const nvinfer1::PluginTensorDesc& out = inOut[1];

  return (in.type == out.type) && (in.format == out.format);
}

void LinalgNormPlugin::terminate() noexcept {}

size_t LinalgNormPlugin::getWorkspaceSize(int maxBatchSize) const noexcept {
  return 0;
}

int LinalgNormPlugin::enqueue(
    int batchSize,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
  //NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<long> batch_input_sizes = input_sizes_;
  //NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<long> batch_output_sizes = output_sizes_;
  batch_input_sizes.insert(batch_input_sizes.begin(), batchSize);
  batch_output_sizes.insert(batch_output_sizes.begin(), batchSize);

  at::Tensor input = at::from_blob(
      (void*)inputs[0], batch_input_sizes, [](void*) {}, tensor_options_);
  at::Tensor output = at::from_blob(
      outputs[0], batch_output_sizes, [](void*) {}, tensor_options_);
  //NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<int64_t> axes_double(axes_.begin(), axes_.end());

  at::cuda::CUDAStreamGuard guard(
      c10::cuda::getStreamFromExternal(stream, c10::cuda::current_device()));

  //NOLINTNEXTLINE(bugprone-branch-clone)
  if (order_int_.has_value()) {
    at::linalg_norm_out(output, input, (int64_t)order_int_.value(), axes_double, (bool)keep_dims_);
  } else if (order_str_.has_value()){
    at::linalg_norm_out(output, input, order_str_.value(), axes_double, (bool)keep_dims_);
  } else if (order_inf_.has_value()) {
    //NOLINTNEXTLINE(bugprone-branch-clone)
    if (order_inf_.value().compare("INF") == 0) {
      at::linalg_norm_out(output, input, INFINITY, axes_double, (bool)keep_dims_);
    } else {
      at::linalg_norm_out(output, input, -INFINITY, axes_double, (bool)keep_dims_);
    }
  } else {
    at::linalg_norm_out(output, input, c10::nullopt, axes_double, (bool)keep_dims_);
  }

  return 0;
}

/*
 * LinalgNormPluginCreator class implementations
 */
LinalgNormPluginCreator::LinalgNormPluginCreator() = default;

const char* LinalgNormPluginCreator::getPluginNamespace() const noexcept {
  return "fx2trt";
}

const char* LinalgNormPluginCreator::getPluginName() const noexcept {
  return "LinalgNormPlugin";
}

const char* LinalgNormPluginCreator::getPluginVersion() const noexcept {
  return "1";
}

nvinfer1::IPluginV2* LinalgNormPluginCreator::createPlugin(
    const char* name,
    const nvinfer1::PluginFieldCollection* fc) noexcept {
  std::optional<int32_t> order_int_opt;
  std::optional<std::string> order_inf_opt;
  std::optional<std::string> order_str_opt;
  //NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<int32_t> axes = {};
  int32_t keep_dims = 0;

  for (int i = 0; i < fc->nbFields; i++) {
    std::string field_name(fc->fields[i].name);
    if (field_name.compare("ord_int") == 0) {
      order_int_opt = *static_cast<const int32_t*>(fc->fields[i].data);
    } else if (field_name.compare("ord_str") == 0) {
      // This is a temporary way of reading "ord" field data,
      // because TensorRT PluginField can only pass in the first character of "ord" field string data
      const char* ord_head_char = (static_cast<const char*>(fc->fields[i].data));
      if (*ord_head_char == 'f') order_str_opt = "fro";
      else if (*ord_head_char == 'n') order_str_opt = "nuc";
    } else if (field_name.compare("ord_float") == 0) {
      double d = *static_cast<const double*>(fc->fields[i].data);
      if (d > 0) order_inf_opt = "INF";
      else order_inf_opt = "-INF";
    } else if (field_name.compare("dim_val") == 0) {
      auto axes_values = static_cast<const int32_t*>(fc->fields[i].data);
      axes.assign(axes_values, axes_values + fc->fields[i].length);
    } else if (field_name.compare("keep_dims") == 0) {
      keep_dims = *static_cast<const int32_t*>(fc->fields[i].data);
    }
  }

  LinalgNormPlugin* plugin = {};
  plugin = new LinalgNormPlugin(order_int_opt, order_inf_opt, order_str_opt, axes, keep_dims, at::TensorOptions(), std::vector<int64_t>(), std::vector<int64_t>());
  return plugin;
}

nvinfer1::IPluginV2* LinalgNormPluginCreator::deserializePlugin(
    const char* name,
    const void* serialData,
    size_t serialLength) noexcept {
  auto plugin = new LinalgNormPlugin((const char*)serialData, serialLength);
  return plugin;
}

const nvinfer1::PluginFieldCollection* LinalgNormPluginCreator::getFieldNames() noexcept {
  return nullptr;
}


LinalgNormPluginCreator2::LinalgNormPluginCreator2() = default;

const char* LinalgNormPluginCreator2::getPluginNamespace() const noexcept {
  return "fx2trt";
}

const char* LinalgNormPluginCreator2::getPluginName() const noexcept {
  return "LinalgNormPluginfx2trt";
}

const char* LinalgNormPluginCreator2::getPluginVersion() const noexcept {
  return "1";
}

nvinfer1::IPluginV2* LinalgNormPluginCreator2::createPlugin(
    const char* name,
    const nvinfer1::PluginFieldCollection* fc) noexcept {
  std::optional<int32_t> order_int_opt;
  std::optional<std::string> order_inf_opt;
  std::optional<std::string> order_str_opt;
  //NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<int32_t> axes = {};
  int32_t keep_dims = 0;

  for (int i = 0; i < fc->nbFields; i++) {
    std::string field_name(fc->fields[i].name);
    if (field_name.compare("ord_int") == 0) {
      order_int_opt = *static_cast<const int32_t*>(fc->fields[i].data);
    } else if (field_name.compare("ord_str") == 0) {
      // This is a temporary way of reading "ord" field data,
      // because TensorRT PluginField can only pass in the first character of "ord" field string data
      const char* ord_head_char = (static_cast<const char*>(fc->fields[i].data));
      if (*ord_head_char == 'f') order_str_opt = "fro";
      else if (*ord_head_char == 'n') order_str_opt = "nuc";
    } else if (field_name.compare("ord_float") == 0) {
      double d = *static_cast<const double*>(fc->fields[i].data);
      if (d > 0) {
        order_inf_opt = "INF";
      } else {
        order_inf_opt = "-INF";
      }
    } else if (field_name.compare("dim_val") == 0) {
      auto axes_values = static_cast<const int32_t*>(fc->fields[i].data);
      axes.assign(axes_values, axes_values + fc->fields[i].length);
    } else if (field_name.compare("keep_dims") == 0) {
      keep_dims = *static_cast<const int32_t*>(fc->fields[i].data);
    }
  }

  LinalgNormPlugin* plugin = {};
  plugin = new LinalgNormPlugin(order_int_opt, order_inf_opt, order_str_opt, axes, keep_dims, at::TensorOptions(), std::vector<int64_t>(), std::vector<int64_t>());
  return plugin;
}

nvinfer1::IPluginV2* LinalgNormPluginCreator2::deserializePlugin(
    const char* name,
    const void* serialData,
    size_t serialLength) noexcept {
  auto plugin = new LinalgNormPlugin((const char*)serialData, serialLength);
  return plugin;
}

const nvinfer1::PluginFieldCollection* LinalgNormPluginCreator2::getFieldNames() noexcept {
  return nullptr;
}

} // namespace fx2trt
