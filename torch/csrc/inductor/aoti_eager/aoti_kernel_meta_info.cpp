#if !defined(C10_MOBILE) && !defined(ANDROID)
#include <torch/csrc/inductor/aoti_eager/aoti_kernel_meta_info.h>

#include <filesystem>
#include <fstream>

namespace torch::inductor {

TensorMetaInfo::TensorMetaInfo(
    bool is_symbolic,
    c10::ScalarType dtype,
    c10::Device device,
    std::vector<int64_t> sizes,
    std::vector<int64_t> strides)
    : is_symbolic(is_symbolic),
      dtype(dtype),
      device(device),
      sizes(sizes),
      strides(strides) {}

bool TensorMetaInfo::operator==(const TensorMetaInfo& other) const {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      !is_symbolic, "To support symbolic shape now");
  return is_symbolic == other.is_symbolic && dtype == other.dtype &&
      device.type() == other.device.type() && sizes == other.sizes &&
      strides == other.strides;
}

bool TensorMetaInfo::sanityCheck(const TensorMetaInfo& tensor_meta_info) {
  if (tensor_meta_info.dtype == c10::ScalarType::Undefined ||
      tensor_meta_info.device ==
          c10::Device(c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES) ||
      tensor_meta_info.sizes.empty() || tensor_meta_info.strides.empty()) {
    return false;
  }

  return true;
}

AOTIKernelMetaInfo TensorMetaInfo::fromConfig(
    const std::vector<std::string>& tensors_meta_info) {
  auto parse_symbolic = [](const std::string& symbolic_str) -> bool {
    return symbolic_str == "true";
  };

  auto parse_dtype = [](const std::string& dtype_str) -> c10::ScalarType {
    std::string to_remove = "torch.";
    std::string canonicalized_dtype_str = dtype_str;
    size_t start_pos = dtype_str.find(to_remove);
    if (start_pos != std::string::npos) {
      canonicalized_dtype_str =
          dtype_str.substr(start_pos + to_remove.length());
    }

    if (canonicalized_dtype_str == "float32") {
      return c10::ScalarType::Float;
    } else if (canonicalized_dtype_str == "int32") {
      return c10::ScalarType::Int;
    } else if (canonicalized_dtype_str == "int64") {
      return c10::ScalarType::Long;
    } else if (canonicalized_dtype_str == "bool") {
      return c10::ScalarType::Bool;
    } else if (canonicalized_dtype_str == "bfloat16") {
      return c10::ScalarType::BFloat16;
    } else if (canonicalized_dtype_str == "float16") {
      return c10::ScalarType::Half;
    } else if (canonicalized_dtype_str == "float64") {
      return c10::ScalarType::Double;
    } else if (canonicalized_dtype_str == "uint8") {
      return c10::ScalarType::Byte;
    } else if (canonicalized_dtype_str == "int8") {
      return c10::ScalarType::Char;
    } else if (canonicalized_dtype_str == "complex64") {
      return c10::ScalarType::ComplexFloat;
    } else if (canonicalized_dtype_str == "complex128") {
      return c10::ScalarType::ComplexDouble;
    } else {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          false, "Unsupported dtype: ", canonicalized_dtype_str);
      return c10::ScalarType::Undefined;
    }
  };

  auto parse_device = [](const std::string& device_str) -> c10::Device {
    if (device_str == "cpu") {
      return c10::Device(c10::DeviceType::CPU);
    } else if (device_str == "cuda") {
      return c10::Device(c10::DeviceType::CUDA);
    } else {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          false, "Unsupported device: ", device_str);
      return c10::Device(c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES);
    }
  };

  auto parse_sizes_or_strides =
      [](const std::string& sizes_or_strides_str) -> std::vector<int64_t> {
    std::vector<int64_t> sizes_or_strides;
    std::stringstream ss(sizes_or_strides_str);
    std::string size_or_stride_str;
    while (getline(ss, size_or_stride_str, ',')) {
      sizes_or_strides.push_back(std::stoi(size_or_stride_str));
    }
    return sizes_or_strides;
  };

  // config file is in the following format:
  //  ${dtype};${device};${sizes};${strides}
  auto parse_tensor_meta_info = [&](const std::string& line) -> TensorMetaInfo {
    std::stringstream ss(line);
    std::string symbolic_str, dtype_str, device_str, sizes_str, strides_str;
    getline(ss, symbolic_str, ';');
    getline(ss, device_str, ';');
    getline(ss, dtype_str, ';');
    getline(ss, sizes_str, ';');
    getline(ss, strides_str, ';');
    sizes_str.erase(
        std::remove(sizes_str.begin(), sizes_str.end(), '['), sizes_str.end());
    sizes_str.erase(
        std::remove(sizes_str.begin(), sizes_str.end(), ']'), sizes_str.end());
    strides_str.erase(
        std::remove(strides_str.begin(), strides_str.end(), '['),
        strides_str.end());
    strides_str.erase(
        std::remove(strides_str.begin(), strides_str.end(), ']'),
        strides_str.end());
    return TensorMetaInfo(
        parse_symbolic(symbolic_str),
        parse_dtype(dtype_str),
        parse_device(device_str),
        parse_sizes_or_strides(sizes_str),
        parse_sizes_or_strides(strides_str));
  };

  // Suppose there are 3 input tensors, and the config file format is:
  //   ${tensor1_dtype};${tensor1_device};${tensor1_sizes};${tensor1_strides}
  //   ${tensor2_dtype};${tensor2_device};${tensor2_sizes};${tensor2_strides}
  //   ${tensor3_dtype};${tensor3_device};${tensor3_sizes};${tensor3_strides}
  // Parse the config file line by line:
  //   1. Parse each line into a TensorMetaInfo object
  //   2. Sanity check the TensorMetaInfo object
  AOTIKernelMetaInfo aoti_kernel_meta_info;
  for (auto& item : tensors_meta_info) {
    auto tensor_meta_info = parse_tensor_meta_info(item);
    if (sanityCheck(tensor_meta_info)) {
      aoti_kernel_meta_info.push_back(tensor_meta_info);
    }
  }

  return aoti_kernel_meta_info;
}

size_t TensorMetaInfoHash::operator()(
    const TensorMetaInfo& tensor_meta_info) const {
  auto hash = std::hash<bool>()(tensor_meta_info.is_symbolic);
  hash = c10::hash_combine(
      hash, std::hash<c10::ScalarType>()(tensor_meta_info.dtype));
  hash = c10::hash_combine(
      hash, std::hash<c10::DeviceType>()(tensor_meta_info.device.type()));
  for (auto& e : tensor_meta_info.sizes) {
    hash = c10::hash_combine(hash, std::hash<int64_t>()(e));
  }
  for (auto& e : tensor_meta_info.strides) {
    hash = c10::hash_combine(hash, std::hash<int64_t>()(e));
  }
  return hash;
}

size_t AOTIKernelMetaInfoHash::operator()(
    const AOTIKernelMetaInfo& aoti_kernel_meta_info) const {
  size_t hash = 0;
  for (auto& e : aoti_kernel_meta_info) {
    hash = c10::hash_combine(hash, TensorMetaInfoHash()(e));
  }
  return hash;
}

} // namespace torch::inductor
#endif