#if !defined(C10_MOBILE) && !defined(ANDROID)
#include <torch/csrc/inductor/aoti_eager/kernel_meta_info.h>

#include <torch/csrc/inductor/aoti_eager/kernel_checker.h>
#include <torch/csrc/inductor/aoti_eager/static_checker.h>

#include <filesystem>
#include <fstream>

namespace torch::inductor {

TensorMetaInfo::TensorMetaInfo(const at::Tensor& src_tensor)
    : is_symbolic(false),
      device(src_tensor.device()),
      sizes(src_tensor.sym_sizes().vec()),
      strides(src_tensor.sym_strides().vec()) {
  for (const auto& size : sizes) {
    if (size.is_symbolic()) {
      is_symbolic = true;
      break;
    }
  }

  if (!is_symbolic) {
    for (const auto& stride : strides) {
      if (stride.is_symbolic()) {
        is_symbolic = true;
        break;
      }
    }
  }

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      !is_symbolic,
      "Eager through torch.compile does not support symbolic shape now.");
  // TODO: Support symbolic shape
  tensor_checker = std::make_shared<StaticTensorChecker>(src_tensor);
}

TensorMetaInfo::TensorMetaInfo(
    bool is_symbolic,
    c10::ScalarType dtype,
    c10::Device device,
    std::vector<c10::SymInt> sizes,
    std::vector<c10::SymInt> strides)
    : is_symbolic(is_symbolic),
      dtype(dtype),
      device(device),
      sizes(sizes),
      strides(strides) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      !is_symbolic,
      "Eager through torch.compile does not support symbolic shape now");
  tensor_checker = std::make_shared<StaticTensorChecker>(*this);
}

bool TensorMetaInfo::operator==(const TensorMetaInfo& other) const {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      !is_symbolic, "To support symbolic shape now");
  return tensor_checker->check(other);
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

AOTIKernelMetaInfo TensorMetaInfo::loadFromFile(
    const std::vector<nlohmann::json>& tensors_meta_info) {
  auto parse_symbolic = [](const std::string& symbolic_str) -> bool {
    std::string upper_symbolic_str;
    upper_symbolic_str.resize(symbolic_str.size());
    std::transform(
        symbolic_str.begin(),
        symbolic_str.end(),
        upper_symbolic_str.begin(),
        [](unsigned char c) -> unsigned char { return std::toupper(c); });
    return upper_symbolic_str == "TRUE";
  };

  auto parse_dtype = [](const std::string& dtype_str) -> c10::ScalarType {
    // The dtype format is torch.float32, float32, torch.int32, int32, etc.
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

  // config file is in the following format:
  //  ${is_symbolic};${dtype};${device};[${sizes}];[${strides}]
  auto parse_tensor_meta_info =
      [&](const nlohmann::json& line) -> TensorMetaInfo {
    std::string symbolic_str = line["is_symbloic"];
    std::string device_str = line["device_type"];
    std::string dtype_str = line["dtype"];
    std::string sizes_str = line["sizes"];
    std::string strides_str = line["strides"];
    nlohmann::json sizes_json = nlohmann::json::parse(sizes_str);
    nlohmann::json strides_json = nlohmann::json::parse(strides_str);

    auto is_symbolic = parse_symbolic(symbolic_str);
    TORCH_INTERNAL_ASSERT(
        is_symbolic == false,
        "Eager through torch.compile does not support symbolic shape now");
    std::vector<int> sizes_vec = sizes_json.get<std::vector<int>>();
    std::vector<int> strides_vec = strides_json.get<std::vector<int>>();

    std::vector<c10::SymInt> sizes;
    std::vector<c10::SymInt> strides;
    std::transform(
        sizes_vec.begin(),
        sizes_vec.end(),
        std::back_inserter(sizes),
        [](int size) { return c10::SymInt(size); });
    std::transform(
        strides_vec.begin(),
        strides_vec.end(),
        std::back_inserter(strides),
        [](int stride) { return c10::SymInt(stride); });

    return TensorMetaInfo(
        is_symbolic,
        parse_dtype(dtype_str),
        parse_device(device_str),
        sizes,
        strides);
  };

  // Suppose there are 3 input tensors, and the config file format will be as
  // follows:
  //   ${is_symbolic};${tensor1_dtype};${tensor1_device};[${tensor1_sizes}];[${tensor1_strides}]
  //   ${is_symbolic};${tensor2_dtype};${tensor2_device};[${tensor2_sizes}];[${tensor2_strides}]
  //   ${is_symbolic};${tensor3_dtype};${tensor3_device};[${tensor3_sizes}];[${tensor3_strides}]
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
    if (!e.is_symbolic()) {
      hash = c10::hash_combine(hash, std::hash<int64_t>()(e.expect_int()));
    }
  }

  for (auto& e : tensor_meta_info.strides) {
    if (!e.is_symbolic()) {
      hash = c10::hash_combine(hash, std::hash<int64_t>()(e.expect_int()));
    }
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
