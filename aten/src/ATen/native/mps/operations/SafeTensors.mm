// Copyright © 2025 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

#include <nlohmann/json.hpp>

#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace at::native::mps {

static at::ScalarType safetensors_dtype_to_scalar_type(const std::string& dtype) {
  static const std::unordered_map<std::string, at::ScalarType> dtype_map = {
      {"F32", at::kFloat},
      {"F16", at::kHalf},
      {"BF16", at::kBFloat16},
      {"I64", at::kLong},
      {"I32", at::kInt},
      {"I16", at::kShort},
      {"I8", at::kChar},
      {"U8", at::kByte},
      {"BOOL", at::kBool},
  };
  auto it = dtype_map.find(dtype);
  TORCH_CHECK(
      it != dtype_map.end(),
      "Unsupported safetensors dtype: ",
      dtype,
      ". MPS does not support F64 (float64).");
  return it->second;
}

std::unordered_map<std::string, at::Tensor> mps_load_safetensors(
    const std::string& filename) {
  TORCH_CHECK(at::hasMPS(), "MPS device is not available");

  int fd = ::open(filename.c_str(), O_RDONLY);
  TORCH_CHECK(
      fd >= 0,
      "Failed to open '",
      filename,
      "': ",
      strerror(errno));
  struct FileGuard {
    int fd;
    ~FileGuard() {
      ::close(fd);
    }
  } file_guard{fd};

  // Safetensors format: 8-byte little-endian header length, then JSON header,
  // then raw tensor data. All multibyte integers are little-endian; x86/arm are
  // both little-endian so no byte-swapping needed.
  uint64_t header_size = 0;
  TORCH_CHECK(
      ::read(fd, &header_size, sizeof(header_size)) ==
          (ssize_t)sizeof(header_size),
      "Failed to read safetensors header length from '",
      filename,
      "'");

  std::string header_str(header_size, '\0');
  TORCH_CHECK(
      ::read(fd, header_str.data(), header_size) == (ssize_t)header_size,
      "Failed to read safetensors header from '",
      filename,
      "'");

  nlohmann::json header;
  try {
    header = nlohmann::json::parse(header_str);
  } catch (const nlohmann::json::parse_error& e) {
    TORCH_CHECK(false, "Failed to parse safetensors header: ", e.what());
  }

  // Tensor data begins immediately after the 8-byte length field and the header.
  const size_t data_base_offset = sizeof(header_size) + header_size;

  struct TensorLoad {
    std::string name;
    at::Tensor tensor;
    void* cpu_ptr;
    size_t file_offset;
    size_t byte_count;
  };

  std::vector<TensorLoad> loads;
  loads.reserve(header.size());

  for (auto& [key, meta] : header.items()) {
    if (key == "__metadata__") {
      continue;
    }
    auto dtype =
        safetensors_dtype_to_scalar_type(meta.at("dtype").get<std::string>());
    auto shape = meta.at("shape").get<std::vector<int64_t>>();
    auto offsets = meta.at("data_offsets").get<std::vector<size_t>>();
    TORCH_CHECK(
        offsets.size() == 2,
        "Expected data_offsets to have 2 elements for tensor '",
        key,
        "'");

    at::Tensor t = at::empty(
        shape, at::TensorOptions().dtype(dtype).device(at::kMPS));

    id<MTLBuffer> buffer = getMTLBufferStorage(t);
    void* cpu_ptr = [buffer contents];
    TORCH_CHECK(
        cpu_ptr != nullptr,
        "MPS buffer for tensor '",
        key,
        "' does not have CPU-accessible memory. "
        "load_safetensors requires unified memory (Apple Silicon).");

    loads.push_back(
        {key,
         std::move(t),
         cpu_ptr,
         data_base_offset + offsets[0],
         offsets[1] - offsets[0]});
  }

  // Read all tensor data in parallel: pread is thread-safe and offset-based,
  // so multiple threads can issue concurrent reads on the same fd without
  // serializing on a mutex. GCD distributes work across available CPU cores.
  std::vector<std::string> errors(loads.size());
  auto* loads_ptr = &loads;
  auto* errors_ptr = &errors;
  dispatch_apply(
      loads.size(),
      dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0),
      ^(size_t i) {
        auto& load = (*loads_ptr)[i];
        ssize_t n =
            ::pread(fd, load.cpu_ptr, load.byte_count, (off_t)load.file_offset);
        if (n != (ssize_t)load.byte_count) {
          (*errors_ptr)[i] = "Failed to read tensor '" + load.name +
              "': expected " + std::to_string(load.byte_count) + " bytes, got " +
              std::to_string(n < 0 ? 0 : n);
        }
      });

  for (const auto& err : errors) {
    TORCH_CHECK(err.empty(), err);
  }

  std::unordered_map<std::string, at::Tensor> result;
  result.reserve(loads.size());
  for (auto& load : loads) {
    result[std::move(load.name)] = std::move(load.tensor);
  }
  return result;
}

} // namespace at::native::mps
