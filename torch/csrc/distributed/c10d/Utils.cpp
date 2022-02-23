#include <c10d/Utils.hpp>

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

namespace c10d {

const char* kDistDebugEnvVar = "TORCH_DISTRIBUTED_DEBUG";
const char* kDistDebugDetailLogLevel = "DETAIL";
const char* kDistDebugInfoLogLevel = "INFO";
const char* kDistDebugOffLogLevel = "OFF";

std::string parse_env(const char* env_var_name) {
  char* stringValue = std::getenv(env_var_name);
  std::string res = "N/A";
  if (stringValue != nullptr) {
    res = stringValue;
  }
  return res;
}

DistributedDebugLevel parseDistDebugLevel() {
  std::string debugLevel = parse_env(kDistDebugEnvVar);
  const char* levelStr{nullptr};
  if (debugLevel.compare("N/A") == 0) {
    levelStr = kDistDebugOffLogLevel;
  } else {
    levelStr = debugLevel.c_str();
    TORCH_CHECK(
        strncmp(
            levelStr,
            kDistDebugDetailLogLevel,
            strlen(kDistDebugDetailLogLevel)) == 0 ||
            strncmp(
                levelStr,
                kDistDebugInfoLogLevel,
                strlen(kDistDebugInfoLogLevel)) == 0 ||
            strncmp(
                levelStr,
                kDistDebugOffLogLevel,
                strlen(kDistDebugOffLogLevel)) == 0,
        c10::str(
            "Expected environment variable TORCH_DISTRIBUTED_DEBUG to be one of ",
            kDistDebugDetailLogLevel,
            " ",
            kDistDebugInfoLogLevel,
            " ",
            kDistDebugOffLogLevel,
            " "));
    C10_LOG_FIRST_N(INFO, 1)
        << "TORCH_DISTRIBUTED_DEBUG level parsed as " << levelStr;
  }

  static std::unordered_map<std::string, DistributedDebugLevel> mapping = {
      {kDistDebugOffLogLevel, DistributedDebugLevel::OFF},
      {kDistDebugInfoLogLevel, DistributedDebugLevel::INFO},
      {kDistDebugDetailLogLevel, DistributedDebugLevel::DETAIL}};

  auto it = mapping.find(levelStr);
  TORCH_CHECK(
      it != mapping.end(),
      "Invalid string value for distributed debug mode: ",
      levelStr);
  return it->second;
}

std::vector<at::Tensor> getTensorShapes(
    const std::vector<at::Tensor>& tensors) {
  std::vector<at::Tensor> shapeTensors;
  shapeTensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    // Use `at::tensor()` to copy the data underlying `sizes()` since it may be
    // released elsewhere.
    at::Tensor shapesTensor =
        at::tensor(tensor.sizes(), at::TensorOptions().dtype(at::kLong));
    shapeTensors.emplace_back(std::move(shapesTensor));
  }
  return shapeTensors;
}

} // namespace c10d
