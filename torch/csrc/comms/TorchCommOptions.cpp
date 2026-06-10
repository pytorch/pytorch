// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/comms/TorchCommOptions.hpp>

#include <sstream>
#include <type_traits>

#include <torch/csrc/comms/utils/Utils.hpp>

namespace torch::comms {

CommOptions::CommOptions() {
  // Check environment variables for options
  abort_process_on_timeout_or_error =
      env_to_value<bool>("TORCHCOMM_ABORT_ON_ERROR", true);

  // Get timeout from environment variable - using 600s as default, which is the
  // same timeout as what ProcessGroupNCCL uses
  uint64_t timeout_seconds =
      env_to_value<uint64_t>("TORCHCOMM_TIMEOUT_SECONDS", 600);
  timeout = std::chrono::seconds(timeout_seconds);

  // Initialize hints to empty map (don't read from environment)
  hints = std::unordered_map<std::string, std::string>();
}

bool CommOptions::operator==(const CommOptions& other) const {
  return (
      abort_process_on_timeout_or_error ==
          other.abort_process_on_timeout_or_error &&
      timeout == other.timeout && hints == other.hints);
}

template <typename T>
T CommOptions::getHint(std::string_view key, const T& default_value) const {
  auto it = hints.find(std::string(key));
  if (it == hints.end()) {
    return default_value;
  }
  const auto& value = it->second;
  if constexpr (std::is_same_v<T, bool>) {
    return string_to_bool(value);
  } else if constexpr (std::is_same_v<T, std::string>) {
    return value;
  } else if constexpr (std::is_unsigned_v<T>) {
    return static_cast<T>(std::stoull(value));
  } else {
    T result;
    std::istringstream ss(value);
    ss >> result;
    if (ss.fail() || !ss.eof()) {
      throw std::invalid_argument(
          "Invalid hint value for key '" + std::string(key) + "': " + value);
    }
    return result;
  }
}

template bool CommOptions::getHint<bool>(std::string_view, const bool&) const;
template size_t CommOptions::getHint<size_t>(std::string_view, const size_t&)
    const;
template std::string CommOptions::getHint<std::string>(
    std::string_view,
    const std::string&) const;

} // namespace torch::comms
