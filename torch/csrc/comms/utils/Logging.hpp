// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <c10/util/Logging.h>
#include <fmt/core.h>

#include <torch/csrc/comms/TorchCommBackend.hpp>

inline std::string getCommNamePrefix(torch::comms::TorchCommBackend* comm) {
  return comm ? fmt::format("[name={}]", comm->getCommName()) : "";
}

inline std::string getRankPrefix(torch::comms::TorchCommBackend* comm) {
  try {
    return comm ? fmt::format("[rank={}]", comm->getRank()) : "";
  } catch (...) {
    return "";
  }
}

#define TC_LOG_METADATA(comm) \
  "[TC]" << ::getRankPrefix(comm) << ::getCommNamePrefix(comm) << " "

// level is one of the following: INFO, WARNING, ERROR, FATAL
#define TC_LOG_WITH_PREFIX_BUILDER(level, comm) \
  LOG(level) << TC_LOG_METADATA(comm)
#define TC_LOG_PICKER(x, level, comm, FUNC, ...) FUNC
#define TC_LOG(...)                            \
  TC_LOG_PICKER(                               \
      ,                                        \
      ##__VA_ARGS__,                           \
      TC_LOG_WITH_PREFIX_BUILDER(__VA_ARGS__), \
      TC_LOG_WITH_PREFIX_BUILDER(__VA_ARGS__, getDefaultCommunicator()))

namespace {

// Logging initialization is owned by PyTorch (c10) when running inside the
// torch process, so this is intentionally a no-op. Kept for call-site
// compatibility with the original torchcomms code.
[[maybe_unused]] void tryTorchCommLoggingInit(std::string_view /*name*/) {}

[[maybe_unused]] torch::comms::TorchCommBackend* getDefaultCommunicator() {
  static torch::comms::TorchCommBackend* defaultCommunicator = nullptr;
  return defaultCommunicator;
}

} // namespace
