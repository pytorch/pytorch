// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#ifdef USE_C10D_NCCL

#include <c10/util/Logging.h>
#include <fmt/format.h>

#include <string>

// Lightweight logging helpers for the in-tree NCCL TorchComms backend.
//
// Upstream torchcomms routes TC_LOG through glog and lazily calls
// InitGoogleLogging. PyTorch builds with USE_GLOG=OFF by default, so this
// header routes TC_LOG through c10's LOG() (c10/util/Logging.h provides a
// non-glog LOG() implementation) and drops the glog-init dance and the
// getDefaultCommunicator() fallback entirely.

namespace c10d::nccl2 {

// Prefix helpers are templated on the comm type so this header carries no
// dependency on the backend class (which itself includes this header).
template <typename Comm>
inline std::string getCommNamePrefix(Comm* comm) {
  return comm ? fmt::format("[name={}]", comm->getCommName()) : "";
}

template <typename Comm>
inline std::string getRankPrefix(Comm* comm) {
  try {
    return comm ? fmt::format("[rank={}]", comm->getRank()) : "";
  } catch (...) {
    return "";
  }
}

} // namespace c10d::nccl2

#define TC_LOG_METADATA(comm)                  \
  "[TC]" << ::c10d::nccl2::getRankPrefix(comm) \
         << ::c10d::nccl2::getCommNamePrefix(comm) << " "

// level is one of: INFO, WARNING, ERROR, FATAL
#define TC_LOG_WITH_PREFIX_BUILDER(level, comm) \
  LOG(level) << TC_LOG_METADATA(comm)
#define TC_LOG_PLAIN(level) LOG(level) << "[TC] "
#define TC_LOG_PICKER(x, level, comm, FUNC, ...) FUNC
#define TC_LOG(...)                            \
  TC_LOG_PICKER(                               \
      ,                                        \
      ##__VA_ARGS__,                           \
      TC_LOG_WITH_PREFIX_BUILDER(__VA_ARGS__), \
      TC_LOG_PLAIN(__VA_ARGS__))

#endif // USE_C10D_NCCL
