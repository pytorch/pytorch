// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/comms/gloo/TorchCommGloo.hpp>

#include <cstring>
#include <string>

#include <fmt/core.h>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp> // @manual
#include <torch/csrc/distributed/c10d/TCPStore.hpp> // @manual

#include <torch/csrc/comms/utils/Logging.hpp>
#include <torch/csrc/comms/utils/TracingGuard.hpp>
#include <torch/csrc/comms/utils/Utils.hpp>

namespace torch::comms {

InitHandle TorchCommGloo::getInitHandle() const {
  std::string storeAddr;
  if (reconfigure_store_) {
    auto* tcpStore = dynamic_cast<c10d::TCPStore*>(reconfigure_store_.get());
    if (!tcpStore) {
      auto* prefixStore =
          dynamic_cast<c10d::PrefixStore*>(reconfigure_store_.get());
      if (prefixStore) {
        tcpStore = dynamic_cast<c10d::TCPStore*>(
            prefixStore->getUnderlyingNonPrefixStore().get());
      }
    }
    if (tcpStore) {
      storeAddr =
          fmt::format("{}:{}", tcpStore->getHost(), tcpStore->getPort());
    }
  }
  return fmt::format("gloo:{}:{}:{}", rank_, uuid_, storeAddr);
}

namespace {

struct GlooHandleInfo {
  int rank;
  int64_t uuid;
  std::string storeAddr;
};

GlooHandleInfo parseGlooHandle(const InitHandle& handle) {
  auto first = handle.find(':');
  if (first == std::string::npos) {
    return {-1, -1, ""};
  }
  auto second = handle.find(':', first + 1);
  if (second == std::string::npos) {
    return {std::stoi(handle.substr(first + 1)), -1, ""};
  }
  int rank = std::stoi(handle.substr(first + 1, second - first - 1));
  auto third = handle.find(':', second + 1);
  if (third == std::string::npos) {
    return {rank, std::stoll(handle.substr(second + 1)), ""};
  }
  int64_t uuid = std::stoll(handle.substr(second + 1, third - second - 1));
  std::string storeAddr = handle.substr(third + 1);
  return {rank, uuid, storeAddr};
}

std::string extractGlooStoreAddr(
    const std::variant<std::unordered_set<InitHandle>, std::vector<InitHandle>>&
        handles) {
  std::string addr;
  std::visit(
      [&](const auto& h) {
        for (const auto& handle : h) {
          auto info = parseGlooHandle(handle);
          if (!info.storeAddr.empty()) {
            addr = info.storeAddr;
            return;
          }
        }
      },
      handles);
  return addr;
}

c10::intrusive_ptr<c10d::Store> connectGlooStore(
    const std::string& storeAddr,
    const std::string& prefix,
    std::chrono::milliseconds timeout) {
  auto colonPos = storeAddr.rfind(':');
  TORCH_INTERNAL_ASSERT(
      colonPos != std::string::npos,
      "Invalid store address in handle: ",
      storeAddr);
  std::string host = storeAddr.substr(0, colonPos);
  uint16_t port =
      static_cast<uint16_t>(std::stoi(storeAddr.substr(colonPos + 1)));

  c10d::TCPStoreOptions opts;
  opts.port = port;
  opts.isServer = false;
  opts.waitWorkers = false;
  opts.useLibUV = true;
  opts.timeout = timeout;

  return c10::make_intrusive<c10d::PrefixStore>(
      prefix, c10::make_intrusive<c10d::TCPStore>(host, opts));
}

} // namespace

c10::intrusive_ptr<TorchWork> TorchCommGloo::reconfigure(
    const ReconfigureOptions& opts) {
  context_.reset();
  store_.reset();
  collectiveCounter_ = 0;
  comm_state_ = CommState::NORMAL;

  int new_size = static_cast<int>(
      std::visit([](const auto& h) { return h.size(); }, opts.handles));
  comm_size_ = new_size;

  auto reconfigureTimeout = opts.timeout.value_or(options_.timeout);
  auto storeAddr = extractGlooStoreAddr(opts.handles);

  if (!storeAddr.empty() && new_size > 1) {
    auto store = connectGlooStore(
        storeAddr, fmt::format("{}/{}", name_, opts.uuid), reconfigureTimeout);
    rank_ = static_cast<int>(store->add("rank_counter", 1)) - 1;
  } else if (new_size > 1) {
    auto [rank, envCommSize] = query_ranksize();
    (void)envCommSize;
    rank_ = rank;
  } else {
    rank_ = 0;
  }

  auto storePrefix = fmt::format("{}/reconfigure/{}", name_, opts.uuid);

  CommOptions connectOpts = options_;
  connectOpts.timeout = reconfigureTimeout;
  connectOpts.store = reconfigure_store_;

  connectGlooContext(connectOpts, storePrefix);

  init_state_ = InitializationState::INITIALIZED;
  uuid_ = opts.uuid;

  TracingGuard tracingGuard(name_, comm_size_, "reconfigure", rank_);

  TC_LOG(INFO, this) << "TorchCommGloo reconfigure completed for rank: "
                     << rank_;

  return c10::make_intrusive<TorchWorkCompleted>();
}

} // namespace torch::comms
