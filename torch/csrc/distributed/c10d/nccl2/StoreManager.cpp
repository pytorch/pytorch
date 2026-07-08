// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <torch/csrc/distributed/c10d/nccl2/StoreManager.hpp>

#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/nccl2/Logging.hpp>
#include <torch/csrc/distributed/c10d/nccl2/Utils.hpp>

namespace c10d::nccl2 {

c10::intrusive_ptr<c10d::Store> createPrefixStore(
    const std::string& prefix,
    std::chrono::milliseconds timeout) {
  const char* master_addr_env = std::getenv("MASTER_ADDR");
  TORCH_INTERNAL_ASSERT(
      master_addr_env != nullptr, "MASTER_ADDR env is not set");
  std::string host{master_addr_env};
  const char* master_port_env = std::getenv("MASTER_PORT");
  TORCH_INTERNAL_ASSERT(
      master_port_env != nullptr, "MASTER_PORT env is not set");
  int port{std::stoi(master_port_env)};

  auto [rank, comm_size] = query_ranksize();
  (void)comm_size;

  c10d::TCPStoreOptions opts;
  opts.port = port;
  opts.isServer = (rank == 0);
  opts.waitWorkers = false;
  opts.useLibUV = true;
  opts.timeout = timeout;

  return c10::make_intrusive<c10d::PrefixStore>(
      prefix, c10::make_intrusive<c10d::TCPStore>(host, opts));
}

c10::intrusive_ptr<c10d::Store> dupPrefixStore(
    const std::string& prefix,
    const c10::intrusive_ptr<c10d::Store>& bootstrapStore,
    std::chrono::milliseconds timeout) {
  const char* master_addr_env = std::getenv("MASTER_ADDR");
  TORCH_INTERNAL_ASSERT(
      master_addr_env != nullptr, "MASTER_ADDR env is not set");
  std::string host{master_addr_env};

  auto [rank, comm_size] = query_ranksize();
  (void)comm_size;

  const std::string key = "dup_store_port";
  c10::intrusive_ptr<c10d::TCPStore> tcpStore;

  if (rank == 0) {
    c10d::TCPStoreOptions opts;
    opts.port = 0;
    opts.isServer = true;
    opts.waitWorkers = false;
    opts.useLibUV = true;
    opts.timeout = timeout;
    tcpStore = c10::make_intrusive<c10d::TCPStore>(host, opts);

    std::string portStr = std::to_string(tcpStore->getPort());
    bootstrapStore->set(
        key, std::vector<uint8_t>(portStr.begin(), portStr.end()));
  } else {
    bootstrapStore->wait({key}, timeout);
    auto portVec = bootstrapStore->get(key);
    uint16_t port = static_cast<uint16_t>(
        std::stoi(std::string(portVec.begin(), portVec.end())));

    c10d::TCPStoreOptions opts;
    opts.port = port;
    opts.isServer = false;
    opts.waitWorkers = false;
    opts.useLibUV = true;
    opts.timeout = timeout;
    tcpStore = c10::make_intrusive<c10d::TCPStore>(host, opts);
  }

  return c10::make_intrusive<c10d::PrefixStore>(prefix, tcpStore);
}

} // namespace c10d::nccl2
