// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#ifdef USE_C10D_NCCL

#include <memory>
#include <string>
#include <unordered_map>

#include <ATen/ATen.h>
#include <torch/csrc/distributed/c10d/Store.hpp>

#include <nccl.h>
#include <torch/csrc/distributed/c10d/nccl2/NcclApi.hpp>

namespace c10d::nccl2 {

class NCCLBootstrap {
 public:
  NCCLBootstrap(
      c10::intrusive_ptr<c10d::Store> store,
      c10::Device device,
      int rank,
      int comm_size,
      uint64_t generation,
      std::shared_ptr<NcclApi> nccl_api,
      std::chrono::milliseconds timeout);

  // Delete copy and move operations
  NCCLBootstrap(const NCCLBootstrap&) = delete;
  NCCLBootstrap& operator=(const NCCLBootstrap&) = delete;
  NCCLBootstrap(NCCLBootstrap&&) = delete;
  NCCLBootstrap& operator=(NCCLBootstrap&&) = delete;

  ncclComm_t createNcclComm(
      const std::string& name,
      const std::unordered_map<std::string, std::string>& hints = {});

  int getRank() {
    return rank_;
  }
  int getSize() {
    return comm_size_;
  }
  c10::Device getDevice() {
    return device_;
  }

 private:
  ncclUniqueId exchangeUniqueId(std::string_view name);

 private:
  const std::chrono::milliseconds timeout_;
  const uint64_t generation_;

  c10::intrusive_ptr<c10d::Store> store_;
  c10::Device device_;
  std::shared_ptr<NcclApi> nccl_api_;
  int rank_;
  int comm_size_;
};

// Helper function to populate NCCL config from hints
void populateNcclConfigFromHints(
    ncclConfig_t& config,
    const std::unordered_map<std::string, std::string>& hints,
    const std::string& name);

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
