#pragma once

#ifdef USE_C10D_XCCL

#include <chrono>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

#include <ATen/ATen.h>
#include <torch/csrc/distributed/c10d/Store.hpp>

#include <oneapi/ccl.h>
#include <torch/csrc/distributed/c10d/xccl2/XcclApi.hpp>

namespace c10d::xccl2 {

// Communicator bootstrap for the XCCL TorchComms backend. Ported from
// torchcomms' TorchCommXCCLBootstrap; the oneCCL unique ID is exchanged
// through the c10d Store supplied by the ProcessGroup.
class XCCLBootstrap {
 public:
  XCCLBootstrap(
      c10::intrusive_ptr<c10d::Store> store,
      c10::Device device,
      int rank,
      int comm_size,
      uint64_t generation,
      std::shared_ptr<XcclApi> xccl_api,
      std::chrono::milliseconds timeout);

  // Delete copy and move operations
  XCCLBootstrap(const XCCLBootstrap&) = delete;
  XCCLBootstrap& operator=(const XCCLBootstrap&) = delete;
  XCCLBootstrap(XCCLBootstrap&&) = delete;
  XCCLBootstrap& operator=(XCCLBootstrap&&) = delete;

  onecclComm_t createXcclComm(
      const std::string& name,
      const onecclConfig_t& config);

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
  onecclUniqueId exchangeUniqueId(std::string_view name);

 private:
  const std::chrono::milliseconds timeout_;
  const uint64_t generation_;

  c10::intrusive_ptr<c10d::Store> store_;
  c10::Device device_;
  std::shared_ptr<XcclApi> xccl_api_;
  int rank_;
  int comm_size_;
};

// Helper function to populate XCCL config from hints
void populateXcclConfigFromHints(
    onecclConfig_t& config,
    const std::unordered_map<std::string, std::string>& hints,
    const std::string& name);

} // namespace c10d::xccl2

#endif // USE_C10D_XCCL
