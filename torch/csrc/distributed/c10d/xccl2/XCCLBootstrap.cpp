#ifdef USE_C10D_XCCL

#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUFunctions.h>
#include <fmt/core.h>
#include <oneapi/ccl.h>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/xccl2/Logging.hpp>
#include <torch/csrc/distributed/c10d/xccl2/ProcessGroupXCCL.hpp>
#include <torch/csrc/distributed/c10d/xccl2/XCCLBootstrap.hpp>
#include <cstring>
#include <set>

namespace c10d::xccl2 {

XCCLBootstrap::XCCLBootstrap(
    c10::intrusive_ptr<c10d::Store> store,
    c10::Device device,
    int rank,
    int comm_size,
    uint64_t generation,
    std::shared_ptr<XcclApi> xccl_api,
    std::chrono::milliseconds timeout)
    : timeout_(timeout),
      generation_(generation),
      store_(std::move(store)),
      device_(device),
      xccl_api_(std::move(xccl_api)),
      rank_(rank),
      comm_size_(comm_size) {
  TORCH_CHECK(store_ != nullptr, "XCCLBootstrap requires a store");

  if (device_.index() == -1) {
    const auto device_count = c10::xpu::device_count_ensure_non_zero();
    device_ = c10::Device(
        c10::kXPU, static_cast<c10::DeviceIndex>(rank_ % device_count));
    TC_LOG(INFO) << "User did not provide device ID; using device xpu:"
                 << static_cast<int>(device_.index());
  }
}

onecclUniqueId XCCLBootstrap::exchangeUniqueId(std::string_view name) {
  onecclUniqueId uniqueId;

  auto store =
      c10::make_intrusive<::c10d::PrefixStore>(std::string(name), store_);
  auto key = fmt::format("xccl_storekey_{}", generation_);
  if (rank_ == 0) {
    // Generate unique ID on rank 0
    onecclResult_t xcclErr = xccl_api_->getUniqueId(&uniqueId);
    TORCH_CHECK(
        xcclErr == onecclSuccess,
        "Failed to get XCCL unique ID: ",
        xccl_api_->getErrorString(xcclErr));

    // Set the unique ID in the store
    std::vector<uint8_t> vec(
        reinterpret_cast<uint8_t*>(&uniqueId),
        reinterpret_cast<uint8_t*>(&uniqueId) + sizeof(uniqueId));
    store->set(key, vec);
  } else {
    // Other ranks read the broadcast ID
    store->wait({key}, timeout_);
    auto vec = store->get(key);
    TORCH_CHECK(
        vec.size() == sizeof(onecclUniqueId), "Invalid XCCL unique ID size");
    std::memcpy(&uniqueId, vec.data(), sizeof(onecclUniqueId));
  }

  return uniqueId;
}

// TorchComm-layer hint keys that are not part of onecclConfig.
static const std::set<std::string> kLayerHints = {
    std::string(kHintIsHighPriorityStream),
    std::string(kHintMaxEventPoolSize),
};

// NCCL-specific hint keys that onecclConfig has no equivalent for. Listed
// explicitly so that a user porting a hint dict over from nccl2 gets a targeted
// warning rather than the generic "unsupported hint" one.
static const std::set<std::string> kNcclOnlyHints = {
    "trafficClass",
    "traffic_class",
    "commName",
    "collnetEnable",
    "collnet_enable",
    "CTAPolicy",
    "cta_policy",
    "shrinkShare",
    "nvlsCTAs",
    "nvls_ctas",
    "nChannelsPerNetPeer",
    "n_channels_per_net_peer",
    "nvlinkCentricSched",
    "nvlink_centric_sched",
};

// Helper function to populate XCCL config from hints
void populateXcclConfigFromHints(
    onecclConfig_t& config,
    const std::unordered_map<std::string, std::string>& hints,
    const std::string& name) {
  // Iterate over the hints and set the corresponding fields in the config.
  // String fields are const char*, and oneCCL does not document whether it
  // keeps the pointer past communicator creation, so they are strdup'd and
  // deliberately never freed.

  for (const auto& [key, val] : hints) {
    if (kLayerHints.count(key)) {
      continue;
    } else if (key == "blocking") {
      config.blocking = std::stoi(val);
      TC_LOG(INFO) << "[comm=" << name
                   << "] Setting config.blocking=" << config.blocking;
    } else if (key == "cgaClusterSize" || key == "cga_cluster_size") {
      config.cgaClusterSize = std::stoi(val);
      TC_LOG(INFO) << "[comm=" << name << "] Setting config.cgaClusterSize="
                   << config.cgaClusterSize;
    } else if (key == "minCTAs" || key == "min_ctas") {
      config.minCTAs = std::stoi(val);
      TC_LOG(INFO) << "[comm=" << name
                   << "] Setting config.minCTAs=" << config.minCTAs;
    } else if (key == "maxCTAs" || key == "max_ctas") {
      config.maxCTAs = std::stoi(val);
      TC_LOG(INFO) << "[comm=" << name
                   << "] Setting config.maxCTAs=" << config.maxCTAs;
    } else if (key == "netName") {
      config.netName = strdup(val.c_str());
      TC_LOG(INFO) << "[comm=" << name
                   << "] Setting config.netName=" << config.netName;
    } else if (key == "splitShare" || key == "split_share") {
      config.splitShare = std::stoi(val);
      TC_LOG(INFO) << "[comm=" << name
                   << "] Setting config.splitShare=" << config.splitShare;
    } else if (kNcclOnlyHints.count(key)) {
      TC_LOG(WARNING) << "XCCL hint '" << key
                      << "' is NCCL-specific and not supported by oneCCL, "
                         "ignoring for comm '"
                      << name << "'";
    } else {
      TC_LOG(WARNING)
          << "XCCL hint '" << key
          << "' is not supported in this XCCL version, ignoring for comm '"
          << name << "'";
    }
  }
}

onecclComm_t XCCLBootstrap::createXcclComm(
    const std::string& name,
    const onecclConfig_t& base_config) {
  c10::OptionalDeviceGuard deviceGuard(device_);
  onecclUniqueId uniqueId;
  onecclComm_t xccl_comm = nullptr;

  uniqueId = exchangeUniqueId(name);

  onecclConfig_t config = base_config;

  // oneCCL binds the communicator to the device selected on the API side, which
  // is tracked separately from the c10 device guard above, so set it explicitly
  // before initializing the communicator.
  onecclResult_t xcclErr = xccl_api_->setDevice(device_.index());
  TORCH_CHECK(
      xcclErr == onecclSuccess,
      "Failed to set oneCCL device: ",
      xccl_api_->getErrorString(xcclErr));

  xcclErr = xccl_api_->commInitRankConfig(
      &xccl_comm, comm_size_, uniqueId, rank_, &config);
  if (xcclErr != onecclSuccess && xccl_comm != nullptr) {
    // oneCCL may hand back a communicator even on failure. Destroy it rather
    // than abort, since onecclCommAbort is not implemented.
    onecclResult_t destroyErr = xccl_api_->commDestroy(xccl_comm);
    if (destroyErr != onecclSuccess) {
      TC_LOG(ERROR) << "Failed to destroy XCCL communicator after "
                    << "initialization failure: "
                    << xccl_api_->getErrorString(destroyErr);
    }
    xccl_comm = nullptr;
  }
  TORCH_CHECK(
      xccl_comm != nullptr,
      "Failed to initialize XCCL communicator: ",
      xccl_api_->getErrorString(xcclErr));

  return xccl_comm;
}

} // namespace c10d::xccl2

#endif // USE_C10D_XCCL
