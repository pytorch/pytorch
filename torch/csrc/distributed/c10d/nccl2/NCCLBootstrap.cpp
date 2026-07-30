// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef USE_C10D_NCCL

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <fmt/core.h>
#include <nccl.h>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/nccl2/Logging.hpp>
#include <torch/csrc/distributed/c10d/nccl2/NCCLBootstrap.hpp>
#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/nccl2/Utils.hpp>
#include <set>

namespace c10d::nccl2 {

NCCLBootstrap::NCCLBootstrap(
    c10::intrusive_ptr<c10d::Store> store,
    c10::Device device,
    int rank,
    int comm_size,
    uint64_t generation,
    std::shared_ptr<NcclApi> nccl_api,
    std::chrono::milliseconds timeout)
    : timeout_(timeout),
      generation_(generation),
      store_(std::move(store)),
      device_(device),
      nccl_api_(std::move(nccl_api)),
      rank_(rank),
      comm_size_(comm_size) {
  TORCH_CHECK(store_ != nullptr, "NCCLBootstrap requires a store");

  if (device_.index() == -1) {
    const auto device_count = c10::cuda::device_count_ensure_non_zero();
    device_ = c10::Device(
        c10::kCUDA, static_cast<c10::DeviceIndex>(rank_ % device_count));
    TC_LOG(INFO) << "User did not provide device ID; using device cuda:"
                 << static_cast<int>(device_.index());
  }
}

ncclUniqueId NCCLBootstrap::exchangeUniqueId(std::string_view name) {
  ncclUniqueId uniqueId;

  auto store =
      c10::make_intrusive<::c10d::PrefixStore>(std::string(name), store_);
  auto key = fmt::format("nccl_storekey_{}", generation_);
  if (rank_ == 0) {
    // Generate unique ID on rank 0
    ncclResult_t ncclErr = nccl_api_->getUniqueId(&uniqueId);
    if (ncclErr != ncclSuccess) {
      throw std::runtime_error(
          "Failed to get NCCL unique ID: " +
          std::string(nccl_api_->getErrorString(ncclErr)));
    }

    // Set the unique ID in the store
    std::vector<uint8_t> vec(
        reinterpret_cast<uint8_t*>(&uniqueId),
        reinterpret_cast<uint8_t*>(&uniqueId) + sizeof(uniqueId));
    store->set(key, vec);
  } else {
    // Other ranks read the broadcast ID
    store->wait({key}, timeout_);
    auto vec = store->get(key);
    if (vec.size() != sizeof(ncclUniqueId)) {
      throw std::runtime_error("Invalid NCCL unique ID size");
    }
    uniqueId = *(reinterpret_cast<const ncclUniqueId*>(vec.data()));
  }

  return uniqueId;
}

// TorchComm-layer hint keys that are consumed by the backend init code
// (ProcessGroupNCCL::init), not by ncclConfig.  Skip them here to avoid
// spurious "unsupported hint" warnings.
static const std::set<std::string> kLayerHints = {
    "is_high_priority_stream",
    std::string(kHintMaxEventPoolSize),
};

// Helper function to populate NCCL config from hints
void populateNcclConfigFromHints(
    ncclConfig_t& config,
    const std::unordered_map<std::string, std::string>& hints,
    const std::string& name) {
  // Iterate over the hints and set the corresponding fields in the config.  For
  // string arguments, NCCL uses a "const char*" instead of a std::string.  The
  // strings only need to be valid for the duration of the
  // ncclCommInitRankConfig call, so we use .c_str() directly.

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
      config.netName = val.c_str();
      TC_LOG(INFO) << "[comm=" << name
                   << "] Setting config.netName=" << config.netName;
    } else if (key == "splitShare" || key == "split_share") {
      config.splitShare = std::stoi(val);
      TC_LOG(INFO) << "[comm=" << name
                   << "] Setting config.splitShare=" << config.splitShare;
    }
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 0)
    else if (key == "trafficClass" || key == "traffic_class") {
      config.trafficClass = std::stoi(val);
      TC_LOG(INFO) << "[comm=" << name
                   << "] Setting config.trafficClass=" << config.trafficClass;
    } else if (key == "commName") {
      config.commName = val.c_str();
      TC_LOG(INFO) << "[comm=" << name
                   << "] Setting config.commName=" << config.commName;
    } else if (key == "collnetEnable" || key == "collnet_enable") {
      config.collnetEnable = std::stoi(val);
      TC_LOG(INFO) << "[comm=" << name
                   << "] Setting config.collnetEnable=" << config.collnetEnable;
    } else if (key == "CTAPolicy" || key == "cta_policy") {
      config.CTAPolicy = std::stoi(val);
      TC_LOG(INFO) << "[comm=" << name
                   << "] Setting config.CTAPolicy=" << config.CTAPolicy;
    } else if (key == "shrinkShare") {
      config.shrinkShare = std::stoi(val);
      TC_LOG(INFO) << "[comm=" << name
                   << "] Setting config.shrinkShare=" << config.shrinkShare;
    } else if (key == "nvlsCTAs" || key == "nvls_ctas") {
      config.nvlsCTAs = std::stoi(val);
      TC_LOG(INFO) << "[comm=" << name
                   << "] Setting config.nvlsCTAs=" << config.nvlsCTAs;
    }
#elif NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
    else if (key == "nChannelsPerNetPeer" || key == "n_channels_per_net_peer") {
      config.nChannelsPerNetPeer = std::stoi(val);
      TC_LOG(INFO) << "[comm=" << name
                   << "] Setting config.nChannelsPerNetPeer="
                   << config.nChannelsPerNetPeer;
    } else if (key == "nvlinkCentricSched" || key == "nvlink_centric_sched") {
      config.nvlinkCentricSched = std::stoi(val);
      TC_LOG(INFO) << "[comm=" << name << "] Setting config.nvlinkCentricSched="
                   << config.nvlinkCentricSched;
    }
#endif
    else {
      TC_LOG(WARNING)
          << "NCCL hint '" << key
          << "' is not supported in this NCCL version, ignoring for comm '"
          << name << "'";
    }
  }
}

ncclComm_t NCCLBootstrap::createNcclComm(
    const std::string& name,
    const std::unordered_map<std::string, std::string>& hints) {
  c10::cuda::CUDAGuard gpuGuard(device_);
  ncclUniqueId uniqueId;
  ncclComm_t nccl_comm = nullptr;

  uniqueId = exchangeUniqueId(name);

  // TODO: add logging on failures and successes
  // TODO: use scalable init
  // TODO: get the local rank
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 0)
  config.commName = name.c_str();
#endif

  // Populate NCCL config from user-provided hints
  populateNcclConfigFromHints(config, hints, name);

  ncclResult_t ncclErr = nccl_api_->commInitRankConfig(
      &nccl_comm, comm_size_, uniqueId, rank_, &config);
  if (ncclErr != ncclSuccess || nccl_comm == nullptr) {
    throw std::runtime_error(
        "Failed to initialize NCCL communicator: " +
        std::string(nccl_api_->getErrorString(ncclErr)));
  }

  return nccl_comm;
}

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
