#include <torch/csrc/distributed/c10d/intra_node_comm.hpp>

#include <torch/csrc/distributed/c10d/DMAConnectivity.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>

// #include <cuda_runtime.h>

namespace c10d::intra_node_comm {

// NOLINTNEXTLINE(misc-use-internal-linkage)
bool isIntraNodeCommSupported();

static std::vector<std::string> ENABLE_INTRA_NODE_COMM = {
    "ENABLE_INTRA_NODE_COMM"};
// Forces detectedTopology() to return Topology::FULLY_CONNECTED, so
// IntraNodeComm can be used even without NVLink connection. This is only used
// for testing purposes.
static std::vector<std::string> TEST_INTRA_NODE_COMM = {"TEST_INTRA_NODE_COMM"};

static int intraNodeCommIdx = 0;

/**
 * Query the nvlink connection among devices.
 */
static NvlMesh getNvlMesh(const std::vector<int>& rankToDeviceIdx) {
  auto connectivity = detect_dma_connectivity(c10::DeviceType::CUDA, "nvlink");
  NvlMesh nvlMesh = {};
  for (size_t srcRank = 0; srcRank < kMaxDevices; ++srcRank) {
    for (size_t dstRank = 0; dstRank < kMaxDevices; ++dstRank) {
      if (srcRank < rankToDeviceIdx.size() &&
          dstRank < rankToDeviceIdx.size()) {
        nvlMesh[srcRank][dstRank] =
            connectivity
                ->matrix[rankToDeviceIdx[srcRank]][rankToDeviceIdx[dstRank]];
      }
    }
  }
  return nvlMesh;
}

/**
 * Detech topology given a NvlMesh.
 */
static Topology detectTopology(const NvlMesh nvlMesh, size_t worldSize) {
  if (getCvarBool(TEST_INTRA_NODE_COMM, false)) {
    return Topology::FULLY_CONNECTED;
  }
  bool fullyConnected = true;
  for (size_t i = 0; i < worldSize - 1; ++i) {
    for (size_t j = i + 1; j < worldSize; ++j) {
      if (nvlMesh[i][j] == 0 || nvlMesh[j][i] == 0) {
        fullyConnected = false;
      }
    }
  }
  if (fullyConnected) {
    LOG(INFO) << "IntraNodeComm: Topology::FULLY_CONNECTED";
    return Topology::FULLY_CONNECTED;
  }
  LOG(INFO) << "IntraNodeComm: Topology::UNKNOWN";
  return Topology::UNKNOWN;
};

IntraNodeComm::IntraNodeComm(
    c10::intrusive_ptr<c10d::Store> store,
    size_t rank,
    size_t worldSize,
    std::optional<size_t> bufferSize)
    : store_(std::move(store)),
      rank_(rank),
      worldSize_(worldSize),
      bufferSize_(bufferSize.has_value() ? *bufferSize : kDefaultBufferSize) {}

IntraNodeComm::~IntraNodeComm() {
  if (!isInitialized_) {
    return;
  }
  auto allocator = get_allocator(c10::DeviceType::CUDA);
  allocator->free(symmetricMemoryPtr_);
}

bool IntraNodeComm::isEnabled() {
  return getCvarBool(ENABLE_INTRA_NODE_COMM, false);
}

/**
 * Use c10d::Store to perform allgather on a trivially copyable type.
 */
template <typename T>
static std::vector<T> storeAllGather(
    const c10::intrusive_ptr<c10d::Store>& store,
    const std::string& prefix,
    size_t rank,
    size_t worldSize,
    T val) {
  static_assert(std::is_trivially_copyable_v<T>);

  std::vector<std::string> peerKeys;
  for (size_t r = 0; r < worldSize; ++r) {
    std::ostringstream oss;
    oss << prefix << "-" << r;
    peerKeys.push_back(oss.str());
  }

  {
    std::vector<uint8_t> payload(
        reinterpret_cast<uint8_t*>(&val),
        reinterpret_cast<uint8_t*>(&val) + sizeof(T));
    store->set(peerKeys[rank], payload);
  }

  std::vector<T> peerVals;
  for (size_t r = 0; r < worldSize; ++r) {
    if (r == rank) {
      peerVals.push_back(val);
      continue;
    }
    store->wait({peerKeys[r]});
    auto payload = store->get(peerKeys[r]);
    TORCH_CHECK(payload.size() == sizeof(T));
    T peerVal{};
    std::memcpy(&peerVal, payload.data(), sizeof(T));
    peerVals.push_back(peerVal);
  }
  return peerVals;
}

bool IntraNodeComm::rendezvous() {
  if (isInitialized_) {
    return true;
  }
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
  if (!isIntraNodeCommSupported() || worldSize_ < 2 ||
      worldSize_ > kMaxDevices) {
    return false;
  }

  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  deviceIdx_ = at::cuda::current_device();

  // Exchange hostname and device bus ID
  struct DevInfo {
    // NOLINTNEXTLINE
    char hostname[HOST_NAME_MAX + 1];
    int deviceIdx;
  };

  DevInfo devInfo{};
  gethostname(devInfo.hostname, sizeof(devInfo.hostname));
  devInfo.deviceIdx = deviceIdx_;

  auto peerDevInfos =
      storeAllGather(store_, "handshake-0", rank_, worldSize_, devInfo);

  std::vector<int> rankToDeviceIdx;
  for (const auto& info : peerDevInfos) {
    if (strcmp(info.hostname, peerDevInfos.front().hostname) != 0) {
      LOG(WARNING) << "Aborting IntraNodeComm::rendezvous because some "
                      "participants are not on the same host ("
                   << info.hostname << ", " << devInfo.hostname << ")";
      return false;
    }
    rankToDeviceIdx.emplace_back(info.deviceIdx);
  }

  // Query nvlink connection
  auto nvlMesh = getNvlMesh(rankToDeviceIdx);

  // Detect topology
  topology_ = detectTopology(nvlMesh, worldSize_);
  if (topology_ != Topology::FULLY_CONNECTED) {
    return false;
  }

  auto groupName = "IntraNodeComm" + std::to_string(intraNodeCommIdx++);
  set_group_info(
      groupName, static_cast<int>(rank_), static_cast<int>(worldSize_), store_);
  auto allocator = get_allocator(c10::DeviceType::CUDA);
  symmetricMemoryPtr_ = allocator->alloc(bufferSize_, deviceIdx_, groupName);
  symmetricMemory_ = allocator->rendezvous(symmetricMemoryPtr_);
  isInitialized_ = true;
  return true;
#endif
  return false;
}

} // namespace c10d::intra_node_comm
