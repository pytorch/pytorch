#include <torch/csrc/distributed/c10d/intra_node_comm.hpp>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Logging.h>
#include <torch/csrc/distributed/c10d/Utils.hpp>

#include <iostream>
#include <utility>

#include <fcntl.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/driver_api.h>
#include <nvml.h>
#endif

#include <cuda_runtime.h>

namespace c10d::intra_node_comm {

static std::vector<std::string> ENABLE_INTRA_NODE_COMM = {
    "ENABLE_INTRA_NODE_COMM"};
// Forces detectedTopology() to return Topology::FULLY_CONNECTED, so
// IntraNodeComm can be used even without NVLink connection. This is only used
// for testing purposes.
static std::vector<std::string> TEST_INTRA_NODE_COMM = {"TEST_INTRA_NODE_COMM"};

////////////////////////////////////////////////////////////////////////////////
// CUDA Functions
////////////////////////////////////////////////////////////////////////////////

bool isIntraNodeCommSupported();

std::optional<HybridCubeMesh> getHybridCubeMesh(NvlMesh nvlMesh);

void* initP2pState();

void* initTopoInfo(Topology topology, NvlMesh nvlMesh, size_t rank);

////////////////////////////////////////////////////////////////////////////////
// Topology Detection
////////////////////////////////////////////////////////////////////////////////

static std::ostream& operator<<(std::ostream& os, const NvlMesh& nvlMesh) {
  std::ostringstream oss;
  for (size_t i = 0; i < kMaxDevices; ++i) {
    for (size_t j = 0; j < kMaxDevices; ++j) {
      oss << nvlMesh[i][j] << " ";
    }
    oss << '\n';
  }
  os << oss.str();
  return os;
}

static bool isSame(NvlMesh lhs, NvlMesh rhs) {
  for (size_t i = 0; i < kMaxDevices; ++i) {
    for (size_t j = 0; j < kMaxDevices; ++j) {
      if (lhs[i][j] != rhs[i][j]) {
        return false;
      }
    }
  }
  return true;
}

/**
 * Query the nvlink connection among devices.
 */
static NvlMesh getNvlMesh(const std::vector<std::string>& rankToBusId) {
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
  using namespace c10::cuda;

  NvlMesh nvlMesh = {};
  auto driverApi = DriverAPI::get();
  if (driverApi == nullptr) {
    return nvlMesh;
  }

  const auto worldSize = rankToBusId.size();
  std::vector<nvmlDevice_t> devices(worldSize, nullptr);
  std::unordered_map<std::string, size_t> busIdToRank;
  std::vector<size_t> switchLinkCount(worldSize, 0);

  for (size_t r = 0; r < worldSize; ++r) {
    busIdToRank.emplace(rankToBusId[r], r);
    TORCH_CHECK(
        driverApi->nvmlDeviceGetHandleByPciBusId_v2_(
            rankToBusId[r].c_str(), &devices[r]) == NVML_SUCCESS);
  }

  // TODO: find a better way to determine this
  constexpr size_t kMaxNvLinks = 20;

  // For each device, loop over devices connected to it via NVLink
  for (size_t idx = 0; idx < worldSize; ++idx) {
    for (size_t link = 0; link < kMaxNvLinks; ++link) {
      nvmlReturn_t ret;
      nvmlIntNvLinkDeviceType_t deviceType;
      ret = driverApi->nvmlDeviceGetNvLinkRemoteDeviceType_(
          devices[idx], link, &deviceType);
      if (ret != NVML_SUCCESS) {
        // We've exhausted the NVLinks connected to this device.
        // This error is benign. There doesn't seem to be a reliable
        // way to obtain the maximum link value that can be passed to
        // the API, so we simply increment the link value until the
        // API fails or we hit a predefined maximum value.
        break;
      }
      // Remote device is GPU
      if (deviceType == NVML_NVLINK_DEVICE_TYPE_GPU) {
        nvmlPciInfo_t pciInfo;
        ret = driverApi->nvmlDeviceGetNvLinkRemotePciInfo_v2_(
            devices[idx], link, &pciInfo);
        if (ret != NVML_SUCCESS) {
          // Unexpected error. Return an empty NvlMesh
          return {};
        }
        auto it = busIdToRank.find(pciInfo.busId);
        if (it != busIdToRank.end()) {
          if (idx != it->second) {
            nvlMesh[idx][it->second] += 1;
          }
        }
        // Remote device is NVSwitch
      } else if (deviceType == NVML_NVLINK_DEVICE_TYPE_SWITCH) {
        switchLinkCount[idx] += 1;
      }
    }
  }
  // Process NVSwitch connections. For simplicity, we assume
  // all NVSwitches are interconnected.
  for (size_t i = 0; i < worldSize; ++i) {
    for (size_t j = 0; j < worldSize; ++j) {
      if (i == j) {
        continue;
      }
      nvlMesh[i][j] += std::min(switchLinkCount[i], switchLinkCount[j]);
    }
  }
  return nvlMesh;
#else
  return {};
#endif
}

/**
 * Determine if the devices form a hybrid cube mesh
 * topology given a NvlMesh.
 */
static bool isHybridCubeMesh(const NvlMesh nvlMesh) {
  std::array<size_t, kMaxDevices> numNeighbors = {};
  for (size_t i = 0; i < kMaxDevices; ++i) {
    for (size_t j = 0; j < kMaxDevices; ++j) {
      if (nvlMesh[i][j] > 0) {
        numNeighbors[i] += 1;
      }
    }
  }
  for (size_t i = 0; i < kMaxDevices; ++i) {
    // TODO: this is insufficent and needs revisit
    if (numNeighbors[i] != 4) {
      return false;
    }
  }
  return true;
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
  if (worldSize == kMaxDevices && getHybridCubeMesh(nvlMesh) != std::nullopt) {
    LOG(INFO) << "IntraNodeComm: Topology::HYBRID_CUBE_MESH";
    return Topology::HYBRID_CUBE_MESH;
  }
  LOG(INFO) << "IntraNodeComm: Topology::UNKNOWN";
  return Topology::UNKNOWN;
};

////////////////////////////////////////////////////////////////////////////////
// Rendezvous and Initialization
////////////////////////////////////////////////////////////////////////////////

IntraNodeComm::IntraNodeComm(
    c10::intrusive_ptr<c10d::Store> store,
    size_t rank,
    size_t worldSize,
    std::optional<size_t> bufferSize)
    : store_(std::move(store)),
      rank_(rank),
      worldSize_(worldSize),
      bufferSize_(bufferSize.has_value() ? *bufferSize : kDefaultBufferSize),
      barrierReady_(at::cuda::CUDAEvent()) {}

IntraNodeComm::~IntraNodeComm() {
  if (!isInitialized_) {
    return;
  }
  // Intentionally releasing resources without synchronizing devices. The
  // teardown logic is safe for propoerly sync'd user program. We don't want
  // improperly sync'd user program to hang here.
  for (size_t r = 0; r < worldSize_; ++r) {
    if (r == rank_) {
      continue;
    }
    AT_CUDA_CHECK(cudaIpcCloseMemHandle(p2pStates_[r]));
    AT_CUDA_CHECK(cudaIpcCloseMemHandle(buffers_[r]));
  }
  AT_CUDA_CHECK(cudaFree(p2pStates_[rank_]));
  AT_CUDA_CHECK(cudaFree(buffers_[rank_]));
  if (topoInfo_ != nullptr) {
    AT_CUDA_CHECK(cudaFree(topoInfo_));
  }
  AT_CUDA_CHECK(cudaFree(p2pStatesDev_));
  AT_CUDA_CHECK(cudaFree(buffersDev_));
}

bool IntraNodeComm::isEnabled() {
  return getCvarBool(ENABLE_INTRA_NODE_COMM, false);
}

/**
 * Use c10d::Store to perform allgather on a trivially copyable type.
 */
template <typename T>
std::vector<T> storeAllGather(
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

  auto deviceIdx = at::cuda::current_device();
  c10::cuda::CUDAGuard guard(deviceIdx);

  // First hand shake: exchange hostname and device bus ID
  struct DevInfo {
    char hostname[HOST_NAME_MAX + 1];
    char busId[80];
  };

  DevInfo devInfo{};
  gethostname(devInfo.hostname, sizeof(devInfo.hostname));
  cudaDeviceProp prop{};
  AT_CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceIdx));
  snprintf(
      devInfo.busId,
      sizeof(devInfo.busId),
      NVML_DEVICE_PCI_BUS_ID_FMT,
      prop.pciDomainID,
      prop.pciBusID,
      prop.pciDeviceID);

  auto peerDevInfos =
      storeAllGather(store_, "handshake-0", rank_, worldSize_, devInfo);

  std::vector<std::string> rankToBusId;
  for (const auto& info : peerDevInfos) {
    if (strcmp(info.hostname, peerDevInfos.front().hostname) != 0) {
      LOG(WARNING) << "Aborting IntraNodeComm::rendezvous because some "
                      "participants are not on the same host ("
                   << info.hostname << ", " << devInfo.hostname << ")";
      return false;
    }
    rankToBusId.emplace_back(info.busId);
  }

  // Verify unique devices
  {
    std::unordered_set uniqueBusIds(rankToBusId.begin(), rankToBusId.end());
    TORCH_CHECK(
        uniqueBusIds.size() == worldSize_,
        "IntraNodeComm::rendezvous: detected overlapping devices across ranks. "
        "Please properly set device via torch.cuda.set_device() before "
        "initiating rendezvous.");
  }

  // Query nvlink connection
  auto nvlMesh = getNvlMesh(rankToBusId);

  // Detect topology
  Topology topology = detectTopology(nvlMesh, worldSize_);

  // Initialize p2p state
  auto p2pState = initP2pState();

  // Allocate buffer
  void* buffer = nullptr;
  AT_CUDA_CHECK(cudaMalloc(&buffer, bufferSize_));

  // Second handshake: exchange topology and CUDA IPC handles
  struct IpcInfo {
    NvlMesh nvlMesh;
    Topology topology;
    cudaIpcMemHandle_t p2pStateHandle, bufferHandle;
  };

  // Make p2p state and buffer available for IPC
  cudaIpcMemHandle_t p2pStateHandle, bufferHandle;
  AT_CUDA_CHECK(cudaIpcGetMemHandle(&p2pStateHandle, p2pState));
  AT_CUDA_CHECK(cudaIpcGetMemHandle(&bufferHandle, buffer));

  IpcInfo ipcInfo{
      .nvlMesh = nvlMesh,
      .topology = topology,
      .p2pStateHandle = p2pStateHandle,
      .bufferHandle = bufferHandle};

  auto peerIpcInfos =
      storeAllGather(store_, "handshake-1", rank_, worldSize_, ipcInfo);

  for (const auto& info : peerIpcInfos) {
    if (!isSame(info.nvlMesh, peerIpcInfos.front().nvlMesh) ||
        info.topology != peerIpcInfos.front().topology) {
      LOG(WARNING) << "Aborting IntraNodeComm::rendezvous because some "
                      "participants are observing different topologies ("
                   << int(info.topology) << " and " << int(topology) << ")";
      AT_CUDA_CHECK(cudaFree(p2pState));
      AT_CUDA_CHECK(cudaFree(buffer));
      return false;
    }
  }

  std::array<void*, kMaxDevices> p2pStates = {}, buffers = {};
  for (size_t r = 0; r < peerIpcInfos.size(); ++r) {
    if (r == rank_) {
      p2pStates[r] = p2pState;
      buffers[r] = buffer;
    } else {
      AT_CUDA_CHECK(cudaIpcOpenMemHandle(
          &p2pStates[r],
          peerIpcInfos[r].p2pStateHandle,
          cudaIpcMemLazyEnablePeerAccess));
      AT_CUDA_CHECK(cudaIpcOpenMemHandle(
          &buffers[r],
          peerIpcInfos[r].bufferHandle,
          cudaIpcMemLazyEnablePeerAccess));
    }
  }
  void* p2pStatesDev = nullptr;
  AT_CUDA_CHECK(cudaMalloc(&p2pStatesDev, sizeof(p2pStates)));
  AT_CUDA_CHECK(cudaMemcpy(
      p2pStatesDev,
      p2pStates.data(),
      sizeof(p2pStates),
      cudaMemcpyHostToDevice));

  void* buffersDev = nullptr;
  AT_CUDA_CHECK(cudaMalloc(&buffersDev, sizeof(buffers)));
  AT_CUDA_CHECK(cudaMemcpy(
      buffersDev, buffers.data(), sizeof(buffers), cudaMemcpyHostToDevice));

  void* topoInfo = initTopoInfo(topology, nvlMesh, rank_);

  isInitialized_ = true;
  topology_ = topology;
  std::copy(p2pStates.begin(), p2pStates.end(), p2pStates_.begin());
  std::copy(buffers.begin(), buffers.end(), buffers_.begin());
  p2pStatesDev_ = p2pStatesDev;
  buffersDev_ = buffersDev;
  topoInfo_ = topoInfo;
  return true;
#endif
  return false;
}

} // namespace c10d::intra_node_comm
