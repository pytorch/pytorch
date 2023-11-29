#include <torch/csrc/distributed/c10d/intra_node_comm.hpp>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/driver_api.h>
#include <c10/util/Logging.h>
#include <torch/csrc/distributed/c10d/Utils.hpp>

#include <cuda_runtime.h>
#include <nvml.h>

#include <iostream>
#include <random>

#include <fcntl.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace c10d {

////////////////////////////////////////////////////////////////////////////////
// Rendezvous Utilities
////////////////////////////////////////////////////////////////////////////////

class SharedMemoryPtrBase {
 public:
  SharedMemoryPtrBase(
      const std::string& rdzvId,
      size_t rank,
      size_t worldSize,
      size_t allocSize,
      std::function<void(void*)> initializer,
      std::function<void(void*)> destructor);

  ~SharedMemoryPtrBase();

 private:
  std::string shmName_;
  std::string initSemName_;
  std::string tearDownSemName_;
  size_t rank_;
  size_t worldSize_;
  size_t allocSize_;
  std::function<void(void*)> destructor_;

  sem_t* initSem_;
  sem_t* tearDownSem_;
  int shmFd_;

 protected:
  void* shared_;
};

template <typename T>
class SharedMemoryPtr : public SharedMemoryPtrBase {
 public:
  template <typename... ConstructorArgs>
  SharedMemoryPtr(
      const std::string& rdzvId,
      size_t rank,
      size_t worldSize,
      ConstructorArgs... args)
      : SharedMemoryPtrBase(
            rdzvId,
            rank,
            worldSize,
            sizeof(T),
            [args...](void* ptr) { new (ptr) T(args...); },
            [](void* ptr) { static_cast<T*>(ptr)->~T(); }) {}

  SharedMemoryPtr(const SharedMemoryPtr&) = delete;
  SharedMemoryPtr& operator=(const SharedMemoryPtr&) = delete;

  T* operator->() const {
    return static_cast<T*>(shared_);
  }

  T& operator*() const {
    return *static_cast<T*>(shared_);
  }
};

class TwoPhaseExchange {
 public:
  TwoPhaseExchange(size_t worldSize);
  ~TwoPhaseExchange();

  void run(std::function<void()> writeFn, std::function<void()> gatherFn);

 private:
  size_t worldSize_;
  size_t barrierCnt_;
  pthread_mutex_t mutex_;
  pthread_cond_t cond_;
};

SharedMemoryPtrBase::SharedMemoryPtrBase(
    const std::string& rdzvId,
    size_t rank,
    size_t worldSize,
    size_t allocSize,
    std::function<void(void*)> initializer,
    std::function<void(void*)> destructor)
    : shmName_(rdzvId + "-shm"),
      initSemName_(rdzvId + "-initSem"),
      tearDownSemName_(rdzvId + "-tearDownSem"),
      rank_(rank),
      worldSize_(worldSize),
      allocSize_(allocSize),
      destructor_(destructor) {
  initSem_ = sem_open(initSemName_.c_str(), O_CREAT, S_IRUSR | S_IWUSR, 0);
  TORCH_CHECK(initSem_ != SEM_FAILED, "Failed to open semaphore");

  tearDownSem_ =
      sem_open(tearDownSemName_.c_str(), O_CREAT, S_IRUSR | S_IWUSR, 0);
  TORCH_CHECK(tearDownSem_ != SEM_FAILED, "Failed to open semaphore");

  shmFd_ = shm_open(shmName_.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  TORCH_CHECK(shmFd_ != -1, "Failed to open shared memory");

  TORCH_CHECK(
      ftruncate(shmFd_, allocSize) != -1, "Failed to truncate shared memory");

  shared_ =
      mmap(NULL, allocSize, PROT_READ | PROT_WRITE, MAP_SHARED, shmFd_, 0);
  TORCH_CHECK(shared_ != MAP_FAILED, "Failed to map shared memory");

  if (rank_ == 0) {
    initializer(shared_);
    for (size_t i = 1; i < worldSize_; ++i) {
      sem_post(initSem_);
    }
  } else {
    sem_wait(initSem_);
  }
}

SharedMemoryPtrBase::~SharedMemoryPtrBase() {
  if (rank_ != 0) {
    sem_post(tearDownSem_);
  } else {
    for (size_t i = 1; i < worldSize_; ++i) {
      sem_wait(tearDownSem_);
    }
    destructor_(shared_);
  }
  munmap(shared_, allocSize_);
  close(shmFd_);
  sem_close(tearDownSem_);
  sem_close(initSem_);

  if (rank_ == 0) {
    shm_unlink(shmName_.c_str());
    shm_unlink(tearDownSemName_.c_str());
    sem_unlink(initSemName_.c_str());
  }
}

TwoPhaseExchange::TwoPhaseExchange(size_t worldSize)
    : worldSize_(worldSize), barrierCnt_(0) {
  pthread_mutexattr_t mutexAttr;
  pthread_condattr_t condAttr;

  TORCH_CHECK(
      pthread_mutexattr_init(&mutexAttr) == 0,
      "Failed to initialize mutex attributes");

  TORCH_CHECK(
      pthread_mutexattr_setpshared(&mutexAttr, PTHREAD_PROCESS_SHARED) == 0,
      "Failed to set mutex as shared");

  TORCH_CHECK(
      pthread_mutex_init(&mutex_, &mutexAttr) == 0,
      "Failed to initialize mutex");

  TORCH_CHECK(
      pthread_condattr_init(&condAttr) == 0,
      "Failed to initialize cond attributes");

  TORCH_CHECK(
      pthread_condattr_setpshared(&condAttr, PTHREAD_PROCESS_SHARED) == 0,
      "Failed to set cond as shared");

  TORCH_CHECK(
      pthread_cond_init(&cond_, &condAttr) == 0, "Failed to initialize cond");

  pthread_mutexattr_destroy(&mutexAttr);
  pthread_condattr_destroy(&condAttr);
}

TwoPhaseExchange::~TwoPhaseExchange() {
  pthread_mutex_destroy(&mutex_);
  pthread_cond_destroy(&cond_);
}

void TwoPhaseExchange::run(
    std::function<void()> writeFn,
    std::function<void()> gatherFn) {
  // Phase 1: write
  pthread_mutex_lock(&mutex_);
  writeFn();
  barrierCnt_ += 1;
  if (barrierCnt_ != worldSize_) {
    pthread_cond_wait(&cond_, &mutex_);
  } else {
    pthread_cond_broadcast(&cond_);
  }
  pthread_mutex_unlock(&mutex_);

  // Phase 2: gather
  pthread_mutex_lock(&mutex_);
  gatherFn();
  barrierCnt_ -= 1;
  if (barrierCnt_ != 0) {
    pthread_cond_wait(&cond_, &mutex_);
  } else {
    pthread_cond_broadcast(&cond_);
  }
  pthread_mutex_unlock(&mutex_);
}

////////////////////////////////////////////////////////////////////////////////
// Topology Detection
////////////////////////////////////////////////////////////////////////////////

// TODO: determine via GPU model
static constexpr size_t kMaxNvLinks = 20;

static std::ostream& operator<<(std::ostream& os, const NvlMesh& nvlMesh) {
  std::ostringstream oss;
  for (size_t i = 0; i < kMaxDevices; ++i) {
    for (size_t j = 0; j < kMaxDevices; ++j) {
      oss << nvlMesh[i][j] << " ";
    }
    oss << std::endl;
  }
  os << oss.str();
  return os;
}

/**
 * Query the nvlink connection among devices.
 */
static NvlMesh getNvlMesh(size_t worldSize) {
  using namespace c10::cuda;

  nvmlDevice_t devices[worldSize];
  std::unordered_map<std::string, size_t> busIdToIdx;
  size_t switchLinkCount[worldSize] = {};
  NvlMesh nvlMesh = {};

  for (size_t idx = 0; idx < worldSize; ++idx) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, idx) == cudaErrorInvalidDevice) {
      continue;
    }
    char busId[80];
    snprintf(
        busId,
        sizeof(busId),
        NVML_DEVICE_PCI_BUS_ID_FMT,
        prop.pciDomainID,
        prop.pciBusID,
        prop.pciDeviceID);
    TORCH_CHECK(
        DriverAPI::get()->nvmlDeviceGetHandleByPciBusId_v2_(
            busId, &devices[idx]) == NVML_SUCCESS);
    busIdToIdx.emplace(std::make_pair(busId, idx));
  }

  for (size_t idx = 0; idx < worldSize; ++idx) {
    for (size_t link = 0; link < kMaxNvLinks; ++link) {
      nvmlReturn_t ret;
      nvmlIntNvLinkDeviceType_t deviceType;
      ret = DriverAPI::get()->nvmlDeviceGetNvLinkRemoteDeviceType_(
          devices[idx], link, &deviceType);
      if (ret != NVML_SUCCESS) {
        break;
      }
      // Remote device is GPU
      if (deviceType == NVML_NVLINK_DEVICE_TYPE_GPU) {
        nvmlPciInfo_t pciInfo;
        ret = DriverAPI::get()->nvmlDeviceGetNvLinkRemotePciInfo_v2_(
            devices[idx], link, &pciInfo);
        if (ret != NVML_SUCCESS) {
          break;
        }
        auto it = busIdToIdx.find(pciInfo.busId);
        if (it != busIdToIdx.end()) {
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
  // Process NVSwitch connections
  for (size_t i = 0; i < worldSize; ++i) {
    for (size_t j = 0; j < worldSize; ++j) {
      if (i == j) {
        continue;
      }
      nvlMesh[i][j] = std::min(switchLinkCount[i], switchLinkCount[j]);
    }
  }
  return nvlMesh;
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
    if (numNeighbors[i] != 4) {
      return false;
    }
  }
  return true;
}

/**
 * Detech topology given a NvlMesh.
 */
static Topology detectTopology(
    const NvlMesh nvlMesh,
    std::vector<int> deviceIds) {
  TORCH_CHECK(deviceIds.size() >= 2);
  bool fullyConnected = true;
  for (size_t i = 0; i < deviceIds.size() - 1; ++i) {
    for (size_t j = i + 1; j < deviceIds.size(); ++j) {
      if (nvlMesh[deviceIds[i]][deviceIds[j]] == 0) {
        fullyConnected = false;
      }
    }
  }
  if (fullyConnected) {
    std::cout << "IntraNodeComm: Topology::FULLY_CONNECTED" << std::endl;
    LOG(INFO) << "IntraNodeComm: Topology::FULLY_CONNECTED";
    return Topology::FULLY_CONNECTED;
  }
  if (deviceIds.size() == kMaxDevices && isHybridCubeMesh(nvlMesh)) {
    std::cout << "IntraNodeComm: Topology::HYBRID_CUBE_MESH" << std::endl;
    LOG(INFO) << "IntraNodeComm: Topology::HYBRID_CUBE_MESH";
    return Topology::HYBRID_CUBE_MESH;
  }
  std::cout << "IntraNodeComm: Topology::UNKNOWN" << std::endl;
  return Topology::UNKNOWN;
};

////////////////////////////////////////////////////////////////////////////////
// Rendezvous and Initialization
////////////////////////////////////////////////////////////////////////////////

IntraNodeComm::IntraNodeComm(
    Topology topology,
    std::array<void*, kMaxDevices> p2pStates,
    std::array<void*, kMaxDevices> buffers,
    size_t rank,
    size_t worldSize)
    : topology_(topology),
      p2pStates_(p2pStates),
      buffers_(buffers),
      rank_(rank),
      worldSize_(worldSize) {}

IntraNodeComm::~IntraNodeComm() {
  // TODO
}

void* initFcP2pState();
void* initHcmP2pState(NvlMesh nvlMesh, size_t rank);

void* initP2pState(Topology topology, NvlMesh nvlMesh, size_t rank) {
  if (topology == Topology::FULLY_CONNECTED) {
    return initFcP2pState();
  } else if (topology == Topology::HYBRID_CUBE_MESH) {
    return initHcmP2pState(nvlMesh, rank);
  } else {
    LOG(FATAL) << "Invalid topology";
    return nullptr;
  }
}

/**
 * Rendezvous via shared memory given a rendezvous ID.
 *
 * Use this if we know all participants are from the same host.
 */
c10::intrusive_ptr<IntraNodeComm> IntraNodeComm::rendezvous(
    const std::string& rdzvId,
    size_t rank,
    size_t worldSize) {
  if (!parseEnvVarFlag(ENABLE_INTRA_NODE_COMM) || worldSize == 1 ||
      worldSize > kMaxDevices) {
    return nullptr;
  }

  // Data structure for rendezvous
  struct Shared {
    std::array<int, kMaxDevices> deviceIds;
    std::array<cudaIpcMemHandle_t, kMaxDevices> p2pStateHandles;
    std::array<cudaIpcMemHandle_t, kMaxDevices> bufferHandles;
    TwoPhaseExchange twoPhaseExchange;

    Shared(size_t worldSize) : twoPhaseExchange(worldSize) {
      deviceIds.fill(-1);
    }
  };

  // Initialize shared memory for rendezvous
  SharedMemoryPtr<Shared> peerInfo(rdzvId, rank, worldSize, worldSize);

  // Gether peer device Ids
  std::vector<int> deviceIds;
  peerInfo->twoPhaseExchange.run(
      [&]() { peerInfo->deviceIds[rank] = at::cuda::current_device(); },
      [&]() {
        for (const auto& deviceId : peerInfo->deviceIds) {
          if (deviceId != -1) {
            deviceIds.push_back(deviceId);
          }
        }
      });
  TORCH_CHECK(
      deviceIds.size() == worldSize,
      "More than one rank is using the same device. "
      "Make sure every rank is assigned with a unique device. ");

  // Query nvlink connection
  auto nvlMesh = getNvlMesh(worldSize);

  // Detect topology
  Topology topology = detectTopology(nvlMesh, deviceIds);
  if (topology == Topology::UNKNOWN) {
    return nullptr;
  }

  // Initialize p2p state
  auto p2pState = initP2pState(topology, nvlMesh, rank);

  // Allocate buffer
  void* buffer = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&buffer, kMaxIntraNodeSize * 2));

  // Make p2p state and buffer available for IPC
  cudaIpcMemHandle_t p2pStateHandle, bufferHandle;
  AT_CUDA_CHECK(cudaIpcGetMemHandle(&p2pStateHandle, p2pState));
  AT_CUDA_CHECK(cudaIpcGetMemHandle(&bufferHandle, buffer));

  // Exchange IPC handles for p2p state and buffer
  std::array<cudaIpcMemHandle_t, kMaxDevices> p2pStateHandles, bufferHandles;
  peerInfo->twoPhaseExchange.run(
      [&]() {
        peerInfo->p2pStateHandles[rank] = p2pStateHandle;
        peerInfo->bufferHandles[rank] = bufferHandle;
      },
      [&]() {
        p2pStateHandles = peerInfo->p2pStateHandles;
        bufferHandles = peerInfo->bufferHandles;
      });

  // Map peer p2p states and buffers to the process address space
  std::array<void*, kMaxDevices> p2pStates, buffers;
  for (size_t r = 0; r < worldSize; ++r) {
    if (r == rank) {
      p2pStates[r] = p2pState;
      buffers[r] = buffer;
    } else {
      AT_CUDA_CHECK(cudaIpcOpenMemHandle(
          &p2pStates[r], p2pStateHandles[r], cudaIpcMemLazyEnablePeerAccess));
      AT_CUDA_CHECK(cudaIpcOpenMemHandle(
          &buffers[r], bufferHandles[r], cudaIpcMemLazyEnablePeerAccess));
    }
  }
  return c10::make_intrusive<IntraNodeComm>(
      topology, p2pStates, buffers, rank, worldSize);
}

std::string generateUUID() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 15);
  std::uniform_int_distribution<> dis2(8, 11);

  std::stringstream ss;
  int i;

  ss << std::hex;
  for (i = 0; i < 8; i++) {
    ss << dis(gen);
  }
  ss << "-";
  for (i = 0; i < 4; i++) {
    ss << dis(gen);
  }
  ss << "-4"; // Version 4 : Random
  for (i = 0; i < 3; i++) {
    ss << dis(gen);
  }
  ss << "-";
  ss << dis2(gen);

  for (i = 0; i < 3; i++) {
    ss << dis(gen);
  }
  ss << "-";
  for (i = 0; i < 12; i++) {
    ss << dis(gen);
  }

  return ss.str();
}

/**
 * Rendezvous via c10::Store.
 *
 * Use this if we don't know if all participants are from the same host. This
 * function returns nullptr for all participants if not all of them are from
 * the same host.
 */
c10::intrusive_ptr<IntraNodeComm> IntraNodeComm::rendezvousViaStore(
    c10::intrusive_ptr<c10d::Store> store,
    const std::string& prefix,
    size_t rank,
    size_t worldSize) {
  if (!parseEnvVarFlag(ENABLE_INTRA_NODE_COMM) || worldSize == 1 ||
      worldSize > kMaxDevices) {
    return nullptr;
  }
  std::array<char, HOST_NAME_MAX + 1> buf = {};
  gethostname(buf.data(), buf.size());
  std::string hostname(buf.data());

  std::vector<std::string> hostnameKeys;
  for (size_t r = 0; r < worldSize; ++r) {
    std::ostringstream oss;
    oss << prefix << "-" << rank << "-hostname";
    hostnameKeys.push_back(oss.str());
  }
  store->set(hostnameKeys[rank], hostname);

  std::vector<std::string> hostnames;
  for (size_t r = 0; r < worldSize; ++r) {
    if (r == rank) {
      hostnames.push_back(hostname);
      continue;
    }
    store->wait({hostnameKeys[r]});
    hostnames.push_back(store->get_to_str(hostnameKeys[r]));
  }

  for (const auto& hn : hostnames) {
    if (hn != hostname) {
      return nullptr;
    }
  }

  std::string uuid;
  std::string rdzvIdKey = prefix + "-IntraHostCommRdzvId";
  if (rank == 0) {
    uuid = generateUUID();
    store->set(rdzvIdKey, uuid);
  } else {
    store->wait({rdzvIdKey});
    uuid = store->get_to_str(rdzvIdKey);
  }
  return rendezvous(uuid, rank, worldSize);
}

AllReduceAlgo selectAllReduceAlgo(
    const at::Tensor& input,
    Topology topology,
    size_t worldSize);

bool IntraNodeComm::shouldUseIntraNodeAllReduce(const at::Tensor& input) {
  auto algo = selectAllReduceAlgo(input, topology_, worldSize_);
  return algo != AllReduceAlgo::NONE;
}

at::Tensor oneShotAllReduce(
    const at::Tensor& input,
    std::array<void*, kMaxDevices> buffers,
    std::array<void*, kMaxDevices> p2pStates,
    size_t rank,
    size_t worldSize);

at::Tensor twoShotAllReduce(
    const at::Tensor& input,
    std::array<void*, kMaxDevices> p2pStates,
    std::array<void*, kMaxDevices> buffers,
    size_t rank,
    size_t worldSize);

at::Tensor hybridCubeMeshAllReduce(
    const at::Tensor& input,
    std::array<void*, kMaxDevices> p2pStates,
    std::array<void*, kMaxDevices> buffers,
    size_t rank,
    size_t worldSize);

at::Tensor IntraNodeComm::allReduce(const at::Tensor& input) {
  auto algo = selectAllReduceAlgo(input, topology_, worldSize_);
  switch (algo) {
    case AllReduceAlgo::ONE_SHOT:
      return oneShotAllReduce(input, p2pStates_, buffers_, rank_, worldSize_);
    case AllReduceAlgo::TWO_SHOT:
      return twoShotAllReduce(input, p2pStates_, buffers_, rank_, worldSize_);
    case AllReduceAlgo::HCM:
      return hybridCubeMeshAllReduce(
          input, p2pStates_, buffers_, rank_, worldSize_);
    default:
      LOG(FATAL) << "FOOBAR";
      return input;
  }
}

} // namespace c10d
