#include <torch/csrc/distributed/c10d/intra_node_comm.hpp>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/driver_api.h>
#include <torch/csrc/distributed/c10d/Utils.hpp>

#include <cuda_runtime.h>
#include <nvml.h>

#include <iostream>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace c10d {

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

TwoPhaseGather::TwoPhaseGather(size_t worldSize)
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

TwoPhaseGather::~TwoPhaseGather() {
  pthread_mutex_destroy(&mutex_);
  pthread_cond_destroy(&cond_);
}

// TODO: maybe call it TwoPhaseExchange?
void TwoPhaseGather::run(
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

// TODO: determine via GPU model
static constexpr size_t kMaxNvLinks = 12;

static NvlMesh getNvlMesh() {
  using namespace c10::cuda;

  nvmlDevice_t devices[kMaxDevices];
  std::unordered_map<std::string, size_t> busIdToIdx;
  NvlMesh nvlMesh = {};

  for (size_t idx = 0; idx < kMaxDevices; ++idx) {
    cudaDeviceProp prop{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, idx));
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

  for (size_t idx = 0; idx < kMaxDevices; ++idx) {
    for (size_t link = 0; link < kMaxNvLinks; ++link) {
      nvmlPciInfo_t pciInfo;
      TORCH_CHECK(
          DriverAPI::get()->nvmlDeviceGetNvLinkRemotePciInfo_v2_(
              devices[idx], link, &pciInfo) == NVML_SUCCESS);
      auto it = busIdToIdx.find(pciInfo.busId);
      if (it != busIdToIdx.end()) {
        if (idx != it->second) {
          nvlMesh[idx][it->second] += 1;
        }
      }
    }
  }
  return nvlMesh;
}

static bool isHybridCubeMesh(NvlMesh nvlMesh) {
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

IntraNodeComm::IntraNodeComm(
    std::array<void*, kMaxDevices> buffers,
    std::array<uint32_t*, kMaxDevices> barriers,
    size_t rank,
    size_t worldSize)
    : buffers_(buffers),
      barriers_(barriers),
      rank_(rank),
      worldSize_(worldSize),
      generation_(0),
      nvlMesh_(getNvlMesh()) {
  if (worldSize_ == kMaxDevices) {
    isHybridCubeMesh_ = isHybridCubeMesh(nvlMesh_);
  }
}

IntraNodeComm::~IntraNodeComm() {
  // TODO
}

c10::intrusive_ptr<IntraNodeComm> IntraNodeComm::rendezvous(
    const std::string& rdzvId,
    size_t rank,
    size_t worldSize) {
  // Impose this constraint to avoid confusion
  TORCH_CHECK(rank == static_cast<size_t>(at::cuda::current_device()));

  struct Shared {
    std::array<int, kMaxDevices> devices;
    std::array<cudaIpcMemHandle_t, kMaxDevices> barrierHandles;
    std::array<cudaIpcMemHandle_t, kMaxDevices> bufferHandles;
    TwoPhaseGather twoPhaseGather;

    Shared(size_t worldSize) : twoPhaseGather(worldSize) {}
  };
  SharedMemoryPtr<Shared> peerInfo(rdzvId, rank, worldSize, worldSize);

  void* buffer = nullptr;
  // TODO: explain * 2
  C10_CUDA_CHECK(cudaMalloc(&buffer, kMaxIntraNodeSize * 2));
  cudaIpcMemHandle_t bufferHandle;
  AT_CUDA_CHECK(cudaIpcGetMemHandle(&bufferHandle, buffer));

  uint32_t* barrier = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&barrier, sizeof(uint32_t) * kMaxDevices));
  C10_CUDA_CHECK(cudaMemset(barrier, 0, sizeof(uint32_t) * kMaxDevices));
  cudaIpcMemHandle_t barrierHandle;
  AT_CUDA_CHECK(cudaIpcGetMemHandle(&barrierHandle, barrier));

  std::array<int, kMaxDevices> devices;
  std::array<cudaIpcMemHandle_t, kMaxDevices> bufferHandles;
  std::array<cudaIpcMemHandle_t, kMaxDevices> barrierHandles;
  peerInfo->twoPhaseGather.run(
      [&]() {
        peerInfo->devices[rank] = at::cuda::current_device();
        peerInfo->bufferHandles[rank] = bufferHandle;
        peerInfo->barrierHandles[rank] = barrierHandle;
      },
      [&]() {
        bufferHandles = peerInfo->bufferHandles;
        barrierHandles = peerInfo->barrierHandles;
        devices = peerInfo->devices;
      });

  // Check for device uniqueness
  std::unordered_set<int> uniqueDevices(devices.begin(), devices.end());
  TORCH_CHECK(
      uniqueDevices.size() == worldSize,
      "More than one rank is using the same device. "
      "Make sure every rank is assigned with a unique device. ");

  std::array<void*, kMaxDevices> buffers;
  std::array<uint32_t*, kMaxDevices> barriers;
  for (size_t r = 0; r < worldSize; ++r) {
    if (r == rank) {
      buffers[r] = buffer;
      barriers[r] = barrier;
    } else {
      AT_CUDA_CHECK(cudaIpcOpenMemHandle(
          &buffers[r], bufferHandles[r], cudaIpcMemLazyEnablePeerAccess));

      AT_CUDA_CHECK(cudaIpcOpenMemHandle(
          reinterpret_cast<void**>(&barriers[r]),
          barrierHandles[r],
          cudaIpcMemLazyEnablePeerAccess));
    }
  }
  return c10::make_intrusive<IntraNodeComm>(buffers, barriers, rank, worldSize);
}

at::Tensor oneShotAllReduce(
    const at::Tensor& input,
    std::array<void*, kMaxDevices> buffers,
    std::array<uint32_t*, kMaxDevices> barriers,
    uint32_t barrierFlag,
    size_t rank,
    size_t worldSize);

at::Tensor hybridCubeOneShotAllReduce(
    const at::Tensor& input,
    std::array<void*, kMaxDevices> buffers,
    std::array<uint32_t*, kMaxDevices> barriers,
    uint32_t barrierFlag,
    size_t rank,
    size_t worldSize,
    NvlMesh nvlMesh_);

at::Tensor IntraNodeComm::allReduce(const at::Tensor& input) {
  TORCH_CHECK(static_cast<size_t>(input.numel()) < kMaxIntraNodeSize);
  if (isHybridCubeMesh_ && parseEnvVarFlag("ENABLE_HCM_ALLREDUCE")) {
    // The hybrid cube mesh algorithm uses the barrier twice
    generation_ = (generation_ + 2) % 93187;
    return hybridCubeOneShotAllReduce(
        input, buffers_, barriers_, generation_, rank_, worldSize_, nvlMesh_);
  }
  generation_ = (generation_ + 1) % 93187;
  return oneShotAllReduce(
      input, buffers_, barriers_, generation_, rank_, worldSize_);
}

} // namespace c10d
