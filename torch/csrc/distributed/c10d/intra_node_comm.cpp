#include <torch/csrc/distributed/c10d/intra_node_comm.hpp>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

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

TwoPhaseSync::TwoPhaseSync(size_t worldSize)
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

TwoPhaseSync::~TwoPhaseSync() {
  pthread_mutex_destroy(&mutex_);
  pthread_cond_destroy(&cond_);
}

// TODO: maybe call it TwoPhaseExchange?
void TwoPhaseSync::run(
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

IntraNodeComm::IntraNodeComm(
    std::array<void*, kMaxDevices> buffers,
    std::array<uint32_t*, kMaxDevices> barriers,
    size_t rank,
    size_t worldSize)
    : buffers_(buffers),
      barriers_(barriers),
      rank_(rank),
      worldSize_(worldSize),
      generation_(0) {}

IntraNodeComm::~IntraNodeComm() {
  // TODO
}

c10::intrusive_ptr<IntraNodeComm> IntraNodeComm::rendezvous(
    const std::string& rdzvId,
    size_t rank,
    size_t worldSize) {
  struct Shared {
    std::array<int, kMaxDevices> devices;
    std::array<cudaIpcMemHandle_t, kMaxDevices> barrierHandles;
    std::array<cudaIpcMemHandle_t, kMaxDevices> bufferHandles;
    TwoPhaseSync twoPhaseSync;

    Shared(size_t worldSize) : twoPhaseSync(worldSize) {}
  };
  SharedMemoryPtr<Shared> peerInfo(rdzvId, rank, worldSize, worldSize);

  void* buffer = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&buffer, kMaxIntraNodeSize));
  cudaIpcMemHandle_t bufferHandle;
  AT_CUDA_CHECK(cudaIpcGetMemHandle(&bufferHandle, buffer));

  uint32_t* barrier = nullptr;
  // TODO: correct size
  // TODO: memset
  C10_CUDA_CHECK(cudaMalloc(&barrier, kMaxIntraNodeSize));
  cudaIpcMemHandle_t barrierHandle;
  AT_CUDA_CHECK(cudaIpcGetMemHandle(&barrierHandle, barrier));

  std::array<int, kMaxDevices> devices;
  std::array<cudaIpcMemHandle_t, kMaxDevices> bufferHandles;
  std::array<cudaIpcMemHandle_t, kMaxDevices> barrierHandles;
  peerInfo->twoPhaseSync.run(
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

at::Tensor IntraNodeComm::allReduce(const at::Tensor& input) {
  // TODO: check size
  // TODO: two-shot
  // TODO: memset barrier
  generation_ += 1;
  return oneShotAllReduce(
      input, buffers_, barriers_, generation_, rank_, worldSize_);
}

} // namespace c10d
