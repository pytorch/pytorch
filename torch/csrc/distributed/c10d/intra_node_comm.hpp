#pragma once

#include <pthread.h>
#include <semaphore.h>

#include <ATen/ATen.h>

namespace c10d {

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

class TwoPhaseSync {
 public:
  TwoPhaseSync(size_t worldSize);
  ~TwoPhaseSync();

  void run(std::function<void()> writeFn, std::function<void()> gatherFn);

 private:
  size_t worldSize_;
  size_t barrierCnt_;
  pthread_mutex_t mutex_;
  pthread_cond_t cond_;
};

static constexpr size_t kMaxDevices = 8;
static constexpr size_t kMaxIntraNodeSize = 50 * 1024 * 1024;

class TORCH_API IntraNodeComm : public c10::intrusive_ptr_target {
 public:
  IntraNodeComm(
      std::array<void*, kMaxDevices> buffers,
      std::array<uint32_t*, kMaxDevices> barriers,
      size_t rank,
      size_t worldSize);

  ~IntraNodeComm();

  static c10::intrusive_ptr<IntraNodeComm> rendezvous(
      const std::string& rdzvId,
      size_t rank,
      size_t worldSize);

  at::Tensor allReduce(const at::Tensor& input);

 private:
  std::array<void*, kMaxDevices> buffers_;
  std::array<uint32_t*, kMaxDevices> barriers_;
  size_t rank_;
  size_t worldSize_;
  uint32_t generation_;
};

} // namespace c10d
