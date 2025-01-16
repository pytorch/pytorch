#pragma once

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/SymmetricMemory.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

namespace c10d::intra_node_comm {

using namespace c10d::symmetric_memory;

constexpr size_t kMaxDevices = 8;
constexpr size_t kDefaultBufferSize = 10ull * 1024 * 1024;

using NvlMesh = std::array<std::array<size_t, kMaxDevices>, kMaxDevices>;

enum class Topology : uint8_t {
  UNKNOWN = 0,
  FULLY_CONNECTED = 1,
};

enum class AllReduceAlgo : uint8_t {
  NONE = 0,
  ONE_SHOT = 1,
  TWO_SHOT = 2,
};

// NOTE: this class will be be removed soon in favor of SymmetricMemory
class TORCH_API IntraNodeComm : public c10::intrusive_ptr_target {
 public:
  IntraNodeComm(
      c10::intrusive_ptr<c10d::Store> store,
      size_t rank,
      size_t worldSize,
      std::optional<size_t> bufferSize = std::nullopt);

  ~IntraNodeComm() override;

  static bool isEnabled();

  /**
   * Performs rendezvous.
   * If rendezvous fails, the IntraNodeComm object will be in an invalid
   * state and it is the caller's responsibility to dispose it.
   */
  bool rendezvous();

  /**
   * Selects a AllReduceAlgo that we think will outperform nccl.
   * Returns AllReduceAlgo::NONE if we don't think we can outperform nccl.
   */
  AllReduceAlgo selectAllReduceAlgo(const at::Tensor& input);

  at::Tensor allReduce(const at::Tensor& input, AllReduceAlgo algo);

 private:
  at::Tensor oneShotAllReduce(
      const at::Tensor& input,
      at::cuda::CUDAStream& stream);

  at::Tensor twoShotAllReduce(
      const at::Tensor& input,
      at::cuda::CUDAStream& stream);

  c10::intrusive_ptr<Store> store_;
  size_t rank_;
  size_t worldSize_;
  size_t bufferSize_;

  /**
   * Members initialized after rendezvous
   */
  bool isInitialized_ = false;
  int deviceIdx_{0};
  Topology topology_ = Topology::UNKNOWN;
  void* symmetricMemoryPtr_ = nullptr;
  c10::intrusive_ptr<SymmetricMemory> symmetricMemory_ = nullptr;
};

class IntraNodeCommWork : public c10d::Work {
 public:
  bool wait(std::chrono::milliseconds timeout = kNoTimeout) override {
    return true;
  }
};

TORCH_API int64_t getIntraNodeCommUsageCounter();

} // namespace c10d::intra_node_comm
