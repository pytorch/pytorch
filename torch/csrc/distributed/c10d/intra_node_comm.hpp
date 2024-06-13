#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

namespace c10d::intra_node_comm {

constexpr size_t kMaxDevices = 8;
constexpr size_t kDefaultBufferSize = 10ull * 1024 * 1024;

using NvlMesh = std::array<std::array<size_t, kMaxDevices>, kMaxDevices>;
using HybridCubeMesh = std::array<std::array<int, 4>, kMaxDevices>;

enum class Topology : uint8_t {
  UNKNOWN = 0,
  FULLY_CONNECTED = 1,
  HYBRID_CUBE_MESH = 2
};

enum class AllReduceAlgo : uint8_t {
  NONE = 0,
  ONE_SHOT = 1,
  TWO_SHOT = 2,
  HCM = 3
};

class TORCH_API IntraNodeComm : public c10::intrusive_ptr_target {
 public:
  IntraNodeComm(
      c10::intrusive_ptr<c10d::Store> store,
      size_t rank,
      size_t worldSize,
      std::optional<size_t> bufferSize = c10::nullopt);

  ~IntraNodeComm() override;

  static bool isEnabled();

  /**
   * Performs rendezvous.
   * If rendezvous fails, the IntraNodeComm object will be in an invalid
   * state and it is the caller's responsibility to dispose it.
   */
  bool rendezvous();

  Topology getTopology() {
    return topology_;
  }

  size_t getBufferSize() {
    return bufferSize_;
  }

  /**
   * Selects a AllReduceAlgo that we think will outperform nccl.
   * Returns AllReduceAlgo::NONE if we don't think we can outperform nccl.
   */
  AllReduceAlgo selectAllReduceAlgo(const at::Tensor& input);

  at::Tensor allReduce(const at::Tensor& input, AllReduceAlgo algo);

  /**
   * Perform a barrier among the specified ranks.
   */
  void barrier(std::optional<std::vector<int64_t>> ranks = c10::nullopt);

  at::Tensor getBuffer(
      size_t rank,
      const std::vector<int64_t>& sizes,
      c10::ScalarType dtype,
      int64_t storageOffset);

 private:
  at::Tensor oneShotAllReduce(
      const at::Tensor& input,
      at::cuda::CUDAStream& stream);

  at::Tensor twoShotAllReduce(
      const at::Tensor& input,
      at::cuda::CUDAStream& stream);

  at::Tensor hybridCubeMeshAllReduce(
      const at::Tensor& input,
      at::cuda::CUDAStream& stream);

  c10::intrusive_ptr<Store> store_;
  size_t rank_;
  size_t worldSize_;
  size_t bufferSize_;
  at::cuda::CUDAEvent barrierReady_;

  /**
   * Members initialized after rendezvous
   */
  bool isInitialized_ = false;
  Topology topology_ = Topology::UNKNOWN;
  std::array<void*, kMaxDevices> p2pStates_{};
  std::array<void*, kMaxDevices> buffers_{};
  void* p2pStatesDev_{};
  void* buffersDev_{};
  void* topoInfo_{};
};

/**
 * NOTE [IntraNodeComm Stream Semantics]
 *
 * ProcessGroupNCCL launches kernels differently from the conventional PyTorch
 * CUDA semantics: it always launches collective kernels onto a dedicated
 * communication stream. Therefore, it needs to:
 *
 * - Synchronize the calling stream and the comm stream.
 * - Ensure the memory safety of the operands (via record_stream or stashing).
 * - Synchronize the waiting stream with the comm stream.
 *
 * Unconditionally performing these tasks makes sense when we expect most of the
 * communication to benefit from compute/comm overlap. However, IntraNodeComm
 * primarily aims to optimize small, latency-sensitive, blocking communication,
 * in which the overhead incurred by the above steps can be quite pronounced.
 *
 * Thus, IntraNodeComm follows the conventional PyTorch CUDA semantics and
 * launches kernels onto the stream specified by the user. Although the user
 * can perform neccessary synchronization via wait_stream, to provide a UX
 * consistent to that of ProcessGroupNCCL, the neccessary stream
 * synchronization can also be performed via IntraNodeWork::wait().
 */
class IntraNodeCommWork : public c10d::Work {
 public:
  IntraNodeCommWork() : c10d::Work() {
    event_.record();
  }

  bool wait(std::chrono::milliseconds timeout = kNoTimeout) override {
    event_.block(at::cuda::getCurrentCUDAStream());
    return true;
  }

 private:
  at::cuda::CUDAEvent event_;
};

TORCH_API int64_t getIntraNodeCommUsageCounter();

} // namespace c10d::intra_node_comm
