#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

namespace c10d {
namespace intra_node_comm {

constexpr size_t kMaxDevices = 8;
constexpr size_t kDefaultBufferSize = 10 * 1024 * 1024;

using NvlMesh = std::array<std::array<size_t, kMaxDevices>, kMaxDevices>;
using HybridCubeMesh = std::array<std::array<int, 4>, kMaxDevices>;

enum class Topology { UNKNOWN = 0, FULLY_CONNECTED = 1, HYBRID_CUBE_MESH = 2 };

enum class AllReduceAlgo { NONE = 0, ONE_SHOT = 1, TWO_SHOT = 2, HCM = 3 };

class TORCH_API IntraNodeComm : public c10::intrusive_ptr_target {
 public:
  IntraNodeComm(
      Topology topology,
      std::array<void*, kMaxDevices> p2pStates,
      std::array<void*, kMaxDevices> buffers,
      void* p2pStatesDev,
      void* buffersDev,
      void* topoInfo,
      size_t rank,
      size_t worldSize,
      size_t bufferSize = kDefaultBufferSize);

  ~IntraNodeComm();

  /**
   * Rendezvous via a c10d::Store.
   * This function may return nullptr if intra-node comm is not applicable.
   * It guarantees all participants either succeeds or abort.
   */
  static c10::intrusive_ptr<IntraNodeComm> rendezvous(
      c10::intrusive_ptr<c10d::Store> store,
      const std::string& prefix,
      size_t rank,
      size_t worldSize,
      size_t bufferSize = kDefaultBufferSize);

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

  at::Tensor hybridCubeMeshAllReduce(
      const at::Tensor& input,
      at::cuda::CUDAStream& stream);

  Topology topology_;
  std::array<void*, kMaxDevices> p2pStates_;
  std::array<void*, kMaxDevices> buffers_;
  void* p2pStatesDev_;
  void* buffersDev_;
  void* topoInfo_;
  size_t rank_;
  size_t worldSize_;
  size_t bufferSize_;
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

} // namespace intra_node_comm
} // namespace c10d
