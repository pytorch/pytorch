#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/distributed/c10d/Store.hpp>

namespace c10d {
namespace intra_node_comm {

constexpr size_t kMaxDevices = 8;
constexpr size_t kMaxIntraNodeSize = 50 * 1024 * 1024;

using NvlMesh = std::array<std::array<size_t, kMaxDevices>, kMaxDevices>;

enum class Topology { UNKNOWN = 0, FULLY_CONNECTED = 1, HYBRID_CUBE_MESH = 2 };

enum class AllReduceAlgo { NONE = 0, ONE_SHOT = 1, TWO_SHOT = 2, HCM = 3 };

class TORCH_API IntraNodeComm : public c10::intrusive_ptr_target {
 public:
  IntraNodeComm(
      Topology topology,
      std::array<void*, kMaxDevices> p2pStates,
      std::array<void*, kMaxDevices> buffers,
      size_t rank,
      size_t worldSize);

  ~IntraNodeComm();

  /**
   * Rendezvous via shared memory given a rendezvous ID.
   * Use this if we know all participants are from the same host.
   */
  static c10::intrusive_ptr<IntraNodeComm> rendezvous(
      const std::string& rdzvId,
      size_t rank,
      size_t worldSize);

  /**
   * Rendezvous via c10::Store. This variant properly handles the case where
   * not all participants are from the same host, in which case nullptr is
   * returned to all participants.
   */
  static c10::intrusive_ptr<IntraNodeComm> rendezvousViaStore(
      c10::intrusive_ptr<c10d::Store> store,
      const std::string& prefix,
      size_t rank,
      size_t worldSize);

  /**
   * Selects a AllReduceAlgo that we think will outperform nccl.
   * Returns AllReduceAlgo::NONE if we don't think we can outperform nccl.
   */
  AllReduceAlgo selectAllReduceAlgo(const at::Tensor& input);

  at::Tensor allReduce(const at::Tensor& input, AllReduceAlgo algo);

 private:
  Topology topology_;
  std::array<void*, kMaxDevices> p2pStates_;
  std::array<void*, kMaxDevices> buffers_;
  size_t rank_;
  size_t worldSize_;
};

} // namespace intra_node_comm
} // namespace c10d
