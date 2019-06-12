#pragma once

#include "../DataChannel.hpp"
#include "DataChannelUtils.hpp"

#include <nccl.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define NCCL_CHECK(cmd)                                                   \
  do {                                                                    \
    ncclResult_t error = cmd;                                             \
    if (error != ncclSuccess) {                                           \
      std::string err = "NCCL error in: " + std::string(__FILE__) + ":" + \
          std::to_string(__LINE__) + ", " +                               \
          std::string(ncclGetErrorString(error));                         \
      throw std::runtime_error(err);                                      \
    }                                                                     \
  } while (0)

namespace thd {

// Type aliasing
using NcclResourcePair =
    std::pair<std::vector<ncclComm_t>*, std::vector<cudaEvent_t>*>;

struct DataChannelNccl : DataChannel {
  // Nothing to implement
  struct RequestNccl : DataChannel::Request {};

  // Wrapper on the pair of NCCL resources
  class NcclResources {
   public:
    NcclResources() = default;
    NcclResources(
        std::unique_ptr<std::vector<ncclComm_t>>&& ncclComm,
        std::unique_ptr<std::vector<cudaEvent_t>>&& event)
        :

          _commEventPair(std::pair<
                         std::unique_ptr<std::vector<ncclComm_t>>,
                         std::unique_ptr<std::vector<cudaEvent_t>>>(
              std::move(ncclComm),
              std::move(event))) {}
    // Delete copy and assignment ctors
    NcclResources(const NcclResources&) = delete;
    NcclResources& operator=(const NcclResources&) = delete;

    // Move ctors by default
    NcclResources(NcclResources&&) = default;
    NcclResources& operator=(NcclResources&&) = default;

    // Nccl Communicator Getter
    std::vector<ncclComm_t>* ncclComms() {
      return _commEventPair.first.get();
    }

    // Nccl CUDA event Getter
    std::vector<cudaEvent_t>* ncclCudaEvents() {
      return _commEventPair.second.get();
    }

   private:
    std::pair<
        std::unique_ptr<std::vector<ncclComm_t>>,
        std::unique_ptr<std::vector<cudaEvent_t>>>
        _commEventPair;
  };

  // Constructor
  DataChannelNccl(InitMethod::Config config, int timeout = -1);
  virtual ~DataChannelNccl();

  bool init() override;
  void destroy() override;

  rank_type getRank() override;
  rank_type getNumProcesses() override;

  void allReduce(
      std::vector<at::Tensor>& data,
      THDReduceOp operation,
      THDGroup = THDGroupWORLD) override;

  void allReduce(
      at::Tensor& data,
      THDReduceOp operation,
      THDGroup groupId = THDGroupWORLD) override;

  void allGather(
      std::vector<at::Tensor>& output,
      std::vector<at::Tensor>& input,
      THDGroup groupId = THDGroupWORLD) override;

  void allGather(
      std::vector<at::Tensor>& output,
      at::Tensor& input,
      THDGroup groupId = THDGroupWORLD) override;

  void reduce(
      std::vector<at::Tensor>& input,
      THDReduceOp operation,
      rank_type dstRank,
      THDGroup groupId = THDGroupWORLD) override;

  void reduce(
      at::Tensor& data,
      THDReduceOp operation,
      rank_type dstRank,
      THDGroup groupId = THDGroupWORLD) override;

  void broadcast(
      std::vector<at::Tensor>& data,
      rank_type srcRank,
      THDGroup groupId = THDGroupWORLD) override;

  void broadcast(
      at::Tensor& data,
      rank_type srcRank,
      THDGroup groupId = THDGroupWORLD) override;

  void barrier(THDGroup groupId = THDGroupWORLD) override;

  THDGroup newGroup(const std::vector<rank_type>& ranks) override;

  void clearGroupCache(THDGroup groupId = THDGroupWORLD) override;

  // Not supported functions
  void gather(
      std::vector<at::Tensor>& output,
      at::Tensor& input,
      rank_type dstRank,
      THDGroup groupId = THDGroupWORLD) override;

  void scatter(
      std::vector<at::Tensor>& input,
      at::Tensor& output,
      rank_type srcRank,
      THDGroup groupId = THDGroupWORLD) override;

  void send(Scalar& data, rank_type dstRank) override;

  void send(at::Tensor& data, rank_type dstRank) override;

  void receive(Scalar& data, rank_type srcRank) override;

  rank_type receive(at::Tensor& data) override;

  void receive(at::Tensor& data, rank_type srcRank) override;

  RequestNccl* isend(at::Tensor& data, rank_type dstRank) override;

  RequestNccl* ireceive(at::Tensor& data, rank_type srcRank) override;

 private:
  // Current process' rank
  rank_type _rank;
  // Number of processes in network
  rank_type _numProcesses;

  // Accept waiting timeout in milliseconds, optional
  int _timeout;
  // Master's address
  std::string _masterAddr;
  // Master's port
  port_type _masterPort;
  // Socket on which the master is listening
  int _masterListeningSocket;
  /**
   * Sockets on which the master is sending to each slave
   * Note that the sockets in the vector can be in arbitrary order and
   * are not sorted by ranks
   */
  std::vector<int> _masterSendingSockets;
  /**
   * Slave socket, which is used for all other slave ranks other than the master
   * rank (rank 0) to receive rank 0's broadcasted Unique NCCL ID
   * that is used for building the NCCL communicator
   */
  int _slaveSocket;

  // Number of GPUs on each node
  int _numGPUs;
  // Mutex for Nccl Data Channel
  std::mutex _mutex;

  /**
   * The GPU devices each group is currently using.
   * The GPU devices are stored in a device sequence and the cache NCCL
   * communicator is associated with this GPU device sequence
   *
   * e.g. If the group only uses device 0, then the value of
   *      the used device string stored (value of the hashmap) would be "0".
   *
   *      If the group uses device 0 - 7 and the each tensor of the
   *      input tensor list is on device, 0, 1, 2, 3, 4, 5, 6, 7 separately,
   *      then the value of the used device string stored would be
   *      "0,1,2,3,4,5,6,7"
   *
   *      If the group uses device 0 - 7 and the each tensor of the
   *      input tensor list is on device, 0, 4, 5, 6, 7, 1, 2, 3 separately,
   *      then the value of the used device string stored would be
   *      "0,4,5,6,7,1,2,3"
   *
   *      Note that the order of the device for the tensor list matters.
   *
   *      Also note that each group only caches a single NCCL communicator
   *      associated with the current "used device string".
   *
   *      If a new device string appears, the previous
   *      cached communicator will be destroyed and a new one with the new
   *      device string will be built
   */
  std::unordered_map<THDGroup, std::vector<std::string>> _groupDevices;

  /**
   * NCCL resources for for each THDGroup including:
   * NCCL communicator for the current group
   * Cuda Events for all GPUs for NCCL operations of the current group
   */
  std::unordered_map<THDGroup, std::vector<NcclResources>> _groupNcclResources;

  // Existing groups
  std::unordered_map<THDGroup, DataChannel::Group> _groups;

  // Helper function that gets the NCCL communicator
  NcclResourcePair _getNcclResourcePair(
      std::vector<at::Tensor>& input,
      THDGroup groupId);

  /**
   * Helper function that broadcasts the NCCL unique ID to everyone in the rank
   * NCCLID pointed by ncclId of Rank 0 will be sent to other ranks' NCCID
   * pointed by ncclId
   */
  void broadcastUniqueNcclId(ncclUniqueId* ncclId);

  // Helper that checks the input and output tensors
  bool _tensorCheckHelper(
      const std::vector<at::Tensor>& input,
      const std::vector<at::Tensor>& output,
      size_t outputOverInput = 1);

  // Helper that destroys a group's NCCL resources
  void _destroyNcclResources(THDGroup groupId);

  // Group validity checker
  void _checkGroupIdValid(THDGroup groupId);

  // Helper fucntion that destroys all the open sockets
  void _destroySockets();
};

} // namespace thd
