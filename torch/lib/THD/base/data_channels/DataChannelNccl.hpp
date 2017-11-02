#pragma once

#include "../DataChannel.hpp"
#include "DataChannelUtils.hpp"

#include <nccl.h>

#include <utility>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>


#define NCCL_CHECK(cmd) do {                                  \
  ncclResult_t error = cmd;                                   \
  if (error != ncclSuccess) {                                 \
    std::string err = "NCCL error in: " +                     \
                      std::string(__FILE__) + ":" +           \
                      std::to_string(__LINE__) + ", " +       \
                      std::string(ncclGetErrorString(error)); \
    throw std::runtime_error(err);                            \
  }                                                           \
} while (0)


namespace thd {

struct DataChannelNccl : DataChannel {
  struct RequestNccl : DataChannel::Request {

    RequestNccl(QueueWorker::Request&& request);
    virtual ~RequestNccl();

    virtual bool isCompleted() override;
    virtual void wait() override;

  private:
    QueueWorker::Request _request;
  };

  DataChannelNccl(InitMethod::Config config, int timeout = -1);
  virtual ~DataChannelNccl();

  bool init() override;
  void destroy() override;

  rank_type getRank() override;
  rank_type getNumProcesses() override;

  void allReduce(std::vector<at::Tensor>& input,
                 std::vector<at::Tensor>& output,
                 THDReduceOp operation,
                 THDGroup = THDGroupWORLD) override;

  void allReduce(at::Tensor& data,
                 THDReduceOp operation,
                 THDGroup groupId = THDGroupWORLD) override;

  void allGather(std::vector<at::Tensor>& input,
                 std::vector<at::Tensor>& output,
                 THDGroup groupId = THDGroupWORLD) override;

  void allGather(std::vector<at::Tensor>& output,
                 at::Tensor& input,
                 THDGroup groupId = THDGroupWORLD) override;

  void reduce(std::vector<at::Tensor>& input,
              THDReduceOp operation,
              rank_type dstRank,
              THDGroup groupId = THDGroupWORLD) override;

  void reduce(at::Tensor& data,
              THDReduceOp operation,
              rank_type dstRank,
              THDGroup groupId = THDGroupWORLD) override;

  void broadcast(std::vector<at::Tensor>& data,
                 rank_type srcRank,
                 THDGroup groupId = THDGroupWORLD) override;

  void broadcast(at::Tensor& data,
                 rank_type srcRank,
                 THDGroup groupId = THDGroupWORLD) override;

  void barrier(THDGroup groupId = THDGroupWORLD) override;

  THDGroup newGroup(const std::vector<rank_type>& ranks) override;

  void destroyGroup(THDGroup groupId = THDGroupWORLD) override;

  // Not supported functions
  void gather(std::vector<at::Tensor>& output,
              at::Tensor& input,
              rank_type dstRank,
              THDGroup groupId = THDGroupWORLD) override;

  void scatter(std::vector<at::Tensor>& input,
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
  // Number of GPUs on each node
  int _numGPUs;
  // Mutex for Nccl Data Channel
  std::mutex _mutex;

  /**
   * GPU device ID list for each group, each group should only have one device
   * list be associated with
   */
  std::unordered_map<THDGroup, std::string> _groupDevices;
  /**
   * Communicator for each THDGroup
   * Cuda Events for all GPUs for NCCL operations
   * Each communicator vector will be operating on a different set of
   * CUDA events
   */
  std::unordered_map<THDGroup,
                     std::pair<std::unique_ptr<std::vector<ncclComm_t>>,
                               std::unique_ptr<std::vector<cudaEvent_t>>>>
                    _ncclCommsAndEvents;

  // Existing groups
  std::unordered_map<THDGroup, DataChannel::Group> _groups;

  // Helper function that gets the NCCL communicator
  std::pair<std::vector<ncclComm_t>*, std::vector<cudaEvent_t>*>
    _getNcclCommsAndEvents(std::vector<at::Tensor>& input,
                           THDGroup groupId);

  // Helper function that broadcasts the NCCL unique ID to everyone in the rank
  void broadcastUniqueNcclId(ncclUniqueId* srcNcclId,
                             ncclUniqueId* dstNcclId);

  // Helper that checks the input and output tensors
  void _tensorCheckHelper(const std::vector<at::Tensor>& input,
                          const std::vector<at::Tensor>& output,
                          size_t outputOverInput = 1);

  // Group validity checker
  void _checkGroupIdValid(THDGroup groupId);
};

} // namespace thd
