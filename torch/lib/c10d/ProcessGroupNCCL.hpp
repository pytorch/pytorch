#pragma once

#include <mutex>
#include <unordered_map>

#include <c10d/NCCLUtils.hpp>
#include <c10d/ProcessGroup.hpp>
#include <c10d/Store.hpp>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>

namespace c10d {

// ProcessGroupNCCL implements NCCL bindings for c10d.
//
// All functions of the class are expected to be called in the same order
// across all processes in the process group.  This is the only way that we
// can guarantee to match up the same calls among all processes.
//
// All NCCL functions provided by this class are asynchronous functions. More
// specifically, each NCCL call is scheduled on a separate CUDA stream that is
// different from the current CUDA stream. This is for the purpose of
// achieving potentially concurrency and better performance. As a result,
// it is the callers' responsibilty to make sure that the CUDA stream their
// code works on needs to wait for the NCCL operation from
// this class.
//
// This can be done by calling:
//
// either WorkNCCL::wait() or WorkNCCL::synchronize(), both achieves the same
// functionality and are synonyms.
//
// Note that WorkNCCL::isSuccess() and WorkNCCL::isCompleted() will always
// return true since ProcessGroupNCCL is single threaded. Every single NCCL
// or CUDA failure will simply raise std::runtime_error.
//
// Therefore, WorkNCCL::exception() is not supported since isSuccess() always
// returns true.
//
// Also note that WorkNCCL::finishedGPUExecution() is a helper function only
// provided by ProcessGroupNCCL to check if the NCCL operation of WorkNCCL has
// finished execution on the GPU (not just scheduled).
//
// Example on using the NCCL process group
//
//   ProcessGroupNCCL pg(store, rank, size);
//   std::shared_ptr<WorkNCCL> work = pg.allreduce(tensors);
//
//   // At this point, NCCL kernel has already by queued successfully
//   // Now, let current stream wait for the NCCL to finish, this function is
//   // async operation as well
//
//   work->wait()
//
//   // Now continue on other work in the current stream.
class ProcessGroupNCCL : public ProcessGroup {
 public:
  class WorkNCCL : public ProcessGroup::Work {
   public:
    // Constructor takes a list of CUDA devices
    WorkNCCL(const std::vector<at::Device>& devices);
    virtual ~WorkNCCL();

    // Checks if request has completed. In this specific case of NCCL, it checks
    // if the NCCL operation has completed on the GPU in its own NCCL stream.
    // Non-blocking operation.
    bool isCompleted() override;

    // Let current stream wait on the completing of the NCCL work
    // always return true and will throw if there are exceptions
    // Non-blocking operation
    bool wait() override;

    // Will always return true
    bool isSuccess() const override;

    // Same as wait()
    void synchronize() override;

    // Not supported by WorkNCCL
    const std::exception& exception() const override;

    // Helper function that checks if the NCCL kernels have finished
    // execution on the GPUs
    bool finishedGPUExecution() const;

   protected:
    // The cached list of CUDA devices to operate on
    std::vector<at::Device> devices_;

    // The CUDA events tracking this work item on multiple CUDA devices
    std::vector<at::cuda::CUDAEvent> cudaEvents_;

    friend class ProcessGroupNCCL;
  };

  // Constructor will also check the number of available GPUs in the system
  ProcessGroupNCCL(const std::shared_ptr<Store>& store, int rank, int size);

  virtual ~ProcessGroupNCCL();

  std::shared_ptr<ProcessGroup::Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  std::shared_ptr<ProcessGroup::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors) override;

  // Unsupported Ops
  std::shared_ptr<ProcessGroup::Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  std::shared_ptr<ProcessGroup::Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  std::shared_ptr<ProcessGroup::Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  std::shared_ptr<ProcessGroup::Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int* srcRank,
      int tag) override;

  std::shared_ptr<ProcessGroup::Work> barrier() override;

  std::unordered_map<int, int> getGroupRank() override;

 protected:
  // Helper that broadcasts nccl unique ID to all ranks through the store
  void broadcastUniqueNCCLID(ncclUniqueId* ncclID);

  // Helper that either looks up the cached NCCL communicators or creates
  // a new set of NCCL communicators as a cache entry
  std::vector<std::shared_ptr<NCCLComm>>& getNCCLComm(
      const std::string& devicesKey,
      const std::vector<at::Device>& devices);

  // Tensor checker helper
  void tensorCheckHelper(
      const std::vector<at::Tensor>& input,
      const std::vector<at::Tensor>& output,
      int outputOverInput = 1);

  // Store that is used to exchange each Ranks's NCCL unique ID
  std::shared_ptr<Store> store_;

  // The NCCL communicator that the process group has cached.
  // The key is a list of GPU devices that an operation is operating on
  // The GPU devices are stored in a device sequence and the cache NCCL
  // communicator is associated with this GPU device sequence
  //
  // e.g. If the process group op only uses device 0, then the value of
  // the used device string stored (value of the hashmap) would be "0".
  //
  //      If the process group op uses device 0 - 7 and the each tensor of the
  //      input tensor list is on device, 0, 1, 2, 3, 4, 5, 6, 7 separately,
  //      then the value of the used device string (key) stored would be
  //      "0,1,2,3,4,5,6,7"
  //
  //      If the process group op uses device 0 - 7 and the each tensor of the
  //      input tensor list is on device, 0, 4, 5, 6, 7, 1, 2, 3 separately,
  //      then the value of the used device string stored would be
  //      "0,4,5,6,7,1,2,3"
  //
  //      Note that the order of the device for the tensor list matters.
  std::unordered_map<std::string, std::vector<std::shared_ptr<NCCLComm>>>
      devNCCLCommMap_;

  // The CUDA steams used by NCCL kernels
  std::unordered_map<std::string, std::vector<at::cuda::CUDAStream>>
      ncclStreams_;

  // The CUDA events used to sync NCCL streams
  std::unordered_map<std::string, std::vector<at::cuda::CUDAEvent>> ncclEvents_;

  // ID of this process group
  std::string processGroupID_;

  // processGroupID tracking
  static std::mutex pgTrackingLock_;
  static std::unordered_map<ssize_t, ssize_t> pgUniqueNCCLIDCnt_;
  static ssize_t processGroupCounter_;
};

} // namespace c10d
