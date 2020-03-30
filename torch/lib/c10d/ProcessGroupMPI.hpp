#pragma once

#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <c10d/ProcessGroup.hpp>
#include <c10d/Types.hpp>
#include <c10d/Utils.hpp>

#include <mpi.h>

namespace c10d {

// WorkEntry is the state associated with a single MPI run instance.
// It include the source Tensor list and destination Tensor list, as well as
// The actual run function that will operate either on src or dst or both.
struct WorkEntry {
  explicit WorkEntry(
      std::vector<at::Tensor>* srcPtr,
      std::vector<at::Tensor>* dstPtr,
      std::function<void(std::unique_ptr<WorkEntry>&)> run)
      : run(run) {
    if (srcPtr) {
      src = *srcPtr;
    }
    if (dstPtr) {
      dst = *dstPtr;
    }
  }

  // Not copyable
  WorkEntry(const WorkEntry&) = delete;
  // Not copy assignable
  WorkEntry& operator=(const WorkEntry&) = delete;

  // For input and output tensors (in-place), we will always use src
  std::vector<at::Tensor> src;
  std::vector<at::Tensor> dst;
  // src rank returned, for recv only
  int* srcRank = nullptr;
  std::function<void(std::unique_ptr<WorkEntry>&)> run;
};

// ProcessGroupMPI implements MPI bindings for c10d.
//
// All functions on this class are expected to be called in the same
// order across processes in the group. This is the only way that we
// can guarantee to match up the same calls across processes.
//
// All MPI functions provided by this class is asynchronously scheduled on a
// Worker thread. Therefore, ProcessGroupMPI requires the MPI implementation
// that is used to have a minimum thread support value of MPI_THREAD_SERIALIZED.
// That is, The process may be multi-threaded, and multiple threads may make
// MPI calls, but only one at a time: MPI calls are not made concurrently from
// two distinct threads (all MPI calls are serialized). However, with
// MPI_THREAD_SERIALIZED, ProcessGroupMPI will only support a singe process
// group. In other words, no more than 1 process group can be created globally.
//
// If you would like to use multiple ProcessGroupMPI, it requres your MPI
// implemenation to have a thread support value of MPI_THREAD_MULTIPLE, that is,
// multiple threads may call MPI, with no restriction.
//
// Also note that ProcessGroupMPI only supports a single Tensor operation. In
// other words, the size of the input Tensor vector should always be 1.
//
// CUDA tensor can be supported if the MPI used is CUDA-aware MPI, and
// ProcessGroupMPI will automatically detect this support.
class ProcessGroupMPI : public ProcessGroup {
 public:
  class WorkMPI : public ProcessGroup::Work {
   protected:
    friend class ProcessGroupMPI;
  };

  class AsyncWork : public ProcessGroup::Work {
   public:
    AsyncWork(at::Tensor tensor, MPI_Request request);
    virtual ~AsyncWork();

    bool isCompleted() override;

    bool isSuccess() const override;

    int sourceRank() const override;

    bool wait() override;

    void abort() override;

   protected:
    void populateException();

    at::Tensor tensor_;
    MPI_Request request_;
    MPI_Status status_;
  };

  // Constructor will spawn up the worker thread loop
  explicit ProcessGroupMPI(int rank, int size, MPI_Comm pgComm);

  virtual ~ProcessGroupMPI();

  // Abort the MPI program, needs to be called when exception is detected
  void abort();

  std::shared_ptr<ProcessGroup::Work> broadcast(
      std::vector<at::Tensor>& data,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  std::shared_ptr<ProcessGroup::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allgather_base(
      at::Tensor& outputbuffer,
      at::Tensor& inputbuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  std::shared_ptr<ProcessGroup::Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  std::shared_ptr<ProcessGroup::Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag);

  std::shared_ptr<ProcessGroup::Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag);

  std::shared_ptr<ProcessGroup::Work> recvAnysource(
      std::vector<at::Tensor>& tensor,
      int tag);

  std::shared_ptr<ProcessGroup::Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  // Creating a new ProcessGroupMPI, will initiialize MPI if not initialized
  static std::shared_ptr<ProcessGroupMPI> createProcessGroupMPI(
      std::vector<int> ranks = {});

 protected:
  using WorkType =
      std::tuple<std::unique_ptr<WorkEntry>, std::shared_ptr<WorkMPI>>;
  // Worker thread loop
  void runLoop();
  // Helper function that is called by the destructor
  void destroy();

  std::shared_ptr<ProcessGroup::Work> enqueue(std::unique_ptr<WorkEntry> entry);

  bool stop_;

  std::mutex pgMutex_;
  std::thread workerThread_;

  std::deque<WorkType> queue_;
  std::condition_variable queueProduceCV_;
  std::condition_variable queueConsumeCV_;

  // Global states
  static void initMPIOnce();
  static void mpiExit();
  static std::once_flag onceFlagInitMPI;

  static std::mutex pgGlobalMutex_;
  static int mpiThreadSupport_;

  MPI_Comm pgComm_;
};

} // namespace c10d
