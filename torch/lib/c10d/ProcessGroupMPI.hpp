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

namespace c10d {

// WorkEntry is the state associated with a single MPI run instance.
// It include the source Tensor list and destination Tensor list, as well as
// The actual run function that will operate either on src or dst or both.
struct WorkEntry {
  explicit WorkEntry(
      std::vector<at::Tensor>* src,
      std::vector<at::Tensor>* dst,
      std::function<void(std::unique_ptr<WorkEntry>&)> run)
      : src(src), dst(dst), run(run) {}

  // Not copyable
  WorkEntry(const WorkEntry&) = delete;
  // Not copy assignable
  WorkEntry& operator=(const WorkEntry&) = delete;

  // For input and output tensors (in-place), we will always use src
  std::vector<at::Tensor>* src;
  std::vector<at::Tensor>* dst;
  // src rank returned, for recv only
  int* srcRank;
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
   public:
    WorkMPI();
    virtual ~WorkMPI();

    // Checks if request has completed. Non-blocking operation.
    bool isCompleted() const override;

    // Returns if the work completed successfully
    // if false, the exception function can be called to get details.
    bool isSuccess() const override;

    // No op for the case of MPI
    virtual void synchronize() override;

    // Waits until request completes. Blocking operation
    // Returns false if the work completed with an exception
    bool wait() override;

    // Return the exception if wait() returned false.
    const std::exception& exception() const override;

   protected:
    void finish();
    void finishWithException(std::exception_ptr caughtWorkException);

    std::mutex workMutex_;
    std::condition_variable workCV_;
    std::atomic<bool> completed_;

    std::exception_ptr workException_;

    friend class ProcessGroupMPI;
  };

  // Constructor will spawn up the worker thread loop
  explicit ProcessGroupMPI(int rank, int size);

  virtual ~ProcessGroupMPI();

  // Abort the MPI program, needs to be called when exception is detected
  void abort();

  std::shared_ptr<ProcessGroup::Work> broadcast(
      std::vector<at::Tensor>& data,
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
      int dstRank);

  std::shared_ptr<ProcessGroup::Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank);

  std::shared_ptr<ProcessGroup::Work> recvAnysource(
      std::vector<at::Tensor>& tensor,
      int* srcRank);

  std::shared_ptr<ProcessGroup::Work> barrier();

  // Creating a new ProcessGroupMPI, will initiialize MPI if not initialized
  static std::shared_ptr<ProcessGroupMPI> createProcessGroupMPI();

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
  static std::once_flag onceFlagInitMPI;

  static std::mutex pgGlobalMutex_;
  static int numProcessGroups_;
  static int mpiThreadSupport_;
};

} // namespace c10d
