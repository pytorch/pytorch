#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <ATen/ATen.h>

#include <c10d/Types.hpp>

// *************************************************************************
// PROCESS GROUP collective communication API IS BEING CHANGED BETWEEN
// versions 1.7 and 1.8.
// PLEASE DO NOT ADD ANY DEPENDENCIES.
// SEE RFC: https://github.com/pytorch/pytorch/issues/39662
// *************************************************************************

constexpr auto kNoTimeout = std::chrono::milliseconds(0);

namespace c10d {

enum class OpType : std::uint8_t {
  BROADCAST = 0,
  ALLREDUCE = 1,
  ALLREDUCE_COALESCED = 2,
  REDUCE = 3,
  ALLGATHER = 4,
  ALLGATHER_BASE = 5,
  ALLGATHER_COALESCED = 6,
  GATHER = 7,
  SCATTER = 8,
  REDUCE_SCATTER = 9,
  ALLTOALL_BASE = 10,
  ALLTOALL = 11,
  SEND = 12,
  RECV = 13,
  RECVANYSOURCE = 14,
  BARRIER = 15,
  UNKNOWN = 100,
};

// Converts OpType to human readable string.
std::string opTypeToString(OpType opType);

// Whether or not an OP is an p2p op (SEND, RECV, RECVANYSOURCE)
bool isP2POp(OpType opType);

// ProcessGroup is a base class that captures collective and point to
// point communication in a fixed set of processes.
//
// The functions specified in the class below describe the API alone;
// implementations are provided in subclasses.
//
// Every function that performs I/O is executed asynchronously by a
// thread pool owned by the ProcessGroup (by default). They return an
// object that can be used to wait for completion or error.
//
// The ProcessGroup can instantiate subgroups with fewer or an equal
// number of members. Implementations must take care that multiple
// process groups can be used in parallel and synchronize accordingly.
//
// The ProcessGroup assumes a fixed set of processes. If the set
// changes, existing instances must be destructed and instantiation
// and initialization must start from scratch. For members of the
// process group to find each other (referred to as rendezvous from
// hereon)
//
class ProcessGroup {
 public:

  // Please do not use ProcessGroup::Work API, it is going away, to be
  // replaced by ivalue::Future.
  // Python binding for this class might change, please do not assume
  // this will be bound using pybind.
  class Work {
   public:
    Work(int rank = -1, OpType opType = OpType::UNKNOWN, const char* profilingTitle = nullptr);

    virtual ~Work();

    // Checks if request has completed. Non-blocking operation.
    virtual bool isCompleted();

    // Returns if the work completed successfully.
    // If false, the exception function can be called to get details.
    virtual bool isSuccess() const;

    // Returns exception if isSuccess() returned false.
    virtual std::exception_ptr exception() const;

    // Returns source rank if this objects represents a recv-from-any.
    virtual int sourceRank() const;

    // Returns result tensors, if applicable.
    virtual std::vector<at::Tensor> result();

    // Ensures that operations on the output tensors that are invoked
    // after this function returns are correctly sequenced after the
    // asynchronous completion of this work.
    //
    // For CUDA tensors, it inserts stream synchronization such that
    // the streams of the caller wait for completion of the
    // asynchronous operations on the destination tensors.
    //
    // For CPU tensors, it is currently a nop.
    //
    // This function should only be used if the caller polls for
    // completion through the `isCompleted` function, it has returned
    // true, and the `isSuccess` function also has returned true.
    //
    virtual void synchronize();

    // Waits until request completes. Blocking operation.
    // Throws if the work completed with an exception.
    // Returns false if the work is aborted.
    // Otherwise, it always returns true, indicating the work is completed.
    //
    // Functionally equivalent to:
    //
    //   while (!isCompleted()) { /* nop */ }
    //   auto success = isSuccess();
    //   if (!success) { std::rethrow_exception(exception()); }
    //   return success;
    //
    virtual bool wait(std::chrono::milliseconds timeout = kNoTimeout);

    virtual void abort();

    // Returns a Future object that will be associated with the completion of
    // work. Only NCCL backend is currently supported.
    virtual c10::intrusive_ptr<c10::ivalue::Future> getFuture();

    OpType retrieveOpType();

   protected:
    // Completes the work object and optionally sets the exception in a
    // thread-safe manner. Notifies all waiting condition variables as well.
    void finish(std::exception_ptr exception = nullptr);

    // Similar to finish, but throws an exception if one is already set or
    // provided by the user.
    void finishAndThrow(std::exception_ptr exception);

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool completed_ = false;
    std::exception_ptr exception_;

    // Current rank of the node.
    const int rank_;

    // Operation type that this work object refers to.
    OpType opType_;

    // When profiling, the callback to record end of operation event. This
    // callback needs to be called when collective operation is complete.
    std::function<void()> recordFunctionEndCallback_;
  };

  explicit ProcessGroup(int rank, int size);
  virtual ~ProcessGroup();

  int getRank() const {
    return rank_;
  }

  int getSize() const {
    return size_;
  }

  virtual std::shared_ptr<ProcessGroup::Work> broadcast(
      std::vector<at::Tensor>& data,
      const BroadcastOptions& opts = BroadcastOptions()) = 0;

  virtual std::shared_ptr<ProcessGroup::Work> allreduce(
      std::vector<at::Tensor>& data,
      const AllreduceOptions& opts = AllreduceOptions()) = 0;

  // This will be moved out of ProcessGroup, do not add dependencies on this
  // function.
  virtual std::shared_ptr<ProcessGroup::Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts = AllreduceCoalescedOptions()) = 0;

  virtual std::shared_ptr<ProcessGroup::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) = 0;

  virtual std::shared_ptr<ProcessGroup::Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) = 0;

  // Gathers a single tensor inputBuffer into a single buffer outputBuffer that
  // is interpreted as a contigious collection of size inputBuffer * WORLD_SIZE.
  // For implementers of ProcessGroup API and advanced users only.
  virtual std::shared_ptr<ProcessGroup::Work> allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) = 0;

  // This function is deprecated and will be moved out of ProcessGroup to comms:
  // * do not add dependencies on this function,
  // * do not implement it in your ProcessGroup, implement allgather_base
  //   instead.
  virtual std::shared_ptr<ProcessGroup::Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions());

  virtual std::shared_ptr<ProcessGroup::Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) = 0;

  virtual std::shared_ptr<ProcessGroup::Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) = 0;

  virtual std::shared_ptr<ProcessGroup::Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) = 0;

  virtual std::shared_ptr<ProcessGroup::Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) {
    throw std::runtime_error("ProcessGroup does not support alltoall");
  }

  virtual std::shared_ptr<ProcessGroup::Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) {
    throw std::runtime_error("ProcessGroup does not support alltoall");
  }

  virtual std::shared_ptr<ProcessGroup::Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) = 0;

  virtual std::shared_ptr<ProcessGroup::Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) = 0;

  virtual std::shared_ptr<ProcessGroup::Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) = 0;

  virtual std::shared_ptr<ProcessGroup::Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) = 0;

 protected:
  const int rank_;
  const int size_;
};

} // namespace c10d
