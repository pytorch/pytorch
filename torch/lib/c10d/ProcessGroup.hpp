#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <ATen/ATen.h>

#include <c10d/Types.hpp>

namespace c10d {

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
  class Work {
   public:
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
    virtual std::vector<at::Tensor> result() const;

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
    //
    // Functionally equivalent to:
    //
    //   while (!isCompleted()) { /* nop */ }
    //   auto success = isSuccess();
    //   if (!success) { std::rethrow_exception(exception()); }
    //   return success;
    //
    virtual void wait();

   protected:
    void finish(std::exception_ptr exception = nullptr);

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool completed_ = false;
    std::exception_ptr exception_;
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
