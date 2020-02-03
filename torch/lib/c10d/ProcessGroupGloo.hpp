#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include <gloo/algorithm.h>
#include <gloo/common/error.h>
#include <gloo/context.h>
#include <gloo/rendezvous/store.h>
#include <gloo/transport/device.h>

#include <torch/csrc/utils/hash.h>

#ifdef USE_CUDA
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#endif

#include <c10d/ProcessGroup.hpp>
#include <c10d/Store.hpp>
#include <c10d/Types.hpp>
#include <c10d/Utils.hpp>

namespace c10d {

// ProcessGroupGloo implements Gloo bindings for c10d.
//
// All functions on this class are expected to be called in the same
// order across processes in the group. This is the only way that we
// can guarantee to match up the same calls across processes. For
// multi-threaded usage of process groups, you can use consider using
// multiple process group instances.
//
// The Gloo algorithms that this class calls into are cached by their
// signature (see description of AlgorithmKey above). This cache works
// as follows: every function call instantiates an AlgorithmKey and
// looks in the cache for existing entries. If there is one, it is
// removed from the cache and returned to the caller. If there are
// none, a new entry is created and returned. If an entry was created
// before, but is still in use, the call will block and wait until the
// entry is returned to the cache.
//
// In the future, we hope to extend this to allow multiple entries per
// key, to enable parallelism for a single key. The number of entries
// per key must always be identical for all processes. This maximum
// number can be automatically tuned, but only if we let a single
// process take charge, and have it broadcast the limits.
//
class ProcessGroupGloo : public ProcessGroup {
 public:
  // AsyncWork is the Gloo specific superclass for asynchronous work items.
  // We can split asynchronous work into 3 phases:
  // 1) Sanity checks and prepare input (e.g. memcpy)
  // 2) Run operation on background thread
  // 3) Synchronize with completion on foreground thread
  //
  // There is state to be shared between these 3 phases and all of this state
  // is captured in the AsyncWork class and its derivatives.
  //
  // Note: while we are porting operations to use new style collectives, there
  // is a split between operations using the existing caching approach and
  // operations using the new AsyncWork base class. Over time we will port
  // all operations and perform needed cleanup.
  //
  class AsyncWork : public ProcessGroup::Work {
   public:
    static void execute(std::shared_ptr<AsyncWork> work) {
      std::exception_ptr eptr;
      try {
        work->run();
      } catch (...) {
        eptr = std::current_exception();
      }
      work->finish(eptr);
    }

    virtual void run() = 0;

   protected:
    friend class ProcessGroupGloo;
  };

  // For send and recv operations there is no need to pass them to the
  // thread pool as they are entirely completed by the device thread.
  // This work object is used to synchronize completion of the send or
  // recv operation. It keeps a reference to the tensor it is
  // operating on to prevent it from being deallocated while the
  // operation is still in flight.
  class SendWork : public ProcessGroup::Work {
   public:
    explicit SendWork(
        at::Tensor& tensor,
        std::unique_ptr<::gloo::transport::UnboundBuffer> buffer);

    bool wait() override;

    void abort() override;

   protected:
    at::Tensor tensor_;
    std::unique_ptr<::gloo::transport::UnboundBuffer> buffer_;
  };

  class RecvWork : public ProcessGroup::Work {
   public:
    explicit RecvWork(
        at::Tensor& tensor,
        std::unique_ptr<::gloo::transport::UnboundBuffer> buffer);

    int sourceRank() const override;

    bool wait() override;

    void abort() override;

   protected:
    at::Tensor tensor_;
    std::unique_ptr<::gloo::transport::UnboundBuffer> buffer_;
    int srcRank_;
  };

  struct Options {
    explicit Options();

    std::vector<std::shared_ptr<::gloo::transport::Device>> devices;
    std::chrono::milliseconds timeout;
    int threads;
  };

  // Helper functions to create a new device object.
  // They are static functions on this class to keep them logically
  // separate from the rest of the code base (e.g. torch/csrc/distributed).

  // Create new device instance for specific interface.
  static std::shared_ptr<::gloo::transport::Device> createDeviceForInterface(
      const std::string& interface);

  // Create new device instance for specific hostname or address.
  static std::shared_ptr<::gloo::transport::Device> createDeviceForHostname(
      const std::string& hostname);

  // Create new device instance.
  // It tries to resolve this machine's hostname and bind to that address.
  // If that fails (i.e. the hostname doesn't resolve to an address), it
  // falls back to binding to the loopback address.
  static std::shared_ptr<::gloo::transport::Device> createDefaultDevice();

  explicit ProcessGroupGloo(
      const std::shared_ptr<Store>& store,
      int rank,
      int size,
      Options options = Options());

  virtual ~ProcessGroupGloo();

  std::shared_ptr<ProcessGroup::Work> broadcast(
      std::vector<at::Tensor>& tensors,
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
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& output_lists,
      std::vector<at::Tensor>& input_list,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> gather(
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      const GatherOptions& opts = GatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> scatter(
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs,
      const ScatterOptions& opts = ScatterOptions()) override;

  std::shared_ptr<ProcessGroup::Work> reduce_scatter(
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

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
      int tag) override;

  std::shared_ptr<ProcessGroup::Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

 protected:
  std::unique_ptr<::gloo::rendezvous::Store> store_;

  // Every Gloo context represents a set of connections to its peers.
  // In order to use more than one device (or allow for parallelism on
  // a single device), you need multiple contexts.
  std::vector<std::shared_ptr<::gloo::Context>> contexts_;
  std::vector<std::thread> threads_;
  bool stop_;

  // Incremented for every collective we kick off.
  // The value is used as tag for collective operations. Collectives are kicked
  // off in identical order across processes. Therefore the tag can be used
  // to match up operations during concurrent execution.
  uint32_t collectiveCounter_;

  // Returns next collective tag to use (uses collectiveCounter_).
  uint32_t nextTag();

  // Returns the context to use for the specified tag.
  // With `nextTag` returning an increasing number, this should lead
  // to contexts being used in a round-robin fashion.
  std::shared_ptr<::gloo::Context> getContext(uint32_t tag);

  // Entrypoint for worker threads.
  void runLoop(int workerIndex);

  // Queue work to run on worker thread.
  void enqueue(std::shared_ptr<AsyncWork> work);

  // Keep both a queue of pending work, and a vector with in progress work.
  // Both of these can only be mutated when holding the queue lock.
  // We keep both around instead of just the queue, so we can grab a weak_ptr
  // to all in progress and pending work when executing a barrier.
  // When executing a barrier, we need to ensure that all prior work
  // has completed before completing itself.
  std::deque<std::shared_ptr<AsyncWork>> workQueue_;
  std::vector<std::shared_ptr<AsyncWork>> workInProgress_;
  std::mutex workMutex_;
  std::condition_variable workProduceCV_;
  std::condition_variable workConsumeCV_;
};

} // namespace c10d
