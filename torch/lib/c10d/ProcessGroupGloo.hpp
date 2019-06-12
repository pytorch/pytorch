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
#include <ATen/cuda/CUDAStream.h>
#endif

#include <c10d/ProcessGroup.hpp>
#include <c10d/Store.hpp>
#include <c10d/Types.hpp>
#include <c10d/Utils.hpp>

namespace c10d {

// AlgorithmKey is a const identifier for a Gloo algorithm.
//
// It captures the set of participating devices, the source device,
// destination device, source rank, destination rank, reduction type
// (if applicable), etcetera. This key is used to cache instances of a
// Gloo algorithm for reuse. The number of cached instances can vary
// over time and is agreed upon between all processes in the group.
//
// When we're dealing with multiple entries per key, it is also used
// to broadcast the number of entries such that all processes agree.
//
struct AlgorithmKey {
  bool operator==(const AlgorithmKey& other) const {
    return (collectiveType == other.collectiveType) && (type == other.type) &&
        (devices == other.devices) && (srcSizes == other.srcSizes) &&
        (dstSizes == other.dstSizes) && (srcRank == other.srcRank) &&
        (dstRank == other.dstRank) && (srcTensor == other.srcTensor) &&
        (dstTensor == other.dstTensor) && (reduceOp == other.reduceOp);
  }

  CollectiveType collectiveType = CollectiveType::UNUSED;
  at::Type* type = nullptr;
  std::vector<int> devices;
  std::vector<std::vector<int64_t>> srcSizes;
  std::vector<std::vector<int64_t>> dstSizes;
  int srcRank = -1;
  int dstRank = -1;
  int srcTensor = -1;
  int dstTensor = -1;
  ReduceOp reduceOp = ReduceOp::UNUSED;

  // This function is called by torch::hash<AlgorithmKey>
  static size_t hash(const AlgorithmKey& k) {
    return torch::get_hash(
        k.collectiveType,
        k.type,
        k.devices,
        k.srcSizes,
        k.dstSizes,
        k.srcRank,
        k.dstRank,
        k.srcTensor,
        k.dstTensor,
        k.reduceOp);
  }
};

// AlgorithmEntry is the state associated with a single algorithm instance.
//
// Keeping Gloo algorithms around for reuse is a win, since most of
// them end up allocating some memory, constructing them takes some
// time, and they may do some I/O (to setup communication buffers
// between processes). Also, until it supports executing on arbitrary
// memory, we need to hold on to memory that we instantiated the
// algorithm with. The lifecycle of memory in ATen is arbitrary, so to
// do caching, this entry holds on to memory that we copy to/from.
//
// Every unique call (in terms of number of tensors, tensor types,
// tensor sizes, etc.) gets its own entry. In the future we may extend
// this to allow multiple entries per unique call, to better exploit
// parallelism for calls with the same signature.
//
struct AlgorithmEntry {
  AlgorithmKey key;
  std::unique_ptr<::gloo::Algorithm> algorithm;
  std::vector<at::Tensor> src;
  std::vector<at::Tensor> dst;
  std::function<void()> run;

#ifdef USE_CUDA
  // For CUDA tensors, the following happens:
  //
  // - Input tensor A is copied to persistent tensor B on the stream
  //   associated with the device that stores A
  // - This stream is recorded in an event (see events below) so that
  //   the copy can be synchronized.
  // - The private stream (see streams below) that is used to execute
  //   the algorithm on a worker thread waits for this event such that
  //   we know the copy to tensor B has finished.
  // - Once the algorithm has finished executing, the work object
  //   associated with the execution records the private streams in
  //   its own events. Then, when the wait() function on the work
  //   object is called, the streams of the caller are synchronized
  //   with asynchronous completion of the memory copies back to the
  //   destination tensors.
  //
  // This approach means the caller of the process group function can
  // retain asynchrony (no need for synchronizing its CUDA streams).
  // Once the wait() function on the associated work object returns
  // true, the caller can launch new CUDA kernels and they will be
  // correctly sequenced.
  //
  std::vector<at::cuda::CUDAStream> streams;
  std::vector<at::cuda::CUDAEvent> events;
#endif

  // Used to synchronize between calling thread and worker threads.
  std::mutex m;
  std::condition_variable cv;
  bool busy = false;

  // Default constructor must be specified.
  AlgorithmEntry() = default;

  // Must not be copyable.
  // This is implied by the std::unique_ptr member field, but serves
  // as documentation in case it ever is removed.
  AlgorithmEntry& operator=(const AlgorithmEntry&) = delete;
  AlgorithmEntry(const AlgorithmEntry&) = delete;
};

} // namespace c10d

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
    bool isCompleted() override;
    bool isSuccess() const override;
    void synchronize() override;
    bool wait() override;
    const std::exception& exception() const override;

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
    std::mutex m_;
    std::condition_variable cv_;
    bool completed_ = false;
    std::exception_ptr eptr_;

    void finish(std::exception_ptr ptr);

    friend class ProcessGroupGloo;
  };

  class WorkGloo : public ProcessGroup::Work {
   public:
    explicit WorkGloo();
    virtual ~WorkGloo();

    bool isCompleted() override;
    bool isSuccess() const override;
    void synchronize() override;
    bool wait() override;
    const std::exception& exception() const override;

   protected:
    void finish(const AlgorithmEntry& entry);
    void finishWithException(const ::gloo::Exception& ex);

    std::mutex m_;
    std::condition_variable cv_;
    std::atomic<bool> completed_;

    // Use pointer to ::gloo::Exception because it doesn't have a
    // default constructor and constructing an empty std::unique_ptr
    // is probably cheaper (this is highly speculative).
    std::unique_ptr<::gloo::Exception> ex_;

#ifdef USE_CUDA
    // List of devices and events so that we can synchronize the
    // streams of the caller with the kernels that were launched
    // asynchronously to finish this operation.
    //
    // These events are private to a single work instance. An event
    // captures the progress of a stream at a single point in time. If
    // we were to use events stored on the algorithm entry, then
    // multiple work instances might end up using the same events, and
    // end up interfering with each other (causing unnecessary
    // synchronization delays). Using events that are private to a
    // single work instance avoids this. Ad hoc benchmarks showed that
    // event construction is relatively cheap: creating 8 events takes
    // 3 microseconds on a fast machine.
    //
    // Also see CUDA comment in AlgorithmEntry struct.
    //
    bool cuda_;
    std::vector<int> devices_;
    std::vector<at::cuda::CUDAEvent> events_;
#endif

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

    virtual ~SendWork() = default;

    bool isCompleted() override;

    bool isSuccess() const override;

    void synchronize() override;

    bool wait() override;

    const std::exception& exception() const override;

   protected:
    at::Tensor tensor_;
    std::unique_ptr<::gloo::transport::UnboundBuffer> buffer_;
  };

  class RecvWork : public ProcessGroup::Work {
   public:
    explicit RecvWork(
        at::Tensor& tensor,
        std::unique_ptr<::gloo::transport::UnboundBuffer> buffer,
        int* srcRank);

    virtual ~RecvWork() = default;

    bool isCompleted() override;

    bool isSuccess() const override;

    void synchronize() override;

    bool wait() override;

    const std::exception& exception() const override;

   protected:
    at::Tensor tensor_;
    std::unique_ptr<::gloo::transport::UnboundBuffer> buffer_;
    int* srcRank_;
  };

  struct Options {
    explicit Options();

    std::vector<std::shared_ptr<::gloo::transport::Device>> devices;
    std::chrono::milliseconds timeout;
    int threads;

    // This controls how many Gloo algorithm instances are created for
    // a single identifying key. If you have many identical calls with
    // tensors of identical size and need to parallelize, this should
    // be greater than 1. More cache entries means more memory usage.
    // The default value is 1.
    int cacheNumAlgorithmEntries;
  };

  explicit ProcessGroupGloo(
      const std::shared_ptr<Store>& store,
      int rank,
      int size,
      Options options = Options());

  virtual ~ProcessGroupGloo();

  std::shared_ptr<Work> broadcast(
      std::vector<at::Tensor>& data,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  std::shared_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  // Unsupported Ops
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
  using KeyType = AlgorithmKey;
  using EntryType = std::unique_ptr<AlgorithmEntry>;
  using HashType = torch::hash<AlgorithmKey>;
  using WorkType = std::
      tuple<AlgorithmEntry*, std::shared_ptr<WorkGloo>, std::function<void()>>;

  std::unique_ptr<::gloo::rendezvous::Store> store_;
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

  void runLoop(void);

  void runSingle(WorkType work);

  void createAlgorithm(AlgorithmEntry& entry);

  template <typename T>
  void createAllreduce(AlgorithmEntry& entry);

  template <typename T>
  void createBroadcast(AlgorithmEntry& entry);

  // Construct creates AlgorithmEntry for specified key.
  EntryType construct(const KeyType& key);

  // Checkout constructs new AlgorithmEntry or returns existing one.
  AlgorithmEntry* checkout(const KeyType& key);

  // The maximum number of cached algorithms for a single key.
  const int cacheNumAlgorithmEntries_;

  // Index of the next algorithm to use for a particular key.
  // Note that this index must be the same for all particating processes.
  std::unordered_map<KeyType, int, HashType> cacheCurrentEntry_;

  // The list of cached algorithms, by algorithm key.
  std::unordered_map<KeyType, std::vector<EntryType>, HashType> cache_;

  std::shared_ptr<Work> enqueue(AlgorithmEntry* entry);

  // Queue std::function to run on the background thread pool.
  void enqueue(std::function<void()> fn);

  std::deque<WorkType> queue_;
  std::mutex queueMutex_;
  std::condition_variable queueProduceCV_;
  std::condition_variable queueConsumeCV_;
};

} // namespace c10d
