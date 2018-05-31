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

#include "ProcessGroup.hpp"
#include "Store.hpp"
#include "Types.hpp"
#include "Utils.hpp"

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
  static std::size_t hash(const AlgorithmKey& k) {
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
  class WorkGloo : public ProcessGroup::Work {
   public:
    explicit WorkGloo();
    virtual ~WorkGloo();

    // Checks if request has completed. Non-blocking operation.
    bool isCompleted() const override;

    // Returns if the work completed successfully.
    // If false, the exception function can be called to get details.
    bool isSuccess() const override;

    // Waits until request completes. Blocking operation.
    // Returns false if the work completed with an exception.
    bool wait() override;

    // Returns exception if wait() returned false.
    const std::exception& exception() const override;

   protected:
    void finish();
    void finishWithException(const ::gloo::Exception& ex);

    std::mutex m_;
    std::condition_variable cv_;
    std::atomic<bool> completed_;

    // Use pointer to ::gloo::Exception because it doesn't have a
    // default constructor and constructing an empty std::unique_ptr
    // is probably cheaper (this is highly speculative).
    std::unique_ptr<::gloo::Exception> ex_;

    friend class ProcessGroupGloo;
  };

  struct Options {
    explicit Options();

    std::vector<std::shared_ptr<::gloo::transport::Device>> devices;
    std::chrono::milliseconds timeout;
    int threads;
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

 protected:
  using KeyType = AlgorithmKey;
  using EntryType = std::unique_ptr<AlgorithmEntry>;
  using HashType = torch::hash<AlgorithmKey>;
  using WorkType = std::tuple<AlgorithmEntry*, std::shared_ptr<WorkGloo>>;

  std::unique_ptr<::gloo::rendezvous::Store> store_;
  std::vector<std::shared_ptr<::gloo::Context>> contexts_;
  std::vector<std::thread> threads_;
  bool stop_;

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

  std::unordered_map<KeyType, EntryType, HashType> cache_;

  std::shared_ptr<Work> enqueue(AlgorithmEntry* entry);

  std::deque<WorkType> queue_;
  std::mutex queueMutex_;
  std::condition_variable queueProduceCV_;
  std::condition_variable queueConsumeCV_;
};

} // namespace c10d
