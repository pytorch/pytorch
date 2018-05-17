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
// over time and is agreed upon between all processes in the group. It
// is also used in identifying the algorithm type for which to change
// the maximum alotted number of instances.
//
struct AlgorithmKey {
  bool operator==(const AlgorithmKey &other) const {
    return
      (collectiveType == other.collectiveType) &&
      (type == other.type) &&
      (devices == other.devices) &&
      (srcSizes == other.srcSizes) &&
      (dstSizes == other.dstSizes) &&
      (srcDevice == other.srcDevice) &&
      (dstDevice == other.dstDevice) &&
      (srcRank == other.srcRank) &&
      (dstRank == other.dstRank) &&
      (reduceOp == other.reduceOp);
  }

  CollectiveType collectiveType = CollectiveType::UNUSED;
  at::Type* type = nullptr;
  std::vector<int> devices;
  std::vector<std::vector<int64_t>> srcSizes;
  std::vector<std::vector<int64_t>> dstSizes;
  int srcDevice = -1;
  int dstDevice = -1;
  int srcRank = -1;
  int dstRank = -1;
  ReduceOp reduceOp = ReduceOp::UNUSED;
};

struct AlgorithmEntry {
  AlgorithmKey key;
  std::unique_ptr<::gloo::Algorithm> algorithm;
  std::vector<at::Tensor> src;
  std::vector<at::Tensor> dst;
  std::function<void(std::unique_ptr<AlgorithmEntry>&)> run;

  explicit AlgorithmEntry() {
  }

  // Must not be copied
  AlgorithmEntry & operator=(const AlgorithmEntry&) = delete;
  AlgorithmEntry(const AlgorithmEntry&) = delete;
};

} // namespace c10d

MAKE_HASHABLE(
    ::c10d::AlgorithmKey,
    t.collectiveType,
    t.type,
    t.devices,
    t.srcSizes,
    t.dstSizes,
    t.srcDevice,
    t.dstDevice,
    t.srcRank,
    t.dstRank,
    t.reduceOp);

namespace c10d {

class ProcessGroupGloo : public ProcessGroup {
 public:
  class WorkGloo : public Work {
   public:
    explicit WorkGloo();
    virtual ~WorkGloo();

    // Checks if request has completed. Non-blocking operation.
    bool isCompleted() const override;

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
      int size);

  virtual ~ProcessGroupGloo();

  void initialize();

  void initialize(Options& options);

  void destroy();

  std::shared_ptr<Work> broadcast(
      std::vector<at::Tensor>& data,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  std::shared_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

 protected:
  std::unique_ptr<::gloo::rendezvous::Store> store_;
  std::vector<std::shared_ptr<::gloo::Context>> contexts_;

  using KeyType = AlgorithmKey;
  using EntryType = std::unique_ptr<AlgorithmEntry>;

  EntryType construct(const KeyType& key);
  EntryType checkout(const KeyType& key);

  std::mutex m_;
  std::unordered_map<KeyType, int> cacheCreated_;
  std::unordered_map<KeyType, EntryType> cache_;
  std::condition_variable cacheCV_;

  using WorkType = std::tuple<EntryType, std::shared_ptr<WorkGloo>>;
  std::deque<WorkType> queue_;
  std::condition_variable queueProduceCV_;
  std::condition_variable queueConsumeCV_;

  std::vector<std::thread> threads_;
  bool stop_;

  void runLoop(void);
  void runSingle(WorkType work);
  void createAlgorithm(AlgorithmEntry& entry);

  template <typename T>
  void createAllreduce(AlgorithmEntry& entry);
};

} // namespace c10d
