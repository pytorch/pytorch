#pragma once

#ifdef USE_C10D_GLOO

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <gloo/algorithm.h>
#include <gloo/common/error.h>
#include <gloo/context.h>
#include <gloo/rendezvous/store.h>
#include <gloo/transport/device.h>

#include <c10/util/hash.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>

#include <ATen/ThreadLocalState.h>

namespace c10d {

constexpr const char* GLOO_BACKEND_NAME = "gloo";

// ProcessGroupGloo implements Gloo bindings for c10d.
//
// All functions on this class are expected to be called in the same
// order across processes in the group. This is the only way that we
// can guarantee to match up the same calls across processes. For
// multi-threaded usage of process groups, you can consider using
// multiple process group instances.
//
class TORCH_API ProcessGroupGloo : public Backend {
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
  // FIXME: This probably should be called WorkGloo since the work is executed
  // in sync mode by a background thread.
  class TORCH_API AsyncWork : public Work {
   public:
    explicit AsyncWork(
        std::vector<std::vector<at::Tensor>> outputTensors,
        OpType opType,
        uint64_t seq,
        const char* profilingTitle = nullptr,
        const std::optional<std::vector<at::Tensor>>& inputTensors =
            std::nullopt);

    ~AsyncWork() override = default;

    static void execute(const c10::intrusive_ptr<AsyncWork>& work);

    virtual void run() = 0;

    std::vector<at::Tensor> result() override;

    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;
    uint64_t getSequencenumber() const override;

    inline at::ThreadLocalState getTLS() const {
      return tls_;
    }

   protected:
    friend class ProcessGroupGloo;

   private:
    void finishWorkGloo();
    void finishWorkGlooError(const std::exception_ptr& eptr);
    inline void recordAsyncWorkProfilingInfo(
        const char* profilingTitle,
        const std::optional<std::vector<at::Tensor>>& inputTensors);

    const std::vector<std::vector<at::Tensor>> outputTensors_;
    c10::intrusive_ptr<at::ivalue::Future> future_;
    std::function<void()> recordFunctionBeforeCallback_;
    const uint64_t seq_;
    at::ThreadLocalState tls_;
  };

  // Wrap c10d store as Gloo store
  class TORCH_API GlooStore : public ::gloo::rendezvous::Store {
   public:
    GlooStore(c10::intrusive_ptr<::c10d::Store> store)
        : store_(std::move(store)) {}

    void setUint(const std::string& key, const std::vector<uint8_t>& value) {
      store_->set(key, value);
    }

    void set(const std::string& key, const std::vector<char>& value) override {
      std::vector<uint8_t> tmp(value.begin(), value.end());
      store_->set(key, tmp);
    }

    std::vector<uint8_t> getUint(const std::string& key) {
      auto value = store_->get(key);
      return value;
    }

    std::vector<char> get(const std::string& key) override {
      auto value = store_->get(key);
      return std::vector<char>(value.begin(), value.end());
    }

    void wait(const std::vector<std::string>& keys) override {
      store_->wait(keys, ::c10d::Store::kDefaultTimeout);
    }

    void wait(
        const std::vector<std::string>& keys,
        const std::chrono::milliseconds& timeout) override {
      store_->wait(keys, timeout);
    }

#ifdef GLOO_STORE_HAS_STORE_V2
    bool has_v2_support() override {
      return store_->hasExtendedApi();
    }

    std::vector<std::vector<char>> multi_get(
        const std::vector<std::string>& keys) override {
      std::vector<std::vector<char>> res;
      for (auto& value : store_->multiGet(keys)) {
        res.emplace_back(value.begin(), value.end());
      }
      return res;
    }

    void multi_set(
        const std::vector<std::string>& keys,
        const std::vector<std::vector<char>>& values) override {
      std::vector<std::vector<uint8_t>> u_values;
      u_values.reserve(values.size());
      for (auto& value : values) {
        u_values.emplace_back(value.begin(), value.end());
      }
      store_->multiSet(keys, u_values);
    }

    void append(const std::string& key, const std::vector<char>& value)
        override {
      std::vector<uint8_t> tmp(value.begin(), value.end());
      return store_->append(key, tmp);
    }

    int64_t add(const std::string& key, int64_t value) override {
      return store_->add(key, value);
    }
#endif

   protected:
    c10::intrusive_ptr<::c10d::Store> store_;
  };

  // For send and recv operations there is no need to pass them to the
  // thread pool as they are entirely completed by the device thread.
  // This work object is used to synchronize completion of the send or
  // recv operation. It keeps a reference to the tensor it is
  // operating on to prevent it from being deallocated while the
  // operation is still in flight.
  class TORCH_API SendWork : public Work {
   public:
    explicit SendWork(
        at::Tensor& tensor,
        std::unique_ptr<::gloo::transport::UnboundBuffer> buffer,
        uint64_t seq);

    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;

    void abort() override;

    uint64_t getSequencenumber() const override;

   protected:
    at::Tensor tensor_;
    std::unique_ptr<::gloo::transport::UnboundBuffer> buffer_;
    const uint64_t seq_;
  };

  class TORCH_API RecvWork : public Work {
   public:
    explicit RecvWork(
        at::Tensor& tensor,
        std::unique_ptr<::gloo::transport::UnboundBuffer> buffer,
        OpType opType,
        uint64_t seq,
        const char* profilingTitle = nullptr);

    int sourceRank() const override;

    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;

    void abort() override;

    uint64_t getSequencenumber() const override;

   protected:
    at::Tensor tensor_;
    std::unique_ptr<::gloo::transport::UnboundBuffer> buffer_;
    int srcRank_;
    const uint64_t seq_;
  };

  struct TORCH_API Options : public Backend::Options {
    explicit Options(
        std::chrono::milliseconds timeout = kBackendDefaultTimeout);

    // return intrusive_ptr of the object
    static c10::intrusive_ptr<Options> create(
        std::chrono::milliseconds timeout = kBackendDefaultTimeout) {
      return c10::make_intrusive<Options>(timeout);
    }

    std::vector<std::shared_ptr<::gloo::transport::Device>> devices;
    int threads;
  };

  const std::string getBackendName() const override {
    return std::string(GLOO_BACKEND_NAME);
  }

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

  // Create ProcessGroupGloo instance.
  static c10::intrusive_ptr<ProcessGroupGloo> createProcessGroupGloo(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size,
      std::chrono::milliseconds timeout);

  explicit ProcessGroupGloo(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<Options> options = Options::create());

  ~ProcessGroupGloo() override;

  c10::intrusive_ptr<Options> getOptions() {
    return options_;
  }

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work> allreduce_sparse(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& output_tensor,
      at::Tensor& input_tensor,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& output_lists,
      std::vector<at::Tensor>& input_list,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      const GatherOptions& opts = GatherOptions()) override;

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs,
      const ScatterOptions& opts = ScatterOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputCounts,
      std::vector<int64_t>& inputCounts,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override;

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  void enableCollectivesTiming() override;

  const std::unique_ptr<::gloo::rendezvous::Store>& _getStore() const {
    return store_;
  }

  // Similar to barrier(), but blocks rank 0 until all other ranks have
  // acknowledged that they are alive (through send/recv from rank 0). Rank 0
  // is able to report all failed ranks if waitAllRanks = true, otherwise
  // reports the first rank it detected as failed.
  void monitoredBarrier(
      const BarrierOptions& opts = BarrierOptions(),
      bool waitAllRanks = false) override;

  // Agrees on an initial sequence number for the whole group by having rank 0
  // create it and broadcast it to other ranks using the store.
  void setSequenceNumberForGroup() override;

  // Retrieves the current sequence number for the whole group, which should be
  // in sync. If the returned number is not consistent across the group, it
  // may indicate that there is some sort of collective desynchronization.
  uint64_t getSequenceNumberForGroup() override;

  int getNumThreads() {
    return options_->threads;
  }

 protected:
  std::unique_ptr<::gloo::rendezvous::Store> store_;
  const c10::intrusive_ptr<Options> options_;

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
  void enqueue(c10::intrusive_ptr<AsyncWork> work);

  // Keep both a queue of pending work, and a vector with in progress work.
  // Both of these can only be mutated when holding the queue lock.
  // We keep both around instead of just the queue, so we can grab a weak_ptr
  // to all in progress and pending work when executing a barrier.
  // When executing a barrier, we need to ensure that all prior work
  // has completed before completing itself.
  std::deque<c10::intrusive_ptr<AsyncWork>> workQueue_;
  std::vector<c10::intrusive_ptr<AsyncWork>> workInProgress_;
  std::mutex workMutex_;
  std::condition_variable workProduceCV_;
  std::condition_variable workConsumeCV_;
  uint64_t seq_{0};
};

} // namespace c10d

#endif // USE_C10D_GLOO
