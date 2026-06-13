#pragma once

#ifdef USE_C10D_MPS

#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>

namespace jaccl {
class Group;
} // namespace jaccl

namespace c10d {

constexpr const char* MPS_BACKEND_NAME = "mps";

// RDMA-only MPS backend, layered on top of MLX's JACCL library. Construction
// requires an Apple Thunderbolt RDMA device that accepts ibv_alloc_pd on every
// rank — otherwise this throws and the user is expected to fall back to the
// gloo backend.
class TORCH_API ProcessGroupMPS : public Backend {
 public:
  class WorkMPS : public Work {
   public:
    WorkMPS(
        OpType opType,
        const char* profilingTitle = nullptr,
        const std::optional<std::vector<at::Tensor>>& inputTensors =
            std::nullopt);
    ~WorkMPS() override = default;

    bool isCompleted() override;
    bool isSuccess() const override;
    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

    void finishWork();
    void finishWorkError(const std::exception_ptr& eptr);

   protected:
    friend class ProcessGroupMPS;
    c10::intrusive_ptr<at::ivalue::Future> future_;
    std::vector<at::Tensor> outputTensors_;
  };

  struct TORCH_API Options : public Backend::Options {
    explicit Options(
        std::chrono::milliseconds timeout = kBackendDefaultTimeout);

    static c10::intrusive_ptr<Options> create(
        std::chrono::milliseconds timeout = kBackendDefaultTimeout) {
      return c10::make_intrusive<Options>(timeout);
    }
  };

  ProcessGroupMPS(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<Options> options = Options::create());

  ~ProcessGroupMPS() override;

  const std::string getBackendName() const override {
    return std::string(MPS_BACKEND_NAME);
  }

  c10::intrusive_ptr<Backend::Options> getBackendOptions() override {
    return c10::static_intrusive_pointer_cast<Backend::Options>(options_);
  }

  c10::intrusive_ptr<Options> getOptions() {
    return options_;
  }

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

 private:
  at::Tensor syncAndCopyToCPU(const at::Tensor& tensor);
  void copyToMPS(const at::Tensor& cpuTensor, at::Tensor& mpsTensor);

  void enqueue(std::function<void()> fn);
  void runLoop();

  c10::intrusive_ptr<Store> store_;
  c10::intrusive_ptr<Options> options_;
  std::shared_ptr<::jaccl::Group> jacclGroup_;

  std::thread workerThread_;
  bool stop_{false};
  std::deque<std::function<void()>> workQueue_;
  std::mutex workMutex_;
  std::condition_variable workCV_;
};

} // namespace c10d

#endif // USE_C10D_MPS
