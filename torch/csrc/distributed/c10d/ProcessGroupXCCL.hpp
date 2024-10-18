#pragma once

#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#ifdef USE_C10D_XCCL
#include <ATen/xpu/XPUEvent.h>
#include <oneapi/ccl.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <exception>
#include <memory>
#include <vector>

#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <c10/core/StreamGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
namespace c10d {

static std::vector<std::string> TORCH_XCCL_BLOCKING_WAIT = {
    "TORCH_XCCL_BLOCKING_WAIT",
    "XCCL_BLOCKING_WAIT"};

using xcclComm_t = ccl::communicator;
constexpr const char* XCCL_BACKEND_NAME = "xccl";

class TORCH_API ProcessGroupXCCL : public Backend {
 public:
  class WorkXCCL : public Work {
   public:
    WorkXCCL(
        at::Device& device,
        int rank,
        OpType opType,
        const std::optional<std::vector<at::Tensor>>& inputs = std::nullopt);
    WorkXCCL(const WorkXCCL& w);
    ~WorkXCCL() override;

    bool isCompleted() override;

    void abort() override {
      TORCH_CHECK(false, "ProcessGroupXCCL::WorkXCCL::abort not implemented");
    }

    void synchronize() override;

    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;

    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
      return future_;
    }

    std::vector<at::Tensor> result() override {
      return *outputs_;
    }

   protected:
    at::Device device_;
    std::shared_ptr<at::xpu::XPUEvent> xcclEndEvent_;
    bool blockingWait_ = false;
    std::chrono::time_point<std::chrono::steady_clock> workStartTime_;

   private:
    void synchronizeInternal(std::chrono::milliseconds timeout);
    std::shared_ptr<std::vector<at::Tensor>> outputs_;
    c10::intrusive_ptr<at::ivalue::Future> future_;
    friend class ProcessGroupXCCL;
  };

  ProcessGroupXCCL(const c10::intrusive_ptr<Store>& store, int rank, int size);

  C10_DEPRECATED ProcessGroupXCCL(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size,
      const std::string& groupName)
      : ProcessGroupXCCL(store, rank, size) {}

  ~ProcessGroupXCCL() override;

  const std::string getBackendName() const override {
    return std::string(XCCL_BACKEND_NAME);
  }

  std::shared_ptr<xcclComm_t> getXCCLComm(
      const std::string& deviceKey,
      at::Device& device);

  virtual c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> initWork(
      at::Device& device,
      int rank,
      OpType opType,
      const std::vector<at::Tensor>& inputs = {},
      const std::vector<at::Tensor>& outputs = {});

  template <typename Fn>
  c10::intrusive_ptr<Work> collective(
      at::Tensor& input,
      at::Tensor& output,
      Fn fn,
      OpType opType) {
    auto inputs = std::vector<at::Tensor>{input};
    auto outputs = std::vector<at::Tensor>{output};
    return collective<Fn>(
        inputs,
        outputs,
        fn,
        [](at::xpu::XPUStream&,
           c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>&) {},
        [](at::xpu::XPUStream&,
           c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>&) {},
        opType);
  }

  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<Work> collective(
      std::vector<at::Tensor>& inputs,
      std::vector<at::Tensor>& outputs,
      Fn fn,
      PreProcess pre,
      PostProcess post,
      OpType opType);

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  void setSequenceNumberForGroup() override {}

 protected:
  std::unordered_map<std::string, at::xpu::XPUStream> xcclStreamsMap_;
  std::unordered_map<std::string, at::xpu::XPUEvent> xcclEventsMap_;
  std::unordered_map<std::string, std::shared_ptr<xcclComm_t>> devXCCLCommMap_;
  c10::intrusive_ptr<Store> store_;
  std::mutex mutex_;
  bool blockingWait_ = false;

 private:
  std::mutex kvs_mutex;
  ccl::shared_ptr_class<ccl::kvs> kvs;

  ccl::shared_ptr_class<ccl::kvs> get_kvs(int rank, c10d::Store& store) {
    std::lock_guard<std::mutex> lock(kvs_mutex);
    if (kvs)
      return kvs;
    std::string storeKey = "xccl_kvs";
    // Rank 0 broadcast the bootstrap network information to other ranks
    if (rank == 0) {
      kvs = ccl::create_main_kvs();
      ccl::kvs::address_type main_addr = kvs->get_address();
      auto ccl_kvs_addr =
          std::vector<uint8_t>(main_addr.begin(), main_addr.end());
      store.set(storeKey, ccl_kvs_addr);
    } else {
      auto ccl_kvs_addr = store.get(storeKey);
      if (ccl_kvs_addr.size() != ccl::kvs::address_max_size) {
        throw std::runtime_error("Unexpected ccl kvs addr from the store\n");
      }
      ccl::kvs::address_type main_addr;
      std::copy_n(
          ccl_kvs_addr.begin(), ccl::kvs::address_max_size, main_addr.begin());
      kvs = ccl::create_kvs(main_addr);
    }
    return kvs;
  }
};
} // namespace c10d

#endif // USE_C10D_XCCL
