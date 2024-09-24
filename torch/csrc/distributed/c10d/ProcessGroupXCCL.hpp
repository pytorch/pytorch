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
using XCCL_KVS = ccl::shared_ptr_class<ccl::kvs>;
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

    bool isSuccess() const override {
      TORCH_CHECK(
          false, "ProcessGroupXCCL::WorkXCCL::isSuccess not implemented");
    }

    void abort() override {
      TORCH_CHECK(false, "ProcessGroupXCCL::WorkXCCL::abort not implemented");
    }

    void synchronize() override;

    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;

    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
      return future_;
    }

    std::vector<at::Tensor> result() override {
      TORCH_CHECK(false, "ProcessGroupXCCL::WorkXCCL::result not implemented");
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
      OpType opType);

  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<Work> collective(
      at::Tensor& input,
      at::Tensor& output,
      Fn fn,
      PreProcess pre,
      PostProcess post,
      OpType opType);

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override {
    TORCH_CHECK(false, "ProcessGroupXCCL::allreduce_coalesced not implemented");
  }

  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override {
    TORCH_CHECK(false, "ProcessGroupXCCL::reduce not implemented");
  }

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override {
    TORCH_CHECK(false, "ProcessGroupXCCL::broadcast not implemented");
  }

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    TORCH_CHECK(false, "ProcessGroupXCCL::allgather not implemented");
  }

  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputbuffer,
      at::Tensor& inputbuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    TORCH_CHECK(false, "ProcessGroupXCCL::_allgather_base not implemented");
  }

  c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    TORCH_CHECK(false, "ProcessGroupXCCL::allgather_coalesced not implemented");
  }

  c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    TORCH_CHECK(
        false,
        "ProcessGroupXCCL::allgather_into_tensor_coalesced not implemented");
  }

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    TORCH_CHECK(false, "ProcessGroupXCCL::reduce_scatter not implemented");
  }

  c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    TORCH_CHECK(
        false, "ProcessGroupXCCL::_reduce_scatter_base not implemented");
  }

  c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    TORCH_CHECK(
        false,
        "ProcessGroupXCCL::reduce_scatter_tensor_coalesced not implemented");
  }

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override {
    TORCH_CHECK(false, "ProcessGroupXCCL::barrier not implemented");
  }

  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override {
    TORCH_CHECK(false, "ProcessGroupXCCL::alltoall_base not implemented");
  }

  c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override {
    TORCH_CHECK(false, "ProcessGroupXCCL::alltoall not implemented");
  }

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override {
    TORCH_CHECK(false, "ProcessGroupXCCL::send not implemented");
  }

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override {
    TORCH_CHECK(false, "ProcessGroupXCCL::recv not implemented");
  }

  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override {
    TORCH_CHECK(false, "ProcessGroupXCCL::gather not implemented");
  }

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override {
    TORCH_CHECK(false, "ProcessGroupXCCL::scatter not implemented");
  }

 protected:
  std::unordered_map<std::string, at::xpu::XPUStream> xcclStreams_;
  std::unordered_map<std::string, std::shared_ptr<xcclComm_t>>
      inInitializationCommMap_;
  std::unordered_map<std::string, std::shared_ptr<xcclComm_t>> devXCCLCommMap_;
  c10::intrusive_ptr<Store> store_;
  std::mutex mutex_;
  bool blockingWait_ = false;

 private:
  XCCL_KVS kvs;
  std::mutex kvs_mutex;
  XCCL_KVS get_kvs(int rank, c10d::Store& store) {
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

namespace {
int getXCCLEnvVar(std::string envVarName) {
  char* stringValue = std::getenv(envVarName.c_str());
  if (stringValue != nullptr) {
    try {
      int val = std::stoi(stringValue);
      return val;
    } catch (std::exception& e) {
      TORCH_CHECK(
          false,
          "Invalid value for environment variable: " + std::string(envVarName));
    }
  } else {
    return -1;
  }
}

template <typename T>
void setXCCLEnvVar(const std::string& envVarName, T val) {
  if constexpr (std::is_same_v<T, int>) {
    setenv(envVarName.c_str(), std::to_string(val).c_str(), 1);
  } else if constexpr (std::is_same_v<T, std::string>) {
    setenv(envVarName.c_str(), val.c_str(), 1);
  }
}

bool with_mpirun() {
  return (getenv("MPI_LOCALRANKID") || getenv("MPI_LOCALNRANKS") ||
          getenv("PMI_RANK") || getenv("PMI_SIZE") || getenv("PMIX_RANK"))
      ? true
      : false;
}

} // namespace
} // namespace c10d

#endif // USE_C10D_XCCL
