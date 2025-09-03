#pragma once

#include <torch/extension.h>

#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <chrono>

#include <pybind11/chrono.h>

#include <torch/csrc/distributed/c10d/ProcessGroup.h>
#include <torch/csrc/distributed/c10d/Work.h>
#include <torch/csrc/distributed/c10d/Store.h>
#include <torch/csrc/distributed/c10d/Types.h>
#include <torch/csrc/distributed/c10d/Utils.h>

namespace c10d {

//
// ProcessGroupTest implements dummy bindings for c10d.
//

class ProcessGroupTest : public ProcessGroup {
 public:
  class WorkTest : public Work {
   public:
    WorkTest() {}

    virtual ~WorkTest();
    bool isCompleted() override;
    bool isSuccess() const override;
    bool wait(std::chrono::milliseconds timeout) override;

   protected:
    friend class ProcessGroupTest;
  };

  explicit ProcessGroupTest(int rank = -1, int size = -1);
  virtual ~ProcessGroupTest();

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& data,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts = AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& tensor,
      int tag) override;

  // Create a new ProcessGroupTest instance
  static c10::intrusive_ptr<ProcessGroup> createProcessGroupTest(
      const c10::intrusive_ptr<::c10d::Store>& store,
      int rank,
      int size,
      const std::chrono::duration<float>& timeout);

  static void ProcessGroupTestConstructor() __attribute__((constructor)) {
      py::object module = py::module::import("torch.distributed");
      py::object register_backend = module.attr("Backend").attr("register_backend");
      // The first parameter is the backend name used by user in invoking
      // torch.distributed.init_process_group().
      // Note it could be different with module name. For example, the module
      // name is "torch_test" but the backend name is "test".
      // The second parameter is the instantiation function.
      register_backend("test", py::cpp_function(createProcessGroupTest));
  }

};

} // namespace c10d
