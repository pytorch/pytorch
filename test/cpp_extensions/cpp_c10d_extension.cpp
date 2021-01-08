#include "cpp_c10d_extension.hpp"

#include <map>

namespace c10d {

ProcessGroupTest::WorkTest::~WorkTest() {}

bool ProcessGroupTest::WorkTest::isCompleted() {
  return true;
}

bool ProcessGroupTest::WorkTest::isSuccess() const {
  return true;
}

bool ProcessGroupTest::WorkTest::wait(std::chrono::milliseconds /* unused */) {
  return true;
}

ProcessGroupTest::ProcessGroupTest(int rank, int size)
    : ProcessGroup(rank, size) {}

ProcessGroupTest::~ProcessGroupTest() {}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupTest::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  return c10::make_intrusive<ProcessGroupTest::WorkTest>();
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupTest::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  return c10::make_intrusive<ProcessGroupTest::WorkTest>();
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupTest::allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts) {
  throw std::runtime_error("ProcessGroupTest does not support allreduce_coalesced");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupTest::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  throw std::runtime_error("ProcessGroupTest does not support reduce");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupTest::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  throw std::runtime_error("ProcessGroupTest does not support allgather");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupTest::allgather_base(
    at::Tensor& outputBuffer,
    at::Tensor& inputBuffer,
    const AllgatherOptions& opts) {
  throw std::runtime_error("ProcessGroupTest does not support allgather_base");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupTest::barrier(
    const BarrierOptions& opts) {
  return c10::make_intrusive<ProcessGroupTest::WorkTest>();
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupTest::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  throw std::runtime_error("ProcessGroupTest does not support gather");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupTest::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  throw std::runtime_error("ProcessGroupTest does not support scatter");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupTest::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  throw std::runtime_error("ProcessGroupTest does not support reduce_scatter");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupTest::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  throw std::runtime_error("ProcessGroupTest does not support send");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupTest::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  throw std::runtime_error("ProcessGroupTest does not support recv");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupTest::recvAnysource(
    std::vector<at::Tensor>& tensor,
    int tag) {
  throw std::runtime_error("ProcessGroupTest does not support recvAnysource");
}

c10::intrusive_ptr<ProcessGroup> ProcessGroupTest::createProcessGroupTest(
    const c10::intrusive_ptr<::c10d::Store>& store,
    int rank,
    int size,
    const std::chrono::duration<float>& timeout) {
  return c10::make_intrusive<ProcessGroupTest>(rank, size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createProcessGroupTest", &ProcessGroupTest::createProcessGroupTest);
}

} // namespace c10d
