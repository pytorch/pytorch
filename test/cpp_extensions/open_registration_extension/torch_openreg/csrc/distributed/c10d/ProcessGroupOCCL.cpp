#include <stdexcept>

#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>
#include <c10/util/string_view.h>
#include <torch/headeronly/core/DeviceType.h>

#include "ProcessGroupOCCL.hpp"
namespace {

void checkTensorDevice(
    const at::Tensor& tensor,
    c10::string_view opName,
    c10::string_view argName) {
  if (!tensor.defined()) {
    return;
  }

  TORCH_CHECK(
      tensor.device().type() == c10::DeviceType::PrivateUse1,
      "ProcessGroupOCCL only supports OpenReg (PrivateUse1) tensors. Got ",
      tensor.device().type(),
      " for argument '",
      argName,
      "' in ",
      opName,
      ".");
}

void checkTensorList(
    const std::vector<at::Tensor>& tensors,
    c10::string_view opName,
    c10::string_view argName) {
  for (const auto& tensor : tensors) {
    checkTensorDevice(tensor, opName, argName);
  }
}

void checkTensorListOfLists(
    const std::vector<std::vector<at::Tensor>>& tensorLists,
    c10::string_view opName,
    c10::string_view argName) {
  for (const auto& tensors : tensorLists) {
    checkTensorList(tensors, opName, argName);
  }
}

} // namespace

/**
    @brief Currently, a no-op OCCL ProcessGroup stub for registration, tests and educational purposes.

    This translation unit provides a minimal ProcessGroup backend named "OCCL" (short
    for OpenReg Collective Communications Library) that only validates input shapes/devices
    and immediately returns a completed Work object without performing any actual communication.
    It is intended for:
        - Out-of-tree custom CCL backend registration and linkage tests
        - API surface validation and call-site integration

    All collective point-to-point and synchronization calls are implemented as
    immediate, successful completions (unless a precondition check fails and an
    exception is thrown). As a result, these calls do not exchange data, do not
    synchronize ranks, and do not modify input/output tensors. OCCL is for testing
    and educational purposes only; it does not provide any real distributed functionality.

    ------------------------------------------------------------------------------
    Class: ProcessGroupOCCL::DummyWork

    @brief An immediately-completed Work instance.

    - Construction marks an internal c10::ivalue::Future as completed.
    - isCompleted(): always true after construction.
    - isSuccess(): true unless an error was explicitly set (not expected here).
    - wait(timeout): returns immediately; the Future is already complete.
    - synchronize(): no-op; there is no associated device or stream work.
    - abort(): attempts to set an error only if the Future is not completed;
                         has no effect in the current implementation since the Future is
                         completed in the constructor.
    - getFuture(): returns the already-completed Future.

    Notes:
    - No background progress, streams, or device synchronization is involved.
    - Intended solely as a lightweight placeholder Work for API conformance.

    ------------------------------------------------------------------------------
    Class: ProcessGroupOCCL

    @brief A stub ProcessGroup backend whose collectives validate inputs and then
                 return a DummyWork that is already complete.

    General behavior across all methods below:
    - Input validation: Methods call internal helpers (e.g., checkTensorList,
        checkTensorListOfLists, checkTensorDevice) to validate that inputs are
        defined, have expected dtypes/devices, and adhere to basic invariants.
        This is for building connection with out-of-tree accelerator backend (OpenReg)
        and as a guard for blocking any changes from upstream that will break the compatibility.
    - No communication: No data movement or cross-rank coordination occurs.
    - Options ignored: All options/attributes (e.g., reduce ops, tags, counts,
        timeouts) are accepted but not used to affect behavior.
    - Immediate completion: A ProcessGroupOCCL::DummyWork is returned, whose
        future is already completed successfully.
    - No side-effects: Output tensors are not populated or modified.

    Limitations and caveats:
    - Does not guarantee any semantic property of real collectives (e.g., data
        equivalence across ranks, synchronization, ordering).
    - Barrier does not synchronize ranks; it simply returns a completed Work.
    - Safe for use in unit tests only where communication effects are not required.

    ------------------------------------------------------------------------------
    Factory:

    @fn createProcessGroupOCCL(const c10::intrusive_ptr<c10d::Store>& store, int rank, int size, const std::chrono::duration<float>& timeout)
            Constructs a ProcessGroupOCCL using the given rank and size. The store and
            timeout are accepted for API parity but are not used to affect behavior.
            - @param store: Process group store (ignored).
            - @param rank: This process rank.
            - @param size: World size.
            - @param timeout: Operation timeout hint (ignored).
            - @return A new ProcessGroupOCCL instance.

    ------------------------------------------------------------------------------
    Usage notes:
    - Suitable for tests that only check callability/flow, not correctness of
        distributed results.
    - Do not rely on returned outputs or synchronization semantics.
    - In production, replace with a real backend (e.g., NCCL, Gloo, MPI).
*/
namespace c10d {

// WorkOCCL -----------------------------------------------------------------

ProcessGroupOCCL::DummyWork::DummyWork()
    : Work(-1, OpType::UNKNOWN),
      future_(c10::make_intrusive<c10::ivalue::Future>(c10::NoneType::get())) {
  if (!future_->completed()) {
    future_->markCompleted(c10::IValue());
  }
  finish();
}

ProcessGroupOCCL::DummyWork::~DummyWork() = default;

bool ProcessGroupOCCL::DummyWork::isCompleted() {
  return future_->completed();
}

bool ProcessGroupOCCL::DummyWork::isSuccess() const {
  return !future_->hasError();
}

bool ProcessGroupOCCL::DummyWork::wait(std::chrono::milliseconds /* timeout */) {
  future_->wait();
  return !future_->hasError();
}

void ProcessGroupOCCL::DummyWork::synchronize() {

}

void ProcessGroupOCCL::DummyWork::abort() {
  if (!future_->completed()) {
    future_->setError(std::make_exception_ptr(std::runtime_error("OCCL work aborted")));
  }
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupOCCL::DummyWork::getFuture() {
  return future_;
}

// Options ------------------------------------------------------------------

ProcessGroupOCCL::Options::Options(std::chrono::milliseconds timeout)
    : Backend::Options(OCCL_BACKEND_NAME, timeout) {}

// ProcessGroupOCCL ---------------------------------------------------------

ProcessGroupOCCL::ProcessGroupOCCL(int rank, int size)
    : Backend(rank, size), options_(Options::create()) {}

ProcessGroupOCCL::~ProcessGroupOCCL() = default;

c10::intrusive_ptr<Work> ProcessGroupOCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& /* opts */) {
  checkTensorList(tensors, "broadcast", "tensors");
  return c10::make_intrusive<ProcessGroupOCCL::DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& /* opts */) {
  checkTensorList(tensors, "allreduce", "tensors");
  return c10::make_intrusive<ProcessGroupOCCL::DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::allreduce_sparse(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& /* opts */) {
  checkTensorList(tensors, "allreduce_sparse", "tensors");
  return c10::make_intrusive<ProcessGroupOCCL::DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& /* opts */) {
  checkTensorList(tensors, "allreduce_coalesced", "tensors");
  return c10::make_intrusive<ProcessGroupOCCL::DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& /* opts */) {
  checkTensorList(tensors, "reduce", "tensors");
  return c10::make_intrusive<ProcessGroupOCCL::DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::_reduce_scatter_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const ReduceScatterOptions& /* opts */) {
  checkTensorDevice(outputTensor, "_reduce_scatter_base", "outputTensor");
  checkTensorDevice(inputTensor, "_reduce_scatter_base", "inputTensor");
  return c10::make_intrusive<ProcessGroupOCCL::DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::_allgather_base(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const AllgatherOptions& /* opts */) {
  checkTensorDevice(output_tensor, "_allgather_base", "output_tensor");
  checkTensorDevice(input_tensor, "_allgather_base", "input_tensor");
  return c10::make_intrusive<ProcessGroupOCCL::DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs,
    const AllgatherOptions& /* opts */) {
  checkTensorListOfLists(outputs, "allgather", "outputs");
  checkTensorList(inputs, "allgather", "inputs");
  return c10::make_intrusive<ProcessGroupOCCL::DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& output_lists,
    std::vector<at::Tensor>& input_list,
    const AllgatherOptions& /* opts */) {
  checkTensorListOfLists(output_lists, "allgather_coalesced", "output_lists");
  checkTensorList(input_list, "allgather_coalesced", "input_list");
  return c10::make_intrusive<ProcessGroupOCCL::DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const AllgatherOptions& /* opts */) {
  checkTensorList(outputs, "allgather_into_tensor_coalesced", "outputs");
  checkTensorList(inputs, "allgather_into_tensor_coalesced", "inputs");
  return c10::make_intrusive<ProcessGroupOCCL::DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::gather(
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs,
    const GatherOptions& /* opts */) {
  checkTensorListOfLists(outputs, "gather", "outputs");
  checkTensorList(inputs, "gather", "inputs");
  return c10::make_intrusive<ProcessGroupOCCL::DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::scatter(
    std::vector<at::Tensor>& outputs,
    std::vector<std::vector<at::Tensor>>& inputs,
    const ScatterOptions& /* opts */) {
  checkTensorList(outputs, "scatter", "outputs");
  checkTensorListOfLists(inputs, "scatter", "inputs");
  return c10::make_intrusive<ProcessGroupOCCL::DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::reduce_scatter(
    std::vector<at::Tensor>& outputs,
    std::vector<std::vector<at::Tensor>>& inputs,
    const ReduceScatterOptions& /* opts */) {
  checkTensorList(outputs, "reduce_scatter", "outputs");
  checkTensorListOfLists(inputs, "reduce_scatter", "inputs");
  return c10::make_intrusive<ProcessGroupOCCL::DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const ReduceScatterOptions& /* opts */) {
  checkTensorList(outputTensors, "reduce_scatter_tensor_coalesced", "outputTensors");
  checkTensorList(inputTensors, "reduce_scatter_tensor_coalesced", "inputTensors");
  return c10::make_intrusive<ProcessGroupOCCL::DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& /* outputCounts */,
    std::vector<int64_t>& /* inputCounts */,
    const AllToAllOptions& /* opts */) {
  checkTensorDevice(outputTensor, "alltoall_base", "outputTensor");
  checkTensorDevice(inputTensor, "alltoall_base", "inputTensor");
  return c10::make_intrusive<ProcessGroupOCCL::DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::send(
    std::vector<at::Tensor>& tensors,
    int /* dstRank */,
    int /* tag */) {
  checkTensorList(tensors, "send", "tensors");
  return c10::make_intrusive<ProcessGroupOCCL::DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::recv(
    std::vector<at::Tensor>& tensors,
    int /* srcRank */,
    int /* tag */) {
  checkTensorList(tensors, "recv", "tensors");
  return c10::make_intrusive<ProcessGroupOCCL::DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int /* tag */) {
  checkTensorList(tensors, "recvAnysource", "tensors");
  return c10::make_intrusive<ProcessGroupOCCL::DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::barrier(
    const BarrierOptions& /* opts */) {
  return c10::make_intrusive<ProcessGroupOCCL::DummyWork>();
}

c10::intrusive_ptr<ProcessGroupOCCL> createProcessGroupOCCL(
    const c10::intrusive_ptr<c10d::Store>& store,
    int rank,
    int size,
    const std::chrono::duration<float>& timeout) {
  return c10::make_intrusive<ProcessGroupOCCL>(rank, size);
}

} // namespace c10d
