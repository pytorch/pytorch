#ifdef USE_C10D_XCCL

#include <torch/csrc/distributed/c10d/xccl2/ProcessGroupXCCL.hpp>

#include <oneapi/ccl.h>
#include <torch/csrc/distributed/c10d/xccl2/Logging.hpp>
#include <torch/csrc/distributed/c10d/xccl2/XCCLCachingAllocatorHook.hpp>
#include <stdexcept>
#include <string>
#include <variant>

namespace c10d::xccl2 {

namespace {

// Scaling factor for a PREMUL_SUM reduction: either a per-element device tensor
// or a host scalar.
using PreMulSumFactorT = std::variant<at::Tensor, double>;

// Extract the scaling factor from a c10d PREMUL_SUM ReduceOp supplement.
// torchcomms carried the factor on its own ReduceOp type (op.factor()); c10d
// instead hangs it off the ReduceOp supplement, so unpack it here.
PreMulSumFactorT getPreMulSumFactor(const ::c10d::ReduceOp& op) {
  TORCH_CHECK(
      op.supplement_ != nullptr,
      "PREMUL_SUM operation requires a supplement, but none was provided");
  const auto* preMulSupplement =
      dynamic_cast<const ::c10d::NCCLPreMulSumSupplement*>(
          op.supplement_.get());
  TORCH_CHECK(
      preMulSupplement != nullptr,
      "PREMUL_SUM operation supplement must be of type NCCLPreMulSumSupplement");
  if (preMulSupplement->tensor_factor.defined()) {
    return preMulSupplement->tensor_factor;
  }
  return preMulSupplement->double_factor;
}

onecclDataType_t getXcclDataTypeInternal(const at::Tensor& tensor) {
  switch (tensor.scalar_type()) {
    case at::ScalarType::Float:
      return onecclFloat32;
    case at::ScalarType::Double:
      return onecclFloat64;
    case at::ScalarType::Half:
      return onecclFloat16;
    case at::ScalarType::BFloat16:
      return onecclBfloat16;
    case at::ScalarType::Int:
      return onecclInt32;
    case at::ScalarType::Long:
      return onecclInt64;
    case at::ScalarType::Char:
      return onecclInt8;
    case at::ScalarType::Byte:
    case at::ScalarType::Bool:
      return onecclUint8;
    default:
      TORCH_CHECK(false, "Unsupported tensor data type for XCCL");
  }
}

template <typename T, onecclDataType_t dataType>
void createPreMulSum(
    onecclRedOp_t* op,
    const PreMulSumFactorT& factor,
    const onecclComm_t& comm,
    XcclApi* xccl_api) {
  const bool is_tensor = std::holds_alternative<at::Tensor>(factor);
  const auto residence =
      is_tensor ? onecclScalarDevice : onecclScalarHostImmediate;

  at::Tensor tensor = is_tensor ? std::get<at::Tensor>(factor) : at::Tensor();
  T scalar_factor = is_tensor ? T{} : static_cast<T>(std::get<double>(factor));
  void* scalar = is_tensor ? tensor.data_ptr() : &scalar_factor;

  TORCH_INTERNAL_ASSERT(
      is_tensor ? dataType == getXcclDataTypeInternal(tensor)
                : dataType != onecclBfloat16,
      "PreMulSum factor type must match input data type");
  XCCL_CHECK(
      xccl_api,
      xccl_api->redOpCreatePreMulSum(op, scalar, dataType, residence, comm),
      "XCCL redOpCreatePreMulSum failed");
}

} // namespace

ProcessGroupXCCL::RedOpRAII::RedOpRAII(onecclRedOp_t op) : xcclRedOp_(op) {}

ProcessGroupXCCL::RedOpRAII::RedOpRAII(
    const ::c10d::ReduceOp& op,
    onecclComm_t comm,
    const onecclDataType_t dataType,
    std::shared_ptr<XcclApi> xccl_api)
    : comm_(comm), xccl_api_(std::move(xccl_api)) {
  TORCH_INTERNAL_ASSERT(
      op == ::c10d::ReduceOp::PREMUL_SUM,
      "Constructing premul_sum RedOpRAII with non-premul_sum RedOpType");

  const auto factor = getPreMulSumFactor(op);
  switch (dataType) {
    case onecclFloat16:
      createPreMulSum<at::Half, onecclFloat16>(
          &xcclRedOp_, factor, comm, xccl_api_.get());
      break;
    case onecclFloat32:
      createPreMulSum<float, onecclFloat32>(
          &xcclRedOp_, factor, comm, xccl_api_.get());
      break;
    case onecclBfloat16:
      createPreMulSum<float, onecclBfloat16>(
          &xcclRedOp_, factor, comm, xccl_api_.get());
      break;
    case onecclFloat64:
      createPreMulSum<double, onecclFloat64>(
          &xcclRedOp_, factor, comm, xccl_api_.get());
      break;
    default:
      TORCH_CHECK(
          false, "PreMulSum Data type must be half, float, bfloat16 or double");
  }
}

ProcessGroupXCCL::RedOpRAII::~RedOpRAII() {
  if (comm_) {
    XCCL_CHECK_IGNORE(
        xccl_api_,
        xccl_api_->redOpDestroy(xcclRedOp_, comm_),
        "XCCL redOpDestroy failed");
  }
}

size_t ProcessGroupXCCL::wordSize(onecclDataType_t type) const {
  switch (type) {
    case onecclInt8:
    case onecclUint8:
      return 1;
    case onecclFloat16:
    case onecclBfloat16:
      return 2;
    case onecclInt32:
    case onecclUint32:
    case onecclFloat32:
      return 4;
    case onecclInt64:
    case onecclUint64:
    case onecclFloat64:
      return 8;
    default:
      return 0;
  }
}

onecclDataType_t ProcessGroupXCCL::getXcclDataType(const at::Tensor& tensor) {
  return getXcclDataTypeInternal(tensor);
}

ProcessGroupXCCL::RedOpRAII ProcessGroupXCCL::getXcclReduceOp(
    const ::c10d::ReduceOp& op,
    onecclComm_t comm,
    const onecclDataType_t dataType) {
  switch (op) {
    case ::c10d::ReduceOp::SUM:
      return onecclSum;
    case ::c10d::ReduceOp::PRODUCT:
      return onecclProd;
    case ::c10d::ReduceOp::MIN:
      return onecclMin;
    case ::c10d::ReduceOp::MAX:
      return onecclMax;
    case ::c10d::ReduceOp::PREMUL_SUM:
      return RedOpRAII(op, comm, dataType, xccl_api_);
    case ::c10d::ReduceOp::AVG:
      return onecclAvg;
    case ::c10d::ReduceOp::BAND:
      // XCCL doesn't have bitwise AND
      TORCH_CHECK(false, "Unsupported BAND reduce operation");
    case ::c10d::ReduceOp::BOR:
      // XCCL doesn't have bitwise OR
      TORCH_CHECK(false, "Unsupported BOR reduce operation");
    case ::c10d::ReduceOp::BXOR:
      // XCCL doesn't have bitwise XOR
      TORCH_CHECK(false, "Unsupported BXOR reduce operation");
    default:
      TORCH_CHECK(false, "Unsupported reduce operation");
  }
}

void ProcessGroupXCCL::checkWorkQueue() {
  WorkXCCL::WorkStatus status = workq_.garbageCollect();

  switch (status) {
    case WorkXCCL::WorkStatus::TIMEDOUT:
      comm_state_ = CommState::TIMEOUT;
      break;
    case WorkXCCL::WorkStatus::ERROR:
      comm_state_ = CommState::ERROR;
      break;
    default:
      // For COMPLETED, NOT_STARTED, and INPROGRESS, no state change needed
      break;
  }
}

// The timeout thread cannot make XCCL calls. The only XPU call it can make is
// an event query via the work status checks.
//
// NOTE: unlike nccl2's watchdog, this one does not poll the communicator for
// asynchronous errors. oneCCL declares onecclCommGetAsyncError with
// CCL_C_NOT_IMPLEMENTED, which expands to __attribute__((error(...))), so
// referencing it does not even compile (see XcclApi.cpp). Communicator faults
// are therefore only observed once they make a work item time out.
void ProcessGroupXCCL::timeoutWatchdog() noexcept {
  TC_LOG(INFO, this) << "Timeout thread starting for rank: " << rank_;
  while (!shutdown_) {
    {
      std::unique_lock<std::mutex> lock(timeout_mutex_);
      // Wait for a shorter interval to check work objects periodically
      // Wake up either after 1 second or immediately if shutdown is requested
      timeout_cv_.wait_for(
          lock, std::chrono::seconds(1), [this]() { return shutdown_.load(); });

      // If we're shutting down, exit the loop
      if (shutdown_) {
        break;
      }
    }

    // Check work objects for completion or timeout
    checkWorkQueue();
    if (shutdown_) {
      break;
    }
    if (comm_state_ != CommState::NORMAL &&
        abort_process_on_timeout_or_error_) {
      // Log the error and abort the process.  We cannot abort the XCCL
      // communicator as it is not safe to call XCCL operations from
      // multiple threads at the same time.
      if (comm_state_ == CommState::TIMEOUT) {
        TC_LOG(ERROR, this)
            << "Aborting process due to timeout on rank " << rank_
            << " - timeout watchdog detected operation timeout";
      } else if (comm_state_ == CommState::ERROR) {
        TC_LOG(ERROR, this) << "Aborting process due to error on rank " << rank_
                            << " - timeout watchdog detected operation error. ";
      }

      runAbortHooks();

      ::abort();
    }
  }

  TC_LOG(INFO, this) << "Timeout thread exiting for rank: " << rank_;
}

void ProcessGroupXCCL::checkInitialized() const {
  TORCH_CHECK(
      init_state_ == InitializationState::INITIALIZED,
      "ProcessGroupXCCL not initialized");
}

void ProcessGroupXCCL::checkAndAbortIfTimedOutOrError() {
  // First, check work queue status
  checkWorkQueue();

  if (comm_state_ == CommState::TIMEOUT) {
    // NOTE: nccl2 aborts the communicator here before reporting the timeout.
    // oneCCL has no abort (onecclCommAbort is CCL_C_NOT_IMPLEMENTED), so the
    // communicator is left as-is and the timeout is reported directly. When
    // abort_process_on_timeout_or_error_ is false the caller gets an exception
    // while peers may still be blocked in the collective.
    if (abort_process_on_timeout_or_error_) {
      TC_LOG(ERROR, this) << "Aborting process due to timeout";
      runAbortHooks();
      ::abort();
    } else {
      TORCH_CHECK(false, "XCCL operation timed out");
    }
  } else if (comm_state_ == CommState::ERROR) {
    throwAsyncError(false);
  }
}

c10::intrusive_ptr<WorkXCCL> ProcessGroupXCCL::createWork(
    c10::xpu::XPUStream stream,
    std::chrono::milliseconds timeout,
    const std::vector<at::Tensor>& inputTensors) {
  // Only create the work object without enqueuing it
  return c10::make_intrusive<WorkXCCL>(this, stream, timeout, inputTensors);
}

c10::intrusive_ptr<WorkXCCL> ProcessGroupXCCL::createWork(
    c10::xpu::XPUStream stream,
    std::chrono::milliseconds timeout,
    const at::Tensor& inputTensor) {
  // Single-tensor overload to avoid vector allocation
  return c10::make_intrusive<WorkXCCL>(this, stream, timeout, inputTensor);
}

void ProcessGroupXCCL::enqueueWork(
    c10::intrusive_ptr<WorkXCCL> work,
    c10::xpu::XPUStream stream) {
  // Add work to stream's queue after events have been recorded
  workq_.enqueueWork(std::move(work), stream);
}

c10::xpu::XPUStream ProcessGroupXCCL::getOperationStream(bool async_op) {
  if (async_op) {
    // Get current PyTorch XPU stream for this device
    auto current_stream = c10::xpu::getCurrentXPUStream(device_.index());
    TORCH_CHECK(
        dependency_event_.has_value() && internal_stream_.has_value(),
        "XCCL stream resources are not initialized");
    auto& dependency_event = dependency_event_.value();
    const auto& internal_stream = internal_stream_.value();

    // Record event on current stream and wait for it on internal stream, so
    // the internal stream is ordered after whatever the caller queued.
    dependency_event.record(current_stream);
    dependency_event.block(internal_stream);

    return internal_stream;
  }
  // Use the current PyTorch XPU stream for synchronous operations
  return c10::xpu::getCurrentXPUStream(device_.index());
}

void ProcessGroupXCCL::ensureTensorContiguous(const at::Tensor& tensor) {
  TORCH_CHECK(
      tensor.is_contiguous(), "Tensor must be contiguous for XCCL operations");
}

void ProcessGroupXCCL::register_address(void* addr, size_t len) {
  if (xccl_comm_ == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(memory_registration_mutex_);
  TORCH_CHECK(
      !memoryRegistrationHandles_.count(addr),
      "Memory already registered with XCCL");
  void* handle = nullptr;
  XCCL_CHECK(
      xccl_api_,
      xccl_api_->commRegister(xccl_comm_, addr, len, &handle),
      "Failed to register memory with XCCL");
  memoryRegistrationHandles_.emplace(addr, RegistrationHandle{handle, len});
}

void ProcessGroupXCCL::deregister_address(void* addr) {
  if (xccl_comm_ == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(memory_registration_mutex_);
  auto it = memoryRegistrationHandles_.find(addr);
  if (it == memoryRegistrationHandles_.end()) {
    return;
  }
  XCCL_CHECK(
      xccl_api_,
      xccl_api_->commDeregister(xccl_comm_, it->second.handle),
      "Failed to deregister memory with XCCL");
  memoryRegistrationHandles_.erase(it);
}

void ProcessGroupXCCL::attachMemoryHook() {
  XCCLCachingAllocatorHook::getInstance().registerComm(this);
}

void ProcessGroupXCCL::detachMemoryHook() {
  XCCLCachingAllocatorHook::getInstance().deregisterComm(this);
}

} // namespace c10d::xccl2

#endif // USE_C10D_XCCL
