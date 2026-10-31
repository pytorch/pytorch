#ifdef USE_C10D_XCCL

#include <torch/csrc/distributed/c10d/xccl2/ProcessGroupXCCL.hpp>

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>

#include <c10/core/ScalarType.h>
#include <c10/core/StreamGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>

#include <torch/csrc/distributed/c10d/xccl2/Logging.hpp>
#include <torch/csrc/distributed/c10d/xccl2/TracingGuard.hpp>
#include <torch/csrc/distributed/c10d/xccl2/XCCLBootstrap.hpp>

namespace c10d::xccl2 {

namespace {

void checkSameDtype(const at::Tensor& reference, const at::Tensor& tensor) {
  if (reference.scalar_type() != tensor.scalar_type()) {
    C10_THROW_ERROR(TypeError, "Tensors must have identical type");
  }
}

void checkSameDtype(
    const at::Tensor& reference,
    const std::vector<at::Tensor>& tensors) {
  for (const auto& tensor : tensors) {
    checkSameDtype(reference, tensor);
  }
}

// Scale `tensor` in place by the PREMUL_SUM factor carried on the c10d
// ReduceOp supplement. torchcomms read this from its own ReduceOp::factor().
void applyPreMulFactor(at::Tensor& tensor, const ::c10d::ReduceOp& op) {
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
    tensor.mul_(preMulSupplement->tensor_factor);
  } else {
    tensor.mul_(preMulSupplement->double_factor);
  }
}

// oneCCL does not implement PREMUL_SUM or AVG natively, so both are lowered to
// SUM here: PREMUL_SUM by pre-scaling a copy of the input, AVG by dividing the
// result afterwards at the call site.
//   PREMUL_SUM: https://github.com/uxlfoundation/oneCCL/issues/196
//   AVG:        https://github.com/uxlfoundation/oneCCL/issues/195
template <typename T>
std::pair<T, ::c10d::ReduceOp> getMaybeScaledInputsAndNewOp(
    const T& input,
    const ::c10d::ReduceOp& op,
    c10::xpu::XPUStream stream) {
  if (op == ::c10d::ReduceOp::PREMUL_SUM) {
    c10::StreamGuard guard(stream);
    if constexpr (std::is_same_v<T, at::Tensor>) {
      at::Tensor scaled_input = input.clone();
      applyPreMulFactor(scaled_input, op);
      return {std::move(scaled_input), ::c10d::ReduceOp(::c10d::ReduceOp::SUM)};
    } else if constexpr (std::is_same_v<T, std::vector<at::Tensor>>) {
      std::vector<at::Tensor> scaled_inputs;
      scaled_inputs.reserve(input.size());
      for (const auto& tensor : input) {
        at::Tensor scaled_input = tensor.clone();
        applyPreMulFactor(scaled_input, op);
        scaled_inputs.push_back(std::move(scaled_input));
      }
      return {
          std::move(scaled_inputs), ::c10d::ReduceOp(::c10d::ReduceOp::SUM)};
    } else {
      TORCH_CHECK(false, "Unsupported type for PREMUL_SUM scaling");
    }
  } else if (op == ::c10d::ReduceOp::AVG) {
    return {input, ::c10d::ReduceOp(::c10d::ReduceOp::SUM)};
  }
  return {input, op};
}

// Divisor applied after a SUM to emulate AVG. Integer tensors truncate so the
// result stays integral and rounds toward zero for negative values.
std::optional<std::string_view> avgRoundingMode(const at::Tensor& tensor) {
  if (c10::isIntegralType(tensor.scalar_type(), /*includeBool=*/false)) {
    return "trunc";
  }
  return std::nullopt;
}

// Allocate a [tensors.size(), *tensors[0].sizes()] staging buffer, optionally
// filled with the contents of `tensors`. torchcomms drove the copies through
// XpuApi::memcpyAsync; XpuApi is a driver-mock layer that was not ported, so
// the stream-ordered ATen copy is used instead.
at::Tensor createFlattenedTensor(
    const std::vector<at::Tensor>& tensors,
    c10::xpu::XPUStream stream,
    bool copy_data = false) {
  TORCH_CHECK(!tensors.empty(), "tensors list is empty");
  const auto numel = tensors[0].numel();
  const auto sizes = tensors[0].sizes();
  for (const auto& tensor : tensors) {
    TORCH_CHECK(
        tensor.numel() == numel,
        "All tensors must have the same number of elements");
    TORCH_CHECK(
        tensor.sizes() == sizes, "All tensors must have the same shape");
  }

  c10::StreamGuard guard(stream);

  auto shape = tensors[0].sizes().vec();
  shape.insert(shape.begin(), static_cast<int64_t>(tensors.size()));
  at::Tensor flattened_tensor = at::empty(shape, tensors[0].options());

  if (copy_data) {
    for (size_t i = 0; i < tensors.size(); ++i) {
      flattened_tensor[static_cast<int64_t>(i)].copy_(
          tensors[i], /*non_blocking=*/true);
    }
  }

  return flattened_tensor;
}

void copyFlattenedTensorToTensors(
    const at::Tensor& flattened_tensor,
    const std::vector<at::Tensor>& tensors,
    c10::xpu::XPUStream stream) {
  TORCH_CHECK(!tensors.empty(), "tensors list is empty");
  const auto numel = tensors[0].numel();
  for (const auto& tensor : tensors) {
    TORCH_CHECK(
        tensor.numel() == numel,
        "All tensors must have the same number of elements");
  }
  TORCH_CHECK(
      static_cast<size_t>(flattened_tensor.numel()) == tensors.size() * numel,
      "Flattened tensor size does not match the total number of elements in tensors");

  c10::StreamGuard guard(stream);
  for (size_t i = 0; i < tensors.size(); ++i) {
    tensors[i].copy_(
        flattened_tensor[static_cast<int64_t>(i)], /*non_blocking=*/true);
  }
}

} // namespace

XCCLException::XCCLException(
    XcclApi& api,
    const std::string& message,
    onecclResult_t result)
    : message_(
          message + ": " + std::string(api.getErrorString(result)) + " (" +
          std::to_string(static_cast<int>(result)) + ")"),
      result_(result) {}

const char* XCCLException::what() const noexcept {
  return message_.c_str();
}

onecclResult_t XCCLException::getResult() const noexcept {
  return result_;
}

ProcessGroupXCCL::~ProcessGroupXCCL() {
  if (init_state_ == InitializationState::INITIALIZED) {
    TC_LOG(WARNING, this)
        << "ProcessGroupXCCL " << name_
        << " was not finalized before destruction. "
        << "This may indicate a resource leak. Please call finalize() explicitly.";

    // The allocator hook holds a raw pointer to this backend; drop it before
    // anything else so a concurrent allocator event cannot reach a dead object.
    detachMemoryHook();

    // Signal shutdown to timeout watchdog thread to prevent it from accessing
    // this object after destruction
    shutdown_ = true;
    // Wake up the timeout watchdog thread
    {
      std::lock_guard<std::mutex> lock(timeout_mutex_);
      timeout_cv_.notify_all();
    }

    // Wait for timeout thread to finish. If we're being called from within
    // the timeout thread itself (e.g., garbageCollect popped a work item whose
    // destruction released the last reference to this backend), we must detach
    // instead of join to avoid a deadlock.
    if (timeout_thread_.joinable()) {
      if (std::this_thread::get_id() != timeout_thread_.get_id()) {
        timeout_thread_.join();
      } else {
        timeout_thread_.detach(); // NOLINT(facebook-hte-BadCall-detach)
      }
    }

    if (xccl_comm_) {
      if (xccl_api_) {
        // Use destroy to free resources since oneCCL doesn't have abort api
        XCCL_CHECK_IGNORE(
            xccl_api_,
            xccl_api_->commDestroy(xccl_comm_),
            "Failed to destroy XCCL communicator");
      }
      xccl_comm_ = nullptr;
    }
  }
}

void ProcessGroupXCCL::init(at::Device device) {
  TC_LOG(INFO, this) << "Initializing ProcessGroupXCCL for device: " << device;
  device_ = device;

  TORCH_CHECK(
      init_state_ != InitializationState::INITIALIZED,
      "ProcessGroupXCCL already initialized");
  TORCH_CHECK(
      init_state_ != InitializationState::FINALIZED,
      "ProcessGroupXCCL already finalized");

  if (!xccl_api_) {
    xccl_api_ = std::make_shared<DefaultXcclApi>();
  }

  if (device_.index() == -1 || xccl_comm_ == nullptr) {
    auto bootstrap = std::make_unique<XCCLBootstrap>(
        store_,
        device_,
        getRank(),
        getSize(),
        bootstrap_generation_++,
        xccl_api_,
        options_c10d_->timeout);
    device_ = bootstrap->getDevice();

    if (xccl_comm_ == nullptr) {
      onecclConfig_t config = ONECCL_CONFIG_INITIALIZER;
      populateXcclConfigFromHints(
          config, options_c10d_->hints, std::string(kBackendName));
      xccl_comm_ = bootstrap->createXcclComm(name_, config);
    }
  }

  initXcclResources();

  init_state_ = InitializationState::INITIALIZED;
  TracingGuard tracingGuard(name_, comm_size_, "init", rank_, sequence_number_);

  TC_LOG(INFO, this) << "ProcessGroupXCCL initialized for rank: " << rank_;
}

void ProcessGroupXCCL::initFromComm(
    onecclComm_t comm,
    at::Device device,
    std::shared_ptr<XcclApi> xccl_api) {
  TORCH_CHECK(
      init_state_ == InitializationState::UNINITIALIZED,
      "ProcessGroupXCCL already initialized or finalized");

  xccl_api_ = std::move(xccl_api);
  xccl_comm_ = comm;
  device_ = device;

  initXcclResources();

  init_state_ = InitializationState::INITIALIZED;
  TracingGuard tracingGuard(name_, comm_size_, "init", rank_, sequence_number_);

  TC_LOG(INFO, this) << "ProcessGroupXCCL initialized from split for rank: "
                     << rank_;
}

c10::intrusive_ptr<::c10d::Backend> ProcessGroupXCCL::split(
    const c10::intrusive_ptr<::c10d::Store>& store,
    const std::vector<int>& ranks,
    const c10::intrusive_ptr<::c10d::Backend::Options>& opts) {
  std::unordered_set<int> seen;
  for (int r : ranks) {
    TORCH_CHECK(
        r >= 0 && r < getSize(),
        "ProcessGroupXCCL::split: rank ",
        r,
        " is out of range [0, ",
        getSize(),
        ")");
    TORCH_CHECK(
        seen.insert(r).second,
        "ProcessGroupXCCL::split: rank ",
        r,
        " appears more than once in ranks");
  }

  auto xcclOpts = c10::dynamic_intrusive_pointer_cast<Options>(opts);
  TORCH_CHECK(
      xcclOpts != nullptr,
      "ProcessGroupXCCL::split requires ProcessGroupXCCL::Options");

  // onecclCommSplit is collective over the parent communicator, so the parent
  // must be initialized before we split.
  if (init_state_ != InitializationState::INITIALIZED) {
    auto bound = getBoundDeviceId();
    ensureInitialized(
        bound.has_value()
            ? *bound
            : at::Device(
                  at::kXPU,
                  static_cast<c10::DeviceIndex>(c10::xpu::current_device())));
  }
  checkAndAbortIfTimedOutOrError();

  int color = ONECCL_SPLIT_NOCOLOR;
  int newRank = -1;
  auto it = std::find(ranks.begin(), ranks.end(), getRank());
  if (it != ranks.end()) {
    color = *std::min_element(ranks.begin(), ranks.end());
    newRank = static_cast<int>(std::distance(ranks.begin(), it));
  }

  onecclConfig_t config = ONECCL_CONFIG_INITIALIZER;
  populateXcclConfigFromHints(
      config, xcclOpts->hints, std::string(kBackendName));

  c10::OptionalDeviceGuard deviceGuard(device_);
  onecclComm_t new_comm = nullptr;
  // oneCCL rejects a negative key, so ranks outside the new group pass their
  // parent rank; the key is ignored for ONECCL_SPLIT_NOCOLOR.
  const int key = newRank >= 0 ? newRank : getRank();
  XCCL_CHECK(
      xccl_api_,
      xccl_api_->commSplit(xccl_comm_, color, key, &new_comm, &config),
      "XCCL commSplit failed");

  // Ranks outside the new group get no child backend.
  if (newRank == -1) {
    return nullptr;
  }

  auto childOpts = Options::create(xcclOpts->is_high_priority_stream);
  childOpts->timeout = xcclOpts->timeout;
  childOpts->hints = xcclOpts->hints;
  childOpts->group_name = xcclOpts->group_name;
  childOpts->group_desc = xcclOpts->group_desc;

  // The child's membership in world-rank terms. Map through this group's own
  // map, not the passed-in Options': an empty map means "spans the world in
  // rank order", and a merged group inherits its parent's shorter map, so
  // treat anything too short the same way instead of indexing out of bounds.
  const auto& parentRanks = options_c10d_->global_ranks_in_group;
  const bool parentSpansWorld =
      parentRanks.size() < static_cast<size_t>(getSize());
  std::vector<uint64_t> childRanks;
  childRanks.reserve(ranks.size());
  for (int r : ranks) {
    childRanks.push_back(
        parentSpansWorld ? static_cast<uint64_t>(r) : parentRanks[r]);
  }
  childOpts->global_ranks_in_group = std::move(childRanks);

  auto child = c10::make_intrusive<ProcessGroupXCCL>(
      store, newRank, static_cast<int>(ranks.size()), childOpts);
  child->initFromComm(new_comm, device_, xccl_api_);
  return c10::static_intrusive_pointer_cast<::c10d::Backend>(child);
}

void ProcessGroupXCCL::initXcclResources() {
  c10::OptionalDeviceGuard deviceGuard(device_);

  is_high_priority_stream_ = options_c10d_->is_high_priority_stream;

  if (!internal_stream_) {
    internal_stream_.emplace(
        c10::xpu::getStreamFromPool(is_high_priority_stream_, device_.index()));
  }

  if (!dependency_event_) {
    dependency_event_.emplace();
  }

  if (!barrier_buffer_) {
    barrier_buffer_ =
        c10::xpu::XPUCachingAllocator::get()->allocate(sizeof(float));
  }

  XCCL_CHECK(
      xccl_api_,
      xccl_api_->commUserRank(xccl_comm_, &rank_),
      "XCCL commUserRank failed");

  XCCL_CHECK(
      xccl_api_,
      xccl_api_->commCount(xccl_comm_, &comm_size_),
      "XCCL commCount failed");

  xccl_api_->setVersionInfo();
  TC_LOG(INFO, this) << "XCCL Version: " << xccl_api_->getVersion()
                     << " (Major: " << xccl_api_->getMajorVersion()
                     << " Minor: " << xccl_api_->getMinorVersion()
                     << " Patch: " << xccl_api_->getPatchVersion() << ")";

  if (!shutdown_) {
    timeout_thread_ = std::thread(&ProcessGroupXCCL::timeoutWatchdog, this);
  }

  attachMemoryHook();
}

[[noreturn]] void ProcessGroupXCCL::throwAsyncError(bool abort_comm) {
  onecclResult_t asyncErr = onecclSuccess;
  onecclResult_t res = xccl_api_->commGetAsyncError(xccl_comm_, &asyncErr);
  if (res != onecclSuccess) {
    TC_LOG(WARNING) << "commGetAsyncError returned " << res;
    asyncErr = res;
  }
  if (asyncErr == onecclSuccess) {
    asyncErr = onecclSystemError;
  }
  XCCLException xcclException(*xccl_api_, "XCCL Async Error", asyncErr);
  if (abort_comm) {
    abortXcclComm();
  } else if (abort_process_on_timeout_or_error_) {
    TC_LOG(ERROR) << "Aborting process due to error: " << xcclException.what();
    runAbortHooks();
    ::abort();
  }
  // @allow-raw-throw: its what() is logged above before it is thrown
  throw std::move(xcclException);
}

void ProcessGroupXCCL::abort() {
  abortXcclComm();
  comm_state_ = CommState::ERROR;
}

::c10d::ErrorType ProcessGroupXCCL::getError() {
  switch (comm_state_.load()) {
    case CommState::TIMEOUT:
      return ::c10d::ErrorType::TIMEOUT;
    case CommState::ERROR:
      return ::c10d::ErrorType::COMM_ERROR;
    default:
      return ::c10d::ErrorType::SUCCESS;
  }
}

void ProcessGroupXCCL::finalize() {
  TORCH_CHECK(
      init_state_ != InitializationState::UNINITIALIZED,
      "ProcessGroupXCCL not initialized");
  TORCH_CHECK(
      init_state_ != InitializationState::FINALIZED,
      "ProcessGroupXCCL already finalized");
  init_state_ = InitializationState::FINALIZED;

  // Drop the caching-allocator registrations first: the hook holds a raw
  // pointer to this backend and can fire from an allocator thread.
  detachMemoryHook();

  // Signal shutdown to timeout watchdog
  shutdown_ = true;

  // Wake up the timeout watchdog thread
  {
    std::lock_guard<std::mutex> lock(timeout_mutex_);
    timeout_cv_.notify_all();
  }

  // Wait for timeout thread to finish
  if (timeout_thread_.joinable()) {
    timeout_thread_.join();
  }

  // Wait for all pending work objects to complete and get final status
  auto work_status = workq_.finalize();

  if (comm_state_ == CommState::ERROR) {
    throwAsyncError(true);
  }

  if (comm_state_ == CommState::TIMEOUT) {
    abortXcclComm();
    TORCH_CHECK(false, "Work timed out during finalize");
  }

  TORCH_CHECK(
      work_status != WorkXCCL::WorkStatus::NOT_STARTED &&
          work_status != WorkXCCL::WorkStatus::INPROGRESS,
      "WorkQ finalize returned in progress or not started state");

  // Update comm_state_ based on the work status
  if (work_status == WorkXCCL::WorkStatus::TIMEDOUT) {
    comm_state_ = CommState::TIMEOUT;
    abortXcclComm();
    TORCH_CHECK(false, "Work timed out during finalize");
  } else if (work_status == WorkXCCL::WorkStatus::ERROR) {
    comm_state_ = CommState::ERROR;
    throwAsyncError(true);
  }

  // Clean up event pool
  event_pool_->clear();

  barrier_buffer_.clear();

  dependency_event_.reset();
  internal_stream_.reset();

  // Destroy XCCL communicator.
  // TODO: should probably not call this after calling abort.
  if (xccl_comm_) {
    XCCL_CHECK_IGNORE(
        xccl_api_,
        xccl_api_->commDestroy(xccl_comm_),
        "XCCL commDestroy failed");
    xccl_comm_ = nullptr;
  }
}

void ProcessGroupXCCL::abortXcclComm() {
  // Deregister allocator segments while the communicator is still valid.
  detachMemoryHook();

  if (xccl_comm_) {
    // NOTE: oneCCL has no working abort -- XcclApi::commAbort returns
    // onecclNotImplemented because onecclCommAbort is declared
    // CCL_C_NOT_IMPLEMENTED. The communicator is dropped here without being
    // torn down, so peers blocked in a collective stay blocked.
    onecclResult_t res = xccl_api_->commAbort(xccl_comm_);
    if (res != onecclSuccess) {
      TC_LOG(WARNING) << "commAbort returned " << res;
    }
    xccl_comm_ = nullptr;
  }
  if (abort_process_on_timeout_or_error_) {
    TC_LOG(ERROR, this) << "Aborting process due to error or timeout";
    runAbortHooks();
    ::abort();
  }
}

int64_t ProcessGroupXCCL::getCommPtr() const {
  return reinterpret_cast<int64_t>(xccl_comm_);
}

void ProcessGroupXCCL::checkTensorDevice(const at::Tensor& tensor) const {
  TORCH_CHECK(
      tensor.device() == device_,
      "Tensor is on device " + tensor.device().str() +
          " but this ProcessGroupXCCL is bound to device " + device_.str());
}

void ProcessGroupXCCL::checkTensorsDevice(
    const std::vector<at::Tensor>& tensors) const {
  for (const auto& tensor : tensors) {
    checkTensorDevice(tensor);
  }
}

// Point-to-Point Operations
c10::intrusive_ptr<WorkXCCL> ProcessGroupXCCL::sendImpl(
    const at::Tensor& tensor,
    int dst,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);
  checkTensorDevice(tensor);

  TracingGuard tracingGuard(
      name_, comm_size_, "send", dst, sequence_number_, tensor, tensor);

  c10::xpu::XPUStream stream = getOperationStream(async_op);
  auto work = async_op ? createWork(stream, timeout, tensor)
                       : createWork(stream, timeout);

  work->recordStart("send");

  if (dst == rank_) [[unlikely]] {
    throw XCCLException(
        *xccl_api_,
        "XCCL send called with destination rank equal to current rank " +
            std::to_string(rank_) + "; operation would hang",
        onecclInvalidUsage);
  }

  XCCL_CHECK(
      xccl_api_,
      xccl_api_->send(
          tensor.data_ptr(),
          tensor.numel(),
          getXcclDataType(tensor),
          dst,
          xccl_comm_,
          stream),
      "XCCL send failed");

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkXCCL> ProcessGroupXCCL::recvImpl(
    at::Tensor& tensor,
    int src,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);
  checkTensorDevice(tensor);

  TracingGuard tracingGuard(
      name_, comm_size_, "recv", src, sequence_number_, tensor, tensor);

  c10::xpu::XPUStream stream = getOperationStream(async_op);
  auto work = createWork(stream, timeout);

  work->recordStart("recv");

  if (src == rank_) [[unlikely]] {
    throw XCCLException(
        *xccl_api_,
        "XCCL recv called with source rank equal to current rank " +
            std::to_string(rank_) + "; operation would hang",
        onecclInvalidUsage);
  }

  XCCL_CHECK(
      xccl_api_,
      xccl_api_->recv(
          tensor.data_ptr(),
          tensor.numel(),
          getXcclDataType(tensor),
          src,
          xccl_comm_,
          stream),
      "XCCL recv failed");

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

// Batch P2P Operations
c10::intrusive_ptr<WorkXCCL> ProcessGroupXCCL::batch_op_issue(
    const std::vector<BatchSendRecv::P2POp>& ops,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();

  TORCH_CHECK(!ops.empty(), "Cannot issue empty batch operation");

  // Collect input and output tensors for work tracking
  std::vector<at::Tensor> input_tensors;
  std::vector<at::Tensor> output_tensors;

  for (const auto& op : ops) {
    ensureTensorContiguous(op.tensor);
    checkTensorDevice(op.tensor);
    if (op.type == BatchSendRecv::P2POp::OpType::SEND) {
      input_tensors.push_back(op.tensor);
    } else if (op.type == BatchSendRecv::P2POp::OpType::RECV) {
      output_tensors.push_back(op.tensor);
    } else {
      TORCH_CHECK(false, "Unknown op type in batch_op_issue");
    }
  }

  TracingGuard tracingGuard(
      name_,
      comm_size_,
      "batch_op_issue",
      rank_,
      sequence_number_,
      input_tensors,
      output_tensors);

  c10::xpu::XPUStream stream = getOperationStream(async_op);
  auto work = createWork(stream, timeout, input_tensors);

  work->recordStart("batch_op_issue");

  XCCL_CHECK(
      xccl_api_,
      xccl_api_->groupStart(),
      "XCCL groupStart failed in batch_op_issue");

  // Issue all sends before any recvs within the group. WORKAROUND for a oneCCL
  // grouped-P2P deadlock: group ops are replayed in enqueue order and a recv
  // blocks in a host-side IPC handle exchange until its peer's matching send is
  // reached, so recv-before-send on both peers (e.g. batch_isend_irecv with
  // irecv first) deadlocks. Sends first avoids this; relative order within
  // sends and within recvs is preserved.
  for (const auto& op : ops) {
    if (op.type != BatchSendRecv::P2POp::OpType::SEND) {
      continue;
    }

    auto result = xccl_api_->send(
        op.tensor.data_ptr(),
        op.tensor.numel(),
        getXcclDataType(op.tensor),
        op.peer,
        xccl_comm_,
        stream);

    if (result != onecclSuccess) [[unlikely]] {
      XCCL_CHECK_IGNORE(
          xccl_api_,
          xccl_api_->groupEnd(), // Clean up group on error
          "XCCL groupEnd failed during error cleanup after send failure in batch_op_issue");
      throw XCCLException(
          *xccl_api_, "XCCL send failed in batch_op_issue", result);
    }
  }

  for (const auto& op : ops) {
    if (op.type != BatchSendRecv::P2POp::OpType::RECV) {
      continue;
    }

    auto result = xccl_api_->recv(
        op.tensor.data_ptr(),
        op.tensor.numel(),
        getXcclDataType(op.tensor),
        op.peer,
        xccl_comm_,
        stream);

    if (result != onecclSuccess) [[unlikely]] {
      XCCL_CHECK_IGNORE(
          xccl_api_,
          xccl_api_->groupEnd(), // Clean up group on error
          "XCCL groupEnd failed during error cleanup after recv failure in batch_op_issue");
      throw XCCLException(
          *xccl_api_, "XCCL recv failed in batch_op_issue", result);
    }
  }

  XCCL_CHECK(
      xccl_api_,
      xccl_api_->groupEnd(),
      "XCCL groupEnd failed in batch_op_issue");

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

// Collective Operations
c10::intrusive_ptr<WorkXCCL> ProcessGroupXCCL::broadcastImpl(
    at::Tensor& tensor,
    int root,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);
  checkTensorDevice(tensor);

  TracingGuard tracingGuard(
      name_, comm_size_, "broadcast", root, sequence_number_, tensor, tensor);

  c10::xpu::XPUStream stream = getOperationStream(async_op);

  auto work = async_op ? createWork(stream, timeout, tensor)
                       : createWork(stream, timeout);

  work->recordStart("broadcast");

  const auto dataType = getXcclDataType(tensor);
  XCCL_CHECK(
      xccl_api_,
      xccl_api_->broadcast(
          tensor.data_ptr(),
          tensor.data_ptr(),
          tensor.numel(),
          dataType,
          root,
          xccl_comm_,
          stream),
      "XCCL broadcast failed");

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkXCCL> ProcessGroupXCCL::all_reduce(
    at::Tensor& tensor,
    const ::c10d::ReduceOp& op,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);
  checkTensorDevice(tensor);

  TracingGuard tracingGuard(
      name_, comm_size_, "all_reduce", rank_, sequence_number_, tensor, tensor);

  c10::xpu::XPUStream stream = getOperationStream(async_op);
  auto work = async_op ? createWork(stream, timeout, tensor)
                       : createWork(stream, timeout);

  work->recordStart("all_reduce");

  // all_reduce is in place, so PREMUL_SUM scales the caller's buffer directly
  // rather than a clone. See getMaybeScaledInputsAndNewOp for the oneCCL
  // limitation this works around.
  const auto maybe_new_op = [&]() -> ::c10d::ReduceOp {
    if (op == ::c10d::ReduceOp::PREMUL_SUM) {
      c10::StreamGuard guard(stream);
      applyPreMulFactor(tensor, op);
      return ::c10d::ReduceOp(::c10d::ReduceOp::SUM);
    } else if (op == ::c10d::ReduceOp::AVG) {
      return ::c10d::ReduceOp(::c10d::ReduceOp::SUM);
    }
    return op;
  }();

  const auto dataType = getXcclDataType(tensor);
  XCCL_CHECK(
      xccl_api_,
      xccl_api_->allReduce(
          tensor.data_ptr(),
          tensor.data_ptr(), // In-place operation
          tensor.numel(),
          dataType,
          getXcclReduceOp(maybe_new_op, xccl_comm_, dataType),
          xccl_comm_,
          stream),
      "XCCL allReduce failed");

  if (op == ::c10d::ReduceOp::AVG) {
    // oneCCL has no native AVG, so every rank divides the SUM by comm_size.
    c10::StreamGuard guard(stream);
    tensor.div_(comm_size_, avgRoundingMode(tensor));
  }

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkXCCL> ProcessGroupXCCL::reduceImpl(
    const at::Tensor& tensor,
    int root,
    const ::c10d::ReduceOp& op,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);
  checkTensorDevice(tensor);

  TracingGuard tracingGuard(
      name_, comm_size_, "reduce", root, sequence_number_, tensor, tensor);

  c10::xpu::XPUStream stream = getOperationStream(async_op);

  auto work = async_op ? createWork(stream, timeout, tensor)
                       : createWork(stream, timeout);

  work->recordStart("reduce");

  const auto [maybe_scaled_tensor, maybe_new_op] =
      getMaybeScaledInputsAndNewOp(tensor, op, stream);

  const auto dataType = getXcclDataType(tensor);
  XCCL_CHECK(
      xccl_api_,
      xccl_api_->reduce(
          maybe_scaled_tensor.data_ptr(),
          tensor.data_ptr(),
          tensor.numel(),
          dataType,
          getXcclReduceOp(maybe_new_op, xccl_comm_, dataType),
          root,
          xccl_comm_,
          stream),
      "XCCL reduce failed");

  if (rank_ == root && op == ::c10d::ReduceOp::AVG) {
    c10::StreamGuard guard(stream);
    maybe_scaled_tensor.div_(comm_size_, avgRoundingMode(maybe_scaled_tensor));
  }

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

} // namespace c10d::xccl2

#endif // USE_C10D_XCCL
