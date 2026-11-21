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

// AVG is lowered to SUM plus a div_ that fails for bool only after the
// collective has run, so reject it up front.
void checkReduceOpDtype(const at::Tensor& tensor, const ::c10d::ReduceOp& op) {
  TORCH_CHECK_TYPE(
      tensor.scalar_type() != at::kBool || op != ::c10d::ReduceOp::AVG,
      "Cannot use ReduceOp.AVG with boolean inputs");
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
// TODO: oneCCL recent release added the PREMUL_SUM and AVG native support.
// However, the AVG has some accuracy issues, which are being investigated.
// Both of these implementations are planned to be evaluated after XCCL2
// landing, for both performance and functionality.
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

    // Stop the watchdog so it cannot access this object after destruction.
    stopWatchdog();

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
    abortProcess(std::string("error: ") + xcclException.what());
  }
  // @allow-raw-throw: its what() is logged above before it is thrown
  throw std::move(xcclException);
}

bool ProcessGroupXCCL::isInitialized() {
  return init_state_ == InitializationState::INITIALIZED &&
      xccl_comm_ != nullptr;
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

  // Stop the watchdog first
  stopWatchdog();

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
}

void ProcessGroupXCCL::stopWatchdog() {
  shutdown_ = true;
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
}

[[noreturn]] void ProcessGroupXCCL::abortProcess(const std::string& reason) {
  TC_LOG(ERROR, this) << "Aborting process on rank " << rank_ << " due to "
                      << reason;
  runAbortHooks();
  ::abort();
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
  try {
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
        throw XCCLException(
            *xccl_api_, "XCCL recv failed in batch_op_issue", result);
      }
    }
  } catch (...) {
    XCCL_CHECK_IGNORE(
        xccl_api_,
        xccl_api_->groupEnd(),
        "XCCL groupEnd failed during error cleanup in batch_op_issue");
    throw;
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
  checkReduceOpDtype(tensor, op);

  TracingGuard tracingGuard(
      name_, comm_size_, "all_reduce", rank_, sequence_number_, tensor, tensor);

  c10::xpu::XPUStream stream = getOperationStream(async_op);
  auto work = async_op ? createWork(stream, timeout, tensor)
                       : createWork(stream, timeout);

  work->recordStart("all_reduce");

  // all_reduce is in place, so PREMUL_SUM scales the caller's buffer directly
  // rather than a clone. See getMaybeScaledInputsAndNewOp for the oneCCL
  // limitation this works around.
  // TODO: oneCCL recent release added the PREMUL_SUM and AVG native support.
  // Both of these native implementations are planned to be evaluated after
  // XCCL2 landing, for both performance and functionality.
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
          getXcclReduceOp(maybe_new_op, xccl_comm_, tensor),
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
  checkReduceOpDtype(tensor, op);

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
          getXcclReduceOp(maybe_new_op, xccl_comm_, tensor),
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

c10::intrusive_ptr<WorkXCCL> ProcessGroupXCCL::all_gather(
    const std::vector<at::Tensor>& tensor_list,
    const at::Tensor& tensor,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);

  TORCH_CHECK(
      tensor_list.size() == static_cast<size_t>(comm_size_),
      "tensor_list size must equal comm_size for all_gather");

  // Sizes may differ per rank (the all_gather_v case); only this rank's slot
  // has to match what we contribute.
  bool uneven = false;
  for (const auto& t : tensor_list) {
    ensureTensorContiguous(t);
    uneven = uneven || t.numel() != tensor.numel();
  }

  TORCH_CHECK(
      tensor_list[rank_].numel() == tensor.numel(),
      "all_gather: output tensor size at this rank must match input tensor "
      "size");

  checkTensorDevice(tensor);
  checkTensorsDevice(tensor_list);
  checkSameDtype(tensor, tensor_list);

  TracingGuard tracingGuard(
      name_,
      comm_size_,
      "all_gather",
      rank_,
      sequence_number_,
      tensor_list,
      {tensor});

  c10::xpu::XPUStream stream = getOperationStream(async_op);

  auto work = async_op ? createWork(stream, timeout, tensor)
                       : createWork(stream, timeout);

  work->recordStart("all_gather");

  if (uneven) {
    // oneCCL has no all_gatherv, so an uneven gather is issued as one grouped
    // broadcast per rank.
    XCCL_CHECK(
        xccl_api_,
        xccl_api_->groupStart(),
        "XCCL groupStart failed in all_gather");

    for (int i = 0; i < comm_size_; ++i) {
      const at::Tensor& output = tensor_list[i];
      // oneCCL wants a valid send buffer on every rank, so non-root ranks
      // point at their own output; only the root's data is broadcast.
      const at::Tensor& input = (i == rank_) ? tensor : output;

      if (output.numel() == 0) [[unlikely]] {
        continue;
      }

      onecclResult_t result = xccl_api_->broadcast(
          input.data_ptr(),
          output.data_ptr(),
          output.numel(),
          getXcclDataType(output),
          i,
          xccl_comm_,
          stream);
      if (result != onecclSuccess) [[unlikely]] {
        XCCL_CHECK_IGNORE(
            xccl_api_,
            xccl_api_->groupEnd(),
            "XCCL groupEnd failed while unwinding all_gather");
        throw XCCLException(
            *xccl_api_, "XCCL broadcast failed in all_gather", result);
      }
    }

    XCCL_CHECK(
        xccl_api_, xccl_api_->groupEnd(), "XCCL groupEnd failed in all_gather");
  } else {
    at::Tensor output_flattened = createFlattenedTensor(tensor_list, stream);

    XCCL_CHECK(
        xccl_api_,
        xccl_api_->allGather(
            tensor.data_ptr(),
            output_flattened.data_ptr(),
            tensor.numel(),
            getXcclDataType(tensor),
            xccl_comm_,
            stream),
        "XCCL allGather failed in all_gather");

    if (output_flattened.numel() != 0) [[likely]] {
      copyFlattenedTensorToTensors(output_flattened, tensor_list, stream);
    }
  }

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkXCCL> ProcessGroupXCCL::allGatherSingleImpl(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);
  ensureTensorContiguous(input);
  checkTensorDevice(output);
  checkTensorDevice(input);
  checkSameDtype(input, output);

  TORCH_CHECK(
      output.numel() == input.numel() * comm_size_,
      "Output tensor size must be input_size * comm_size for allGatherSingleImpl");

  TracingGuard tracingGuard(
      name_,
      comm_size_,
      "allGatherSingleImpl",
      rank_,
      sequence_number_,
      input,
      output);

  c10::xpu::XPUStream stream = getOperationStream(async_op);

  auto work = async_op ? createWork(stream, timeout, input)
                       : createWork(stream, timeout);

  work->recordStart("allGatherSingleImpl");

  XCCL_CHECK(
      xccl_api_,
      xccl_api_->allGather(
          input.data_ptr(),
          output.data_ptr(),
          input.numel(),
          getXcclDataType(input),
          xccl_comm_,
          stream),
      "XCCL allGather failed in allGatherSingleImpl");

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkXCCL> ProcessGroupXCCL::reduce_scatter(
    at::Tensor& output,
    const std::vector<at::Tensor>& input_list,
    const ::c10d::ReduceOp& op,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);
  checkReduceOpDtype(output, op);

  TORCH_CHECK(
      input_list.size() == static_cast<size_t>(comm_size_),
      "reduce_scatter: input_list size must equal comm_size");

  bool uneven = false;
  for (const auto& t : input_list) {
    ensureTensorContiguous(t);
    uneven = uneven || t.numel() != output.numel();
  }

  TORCH_CHECK(
      input_list[rank_].numel() == output.numel(),
      "reduce_scatter: input tensor size at this rank must match output "
      "tensor size");

  checkTensorDevice(output);
  checkTensorsDevice(input_list);
  checkSameDtype(output, input_list);

  TracingGuard tracingGuard(
      name_,
      comm_size_,
      "reduce_scatter",
      rank_,
      sequence_number_,
      input_list,
      {output});

  c10::xpu::XPUStream stream = getOperationStream(async_op);

  auto work = async_op ? createWork(stream, timeout, input_list)
                       : createWork(stream, timeout);

  work->recordStart("reduce_scatter");

  if (!uneven && output.numel() == 0) [[unlikely]] {
    work->recordEnd();
    enqueueWork(work, stream);
    return work;
  }

  const auto [maybe_scaled_input_list, maybe_new_op] =
      getMaybeScaledInputsAndNewOp(input_list, op, stream);

  if (uneven) {
    // oneCCL has no reduce_scatterv, so an uneven scatter is issued as one
    // grouped reduce per rank.
    XCCL_CHECK(
        xccl_api_,
        xccl_api_->groupStart(),
        "XCCL groupStart failed in reduce_scatter");

    for (int i = 0; i < comm_size_; ++i) {
      const auto& input_tensor = maybe_scaled_input_list[i];
      if (input_tensor.numel() == 0) [[unlikely]] {
        continue;
      }
      const auto dataType = getXcclDataType(input_tensor);
      // oneCCL hangs when a non-root rank passes a null receive buffer, so
      // non-root ranks point at their own input; oneCCL only writes on root.
      void* recv_ptr =
          (i == rank_) ? output.data_ptr() : input_tensor.data_ptr();
      onecclResult_t result = xccl_api_->reduce(
          input_tensor.data_ptr(),
          recv_ptr,
          input_tensor.numel(),
          dataType,
          getXcclReduceOp(maybe_new_op, xccl_comm_, input_tensor),
          i,
          xccl_comm_,
          stream);
      if (result != onecclSuccess) [[unlikely]] {
        XCCL_CHECK_IGNORE(
            xccl_api_,
            xccl_api_->groupEnd(),
            "XCCL groupEnd failed during error cleanup after reduce failure in reduce_scatter");
        throw XCCLException(
            *xccl_api_, "XCCL reduce failed in reduce_scatter", result);
      }
    }

    XCCL_CHECK(
        xccl_api_,
        xccl_api_->groupEnd(),
        "XCCL groupEnd failed in reduce_scatter");
  } else {
    at::Tensor input_flattened = createFlattenedTensor(
        maybe_scaled_input_list, stream, /*copy_data=*/true);

    const auto dataType = getXcclDataType(output);

    XCCL_CHECK(
        xccl_api_,
        xccl_api_->reduceScatter(
            input_flattened.data_ptr(),
            output.data_ptr(),
            output.numel(),
            dataType,
            getXcclReduceOp(maybe_new_op, xccl_comm_, output),
            xccl_comm_,
            stream),
        "XCCL reduceScatter failed in reduce_scatter");
  }

  if (op == ::c10d::ReduceOp::AVG) {
    c10::StreamGuard guard(stream);
    output.div_(comm_size_, avgRoundingMode(output));
  }

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkXCCL> ProcessGroupXCCL::reduceScatterSingleImpl(
    at::Tensor& output,
    const at::Tensor& input,
    const ::c10d::ReduceOp& op,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);
  ensureTensorContiguous(input);
  checkTensorDevice(output);
  checkTensorDevice(input);
  checkSameDtype(input, output);
  checkReduceOpDtype(output, op);

  TORCH_CHECK(
      input.numel() == output.numel() * comm_size_,
      "Input tensor size must be output_size * comm_size for reduceScatterSingleImpl");

  TracingGuard tracingGuard(
      name_,
      comm_size_,
      "reduceScatterSingleImpl",
      rank_,
      sequence_number_,
      input,
      output);

  c10::xpu::XPUStream stream = getOperationStream(async_op);

  auto work = async_op ? createWork(stream, timeout, input)
                       : createWork(stream, timeout);

  work->recordStart("reduceScatterSingleImpl");

  const auto [maybe_scaled_input, maybe_new_op] =
      getMaybeScaledInputsAndNewOp(input, op, stream);

  const auto dataType = getXcclDataType(maybe_scaled_input);
  XCCL_CHECK(
      xccl_api_,
      xccl_api_->reduceScatter(
          maybe_scaled_input.data_ptr(),
          output.data_ptr(),
          output.numel(),
          dataType,
          getXcclReduceOp(maybe_new_op, xccl_comm_, maybe_scaled_input),
          xccl_comm_,
          stream),
      "XCCL reduceScatter failed in reduceScatterSingleImpl");

  if (op == ::c10d::ReduceOp::AVG) {
    c10::StreamGuard guard(stream);
    output.div_(comm_size_, avgRoundingMode(output));
  }

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkXCCL> ProcessGroupXCCL::allToAllSingleImpl(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);
  ensureTensorContiguous(input);
  checkTensorDevice(output);
  checkTensorDevice(input);
  checkSameDtype(input, output);

  TORCH_CHECK(
      input.numel() == output.numel(),
      "Input and output tensors must have same size for allToAllSingleImpl");

  TORCH_CHECK(
      input.numel() % comm_size_ == 0,
      "Tensor size must be divisible by comm_size for allToAllSingleImpl");

  TracingGuard tracingGuard(
      name_,
      comm_size_,
      "allToAllSingleImpl",
      rank_,
      sequence_number_,
      input,
      output);

  c10::xpu::XPUStream stream = getOperationStream(async_op);

  auto work = async_op ? createWork(stream, timeout, input)
                       : createWork(stream, timeout);

  work->recordStart("allToAllSingleImpl");

  size_t chunk_size = input.numel() / comm_size_;
  const auto dataType = getXcclDataType(input);

  XCCL_CHECK(
      xccl_api_,
      xccl_api_->allToAll(
          input.data_ptr(),
          output.data_ptr(),
          chunk_size,
          dataType,
          xccl_comm_,
          stream),
      "XCCL allToAll failed in allToAllSingleImpl");

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkXCCL> ProcessGroupXCCL::all_to_all_v_single(
    at::Tensor& output,
    const at::Tensor& input,
    const std::vector<uint64_t>& output_split_sizes,
    const std::vector<uint64_t>& input_split_sizes,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);
  ensureTensorContiguous(input);
  checkTensorDevice(output);
  checkTensorDevice(input);
  checkSameDtype(input, output);

  TORCH_CHECK(
      input_split_sizes.size() == static_cast<size_t>(comm_size_),
      "input_split_sizes length must equal comm_size for all_to_all_v_single");

  TORCH_CHECK(
      output_split_sizes.size() == static_cast<size_t>(comm_size_),
      "output_split_sizes length must equal comm_size for all_to_all_v_single");

  // Validate that split sizes sum does not exceed tensor dimensions
  uint64_t input_total = 0;
  uint64_t output_total = 0;
  for (int i = 0; i < comm_size_; ++i) {
    input_total += input_split_sizes[i];
    output_total += output_split_sizes[i];
  }

  TORCH_CHECK(
      input_total <= static_cast<uint64_t>(input.size(0)),
      "Sum of input_split_sizes exceeds input tensor size for all_to_all_v_single");

  TORCH_CHECK(
      output_total <= static_cast<uint64_t>(output.size(0)),
      "Sum of output_split_sizes exceeds output tensor size for all_to_all_v_single");

  TracingGuard tracingGuard(
      name_,
      comm_size_,
      "all_to_all_v_single",
      rank_,
      sequence_number_,
      input,
      output);

  c10::xpu::XPUStream stream = getOperationStream(async_op);
  auto work = async_op ? createWork(stream, timeout, input)
                       : createWork(stream, timeout);

  work->recordStart("all_to_all_v_single");

  std::vector<size_t> sendcounts(comm_size_);
  std::vector<size_t> recvcounts(comm_size_);
  std::vector<size_t> senddispls(comm_size_);
  std::vector<size_t> recvdispls(comm_size_);

  // Calculate the number of elements per slice along the first dimension
  // For a tensor with shape [N, D1, D2, ..., Dk], each slice of size S along
  // dim 0 contains S * D1 * D2 * ... * Dk elements
  // Use input tensor for send counts and output tensor for recv counts
  size_t send_elements_per_slice =
      input.numel() ? input.numel() / input.size(0) : 0;
  size_t recv_elements_per_slice =
      output.numel() ? output.numel() / output.size(0) : 0;
  const auto dataType = getXcclDataType(input);
  const size_t type_size = wordSize(dataType);

  size_t sendoffset = 0;
  size_t recvoffset = 0;
  // Row offsets along dim 0, used to build the self-copy views below.
  std::vector<int64_t> sendrows(comm_size_);
  std::vector<int64_t> recvrows(comm_size_);
  int64_t sendrow = 0;
  int64_t recvrow = 0;
  for (int i = 0; i < comm_size_; ++i) {
    sendcounts[i] = input_split_sizes[i] * send_elements_per_slice;
    recvcounts[i] = output_split_sizes[i] * recv_elements_per_slice;
    senddispls[i] = sendoffset;
    recvdispls[i] = recvoffset;
    sendoffset += sendcounts[i];
    recvoffset += recvcounts[i];
    sendrows[i] = sendrow;
    recvrows[i] = recvrow;
    sendrow += static_cast<int64_t>(input_split_sizes[i]);
    recvrow += static_cast<int64_t>(output_split_sizes[i]);
  }

  char* sptr = static_cast<char*>(input.data_ptr());
  char* rptr = static_cast<char*>(output.data_ptr());

  XCCL_CHECK(
      xccl_api_,
      xccl_api_->groupStart(),
      "XCCL groupStart failed in all_to_all_v_single");

  for (int i = 0; i < comm_size_; ++i) {
    if (sendcounts[i] == 0 && recvcounts[i] == 0) [[unlikely]] {
      continue;
    }
    if (sendcounts[i] > 0) {
      if (i == rank_) {
        // Own contribution never goes on the wire; copy it locally.
        c10::StreamGuard guard(stream);
        output
            .slice(
                0,
                recvrows[i],
                recvrows[i] + static_cast<int64_t>(output_split_sizes[i]))
            .copy_(
                input.slice(
                    0,
                    sendrows[i],
                    sendrows[i] + static_cast<int64_t>(input_split_sizes[i])),
                /*non_blocking=*/true);
        continue;
      } else {
        onecclResult_t result = xccl_api_->send(
            sptr + senddispls[i] * type_size,
            sendcounts[i],
            dataType,
            i,
            xccl_comm_,
            stream);
        if (result != onecclSuccess) [[unlikely]] {
          XCCL_CHECK_IGNORE(
              xccl_api_,
              xccl_api_->groupEnd(),
              "XCCL groupEnd failed during error cleanup after send failure in all_to_all_v_single");
          throw XCCLException(
              *xccl_api_, "XCCL send failed in all_to_all_v_single", result);
        }
      }
    }

    if (recvcounts[i] > 0) {
      onecclResult_t result = xccl_api_->recv(
          rptr + recvdispls[i] * type_size,
          recvcounts[i],
          dataType,
          i,
          xccl_comm_,
          stream);
      if (result != onecclSuccess) [[unlikely]] {
        XCCL_CHECK_IGNORE(
            xccl_api_,
            xccl_api_->groupEnd(),
            "XCCL groupEnd failed during error cleanup after recv failure in all_to_all_v_single");
        throw XCCLException(
            *xccl_api_, "XCCL recv failed in all_to_all_v_single", result);
      }
    }
  }

  XCCL_CHECK(
      xccl_api_,
      xccl_api_->groupEnd(),
      "XCCL groupEnd failed in all_to_all_v_single");

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkXCCL> ProcessGroupXCCL::all_to_all(
    const std::vector<at::Tensor>& output_tensor_list,
    const std::vector<at::Tensor>& input_tensor_list,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();

  TORCH_CHECK(
      output_tensor_list.size() == static_cast<size_t>(comm_size_) &&
          input_tensor_list.size() == static_cast<size_t>(comm_size_),
      "Tensor list sizes must equal comm_size for all_to_all");

  for (int i = 0; i < comm_size_; ++i) {
    ensureTensorContiguous(input_tensor_list[i]);
    ensureTensorContiguous(output_tensor_list[i]);
  }

  checkTensorsDevice(input_tensor_list);
  checkTensorsDevice(output_tensor_list);

  TracingGuard tracingGuard(
      name_,
      comm_size_,
      "all_to_all",
      rank_,
      sequence_number_,
      input_tensor_list,
      output_tensor_list);

  c10::xpu::XPUStream stream = getOperationStream(async_op);
  auto work = async_op ? createWork(stream, timeout, input_tensor_list)
                       : createWork(stream, timeout);

  work->recordStart("all_to_all");

  XCCL_CHECK(
      xccl_api_,
      xccl_api_->groupStart(),
      "XCCL groupStart failed in all_to_all");

  for (int i = 0; i < comm_size_; ++i) {
    if (input_tensor_list[i].numel() == 0) [[unlikely]] {
      continue;
    }
    if (i == rank_) {
      // Own contribution never goes on the wire; copy it locally.
      c10::StreamGuard guard(stream);
      output_tensor_list[i].copy_(input_tensor_list[i], /*non_blocking=*/true);
      continue;
    } else {
      onecclResult_t result = xccl_api_->send(
          input_tensor_list[i].data_ptr(),
          input_tensor_list[i].numel(),
          getXcclDataType(input_tensor_list[i]),
          i,
          xccl_comm_,
          stream);
      if (result != onecclSuccess) [[unlikely]] {
        XCCL_CHECK_IGNORE(
            xccl_api_,
            xccl_api_->groupEnd(),
            "XCCL groupEnd failed during error cleanup after send failure in all_to_all");
        throw XCCLException(
            *xccl_api_, "XCCL send failed in all_to_all", result);
      }
    }

    onecclResult_t result = xccl_api_->recv(
        output_tensor_list[i].data_ptr(),
        output_tensor_list[i].numel(),
        getXcclDataType(output_tensor_list[i]),
        i,
        xccl_comm_,
        stream);
    if (result != onecclSuccess) [[unlikely]] {
      XCCL_CHECK_IGNORE(
          xccl_api_,
          xccl_api_->groupEnd(),
          "XCCL groupEnd failed during error cleanup after recv failure in all_to_all");
      throw XCCLException(*xccl_api_, "XCCL recv failed in all_to_all", result);
    }
  }

  XCCL_CHECK(
      xccl_api_, xccl_api_->groupEnd(), "XCCL groupEnd failed in all_to_all");

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkXCCL> ProcessGroupXCCL::barrierImpl(
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();

  TracingGuard tracingGuard(
      name_, comm_size_, "barrier", rank_, sequence_number_);

  c10::xpu::XPUStream stream = getOperationStream(async_op);
  auto work = createWork(stream, timeout);

  work->recordStart("barrier");

  XCCL_CHECK(
      xccl_api_,
      xccl_api_->allReduce(
          barrier_buffer_.get(),
          barrier_buffer_.get(),
          1,
          onecclFloat32,
          onecclSum,
          xccl_comm_,
          stream),
      "XCCL barrier failed");

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkXCCL> ProcessGroupXCCL::scatterImpl(
    at::Tensor& output_tensor,
    const std::vector<at::Tensor>& input_tensor_list,
    int root,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output_tensor);
  checkTensorDevice(output_tensor);

  // Only the root rank needs valid tensors
  if (rank_ == root) {
    TORCH_CHECK(
        input_tensor_list.size() == static_cast<size_t>(comm_size_),
        "input_tensor_list must equal comm_size for scatter");

    for (const auto& t : input_tensor_list) {
      ensureTensorContiguous(t);
      TORCH_CHECK(
          t.numel() == output_tensor.numel(),
          "All input tensors must have same size as output tensor");
    }

    checkTensorsDevice(input_tensor_list);
    checkSameDtype(output_tensor, input_tensor_list);
  }

  TracingGuard tracingGuard(
      name_,
      comm_size_,
      "scatter",
      root,
      sequence_number_,
      input_tensor_list,
      {output_tensor});

  c10::xpu::XPUStream stream = getOperationStream(async_op);
  std::vector<at::Tensor> input_tensors;
  if (async_op && rank_ == root) {
    input_tensors = input_tensor_list;
  }
  auto work = createWork(stream, timeout, input_tensors);

  work->recordStart("scatter");

  // Grouped to avoid a oneCCL hang; see
  // https://github.com/uxlfoundation/oneCCL/issues/193.
  XCCL_CHECK(
      xccl_api_, xccl_api_->groupStart(), "XCCL groupStart failed in scatter");

  if (rank_ == root) {
    for (int i = 0; i < comm_size_; ++i) {
      if (i != root) {
        onecclResult_t result = xccl_api_->send(
            input_tensor_list[i].data_ptr(),
            input_tensor_list[i].numel(),
            getXcclDataType(input_tensor_list[i]),
            i,
            xccl_comm_,
            stream);
        if (result != onecclSuccess) [[unlikely]] {
          XCCL_CHECK_IGNORE(
              xccl_api_,
              xccl_api_->groupEnd(),
              "XCCL groupEnd failed during error cleanup after send failure in scatter");
          throw XCCLException(
              *xccl_api_, "XCCL send failed in scatter", result);
        }
      }
    }

    // Root's own chunk never goes on the wire; copy it locally.
    {
      c10::StreamGuard guard(stream);
      output_tensor.copy_(input_tensor_list[root], /*non_blocking=*/true);
    }
  } else {
    onecclResult_t result = xccl_api_->recv(
        output_tensor.data_ptr(),
        output_tensor.numel(),
        getXcclDataType(output_tensor),
        root,
        xccl_comm_,
        stream);
    if (result != onecclSuccess) [[unlikely]] {
      XCCL_CHECK_IGNORE(
          xccl_api_,
          xccl_api_->groupEnd(),
          "XCCL groupEnd failed during error cleanup after recv failure in scatter");
      throw XCCLException(*xccl_api_, "XCCL recv failed in scatter", result);
    }
  }

  XCCL_CHECK(
      xccl_api_, xccl_api_->groupEnd(), "XCCL groupEnd failed in scatter");

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkXCCL> ProcessGroupXCCL::gatherImpl(
    const std::vector<at::Tensor>& output_tensor_list,
    const at::Tensor& input_tensor,
    int root,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(input_tensor);
  checkTensorDevice(input_tensor);

  // Only the root rank needs valid output tensors
  if (rank_ == root) {
    TORCH_CHECK(
        output_tensor_list.size() == static_cast<size_t>(comm_size_),
        "output_tensor_list size (" +
            std::to_string(output_tensor_list.size()) +
            ") must equal comm_size (" + std::to_string(comm_size_) +
            ") for gather operation");

    for (size_t i = 0; i < output_tensor_list.size(); ++i) {
      const auto& t = output_tensor_list[i];
      ensureTensorContiguous(t);
      TORCH_CHECK(
          t.numel() == input_tensor.numel(),
          "Output tensor at index " + std::to_string(i) + " has size " +
              std::to_string(t.numel()) + " but expected " +
              std::to_string(input_tensor.numel()) + " to match input tensor");
    }

    checkTensorsDevice(output_tensor_list);
    checkSameDtype(input_tensor, output_tensor_list);
  }

  TracingGuard tracingGuard(
      name_,
      comm_size_,
      "gather",
      root,
      sequence_number_,
      {input_tensor},
      output_tensor_list);

  c10::xpu::XPUStream stream = getOperationStream(async_op);

  auto work = async_op ? createWork(stream, timeout, input_tensor)
                       : createWork(stream, timeout);

  work->recordStart("gather");

  // Grouped to avoid a oneCCL hang; see
  // https://github.com/uxlfoundation/oneCCL/issues/193.
  XCCL_CHECK(
      xccl_api_, xccl_api_->groupStart(), "XCCL groupStart failed in gather");

  if (rank_ == root) {
    for (int peer_rank = 0; peer_rank < comm_size_; ++peer_rank) {
      if (peer_rank != root) {
        const auto& peer_tensor = output_tensor_list[peer_rank];
        onecclResult_t result = xccl_api_->recv(
            peer_tensor.data_ptr(),
            peer_tensor.numel(),
            getXcclDataType(peer_tensor),
            peer_rank,
            xccl_comm_,
            stream);
        if (result != onecclSuccess) [[unlikely]] {
          XCCL_CHECK_IGNORE(
              xccl_api_,
              xccl_api_->groupEnd(),
              "XCCL groupEnd failed during error cleanup after recv failure in gather");
          throw XCCLException(*xccl_api_, "XCCL recv failed in gather", result);
        }
      }
    }

    // Root's own chunk never goes on the wire; copy it locally.
    {
      c10::StreamGuard guard(stream);
      output_tensor_list[root].copy_(input_tensor, /*non_blocking=*/true);
    }
  } else {
    onecclResult_t result = xccl_api_->send(
        input_tensor.data_ptr(),
        input_tensor.numel(),
        getXcclDataType(input_tensor),
        root,
        xccl_comm_,
        stream);
    if (result != onecclSuccess) [[unlikely]] {
      XCCL_CHECK_IGNORE(
          xccl_api_,
          xccl_api_->groupEnd(),
          "XCCL groupEnd failed during error cleanup after send failure in gather");
      throw XCCLException(*xccl_api_, "XCCL send failed in gather", result);
    }
  }

  XCCL_CHECK(
      xccl_api_, xccl_api_->groupEnd(), "XCCL groupEnd failed in gather");

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

} // namespace c10d::xccl2

#endif // USE_C10D_XCCL
