#include <c10/util/Exception.h>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>

#ifdef USE_C10D_GLOO

#include <torch/csrc/distributed/c10d/GlooDeviceFactory.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <chrono>
#include <exception>

#ifdef _WIN32
#include <gloo/common/win.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#endif
#include <sys/types.h>

#include <type_traits>
#include <utility>

#include <gloo/allgather.h>
#include <gloo/allgatherv.h>
#include <gloo/allreduce.h>
#include <gloo/alltoall.h>
#include <gloo/alltoallv.h>
#include <gloo/barrier.h>
#include <gloo/broadcast.h>
#include <gloo/gather.h>
#include <gloo/reduce.h>
#include <gloo/scatter.h>

#include <ATen/ThreadLocalState.h>
#include <ATen/native/SparseTensorUtils.h>

#include <c10/util/StringUtil.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>
#include <gloo/config.h>
#include <gloo/rendezvous/context.h>
#include <gloo/rendezvous/prefix_store.h>

#ifdef _WIN32
#define GENERATE_ALL_TYPES(type, func, ...)      \
  switch (type) {                                \
    case ::at::ScalarType::Float:                \
      func<float>(__VA_ARGS__);                  \
      break;                                     \
    case ::at::ScalarType::Double:               \
      func<double>(__VA_ARGS__);                 \
      break;                                     \
    case ::at::ScalarType::Half:                 \
      func<gloo::float16>(__VA_ARGS__);          \
      break;                                     \
    case ::at::ScalarType::BFloat16:             \
      func<c10::BFloat16>(__VA_ARGS__);          \
      break;                                     \
    case ::at::ScalarType::Char:                 \
      func<int8_t>(__VA_ARGS__);                 \
      break;                                     \
    case ::at::ScalarType::Byte:                 \
    case ::at::ScalarType::Bool:                 \
      func<uint8_t>(__VA_ARGS__);                \
      break;                                     \
    case ::at::ScalarType::Int:                  \
      func<int32_t>(__VA_ARGS__);                \
      break;                                     \
    case ::at::ScalarType::Long:                 \
      func<int64_t>(__VA_ARGS__);                \
      break;                                     \
    default:                                     \
      TORCH_CHECK(false, "Invalid scalar type"); \
  }

#define HOST_NAME_MAX 256
#else
#define GENERATE_ALL_TYPES(type, func, args...)  \
  switch (type) {                                \
    case ::at::ScalarType::Float:                \
      func<float>(args);                         \
      break;                                     \
    case ::at::ScalarType::Double:               \
      func<double>(args);                        \
      break;                                     \
    case ::at::ScalarType::Half:                 \
      func<gloo::float16>(args);                 \
      break;                                     \
    case ::at::ScalarType::BFloat16:             \
      func<c10::BFloat16>(args);                 \
      break;                                     \
    case ::at::ScalarType::Char:                 \
      func<int8_t>(args);                        \
      break;                                     \
    case ::at::ScalarType::Byte:                 \
    case ::at::ScalarType::Bool:                 \
      func<uint8_t>(args);                       \
      break;                                     \
    case ::at::ScalarType::Int:                  \
      func<int32_t>(args);                       \
      break;                                     \
    case ::at::ScalarType::Long:                 \
      func<int64_t>(args);                       \
      break;                                     \
    default:                                     \
      TORCH_CHECK(false, "Invalid scalar type"); \
  }
#endif

namespace c10d {

namespace {

using steady_clock_time_point =
    std::chrono::time_point<std::chrono::steady_clock>;

std::chrono::milliseconds getRemainingTime(
    steady_clock_time_point startTime,
    const std::chrono::milliseconds& timeout,
    bool waitAllRanks) {
  if (waitAllRanks) {
    // See Note in monitoredBarrier
    return timeout;
  }
  auto elapsedTime = std::chrono::steady_clock::now() - startTime;
  auto remainingMillis = timeout -
      std::chrono::duration_cast<std::chrono::milliseconds>(elapsedTime);

  // If no more remaining time, return -1 to indicate to caller.
  if (remainingMillis.count() <= 0) {
    return std::chrono::milliseconds(-1);
  }

  return remainingMillis;
}

// Emit a LOG(ERROR) and throws using TORCH_CHECK with the given messages.
void logAndThrow(
    const std::string& logMessage,
    const std::string& errorMessage) {
  LOG(ERROR) << logMessage;
  TORCH_CHECK(false, errorMessage);
}

// For monitoredBarrier, checks remaining time left to finish processing ranks
// and throws error if timeout.
void checkRemainingTime(
    const std::chrono::milliseconds& monitoredBarrierTimeout,
    const std::chrono::milliseconds& remainingTime,
    const std::vector<int>& processedRanks,
    int currentRank) {
  const std::string kNoRemainingTimeError = c10::str(
      "Rank ",
      currentRank,
      " timed out in monitoredBarrier after ",
      monitoredBarrierTimeout.count(),
      " ms.");
  if (remainingTime.count() < 0) {
    std::string rankInfo;
    if (!processedRanks.empty()) {
      rankInfo = c10::str(
          "Successfully processed ranks: ", c10::Join(", ", processedRanks));
    } else {
      rankInfo = "No ranks successfully processed in monitoredBarrier.";
    }
    auto error = c10::str(kNoRemainingTimeError, "\n", rankInfo);
    logAndThrow(error, error);
  }
}

typedef void (*ReduceFunc)(void*, const void*, const void*, size_t);

template <typename T, std::enable_if_t<!std::is_integral_v<T>, int> = 0>
ReduceFunc toFunction(const ReduceOp& r) {
  switch (r) {
    case ReduceOp::SUM:
      return ReduceFunc(&::gloo::sum<T>);
    case ReduceOp::PRODUCT:
      return ReduceFunc(&::gloo::product<T>);
    case ReduceOp::MIN:
      return ReduceFunc(&::gloo::min<T>);
    case ReduceOp::MAX:
      return ReduceFunc(&::gloo::max<T>);
    case ReduceOp::BAND:
      TORCH_CHECK(false, "Cannot use ReduceOp.BAND with non-integral dtype");
      break;
    case ReduceOp::BOR:
      TORCH_CHECK(false, "Cannot use ReduceOp.BOR with non-integral dtype");
      break;
    case ReduceOp::BXOR:
      TORCH_CHECK(false, "Cannot use ReduceOp.BXOR with non-integral dtype");
      break;
    case ReduceOp::AVG:
      TORCH_CHECK(false, "Cannot use ReduceOp.AVG with Gloo");
      break;
    case ReduceOp::PREMUL_SUM:
      TORCH_CHECK(false, "Cannot use ReduceOp.PREMUL_SUM with Gloo");
      break;
    case ReduceOp::UNUSED:
    default:
      break;
  }

  TORCH_CHECK(false, "Unhandled ReduceOp");
}

// Bitwise AND with SFINAE guard for integral types.
template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
void band(void* c, const void* a, const void* b, size_t n) {
  auto tc = static_cast<T*>(c);
  auto ta = static_cast<const T*>(a);
  auto tb = static_cast<const T*>(b);
  for (const auto i : c10::irange(n)) {
    tc[i] = ta[i] & tb[i];
  }
}

// Bitwise OR with SFINAE guard for integral types.
template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
void bor(void* c, const void* a, const void* b, size_t n) {
  auto tc = static_cast<T*>(c);
  auto ta = static_cast<const T*>(a);
  auto tb = static_cast<const T*>(b);
  for (const auto i : c10::irange(n)) {
    tc[i] = ta[i] | tb[i];
  }
}

// Bitwise XOR with SFINAE guard for integral types.
template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
void bxor(void* c, const void* a, const void* b, size_t n) {
  auto tc = static_cast<T*>(c);
  auto ta = static_cast<const T*>(a);
  auto tb = static_cast<const T*>(b);
  for (const auto i : c10::irange(n)) {
    tc[i] = ta[i] ^ tb[i];
  }
}

template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
ReduceFunc toFunction(const ReduceOp& r) {
  switch (r) {
    case ReduceOp::SUM:
      return ReduceFunc(&::gloo::sum<T>);
    case ReduceOp::PRODUCT:
      return ReduceFunc(&::gloo::product<T>);
    case ReduceOp::MIN:
      return ReduceFunc(&::gloo::min<T>);
    case ReduceOp::MAX:
      return ReduceFunc(&::gloo::max<T>);
    case ReduceOp::BAND:
      return ReduceFunc(&band<T>);
    case ReduceOp::BOR:
      return ReduceFunc(&bor<T>);
    case ReduceOp::BXOR:
      return ReduceFunc(&bxor<T>);
    case ReduceOp::AVG:
      TORCH_CHECK(false, "Cannot use ReduceOp.AVG with Gloo");
      break;
    case ReduceOp::PREMUL_SUM:
      TORCH_CHECK(false, "Cannot use ReduceOp.PREMUL_SUM with Gloo");
      break;
    case ReduceOp::UNUSED:
    default:
      break;
  }

  TORCH_CHECK(false, "Unhandled ReduceOp");
}

template <typename T, typename O>
void setInputs(O& opts, std::vector<at::Tensor>& tensors) {
  opts.setInputs(getDataPointers<T>(tensors), tensors[0].numel());
}

template <typename T, typename O>
void setInput(O& opts, at::Tensor& tensor) {
  opts.setInput(getDataPointer<T>(tensor), tensor.numel());
}

template <typename T, typename O>
void setInput(O& opts, at::Tensor& tensor, std::vector<size_t>& counts) {
  opts.setInput(getDataPointer<T>(tensor), counts);
}

template <typename T, typename O>
void setInput(O& opts, at::Tensor& tensor, std::vector<int64_t>& counts) {
  opts.setInput(getDataPointer<T>(tensor), counts);
}

template <typename T, typename O>
void setOutputs(O& opts, std::vector<at::Tensor>& tensors) {
  opts.setOutputs(getDataPointers<T>(tensors), tensors[0].numel());
}

template <typename T, typename O>
void setOutput(O& opts, at::Tensor& tensor) {
  opts.setOutput(getDataPointer<T>(tensor), tensor.numel());
}

template <typename T, typename O>
void setOutput(O& opts, at::Tensor& tensor, std::vector<size_t>& counts) {
  opts.setOutput(getDataPointer<T>(tensor), counts);
}

template <typename T, typename O>
void setOutput(O& opts, at::Tensor& tensor, std::vector<int64_t>& counts) {
  opts.setOutput(getDataPointer<T>(tensor), counts);
}

at::Tensor pinnedLike(at::Tensor& tensor) {
  auto* allocator = at::detail::getCUDAHooks().getPinnedMemoryAllocator();
  auto storage = c10::Storage(
      c10::Storage::use_byte_size_t(),
      static_cast<int64_t>(at::detail::computeStorageNbytes(
          tensor.sizes(), tensor.strides(), tensor.dtype().itemsize())),
      allocator,
      /*resizable=*/false);
  return at::empty({0}, tensor.options().device(at::kCPU))
      .set_(storage, 0, tensor.sizes(), tensor.strides());
}

// This function initializes a vector of CUDA streams, one for every
// tensor in the input tensor vector, and ensures that these streams are
// synchronized with the current default streams. This is needed so
// that new work on the new streams is serialized w.r.t. all operations
// on the tensors.
void initializeStreamsEvents(
    const std::vector<at::Tensor>& tensors,
    std::vector<c10::Stream>& streams,
    std::vector<c10::Event>& events) {
  streams.reserve(tensors.size());
  events.reserve(tensors.size());
  for (const auto i : c10::irange(tensors.size())) {
    c10::Device device = tensors[i].device();
    c10::impl::VirtualGuardImpl impl(device.type());
    // Record event on current stream
    events.emplace_back(device.type());
    events[i].record(impl.getStream(device));
    // Get a non-default stream to execute asynchronous CUDA operations
    // on for this device. This ensures that the default stream used
    // by the caller is not occupied by c10d related operations.
    streams.push_back(
        impl.getStreamFromGlobalPool(device, /*isHighPriority=*/true));
    // Ensure the new stream is synchronized with the current stream.
    events[i].block(streams[i]);

    // `tensors` are created on a different stream. Hence, they must record
    // new streams in this Work to prevent being freed before the Work finishes.
    if (tensors[i].is_sparse()) {
      if (tensors[i].is_coalesced()) {
        impl.recordDataPtrOnStream(
            tensors[i].indices().storage().data_ptr(), streams[i]);
        impl.recordDataPtrOnStream(
            tensors[i].values().storage().data_ptr(), streams[i]);
      } else {
        // We will need to coalesce first, which means new tensors will
        // be allocated on the streams we just allocated, and there
        // is no need to record them separately.
      }
    } else {
      impl.recordDataPtrOnStream(tensors[i].storage().data_ptr(), streams[i]);
    }
  }
}

// This function initializes a vector of CUDA streams, one per device,
// and ensures that these streams are synchronized with the current default
// streams. It is assumed that the tensors in the nested tensor vectors are
// on the same device.
void initializeStreamsEvents(
    std::vector<std::vector<at::Tensor>>& tensors,
    std::vector<c10::Stream>& streams,
    std::vector<c10::Event>& events) {
  // Ensure that the tensors in the nested tensor vectors are on the same
  // device.
  for (const auto& tensorgroup : tensors) {
    const auto device_id = tensorgroup[0].device().index();
    for (const auto& tensor : tensorgroup) {
      if (tensor.device().index() != device_id) {
        TORCH_CHECK(
            false,
            "tensors in the nested tensor vectors need to "
            "be on the same device");
      }
    }
  }

  streams.reserve(tensors.size());
  events.reserve(tensors.size());
  for (const auto i : c10::irange(tensors.size())) {
    c10::Device device = tensors[i][0].device();
    c10::impl::VirtualGuardImpl impl(device.type());
    // Record event on current stream
    events.emplace_back(device.type());
    events[i].record(impl.getStream(device));
    // Get a non-default stream to execute asynchronous CUDA operations
    // on for this output. This ensures that the default stream used
    // by the caller is not occupied by c10d related operations.
    streams.push_back(
        impl.getStreamFromGlobalPool(device, /*isHighPriority=*/true));
    // Ensure the new stream is synchronized with the current stream.
    events[i].block(streams[i]);

    for (at::Tensor& tensor : tensors[i]) {
      // `tensors` are created on a different stream. Hence, they must record
      // new streams in this Work to prevent being freed before the Work
      // finishes.
      impl.recordDataPtrOnStream(tensor.storage().data_ptr(), streams[i]);
    }
  }
}

const auto kLoopbackAddress = "127.0.0.1";

} // namespace

// static
void ProcessGroupGloo::AsyncWork::execute(
    const c10::intrusive_ptr<AsyncWork>& work) {
  if (work->recordFunctionBeforeCallback_) {
    work->recordFunctionBeforeCallback_();
  }
  try {
    work->run();
  } catch (...) {
    work->finishWorkGlooError(std::current_exception());
    return;
  }

  // FIXME: We need to call it here since Future completion requires all
  // the work to be synchronized to CUDA.
  work->synchronize();
  work->finishWorkGloo();
}

std::vector<at::Tensor> ProcessGroupGloo::AsyncWork::result() {
  TORCH_CHECK(
      isCompleted(),
      "Work needs to be completed before calling result(). "
      "Should call wait() before result().");
  TORCH_CHECK(
      outputTensors_.size() <= 1,
      "work result does not support list of lists, use .getFuture() and value()");
  return outputTensors_.empty() ? std::vector<at::Tensor>()
                                : outputTensors_.at(0);
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupGloo::AsyncWork::
    getFuture() {
  return future_;
}

namespace {
c10::intrusive_ptr<c10::ivalue::Future> createFutureAsOutput(
    const std::vector<std::vector<at::Tensor>>& outputTensors) {
  if (outputTensors.size() > 1) {
    return c10::make_intrusive<c10::ivalue::Future>(
        c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  }
  return c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()));
}

void returnFutureWithOutput(
    c10::intrusive_ptr<c10::ivalue::Future>& future,
    const std::vector<std::vector<at::Tensor>>& outputTensors) {
  if (outputTensors.empty()) {
    future->markCompleted(c10::IValue(std::vector<at::Tensor>()));
    return;
  }
  if (outputTensors.size() > 1) {
    future->markCompleted(c10::IValue(outputTensors));
    return;
  }
  future->markCompleted(c10::IValue(outputTensors[0]));
}
} // namespace

inline void ProcessGroupGloo::AsyncWork::recordAsyncWorkProfilingInfo(
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputTensors) {
  auto recordingFunction =
      std::make_shared<at::RecordFunction>(at::RecordScope::USER_SCOPE);
  if (recordingFunction->isActive()) {
    std::function<void()> before_handler =
        [inputTensors, profilingTitle, recordingFunction]() {
          // The work will be started and completed by different threads.
          recordingFunction->_setAsync();
          std::vector<c10::IValue> inputs;
          if (inputTensors) {
            inputs.reserve(inputTensors->size());
            for (const auto& tensor : *inputTensors) {
              inputs.emplace_back(tensor);
            }
          }
          recordingFunction->before(
              profilingTitle,
              c10::ArrayRef<const c10::IValue>(inputs.data(), inputs.size()));
        };
    recordFunctionBeforeCallback_ =
        at::wrapPropagateTLSState(std::move(before_handler));
    std::function<void()> end_handler = [recordingFunction]() {
      recordingFunction->end();
    };
    recordFunctionEndCallback_ = at::wrapPropagateTLSState(end_handler);
  }
}

ProcessGroupGloo::AsyncWork::AsyncWork(
    std::vector<std::vector<at::Tensor>> outputTensors,
    OpType opType,
    uint64_t seq,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputTensors)
    // Profiler: Pass nullptr as profilingTitle to parent constructor to
    // replace default profiler implementation with async version that reports
    // correct timestamps for work that is asynchronously executed.
    : Work(-1, opType, nullptr, inputTensors),
      outputTensors_(std::move(outputTensors)),
      future_(createFutureAsOutput(outputTensors_)),
      seq_(seq) {
  if (profilingTitle != nullptr) {
    recordAsyncWorkProfilingInfo(profilingTitle, inputTensors);
  }
}

uint64_t ProcessGroupGloo::AsyncWork::getSequencenumber() const {
  return seq_;
}

void ProcessGroupGloo::AsyncWork::finishWorkGlooError(
    const std::exception_ptr& eptr) {
  future_->setError(eptr);
  finish(eptr);
}

void ProcessGroupGloo::AsyncWork::finishWorkGloo() {
  returnFutureWithOutput(future_, outputTensors_);
  finish();
}

ProcessGroupGloo::SendWork::SendWork(
    at::Tensor& tensor,
    std::unique_ptr<::gloo::transport::UnboundBuffer> buffer,
    uint64_t seq)
    : Work(
          -1,
          OpType::SEND,
          "gloo:send",
          std::optional<std::vector<at::Tensor>>({tensor})),
      tensor_(tensor),
      buffer_(std::move(buffer)),
      seq_(seq) {}

uint64_t ProcessGroupGloo::SendWork::getSequencenumber() const {
  return seq_;
}

bool ProcessGroupGloo::SendWork::wait(std::chrono::milliseconds timeout) {
  bool sendCompleted = false;
  std::exception_ptr exception{nullptr};
  try {
    if (timeout == kNoTimeout) {
      sendCompleted = buffer_->waitSend();
    } else {
      sendCompleted = buffer_->waitSend(timeout);
    }
  } catch (...) {
    exception = std::current_exception();
  }

  // Completes the Work object and throws the exception.
  finishAndThrow(exception);
  return sendCompleted;
}

void ProcessGroupGloo::SendWork::abort() {
  buffer_->abortWaitSend();
}

ProcessGroupGloo::RecvWork::RecvWork(
    at::Tensor& tensor,
    std::unique_ptr<::gloo::transport::UnboundBuffer> buffer,
    OpType opType,
    uint64_t seq,
    const char* profilingTitle)
    : Work(
          -1,
          opType,
          profilingTitle,
          std::optional<std::vector<at::Tensor>>({tensor})),
      tensor_(tensor),
      buffer_(std::move(buffer)),
      srcRank_(-1),
      seq_(seq) {}

uint64_t ProcessGroupGloo::RecvWork::getSequencenumber() const {
  return seq_;
}

int ProcessGroupGloo::RecvWork::sourceRank() const {
  std::lock_guard<std::timed_mutex> lock(mutex_);
  return srcRank_;
}

bool ProcessGroupGloo::RecvWork::wait(std::chrono::milliseconds timeout) {
  bool recvCompleted = false;
  std::exception_ptr exception{nullptr};
  try {
    if (timeout == kNoTimeout) {
      recvCompleted = buffer_->waitRecv(&srcRank_);
    } else {
      recvCompleted = buffer_->waitRecv(&srcRank_, timeout);
    }
  } catch (...) {
    exception = std::current_exception();
  }

  // Completes the Work object and throws the exception.
  finishAndThrow(exception);
  return recvCompleted;
}

void ProcessGroupGloo::RecvWork::abort() {
  buffer_->abortWaitRecv();
}

ProcessGroupGloo::Options::Options(std::chrono::milliseconds timeout)
    : Backend::Options(GLOO_BACKEND_NAME, timeout), threads(2) {}

namespace {

void socketInitialize() {
#ifdef _WIN32
  ::gloo::init_winsock();
#endif
}

// Gloo assumes that this machine's hostname can always be resolved
// to an address. If it doesn't it throws a runtime error saying
// that it can't be resolved. Instead of catching it, we choose
// to proactively check if an address can be resolved, so we can
// gracefully fall back to an alternative if it doesn't.
bool doesHostnameResolveToUsableAddress(const std::string& hostname) {
  socketInitialize();
  struct addrinfo hints {};
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  struct addrinfo* result = nullptr;
  auto rv = getaddrinfo(hostname.c_str(), nullptr, &hints, &result);
  if (rv < 0) {
    return false;
  }
  struct addrinfo* rp = nullptr;
  for (rp = result; rp != nullptr; rp = rp->ai_next) {
    auto fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    if (fd == -1) {
      continue;
    }
    rv = bind(fd, rp->ai_addr, rp->ai_addrlen);
#ifdef _WIN32
    closesocket(fd);
#else
    close(fd);
#endif
    if (rv == -1) {
      continue;
    }
    break;
  }
  freeaddrinfo(result);
  return rp != nullptr;
}

} // namespace

std::shared_ptr<::gloo::transport::Device> ProcessGroupGloo::
    createDeviceForInterface(const std::string& interface_name) {
  return ::c10d::GlooDeviceFactory::makeDeviceForInterface(interface_name);
}

std::shared_ptr<::gloo::transport::Device> ProcessGroupGloo::
    createDeviceForHostname(const std::string& hostname) {
  TORCH_CHECK(
      doesHostnameResolveToUsableAddress(hostname),
      "Cannot resolve ",
      hostname,
      " to a (local) address");
  return ::c10d::GlooDeviceFactory::makeDeviceForHostname(hostname);
}

#if defined(__linux__) || defined(_WIN32)
std::shared_ptr<::gloo::transport::Device> ProcessGroupGloo::
    createDefaultDevice() {
  // Use the hostname to resolve the network address to
  // use. Note: if the hostname does not resolve to an address (e.g.
  // because of misconfigured /etc/hosts file), this will not work.
  socketInitialize();
  std::array<char, HOST_NAME_MAX> hostname{};
  auto rv = gethostname(hostname.data(), HOST_NAME_MAX);
  if (rv != 0) {
    C10_THROW_ERROR(DistBackendError, std::strerror(errno));
  }

  // Use this machine's hostname if it resolves to an address.
  if (doesHostnameResolveToUsableAddress(hostname.data())) {
    return ::c10d::GlooDeviceFactory::makeDeviceForHostname(hostname.data());
  }

  // Otherwise, use the loopback address.
  TORCH_WARN_ONCE(
      "Unable to resolve hostname to a (local) address. ",
      "Using the loopback address as fallback. ",
      "Manually set the network interface to bind to with GLOO_SOCKET_IFNAME.");
  return createDeviceForHostname(kLoopbackAddress);
}
#endif

#ifdef __APPLE__
std::shared_ptr<::gloo::transport::Device> ProcessGroupGloo::
    createDefaultDevice() {
  // Use the hostname to resolve the network address to
  // use. Note: if the hostname does not resolve to an address (e.g.
  // because of misconfigured /etc/hosts file), this will not work.
  const auto hostNameMax = sysconf(_SC_HOST_NAME_MAX);
  auto hostname = std::unique_ptr<char[]>(new char[hostNameMax]);
  auto rv = gethostname(hostname.get(), hostNameMax);
  if (rv != 0) {
    C10_THROW_ERROR(DistBackendError, std::strerror(errno));
  }

  // Use this machine's hostname if it resolves to an address.
  if (doesHostnameResolveToUsableAddress(hostname.get())) {
    return ::c10d::GlooDeviceFactory::makeDeviceForHostname(hostname.get());
  }

  // Otherwise, use the loopback address.
  TORCH_WARN_ONCE(
      "Unable to resolve hostname to a (local) address. ",
      "Using the loopback address as fallback. ",
      "Manually set the network interface to bind to with GLOO_SOCKET_IFNAME.");
  return createDeviceForHostname(kLoopbackAddress);
}
#endif

ProcessGroupGloo::ProcessGroupGloo(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : Backend(rank, size),
      store_(new GlooStore(store)),
      options_(std::move(options)),
      stop_(false),
      collectiveCounter_(0) {
  auto& devices = options_->devices;
  if (devices.empty()) {
    TORCH_CHECK(false, "No device(s) specified");
  }

  // Create and connect a context for every device.
  //
  // Note that the same device can be specified multiple times, either
  // the same object, or the same logical device as different objects.
  // Either mode is fine and only has performance implications.
  //
  // Using the same object multiple times means all contexts share a
  // single I/O thread. If you use different objects for the same
  // logical device they will have independent I/O threads. The latter
  // option is needed if you have a fast NIC that cannot be saturated
  // by a single I/O thread.
  //
  contexts_.reserve(options_->devices.size());
  for (const auto i : c10::irange(options_->devices.size())) {
    auto context = std::make_shared<::gloo::rendezvous::Context>(rank_, size_);
    auto store = ::gloo::rendezvous::PrefixStore(std::to_string(i), *store_);
    context->setTimeout(options_->timeout);
    try {
      context->connectFullMesh(store, options_->devices[i]);
    } catch (const std::runtime_error& e) {
      auto err = e.what();
      // TORCH_CHECK to print the cpp stacktrace.
      auto msg = c10::str("Gloo connectFullMesh failed with ", err);
      logAndThrow(msg, msg);
    }
    contexts_.push_back(std::move(context));
  }

  // Every worker thread stores the AsyncWork object it's currently
  // working on in the workInProgress_ vector. It must have size equal
  // to the number of workers such that they can simply index into it
  // using the worker index they are started with.
  workInProgress_.resize(options_->threads);

  threads_.resize(options_->threads);
  for (const auto i : c10::irange(threads_.size())) {
    threads_[i] = std::thread(&ProcessGroupGloo::runLoop, this, i);
  }

  init();
}

ProcessGroupGloo::~ProcessGroupGloo() {
  std::unique_lock<std::mutex> lock(workMutex_);
  workConsumeCV_.wait(lock, [&] { return workQueue_.empty(); });

  // Queue is empty, signal stop
  stop_ = true;

  // Release lock to allow threads to terminate
  lock.unlock();

  workProduceCV_.notify_all();

  // Wait for worker threads to terminate
  for (auto& thread : threads_) {
    thread.join();
  }
}

uint32_t ProcessGroupGloo::nextTag() {
  return collectiveCounter_++;
}

std::shared_ptr<::gloo::Context> ProcessGroupGloo::getContext(uint32_t tag) {
  return contexts_[tag % contexts_.size()];
}

void ProcessGroupGloo::runLoop(int workerIndex) {
  std::unique_lock<std::mutex> lock(workMutex_);

  while (!stop_) {
    if (workQueue_.empty()) {
      workProduceCV_.wait(lock);
      continue;
    }

    auto work = std::move(workQueue_.front());
    workQueue_.pop_front();
    workInProgress_[workerIndex] = work;
    lock.unlock();

    // Notify after releasing the lock so that the waiter
    // does not immediately block.
    workConsumeCV_.notify_one();

    AsyncWork::execute(work);
    lock.lock();
    workInProgress_[workerIndex].reset();
  }
}

void ProcessGroupGloo::enqueue(c10::intrusive_ptr<AsyncWork> work) {
  std::unique_lock<std::mutex> lock(workMutex_);
  workQueue_.push_back(std::move(work));
  lock.unlock();

  // Notify after releasing the lock so that the waiter
  // does not immediately block.
  workProduceCV_.notify_one();
}

namespace {

class AsyncBroadcastWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncBroadcastWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      int rootRank,
      int rootTensor,
      uint32_t tag,
      uint64_t seq)
      : ProcessGroupGloo::AsyncWork(
            {inputs},
            OpType::BROADCAST,
            seq,
            "gloo:broadcast",
            inputs),
        context(context),
        inputs(inputs),
        rootRank(rootRank),
        rootTensor(rootTensor),
        tag(tag) {}

  std::shared_ptr<gloo::Context> context;
  std::vector<at::Tensor> inputs{};
  const int rootRank;
  const int rootTensor;
  const uint32_t tag;

  void broadcast(at::Tensor& tensor) {
    const auto& scalarType = tensor.scalar_type();
    gloo::BroadcastOptions opts(context);
    opts.setRoot(rootRank);
    opts.setTag(tag);
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, tensor);
    gloo::broadcast(opts);
  }

  void run() override {
    broadcast(inputs[rootTensor]);

    // Copy to non-root tensors
    for (const auto i : c10::irange(inputs.size())) {
      if (i == static_cast<size_t>(rootTensor)) {
        continue;
      }
      inputs[i].copy_(inputs[rootTensor]);
    }
  }
};

class AsyncBroadcastCUDAWork : public AsyncBroadcastWork {
 public:
  AsyncBroadcastCUDAWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      int rootRank,
      int rootTensor,
      uint32_t tag,
      uint64_t seq)
      : AsyncBroadcastWork(context, inputs, rootRank, rootTensor, tag, seq) {
    initializeStreamsEvents(inputs, streams, events);

    // Create pinned host side tensors.
    tmp = pinnedLike(inputs[rootTensor]);
    c10::OptionalStreamGuard guard;
    if (context->rank == rootRank) {
      guard.reset_stream(streams[rootTensor]);
      tmp.copy_(inputs[rootTensor], /* non_blocking */ true);
    }
  }

  void run() override {
    // Synchronize with copy operation if applicable.
    if (context->rank == rootRank) {
      streams[rootTensor].synchronize();
    }

    // Run broadcast on host side tensors.
    broadcast(tmp);

    // Kick off copy back to the CUDA tensors.
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(streams[i]);
      inputs[i].copy_(tmp, /* non_blocking */ true);
      events[i].record(streams[i]);
    }
  }

  void synchronize() override {
    // Synchronize with the copy back to CUDA tensors.
    for (const auto i : c10::irange(inputs.size())) {
      c10::Device device = inputs[i].device();
      events[i].block(
          c10::impl::VirtualGuardImpl(device.type()).getStream(device));
    }
  }

  at::Tensor tmp;
  std::vector<c10::Stream> streams{};
  std::vector<c10::Event> events{};
};

} // namespace

c10::intrusive_ptr<Work> ProcessGroupGloo::broadcast(
    std::vector<at::Tensor>& inputs,
    const BroadcastOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupGloo::broadcast: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);
  assertRootTensor(
      invalidArgument, opts.rootTensor, static_cast<int64_t>(inputs.size()));
  assertDense(invalidArgument, inputs);
  assertTypeAndSizesMatch(invalidArgument, inputs);

  const auto& device = inputs[0].device();
  switch (device.type()) {
    case at::kCPU:
      break;
    case at::kCUDA:
      // If the user gave us a CUDA tensor then CUDA must be loaded.
      TORCH_INTERNAL_ASSERT(at::hasCUDA());
      break;
    default:
      invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  c10::intrusive_ptr<AsyncBroadcastWork> work;
  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;
  if (device.type() == at::kCPU) {
    work = c10::make_intrusive<AsyncBroadcastWork>(
        std::move(context), inputs, opts.rootRank, opts.rootTensor, tag, seq_);
  } else if (device.type() == at::kCUDA) {
    work = c10::make_intrusive<AsyncBroadcastCUDAWork>(
        std::move(context), inputs, opts.rootRank, opts.rootTensor, tag, seq_);
  } else {
    TORCH_CHECK(false, "Invalid backend");
  }

  enqueue(work);
  return work;
}

namespace {

class AsyncAllreduceWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncAllreduceWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      ReduceOp reduceOp,
      uint32_t tag,
      uint64_t seq)
      : ProcessGroupGloo::AsyncWork(
            {inputs},
            OpType::ALLREDUCE,
            seq,
            "gloo:all_reduce",
            inputs),
        context(context),
        inputs(inputs),
        reduceOp(std::move(reduceOp)),
        tag(tag) {}

  std::shared_ptr<gloo::Context> context;
  std::vector<at::Tensor> inputs{};
  const ReduceOp reduceOp;
  const uint32_t tag;

  void allreduce(std::vector<at::Tensor>& tensors) {
    const auto& scalarType = tensors[0].scalar_type();
    gloo::AllreduceOptions opts(context);
    opts.setReduceFunction(getFunction(scalarType, reduceOp));
    opts.setTag(tag);
    GENERATE_ALL_TYPES(scalarType, setOutputs, opts, tensors);
    gloo::allreduce(opts);
  }

  void run() override {
    allreduce(inputs);
  }

  template <typename T>
  void getFunction(gloo::AllreduceOptions::Func& fn, const ReduceOp op) {
    fn = toFunction<T>(op);
  }

  gloo::AllreduceOptions::Func getFunction(
      const at::ScalarType& dtype,
      const ReduceOp& op) {
    gloo::AllreduceOptions::Func fn;
    GENERATE_ALL_TYPES(dtype, getFunction, fn, op);
    return fn;
  }
};

class AsyncAllreduceCoalescedWork : public AsyncAllreduceWork {
 public:
  AsyncAllreduceCoalescedWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      ReduceOp reduceOp,
      uint32_t tag,
      uint64_t seq)
      : AsyncAllreduceWork(context, inputs, std::move(reduceOp), tag, seq) {}

  void run() override {
    allreduceCoalesced(inputs);
  }

 private:
  void allreduceCoalesced(std::vector<at::Tensor>& tensors) {
    // reduce coalesced, flattened tensors.
    at::Tensor coalescedTensor = flattenDenseTensors(tensors);
    std::vector<at::Tensor> allreduceInput = {coalescedTensor};
    allreduce(allreduceInput);

    // separate and reshape tensors.
    size_t offset = 0;
    for (at::Tensor& tensor : tensors) {
      const int64_t tensorNumel = tensor.numel();
      const c10::IntArrayRef tensorShape = tensor.sizes();
      tensor.copy_(coalescedTensor.slice(0, offset, offset + tensorNumel)
                       .view(tensorShape));
      offset += tensorNumel;
    }
  }
};

class AsyncSparseAllreduceWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncSparseAllreduceWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      uint32_t tag,
      uint64_t seq)
      : ProcessGroupGloo::AsyncWork(
            {inputs},
            OpType::_ALLREDUCE_SPARSE,
            seq,
            "gloo:sparse_all_reduce",
            inputs),
        context(context),
        inputs(inputs),
        tag(tag) {}

  std::shared_ptr<gloo::Context> context;
  std::vector<at::Tensor> inputs{};
  const uint32_t tag;

  // We share dimensionality about the sparse tensors before collecting
  // their contents. We assume here that the maximum number of sparse
  // and dense dimensions is 4. This is stored in a contiguous piece of
  // memory so that we can easily run allgather on it.
  //
  // The layout of this memory is as follows:
  //
  //   - [0:4]: sparse dims
  //   - [4:8]: dense dims
  //   -   [8]: nnz
  //
  class SparseTensorMetadata {
   public:
    static constexpr auto dim = 9;

    // Construct from an existing metadata tensor to facilitate structured
    // access to metadata from peers, after gathering it.
    explicit SparseTensorMetadata(at::Tensor metadata)
        : metadata_(std::move(metadata)),
          data_(metadata_.mutable_data_ptr<int64_t>()) {
      AT_ASSERT(metadata_.scalar_type() == at::kLong);
      AT_ASSERT(metadata_.dim() == 1);
      AT_ASSERT(metadata_.size(0) == dim);
    }

    // Populate the metadata.
    void populate_from_sparse_tensor(const at::Tensor& tensor) {
      const auto sparse_dim = tensor.sparse_dim();
      AT_ASSERT(sparse_dim <= 4);
      for (const auto i : c10::irange(4)) {
        if (i < sparse_dim) {
          data_[i] = tensor.size(i);
        }
      }
      const auto dense_dim = tensor.dense_dim();
      AT_ASSERT(dense_dim <= 4);
      for (const auto i : c10::irange(4)) {
        if (i < dense_dim) {
          data_[i + 4] = tensor.size(sparse_dim + i);
        }
      }
      data_[8] = tensor._nnz();
    }

    std::vector<int64_t> sizes() const {
      std::vector<int64_t> sizes;
      // Sparse sizes
      for (const auto i : c10::irange(4)) {
        if (data_[i] <= 0) {
          break;
        }
        sizes.push_back(data_[i]);
      }
      // Dense sizes
      for (const auto i : c10::irange(4, 8)) {
        if (data_[i] <= 0) {
          break;
        }
        sizes.push_back(data_[i]);
      }
      return sizes;
    }

    int64_t nnz() const {
      return data_[8];
    }

   protected:
    at::Tensor metadata_;
    int64_t* data_;
  };

  // Sparse allreduce is implemented with allgather on indices and values.
  // Every process then sums the resulting sparse tensors locally.
  // The nnz for sparse tensors may be different across processes, so first
  // we run allgather on the nnz, and then allgather with max(nnz).
  at::Tensor allreduce(std::vector<at::Tensor>& tensors) {
    // TODO: This is a massive hack!  There is some confusion about
    // Variable/Tensor inside the body of this function.  Turning off
    // grad smooths over the confusion for now.  This fixes
    // test/test_c10d_gloo.py ProcessGroupGlooTest.test_sparse_allreduce_basics
    //
    // The correct fix is to stop allocating tensors that are not variables,
    // but to conveniently do this c10d must depend on torch not ATen
    at::AutoDispatchBelowAutograd guard;
    auto input = tensors[0];

    // Perform local reduction if we have multiple inputs.
    for (const auto i : c10::irange(1, tensors.size())) {
      input += tensors[i];
    }

    // Need to coalesce before we can access indices and values.
    input = input.coalesce();

    // Gather metadata information from all ranks.
    auto metadata = allgather_metadata(input);

    // Sanity check dimensionality across ranks.
    {
      const auto expected = metadata[context->rank].sizes();
      for (const auto i : c10::irange(context->size)) {
        if (i == context->rank) {
          continue;
        }
        const auto actual = metadata[i].sizes();
        TORCH_CHECK(actual == expected, "Sparse dimensions do not match");
      }
    }

    // Gather all indices and all values.
    auto indices = allgather_indices(input, metadata);
    auto values = allgather_values(input, metadata);

    // Perform global reduction.
    AT_ASSERT(static_cast<int>(indices.size()) == context->size);
    AT_ASSERT(static_cast<int>(values.size()) == context->size);
    auto output = at::sparse_coo_tensor(
        indices[0], values[0], input.sizes(), input.options());
    for (const auto i : c10::irange(1, context->size)) {
      output += at::sparse_coo_tensor(
          indices[i], values[i], input.sizes(), input.options());
    }

    // Coalesce for good measure.
    return output.coalesce();
  }

  void run() override {
    auto output = allreduce(inputs);

    // This copy is needed when we run a multi-gpu version of reduce (multiple
    // inputs per rank).
    for (const auto i : c10::irange(inputs.size())) {
      inputs[i].copy_(output);
    }
  }

 private:
  std::vector<SparseTensorMetadata> allgather_metadata(
      const at::Tensor& tensor) {
    auto buffer =
        at::zeros({context->size, SparseTensorMetadata::dim}, at::kLong);

    // Prepare metadata vector (1 entry per rank)
    std::vector<SparseTensorMetadata> metadata;
    metadata.reserve(context->size);
    for (const auto i : c10::irange(context->size)) {
      metadata.emplace_back(buffer.select(0, i));
    }

    // Populate data for this rank
    metadata[context->rank].populate_from_sparse_tensor(tensor);

    // Allgather metadata
    gloo::AllgatherOptions opts(context);
    opts.setOutput(buffer.mutable_data_ptr<int64_t>(), buffer.numel());
    opts.setTag(tag);
    gloo::allgather(opts);

    return metadata;
  }

  std::vector<at::Tensor> allgather_indices(
      const at::Tensor& tensor,
      const std::vector<SparseTensorMetadata>& metadata) {
    const auto sparseDim = tensor.sparse_dim();

    std::vector<size_t> counts(context->size);
    size_t totalSize = 0;
    for (const auto i : c10::irange(metadata.size())) {
      counts[i] = metadata[i].nnz() * sparseDim;
      totalSize += counts[i];
    }

    auto output = at::empty({static_cast<int64_t>(totalSize)}, at::kLong);

    // tensors copied from cuda may not be contiguous, get a contiguous
    // tensor before use its data_ptr
    auto input = tensor.indices().contiguous();

    // Allgatherv indices.
    gloo::AllgathervOptions opts(context);
    opts.setInput(
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        const_cast<int64_t*>(input.const_data_ptr<int64_t>()),
        input.numel());
    opts.setOutput(output.mutable_data_ptr<int64_t>(), counts);
    opts.setTag(tag);
    gloo::allgatherv(opts);

    // Compile indices tensor per rank.
    std::vector<at::Tensor> indices;
    indices.reserve(metadata.size());
    int64_t offset = 0;
    for (const auto& i : metadata) {
      const auto nnz = i.nnz();
      const auto numel = sparseDim * nnz;
      indices.push_back(
          output.narrow(0, offset, numel).reshape({sparseDim, nnz}));
      offset += numel;
    }

    return indices;
  }

  std::vector<at::Tensor> allgather_values(
      const at::Tensor& tensor,
      const std::vector<SparseTensorMetadata>& metadata) {
    // There are nnz #dense_dim()-dimensional tensors per rank.
    const auto valueShape = tensor.sizes().slice(tensor.sparse_dim());
    int64_t denseNumel = 1;
    for (auto dim : valueShape) {
      denseNumel *= dim;
    }

    std::vector<size_t> counts(context->size);
    int64_t totalSize = 0;
    for (const auto i : c10::irange(metadata.size())) {
      counts[i] = metadata[i].nnz() * denseNumel;
      totalSize += static_cast<int64_t>(counts[i]);
    }

    auto output = at::empty({totalSize}, tensor.scalar_type());

    // Allgatherv indices.
    gloo::AllgathervOptions opts(context);
    // tensors copied from cuda may not be contiguous, get a contiguous
    // tensor before use its data_ptr
    at::Tensor valueTensor = tensor.values().contiguous();
    GENERATE_ALL_TYPES(valueTensor.scalar_type(), setInput, opts, valueTensor);
    GENERATE_ALL_TYPES(
        valueTensor.scalar_type(), setOutput, opts, output, counts);
    opts.setTag(tag);
    gloo::allgatherv(opts);

    // Compile values tensor per rank.
    std::vector<at::Tensor> values;
    values.reserve(metadata.size());
    int64_t offset = 0;
    for (const auto& i : metadata) {
      const auto nnz = i.nnz();
      const auto numel = denseNumel * nnz;
      auto tensorShape = std::vector<int64_t>({(int64_t)nnz});
      std::copy(
          valueShape.begin(),
          valueShape.end(),
          std::back_inserter(tensorShape));
      values.push_back(output.narrow(0, offset, numel).reshape(tensorShape));
      offset += numel;
    }

    return values;
  }
};

class AsyncAllreduceCUDAWork : public AsyncAllreduceWork {
 public:
  AsyncAllreduceCUDAWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      ReduceOp reduceOp,
      uint32_t tag,
      uint64_t seq)
      : AsyncAllreduceWork(context, inputs, std::move(reduceOp), tag, seq) {
    initializeStreamsEvents(inputs, streams, events);

    // Kick off copy from CUDA tensors to pinned CPU tensors.
    tmp.reserve(inputs.size());
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(streams[i]);
      tmp.push_back(pinnedLike(inputs[i]).copy_(inputs[i], true));
    }
  }

  void run() override {
    // Synchronize with copy operations.
    for (const auto i : c10::irange(inputs.size())) {
      streams[i].synchronize();
    }

    // Run allreduce on host side tensors.
    allreduce(tmp);

    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(streams[i]);
      inputs[i].copy_(tmp[i], /* non_blocking */ true);
      events[i].record(streams[i]);
    }
  }

  void synchronize() override {
    // Synchronize with the copy back to CUDA tensors.
    for (const auto i : c10::irange(inputs.size())) {
      c10::Device device = inputs[i].device();
      events[i].block(
          c10::impl::VirtualGuardImpl(device.type()).getStream(device));
    }
  }

  std::vector<at::Tensor> tmp;
  std::vector<c10::Stream> streams{};
  std::vector<c10::Event> events{};
};

class AsyncSparseAllreduceCUDAWork : public AsyncSparseAllreduceWork {
 public:
  AsyncSparseAllreduceCUDAWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      uint32_t tag,
      uint64_t seq)
      : AsyncSparseAllreduceWork(context, inputs, tag, seq) {
    initializeStreamsEvents(inputs, streams, events);

    // Kick off copy from CUDA tensors to CPU tensors.
    // Note that both coalescing the sparse tensor and copying it to CPU
    // memory must be performed asynchronously, or we block the caller.
    tmp.reserve(inputs.size());
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(streams[i]);
      tmp.push_back(
          inputs[i].coalesce().to(at::DeviceType::CPU, /*non_blocking=*/true));
    }
  }

  void run() override {
    // Synchronize with copy operations.
    for (const auto i : c10::irange(inputs.size())) {
      streams[i].synchronize();
    }

    // Run allreduce on host side tensors.
    auto output = allreduce(tmp);

    // Kick off copy back to the CUDA tensors.
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(streams[i]);
      inputs[i].copy_(output, /*non_blocking=*/true);
      events[i].record(streams[i]);
    }
  }

  void synchronize() override {
    // Synchronize with the copy back to CUDA tensors.
    for (const auto i : c10::irange(inputs.size())) {
      c10::Device device = inputs[i].device();
      events[i].block(
          c10::impl::VirtualGuardImpl(device.type()).getStream(device));
    }
  }

  std::vector<at::Tensor> tmp{};
  std::vector<c10::Stream> streams{};
  std::vector<c10::Event> events{};
};

} // namespace

c10::intrusive_ptr<Work> ProcessGroupGloo::allreduce(
    std::vector<at::Tensor>& inputs,
    const AllreduceOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupGloo::allreduce: " + msg);
  };

  assertNonEmpty(invalidArgument, inputs);
  assertLayoutMatch(invalidArgument, inputs);
  assertTypeAndSizesMatch(invalidArgument, inputs);

  const auto& device = inputs[0].device();
  switch (device.type()) {
    case at::kCPU:
      break;
    case at::kCUDA:
      // If the user gave us a CUDA tensor then CUDA must be loaded.
      TORCH_INTERNAL_ASSERT(at::hasCUDA());
      break;
    default:
      invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  const auto& layout = inputs[0].layout();
  if (layout == c10::kSparse && opts.reduceOp != ReduceOp::SUM) {
    invalidArgument(
        "unsupported reduction operation "
        "(allreduce of sparse tensors only works with ReduceOp.SUM)");
  }

  c10::intrusive_ptr<AsyncWork> work;
  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;
  if (device.type() == at::kCPU) {
    if (layout == c10::kStrided) {
      work = c10::make_intrusive<AsyncAllreduceWork>(
          std::move(context), inputs, opts.reduceOp, tag, seq_);
    } else if (layout == c10::kSparse) {
      work = c10::make_intrusive<AsyncSparseAllreduceWork>(
          std::move(context), inputs, tag, seq_);
    } else {
      invalidArgument("unsupported layout");
    }
  } else if (device.type() == at::kCUDA) {
    if (layout == c10::kStrided) {
      work = c10::make_intrusive<AsyncAllreduceCUDAWork>(
          std::move(context), inputs, opts.reduceOp, tag, seq_);
    } else if (layout == c10::kSparse) {
      work = c10::make_intrusive<AsyncSparseAllreduceCUDAWork>(
          std::move(context), inputs, tag, seq_);
    } else {
      invalidArgument("unsupported layout");
    }
  } else {
    TORCH_CHECK(false, "Invalid backend");
  }

  enqueue(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupGloo::allreduce_sparse(
    std::vector<at::Tensor>& inputs,
    const AllreduceOptions& opts) {
  // all reduce sparse calls into default allreduce which
  // implemented with all_gathering indices and values
  // we do ths we do not have a native cuda implementation
  return allreduce(inputs, opts);
}

c10::intrusive_ptr<Work> ProcessGroupGloo::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupGloo::allreduce_coalesced: " + msg);
  };
  assertNonEmpty(invalidArgument, tensors);

  // tensors will be flattened and concatenated (coalesced). This means that
  // input
  // tensors must have the same device, layout and type.
  assertLayoutMatch(invalidArgument, tensors);
  if (!std::all_of(tensors.begin(), tensors.end(), [&](at::Tensor& t) {
        return t.options().type_equal(tensors[0].options());
      })) {
    invalidArgument("tensors must all have the same type");
  }
  if (!std::all_of(tensors.begin(), tensors.end(), [&](at::Tensor& t) {
        return t.device() == tensors[0].device();
      })) {
    invalidArgument("tensors must all be on the same device");
  }

  const c10::Device& device = tensors[0].device();
  const c10::Layout& layout = tensors[0].layout();

  // invalid arguments are detected early here before any calls to nextTag()
  // which result in the collectiveCounter_ being incremented.
  switch (device.type()) {
    case c10::kCPU:
      break;
    default:
      invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  switch (layout) {
    case c10::kStrided:
      break;
    default:
      invalidArgument("unsupported layout");
  }

  c10::intrusive_ptr<AsyncWork> work;
  const uint32_t tag = nextTag();
  std::shared_ptr<gloo::Context> context = getContext(tag);
  ++seq_;
  if (device.type() == c10::kCPU) {
    if (layout == c10::kStrided) {
      work = c10::make_intrusive<AsyncAllreduceCoalescedWork>(
          std::move(context), tensors, opts.reduceOp, tag, seq_);
    } else {
      invalidArgument("unsupported layout");
    }
  } else {
    TORCH_CHECK(false, "Invalid backend");
  }
  enqueue(work);
  return work;
}

namespace {

class AsyncReduceWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncReduceWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      int rootRank,
      int rootTensor,
      ReduceOp reduceOp,
      uint32_t tag,
      uint64_t seq)
      : ProcessGroupGloo::AsyncWork(
            {inputs},
            OpType::REDUCE,
            seq,
            "gloo:reduce",
            inputs),
        context(context),
        inputs(inputs),
        rootRank(rootRank),
        rootTensor(rootTensor),
        reduceOp(std::move(reduceOp)),
        tag(tag) {}

  std::shared_ptr<gloo::Context> context;
  std::vector<at::Tensor> inputs{};
  const int rootRank;
  const int rootTensor;
  const ReduceOp reduceOp;
  const uint32_t tag;

  void reduce(std::vector<at::Tensor>& tensors) {
    const auto& scalarType = tensors[0].scalar_type();
    gloo::ReduceOptions opts(context);
    opts.setRoot(rootRank);
    opts.setTag(tag);
    opts.setReduceFunction(getFunction(scalarType, reduceOp));
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, tensors[0]);
    gloo::reduce(opts);
  }

  void run() override {
    reduce(inputs);
  }

 protected:
  template <typename T>
  void getFunction(gloo::ReduceOptions::Func& fn, const ReduceOp op) {
    fn = toFunction<T>(op);
  }

  gloo::ReduceOptions::Func getFunction(
      const at::ScalarType& dtype,
      const ReduceOp& op) {
    gloo::ReduceOptions::Func fn;
    GENERATE_ALL_TYPES(dtype, getFunction, fn, op);
    return fn;
  }
};

class AsyncReduceCUDAWork : public AsyncReduceWork {
 public:
  AsyncReduceCUDAWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      int rootRank,
      int rootTensor,
      ReduceOp reduceOp,
      uint32_t tag,
      uint64_t seq)
      : AsyncReduceWork(
            context,
            inputs,
            rootRank,
            rootTensor,
            std::move(reduceOp),
            tag,
            seq) {
    initializeStreamsEvents(inputs, streams, events);

    // Kick off copy from CUDA tensors to pinned CPU tensors.
    tmp.reserve(inputs.size());
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(streams[i]);
      tmp.push_back(pinnedLike(inputs[i]).copy_(inputs[i], true));
    }
  }

  void run() override {
    // Synchronize with copy operations.
    for (const auto i : c10::irange(inputs.size())) {
      streams[i].synchronize();
    }

    // Run reduce on host side tensors.
    reduce(tmp);

    // Kick off copy back to the CUDA tensors.
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(streams[i]);
      inputs[i].copy_(tmp[i], /* non_blocking */ true);
      events[i].record(streams[i]);
    }
  }

  void synchronize() override {
    // Synchronize with the copy back to CUDA tensors.
    for (const auto i : c10::irange(inputs.size())) {
      c10::Device device = inputs[i].device();
      events[i].block(
          c10::impl::VirtualGuardImpl(device.type()).getStream(device));
    }
  }

  std::vector<at::Tensor> tmp{};
  std::vector<c10::Stream> streams{};
  std::vector<c10::Event> events{};
};

} // namespace

c10::intrusive_ptr<Work> ProcessGroupGloo::reduce(
    std::vector<at::Tensor>& inputs,
    const ReduceOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupGloo::reduce: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);
  assertRootTensor(
      invalidArgument, opts.rootTensor, static_cast<int64_t>(inputs.size()));
  assertSingleElement(invalidArgument, inputs);
  assertDense(invalidArgument, inputs);

  const auto& device = inputs[0].device();
  switch (device.type()) {
    case at::kCPU:
      break;
    case at::kCUDA:
      // If the user gave us a CUDA tensor then CUDA must be loaded.
      TORCH_INTERNAL_ASSERT(at::hasCUDA());
      break;
    default:
      invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  c10::intrusive_ptr<AsyncReduceWork> work;
  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;
  if (device.type() == at::kCPU) {
    work = c10::make_intrusive<AsyncReduceWork>(
        std::move(context),
        inputs,
        opts.rootRank,
        opts.rootTensor,
        opts.reduceOp,
        tag,
        seq_);
  } else if (device.type() == at::kCUDA) {
    work = c10::make_intrusive<AsyncReduceCUDAWork>(
        std::move(context),
        inputs,
        opts.rootRank,
        opts.rootTensor,
        opts.reduceOp,
        tag,
        seq_);
  } else {
    TORCH_CHECK(false, "Invalid backend");
  }
  enqueue(work);
  return work;
}

namespace {

class AsyncAllgatherWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncAllgatherWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      uint32_t tag,
      uint64_t seq)
      : ProcessGroupGloo::AsyncWork(
            outputs,
            OpType::ALLGATHER,
            seq,
            "gloo:all_gather",
            inputs),
        context(context),
        outputs(outputs),
        inputs(inputs),
        tag(tag) {}

  std::shared_ptr<gloo::Context> context;
  std::vector<std::vector<at::Tensor>> outputs{};
  std::vector<at::Tensor> inputs{};
  const uint32_t tag;

  void allgather(
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs) {
    const auto& scalarType = inputs[0].scalar_type();
    gloo::AllgatherOptions opts(context);
    opts.setTag(tag);

    // Use single flattened input tensor.
    at::Tensor flatInputTensor = flattenDenseTensors(inputs);
    GENERATE_ALL_TYPES(scalarType, setInput, opts, flatInputTensor);

    // Use single flat output tensor.
    // The first dimension corresponds to the index into outputs[N],
    // so copying into the actual output later is easy.
    at::Tensor flatOutputTensor = newLikeFlat(outputs[0]);
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, flatOutputTensor);
    gloo::allgather(opts);

    // Unflatten into output tensors.
    for (auto& outputgroup : outputs) {
      for (const auto j : c10::irange(outputgroup.size())) {
        outputgroup[j].copy_(flatOutputTensor[static_cast<int64_t>(j)]);
      }
    }
  }

  void run() override {
    allgather(outputs, inputs);
  }
};

// Note: current CUDA implementation holds the assumption that the
// tensors in the nested output tensor vectors are on the same device.
class AsyncAllgatherCUDAWork : public AsyncAllgatherWork {
 public:
  AsyncAllgatherCUDAWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      uint32_t tag,
      uint64_t seq)
      : AsyncAllgatherWork(context, outputs, inputs, tag, seq) {
    initializeStreamsEvents(inputs, inputStreams, inputEvents);
    initializeStreamsEvents(outputs, outputStreams, outputEvents);

    // Kick off copy from CUDA tensors to pinned CPU tensors.
    tmpInputs.reserve(inputs.size());
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(inputStreams[i]);
      tmpInputs.push_back(pinnedLike(inputs[i]).copy_(inputs[i], true));
    }

    tmpOutputs.resize(outputs.size());
    for (const auto i : c10::irange(outputs.size())) {
      tmpOutputs[i].reserve(outputs[i].size());
      for (const auto j : c10::irange(outputs[i].size())) {
        tmpOutputs[i].push_back(pinnedLike(outputs[i][j]));
      }
    }
  }

  void run() override {
    // Synchronize with copy operations.
    for (const auto i : c10::irange(inputs.size())) {
      inputStreams[i].synchronize();
    }

    for (const auto i : c10::irange(outputs.size())) {
      outputStreams[i].synchronize();
    }

    // Run allgather on host side tensors.
    allgather(tmpOutputs, tmpInputs);

    // Kick off copy back to the CUDA tensors.
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(outputs.size())) {
      guard.reset_stream(outputStreams[i]);
      for (const auto j : c10::irange(outputs[i].size())) {
        outputs[i][j].copy_(tmpOutputs[i][j], /* non_blocking */ true);
      }
      outputEvents[i].record(outputStreams[i]);
    }
  }

  void synchronize() override {
    // Synchronize with the copy back to CUDA tensors.
    for (const auto i : c10::irange(outputs.size())) {
      c10::Device device = outputs[i][0].device();
      outputEvents[i].block(
          c10::impl::VirtualGuardImpl(device.type()).getStream(device));
    }
  }

  std::vector<at::Tensor> tmpInputs{};
  std::vector<c10::Stream> inputStreams{};
  std::vector<c10::Event> inputEvents{};

  std::vector<std::vector<at::Tensor>> tmpOutputs{};
  std::vector<c10::Stream> outputStreams{};
  std::vector<c10::Event> outputEvents{};
};

// A work that takes an lambda on construction and calls it on wait.
// It is useful for add a continuation to another work, and/or
// composing multiple works together.
class LambdaWork : public Work {
 public:
  LambdaWork(std::function<void(void)> fn) : fn_(std::move(fn)) {}

  bool wait(std::chrono::milliseconds /* unused */) override {
    fn_();
    return true;
  }

 private:
  std::function<void(void)> fn_;
};

} // namespace

c10::intrusive_ptr<Work> ProcessGroupGloo::_reduce_scatter_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const ReduceScatterOptions& opts) {
  std::vector<at::Tensor> outputTensors = {outputTensor};
  std::vector<at::Tensor> inputTensors = {inputTensor};
  return reduce_scatter_tensor_coalesced(outputTensors, inputTensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupGloo::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const ReduceScatterOptions& opts) {
  if (outputTensors.size() != inputTensors.size()) {
    TORCH_CHECK(
        false, "requires input/output tensor lists to have the same length");
  }
  const auto rank = getRank();
  const auto worldSize = getSize();
  std::vector<at::Tensor> buffers;
  for (const auto i : c10::irange(inputTensors.size())) {
    auto inputShape = inputTensors[i].sizes().vec();
    auto outputShape = outputTensors[i].sizes().vec();
    TORCH_CHECK_EQ(outputTensors[i].dtype(), inputTensors[i].dtype());
    TORCH_CHECK_EQ(outputShape[0] * worldSize, inputShape[0]);
    for (size_t i = 1; i < outputShape.size(); ++i) {
      TORCH_CHECK_EQ(outputShape[i], inputShape[i]);
    }
    buffers.push_back(inputTensors[i].clone());
  }
  std::vector<c10::intrusive_ptr<Work>> works;
  for (const auto i : c10::irange(buffers.size())) {
    std::vector<at::Tensor> inp = {buffers[i]};
    AllreduceOptions arOpts;
    arOpts.reduceOp = opts.reduceOp;
    works.push_back(allreduce(inp));
  }
  return c10::make_intrusive<LambdaWork>(
      [rank, worldSize, buffers, outputTensors, works = std::move(works)]() {
        for (const auto i : c10::irange(outputTensors.size())) {
          works[i]->wait();
          outputTensors[i].copy_(buffers[i].chunk(worldSize)[rank]);
        }
      });
}

c10::intrusive_ptr<Work> ProcessGroupGloo::_allgather_base(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const AllgatherOptions& opts) {
  auto tensor_list = at::chunk(output_tensor, this->getSize(), 0);
  std::vector<std::vector<at::Tensor>> outputs = {tensor_list};
  std::vector<at::Tensor> inputs = {input_tensor};
  return this->allgather(outputs, inputs, opts);
}
// Note: current CUDA implementation holds the assumption that the
// tensors in the nested output tensor vectors are on the same device.
c10::intrusive_ptr<Work> ProcessGroupGloo::allgather(
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs,
    const AllgatherOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupGloo::allgather: " + msg);
  };

  if (inputs.empty()) {
    invalidArgument("requires non-empty input tensor list");
  }

  if (inputs.size() != outputs.size()) {
    invalidArgument(
        "requires input/output tensor lists to have the same length");
  }

  for (const auto i : c10::irange(outputs.size())) {
    const auto expected = inputs.size() * getSize();
    const auto actual = outputs[i].size();
    if (actual != expected) {
      invalidArgument(
          "invalid output tensor list at index " + std::to_string(i) +
          " (expected length " + std::to_string(expected) + ", got " +
          std::to_string(actual) + ")");
    }
  }

  assertDense(invalidArgument, inputs);

  // Expect all input/output tensors to have the same type and sizes
  const auto& options = inputs[0].options();
  const auto& sizes = inputs[0].sizes();
  assertTypeAndSizesMatch(invalidArgument, inputs, options, sizes);
  for (const auto& output : outputs) {
    assertTypeAndSizesMatch(invalidArgument, output, options, sizes);
  }

  const auto& device = inputs[0].device();
  switch (device.type()) {
    case at::kCPU:
      break;
    case at::kCUDA:
      // If the user gave us a CUDA tensor then CUDA must be loaded.
      TORCH_INTERNAL_ASSERT(at::hasCUDA());
      break;
    default:
      invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  c10::intrusive_ptr<AsyncAllgatherWork> work;
  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;
  if (device.type() == at::kCPU) {
    work = c10::make_intrusive<AsyncAllgatherWork>(
        std::move(context), outputs, inputs, tag, seq_);
  } else if (device.type() == at::kCUDA) {
    work = c10::make_intrusive<AsyncAllgatherCUDAWork>(
        std::move(context), outputs, inputs, tag, seq_);
  } else {
    TORCH_CHECK(false, "Invalid backend");
  }
  enqueue(work);
  return work;
}

namespace {

class AsyncAllgatherCoalescedWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncAllgatherCoalescedWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<std::vector<at::Tensor>>& output_lists,
      std::vector<at::Tensor>& input_list,
      uint32_t tag,
      uint64_t seq)
      : ProcessGroupGloo::AsyncWork(
            output_lists,
            OpType::ALLGATHER_COALESCED,
            seq,
            "gloo:all_gather",
            input_list),
        context(context),
        output_lists(output_lists),
        input_list(input_list),
        tag(tag) {}

  std::shared_ptr<gloo::Context> context;
  std::vector<std::vector<at::Tensor>> output_lists{};
  std::vector<at::Tensor> input_list{};
  const uint32_t tag;

  void allgather_coalesced() {
    assert(!output_lists.empty());
    assert(!output_lists[0].empty());
    assert(!input_list.empty());

    const auto& scalarType = input_list[0].scalar_type();
    gloo::AllgatherOptions opts(context);
    opts.setTag(tag);

    // Use single flattened input tensor.
    at::Tensor flatInputTensor = flattenDenseTensors(input_list);
    GENERATE_ALL_TYPES(scalarType, setInput, opts, flatInputTensor);

    // Compute total number of elements we need to allocate for all tensors
    // requested.
    int64_t output_numel = 0;
    for (const auto& t : output_lists[0]) {
      output_numel += t.numel();
    }
    output_numel *= static_cast<int64_t>(output_lists.size());
    // Use single flat output tensor.
    at::Tensor flatOutputTensor =
        at::empty({output_numel}, output_lists[0][0].options());
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, flatOutputTensor);
    gloo::allgather(opts);

    int64_t current_element = 0;
    for (auto& output_list : output_lists) {
      for (auto& output_tensor : output_list) {
        output_tensor.copy_(
            flatOutputTensor.narrow(0, current_element, output_tensor.numel())
                .reshape(output_tensor.sizes()),
            true);
        current_element += output_tensor.numel();
      }
    }
  }

  void run() override {
    allgather_coalesced();
  }
};

} // namespace

c10::intrusive_ptr<Work> ProcessGroupGloo::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& output_lists,
    std::vector<at::Tensor>& input_list,
    const AllgatherOptions& /* unused */) {
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupGloo::allgather_coalesced: " + msg);
  };

  if (input_list.empty()) {
    invalidArgument("requires non-empty input tensor list");
  }

  if (output_lists.size() != static_cast<size_t>(getSize())) {
    invalidArgument("output lists should be equal to world size");
  }

  assertSameDevice(invalidArgument, input_list);

  // Expect i'th tensor of each list from 'output_lists' match i'th tensor
  // from 'input_list' in type and size.
  for (const auto& output_list : output_lists) {
    if (output_list.size() != input_list.size()) {
      invalidArgument(
          "invalid output size: (expected length " +
          std::to_string(input_list.size()) + ", got " +
          std::to_string(output_list.size()) + ")");
    }
    for (const auto i : c10::irange(output_list.size())) {
      const auto expected = input_list[i].sizes();
      const auto actual = output_list[i].sizes();
      if (actual != expected) {
        invalidArgument(
            "invalid size of output tensor at index " + std::to_string(i) +
            " (expected length " + toString(expected) + ", got " +
            toString(actual) + ")");
      }
      if (!input_list[i].options().type_equal(output_list[i].options())) {
        invalidArgument(
            "invalid tensor type at index " + std::to_string(i) +
            " (expected " + input_list[i].toString() + ", got " +
            output_list[i].toString() + ")");
      }
    }
  }

  assertDense(invalidArgument, input_list);

  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;
  auto work = c10::make_intrusive<AsyncAllgatherCoalescedWork>(
      std::move(context), output_lists, input_list, tag, seq_);
  enqueue(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupGloo::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const AllgatherOptions& opts) {
  TORCH_CHECK_EQ(outputs.size(), inputs.size());
  std::vector<std::vector<at::Tensor>> output_lists(getSize());
  for (auto& output : outputs) {
    auto chunks = output.chunk(getSize());
    for (const auto i : c10::irange(output_lists.size())) {
      output_lists[i].push_back(std::move(chunks[i]));
    }
  }
  return allgather_coalesced(output_lists, inputs, opts);
}

namespace {

class AsyncGatherWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncGatherWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      int root,
      uint32_t tag,
      uint64_t seq)
      : ProcessGroupGloo::AsyncWork(
            outputs,
            OpType::GATHER,
            seq,
            "gloo:gather",
            inputs),
        context(context),
        outputs(outputs),
        inputs(inputs),
        root(root),
        tag(tag) {}

  std::shared_ptr<gloo::Context> context;
  std::vector<std::vector<at::Tensor>> outputs{};
  std::vector<at::Tensor> inputs{};
  const int root;
  const uint32_t tag;

  void gather(
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs) {
    const auto scalarType = inputs[0].scalar_type();
    gloo::GatherOptions opts(context);
    opts.setRoot(root);
    opts.setTag(tag);

    // Set single temporary tensor on root process.
    // This is later scattered to the separate output tensors.
    at::Tensor flatOutputTensor;
    if (context->rank == root) {
      flatOutputTensor = newLikeFlat(outputs[0]);
      GENERATE_ALL_TYPES(scalarType, setOutput, opts, flatOutputTensor);
    }

    // Set single input tensor on all processes.
    GENERATE_ALL_TYPES(scalarType, setInput, opts, inputs[0]);
    gloo::gather(opts);

    // Unflatten into output tensors on root process.
    if (context->rank == root) {
      for (const auto i : c10::irange(outputs[0].size())) {
        outputs[0][i].copy_(flatOutputTensor[static_cast<int64_t>(i)]);
      }
    }
  }

  void run() override {
    gather(outputs, inputs);
  }
};

// Note: current CUDA implementation holds the assumptions:
//     - inputs.size() is 1
//     - outputs.size() is 1
//     - the size of the nested output tensors is world size, i.e.,
//       outputs[0].size, is world size
class AsyncGatherCUDAWork : public AsyncGatherWork {
 public:
  AsyncGatherCUDAWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      int root,
      uint32_t tag,
      uint64_t seq)
      : AsyncGatherWork(context, outputs, inputs, root, tag, seq) {
    initializeStreamsEvents(inputs, inputStreams, inputEvents);
    initializeStreamsEvents(outputs, outputStreams, outputEvents);

    // Kick off copy from CUDA tensors to pinned CPU tensors.
    tmpInputs.reserve(inputs.size());
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(inputStreams[i]);
      tmpInputs.push_back(pinnedLike(inputs[i]).copy_(inputs[i], true));
    }

    tmpOutputs.resize(outputs.size());
    for (const auto i : c10::irange(outputs.size())) {
      tmpOutputs[i].reserve(outputs[i].size());
      for (const auto j : c10::irange(outputs[i].size())) {
        tmpOutputs[i].push_back(pinnedLike(outputs[i][j]));
      }
    }
  }

  void run() override {
    // Synchronize with copy operations.
    for (const auto i : c10::irange(inputs.size())) {
      inputStreams[i].synchronize();
    }

    for (const auto i : c10::irange(outputs.size())) {
      outputStreams[i].synchronize();
    }

    // Run gather on host side tensors.
    gather(tmpOutputs, tmpInputs);

    // Kick off copy back to the CUDA tensors.
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(outputs.size())) {
      guard.reset_stream(outputStreams[i]);
      for (const auto j : c10::irange(outputs[i].size())) {
        outputs[i][j].copy_(tmpOutputs[i][j], /* non_blocking */ true);
      }
      outputEvents[i].record(outputStreams[i]);
    }
  }

  void synchronize() override {
    // Synchronize with the copy back to CUDA tensors.
    for (const auto i : c10::irange(outputs.size())) {
      c10::Device device = outputs[i][0].device();
      outputEvents[i].block(
          c10::impl::VirtualGuardImpl(device.type()).getStream(device));
    }
  }

  std::vector<at::Tensor> tmpInputs{};
  std::vector<c10::Stream> inputStreams{};
  std::vector<c10::Event> inputEvents{};

  std::vector<std::vector<at::Tensor>> tmpOutputs{};
  std::vector<c10::Stream> outputStreams{};
  std::vector<c10::Event> outputEvents{};
};

} // namespace

c10::intrusive_ptr<Work> ProcessGroupGloo::gather(
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs,
    const GatherOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupGloo::gather: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);
  assertSingleElementInput(invalidArgument, inputs);
  assertDense(invalidArgument, inputs);

  if (getRank() == opts.rootRank) {
    if (outputs.size() != 1) {
      std::stringstream ss;
      ss << "requires a single-element output list containing a list with "
         << getSize() << " tensors.";
      invalidArgument(ss.str());
    } else if (outputs[0].size() != static_cast<size_t>(getSize())) {
      std::stringstream ss;
      ss << "Incorrect output list size " << outputs[0].size()
         << ". Output list size should be " << getSize()
         << ", same as size of the process group.";
      invalidArgument(ss.str());
    }

    const auto& options = inputs[0].options();
    const auto& sizes = inputs[0].sizes();
    assertTypeAndSizesMatch(invalidArgument, outputs[0], options, sizes);
  } else {
    if (!outputs.empty()) {
      invalidArgument("requires empty output on non-root");
    }
  }

  const auto& device = inputs[0].device();
  switch (device.type()) {
    case at::kCPU:
      break;
    case at::kCUDA:
      // If the user gave us a CUDA tensor then CUDA must be loaded.
      TORCH_INTERNAL_ASSERT(at::hasCUDA());
      break;
    default:
      invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  c10::intrusive_ptr<AsyncGatherWork> work;
  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;
  if (device.type() == at::kCPU) {
    work = c10::make_intrusive<AsyncGatherWork>(
        std::move(context), outputs, inputs, opts.rootRank, tag, seq_);
  } else if (device.type() == at::kCUDA) {
    work = c10::make_intrusive<AsyncGatherCUDAWork>(
        std::move(context), outputs, inputs, opts.rootRank, tag, seq_);
  } else {
    TORCH_CHECK(false, "Invalid backend");
  }
  enqueue(work);
  return work;
}

namespace {

class AsyncScatterWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncScatterWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs,
      int root,
      uint32_t tag,
      uint64_t seq)
      : ProcessGroupGloo::AsyncWork(
            {outputs},
            OpType::SCATTER,
            seq,
            "gloo:scatter",
            !inputs.empty() ? std::optional<std::vector<at::Tensor>>(inputs[0])
                            : std::nullopt),
        context(context),
        outputs(outputs),
        inputs(inputs),
        root(root),
        tag(tag) {}

  std::shared_ptr<gloo::Context> context;
  std::vector<at::Tensor> outputs{};
  std::vector<std::vector<at::Tensor>> inputs{};
  const int root;
  const uint32_t tag;

  void scatter(
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs) {
    const auto scalarType = outputs[0].scalar_type();
    gloo::ScatterOptions opts(context);
    opts.setRoot(root);
    opts.setTag(tag);

    // Set list of input tensors on root process
    if (context->rank == root) {
      GENERATE_ALL_TYPES(scalarType, setInputs, opts, inputs[0]);
    }

    // Set single output tensor on all processes
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, outputs[0]);
    gloo::scatter(opts);
  }

  void run() override {
    scatter(outputs, inputs);
  }
};

class AsyncScatterCUDAWork : public AsyncScatterWork {
 public:
  AsyncScatterCUDAWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs,
      int root,
      uint32_t tag,
      uint64_t seq)
      : AsyncScatterWork(context, outputs, inputs, root, tag, seq) {
    initializeStreamsEvents(inputs, inputStreams, inputEvents);
    initializeStreamsEvents(outputs, outputStreams, outputEvents);

    // Kick off copy from CUDA tensors to pinned CPU tensors.
    tmpInputs.resize(inputs.size());
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(inputStreams[i]);
      tmpInputs[i].reserve(inputs[i].size());
      for (const auto j : c10::irange(inputs[i].size())) {
        tmpInputs[i].push_back(
            pinnedLike(inputs[i][j]).copy_(inputs[i][j], true));
      }
    }

    tmpOutputs.reserve(outputs.size());
    for (auto& output : outputs) {
      tmpOutputs.push_back(pinnedLike(output));
    }
  }

  void run() override {
    // Synchronize with copy operations.
    for (const auto i : c10::irange(inputs.size())) {
      inputStreams[i].synchronize();
    }
    for (const auto i : c10::irange(outputs.size())) {
      outputStreams[i].synchronize();
    }

    // Run scatter on host side tensors.
    scatter(tmpOutputs, tmpInputs);

    // Kick off copy back to the CUDA tensors.
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(outputs.size())) {
      guard.reset_stream(outputStreams[i]);
      outputs[i].copy_(tmpOutputs[i], /* non_blocking */ true);
      outputEvents[i].record(outputStreams[i]);
    }
  }

  void synchronize() override {
    // Synchronize with the copy back to CUDA tensors.
    for (const auto i : c10::irange(outputs.size())) {
      c10::Device device = outputs[i].device();
      outputEvents[i].block(
          c10::impl::VirtualGuardImpl(device.type()).getStream(device));
    }
  }

  std::vector<at::Tensor> tmpOutputs{};
  std::vector<c10::Stream> outputStreams{};
  std::vector<c10::Event> outputEvents{};

  std::vector<std::vector<at::Tensor>> tmpInputs{};
  std::vector<c10::Stream> inputStreams{};
  std::vector<c10::Event> inputEvents{};
};

} // namespace

c10::intrusive_ptr<Work> ProcessGroupGloo::scatter(
    std::vector<at::Tensor>& outputs,
    std::vector<std::vector<at::Tensor>>& inputs,
    const ScatterOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupGloo::scatter: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);
  assertSingleElementOutput(invalidArgument, outputs);
  assertDense(invalidArgument, outputs);

  if (getRank() == opts.rootRank) {
    if (inputs.size() != 1) {
      std::stringstream ss;
      ss << "requires a single-element input list containing a list with "
         << getSize() << " tensors";
      invalidArgument(ss.str());
    } else if (inputs[0].size() != static_cast<size_t>(getSize())) {
      std::stringstream ss;
      ss << "Incorrect input list size " << inputs[0].size()
         << ". Input list size should be " << getSize()
         << ", same as size of the process group.";
      invalidArgument(ss.str());
    }
    const auto& options = outputs[0].options();
    const auto& sizes = outputs[0].sizes();
    assertTypeAndSizesMatch(invalidArgument, inputs[0], options, sizes);
  } else {
    if (!inputs.empty()) {
      invalidArgument("requires empty input on non-root");
    }
  }

  const auto& device = outputs[0].device();
  switch (device.type()) {
    case at::kCPU:
      break;
    case at::kCUDA:
      // If the user gave us a CUDA tensor then CUDA must be loaded.
      TORCH_INTERNAL_ASSERT(at::hasCUDA());
      break;
    default:
      invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  c10::intrusive_ptr<AsyncScatterWork> work;
  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;
  if (device.type() == at::kCPU) {
    work = c10::make_intrusive<AsyncScatterWork>(
        std::move(context), outputs, inputs, opts.rootRank, tag, seq_);
  } else if (device.type() == at::kCUDA) {
    work = c10::make_intrusive<AsyncScatterCUDAWork>(
        std::move(context), outputs, inputs, opts.rootRank, tag, seq_);
  } else {
    TORCH_CHECK(false, "Invalid backend");
  }
  enqueue(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupGloo::reduce_scatter(
    std::vector<at::Tensor>& outputs,
    std::vector<std::vector<at::Tensor>>& inputs,
    const ReduceScatterOptions& opts) {
  TORCH_CHECK(false, "ProcessGroupGloo does not support reduce_scatter");
}

namespace {

class AsyncAlltoallWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncAlltoallWork(
      const std::shared_ptr<gloo::Context>& context,
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputCounts,
      std::vector<int64_t>& inputCounts,
      uint32_t tag,
      uint64_t seq)
      : ProcessGroupGloo::AsyncWork(
            {{outputTensor}},
            OpType::ALLTOALL,
            seq,
            "gloo:all_to_all",
            std::optional<std::vector<at::Tensor>>({inputTensor})),
        context(context),
        outputTensor(outputTensor),
        inputTensor(inputTensor),
        outputCounts(std::move(outputCounts)),
        inputCounts(std::move(inputCounts)),
        tag(tag) {}

  std::shared_ptr<gloo::Context> context;
  at::Tensor outputTensor;
  at::Tensor inputTensor;
  std::vector<int64_t> outputCounts{};
  std::vector<int64_t> inputCounts{};
  const uint32_t tag;

  void alltoall(at::Tensor& outputTensor, at::Tensor& inputTensor) {
    const auto scalarType = outputTensor.scalar_type();
    if (outputCounts.empty() && inputCounts.empty()) {
      // Gloo alltoall
      gloo::AlltoallOptions opts(context);
      opts.setTag(tag);
      GENERATE_ALL_TYPES(scalarType, setInput, opts, inputTensor);
      GENERATE_ALL_TYPES(scalarType, setOutput, opts, outputTensor);
      gloo::alltoall(opts);
    } else {
      // Gloo alltoallv
      c10d::checkSplitSizes(inputCounts, inputTensor, context->size);
      c10d::checkSplitSizes(outputCounts, outputTensor, context->size);
      std::vector<int64_t> sendCounts(context->size);
      std::vector<int64_t> recvCounts(context->size);
      std::vector<int64_t> sendOffsets(context->size);
      std::vector<int64_t> recvOffsets(context->size);
      c10d::computeLengthsAndOffsets(
          inputCounts, inputTensor, &sendCounts, &sendOffsets);
      c10d::computeLengthsAndOffsets(
          outputCounts, outputTensor, &recvCounts, &recvOffsets);
      gloo::AlltoallvOptions opts(context);
      opts.setTag(tag);
      GENERATE_ALL_TYPES(scalarType, setInput, opts, inputTensor, sendCounts);
      GENERATE_ALL_TYPES(scalarType, setOutput, opts, outputTensor, recvCounts);
      gloo::alltoallv(opts);
    }
  }

  void run() override {
    alltoall(outputTensor, inputTensor);
  }
};

class AsyncAlltoallCUDAWork : public AsyncAlltoallWork {
 public:
  AsyncAlltoallCUDAWork(
      const std::shared_ptr<gloo::Context>& context,
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputCounts,
      std::vector<int64_t>& inputCounts,
      uint32_t tag,
      uint64_t seq)
      : AsyncAlltoallWork(
            context,
            outputTensor,
            inputTensor,
            outputCounts,
            inputCounts,
            tag,
            seq) {
    initializeStreamsEvents({inputTensor}, inputStreams, inputEvents);
    initializeStreamsEvents({outputTensor}, outputStreams, outputEvents);

    // Kick off copy from CUDA tensors to pinned CPU tensors.
    c10::OptionalStreamGuard guard;
    guard.reset_stream(inputStreams.front());
    cpuInput = pinnedLike(inputTensor).copy_(inputTensor, true);

    guard.reset_stream(outputStreams.front());
    cpuOutput = pinnedLike(outputTensor);
  }

  void run() override {
    // Synchronize with copy operations.
    inputStreams.front().synchronize();
    outputStreams.front().synchronize();

    // Run alltoall on host side tensors.
    alltoall(cpuOutput, cpuInput);

    // Kick off copy back to the CUDA tensors.
    c10::OptionalStreamGuard guard;
    guard.reset_stream(outputStreams.front());
    outputTensor.copy_(cpuOutput, /* non_blocking */ true);
    outputEvents.front().record(outputStreams.front());
  }

  void synchronize() override {
    // Synchronize with the copy back to CUDA tensors.
    c10::Device device = outputTensor.device();
    outputEvents.front().block(
        c10::impl::VirtualGuardImpl(device.type()).getStream(device));
  }

  at::Tensor cpuOutput;
  std::vector<c10::Stream> outputStreams{};
  std::vector<c10::Event> outputEvents{};

  at::Tensor cpuInput;
  std::vector<c10::Stream> inputStreams{};
  std::vector<c10::Event> inputEvents{};
};

} // namespace

c10::intrusive_ptr<Work> ProcessGroupGloo::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputCounts,
    std::vector<int64_t>& inputCounts,
    const AllToAllOptions& /* unused */) {
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupGloo::alltoall_base: " + msg);
  };

  TORCH_CHECK(
      outputTensor.device() == inputTensor.device(),
      "output tensor and input tensor must be on the same type of device");
  assertDense(invalidArgument, {outputTensor});
  assertDense(invalidArgument, {inputTensor});

  const auto& device = outputTensor.device();
  c10::intrusive_ptr<AsyncAlltoallWork> work;
  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;

  if (device.type() == at::kCPU) {
    work = c10::make_intrusive<AsyncAlltoallWork>(
        std::move(context),
        outputTensor,
        inputTensor,
        outputCounts,
        inputCounts,
        tag,
        seq_);
  } else if (device.type() == at::kCUDA) {
    work = c10::make_intrusive<AsyncAlltoallCUDAWork>(
        std::move(context),
        outputTensor,
        inputTensor,
        outputCounts,
        inputCounts,
        tag,
        seq_);
  } else {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }
  enqueue(work);
  return work;
}

static at::Tensor& checkSingleTensor(std::vector<at::Tensor>& tensors) {
  if (tensors.size() != 1) {
    TORCH_CHECK(false, "ProcessGroupGloo::send takes a single tensor");
  }
  auto& tensor = tensors[0];
  if (!tensor.is_contiguous()) {
    TORCH_CHECK(false, "input tensor has to be contiguous");
  }
  if (tensor.is_sparse()) {
    TORCH_CHECK(false, "input tensor has to be dense");
  }
  return tensor;
}

static uint32_t checkTag(int32_t tag) {
  TORCH_CHECK(tag >= 0, "Tag must be nonnegative");
  return (uint32_t)tag;
}

c10::intrusive_ptr<Work> ProcessGroupGloo::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  auto& tensor = checkSingleTensor(tensors);
  auto utag = checkTag(tag);
  auto ptr = tensor.const_data_ptr();
  auto size = tensor.numel() * tensor.element_size();

  // Construct unbound buffer.
  auto context = getContext(tag);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  auto buf = context->createUnboundBuffer(const_cast<void*>(ptr), size);
  buf->send(dstRank, utag);
  ++seq_;

  // The work captures the tensor to prevent it being deallocated and
  // the unbound buffer to synchronize on completion of the send.
  return c10::make_intrusive<SendWork>(tensor, std::move(buf), seq_);
}

c10::intrusive_ptr<Work> ProcessGroupGloo::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  auto& tensor = checkSingleTensor(tensors);
  auto utag = checkTag(tag);
  auto ptr = tensor.mutable_data_ptr();
  auto size = tensor.numel() * tensor.element_size();

  // Construct unbound buffer.
  auto context = getContext(tag);
  auto buf = context->createUnboundBuffer(ptr, size);
  buf->recv(srcRank, utag);
  ++seq_;

  // The work captures the tensor to prevent it being deallocated and
  // the unbound buffer to synchronize on completion of the recv.
  return c10::make_intrusive<RecvWork>(
      tensor, std::move(buf), OpType::RECV, seq_, "gloo:recv");
}

c10::intrusive_ptr<Work> ProcessGroupGloo::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  auto& tensor = checkSingleTensor(tensors);
  auto utag = checkTag(tag);
  auto ptr = tensor.mutable_data_ptr();
  auto size = tensor.numel() * tensor.element_size();

  // Construct unbound buffer.
  auto context = getContext(tag);
  auto buf = context->createUnboundBuffer(ptr, size);

  // Build list of ranks that this operation can recv from. In these
  // bindings we don't differentiate between ranks and can receive
  // from any other process in the group.
  std::vector<int> srcRanks;
  srcRanks.resize(size_);
  for (const auto i : c10::irange(size_)) {
    srcRanks.push_back(i);
  }

  buf->recv(srcRanks, utag);
  ++seq_;

  // The work captures the tensor to prevent it being deallocated and
  // the unbound buffer to synchronize on completion of the recv.
  return c10::make_intrusive<RecvWork>(
      tensor,
      std::move(buf),
      OpType::RECVANYSOURCE,
      seq_,
      "gloo:recvAnySource");
}

namespace {

class AsyncBarrierWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncBarrierWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<c10::weak_intrusive_ptr<AsyncWork>> priorWork,
      uint32_t tag,
      uint64_t seq)
      : ProcessGroupGloo::AsyncWork(
            {},
            OpType::BARRIER,
            seq,
            "gloo:barrier",
            std::nullopt),
        context(context),
        priorWork(std::move(priorWork)),
        tag(tag) {}

  std::shared_ptr<gloo::Context> context;
  std::vector<c10::weak_intrusive_ptr<AsyncWork>> priorWork{};
  const uint32_t tag;

  void run() override {
    // Wait on prior work to complete
    for (auto& weakWork : priorWork) {
      auto work = weakWork.lock();
      if (work) {
        work->wait();
      }
    }

    gloo::BarrierOptions opts(context);
    opts.setTag(tag);
    gloo::barrier(opts);
  }
};

} // namespace

c10::intrusive_ptr<Work> ProcessGroupGloo::barrier(const BarrierOptions& opts) {
  std::vector<c10::weak_intrusive_ptr<AsyncWork>> priorWork;

  // Snapshot all in progress and pending work as weak_ptr.
  // When executing a barrier, we need to ensure that all prior work
  // has completed before completing itself.
  {
    std::unique_lock<std::mutex> lock(workMutex_);
    priorWork.insert(
        priorWork.end(), workInProgress_.begin(), workInProgress_.end());
    priorWork.insert(priorWork.end(), workQueue_.begin(), workQueue_.end());
  }

  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;
  auto work = c10::make_intrusive<AsyncBarrierWork>(
      std::move(context), std::move(priorWork), tag, seq_);
  enqueue(work);
  return work;
}

void ProcessGroupGloo::monitoredBarrier(
    const BarrierOptions& opts,
    bool waitAllRanks) {
  C10_LOG_API_USAGE_ONCE("torch.distributed.monitored_barrier");
  // Use default timeout if no timeout was specified.
  auto monitoredBarrierTimeout =
      (opts.timeout == kUnsetTimeout) ? this->options_->timeout : opts.timeout;
  auto rank = this->getRank();
  auto t1 = nextTag();
  auto t2 = nextTag();
  std::vector<at::Tensor> commTensor = {at::tensor({rank})};
  // only enforce timeout on rank 0. This is so that other ranks aren't timed
  // out first, bringing down the job without reporting which rank timed out.
  if (rank != 0) {
    auto sendWork = send(commTensor, 0, static_cast<int>(t1));
    auto recvWork = recv(commTensor, 0, static_cast<int>(t2));
    try {
      sendWork->wait();
      recvWork->wait();
    } catch (const std::exception& e) {
      const std::string error = c10::str(
          "Rank ",
          rank,
          " successfully reached monitoredBarrier, but received errors while waiting",
          " for send/recv from rank 0. Please check rank 0 logs for faulty rank.");
      logAndThrow(
          error, c10::str(error, "\n Original exception: \n", e.what()));
    }
    return;
  }
  auto startTime = std::chrono::steady_clock::now();
  auto worldSize = this->getSize();
  // Mappings of rank to recvWork/sendWork respectively.
  std::map<int, c10::intrusive_ptr<Work>> recvWorkMap;
  std::map<int, c10::intrusive_ptr<Work>> sendWorkMap;
  // Kick off recvWork and wait to unblock sendWork->wait() from non-zero ranks.
  // Failed/hanging ranks will not ack this call, letting rank 0 know about the
  // failure.
  for (const auto dstRank : c10::irange(1, worldSize)) {
    recvWorkMap.emplace(
        dstRank, recv(commTensor, dstRank, static_cast<int>(t1)));
  }

  auto waitLoop = [&](const std::map<int, c10::intrusive_ptr<Work>>& works) {
    std::vector<int> processedRanks;
    for (auto& work : works) {
      bool rankResponded = false;
      try {
        // Note: if waitAllRanks=false, we recompute the time remaining in
        // barrier and use this recomputed time in wait(). However, if
        // waitAllRanks=true, we use the original timeout, since if we use
        // up the entire timeout waiting for response from rank n, then we
        // won't have any timeout left to query ranks beginning with n + 1.
        auto remainingTime =
            getRemainingTime(startTime, monitoredBarrierTimeout, waitAllRanks);
        if (!waitAllRanks) {
          checkRemainingTime(
              monitoredBarrierTimeout, remainingTime, processedRanks, rank);
        }
        work.second->wait(remainingTime);
        rankResponded = true;
      } catch (const std::exception& e) {
        const std::string error = c10::str(
            "[Rank 0]: Rank ",
            work.first,
            " failed to pass monitoredBarrier in ",
            monitoredBarrierTimeout.count(),
            " ms");
        if (waitAllRanks) {
          LOG(ERROR) << error;
        } else {
          logAndThrow(
              error, c10::str(error, "\n Original exception: \n", e.what()));
        }
      }
      if (rankResponded) {
        processedRanks.push_back(work.first);
      }
    }
    // If we are collecting all failed ranks, check if we need to throw if
    // some ranks have not responded.
    // Ensure all ranks from 1, ... WORLD_SIZE -1 have been successfully
    // processed.
    auto rankFailure =
        (processedRanks.size() != static_cast<size_t>(size_ - 1));
    if (waitAllRanks && rankFailure) {
      std::vector<int> failedRanks;
      for (const auto i : c10::irange(1, size_)) {
        if (std::find(processedRanks.begin(), processedRanks.end(), i) ==
            processedRanks.end()) {
          failedRanks.push_back(i);
        }
      }

      TORCH_INTERNAL_ASSERT(!failedRanks.empty());
      const std::string ranksStr = c10::Join(", ", failedRanks);
      const std::string error = c10::str(
          "[Rank 0]: Ranks ",
          ranksStr,
          " failed to pass monitoredBarrier in ",
          monitoredBarrierTimeout.count(),
          " ms");
      logAndThrow(error, error);
    }
  };

  waitLoop(recvWorkMap);
  // If we've reached here successfully, this means all ranks have acked in
  // monitoredBarrier. Unblock all ranks now by responding to their recv(). This
  // ensures that this is a true barrier in that all ranks  exit it successfully
  // or none of them do.
  for (const auto dstRank : c10::irange(1, worldSize)) {
    sendWorkMap.emplace(
        dstRank, send(commTensor, dstRank, static_cast<int>(t2)));
  }

  waitLoop(sendWorkMap);
}

void ProcessGroupGloo::setSequenceNumberForGroup() {
} // Gloo just starts sequence numbers at 0.

uint64_t ProcessGroupGloo::getSequenceNumberForGroup() {
  return seq_;
}

void ProcessGroupGloo::enableCollectivesTiming() {
  // Nothing to do to enable timing
}

} // namespace c10d

#endif // USE_C10D_GLOO
