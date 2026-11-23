#pragma once

// Types and helpers shared by the NCCL backends (nccl2 and, where built, the
// legacy ProcessGroupNCCL).

#ifdef USE_C10D_NCCL

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <ATen/core/Tensor.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>

namespace c10d {

// Control broadcasting of NCCL uniqueId
static std::vector<std::string> TORCH_NCCL_BCAST_UNIQUEID = {
    "TORCH_NCCL_BCAST_UNIQUEID"};

// Control EagerInit P2P serialization warning
static std::vector<std::string>
    TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING = {
        "TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING"};

// Control whether to always use high priority streams
static std::vector<std::string> TORCH_NCCL_HIGH_PRIORITY = {
    "TORCH_NCCL_HIGH_PRIORITY"};

// Control whether or not wait() is blocking or non-blocking.
static std::vector<std::string> TORCH_NCCL_BLOCKING_WAIT = {
    "TORCH_NCCL_BLOCKING_WAIT",
    "NCCL_BLOCKING_WAIT"};

// TODO: We want to eventually remove this variable and make users to use
// the default value (3 - SkipCleanUp).
// Control whether or not we perform Async Error Handling with NCCL.
static std::vector<std::string> TORCH_NCCL_ASYNC_ERROR_HANDLING = {
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "NCCL_ASYNC_ERROR_HANDLING"};

// Control whether dumping debug info on watchdog
// timeout is enabled. This variable must be set together with
// TORCH_NCCL_ENABLE_MONITORING=1 and TORCH_NCCL_TRACE_BUFFER_SIZE > 0.
static std::vector<std::string> TORCH_NCCL_DUMP_ON_TIMEOUT = {
    "TORCH_NCCL_DUMP_ON_TIMEOUT"};

// Control whether to propagate NCCL errors to all ranks through TCPStore.
static std::vector<std::string> TORCH_NCCL_PROPAGATE_ERROR = {
    "TORCH_NCCL_PROPAGATE_ERROR"};

// Control whether Desync Debug is enabled. This variable must be set
// together with TORCH_NCCL_ASYNC_ERROR_HANDLING.
static std::vector<std::string> TORCH_NCCL_DESYNC_DEBUG = {
    "TORCH_NCCL_DESYNC_DEBUG",
    "NCCL_DESYNC_DEBUG"};

// Enable recording start-events for all ProcessGroupNCCL collectives, and
// compute accurate collective timing per-collective. (Note: end-events are
// recorded by default. Turn on this flag can increase chances of a watchdog
// hang due to performing a CUDA event query which eventually calls
// cudaEventElapsedTime() API.
static std::vector<std::string> TORCH_NCCL_ENABLE_TIMING = {
    "TORCH_NCCL_ENABLE_TIMING",
    "NCCL_ENABLE_TIMING"};

// Enable monitoring thread which aborts the process when the ProcessGroupNCCL
// Watchdog thread gets stuck and no heartbeat is detected after
// TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC. This can happen due to calling CUDA/NCCL
// APIs that may hang. It is Useful to prevent jobs being stuck for a prolonged
// time than necessary tying up cluster resources.
static std::vector<std::string> TORCH_NCCL_ENABLE_MONITORING = {
    "TORCH_NCCL_ENABLE_MONITORING"};

// Control the watchdog heartbeat timeout period after which the monitoring
// thread will abort the process.
static std::vector<std::string> TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC = {
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"};

// Whether to rethrow CUDA Errors in the watchdog (default true)
static std::vector<std::string> TORCH_NCCL_RETHROW_CUDA_ERRORS = {
    "TORCH_NCCL_RETHROW_CUDA_ERRORS"};

// Whether a NCCL timeout tears the process down even when
// TORCH_NCCL_RETHROW_CUDA_ERRORS is off (default false)
static std::vector<std::string> TORCH_NCCL_TEARDOWN_ON_TIMEOUT = {
    "TORCH_NCCL_TEARDOWN_ON_TIMEOUT"};

// The maximum number of events we store in the flight recorder's ring buffer.
// (One event could be the start or end of a collective, for example).
static std::vector<std::string> TORCH_NCCL_TRACE_BUFFER_SIZE = {
    "TORCH_NCCL_TRACE_BUFFER_SIZE"};

// Control how much extra time we will wait for dumping the debugging info
// before we exit and throws timeout exception.
static std::vector<std::string> TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC = {
    "TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC"};

// Control the interval inside the monitoring thread to check the coordinated
// signal from other ranks, e.g. to dump the debugging information.
static std::vector<std::string> TORCH_NCCL_COORD_CHECK_MILSEC = {
    "TORCH_NCCL_COORD_CHECK_MILSEC"};

// Whether to log C++ stack traces on unclean shutdown (default true)
static std::vector<std::string> TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN = {
    "TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN"};

// Whether to include only active collectives in the Flight Recorder trace
// (default false)
static std::vector<std::string> TORCH_NCCL_EXTRA_DUMP_ON_EXEC = {
    "TORCH_NCCL_EXTRA_DUMP_ON_EXEC"};

// Control whether to use CudaEventCache for the collective in watchdog thread.
// We noticed in the past when cuda global lock is held, destroying CudaEvent
// can cause a hang.
static std::vector<std::string> TORCH_NCCL_CUDA_EVENT_CACHE = {
    "TORCH_NCCL_CUDA_EVENT_CACHE"};

// Control the number of ranks each root can cover during NCCL comm init.
static std::vector<std::string> TORCH_NCCL_RANKS_PER_ROOT = {
    "TORCH_NCCL_RANKS_PER_ROOT"};

static std::vector<std::string> TORCH_NCCL_NAN_CHECK = {"TORCH_NCCL_NAN_CHECK"};

constexpr const char* NCCL_BACKEND_NAME = "nccl";

constexpr const char* kStoreDumpKey = "exception_dump";

constexpr const char* kStoreErrorSignalKey = "remote_error";

constexpr const int kWorkStatusUpdatePeriodMs = 30 * 1000; // 30 seconds

constexpr auto kProcessGroupNCCLDefaultTimeout =
    std::chrono::milliseconds(10 * 60 * 1000);

// NoHandling: do not handle asynchronous NCCL errors
// TearDown: tear down process upon error, see `WorkNCCL::handleException`
// CleanUpOnly: just clean up collectives and abort communicators without
// tearing down process SkipCleanUp: (this is a temporary option and can be
// removed in future) tear down process without cleaning up NCCL communicators.
// This should be used as a last resort in case `ncclCommAbort` itself is
// hanging
enum ErrorHandlingMode {
  NoHandling = 0,
  TearDown = 1,
  CleanUpOnly = 2,
  SkipCleanUp = 3
};

#define SHOULD_CLEAN_UP(a) (a != NoHandling && a != SkipCleanUp)

#define SHOULD_TEAR_DOWN(a) (a != NoHandling && a != CleanUpOnly)

// A shelf for stashing tensors between op call and `work.wait()`.
// Used in case of async ops.
class TensorShelf {
 public:
  // Stash tensors so that CachingAllocator cannot recycle them prematurely.
  void stash(std::vector<at::Tensor>& tensors);
  // Stash tensors from another shelf.
  void stash(TensorShelf& other);
  // Unstage the stashed tensors so that CachingAllocator can recycle them.
  // Same as `clear()`.
  void unstash();
  // Whether shelf is empty.
  bool empty();
  // Clear the shelf.
  void clear();

 protected:
  // Get the inner tensor vector. Use with caution as it is not protected by
  // mutex.
  std::vector<at::Tensor>& get();

 private:
  std::vector<at::Tensor> tVector_;
  // Need a mutex to protect `tVector_` because it can be potentially accessed
  // from both main thread and watchdog thread.
  std::mutex mutex_;
};

struct TORCH_API NCCLOptions : Backend::Options {
  // NOTE: timeout in NCCLOptions denotes the timeout for
  // operations. This is only used when blockingWait_ is enabled.
  explicit NCCLOptions(bool is_high_priority_stream = false);
  NCCLOptions(const NCCLOptions&) = default;
  NCCLOptions(NCCLOptions&&) noexcept = default;
  NCCLOptions& operator=(const NCCLOptions&) = delete;
  NCCLOptions& operator=(NCCLOptions&&) noexcept = delete;
  ~NCCLOptions() override = default;

  // return intrusive_ptr of the object
  static c10::intrusive_ptr<NCCLOptions> create(
      bool is_high_priority_stream = false) {
    return c10::make_intrusive<NCCLOptions>(is_high_priority_stream);
  }

  c10::intrusive_ptr<Backend::Options> clone() const override {
    auto copy = c10::make_intrusive<NCCLOptions>(*this);
    if (config.netName != nullptr) {
      // ncclConfig_t::netName is a strdup'ed const char* (see the NCCLConfig
      // pybind setter) and ncclConfig_t records no owner, so a plain copy
      // would leave two Options sharing one allocation. Give the clone its
      // own and tie the allocation to it: copies of this Options share it and
      // free it once, when the last of them goes away. A netName from
      // anywhere else -- the pybind setter, a config also handed to NCCL --
      // is untracked and untouched.
      copy->owned_net_name_ = std::shared_ptr<const char>(
          strdup(config.netName),
          [](const char* p) { std::free(const_cast<char*>(p)); });
      copy->config.netName = copy->owned_net_name_.get();
    }
    return copy;
  }

  // Schedule NCCL operations on high priority CUDA streams
  bool is_high_priority_stream;

  // Configure ranks
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

  // Optional "parent" backend and color to create communicators from
  // via `ncclCommSplit`
  c10::intrusive_ptr<Backend> split_from;
  // Color to use for `ncclCommSplit`, values:
  // * Non-negative value: in group;
  // * NCCL_SPLIT_NOCOLOR (-1): not in group;
  // * NCCL_SPLIT_NOCOLOR - 1: uninitialized.
  // [Note 1]: the type must be `int` instead of `int64_t` because NCCL API
  // accepts int. Otherwise, an implicit conversion may happen at the API call
  // and the value may become negative.
  // [Note 2]: this member is pybinded to Python, the value passed from Python
  // must be within the numerical range of C++ int. Otherwise, Python will
  // raise a RuntimeError saying type is incompatible. See also
  // `_process_group_color` in `distributed_c10d.py`.
  int split_color{NCCL_SPLIT_NOCOLOR - 1};

  // nccl2 only: create the communicator on the first operation instead of
  // when the device is bound. Such a group can't be split from until its
  // first operation.
  bool lazy_init{false};

  // nccl2 only: process-global pre-shared key for NCCL's TLS socket
  // encryption (NCCL 2.32.3+ built with TLS support). See the pybind
  // docstring for constraints.
  std::optional<std::string> encryption_psk;

 private:
  // Keeps clone()'s strdup'ed config.netName alive; see clone().
  std::shared_ptr<const char> owned_net_name_;
};

// Reset the flighrecorder recordings for the current rank.
TORCH_API void reset_nccl_trace();

// Dumps the NCCL comm traces and additional information about the Process
// Group.
TORCH_API std::string dump_nccl_trace(
    bool includeCollectives,
    bool includeStackTraces,
    bool onlyActive);

// Dumps the NCCL comm traces and additional information about the Process
// Group in JSON formatted string.
// We don't include stack traces in JSON format as it is far too much data.
TORCH_API std::string dump_nccl_trace_json(
    bool includeCollectives,
    bool onlyActive);

// Gets a mutable reference to a global optional function.Heartbeat Monitor
// will use this function to dump traces, if available. Inside fbcode, we
// store a function here that uses an internal tool for process tracing
TORCH_API std::optional<
    std::function<void(std::function<void(const std::string&)>)>>&
get_cpp_trace_dumper();

// Similar to get_cpp_trace_dumper, this stores a function defined in
// torch-python layer that lets us check whether the GIL can be acquired,
// helpful for instrumenting in cases where a hang was observed.
typedef bool (*gil_checker_t)();

TORCH_API gil_checker_t& get_gil_checker();

// Per-communicator dump (ncclCommDump) included in dump_nccl_trace output,
// keyed by communicator hash. Set by a backend that tracks its communicators;
// unset means no per-communicator data.
using nccl_comm_dump_map_t = std::
    unordered_map<std::string, std::unordered_map<std::string, std::string>>;
TORCH_API std::function<nccl_comm_dump_map_t()>& get_nccl_comm_dump_map_fn();
} // namespace c10d

#endif // USE_C10D_NCCL
