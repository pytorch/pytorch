#pragma once

#ifdef USE_C10D_NCCL

#include <sched.h>
#include <cstdio>
#include <cstdlib>

#include <memory>
#include <mutex>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/util/Exception.h>
#include <nccl.h>
#include <torch/csrc/cuda/nccl.h>
#include <torch/csrc/distributed/c10d/TraceUtils.h>
#include <optional>

constexpr int64_t kCommInitBusyWaitMillis = 2;

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 14, 0)
#define NCCL_HAS_COMM_NONBLOCKING
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 18, 0)
#define NCCL_HAS_COMM_SPLIT
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 23, 0)
#define NCCL_HAS_INIT_RANK_SCALABLE
#endif

// ncclGetLastError() is enabled only for NCCL versions 2.13+
// ncclRemoteError only exists in NCCL versions 2.13+
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 13, 0)
#define ENABLE_NCCL_GET_LAST_ERROR
#define NCCL_REMOTE_ERROR
#endif

static_assert(
    NCCL_VERSION_CODE >= NCCL_VERSION(2, 7, 0),
    "NCCL version must be 2.7 or later");
// The following macros represent features supported prior to NCCL 2.7,
// therefore we can define them unconditionally, given the static_assert above.
// TODO: remove these macros from code.
#define ENABLE_NCCL_ERROR_CHECKING
#define ENABLE_NCCL_P2P_SUPPORT
// End of macros for NCCL 2.7 and below.

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 11, 0)
#define ENABLE_NCCL_PREMUL_SUM_SUPPORT
#endif

// Note: the first version that supports ncclConfig_t is 2.14. Here we
// fast-forward the version requirement to 2.17 where ncclConfig_t has CTA and
// CGA fields because they have already been pybinded out.
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 17, 0)
#define NCCL_HAS_CONFIG
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 19, 0)
#define NCCL_HAS_COMM_REGISTER
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 0)
#define NCCL_HAS_COMM_WINDOW_REGISTER
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 19, 0)
#define NCCL_HAS_MEM_ALLOC
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 26, 0)
#define NCCL_HAS_QOS
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 24, 0)
#define NCCL_SUPPORTS_FP8
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 0)
#define NCCL_HAS_COLLNET
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 0)
#define NCCL_HAS_CTA_POLICY
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 0)
#define NCCL_HAS_NVLS_CTAS
#endif

// Macro to throw on a non-successful NCCL return value.
#define C10D_NCCL_CHECK(cmd, failureReason)                                   \
  do {                                                                        \
    ncclResult_t result = cmd;                                                \
    if (result != ncclSuccess) {                                              \
      std::string err = "NCCL error in: " + std::string(__FILE__) + ":" +     \
          std::to_string(__LINE__) + ", " + ncclGetErrorWithVersion(result) + \
          "\n" + getNcclErrorDetailStr(result, failureReason);                \
      TORCH_CHECK_WITH(DistBackendError, false, err);                         \
    }                                                                         \
  } while (0)

// Macro to throw on a non-successful NCCL return value for NONBLOCKING calls.
#define C10D_NCCL_CHECK_NONBLOCKING(cmd, failureReason)                       \
  do {                                                                        \
    ncclResult_t result = cmd;                                                \
    if (result != ncclSuccess && result != ncclInProgress) {                  \
      std::string err = "NCCL error in: " + std::string(__FILE__) + ":" +     \
          std::to_string(__LINE__) + ", " + ncclGetErrorWithVersion(result) + \
          "\n" + getNcclErrorDetailStr(result, failureReason);                \
      TORCH_CHECK_WITH(DistBackendError, false, err);                         \
    }                                                                         \
  } while (0)

// Error out if (current time - startTime) is greater than timeout (sec).
#define C10D_CHECK_TIMEOUT(startTime, timeout)                              \
  do {                                                                      \
    auto currentTime = std::chrono::steady_clock::now();                    \
    auto timeElapsed = std::chrono::duration_cast<std::chrono::seconds>(    \
                           currentTime - startTime)                         \
                           .count();                                        \
    if (timeElapsed > timeout) {                                            \
      std::string err = "NCCL timeout in: " + std::string(__FILE__) + ":" + \
          std::to_string(__LINE__);                                         \
      TORCH_CHECK_WITH(DistBackendError, false, err);                       \
    }                                                                       \
  } while (0)

// Macro to throw on a non-successful NCCL return value, non-blocking.
#define C10D_NCCL_CHECK_TIMEOUT_BASE(cmd, comm, failureReason, yield_fn)      \
  do {                                                                        \
    ncclResult_t result = cmd;                                                \
    auto startTimepoint = std::chrono::steady_clock::now();                   \
    auto timeout = nccl_nonblocking_timeout();                                \
    while (result == ncclInProgress) {                                        \
      C10D_CHECK_TIMEOUT(startTimepoint, timeout);                            \
      yield_fn;                                                               \
      ncclCommGetAsyncError(comm, &result);                                   \
    }                                                                         \
    if (result != ncclSuccess) {                                              \
      std::string err = "NCCL error in: " + std::string(__FILE__) + ":" +     \
          std::to_string(__LINE__) + ", " + ncclGetErrorWithVersion(result) + \
          "\n" + getNcclErrorDetailStr(result, failureReason);                \
      TORCH_CHECK_WITH(DistBackendError, false, err);                         \
    }                                                                         \
  } while (0)

// Sleep for kCommInitBusyWaitMillis milliseconds.
#define C10D_SCHED_SLEEP()     \
  std::this_thread::sleep_for( \
      std::chrono::milliseconds(kCommInitBusyWaitMillis))

// Macro to throw exception on a non-successful NCCL return value or timeout.
// This macro uses sched_yield() to yield the CPU.
// Thus suitable for NCCL calls that would quickly turn ncclSuccess, e.g.
// collectives.
#define C10D_NCCL_CHECK_TIMEOUT(cmd, comm, failureReason) \
  C10D_NCCL_CHECK_TIMEOUT_BASE(cmd, comm, failureReason, sched_yield())

// Macro to throw exception on a non-successful NCCL return value or timeout.
// This macro uses sleep to yield the CPU.
// Thus suitable for NCCL calls that would take longer to turn ncclSuccess, e.g.
// ncclCommInitRankConfig, ncclCommFinalize, etc.
#define C10D_NCCL_CHECK_TIMEOUT_SLEEP(cmd, comm, failureReason) \
  C10D_NCCL_CHECK_TIMEOUT_BASE(cmd, comm, failureReason, C10D_SCHED_SLEEP())

#define C10D_NCCL_CHECK_TIMEOUT_GROUPEND(cmd, comm, failureReason)           \
  do {                                                                       \
    ncclResult_t state = cmd;                                                \
    auto startTimepoint = std::chrono::steady_clock::now();                  \
    auto timeout = nccl_nonblocking_timeout();                               \
    if (state == ncclInProgress) {                                           \
      do {                                                                   \
        C10D_CHECK_TIMEOUT(startTimepoint, timeout);                         \
        sched_yield();                                                       \
        ncclCommGetAsyncError(comm->getNcclComm(), &state);                  \
      } while (state == ncclInProgress);                                     \
    }                                                                        \
    if (state != ncclSuccess) {                                              \
      std::string err = "NCCL error in: " + std::string(__FILE__) + ":" +    \
          std::to_string(__LINE__) + ", " + ncclGetErrorWithVersion(state) + \
          "\n" + getNcclErrorDetailStr(state, failureReason);                \
      TORCH_CHECK_WITH(DistBackendError, false, err);                        \
    }                                                                        \
  } while (0)

// Macro to print and abort on a non-successful NCCL return value.
#define C10D_NCCL_ASSERT(cmd)                            \
  do {                                                   \
    ncclResult_t result = cmd;                           \
    if (result != ncclSuccess) {                         \
      std::string err = ncclGetErrorWithVersion(result); \
      fprintf(                                           \
          stderr,                                        \
          "NCCL error in: %s:%d, %s\n",                  \
          __FILE__,                                      \
          __LINE__,                                      \
          err.c_str());                                  \
      abort();                                           \
    }                                                    \
  } while (0)

namespace c10d {

// NCCL type typing
static std::map<at::ScalarType, ncclDataType_t> ncclDataType = {
    {at::kChar, ncclInt8},
    {at::kByte, ncclUint8},
    {at::kFloat, ncclFloat},
    {at::kDouble, ncclDouble},
    {at::kInt, ncclInt32},
    {at::kLong, ncclInt64},
    {at::kHalf, ncclHalf},
    {at::kBool, ncclUint8},
#ifdef NCCL_SUPPORTS_FP8
    {at::kFloat8_e5m2, ncclFloat8e5m2},
    {at::kFloat8_e4m3fn, ncclFloat8e4m3},
#else
    {at::kFloat8_e5m2, ncclUint8},
    {at::kFloat8_e4m3fn, ncclUint8},
#endif
    // NVIDIA GPUs does not support the UZ version standing for "no negative
    // zero".  See https://onnx.ai/onnx/technical/float8.html
    {at::kFloat8_e4m3fnuz, ncclUint8},
    {at::kFloat8_e5m2fnuz, ncclUint8},
#if HAS_NCCL_BF16_DATATYPE
    {at::kBFloat16, ncclBfloat16},
#endif // HAS_NCCL_BF16_DATATYPE
};

TORCH_API size_t hashTensors(const std::vector<at::Tensor>& tensors);
TORCH_API std::string getNcclVersion();
TORCH_API std::tuple<int, int, int> getNcclVersionTuple();
TORCH_API int getNcclVersionNumber();
TORCH_API std::string ncclGetErrorWithVersion(ncclResult_t error);
int nccl_nonblocking_timeout();

// Provides additional detail into NCCL error codes based on when these are
// thrown in the NCCL codebase.
TORCH_API std::string getNcclErrorDetailStr(
    ncclResult_t error,
    std::optional<std::string> processGroupFailureReason = std::nullopt);

// Helper function that gets the data type and issues error if not supported
ncclDataType_t getNcclDataType(at::ScalarType type);

// RAII wrapper for NCCL communicator
class NCCLComm {
  using MutexType = std::recursive_mutex;
  using LockType = std::unique_lock<MutexType>;

 public:
  explicit NCCLComm(ncclComm_t ncclComm);

  NCCLComm() = default;

  ~NCCLComm() noexcept;

  static std::shared_ptr<NCCLComm> create(
      int numRanks,
      int rank,
      ncclUniqueId commId,
      at::DeviceIndex deviceIndex);

#ifdef NCCL_HAS_CONFIG
  static std::shared_ptr<NCCLComm> create(
      int numRanks,
      int rank,
      ncclUniqueId commId,
      at::DeviceIndex deviceIndex,
      ncclConfig_t& config);
#ifdef NCCL_HAS_INIT_RANK_SCALABLE
  static std::shared_ptr<NCCLComm> create_scalable(
      int numRanks,
      int rank,
      std::vector<ncclUniqueId>& commIds,
      at::DeviceIndex deviceIndex,
      ncclConfig_t& config);
#endif // NCCL_HAS_INIT_RANK_SCALABLE
#endif // NCCL_HAS_CONFIG

#ifdef NCCL_HAS_COMM_SPLIT
  static std::shared_ptr<NCCLComm> split(
      NCCLComm* source,
      int color_id,
      int rank,
      ncclConfig_t& config,
      std::vector<uint64_t>& ranks_ull);
#endif // NCCL_HAS_COMM_SPLIT

#if (defined(IS_NCCLX) || defined(USE_ROCM)) && defined(NCCL_COMM_DUMP)
  std::unordered_map<std::string, std::string> ncclCommDump();
#endif

  ncclUniqueId getNcclId();
  at::DeviceIndex getDeviceIndex();

  // Must not be copyable
  NCCLComm(const NCCLComm&) = delete;
  NCCLComm& operator=(const NCCLComm&) = delete;

  // Do not support move assignment as there is no valid use case
  NCCLComm& operator=(NCCLComm&& other) = delete;

  // Move constructable
  // NOLINTNEXTLINE(*-noexcept-move-*)
  NCCLComm(NCCLComm&& other);

  ncclComm_t getNcclComm();

  // Wait for the communicator to be ready. This is a blocking function.
  // Useful in nonblocking mode: NCCL requires the communicator to be ready
  // before issuing a second command.
  // Arguments:
  //   longInterval: if true, wait with sleep of an interval; otherwise, wait
  //   with `sched_yield` which is faster (but acquires CPU more frequently).
  //   Use `longInterval=true` when waiting for initialization or finalize to
  //   complete. Use `longInterval=false` when waiting collective call to return
  //   ncclSuccess.
  void waitReady(bool longInterval);

  std::optional<std::string> getNcclCommFailureReason() const;

  void abort(std::optional<std::string> commFailureReason = std::nullopt);

  // Finalize a communicator -- asking it to flush its operations. When the
  // communicator is marked as nonblocking, this is a nonblocking function;
  // otherwise, it will block till all operations complete.
  void finalize();

  // Destroy a communicator. This is a blocking function.
  void destroy();

  bool isInitialized() const;

  bool isAborted() const;

  uint64_t getCommSplitCounter() const;

  ncclResult_t checkForNcclError();

  ncclResult_t registerSegment(
      void* ptr,
      size_t size,
      bool errorOnRereg = true,
      bool window = false);

  ncclResult_t deregisterSegment(void* ptr, bool window = false);

  std::string repr() const;

  friend class ProcessGroupNCCL;

 protected:
  // Unique nccl_id for this communicator.
  ncclUniqueId ncclId_{};
  bool aborted_{false};
  uint64_t ncclCommSplitCounter_{0};
  ncclResult_t ncclAsyncErr_{ncclSuccess};
  mutable MutexType mutex_;
  // Rank that this communicator corresponds to.
  int rank_{};
  // Optional reason for communicator failure, provided by ProcessGroupNCCL for
  // better error messaging.
  std::optional<std::string> commFailureReason_{};
  bool initialized_{false};
  // Whether this communicator is using nonblocking mode. Recorded during comm
  // creation or split. For safety, we give a default value of true (more
  // protection).
  bool nonBlocking_{true};
  // Device index for which the NCCL comm is created
  at::DeviceIndex deviceIndex_{-1};
#ifdef NCCL_HAS_COMM_REGISTER
  // Stores handlers for tensors registered by NCCL
  std::unordered_map<void*, void*> registeredSegmentHandles_;
#endif // NCCL_HAS_COMM_REGISTER

 private:
  ncclComm_t ncclComm_{nullptr};
};

// Helper that automatically cleans up premul sums.
struct ncclRedOpRAII {
  ncclRedOpRAII() = default;
  ncclRedOpRAII(ncclRedOp_t op) : op_(op) {}
  ncclRedOpRAII(ncclRedOp_t op, ncclComm_t comm)
      : op_(op), comm_(comm), premul_sum_(true) {}
  ncclRedOpRAII(const ncclRedOpRAII&) = delete;
  ncclRedOpRAII& operator=(const ncclRedOpRAII&) = delete;
  ncclRedOpRAII(ncclRedOpRAII&& tmp) noexcept : ncclRedOpRAII() {
    std::swap(tmp.op_, this->op_);
    std::swap(tmp.comm_, this->comm_);
    std::swap(tmp.premul_sum_, this->premul_sum_);
  }
#if defined(ENABLE_NCCL_PREMUL_SUM_SUPPORT)
  ~ncclRedOpRAII() {
    if (premul_sum_) {
      ncclRedOpDestroy(op_, comm_);
    }
  }
#endif // ENABLE_NCCL_PREMUL_SUM_SUPPORT
  operator ncclRedOp_t() const {
    return op_;
  }
  ncclRedOp_t op_{};
  ncclComm_t comm_{};
  bool premul_sum_ = false;
};

void printNcclCommProxyTrace(
    const std::string& dumpReason,
    const std::unordered_map<std::string, std::string>& dumpMap);
} // namespace c10d

#endif // USE_C10D_NCCL
