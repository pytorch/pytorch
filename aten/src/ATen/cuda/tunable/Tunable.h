// Original TunableOp is from onnxruntime.
// https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/framework/tunable.h
// https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/cuda/tunable
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// Adapting TunableOp into PyTorch
// Copyright (c) Advanced Micro Devices, Inc.
//
#pragma once

#include <c10/util/CallOnce.h>
#include <c10/util/StringUtil.h>
#include <c10/util/env.h>

#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#define TUNABLE_LOGV(LEVEL, ...) getTuningContext()->Log(LEVEL, __VA_ARGS__)
#define TUNABLE_LOG1(...) TUNABLE_LOGV(1, __VA_ARGS__)
#define TUNABLE_LOG2(...) TUNABLE_LOGV(2, __VA_ARGS__)
#define TUNABLE_LOG3(...) TUNABLE_LOGV(3, __VA_ARGS__)

namespace at::cuda::tunable {

enum TORCH_CUDA_CPP_API TuningStatus {
  OK = 0,
  FAIL = 1,
  UNSUPPORTED = 2,
};

// Mapping from params signature to kernel id
class TORCH_CUDA_CPP_API ResultEntry {
  public:
    explicit ResultEntry(std::string  key, double time) : key_(std::move(key)), time_(time) {}
    explicit ResultEntry(std::string  key, double time, std::string blas_sig ) : key_(std::move(key)), time_(time), blas_sig_(std::move(blas_sig)) {}
    bool operator==(const ResultEntry& other) const { return key_ == other.key_; }
    bool operator!=(const ResultEntry& other) const { return key_ != other.key_; }
    operator std::string () { return key_; }
    std::string GetKey() const { return key_; }
    double GetTime() const { return time_; }
    friend std::ostream& operator<<(std::ostream& stream, const ResultEntry& entry);
    static ResultEntry Null() { return ResultEntry("Null", 0.0); }
    static ResultEntry Default() { return ResultEntry("Default", 0.0); }

  private:
    std::string key_;
    double time_;
    std::string blas_sig_;
};

typedef std::unordered_map<std::string, ResultEntry> KernelMap;
typedef std::unordered_map<std::string, KernelMap> ResultsMap;
typedef std::unordered_map<std::string, std::unordered_set<std::string>> UntunedMap;

struct TORCH_CUDA_CPP_API TuningResults {
  // Validates if these results are compatible with the libraries
  std::unordered_map<std::string, std::string> validators;

  // Mapping from Callable signature to Callable's tuning result
  ResultsMap results;
};

class TORCH_CUDA_CPP_API TuningResultsManager {
  public:
    TuningResultsManager() = default;
    ~TuningResultsManager() = default;

    KernelMap Lookup(const std::string& op_signature);

    ResultEntry Lookup(const std::string& op_signature, const std::string& params_signature);

    // Scans persisted wildcard entries (keys containing '*') under
    // `op_signature` and returns the first whose token-by-token pattern
    // matches `concrete_params_signature`. Used as a runtime fallback
    // when a concrete miss happens without an active TunableDynamicDimsGuard
    // (e.g., AOTI cpp_wrapper that wasn't compiled with guard emission).
    // Returns ResultEntry::Null() if no wildcard matches.
    //
    // Overlapping patterns: KernelMap is an unordered_map, so when several
    // wildcards match, the winner is iteration-order dependent rather than the
    // most specific. Only the M dim is wildcarded today, so exactly one
    // matches; precedence is follow-up work.
    //
    // A hit is a hint, not a guarantee: '*' does not preserve dim/stride
    // relationships and nothing is re-validated here. If the algorithm cannot
    // service the concrete shape, hipBLASLt/rocBLAS rejects it at Call() time
    // and the caller re-dispatches through gemm_internal / gemm_and_bias --
    // the next matching wildcard is not tried. Falls back to default ATen,
    // never a wrong result.
    ResultEntry LookupWildcardFallback(
        const std::string& op_signature,
        const std::string& concrete_params_signature);

    // Borrow the tuned solution of a neighbouring shape: identical except in one
    // free dim (M, N or batch count), rounded up. Separate from Lookup so a caller
    // can tell a measured hit from a borrow, which is expected to be validated.
    // Unconditional: TunableOp dispatch never calls this, so calling it is the
    // opt-in and there is no flag to consult.
    //
    // Caller contract: benchmark the returned solution and reject it unless
    // Call() reports OK. The match is made on the params signature, which does
    // not carry the batched strides, so a donor can be one the backend refuses
    // for this shape -- it returns FAIL and the caller silently runs its own
    // non-tunable path. A caller that skipped the check would time that
    // fallback under the borrowed key and record a solution that fails on
    // every later call.
    ResultEntry LookupNearest(const std::string& op_signature,
        const std::string& params_signature);

    void AddImpl(const std::string& op_signature,
        const std::string& params_signature,
        ResultEntry best,
        KernelMap& kernel_map);

    void Add(const std::string& op_signature,
        const std::string& params_signature,
        ResultEntry best);

    // Persist a compile-time decision for a shape that was never tuned: a borrowed
    // solution, or Default when the default kernel benchmarked faster. Unlike Add()
    // this writes while tuning is disabled.
    //
    // Accepted limitation: the entry is written in the ordinary format with
    // nothing marking it as borrowed, so an entry no longer implies the tuner
    // measured that exact shape. Writes are first-write-wins, so a later tuning
    // run skips these shapes and a Default recorded here is permanent.
    void RecordBorrowed(const std::string& op_signature,
        const std::string& params_signature,
        ResultEntry entry);

    void Delete(const std::string& op_signature, const std::string& params_signature);

    void DisjointMergeImpl(
        const std::string& op_signature,
        const KernelMap& kernel_map,
        /*out*/ ResultsMap& results);

    void Load(const ResultsMap& results_to_load);

    ResultsMap Dump();

    void DisjointMerge(const std::string& op_signature, const KernelMap& kernel_map);

    size_t GetSize();

    void RecordUntuned( std::ofstream& untuned_file, const std::string& op_signature,
      const std::string& params_signature, const std::string& blas_signature);
    void ClearUntuned();

    void ClearAll();

    void InitRealtimeAppend(
        const std::string& filename,
        const std::unordered_map<std::string, std::string>& validators);

    void AppendResultLine(const std::string& op_sig,
                         const std::string& param_sig,
                         const ResultEntry& result);

    void CloseRealtimeAppend();  // For clean shutdown
  private:
    // Shared body of Add() and RecordBorrowed(). force_persist writes the entry
    // to the results file regardless of the tuning flags, so RecordBorrowed can
    // override the write rule for itself without widening it for Add()'s callers.
    void AddAndMaybePersist(const std::string& op_signature,
        const std::string& params_signature,
        ResultEntry best,
        bool force_persist);

    std::mutex lock_;
    std::mutex realtime_file_mutex_;
    std::unique_ptr<std::ofstream> realtime_out_;
    std::string realtime_filename_;
    ResultsMap results_;
    UntunedMap untuned_results_;
    bool validators_written_ = false;

};

class TORCH_CUDA_CPP_API TuningResultsValidator {
  public:
    using GetFunc = std::function<std::string()>;
    using ValidateFunc = std::function<TuningStatus(const std::string&)>;
    using GetValidateFuncs = std::unordered_map<std::string, std::pair<GetFunc, ValidateFunc>>;

    TuningResultsValidator();
    ~TuningResultsValidator() = default;

    std::unordered_map<std::string, std::string> GetAllValidators() const;
    TuningStatus ValidateAll(const std::unordered_map<std::string, std::string>& to_validate) const;
    void RegisterValidator(const std::string& key, const GetFunc& gf, const ValidateFunc& vf);

  protected:
    static std::string GetPyTorchVersion() ;
    TuningStatus ValidatePyTorchVersion(const std::string& value) const;

  public:
    static constexpr const std::array mandatory_keys{"PT_VERSION"};

  private:
    GetValidateFuncs validators_;
};

struct NumericalCheckConfig {
  bool   enabled{false};
  double atol{1e-5};
  double rtol{1e-5};

  NumericalCheckConfig() = default;
  NumericalCheckConfig(bool e, double a, double r) : enabled(e), atol(a), rtol(r) {}
};

// Per-call dynamic-dims mask. POD, trivially copyable so it composes with
// the existing OpParams DeepCopy(*this) paths without any custom copy logic.
// The four bits select which logical GEMM dim (M/N/K/BATCH) is wildcarded
// when computing DynamicSignature() for that specific op invocation.
struct TORCH_CUDA_CPP_API DynamicDimsMask {
  static constexpr uint8_t M_BIT = 1u << 0;
  static constexpr uint8_t N_BIT = 1u << 1;
  static constexpr uint8_t K_BIT = 1u << 2;
  static constexpr uint8_t BATCH_BIT = 1u << 3;

  uint8_t bits{0};

  constexpr DynamicDimsMask() = default;
  constexpr DynamicDimsMask(bool m, bool n, bool k, bool batch)
      : bits(
            static_cast<uint8_t>(
                (m ? M_BIT : 0) | (n ? N_BIT : 0) | (k ? K_BIT : 0) |
                (batch ? BATCH_BIT : 0))) {}
  explicit constexpr DynamicDimsMask(uint8_t b) : bits(b) {}

  constexpr bool m() const { return (bits & M_BIT) != 0; }
  constexpr bool n() const { return (bits & N_BIT) != 0; }
  constexpr bool k() const { return (bits & K_BIT) != 0; }
  constexpr bool batch() const { return (bits & BATCH_BIT) != 0; }
  constexpr bool any() const { return bits != 0; }
};

class TORCH_CUDA_CPP_API TuningContext {
  public:
    TuningContext();
    ~TuningContext();
    TuningContext(TuningContext &) = delete;
    TuningContext(TuningContext &&) = delete;
    TuningContext &operator=(TuningContext &) = delete;
    TuningContext &operator=(TuningContext &&) = delete;

    void EnableTunableOp(bool value);
    bool IsTunableOpEnabled() const;

    void EnableTuning(bool value);
    bool IsTuningEnabled() const;

    void EnableRecordUntuned(bool value);
    bool IsRecordUntunedEnabled() const;
    std::ofstream& GetUntunedFile();

    void EnableWildcardFallback(bool value);
    bool IsWildcardFallbackEnabled() const;

    // Policy for the compile-time reuse pass: set from Python, read back by
    // the pass through these C++ accessors (the Python query helpers exist
    // only for tests). Not consulted by TuningResultsManager or TunableOp
    // dispatch. Runtime-settable because the pass and the execution that
    // follows it share a process and need opposite values.
    void EnableNearestMatch(bool value);
    bool IsNearestMatchEnabled() const;

    void EnableRecordBorrowed(bool value);
    bool IsRecordBorrowedEnabled() const;

    void EnableNumericsCheck(bool value);
    bool IsNumericsCheckEnabled() const;
    void SetNumericalCheckConfig(bool enabled, double atol, double rtol);
    NumericalCheckConfig GetNumericalCheckConfig() const;

    void SetMaxTuningDurationMs(int max_duration_ms);
    int GetMaxTuningDurationMs() const;

    void SetMaxTuningIterations(int max_iter);
    int GetMaxTuningIterations() const;

    void SetCublasLtRequestedAlgoCount(int count);
    int GetCublasLtRequestedAlgoCount() const;

    void SetMaxWarmupDurationMs(int max_duration_ms);
    int GetMaxWarmupDurationMs() const;

    void SetMaxWarmupIterations(int max_iter);
    int GetMaxWarmupIterations() const;

    void EnableICacheFlush(bool value);
    bool IsICacheFlushEnabled() const;

    void SetRotatingBufferSize(int size);
    int GetRotatingBufferSize() const;

    TuningResultsManager& GetTuningResultsManager();

    TuningResultsValidator& GetTuningResultsValidator();

    TuningResults GetTuningResults();

    TuningStatus LoadTuningResults(const TuningResults& tr);

    void SetFilename(const std::string& filename, bool insert_device_ordinal=false);
    std::string GetFilename() const;

    bool ReadFile(const std::string& filename={});

    template<class... Types>
    void Log(int level, Types... args) {
      if (GetLogOkay() && GetLogLevel() >= level) {
        GetLog() << c10::str(args...) << std::endl;
      }
    }

  private:
    std::string GetLogFilename() const;
    int GetLogLevel() const;
    bool GetLogOkay() const;
    std::ostream& GetLog() const;

    bool enable_;
    bool tuning_enable_;
    bool record_untuned_enable_;
    // Set once before the work they govern, not flipped underneath it. Per-call
    // state goes through the thread-local stack; see TunableDynamicDimsGuard.
    bool nearest_match_enable_;
    bool record_borrowed_enable_;
    bool manager_initialized_;
    bool numerics_check_enable_;
    int max_tuning_duration_ms_;
    int max_tuning_iterations_;
    int cublaslt_requested_algo_count_;
    int max_warmup_duration_ms_;
    int max_warmup_iterations_;
    bool icache_flush_;
    int rotating_buffer_size_;
    mutable TuningResultsManager manager_;
    mutable c10::once_flag manager_init_once_;
    TuningResultsValidator validator_;
    std::string filename_;
    std::ofstream untuned_file_;
    size_t results_count_from_input_file_;
    bool is_shutting_down_;
    bool wildcard_fallback_enabled_;

    NumericalCheckConfig numerics_cfg_;
};

TORCH_CUDA_CPP_API TuningContext* getTuningContext();

// Returns the current top-of-stack DynamicDimsMask for this thread, or an
// all-zero mask when no TunableDynamicDimsGuard is active. Producers in
// Blas.cpp / CUDABlas.cpp / ScaledBlas.cpp call this once per Gemm*Params
// construction and stamp the result onto the params before invoking the
// TunableOp, so DynamicSignature() can read the mask off *this rather than
// from a global TuningContext setting.
//
// Frame note (canonical): the mask is pushed in inductor frame
// (M = mat1.size(0), N = mat2.size(1), K = mat1.size(1)). When BLAS dispatch
// swaps (M, N) -> (n, m) to keep cuBLAS column-major (cublasCommonArgs::
// swapped_mn, or a transposed batched result), producers must remap the bits
// (swap m()<->n()) before stamping; otherwise DynamicSignature() places '*'
// in the wrong slot and LookupWildcardFallback never matches at runtime.
TORCH_CUDA_CPP_API DynamicDimsMask GetCurrentDynamicDimsMask();

// RAII guard that pushes a DynamicDimsMask onto the thread-local stack on
// construction and pops on destruction. Use to wrap a single GEMM call (or a
// scope containing GEMM calls) with the mask that applies to that op.
//
// Lives in at::cuda::tunable so it is reachable from ATen callers and the
// libtorch-owned AOTI C shim. Generated model.so code must use that stable C
// ABI instead of including this private ATen header directly.
//
// Thread affinity: construct and destroy on the same thread, in LIFO order.
// The stack is thread-local and deliberately unsynchronized -- this sits on
// the per-GEMM dispatch path -- so the destructor cannot pop another thread's
// stack. A guard destroyed off-thread (e.g. a PyCapsule GC-finalized on the
// finalizer thread) warns and skips its pop, leaving the owner's entry until
// that thread's stack goes away with it. Out-of-order pops on one thread
// remove the top entry, not necessarily this guard's.
class TORCH_CUDA_CPP_API TunableDynamicDimsGuard {
 public:
  explicit TunableDynamicDimsGuard(DynamicDimsMask mask);
  ~TunableDynamicDimsGuard();

  TunableDynamicDimsGuard(const TunableDynamicDimsGuard&) = delete;
  TunableDynamicDimsGuard& operator=(const TunableDynamicDimsGuard&) = delete;
  TunableDynamicDimsGuard(TunableDynamicDimsGuard&&) = delete;
  TunableDynamicDimsGuard& operator=(TunableDynamicDimsGuard&&) = delete;

 private:
  std::thread::id owner_thread_;
};

class ITimer {
  public:
    ITimer() = default;
    virtual ~ITimer() = default;

    virtual void Start() = 0;
    virtual void End() = 0;

    /// Computes the elapsed time in milliseconds between Start() and End()
    virtual float Duration() = 0;
};

} // namespace at::cuda::tunable
