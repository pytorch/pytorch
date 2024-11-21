// Original TunableOp is from onnxruntime.
// https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/framework/tunable.h
// https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/rocm/tunable
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// Adapting TunableOp into PyTorch
// Copyright (c) Advanced Micro Devices, Inc.
//
#pragma once

#include <c10/util/env.h>
#include <c10/util/CallOnce.h>
#include <c10/util/StringUtil.h>

#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace at::cuda::tunable {

namespace detail {

struct MaybeDelete {
  bool owns_pointer;
  void operator()(std::ostream* os) const { if (owns_pointer) delete os; }
};

using OstreamPtr = std::unique_ptr<std::ostream, MaybeDelete>;

inline OstreamPtr get_stream(const std::string& filename) {
  if (filename == "out") {
    return OstreamPtr { &std::cout, MaybeDelete {false} };
  }
  else if (filename == "err") {
    return OstreamPtr { &std::cerr, MaybeDelete {false} };
  }
  else {
    return OstreamPtr { new std::ofstream {filename.c_str()}, MaybeDelete {true} };
  }
}

}

template<class... Types>
static void TunableLog(int level, Types... args) {
  static const auto env_file =
      c10::utils::get_env("PYTORCH_TUNABLEOP_VERBOSE_FILENAME");
  static const auto env_verbose =
      c10::utils::get_env("PYTORCH_TUNABLEOP_VERBOSE");
  static int level_user = env_verbose ? stoi(env_verbose.value()) : 0;
  static auto streamptr =
      detail::get_stream(env_file ? env_file.value() : "err");
  if (level_user >= level) {
    (*streamptr) << c10::str(args...) << std::endl;
  }
}
#define TUNABLE_LOGV(LEVEL, ...) TunableLog(LEVEL, __VA_ARGS__)
#define TUNABLE_LOG1(...) TUNABLE_LOGV(1, __VA_ARGS__)
#define TUNABLE_LOG2(...) TUNABLE_LOGV(2, __VA_ARGS__)
#define TUNABLE_LOG3(...) TUNABLE_LOGV(3, __VA_ARGS__)

enum TORCH_CUDA_CPP_API TuningStatus {
  OK = 0,
  FAIL = 1,
  UNSUPPORTED = 2,
};

// Mapping from params signature to kernel id
class TORCH_CUDA_CPP_API ResultEntry {
  public:
    explicit ResultEntry(std::string  key, double time) : key_(std::move(key)), time_(time) {}
    bool operator==(const ResultEntry& other) { return key_ == other.key_; }
    bool operator!=(const ResultEntry& other) { return key_ != other.key_; }
    operator std::string () { return key_; }
    std::string GetKey() const { return key_; }
    double GetTime() const { return time_; }
    friend std::ostream& operator<<(std::ostream& stream, const ResultEntry& entry);
    static ResultEntry Null() { return ResultEntry("Null", 0.0); }
    static ResultEntry Default() { return ResultEntry("Default", 0.0); }

  private:
    std::string key_;
    double time_;
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

    void AddImpl(const std::string& op_signature,
        const std::string& params_signature,
        ResultEntry best,
        KernelMap& kernel_map);

    void Add(const std::string& op_signature,
        const std::string& params_signature,
        ResultEntry best);

    void Delete(const std::string& op_signature, const std::string& params_signature);

    void DisjointMergeImpl(
        const std::string& op_signature,
        const KernelMap& kernel_map,
        /*out*/ ResultsMap& results);

    void Load(const ResultsMap& results_to_load);

    ResultsMap Dump();

    void DisjointMerge(const std::string& op_signature, const KernelMap& kernel_map);

    size_t GetSize();

    void RecordUntuned( std::ofstream& untuned_file, const std::string& op_signature, const std::string& params_signature);
  private:
    std::mutex lock_;
    ResultsMap results_;
    UntunedMap untuned_results_;

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

    void EnableNumericsCheck(bool value);
    bool IsNumericsCheckEnabled() const;

    void SetMaxTuningDurationMs(int max_duration_ms);
    int GetMaxTuningDurationMs() const;

    void SetMaxTuningIterations(int max_iter);
    int GetMaxTuningIterations() const;

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

    void WriteFileOnExit(bool value);

    bool ReadFile(const std::string& filename={});
    bool WriteFile(const std::string& filename={});

  private:
    bool enable_;
    bool tuning_enable_;
    bool record_untuned_enable_;
    bool manager_initialized_;
    bool write_file_on_exit_;
    bool numerics_check_enable_;
    int max_tuning_duration_ms_;
    int max_tuning_iterations_;
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
};

TORCH_CUDA_CPP_API TuningContext* getTuningContext();

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
