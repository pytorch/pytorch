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

#include <c10/util/CallOnce.h>

#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace at::cuda::tunable {

static void TunableLog(const std::string& msg) {
  static const char *env = getenv("PYTORCH_TUNABLEOP_VERBOSE");
  if (env != nullptr && strcmp(env, "1") == 0) {
    std::cerr << msg << std::endl;
  }
}
#define TUNABLE_LOG(...) TunableLog(c10::str(__VA_ARGS__))

enum TuningStatus {
  OK = 0,
  FAIL = 1,
  UNSUPPORTED = 2,
};

// Mapping from params signature to kernel id
class ResultEntry {
  public:
    explicit ResultEntry(const std::string& key, double time) : key_(key), time_(time) {}
    bool operator==(const ResultEntry& other) { return key_ == other.key_; }
    bool operator!=(const ResultEntry& other) { return key_ != other.key_; }
    operator std::string () { return key_; }
    friend std::ostream& operator<<(std::ostream& stream, const ResultEntry& entry);
    static ResultEntry Null() { return ResultEntry("Null", 0.0); }
    static ResultEntry Default() { return ResultEntry("Default", 0.0); }

  private:
    std::string key_;
    double time_;
};

typedef std::unordered_map<std::string, ResultEntry> KernelMap;
typedef std::unordered_map<std::string, KernelMap> ResultsMap;

struct TuningResults {
  // Validates if these results are compatible with the libraries
  std::unordered_map<std::string, std::string> validators;

  // Mapping from Callable signature to Callable's tuning result
  ResultsMap results;
};

class TuningResultsManager {
  public:
    TuningResultsManager() = default;
    ~TuningResultsManager() = default;

    KernelMap Lookup(const std::string& op_signature);

    ResultEntry Lookup(const std::string& op_signature, const std::string& params_signature);

    inline void AddImpl(const std::string& op_signature,
        const std::string& params_signature,
        ResultEntry best,
        KernelMap& kernel_map);

    void Add(const std::string& op_signature,
        const std::string& params_signature,
        ResultEntry best);

    void Delete(const std::string& op_signature, const std::string& params_signature);

    inline void DisjointMergeImpl(
        const std::string& op_signature,
        const KernelMap& kernel_map,
        /*out*/ ResultsMap& results);

    void Load(const ResultsMap& results_to_load);

    ResultsMap Dump();

    void DisjointMerge(const std::string& op_signature, const KernelMap& kernel_map);

    size_t GetSize();

  private:
    std::mutex lock_;
    ResultsMap results_;
};

class TuningResultsValidator {
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
    std::string GetPyTorchVersion() const;
    TuningStatus ValidatePyTorchVersion(const std::string& value) const;

  public:
    static constexpr const std::array mandatory_keys{"PT_VERSION"};

  private:
    GetValidateFuncs validators_;
};

class TuningContext {
  public:
    TuningContext();
    ~TuningContext();
    TuningContext(TuningContext &) = delete;
    TuningContext(TuningContext &&) = delete;
    TuningContext &operator=(TuningContext &) = delete;
    TuningContext &operator=(TuningContext &&) = delete;

    void EnableTunableOp();
    void DisableTunableOp();
    bool IsTunableOpEnabled() const;

    void EnableTuning();
    void DisableTuning();
    bool IsTuningEnabled() const;

    void SetMaxTuningDurationMs(int max_duration_ms);
    int GetMaxTuningDurationMs() const;

    void SetMaxTuningIterations(int max_iter);
    int GetMaxTuningIterations() const;

    void SetMaxWarmupDurationMs(int max_duration_ms);
    int GetMaxWarmupDurationMs() const;

    void SetMaxWarmupIterations(int max_iter);
    int GetMaxWarmupIterations() const;

    void EnableTunableOpAndTuning();
    void DisableTunableOpAndTuning();

    TuningResultsManager& GetTuningResultsManager();

    TuningResultsValidator& GetTuningResultsValidator();

    TuningResults GetTuningResults();

    TuningStatus LoadTuningResults(const TuningResults& tr);

    void SetFilename(const std::string& filename);
    std::string GetFilename() const;

  protected:
    bool ReadFile(const std::string& filename);
    bool WriteFile(const std::string& filename);

  private:
    bool enable_;
    bool tuning_enable_;
    bool manager_initialized_;
    int max_tuning_duration_ms_;
    int max_tuning_iterations_;
    int max_warmup_duration_ms_;
    int max_warmup_iterations_;
    mutable TuningResultsManager manager_;
    mutable c10::once_flag manager_init_once_;
    TuningResultsValidator validator_;
    std::string filename_;
    size_t results_count_from_input_file_;
};

TuningContext* getTuningContext();

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
