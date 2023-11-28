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

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace at::cuda::tunable {

enum TuningStatus {
  OK = 0,
  FAIL = 1,
  UNSUPPORTED = 2,
};

// Mapping from params signature to kernel id
typedef std::unordered_map<std::string, int> KernelMap;
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

    KernelMap Lookup(const std::string& op_signature);

    int Lookup(const std::string& op_signature, const std::string& params_signature);

    inline void AddImpl(const std::string& op_signature,
        const std::string& params_signature,
        int best_id,
        KernelMap& kernel_map);

    void Add(const std::string& op_signature, const std::string& params_signature, int best_id);

    void Delete(const std::string& op_signature, const std::string& params_signature);

    inline void DisjointMergeImpl(
        const std::string& op_signature,
        const KernelMap& kernel_map,
        /*out*/ ResultsMap& results);

    void Load(const ResultsMap& results_to_load);

    ResultsMap Dump();

    void DisjointMerge(const std::string& op_signature, const KernelMap& kernel_map);

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

    std::string GetPyTorchGitCommit() const;
    TuningStatus ValidatePyTorchGitCommit(const std::string& value) const;

  public:
    static constexpr const std::array mandatory_keys{"PT_VERSION", "PT_GIT_COMMIT"};

  private:
    GetValidateFuncs validators_;
};

class TuningContext {
  public:
    TuningContext();
    ~TuningContext() = default;

    void EnableTunableOp();
    void DisableTunableOp();
    bool IsTunableOpEnabled() const;

    void EnableTuning();
    void DisableTuning();
    bool IsTuningEnabled() const;

    void SetMaxTuningDurationMs(int max_duration_ms);
    int GetMaxTuningDurationMs() const;

    void EnableTunableOpAndTuning();
    void DisableTunableOpAndTuning();

    TuningResultsManager& GetTuningResultsManager();

    const TuningResultsManager& GetTuningResultsManager() const;

    const TuningResultsValidator& GetTuningResultsValidator() const;

    TuningResults GetTuningResults();

    TuningStatus LoadTuningResults(const TuningResults& tr);

  private:
    bool enable_;
    bool tuning_enable_;
    int max_tuning_duration_ms_;
    TuningResultsManager manager_;
    TuningResultsValidator validator_;
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
