// Original TunableOp is from onnxruntime.
// https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/framework/tunable.h
// https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/rocm/tunable
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// Adapting TunableOp into PyTorch
// Copyright (c) Advanced Micro Devices, Inc.
//
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/cuda/tunable/Tunable.h>
#include <c10/util/Exception.h>

#ifndef _WIN32
#include <cxxabi.h>
#endif

#include <chrono>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace at::cuda {

tunable::TuningContext* getTuningContext() {
  static tunable::TuningContext* obj = new tunable::TuningContext;
  return obj;
}

} // namespace at::cuda

namespace at::cuda::tunable {

// TuningResultsManager

KernelMap TuningResultsManager::Lookup(const std::string& op_signature) {
  std::scoped_lock l{lock_};
  auto it = results_.find(op_signature);
  if (it == results_.cend()) {
    return {};
  }
  return it->second;  // copied
}

int TuningResultsManager::Lookup(const std::string& op_signature, const std::string& params_signature) {
  std::scoped_lock l{lock_};
  auto kernel_map_it = results_.find(op_signature);
  if (kernel_map_it == results_.cend()) {
    return -1;
  }

  const auto& km = kernel_map_it->second;
  auto it = km.find(params_signature);
  if (it == km.cend()) {
    return -1;
  }
  return it->second;
}

inline void TuningResultsManager::AddImpl(const std::string& op_signature,
    const std::string& params_signature,
    int best_id,
    KernelMap& kernel_map) {
  auto it = kernel_map.find(params_signature);
  if (it != kernel_map.end()) {
    if (it->second != best_id) {
      std::cerr << op_signature << "(" << params_signature << ") already has a best kernel "
        << "id=" << it->second << " selected, want to add a different best kernel id=" << best_id
        << ", the new kernel id will be ignored." << std::endl;
    }
    return;
  }

  std::cerr << op_signature << "(" << params_signature << ") -> " << best_id << std::endl;
  kernel_map[params_signature] = best_id;
}

void TuningResultsManager::Add(const std::string& op_signature, const std::string& params_signature, int best_id) {
  std::scoped_lock l{lock_};

  auto it = results_.find(op_signature);
  if (it == results_.end()) {
    it = results_.insert({op_signature, {}}).first;
  }

  AddImpl(op_signature, params_signature, best_id, it->second);
}

void TuningResultsManager::Delete(const std::string& op_signature, const std::string& params_signature) {
  std::scoped_lock l{lock_};

  auto it = results_.find(op_signature);
  if (it == results_.end()) {
    return;
  }

  auto it2 = it->second.find(params_signature);
  if (it2 == it->second.end()) {
    return;
  }

  std::cerr << op_signature << "(" << params_signature << ")" << std::endl;
  it->second.erase(it2);
}

inline void TuningResultsManager::DisjointMergeImpl(
    const std::string& op_signature,
    const KernelMap& kernel_map,
    /*out*/ std::unordered_map<std::string, KernelMap>& results) {
  auto it = results.find(op_signature);
  if (it == results.end()) {
    for (const auto& [param_sig, kernel_id] : kernel_map) {
        std::cerr << op_signature << "(" << param_sig << ") -> " << kernel_id << std::endl;
    }
    results[op_signature] = kernel_map;
    return;
  }

  for (const auto& [params_signature, best_id] : kernel_map) {
    AddImpl(op_signature, params_signature, best_id, it->second);
  }
}

void TuningResultsManager::Load(const std::unordered_map<std::string, KernelMap>& results_to_load) {
  std::scoped_lock l{lock_};
  for (const auto& [op_signature, kernel_map] : results_to_load) {
    DisjointMergeImpl(op_signature, kernel_map, results_);
  }
}

std::unordered_map<std::string, KernelMap> TuningResultsManager::Dump() {
  std::scoped_lock l{lock_};
  return results_;
}

void TuningResultsManager::DisjointMerge(const std::string& op_signature, const KernelMap& kernel_map) {
  std::scoped_lock l{lock_};
  DisjointMergeImpl(op_signature, kernel_map, results_);
}

// TuningResultsValidator

TuningResultsValidator::TuningResultsValidator() {
  RegisterValidator(
      "PT_VERSION",
      [this]() { return GetPyTorchVersion(); },
      [this](auto&& k) { return ValidatePyTorchVersion(std::forward<decltype(k)>(k)); });

  RegisterValidator(
      "PT_GIT_COMMIT",
      [this]() { return GetPyTorchGitCommit(); },
      [this](auto&& k) { return ValidatePyTorchGitCommit(std::forward<decltype(k)>(k)); });
}

std::unordered_map<std::string, std::string> TuningResultsValidator::GetAllValidators() const {
  std::unordered_map<std::string, std::string> ret;
  for (const auto& [key, get_validate_func_pair] : validators_) {
    const GetFunc& getter = get_validate_func_pair.first;
    ret[key] = getter();
  }
  return ret;
}

static bool CheckMandatoryKeys(
    const TuningResultsValidator::GetValidateFuncs& gv_funcs,
    const std::unordered_map<std::string, std::string>& to_check) {
  bool passed = true;
  std::ostringstream oss;
  for (const auto& k : TuningResultsValidator::mandatory_keys) {
    if (gv_funcs.find(k) == gv_funcs.end()) {
      passed = false;
      oss << "key=\"" << k << "\" is not registered for Get and Validate. ";
    }

    if (to_check.find(k) == to_check.end()) {
      passed = false;
      oss << "key=\"" << k << "\" is not provided for validation. ";
    }
  }
  return passed;
}

static bool CheckKeysMatching(
    const TuningResultsValidator::GetValidateFuncs& gv_funcs,
    const std::unordered_map<std::string, std::string>& to_check) {
  auto get_keys = [](const auto& it) -> std::string { return it.first; };
  std::vector<std::string> required_keys;
  std::vector<std::string> provided_keys;
  std::transform(gv_funcs.cbegin(), gv_funcs.cend(), std::back_inserter(required_keys), get_keys);
  std::transform(to_check.cbegin(), to_check.cend(), std::back_inserter(provided_keys), get_keys);
  std::sort(required_keys.begin(), required_keys.end());
  std::sort(provided_keys.begin(), provided_keys.end());

  std::unordered_set<std::string> intersection;
  std::set_intersection(required_keys.cbegin(), required_keys.cend(),
                        provided_keys.cbegin(), provided_keys.cend(),
                        std::inserter(intersection, intersection.end()));
  bool matched = true;
  std::ostringstream oss;
  if (intersection.size() != required_keys.size()) {
    matched = false;
    for (const auto& k : required_keys) {
      if (intersection.find(k) == intersection.end()) {
        oss << "Unmatched validator: \"" << k << "\" is required, but the tuning results does not provide it. ";
      }
    }
  }
  if (intersection.size() != provided_keys.size()) {
    matched = false;
    for (const auto& k : provided_keys) {
      if (intersection.find(k) == intersection.end()) {
        oss << "Unmatched validator: \"" << k << "\" is provided, but onnxruntime is unable to consume it. ";
      }
    }
  }
  return matched;
}

TuningStatus TuningResultsValidator::ValidateAll(
        const std::unordered_map<std::string,
        std::string>& to_validate) const {
  if (!CheckMandatoryKeys(validators_, to_validate)) {
    return FAIL;
  }
  if (!CheckKeysMatching(validators_, to_validate)) {
    return FAIL;
  }

  for (const auto& [key, value] : to_validate) {
    const auto& it = validators_.find(key);
    TORCH_CHECK(it != validators_.cend());
    const ValidateFunc& validator = it->second.second;
    if (!validator(value)) {
      return FAIL;
    }
  }

  return OK;
}

void TuningResultsValidator::RegisterValidator(const std::string& key, const GetFunc& gf, const ValidateFunc& vf) {
  TORCH_CHECK(validators_.find(key) == validators_.end());
  validators_[key] = std::make_pair(gf, vf);
}

std::string TuningResultsValidator::GetPyTorchVersion() const {
  return "TODO.PyTorchVersion";
}

TuningStatus TuningResultsValidator::ValidatePyTorchVersion(const std::string& value) const {
  if (value == GetPyTorchVersion()) {
    return OK;
  }
  return FAIL;
}

std::string TuningResultsValidator::GetPyTorchGitCommit() const {
  return "TODO.PyTorchGitCommit";
}

TuningStatus TuningResultsValidator::ValidatePyTorchGitCommit(const std::string& value) const {
  if (value == GetPyTorchGitCommit()) {
    return OK;
  }
  return FAIL;
}

// TuningContext

TuningContext::TuningContext() : enable_{false}, tuning_enable_{false}, max_tuning_duration_ms_{} {
}

void TuningContext::EnableTunableOp() {
  std::cerr << "Enable TunableOp" << std::endl;
  enable_ = true;
}

void TuningContext::DisableTunableOp() {
  std::cerr << "Disable TunableOp" << std::endl;
  enable_ = false;
}

bool TuningContext::IsTunableOpEnabled() const {
  static const char *env = std::getenv("PYTORCH_TUNABLEOP_ENABLED");
  if (env != nullptr && strcmp(env, "1") == 0) {
    TORCH_WARN_ONCE("PYTORCH_TUNABLEOP_ENABLED=1");
    return true;
  }
  return enable_;
}

void TuningContext::EnableTuning() {
  std::cerr << "Enable Tuning for TunableOp" << std::endl;
  tuning_enable_ = true;
}

void TuningContext::DisableTuning() {
  std::cerr << "Disable Tuning for TunableOp" << std::endl;
  tuning_enable_ = false;
}

bool TuningContext::IsTuningEnabled() const {
  static const char *env = std::getenv("PYTORCH_TUNABLEOP_TUNING");
  if (env != nullptr && strcmp(env, "1") == 0) {
    TORCH_WARN_ONCE("PYTORCH_TUNABLEOP_TUNING=1");
    return true;
  }
  return tuning_enable_;
}

void TuningContext::SetMaxTuningDurationMs(int max_duration_ms) {
  max_tuning_duration_ms_ = max_duration_ms;
}

int TuningContext::GetMaxTuningDurationMs() const {
  return max_tuning_duration_ms_;
}

void TuningContext::EnableTunableOpAndTuning() {
  EnableTunableOp();
  EnableTuning();
}

void TuningContext::DisableTunableOpAndTuning() {
  DisableTunableOp();
  DisableTuning();
}

TuningResultsManager& TuningContext::GetTuningResultsManager() {
  return manager_;
}

const TuningResultsManager& TuningContext::GetTuningResultsManager() const {
  return manager_;
}

const TuningResultsValidator& TuningContext::GetTuningResultsValidator() const {
  return validator_;
}

TuningResults TuningContext::GetTuningResults() {
  TuningResults tr;
  tr.validators = GetTuningResultsValidator().GetAllValidators();
  tr.results = GetTuningResultsManager().Dump();
  return tr;
}

TuningStatus TuningContext::LoadTuningResults(const TuningResults& tr) {
  TORCH_CHECK(GetTuningResultsValidator().ValidateAll(tr.validators));
  GetTuningResultsManager().Load(tr.results);
  return OK;
}

} // namespace at::cuda::tunable
