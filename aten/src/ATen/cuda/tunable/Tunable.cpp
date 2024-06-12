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
#include <c10/util/StringUtil.h>
#include <torch/version.h>

#ifndef _WIN32
#include <cxxabi.h>
#endif

#include <chrono>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace at::cuda::tunable {

namespace {

TuningContext tuning_context;

} // anonymous namespace

TuningContext* getTuningContext() {
  return &tuning_context;
}

std::ostream& operator<<(std::ostream& stream, const ResultEntry& entry) {
  return stream << entry.key_ << "," << entry.time_;
}

// TuningResultsManager

KernelMap TuningResultsManager::Lookup(const std::string& op_signature) {
  std::scoped_lock l{lock_};
  auto it = results_.find(op_signature);
  if (it == results_.cend()) {
    return {};
  }
  return it->second;  // copied
}

ResultEntry TuningResultsManager::Lookup(const std::string& op_signature, const std::string& params_signature) {
  std::scoped_lock l{lock_};
  auto kernel_map_it = results_.find(op_signature);
  if (kernel_map_it == results_.cend()) {
    TUNABLE_LOG3("missing op_signature, returning null ResultEntry");
    return ResultEntry::Null();
  }

  const auto& km = kernel_map_it->second;
  auto it = km.find(params_signature);
  if (it == km.cend()) {
    TUNABLE_LOG3("missing params_signature, returning null ResultEntry");
    return ResultEntry::Null();
  }
  return it->second;
}

inline void TuningResultsManager::AddImpl(const std::string& op_signature,
    const std::string& params_signature,
    ResultEntry best,
    KernelMap& kernel_map) {
  auto it = kernel_map.find(params_signature);
  if (it != kernel_map.end()) {
    if (it->second != best) {
      TUNABLE_LOG1(op_signature, "(", params_signature, ") already has a best kernel ",
          "id=", it->second, " selected, want to add a different best kernel ", best,
          ", the new kernel id will be ignored.");
    }
    return;
  }

  TUNABLE_LOG2(op_signature, "(", params_signature, ") -> ", best);
  kernel_map.emplace(params_signature, best);
}

void TuningResultsManager::Add(const std::string& op_signature, const std::string& params_signature, ResultEntry best) {
  std::scoped_lock l{lock_};

  auto it = results_.find(op_signature);
  if (it == results_.end()) {
    it = results_.insert({op_signature, {}}).first;
  }

  AddImpl(op_signature, params_signature, best, it->second);
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

  TUNABLE_LOG2(op_signature, "(", params_signature, ")");
  it->second.erase(it2);
}

inline void TuningResultsManager::DisjointMergeImpl(
    const std::string& op_signature,
    const KernelMap& kernel_map,
    /*out*/ std::unordered_map<std::string, KernelMap>& results) {
  auto it = results.find(op_signature);
  if (it == results.end()) {
    for (const auto& [param_sig, kernel_id] : kernel_map) {
      TUNABLE_LOG2(op_signature, "(", param_sig, ") -> ", kernel_id);
    }
    results[op_signature] = kernel_map;
    return;
  }

  for (const auto& [params_signature, best] : kernel_map) {
    AddImpl(op_signature, params_signature, best, it->second);
  }
}

void TuningResultsManager::Load(const std::unordered_map<std::string, KernelMap>& results_to_load) {
  TUNABLE_LOG1("Loading results");
  std::scoped_lock l{lock_};
  for (const auto& [op_signature, kernel_map] : results_to_load) {
    DisjointMergeImpl(op_signature, kernel_map, results_);
  }
}

ResultsMap TuningResultsManager::Dump() {
  std::scoped_lock l{lock_};
  return results_;
}

void TuningResultsManager::DisjointMerge(const std::string& op_signature, const KernelMap& kernel_map) {
  std::scoped_lock l{lock_};
  DisjointMergeImpl(op_signature, kernel_map, results_);
}

size_t TuningResultsManager::GetSize() {
  size_t size = 0;
  std::scoped_lock l{lock_};
  for (const auto& [op_signature, kernel_map] : results_) {
    size += kernel_map.size();
  }
  return size;
}

// TuningResultsValidator

TuningResultsValidator::TuningResultsValidator() {
  RegisterValidator(
      "PT_VERSION",
      [this]() { return GetPyTorchVersion(); },
      [this](auto&& k) { return ValidatePyTorchVersion(std::forward<decltype(k)>(k)); });
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
  for (const auto& k : TuningResultsValidator::mandatory_keys) {
    if (gv_funcs.find(k) == gv_funcs.end()) {
      passed = false;
      TUNABLE_LOG1("key=\"", k, "\" is not registered for Get and Validate. ");
    }

    if (to_check.find(k) == to_check.end()) {
      passed = false;
      TUNABLE_LOG1("key=\"", k, "\" is not provided for validation. ");
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
  if (intersection.size() != required_keys.size()) {
    matched = false;
    for (const auto& k : required_keys) {
      if (intersection.find(k) == intersection.end()) {
        TORCH_WARN("Unmatched validator: \"", k, "\" is required, but the tuning results does not provide it. ");
      }
    }
  }
  if (intersection.size() != provided_keys.size()) {
    matched = false;
    for (const auto& k : provided_keys) {
      if (intersection.find(k) == intersection.end()) {
        TORCH_WARN("Unmatched validator: \"", k, "\" is provided, but pytorch is unable to consume it. ");
      }
    }
  }
  return matched;
}

TuningStatus TuningResultsValidator::ValidateAll(
        const std::unordered_map<std::string, std::string>& to_validate) const {
  if (!CheckMandatoryKeys(validators_, to_validate)) {
    return FAIL;
  }
  if (!CheckKeysMatching(validators_, to_validate)) {
    return FAIL;
  }

  for (const auto& [key, value] : to_validate) {
    const auto& it = validators_.find(key);
    if (it == validators_.cend()) {
      TORCH_WARN("Failed to lookup validator using key ", key);
      for (const auto& [key2, val2] : validators_) {
        TORCH_WARN("available key ", key2);
      }
      return FAIL;
    }
    const ValidateFunc& validator = it->second.second;
    if (validator(value) != OK) {
      TORCH_WARN("Failed validator: ", key);
      return FAIL;
    }
  }

  return OK;
}

void TuningResultsValidator::RegisterValidator(const std::string& key, const GetFunc& gf, const ValidateFunc& vf) {
  if (validators_.find(key) != validators_.end()) {
    TORCH_WARN("Attempting to re-register validator with key ", key);
  }
  else {
    validators_[key] = std::make_pair(gf, vf);
  }
}

std::string TuningResultsValidator::GetPyTorchVersion() const {
  return TORCH_VERSION;
}

TuningStatus TuningResultsValidator::ValidatePyTorchVersion(const std::string& value) const {
  if (value == GetPyTorchVersion()) {
    return OK;
  }
  return FAIL;
}

// TuningContext

TuningContext::TuningContext() :
    enable_{false},
    tuning_enable_{true},
    manager_initialized_{false},
    write_file_on_exit_{true},
    numerics_check_enable_{false},
    max_tuning_duration_ms_{30},
    max_tuning_iterations_{100},
    max_warmup_duration_ms_{0},
    max_warmup_iterations_{0},
    icache_flush_{true},
    rotating_buffer_size_{-1},
    filename_{},
    results_count_from_input_file_{0}
{
}

TuningContext::~TuningContext() {
  if (!manager_initialized_) {
    // TuningResultsManager was never initialized, no tuning requested or performed.
    // This can happen in a DDP job where a python process spawns other workers
    // but doesn't do any computation itself.
    return;
  }
  auto filename = GetFilename();
  if (IsTunableOpEnabled() && IsTuningEnabled() && !filename.empty() && write_file_on_exit_) {
    if (results_count_from_input_file_ < GetTuningResultsManager().GetSize()) {
      if (results_count_from_input_file_ > 0) {
        TUNABLE_LOG1("additional tuning results available, rewriting file ", filename);
      }
      else {
        TUNABLE_LOG1("writing file ", filename);
      }
      if (!WriteFile(filename)) {
        TUNABLE_LOG1("failed to write file ", filename);
      }
    }
  }
}

void TuningContext::EnableTunableOp(bool value) {
  enable_ = value;
  if (value) {
    TUNABLE_LOG1("Enable TunableOp");
  }
  else {
    TUNABLE_LOG1("Disable TunableOp");
  }
}

bool TuningContext::IsTunableOpEnabled() const {
  static const char *env = std::getenv("PYTORCH_TUNABLEOP_ENABLED");
  if (env != nullptr && strcmp(env, "1") == 0) {
    return true;
  }
  return enable_;
}

void TuningContext::EnableTuning(bool value) {
  tuning_enable_ = value;
  if (value) {
    TUNABLE_LOG1("Enable Tuning for TunableOp");
  }
  else {
    TUNABLE_LOG1("Disable Tuning for TunableOp");
  }
}

bool TuningContext::IsTuningEnabled() const {
  static const char *env = std::getenv("PYTORCH_TUNABLEOP_TUNING");
  if (env != nullptr && strcmp(env, "0") == 0) {
    return false;
  }
  return tuning_enable_;
}

void TuningContext::WriteFileOnExit(bool value) {
  write_file_on_exit_ = value;
}

void TuningContext::EnableNumericsCheck(bool value) {
  numerics_check_enable_ = value;
}

bool TuningContext::IsNumericsCheckEnabled() const {
  static const char *env = getenv("PYTORCH_TUNABLEOP_NUMERICAL_CHECK");
  if (env != nullptr && strcmp(env, "1") == 0) {
    return true;
  }
  return numerics_check_enable_;
}

void TuningContext::SetMaxTuningDurationMs(int max_duration_ms) {
  max_tuning_duration_ms_ = max_duration_ms < 0 ? 0 : max_duration_ms;
}

int TuningContext::GetMaxTuningDurationMs() const {
  static const char *env = std::getenv("PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS");
  if (env != nullptr) {
    int val = atoi(env);
    return val < 0 ? 0 : val;
  }
  return max_tuning_duration_ms_;
}

void TuningContext::SetMaxTuningIterations(int max_iter) {
  max_tuning_iterations_ = max_iter < 0 ? 0 : max_iter;
}

int TuningContext::GetMaxTuningIterations() const {
  static const char *env = std::getenv("PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS");
  if (env != nullptr) {
    int val = atoi(env);
    return val < 0 ? 0 : val;
  }
  return max_tuning_iterations_;
}

void TuningContext::SetMaxWarmupDurationMs(int max_duration_ms) {
  max_warmup_duration_ms_ = max_duration_ms < 0 ? 0 : max_duration_ms;
}

int TuningContext::GetMaxWarmupDurationMs() const {
  static const char *env = std::getenv("PYTORCH_TUNABLEOP_MAX_WARMUP_DURATION_MS");
  if (env != nullptr) {
    int val = atoi(env);
    return val < 0 ? 0 : val;
  }
  return max_warmup_duration_ms_;
}

void TuningContext::SetMaxWarmupIterations(int max_iter) {
  max_warmup_iterations_ = max_iter < 0 ? 0 : max_iter;
}

int TuningContext::GetMaxWarmupIterations() const {
  static const char *env = std::getenv("PYTORCH_TUNABLEOP_MAX_WARMUP_ITERATIONS");
  if (env != nullptr) {
    int val = atoi(env);
    return val < 0 ? 0 : val;
  }
  return max_warmup_iterations_;
}

void TuningContext::EnableICacheFlush(bool value) {
  icache_flush_ = value;
}

bool TuningContext::IsICacheFlushEnabled() const {
  static const char *env = std::getenv("PYTORCH_TUNABLEOP_ICACHE_FLUSH_ENABLED");
  if (env != nullptr && strcmp(env, "0") == 0) {
    return false;
  }
  return icache_flush_;
}

void TuningContext::SetRotatingBufferSize(int size) {
  rotating_buffer_size_ = size < 0 ? 0 : size;
}

int TuningContext::GetRotatingBufferSize() const {
  static const char *env = std::getenv("PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE");
  if (env != nullptr) {
    constexpr int MB = 1024 * 1024;
    int val = atoi(env);
    return val < 0 ? 0 : val * MB;  // env var is specified as MB, returned as bytes
  }
  else {
    if (rotating_buffer_size_ < 0) {
      // negative buffer size (default) means query for L2 cache size
      int l2_cache_size = at::cuda::getCurrentDeviceProperties()->l2CacheSize;
      return l2_cache_size;
    }
    else {
      return rotating_buffer_size_;
    }
  }
}

TuningResultsManager& TuningContext::GetTuningResultsManager() {
  c10::call_once(manager_init_once_, [this]() {
    manager_initialized_ = true;
    if (GetFilename().empty()) {
      // if SetFilename() was not already called, call it now with the default or env var
      const char *env = std::getenv("PYTORCH_TUNABLEOP_FILENAME");
      std::string filename = (env == nullptr) ? "tunableop_results.csv" : env;
      SetFilename(filename, true);
    }
    auto filename = GetFilename();
    if (!filename.empty()) {
      ReadFile(filename);
      // attempt immediately to open file for writing to catch errors early
      std::ofstream file(filename, std::ios::out | std::ios::app);
      if (!file.good()) {
        TORCH_WARN("failed to open file '", filename, "' for writing; your tuning results will not be saved");
      }
    }
  });
  return manager_;
}

TuningResultsValidator& TuningContext::GetTuningResultsValidator() {
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

void TuningContext::SetFilename(const std::string& filename, bool insert_device_ordinal) {
  filename_ = filename;

  if (filename_.empty()) {
    return;
  }

  if (insert_device_ordinal) {
    // differentiate filename based on device ordinal to avoid
    // use case of one process per device writing to same file
    std::string device = c10::str(int(c10::cuda::current_device()));

    // does filename contain %d to insert device ordinal in specific location?
    const std::string TOKEN("%d");
    std::size_t found = filename_.find(TOKEN);
    if (found != std::string::npos) {
      filename_.replace(found, TOKEN.length(), device);
    }
    else {
      // no %d present, so append device ordinal before final '.'
      found = filename_.rfind(".");
      if (found != std::string::npos) {
        filename_.insert(found, device);
      }
      else {
        // all else fails, just append
        filename_.append(device);
      }
    }
  }
}

std::string TuningContext::GetFilename() const {
  return filename_;
}

bool TuningContext::ReadFile(const std::string& filename_) {
  std::string filename = filename_.empty() ? GetFilename() : filename_;
  TUNABLE_LOG1("reading tuning results from ", filename);
  ResultsMap results;
  std::unordered_map<std::string, std::string> validators;
  std::string line;
  std::ifstream file(filename);
  if (!file) {
    TUNABLE_LOG1("could not open ", filename, " for reading tuning results");
    return false;
  }
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }
    std::string part;
    std::vector<std::string> parts;
    std::stringstream line_as_stream(line);
    while (std::getline(line_as_stream, part, ',')) {
      parts.push_back(part);
    }
    if (parts[0] == "Validator" && parts.size() >= 3) {
      validators[parts[1]] = parts[2];
      TUNABLE_LOG1("Validator ", parts[1], "=", parts[2]);
    }
    else if (parts.size() >= 4) {
      results[parts[0]].emplace(parts[1], ResultEntry(parts[2], atof(parts[3].c_str())));
    }
    else if (parts.size() >= 3) {
      // the timestamp from the file is optional
      results[parts[0]].emplace(parts[1], ResultEntry(parts[2], 0));
    }
    else {
      TUNABLE_LOG1("could not parse line: ", line);
    }
  }
  if (GetTuningResultsValidator().ValidateAll(validators) != FAIL) {
    manager_.Load(results);
    results_count_from_input_file_ = manager_.GetSize();
  }
  else {
    TUNABLE_LOG1("results validator check failed");
    return false;
  }
  return true;
}

bool TuningContext::WriteFile(const std::string& filename_) {
  std::string filename = filename_.empty() ? GetFilename() : filename_;
  std::ofstream file(filename, std::ios::out | std::ios::trunc);
  if (!file.good()) {
    TUNABLE_LOG1("error opening tuning results file for writing ", filename);
    return false;
  }
  auto validators = GetTuningResultsValidator().GetAllValidators();
  for (const auto& [key, val] : validators) {
    file << "Validator," << key << "," << val << std::endl;
  }
  auto results = GetTuningResultsManager().Dump();
  for (const auto& [op_sig, kernelmap] : results) {
    for (const auto& [param_sig, result] : kernelmap) {
      file << op_sig << "," << param_sig << "," << result << std::endl;
    }
  }
  file.close();
  return true;
}

} // namespace at::cuda::tunable
