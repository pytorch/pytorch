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
#include <c10/util/env.h>
#include <torch/version.h>

#ifndef _WIN32
#include <cxxabi.h>
#endif

#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// for validators
#ifdef USE_ROCM
#ifdef _WIN32
#include <hip/hip_version.h>
#else
#include <rocm-core/rocm_version.h>
#endif
#define ROCBLAS_BETA_FEATURES_API
#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#endif

namespace at::cuda::tunable {

TuningContext* getTuningContext() {
  static TuningContext tuning_context;
  return &tuning_context;
}

std::ostream& operator<<(std::ostream& stream, const ResultEntry& entry) {
  static const bool blaslog = c10::utils::get_env("PYTORCH_TUNABLEOP_BLAS_LOG") == "1";
  if (!blaslog) {
    return stream << entry.key_ << "," << entry.time_;
  }
  else {
    return stream << entry.key_ << "," << entry.time_ << ",BLAS_PARAMS: " << entry.blas_sig_;
  }
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
    TUNABLE_LOG3("missing op_signature, returning null ResultEntry for ", op_signature, ",", params_signature);
    return ResultEntry::Null();
  }

  const auto& km = kernel_map_it->second;
  auto it = km.find(params_signature);
  if (it == km.cend()) {
    TUNABLE_LOG3("missing params_signature, returning null ResultEntry for ", op_signature, ",", params_signature);
    return ResultEntry::Null();
  }
  TUNABLE_LOG3("ResultEntry found for ", op_signature, ",", params_signature);
  return it->second;
}

void TuningResultsManager::AddImpl(const std::string& op_signature,
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
  kernel_map.emplace(params_signature, std::move(best));
}

void TuningResultsManager::Add(const std::string& op_signature, const std::string& params_signature, ResultEntry best) {
  bool is_new = false;
  ResultEntry inserted = ResultEntry::Null();

  // ---- mutate maps under results lock ----
  {
    std::scoped_lock l{lock_};
    auto& km = results_[op_signature];  // creates if missing
    is_new = (km.find(params_signature) == km.end());
    AddImpl(op_signature, params_signature, std::move(best), km);
    if (is_new) {
      inserted = km.at(params_signature);  // snapshot for I/O after unlocking
    }
  }
   if (!is_new) return;  // only write once per unique (op, params)

   TuningContext* ctx = getTuningContext();
  if (ctx->IsTuningEnabled() && !ctx->IsRecordUntunedEnabled()) {
    InitRealtimeAppend(ctx->GetFilename(), ctx->GetTuningResultsValidator().GetAllValidators());

    if (is_new && realtime_out_ && realtime_out_->good()) {
      AppendResultLine(op_signature, params_signature, inserted);
    }
  }

}

void TuningResultsManager::RecordUntuned( std::ofstream& untuned_file, const std::string& op_signature,
    const std::string& params_signature, const std::string& blas_signature) {
  std::scoped_lock l{lock_};
  if (!untuned_file.good()) {
    TORCH_WARN_ONCE("failed to open file for writing; untuned gemm will not be saved");
    return;
  } else {
    bool isNew = false;
    auto it = untuned_results_.find(op_signature);
    if (it == untuned_results_.end()) {
      it = untuned_results_.insert({op_signature, {}}).first;
      isNew = true;
    }

    auto it_kernel_map = it->second.find(params_signature);
    if (it_kernel_map == it->second.end()) {
      it->second.insert(params_signature);
      isNew = true;
    }

    if (isNew) {
      static const bool blaslog = c10::utils::get_env("PYTORCH_TUNABLEOP_BLAS_LOG") == "1";
      if (!blaslog) {
        untuned_file << op_signature << "," << params_signature << std::endl;
      }
      else {
        untuned_file << op_signature << "," << params_signature << ",BLAS_PARAMS: " << blas_signature << std::endl;
      }
      TUNABLE_LOG3("Untuned,", op_signature, ",", params_signature);
    }
  }
}

void TuningResultsManager::InitRealtimeAppend(const std::string& filename, const std::unordered_map<std::string, std::string>& validators) {
  std::scoped_lock fl{realtime_file_mutex_};

  if (realtime_out_ && realtime_out_->good() && realtime_filename_ == filename) {
    return;
  }

  if (realtime_out_ && realtime_filename_ != filename) {
    realtime_out_->flush();
    realtime_out_->close();
    realtime_out_.reset();
    validators_written_ = false;
  }

  bool file_exists = false;
  bool file_empty = true;

  {
    std::ifstream check_file(filename);
    if (check_file.good()) {
      file_exists = true;
      file_empty = (check_file.peek() == std::ifstream::traits_type::eof());
    }
  }

  realtime_out_ = std::make_unique<std::ofstream>(filename, std::ios::out | std::ios::app);

  if (!realtime_out_->good()) {
    TORCH_WARN("TunableOp realtime append: failed to open '", filename,"'");
    realtime_out_.reset();
    return;
  }

  if(!file_exists || file_empty) {
    for(const auto& [key, val] : validators) {
      (*realtime_out_) << "Validator," << key << "," << val << std::endl;
      realtime_out_->flush();
    }
    validators_written_ = true;

    TUNABLE_LOG2("Wrote validators to realtime output file");
  }

  realtime_filename_ = filename;
}

void TuningResultsManager::AppendResultLine(const std::string& op_sig, const std::string& param_sig, const ResultEntry& result) {
  std::scoped_lock fl{realtime_file_mutex_};

  if(!realtime_out_ || !realtime_out_->good()) {
    return;
  }

  (*realtime_out_) << op_sig << "," << param_sig << "," << result << std::endl;
  realtime_out_->flush(); //ensure immediate write to disk

  TUNABLE_LOG3("Realtime append: ", op_sig, "(", param_sig, ") -> ", result);
}

void TuningResultsManager::CloseRealtimeAppend() {
  std::scoped_lock fl{realtime_file_mutex_};


  if(realtime_out_) {
    realtime_out_->flush();
    realtime_out_->close();
    realtime_out_.reset();
    TUNABLE_LOG2("Closed realtime output file");
  }
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

void TuningResultsManager::DisjointMergeImpl(
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
      []() { return GetPyTorchVersion(); },
      [this](auto&& k) { return ValidatePyTorchVersion(std::forward<decltype(k)>(k)); });
#ifdef USE_ROCM
  // hip
  {
    // HIP version is more accurate than ROCm version.  User's environment could be a stock
    // ROCm install but with a mix of newer components, making ROCm version meaningless.
    std::string hip_version = c10::str(TORCH_HIP_VERSION);
    RegisterValidator(
       "HIP_VERSION",
       [hip_version]() { return hip_version; },
       [hip_version](auto&& k) {
        TUNABLE_LOG1("HIP_VERSION validation: expect ", k, " to match ", hip_version);
        return hip_version == k ? OK : FAIL;
      });
  }
  // gfx arch
  {
    std::string gcn_arch_name = at::cuda::getCurrentDeviceProperties()->gcnArchName;
    RegisterValidator(
        "GCN_ARCH_NAME",
        [gcn_arch_name]() { return gcn_arch_name; },
        [gcn_arch_name](auto&& k) {
          TUNABLE_LOG1("GCN_ARCH_NAME validation: expect ", k, " to match ", gcn_arch_name);
          return gcn_arch_name == k ? OK : FAIL;
        });
  }
  // rocblas
  {
    size_t rocblas_version_size;
    rocblas_get_version_string_size(&rocblas_version_size);
    std::string rocblas_version(rocblas_version_size - 1, '\0');
    rocblas_get_version_string(rocblas_version.data(), rocblas_version_size);
    RegisterValidator(
        "ROCBLAS_VERSION",
        [rocblas_version]() { return rocblas_version; },
        [rocblas_version](auto&& k) {
          TUNABLE_LOG1("ROCBLAS_VERSION validation: expect ", k, " to match ", rocblas_version);
          return rocblas_version == k ? OK : FAIL;
        });
  }
  // hipblaslt
  {
    int version;
    std::string revision(128, '\0');
    auto handle = at::cuda::getCurrentCUDABlasLtHandle();
    hipblasLtGetVersion(handle, &version);
    hipblasLtGetGitRevision(handle, revision.data());
    std::string hipblaslt_version =
        c10::str(version, "-", revision.c_str());
    RegisterValidator(
        "HIPBLASLT_VERSION",
        [hipblaslt_version]() { return hipblaslt_version; },
        [hipblaslt_version](auto&& k) {
          TUNABLE_LOG1("HIPBLASLT_VERSION validation: expect ", k, " to match ", hipblaslt_version);
          return hipblaslt_version == k ? OK : FAIL;
        });
  }
#endif
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

std::string TuningResultsValidator::GetPyTorchVersion() {
  return TORCH_VERSION;
}

TuningStatus TuningResultsValidator::ValidatePyTorchVersion(const std::string& value) const {
  TUNABLE_LOG1("PT_VERSION validation: expect ", value, " to match ", GetPyTorchVersion());
  if (value == GetPyTorchVersion()) {
    return OK;
  }
  return FAIL;
}

// TuningContext

TuningContext::TuningContext() :
    enable_{false},
    tuning_enable_{true},
    record_untuned_enable_{false},
    manager_initialized_{false},
    numerics_check_enable_{false},
    max_tuning_duration_ms_{30},
    max_tuning_iterations_{100},
    max_warmup_duration_ms_{0},
    max_warmup_iterations_{0},
    icache_flush_{true},
    rotating_buffer_size_{-1},
    results_count_from_input_file_{0},
    is_shutting_down_{false}
{
}

TuningContext::~TuningContext() {
  is_shutting_down_ = true;
  if (!manager_initialized_) {
    // TuningResultsManager was never initialized, no tuning requested or performed.
    // This can happen in a DDP job where a python process spawns other workers
    // but doesn't do any computation itself.
    return;
  }
  TUNABLE_LOG1("Closing File");
  GetTuningResultsManager().CloseRealtimeAppend(); // Since, we do instant logging by default now.

  if (untuned_file_.good()) {
    untuned_file_.close();
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
  static const bool eval = c10::utils::get_env("PYTORCH_TUNABLEOP_ENABLED") == "1";
  if (eval) {
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

void TuningContext::EnableRecordUntuned(bool value) {
  record_untuned_enable_ = value;
  if (value) {
    TUNABLE_LOG1("Enable Record Untuned for TunableOp");
  } else {
    TUNABLE_LOG1("Disable Record Untuned for TunableOp");
    TUNABLE_LOG1("Closing Untuned GEMM Results File");
    untuned_file_.close();
  }
}

bool TuningContext::IsTuningEnabled() const {
  static const bool eval = c10::utils::get_env("PYTORCH_TUNABLEOP_TUNING") == "0";
  if (eval) {
    return false;
  }
  return tuning_enable_;
}

bool TuningContext::IsRecordUntunedEnabled() const {
  static const bool eval = c10::utils::get_env("PYTORCH_TUNABLEOP_RECORD_UNTUNED") == "1";
  if (eval) {
    return true;
  }
  return record_untuned_enable_;
}

std::ofstream& TuningContext::GetUntunedFile(){
  if (!untuned_file_.is_open()) {
    const auto env = c10::utils::get_env("PYTORCH_TUNABLEOP_UNTUNED_FILENAME");
    std::string filename = (!env.has_value()) ? "tunableop_untuned.csv" : env.value();

    std::string device = c10::str(int(c10::cuda::current_device()));
    std::size_t found = filename.rfind('.');
    if (found != std::string::npos) {
      filename.insert(found, device);
    } else {
      // all else fails, just append
      filename.append(device);
    }

    untuned_file_ = std::ofstream(filename, std::ios::out | std::ios::app);
  }
  return untuned_file_;
}


void TuningContext::EnableNumericsCheck(bool value) {
  numerics_check_enable_ = value;
}

NumericalCheckConfig TuningContext::GetNumericalCheckConfig() const {
  const auto env_opt = c10::utils::get_env("PYTORCH_TUNABLEOP_NUMERICAL_CHECK");

  if (!env_opt.has_value()) {
    return numerics_cfg_;
  }

  const std::string& env = env_opt.value();

  if (env == "0") {
    return NumericalCheckConfig(false, 1e-5, 1e-5);
  }

  const size_t underscore = env.find('_');

  TORCH_CHECK(
      underscore != std::string::npos,
      "Invalid PYTORCH_TUNABLEOP_NUMERICAL_CHECK format. "
      "Expected 'atol_rtol', got: ",
      env);

  double atol = 0.0;
  double rtol = 0.0;

  try {
    atol = std::stod(env.substr(0, underscore));
    rtol = std::stod(env.substr(underscore + 1));
  } catch (const std::exception& e) {
    TORCH_CHECK(false, "Failed to parse PYTORCH_TUNABLEOP_NUMERICAL_CHECK: ", e.what());
  }

  TORCH_CHECK( atol > 0.0 && rtol > 0.0, "Tolerance values must be positive. atol=", atol, ", rtol=", rtol);
  return NumericalCheckConfig(true, atol, rtol);
}

void TuningContext::SetNumericalCheckConfig(bool enabled, double atol, double rtol) {
  TORCH_CHECK(atol > 0.0 && rtol > 0.0, "Numerical check tolerances must be positive");
  numerics_cfg_ = {enabled, atol, rtol};
}

bool TuningContext::IsNumericsCheckEnabled() const {
  const auto cfg = GetNumericalCheckConfig();
  return cfg.enabled || numerics_check_enable_;
}

void TuningContext::SetMaxTuningDurationMs(int max_duration_ms) {
  max_tuning_duration_ms_ = max_duration_ms < 0 ? 0 : max_duration_ms;
}

int TuningContext::GetMaxTuningDurationMs() const {
  static const auto env = c10::utils::get_env("PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS");
  if (env.has_value()) {
    int val = stoi(env.value());
    return val < 0 ? 0 : val;
  }
  return max_tuning_duration_ms_;
}

void TuningContext::SetMaxTuningIterations(int max_iter) {
  max_tuning_iterations_ = max_iter < 0 ? 0 : max_iter;
}

int TuningContext::GetMaxTuningIterations() const {
  static const auto env = c10::utils::get_env("PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS");
  if (env.has_value()) {
    int val = stoi(env.value());
    return val < 0 ? 0 : val;
  }
  return max_tuning_iterations_;
}

void TuningContext::SetMaxWarmupDurationMs(int max_duration_ms) {
  max_warmup_duration_ms_ = max_duration_ms < 0 ? 0 : max_duration_ms;
}

int TuningContext::GetMaxWarmupDurationMs() const {
  static const auto env = c10::utils::get_env("PYTORCH_TUNABLEOP_MAX_WARMUP_DURATION_MS");
  if (env.has_value()) {
    int val = stoi(env.value());
    return val < 0 ? 0 : val;
  }
  return max_warmup_duration_ms_;
}

void TuningContext::SetMaxWarmupIterations(int max_iter) {
  max_warmup_iterations_ = max_iter < 0 ? 0 : max_iter;
}

int TuningContext::GetMaxWarmupIterations() const {
  static const auto env = c10::utils::get_env("PYTORCH_TUNABLEOP_MAX_WARMUP_ITERATIONS");
  if (env.has_value()) {
    int val = stoi(env.value());
    return val < 0 ? 0 : val;
  }
  return max_warmup_iterations_;
}

void TuningContext::EnableICacheFlush(bool value) {
  icache_flush_ = value;
}

bool TuningContext::IsICacheFlushEnabled() const {
  static const auto env = c10::utils::get_env("PYTORCH_TUNABLEOP_ICACHE_FLUSH_ENABLED");
  if (env == "0") {
    return false;
  }
  return icache_flush_;
}

void TuningContext::SetRotatingBufferSize(int size) {
  // Any negative rotating buffer size means l2_cache_size
  // see GetRotatingBufferSize
  //
  // size is set in MB like the environment variable
  constexpr int MB = 1024 * 1024;
  rotating_buffer_size_ = size * MB;
}

int TuningContext::GetRotatingBufferSize() const {
  // If the environment variable is negative or not set, return the L2 cache size.
  // The default rotating_buffer_size is -1, but this member function will
  // return l2_cache size.
  // This member function will always return a zero or a positive integer.
  static const auto env = c10::utils::get_env("PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE");
  int l2_cache_size = at::cuda::getCurrentDeviceProperties()->l2CacheSize;
  if (env.has_value()) {  // env variable is set
    constexpr int MB = 1024 * 1024;
    int val = stoi(env.value());
    return val < 0 ? l2_cache_size : val * MB;  // env var is specified as MB, returned as bytes
  }
  else {  // env variable is not set
    if (rotating_buffer_size_ < 0) {
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
      const auto env = c10::utils::get_env("PYTORCH_TUNABLEOP_FILENAME");
      std::string filename = (!env.has_value()) ? "tunableop_results.csv" : env.value();
      SetFilename(filename, true);
    }
    auto filename = GetFilename();
    if (!filename.empty() && !IsRecordUntunedEnabled()) {
      ReadFile(filename);
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
      found = filename_.rfind('.');
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

namespace {

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

} // anonymous namespace

std::string TuningContext::GetLogFilename() const {
  static const auto env_file = c10::utils::get_env("PYTORCH_TUNABLEOP_VERBOSE_FILENAME");
  static std::string val_file = env_file.has_value() ? env_file.value() : "err";
  return val_file;
}

int TuningContext::GetLogLevel() const {
  static const auto env_verbose = c10::utils::get_env("PYTORCH_TUNABLEOP_VERBOSE");
  static int val_verbose = env_verbose.has_value() ? stoi(env_verbose.value()) : 0;
  return val_verbose;
}

bool TuningContext::GetLogOkay() const {
  return !is_shutting_down_;
}

std::ostream& TuningContext::GetLog() const {
  static auto streamptr = get_stream(GetLogFilename());
  return *streamptr;
}

} // namespace at::cuda::tunable
