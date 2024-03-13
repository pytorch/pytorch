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

#include <ATen/cuda/tunable/Tunable.h>
#include <c10/cuda/CUDACachingAllocator.h>

#ifndef _WIN32
#include <cxxabi.h>
#endif

#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace at::cuda::tunable {

template <typename ParamsT>
class Callable {
  public:
    Callable() = default;
    Callable(Callable&&) = default;
    virtual ~Callable() = default;
    virtual TuningStatus Call(const ParamsT*) {
      return FAIL;
    }
    virtual TuningStatus IsSupported(const ParamsT* params) {
      return Call(params);
    }
};

template <typename ParamsT, typename TimerT>
class TunableOp {
  public:
    TunableOp() = default;
    TunableOp(TunableOp&&) = default;
    virtual ~TunableOp() = default;

    TuningStatus operator()(const ParamsT* params) {
      ResultEntry result = ResultEntry::Null();
      TuningContext* ctx = getTuningContext();
      if (ctx->IsTunableOpEnabled()) {
        auto& mgr = ctx->GetTuningResultsManager();
        auto op_sig = Signature();
        auto params_sig = params->Signature();
        result = mgr.Lookup(op_sig, params_sig);
        // If there is not previous tuning result been found, we do the tuning iff tuning is enabled
        if (result == ResultEntry::Null() && ctx->IsTuningEnabled()) {
          result = FindFastest(params);
          mgr.Add(op_sig, params_sig, result);
        }
      }
      else {
        result = ResultEntry::Default();
      }
      if (result == ResultEntry::Null()) {
        TUNABLE_LOG("no result, using default");
        result = ResultEntry::Default();
      }
      auto iter = ops_.find(result);
      TORCH_CHECK(iter != ops_.end());
      return iter->second->Call(params);
    }

    virtual std::string Signature() {
      // According to C++17 standard https://wg21.link/n4659 section 15.7.4
      // > if the operand of typeid refers to the
      // > object under construction or destruction, typeid yields the std::type_info object representing the constructor
      // > or destructor’s class.
      // So delay the op signature generation.
      c10::call_once(signature_init_once_, [this]() { signature_ = CreateSignature(); });
      return signature_;
    }

  protected:
    void RegisterOp(const std::string& name, std::unique_ptr<Callable<ParamsT>> op) {
      this->op_names_.emplace_back(name);
      this->ops_.emplace(name, std::move(op));
    }

  private:
    static void WarmUp(Callable<ParamsT> *op, ParamsT* param, size_t num_iter) {
      for (size_t i = 0; i < num_iter; i++) {
        TORCH_CHECK(op->Call(param) == OK);
      }
    }

    static double Profile(Callable<ParamsT> *op, ParamsT* param, size_t num_iter) {
      TimerT timer{};
      timer.Start();
      for (size_t i = 0; i < num_iter; i++) {
        TORCH_CHECK(op->Call(param) == OK);
      }
      timer.End();
      return timer.Duration() / num_iter;
    }

  protected:
    bool IsNumericsCheckEnabled() {
      static const char *env = getenv("PYTORCH_TUNABLEOP_NUMERICAL_CHECK");
      if (env != nullptr && strcmp(env, "0") == 0) {
        return false;
      }
      return true;
    }

    virtual ResultEntry FindFastest(const ParamsT* params) {
      TuningContext* ctx = getTuningContext();
      auto op_sig = Signature();
      auto params_sig = params->Signature();
      TUNABLE_LOG("finding fastest for ", op_sig, '(', params_sig, ')', " out of ", op_names_.size(), " candidates");
      auto min_duration_ms = std::numeric_limits<double>::infinity();
      std::string id_name = "Default";

      // calcaulte a reference answer for numerical check
      ParamsT* reference_params = params->DeepCopy();
      TORCH_CHECK(ops_[ResultEntry::Default()]->Call(reference_params) == OK);

      // need a copy of params to reuse
      ParamsT* reusable_params = params->DeepCopy();

      for (size_t i = 0; i < op_names_.size(); i++) {
        auto* candidate = ops_[op_names_[i]].get(); // borrow pointer
        auto status = candidate->Call(reusable_params);
        if (status != OK) {
          TUNABLE_LOG("├──unsupported id=", i, ", ", op_sig, '(', params_sig, ") ", op_names_[i]);
          continue;
        }

        if (IsNumericsCheckEnabled()) {
          ParamsT* numerical_params = params->DeepCopy();
          WarmUp(candidate, numerical_params, 1);
          status = reference_params->NumericalCheck(numerical_params);
          numerical_params->Delete();
          if (status != OK) {
            TUNABLE_LOG("├──numerics check failed for id=", i, ", ", op_sig, '(', params_sig, ") ", op_names_[i]);
            continue;
          }
        }

        // collect a small profile
        constexpr const int approx_num_iter = 3;
        auto approx_duration = Profile(candidate, reusable_params, approx_num_iter);
        // bail if too slow
        if (approx_duration > 2 * min_duration_ms) {
          TUNABLE_LOG("├──skip slow instance id=", i, ", ", op_sig, '(', params_sig, ") ", op_names_[i]);
          continue;
        }

        // for warmup does user set max duration, max iters, or both?
        double max_warmup_duration = ctx->GetMaxWarmupDurationMs();
        int max_warmup_iter = ctx->GetMaxWarmupIterations();
        int warmup_iter = 1; // default
        if (max_warmup_duration > 0) {
          int duration_iters = max_warmup_duration / approx_duration;
          if (max_warmup_iter > 0) {
            warmup_iter = std::min(max_warmup_iter, duration_iters);
          }
          else {
            warmup_iter = duration_iters;
          }
        }
        else if (max_warmup_iter > 0) {
          warmup_iter = max_warmup_iter;
        }

        // for tuning does user set max duration, max iters, or both?
        double max_tuning_duration = ctx->GetMaxTuningDurationMs();
        int max_tuning_iter = ctx->GetMaxTuningIterations();
        int tuning_iter = 100; // default
        if (max_tuning_duration > 0) {
          int duration_iters = max_tuning_duration / approx_duration;
          if (max_tuning_iter > 0) {
            tuning_iter = std::min(max_tuning_iter, duration_iters);
          }
          else {
            tuning_iter = duration_iters;
          }
        }
        else if (max_tuning_iter > 0) {
          tuning_iter = max_tuning_iter;
        }

        // do the full warmup followed by tuning
        double warmup_ms = warmup_iter * approx_duration;
        double tuning_ms = tuning_iter * approx_duration;
        TUNABLE_LOG("├──tuning using "
            "warmup iters ", warmup_iter, " [", warmup_ms, " ms] "
            "and tuning iters ", tuning_iter, " [", tuning_ms, " ms] ",
            "instance id=", i, ", ", op_sig, "(", params_sig, ") ", op_names_[i]);
        WarmUp(candidate, reusable_params, warmup_iter);
        auto duration_ms = Profile(candidate, reusable_params, tuning_iter);
        if (duration_ms < min_duration_ms) {
          TUNABLE_LOG("├──found better instance id=", i, ". " , duration_ms, "ms. ", op_names_[i]);
          min_duration_ms = duration_ms;
          id_name = op_names_[i];
        }
      }

      reusable_params->Delete();
      reference_params->Delete();

      TUNABLE_LOG("└──found fastest for ", op_sig, '(', params_sig, ") ", id_name);
      return ResultEntry(id_name, min_duration_ms);
    }

  private:
    std::string CreateSignature() {
#ifndef _WIN32
      const auto* name = typeid(*this).name();
      char buf[256];
      size_t buf_len = 256;
      abi::__cxa_demangle(name, buf, &buf_len, nullptr);
      buf[255] = '\0';
      return buf;
#else
      return typeid(*this).name();
#endif
    }

    mutable c10::once_flag signature_init_once_;
    std::string signature_;

    std::unordered_map<std::string, std::unique_ptr<Callable<ParamsT>>> ops_;
    std::vector<std::string> op_names_;
};

struct OpParams {
  OpParams() {}
  virtual ~OpParams() = default;
  virtual std::string Signature() const = 0;
};

} // namespace at::cuda::tunable
