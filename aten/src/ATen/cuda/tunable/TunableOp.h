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

#include <ATen/cuda/tunable/Tunable.h>
#include <ATen/cuda/tunable/StreamTimer.h>
#include <ATen/cuda/Sleep.h>
#include <ATen/native/TunableOp.h>
#include <c10/cuda/CUDACachingAllocator.h>
#ifndef USE_ROCM
#include <c10/cuda/CUDAGraphsC10Utils.h>
#endif

#ifndef _WIN32
#include <cxxabi.h>
#endif

#ifndef USE_ROCM
#include <mutex>
#endif
#include <cmath>
#include <deque>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace at::cuda::tunable {

template <typename ParamsT>
class Callable {
  public:
    virtual ~Callable() = default;
    virtual TuningStatus Call(const ParamsT* /*unused*/) {
      return FAIL;
    }
    virtual TuningStatus IsSupported(const ParamsT* params) {
      return Call(params);
    }
};

template <typename ParamsT>
class TunableOp {
  public:
    virtual ~TunableOp() = default;

    TuningStatus operator()(const ParamsT* params) {
      ResultEntry result = ResultEntry::Null();
      TuningContext* ctx = getTuningContext();
      if (ctx->IsTunableOpEnabled()) {
        auto& mgr = ctx->GetTuningResultsManager();
        auto op_sig = Signature();
        auto params_sig = params->Signature();
        auto blas_sig = params->BLASSignature();
        result = mgr.Lookup(op_sig, params_sig);
        // If there is not previous tuning result been found, we do the tuning iff tuning is enabled
        if (result == ResultEntry::Null()) {
          bool should_record_untuned = !ctx->IsTuningEnabled();
          if (ctx->IsTuningEnabled()) {
#ifndef USE_ROCM
            bool is_capturing =
                c10::cuda::currentStreamCaptureStatusMayInitCtx() !=
                c10::cuda::CaptureStatus::None;
            if (!is_capturing) {
              RegisterOpCandidates(params);
              result = FindFastest(params);
              mgr.Add(op_sig, params_sig, result);
            } else {
              should_record_untuned = true;
            }
#else
            result = FindFastest(params);
            mgr.Add(op_sig, params_sig, result);
#endif
          }
          if (should_record_untuned && ctx->IsRecordUntunedEnabled()) {
            // or record the gemm into file
            mgr.RecordUntuned(ctx->GetUntunedFile(), op_sig, params_sig, blas_sig);
          }
        }
      }
      else {
        result = ResultEntry::Default();
      }
      if (result == ResultEntry::Null()) {
        TUNABLE_LOG2("no result, using default");
        result = ResultEntry::Default();
      }
      auto* op = GetOp(result.GetKey());
#ifndef USE_ROCM
      if (op == nullptr && RegisterOpForResult(result, params)) {
        op = GetOp(result.GetKey());
      }
#endif
      if (op == nullptr) {
        TUNABLE_LOG2("missing candidate ", result, ", using default");
        result = ResultEntry::Default();
        op = GetOp(result.GetKey());
      }
      TORCH_CHECK(op != nullptr);
      return op->Call(params);
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
#ifndef USE_ROCM
      std::scoped_lock l{ops_lock_};
#endif
      this->op_names_.emplace_back(name);
      this->ops_.emplace(name, std::move(op));
    }

    bool HasOp(const std::string& name) const {
#ifndef USE_ROCM
      std::scoped_lock l{ops_lock_};
#endif
      return this->ops_.find(name) != this->ops_.end();
    }

    Callable<ParamsT>* GetOp(const std::string& name) const {
#ifndef USE_ROCM
      std::scoped_lock l{ops_lock_};
#endif
      auto it = ops_.find(name);
      return it == ops_.end() ? nullptr : it->second.get();
    }

    std::vector<std::string> OpNames() const {
#ifndef USE_ROCM
      std::scoped_lock l{ops_lock_};
#endif
      return op_names_;
    }

    virtual void RegisterOpCandidates(const ParamsT* /*params*/) {}

    virtual std::vector<std::string> CandidateNames(const ParamsT* /*params*/) const {
      return OpNames();
    }

#ifndef USE_ROCM
    virtual bool RegisterOpForResult(
        const ResultEntry& /*result*/,
        const ParamsT* /*params*/) {
      return false;
    }
#endif

  private:
    static void WarmUp(Callable<ParamsT> *op, const std::vector<ParamsT*> &param, size_t num_iter, size_t &offset) {
      TuningContext* ctx = getTuningContext();
      bool do_flush = ctx->IsICacheFlushEnabled();
      for (size_t i = 0; i < num_iter; i++) {
        if (do_flush) {
          at::cuda::flush_icache();
        }
        TORCH_CHECK(op->Call(param[(i+offset++)%param.size()]) == OK);
      }
    }

    static double ProfileSimple(Callable<ParamsT> *op, const std::vector<ParamsT*> &param, size_t num_iter, size_t &offset) {
      TuningContext* ctx = getTuningContext();
      bool do_flush = ctx->IsICacheFlushEnabled();
      StreamTimerNoSync timer{};

      // Small Mandatory Warmup
      // Reduces outliers
      for (size_t i = 0; i < 2; i++) {
        TORCH_CHECK(op->Call(param[(i+offset++)%param.size()]) == OK);
      }

      timer.Start();
      for (size_t i = 0; i < num_iter; i++) {
        if (do_flush) {
          at::cuda::flush_icache();
        }
        TORCH_CHECK(op->Call(param[(i+offset++)%param.size()]) == OK);
      }
      timer.End();
      return timer.Duration() / num_iter;
    }

    static at::native::tunable::Stats ProfileStats(
        Callable<ParamsT>* op, const std::vector<ParamsT*>& param, size_t num_iter, size_t& offset) {
      TuningContext* ctx = getTuningContext();
      bool do_flush = ctx->IsICacheFlushEnabled();
      std::vector<StreamTimerNoSync> timer(num_iter);

      // Small Mandatory Warmup
      // Reduces outliers
      for (size_t i = 0; i < 2; i++) {
        TORCH_CHECK(op->Call(param[(i+offset++)%param.size()]) == OK);
      }

      for (size_t i = 0; i < num_iter; i++) {
        timer[i].Start();
        TORCH_CHECK(op->Call(param[(i+offset++)%param.size()]) == OK);
        timer[i].End();
        if (do_flush) {
          at::cuda::flush_icache();
        }
      }
      at::native::tunable::Stats s;
      for (size_t i = 0; i < num_iter; i++) {
        s.sample_value(timer[i].Duration());
      }
      return s;
    }

  protected:
    virtual ResultEntry FindFastest(const ParamsT* params) {
      TuningContext* ctx = getTuningContext();
      auto op_sig = Signature();
      auto params_sig = params->Signature();
      auto blas_sig = params->BLASSignature();
      auto candidate_names = CandidateNames(params);
      TUNABLE_LOG2("finding fastest for ", op_sig, '(', params_sig, ')', " out of ", candidate_names.size(), " candidates");
      TORCH_CHECK(!candidate_names.empty());
      ParamsT* reference_params = nullptr;

      // numeric check option is controlled by non-static env var, so check it once per tuned operator
      bool do_numerics_check = ctx->IsNumericsCheckEnabled();

      // calculate a reference answer for numerical check
      if (do_numerics_check) {
        reference_params = params->DeepCopy(false);
        auto* default_op = GetOp(ResultEntry::Default().GetKey());
        TORCH_CHECK(default_op != nullptr);
        TORCH_CHECK(default_op->Call(reference_params) == OK);
      }

      // need copies of params to reuse
      // make as many copies as will fill the requested rotating buffer size, if requested
      // rotating_size guaranteed to be >= 0 even though GetRotatingBufferSize() returns int
      size_t rotating_size = ctx->GetRotatingBufferSize();
      bool use_buffer_rotation = (rotating_size > 0);
      size_t param_size = params->GetSize(use_buffer_rotation);
      size_t param_count = (rotating_size / param_size) + 1;
      constexpr size_t MB = 1024ull*1024;
      if (use_buffer_rotation) {
        TUNABLE_LOG2("Rotating buffer ", rotating_size/MB, " MiB. ",
            "Needed Size: ", param_size/MB, " MiB. ",
            "Needed number of param copies: ", param_count);
      }
      TORCH_CHECK(param_count > 0);

      std::vector<ParamsT*> reusable_params(param_count);
      for (size_t i = 0; i < param_count; i++) {
        reusable_params[i] = params->DeepCopy(use_buffer_rotation);
      }

      // for rotating buffer
      size_t offset = 0;
      at::native::tunable::TuningPolicy policy;
      policy.max_tuning_duration_ms = ctx->GetMaxTuningDurationMs();
      policy.max_tuning_iterations = ctx->GetMaxTuningIterations();
      policy.max_warmup_duration_ms = ctx->GetMaxWarmupDurationMs();
      policy.max_warmup_iterations = ctx->GetMaxWarmupIterations();
      const auto tuning_result = at::native::tunable::findFastest(
          candidate_names.size(),
          0,
          policy,
          [&](size_t i) {
            auto* candidate = GetOp(candidate_names[i]);
            TORCH_CHECK(candidate != nullptr);
            return candidate->Call(reusable_params[0]) == OK;
          },
          [&](size_t i, int iterations) {
            auto* candidate = GetOp(candidate_names[i]);
            TORCH_CHECK(candidate != nullptr);
            return ProfileStats(candidate, reusable_params, iterations, offset);
          },
          [&](size_t i) {
            if (!do_numerics_check) {
              return true;
            }
            ParamsT* numerical_params = params->DeepCopy(false);
            auto* candidate = GetOp(candidate_names[i]);
            TORCH_CHECK(candidate != nullptr);
            auto status = candidate->Call(numerical_params);
            if (status == OK) {
              status = reference_params->NumericalCheck(numerical_params);
            }
            numerical_params->Delete();
            return status == OK;
          },
          [&](size_t i, int iterations) {
            auto* candidate = GetOp(candidate_names[i]);
            TORCH_CHECK(candidate != nullptr);
            WarmUp(candidate, reusable_params, iterations, offset);
          });

      for (size_t i = 0; i < reusable_params.size(); i++) {
        reusable_params[i]->Delete();
      }
      if (reference_params) {
        reference_params->Delete();
      }

      double fastest_ms = std::numeric_limits<double>::infinity();
      std::deque<std::string> top_solns;
      for (size_t i = 0; i < tuning_result.candidates.size(); ++i) {
        const auto& result = tuning_result.candidates[i];
        switch (result.status) {
          case at::native::tunable::CandidateStatus::Unsupported:
            TUNABLE_LOG3("├──unsupported id=", i, ", ", op_sig, '(', params_sig, ") ", candidate_names[i]);
            break;
          case at::native::tunable::CandidateStatus::PrunedFirst:
            TUNABLE_LOG3("├──skip slow instance id=", i, ", ", op_sig, '(', params_sig, ") ", candidate_names[i]);
            break;
          case at::native::tunable::CandidateStatus::PrunedSecond:
            TUNABLE_LOG3("├──2nd skip slow instance id=", i, ", ", op_sig, '(', params_sig, ") ", candidate_names[i]);
            break;
          case at::native::tunable::CandidateStatus::NumericalFailure:
            TUNABLE_LOG3("├──numerics check failed for id=", i, ", ", op_sig, '(', params_sig, ") ", candidate_names[i]);
            break;
          case at::native::tunable::CandidateStatus::Profiled:
            TUNABLE_LOG3(result.stats._mean < fastest_ms ? "├──found better instance id="
                                                         : "├──found slower instance id=",
                         i,
                         ". ",
                         result.stats._mean,
                         "ms. ",
                         candidate_names[i],
                         " min ", result.stats._min,
                         " max ", result.stats._max,
                         " mean ", result.stats._mean,
                         " std ", result.stats.stddev());
            if (result.stats._mean < fastest_ms) {
              fastest_ms = result.stats._mean;
              if (top_solns.size() == 5) {
                top_solns.pop_front();
              }
              top_solns.push_back(std::to_string(result.stats._mean) + " " + candidate_names[i]);
            }
            break;
        }
      }
      const auto id_name = std::isfinite(tuning_result.time_ms) ? candidate_names[tuning_result.candidate_index]
                                                                : ResultEntry::Default().GetKey();
      TUNABLE_LOG2("└──found fastest for ", op_sig, '(', params_sig, ") ", id_name);
      TUNABLE_LOG2("└──top five solutions for ", op_sig, '(', params_sig, ") ");
      for (auto it = top_solns.rbegin(); it != top_solns.rend(); ++it) {
        TUNABLE_LOG2("   ", *it);
      }
      return ResultEntry(id_name, tuning_result.time_ms, blas_sig);
    }

  private:
    std::string CreateSignature() {
#ifndef _WIN32
      const auto* name = typeid(*this).name();
      // NOLINTNEXTLINE(*array*)
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
#ifndef USE_ROCM
    mutable std::mutex ops_lock_;
#endif
};

struct OpParams {
  OpParams() = default;
  OpParams(const OpParams&) = default;
  virtual ~OpParams() = default;
  virtual std::string Signature() const = 0;
  virtual std::string BLASSignature() const = 0;
};

} // namespace at::cuda::tunable
