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

    bool CanDispatchResult(
        const ResultEntry& result,
        const ParamsT* params) {
      if (HasOp(result.GetKey())) {
        return true;
      }
#ifndef USE_ROCM
      return RegisterOpForResult(result, params) && HasOp(result.GetKey());
#else
      return false;
#endif
    }

    TuningStatus operator()(const ParamsT* params) {
      // Dynamic TunableOp dispatch matrix:
      //
      // Tuning enabled (compile-time autotune)
      //   1. no dynamic dim + concrete miss -> tune; persist concrete
      //   2. has dynamic dim + concrete miss -> tune; persist concrete + wildcard
      //   3. no dynamic dim + concrete hit  -> nothing
      //   4. has dynamic dim + concrete hit -> if wildcard not persisted, persist wildcard
      //
      // Tuning disabled (runtime)
      //   At runtime we cannot know which dim is dynamic (the AOTI
      //   cpp_wrapper does not propagate that information into the .so),
      //   so:
      //   1. concrete hit                       -> dispatch via concrete
      //   2. concrete miss + wildcard match     -> dispatch via wildcard
      //                                            (LookupWildcardFallback
      //                                            scans persisted wildcards
      //                                            and returns the first
      //                                            whose pattern matches the
      //                                            concrete signature)
      //   3. concrete miss + no wildcard match  -> result stays Null;
      //                                            generic TunableOp falls
      //                                            through to Default, while
      //                                            BLAS callers pre-check and
      //                                            skip to the disabled aten
      //                                            path instead
      ResultEntry result = ResultEntry::Null();
      TuningContext* ctx = getTuningContext();
      if (ctx->IsTunableOpEnabled()) {
        auto& mgr = ctx->GetTuningResultsManager();
        auto op_sig = Signature();
        auto concrete_params_sig = params->Signature();
        // Only this `any()` bit-check is on the cache-hit fast path. The
        // full DynamicSignature() (which performs a fmt::format with
        // several string allocations) is computed lazily, only inside the
        // miss / case-4 branches that actually need it. At runtime
        // dynamic_dims_mask is always empty (no TunableDynamicDimsGuard
        // emitted by AOTI cpp_wrapper) so this stays false on the
        // hot path; only compile-time tuning (when Python pushes a mask
        // via torch.cuda.tunable.dynamic_dims_mask) sets it.
        const bool has_dynamic_dim = params->dynamic_dims_mask.any();

        result = mgr.Lookup(op_sig, concrete_params_sig);
        const bool concrete_hit = (result != ResultEntry::Null());

        if (ctx->IsTuningEnabled()) {
          if (!concrete_hit) {
            // Cases 1 & 2: concrete miss with tuning -> tune + persist.
            bool can_tune = true;
#ifndef USE_ROCM
            can_tune =
                c10::cuda::currentStreamCaptureStatusMayInitCtx() ==
                c10::cuda::CaptureStatus::None;
            if (can_tune) {
              RegisterOpCandidates(params);
            }
#endif
            if (can_tune) {
              result = FindFastest(params);
              mgr.Add(op_sig, concrete_params_sig, result);
              if (has_dynamic_dim) {
                // Case 2: also seed wildcard key with the same tuned kernel.
                // mgr.Add is a no-op when the key already exists, so the
                // first-tuned-wins semantics required by the dynamic-dims
                // contract are preserved.
                mgr.Add(op_sig, params->DynamicSignature(), result);
              }
            } else if (ctx->IsRecordUntunedEnabled()) {
              mgr.RecordUntuned(
                  ctx->GetUntunedFile(),
                  op_sig,
                  concrete_params_sig,
                  params->BLASSignature());
            }
          }
          else if (has_dynamic_dim) {
            // Case 4: concrete hit + dynamic -> ensure wildcard is persisted
            // so subsequent shapes that differ only in the dynamic dims can
            // reuse this kernel without re-tuning.
            auto dynamic_params_sig = params->DynamicSignature();
            if (mgr.Lookup(op_sig, dynamic_params_sig) == ResultEntry::Null()) {
              mgr.Add(op_sig, dynamic_params_sig, result);
            }
          }
          // Case 3: concrete hit + no dynamic -> nothing to do.
        }
        else {
          // Tuning disabled (runtime). We cannot rely on any caller-set
          // dynamic mask here -- AOTI cpp_wrapper does not emit a
          // TunableDynamicDimsGuard so GetCurrentDynamicDimsMask() is
          // empty. On concrete miss, scan all persisted wildcard
          // entries and return any whose token-pattern matches the
          // concrete signature. If none match, result stays Null and
          // the caller (e.g. launchTunableGemmAndBias) falls back to
          // the non-tunable aten path.
          if (!concrete_hit) {
            result = mgr.LookupWildcardFallback(op_sig, concrete_params_sig);
            if (result == ResultEntry::Null() && ctx->IsRecordUntunedEnabled()) {
              auto blas_sig = params->BLASSignature();
              mgr.RecordUntuned(ctx->GetUntunedFile(), op_sig, concrete_params_sig, blas_sig);
            }
          }
          // Case 1: concrete hit -> use it.
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
      if (op == nullptr && CanDispatchResult(result, params)) {
        op = GetOp(result.GetKey());
      }
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

    static at::native::tunable::Stats ProfileStats(Callable<ParamsT> *op, const std::vector<ParamsT*> &param, size_t num_iter, size_t &offset) {
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
      auto min_duration_ms = std::numeric_limits<double>::infinity();
      std::string id_name = "Default";
      ParamsT* reference_params = nullptr;
      auto top_solns = at::native::tunable::FixedSizeStack(5);

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

      for (size_t i = 0; i < candidate_names.size(); i++) {
        auto* candidate = GetOp(candidate_names[i]); // borrow pointer
        TORCH_CHECK(candidate != nullptr);

        auto status = candidate->Call(reusable_params[0]);
        if (status != OK) {
          TUNABLE_LOG3("├──unsupported id=", i, ", ", op_sig, '(', params_sig, ") ", candidate_names[i]);
          continue;
        }

        // collect a small profile
        int approx_num_iter = 3;
        auto s = ProfileStats(candidate, reusable_params, approx_num_iter, offset);
        double approx_duration = s._mean;
        // bail if too slow
        if (approx_duration > 1.5 * min_duration_ms) {
          TUNABLE_LOG3("├──skip slow instance id=", i, ", ", op_sig, '(', params_sig, ") ", candidate_names[i]);
          continue;
        }

        // 2nd phase skip, more aggressive
        approx_num_iter = 10;
        s = ProfileStats(candidate, reusable_params, approx_num_iter, offset);
        approx_duration = s._mean;
        // bail if too slow
        if (approx_duration > 1.15 * min_duration_ms) {
          TUNABLE_LOG3("├──2nd skip slow instance id=", i, ", ", op_sig, '(', params_sig, ") ", candidate_names[i]);
          continue;
        }

        if (do_numerics_check) {
          ParamsT* numerical_params = params->DeepCopy(false);
          auto status = candidate->Call(numerical_params);
          if (status != OK) {
            numerical_params->Delete();
            TUNABLE_LOG3("├──unsupported id=", i, ", ", op_sig, '(', params_sig, ") ", candidate_names[i]);
            continue;
          }
          status = reference_params->NumericalCheck(numerical_params);
          numerical_params->Delete();
          if (status != OK) {
            TUNABLE_LOG3("├──numerics check failed for id=", i, ", ", op_sig, '(', params_sig, ") ", candidate_names[i]);
            continue;
          }
        }

        // for warmup does user set max duration, max iters, or both?
        // warmup is skipped by default, i.e. warmup_iter = 0
        // warmup will be set to the non-zero value of max_warmup_duration
        // or max_warmup_iter
        // if both are non-zero, we take the smaller of the two.
        double max_warmup_duration = ctx->GetMaxWarmupDurationMs();
        int max_warmup_iter = ctx->GetMaxWarmupIterations();
        int warmup_iter = 0; // default
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
        // tuning must run at least 1 iteration
        tuning_iter = std::max(1, tuning_iter);

        // do the full warmup followed by tuning
        double warmup_ms = warmup_iter * approx_duration;
        double tuning_ms = tuning_iter * approx_duration;
        TUNABLE_LOG3("├──tuning using "
            "warmup iters ", warmup_iter, " [", warmup_ms, " ms] "
            "and tuning iters ", tuning_iter, " [", tuning_ms, " ms] ",
            "instance id=", i, ", ", op_sig, "(", params_sig, ") ", candidate_names[i]);
        TUNABLE_LOG3("├──offset at ", offset);
        WarmUp(candidate, reusable_params, warmup_iter, offset);
        s = ProfileStats(candidate, reusable_params, tuning_iter, offset);
        auto s_stddev = s.stddev();
        if (s._mean < min_duration_ms) {
          TUNABLE_LOG3("├──found better instance id=", i, ". " , s._mean, "ms. ", candidate_names[i],
                " min ", s._min,
                " max ", s._max,
                " mean ", s._mean,
                " std ", s_stddev);
          min_duration_ms = s._mean;
          id_name = candidate_names[i];
          std::string current_soln = std::to_string(s._mean) + " " + candidate_names[i];
          top_solns.push(current_soln);
        }
        else {
          TUNABLE_LOG3("├──found slower instance id=", i, ". " , s._mean, "ms. ", candidate_names[i],
                " min ", s._min,
                " max ", s._max,
                " mean ", s._mean,
                " std ", s_stddev);
        }
      }

      for (size_t i = 0; i < reusable_params.size(); i++) {
        reusable_params[i]->Delete();
      }
      if (reference_params) {
        reference_params->Delete();
      }

      TUNABLE_LOG2("└──found fastest for ", op_sig, '(', params_sig, ") ", id_name);
      TUNABLE_LOG2("└──top five solutions for ", op_sig, '(', params_sig, ") ");
      for (auto it = top_solns.rbegin(); it != top_solns.rend(); ++it) {
        TUNABLE_LOG2("   ", *it);
      }
      return ResultEntry(id_name, min_duration_ms, blas_sig);
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
  virtual std::string DynamicSignature() const {
    return Signature();
  }
  virtual std::string BLASSignature() const = 0;

  // Per-instance mask describing which logical GEMM dims are dynamic for
  // this particular op invocation. The producer (Blas.cpp / CUDABlas.cpp /
  // ScaledBlas.cpp) reads at::cuda::tunable::GetCurrentDynamicDimsMask()
  // once and stamps the result here before calling the TunableOp; the
  // Gemm*Params subclasses' DynamicSignature() implementations then read
  // this field instead of the previously-global TuningContext setting.
  //
  // Default-constructed (all-zero) means "no dim is dynamic", which yields
  // a DynamicSignature() byte-identical to Signature() and preserves the
  // legacy concrete-only behavior for callers that don't push a guard.
  DynamicDimsMask dynamic_dims_mask{};

  bool IsDynamicM() const { return dynamic_dims_mask.m(); }
  bool IsDynamicN() const { return dynamic_dims_mask.n(); }
  bool IsDynamicK() const { return dynamic_dims_mask.k(); }
  bool IsDynamicBatch() const { return dynamic_dims_mask.batch(); }
};

} // namespace at::cuda::tunable
