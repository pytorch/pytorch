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
#include <ATen/cuda/tunable/StreamTimer.h>
#include <ATen/cuda/Sleep.h>
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
#include <deque>

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

namespace {

/** http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance */

class Stats {
  public:
    Stats() {
      _n = 0UL;
      _mean = 0.0;
      _M2 = 0.0;
      _sum = 0.0;
      _min = 0.0;
      _max = 0.0;
    }

    void sample_value(const double x) {
      double delta = 0;
      _sum = _sum + x;
      if (0UL == _n) {
          _min = x;
          _max = x;
      }
      else {
          _min = _min < x ? _min : x;
          _max = _max > x ? _max : x;
      }
      _n = _n + 1UL;
      delta = x - _mean;
      _mean = _mean + delta/_n;
      _M2 = _M2 + delta * (x - _mean);
    }

    double variance() const {
      return _M2/(_n-1);
    }

    double stddev() const {
      return std::sqrt(variance());
    }

    unsigned long _n;
    double _mean;
    double _M2;
    double _sum;
    double _min;
    double _max;
};

class FixedSizeStack {
  private:
      std::deque<std::string> stack;
      const size_t max_size;

  public:
      FixedSizeStack(size_t size) : max_size(size) {}

      void push(const std::string& value) {
          if (stack.size() >= max_size) {
              stack.pop_front(); // Remove the oldest entry
          }
          stack.push_back(value); // Add new entry
      }

      auto rbegin() { return stack.rbegin(); }
      auto rend() { return stack.rend(); }
};

} // anonymous namespace

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
      auto status = op->Call(params);
      if (status != OK && result != ResultEntry::Default()) {
        auto& mgr = ctx->GetTuningResultsManager();
        auto op_sig = Signature();
        auto params_sig = params->Signature();
        TUNABLE_LOG2(
            "candidate ",
            result,
            " failed for ",
            op_sig,
            '(',
            params_sig,
            "), deleting and using default");
        mgr.Delete(op_sig, params_sig);
        auto* default_op = GetOp(ResultEntry::Default().GetKey());
        TORCH_CHECK(default_op != nullptr);
        return default_op->Call(params);
      }
      return status;
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

    static Stats ProfileStats(Callable<ParamsT> *op, const std::vector<ParamsT*> &param, size_t num_iter, size_t &offset) {
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
      Stats s;
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
      auto top_solns = FixedSizeStack(5);

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
        // Assume normal distribution.
        // Solution with smallest mean + 2*sigma will be a better solution?
        // if ((s._mean + 2*s_stddev) < (min_duration_ms + 2*min_stddev_ms)) {
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
  virtual std::string BLASSignature() const = 0;
};

} // namespace at::cuda::tunable
