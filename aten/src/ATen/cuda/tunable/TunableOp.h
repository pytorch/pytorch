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

#ifndef _WIN32
#include <cxxabi.h>
#endif

#include <chrono>
#include <thread>
#include <type_traits>

namespace at::cuda::tunable {

template <typename T, typename Arg, typename E = void>
struct HasIsSupportedMethod {
  constexpr static bool value = false;
};

template <typename T, typename Arg>
struct HasIsSupportedMethod<
    T, Arg, std::enable_if_t<std::is_same_v<decltype(std::declval<T>().IsSupported(std::declval<Arg>())), TuningStatus>>> {
  constexpr static bool value = true;
};

// A type erased Callable wrapper. We could have used std::function<TuningStatus<const ParamsT*>> here. However, std::function
// requires the callable object to be CopyConstructible and CopyAssignable. This is not suitable for move only functor
// or move captured lambda. So we create a simple wrapper for our purpose here.
template <typename ParamsT>
class Callable {
 public:
  template <typename T, typename = std::enable_if_t<
                            !std::is_same_v<Callable<ParamsT>, std::remove_cv_t<std::remove_reference_t<T>>>,
                            void>>
  Callable(T&& c) : callable_{std::make_unique<CallableImpl<T>>(std::forward<T>(c))} {}  // NOLINT(google-explicit-constructor)
  Callable(Callable&&) = default;
  TuningStatus operator()(const ParamsT* param) { return (*callable_)(param); }
  TuningStatus IsSupported(const ParamsT* param) { return (*callable_).IsSupported(param); }

 private:
  struct ICallable {
    virtual ~ICallable() = default;
    virtual TuningStatus operator()(const ParamsT*) = 0;
    virtual TuningStatus IsSupported(const ParamsT*) = 0;
  };

  template <typename T>
  struct CallableImpl : ICallable {
    explicit CallableImpl(T&& c) : c_{std::move(c)} {}
    CallableImpl(CallableImpl&&) = default;
    TuningStatus operator()(const ParamsT* param) override { return c_(param); }
    TuningStatus IsSupported(const ParamsT* param) override {
      if constexpr (HasIsSupportedMethod<T, const ParamsT*>::value) {
        return c_.IsSupported(param);
      } else {
        return c_(param);
      }
    }

   private:
    T c_;
  };

  std::unique_ptr<ICallable> callable_;
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

        // Usage is enabled, then we are free to use previous tuning result.
        result = mgr.Lookup(op_sig, params_sig);
        if (result > static_cast<int>(ops_.size())) {
          TUNABLE_LOG("Invalid TunableOp kernel result for ", op_sig, ", result:", result, ", registered op:", ops_.size());
          mgr.Delete(op_sig, params_sig);
          result = ResultEntry::Null();
        }

        // If there is not previous tuning result been found, we do the tuning iff tuning is enabled
        if (result < 0 && ctx->IsTuningEnabled()) {
          auto maybe_proxy_params = PreTuning(params);
          auto result = FindFastest(maybe_proxy_params);
          PostTuning(maybe_proxy_params);
          mgr.Add(op_sig, params_sig, result);
        }
      }
      if (result < 0) {
        TUNABLE_LOG("result < 0, using default_id_; result=", result);
      }
      return (ops_[result < 0 ? default_id_ : result](params));
    }

    // We might want to do some tricks to the `params`, e.g., some op will use a buffer for input and output at the same
    // time, so it will do inplace update to it. If we blindly tune over the `params`, there will be accumulated update
    // to that buffer during FindFastest, which is an undesired side effect. In this case, we must prepare a new (proxy)
    // params struct for the tuning to avoid this side effect.
    virtual const ParamsT* PreTuning(const ParamsT* params) {
      return params;
    }

    virtual void PostTuning(const ParamsT* /*params*/) {
      // Do nothing if we are not playing around with params
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
    // set the default op to be used in non-tuning scenario
    void SetDefaultId(int id) {
      TORCH_CHECK(id < static_cast<int>(ops_.size()), "TunableOp id out of bound");
      default_id_ = id;
    }

    void RegisterOp(std::string&& name, Callable<ParamsT>&& op) {
      this->ops_.emplace_back(std::move(op));
      this->op_names_.emplace_back(std::move(name));
    }

    int NumberOfOps() {
      return this->ops_.size();
    }

  private:
    static void WarmUp(Callable<ParamsT>& op, const ParamsT* param, int num_iter) {
      for (int i = 0; i < num_iter; i++) {
        TORCH_CHECK(op(param) == OK);
      }
    }

    static double Profile(Callable<ParamsT>& op, const ParamsT* param, int num_iter) {
      TimerT timer{};
      timer.Start();
      for (int i = 0; i < num_iter; i++) {
        TORCH_CHECK(op(param) == OK);
      }
      timer.End();
      return timer.Duration() / num_iter;
    }

    static TuningStatus IsSupported(Callable<ParamsT>& op, const ParamsT* params) {
      TuningStatus status = op.IsSupported(params);
      return status;
    }

  protected:
    virtual ResultEntry FindFastest(const ParamsT* params) {
      return FindFastestImpl(params, ops_);
    }

    ResultEntry FindFastestImpl(const ParamsT* params, const std::vector<Callable<ParamsT>>& candidates) {
      TuningContext* ctx = getTuningContext();
      auto op_sig = Signature();
      auto params_sig = params->Signature();
      TUNABLE_LOG("finding fastest for ", op_sig, '(', params_sig, ')', " out of ", candidates.size(), " candidates");
      auto min_duration_ms = std::numeric_limits<double>::infinity();
      int id = -1;
      std::string id_name = "";

      for (size_t i = 0; i < candidates.size(); i++) {
        auto& candidate = const_cast<Callable<ParamsT>&>(candidates[i]);
        auto status = IsSupported(candidate, params);
        if (status != OK) {
          TUNABLE_LOG("├──unsupported id=", i, ", ", op_sig, '(', params_sig, ") ", op_names_[i]);
          continue;
        }

        // do 1 initial tuning in case there is overhead
        WarmUp(candidate, params, 1);
        // collect a small profile
        constexpr const int approx_num_iter = 3;
        auto approx_duration = Profile(candidate, params, approx_num_iter);
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
        WarmUp(candidate, params, warmup_iter);
        auto duration_ms = Profile(candidate, params, tuning_iter);
        if (duration_ms < min_duration_ms) {
          TUNABLE_LOG("├──found better instance, ",
              "new best id=", i, ", old id=", id, ". " , duration_ms, "ms. ", op_names_[i]);
          min_duration_ms = duration_ms;
          id = static_cast<int>(i);
          id_name = op_names_[i];
        }
      }
      TORCH_CHECK(id >= 0, "Could not find viable op");
      TUNABLE_LOG("└──found fastest with id=", id, " for ", op_sig, '(', params_sig, ") ", id_name);
      //std::this_thread::sleep_for(std::chrono::milliseconds(50));
      return ResultEntry(id, min_duration_ms, id_name);
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

    // the default impl to use when tuning is disabled
    int default_id_{0};

    std::vector<Callable<ParamsT>> ops_;
    std::vector<std::string> op_names_;
};

struct OpParams {
  OpParams() {}
  virtual ~OpParams() = default;
  virtual std::string Signature() const = 0;
};

} // namespace at::cuda::tunable
