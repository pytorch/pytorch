#pragma once

#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename Context>
void adam_update(
    int N,
    const float* g,
    const float* m,
    const float* v,
    float* ng,
    float* nm,
    float* nv,
    float beta1,
    float beta2,
    float eps_hat,
    float correction,
    const float* lr,
    Context* /*context*/) {
  for (auto i = 0; i < N; ++i) {
    float gi = g[i];
    float mi = nm[i] = m[i] * beta1 + gi * (1 - beta1);
    float vi = nv[i] = v[i] * beta2 + gi * gi * (1 - beta2);
    ng[i] = lr[0] * correction * mi / (std::sqrt(vi) + eps_hat);
  }
}

template <typename Context>
void adam_compute(
    int N,
    const float* w,
    const float* g,
    const float* m,
    const float* v,
    float* nw,
    float* nm,
    float* nv,
    float beta1,
    float beta2,
    float eps_hat,
    float correction,
    const float* lr,
    Context* /*context*/) {
  for (auto i = 0; i < N; ++i) {
    float gi = g[i];
    float mi = nm[i] = m[i] * beta1 + gi * (1 - beta1);
    float vi = nv[i] = v[i] * beta2 + gi * gi * (1 - beta2);
    float ng = lr[0] * correction * mi / (std::sqrt(vi) + eps_hat);
    nw[i] = w[i] + ng;
  }
}

template <typename T, class Context>
class AdamOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  AdamOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        beta1_(OperatorBase::GetSingleArgument<float>("beta1", 0.9f)),
        beta2_(OperatorBase::GetSingleArgument<float>("beta2", 0.999f)),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 1e-5f)) {}
  bool RunOnDevice() override {
    // Iter live on the CPU
    CAFFE_ENFORCE(OperatorBase::InputIsType<TensorCPU>(ITER));
    CAFFE_ENFORCE(Input(LR).size() == 1);
    CAFFE_ENFORCE(Input(GRAD).size() == Input(PARAM).size());
    CAFFE_ENFORCE(Input(GRAD).size() == Input(MOMENT_1).size());
    CAFFE_ENFORCE(Input(GRAD).size() == Input(MOMENT_2).size());
    Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
    Output(OUTPUT_MOMENT_1)->ResizeLike(Input(MOMENT_1));
    Output(OUTPUT_MOMENT_2)->ResizeLike(Input(MOMENT_2));

    const auto iter =
        OperatorBase::Input<TensorCPU>(ITER).template data<int64_t>()[0];

    const auto t = iter + 1;
    const auto correction =
        std::sqrt(T(1.) - std::pow(beta2_, t)) / (T(1.) - std::pow(beta1_, t));
    adam_compute<Context>(
        Input(GRAD).size(),
        Input(PARAM).template data<T>(),
        Input(GRAD).template data<T>(),
        Input(MOMENT_1).template data<T>(),
        Input(MOMENT_2).template data<T>(),
        Output(OUTPUT_PARAM)->template mutable_data<T>(),
        Output(OUTPUT_MOMENT_1)->template mutable_data<T>(),
        Output(OUTPUT_MOMENT_2)->template mutable_data<T>(),
        beta1_,
        beta2_,
        epsilon_,
        correction,
        Input(LR).template data<T>(),
        &context_);
    return true;
  }

 protected:
  T beta1_{0.9};
  T beta2_{0.999};
  T epsilon_{1e-8};
  INPUT_TAGS(PARAM, MOMENT_1, MOMENT_2, GRAD, LR, ITER);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1, OUTPUT_MOMENT_2);
};

template <typename T, class Context>
class SparseAdamOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SparseAdamOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        beta1_(OperatorBase::GetSingleArgument<float>("beta1", 0.9f)),
        beta2_(OperatorBase::GetSingleArgument<float>("beta2", 0.999f)),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 1e-5f)) {}

  bool RunOnDevice() override {
    // Enforce shapes
    CAFFE_ENFORCE_EQ(Input(PARAM).size(), Input(MOMENT_1).size());
    CAFFE_ENFORCE_EQ(Input(PARAM).size(), Input(MOMENT_2).size());
    CAFFE_ENFORCE_EQ(
        Input(PARAM).size_from_dim(1),
        Input(GRAD).size_from_dim(Input(INDICES).ndim()));
    CAFFE_ENFORCE_EQ(Input(LR).size(), 1);

    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename SIndex>
  bool DoRunWithType() {
    const auto* lr = Input(LR).template data<T>();
    const auto iter =
        OperatorBase::Input<TensorCPU>(ITER).template data<int64_t>()[0];

    const auto t = iter + 1;
    const auto correction =
        std::sqrt(T(1.) - std::pow(beta2_, t)) / (T(1.) - std::pow(beta1_, t));

    auto block_size = Input(PARAM).size() / Input(PARAM).dim(0);
    auto n = Input(GRAD).size() / block_size;

    const auto* paramIn = Input(PARAM).template data<T>();
    const auto* indices = Input(INDICES).template data<SIndex>();
    const auto* gradIn = Input(GRAD).template data<T>();
    const auto* moment1In = Input(MOMENT_1).template data<T>();
    const auto* moment2In = Input(MOMENT_2).template data<T>();
    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<T>();
    auto* moment1Out = Output(OUTPUT_MOMENT_1)->template mutable_data<T>();
    auto* moment2Out = Output(OUTPUT_MOMENT_2)->template mutable_data<T>();

    for (auto i = 0; i < n; ++i) {
      auto idx = indices[i];

      if (block_size == 1) {
        float gi = gradIn[i];
        float mi = moment1Out[idx] =
            moment1In[idx] * beta1_ + gi * (1 - beta1_);
        float vi = moment2Out[idx] =
            moment2In[idx] * beta2_ + gi * gi * (1 - beta2_);
        paramOut[idx] =
            paramIn[idx] + lr[0] * correction * mi / (std::sqrt(vi) + epsilon_);

      } else {
        auto offsetI = i * block_size;
        auto offsetIdx = idx * block_size;

#ifndef NDEBUG
        CAFFE_ENFORCE_GE(
            Input(PARAM).size(),
            block_size + offsetIdx,
            this->debug_def().input(PARAM),
            ", out of bound,  idx:",
            idx,
            " for input i:",
            i,
            " and block size:",
            block_size);
        CAFFE_ENFORCE_GE(
            Input(GRAD).size(),
            block_size + offsetI,
            this->debug_def().input(GRAD),
            ", out of bound idx, idx:",
            idx,
            " for input i:",
            i);
#endif

        adam_compute(
            block_size,
            paramIn + offsetIdx,
            gradIn + offsetI,
            moment1In + offsetIdx,
            moment2In + offsetIdx,
            paramOut + offsetIdx,
            moment1Out + offsetIdx,
            moment2Out + offsetIdx,
            beta1_,
            beta2_,
            epsilon_,
            correction,
            lr,
            &context_);
      }
    }
    return true;
  }

 protected:
  T beta1_;
  T beta2_;
  T epsilon_;
  INPUT_TAGS(PARAM, MOMENT_1, MOMENT_2, INDICES, GRAD, LR, ITER);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1, OUTPUT_MOMENT_2);
};

template <typename T, class Context>
class RowWiseSparseAdamOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  RowWiseSparseAdamOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        beta1_(OperatorBase::GetSingleArgument<float>("beta1", 0.9f)),
        beta2_(OperatorBase::GetSingleArgument<float>("beta2", 0.999f)),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 1e-5f)) {}

  bool RunOnDevice() override {
    // Enforce shapes
    CAFFE_ENFORCE_EQ(Input(PARAM).size(), Input(MOMENT_1).size());
    CAFFE_ENFORCE_EQ(Input(PARAM).dims()[0], Input(MOMENT_2).size());
    CAFFE_ENFORCE_EQ(
        Input(PARAM).size_from_dim(1),
        Input(GRAD).size_from_dim(Input(INDICES).ndim()));
    CAFFE_ENFORCE_EQ(Input(LR).size(), 1);

    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename SIndex>
  bool DoRunWithType() {
    const auto* lr = Input(LR).template data<T>();
    const auto iter =
        OperatorBase::Input<TensorCPU>(ITER).template data<int64_t>()[0];

    const auto t = iter + 1;
    const auto correction =
        std::sqrt(T(1.) - std::pow(beta2_, t)) / (T(1.) - std::pow(beta1_, t));

    auto block_size = Input(PARAM).size() / Input(PARAM).dim(0);
    auto n = Input(GRAD).size() / block_size;

    const auto* paramIn = Input(PARAM).template data<T>();
    const auto* indices = Input(INDICES).template data<SIndex>();
    const auto* gradIn = Input(GRAD).template data<T>();
    const auto* moment1In = Input(MOMENT_1).template data<T>();
    const auto* moment2In = Input(MOMENT_2).template data<T>();
    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<T>();
    auto* moment1Out = Output(OUTPUT_MOMENT_1)->template mutable_data<T>();
    auto* moment2Out = Output(OUTPUT_MOMENT_2)->template mutable_data<T>();

    for (auto i = 0; i < n; ++i) {
      auto idx = indices[i];

      if (block_size == 1) {
        float gi = gradIn[i];
        float mi = moment1Out[idx] =
            moment1In[idx] * beta1_ + gi * (1 - beta1_);
        float vi = moment2Out[idx] =
            moment2In[idx] * beta2_ + gi * gi * (1 - beta2_);
        paramOut[idx] =
            paramIn[idx] + lr[0] * correction * mi / (std::sqrt(vi) + epsilon_);

      } else {
        auto offsetI = i * block_size;
        auto offsetIdx = idx * block_size;

#ifndef NDEBUG
        CAFFE_ENFORCE_GE(
            Input(PARAM).size(),
            block_size + offsetIdx,
            this->debug_def().input(PARAM),
            ", out of bound,  idx:",
            idx,
            " for input i:",
            i,
            " and block size:",
            block_size);
        CAFFE_ENFORCE_GE(
            Input(GRAD).size(),
            block_size + offsetI,
            this->debug_def().input(GRAD),
            ", out of bound idx, idx:",
            idx,
            " for input i:",
            i);
#endif

        const float* w = paramIn + offsetIdx;
        const float* g = gradIn + offsetI;
        const float* m1 = moment1In + offsetIdx;
        const float* m2 = moment2In + idx;
        float* nw = paramOut + offsetIdx;
        float* nm1 = moment1Out + offsetIdx;
        float* nm2 = moment2Out + idx;

        float m2_sum = 0.;
        for (auto j = 0; j < block_size; ++j) {
          float gj = g[j];
          m2_sum += gj * gj;
        }
        float vi = nm2[0] =
            m2[0] * beta2_ + (m2_sum / block_size) * (1 - beta2_);
        for (auto j = 0; j < block_size; ++j) {
          float mi = nm1[j] = m1[j] * beta1_ + g[j] * (1 - beta1_);
          nw[j] = w[j] + lr[0] * correction * mi / (std::sqrt(vi) + epsilon_);
        }
      }
    }
    return true;
  }

 protected:
  T beta1_;
  T beta2_;
  T epsilon_;
  INPUT_TAGS(PARAM, MOMENT_1, MOMENT_2, INDICES, GRAD, LR, ITER);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1, OUTPUT_MOMENT_2);
};
} // namespace caffe2
