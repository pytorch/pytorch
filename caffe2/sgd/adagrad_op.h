#pragma once

#include "caffe2/core/operator.h"
#include "caffe2/perfkernels/adagrad.h"
#if defined(USE_FBGEMM) && !defined(__NVCC__)
#include "fbgemm/FbgemmEmbedding.h"
#endif

namespace caffe2 {

template <typename Context>
void adagrad_update(
    int N,
    const float* w,
    const float* g,
    const float* h,
    float* nw,
    float* nh,
    float epsilon,
    float decay,
    const float* lr,
    Context* /*context*/,
    float weight_decay = 0.f) {
  return adagrad_update(
      N, w, g, h, nw, nh, epsilon, decay, lr[0], weight_decay);
}

template <typename Context>
void adagrad_update_output_effective_lr(
    int N,
    const float* paramIn,
    const float* gradIn,
    const float* momentIn,
    float* paramOut,
    float* momentOut,
    float* effectiveLROut,
    float epsilon,
    float decay,
    const float* lr,
    Context* /*context*/,
    float weight_decay = 0.f) {
  for (auto i = 0; i < N; ++i) {
    float grad = std::fma(weight_decay, paramIn[i], gradIn[i]);
    float moment = momentOut[i] = decay * momentIn[i] + grad * grad;
    float effective_lr = effectiveLROut[i] =
        lr[0] / (std::sqrt(moment) + epsilon);
    paramOut[i] = paramIn[i] + effective_lr * grad;
  }
}

template <typename Context>
void adagrad_update_output_effective_lr_and_update(
    int N,
    const float* paramIn,
    const float* gradIn,
    const float* momentIn,
    float* paramOut,
    float* momentOut,
    float* effectiveLROut,
    float* updateOut,
    float epsilon,
    float decay,
    const float* lr,
    Context* /*context*/,
    float weight_decay = 0.f) {
  for (auto i = 0; i < N; ++i) {
    float grad = std::fma(weight_decay, paramIn[i], gradIn[i]);
    float moment = momentOut[i] = decay * momentIn[i] + grad * grad;
    float effective_lr = effectiveLROut[i] =
        lr[0] / (std::sqrt(moment) + epsilon);
    float update = updateOut[i] = effective_lr * grad;
    paramOut[i] = paramIn[i] + update;
  }
}

template <class Context>
class AdagradOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  AdagradOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5f)),
        decay_(this->template GetSingleArgument<float>("decay", 1.0f)),
        weight_decay_(
            this->template GetSingleArgument<float>("weight_decay", 0.f)) {
    VLOG(1) << "gradient optimization operator in use: "
            << "AdagradOp"
            << " weight_decay_=" << weight_decay_;
  }

  bool RunOnDevice() override {
    CAFFE_ENFORCE_EQ(
        Input(GRAD).numel(),
        Input(MOMENT_1).numel(),
        "PARAM size: ",
        Input(PARAM).numel(),
        ", GRAD size: ",
        Input(GRAD).numel(),
        ", MOMENT_1 size: ",
        Input(MOMENT_1).numel(),
        ", LR size: ",
        Input(LR).numel());

    CAFFE_ENFORCE_EQ(Input(GRAD).numel(), Input(PARAM).numel());
    Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
    Output(OUTPUT_MOMENT_1)->ResizeLike(Input(MOMENT_1));
    if (OutputSize() == 2) {
      adagrad_update<Context>(
          Input(GRAD).numel(),
          Input(PARAM).template data<float>(),
          Input(GRAD).template data<float>(),
          Input(MOMENT_1).template data<float>(),
          Output(OUTPUT_PARAM)->template mutable_data<float>(),
          Output(OUTPUT_MOMENT_1)->template mutable_data<float>(),
          epsilon_,
          decay_,
          Input(LR).template data<float>(),
          &context_,
          weight_decay_);
    } else if (OutputSize() == 3) {
      Output(OUTPUT_EFFECTIVE_LR)->ResizeLike(Input(GRAD));
      adagrad_update_output_effective_lr<Context>(
          Input(GRAD).numel(),
          Input(PARAM).template data<float>(),
          Input(GRAD).template data<float>(),
          Input(MOMENT_1).template data<float>(),
          Output(OUTPUT_PARAM)->template mutable_data<float>(),
          Output(OUTPUT_MOMENT_1)->template mutable_data<float>(),
          Output(OUTPUT_EFFECTIVE_LR)->template mutable_data<float>(),
          epsilon_,
          decay_,
          Input(LR).template data<float>(),
          &context_,
          weight_decay_);
    } else {
      Output(OUTPUT_EFFECTIVE_LR)->ResizeLike(Input(GRAD));
      Output(OUTPUT_UPDATE)->ResizeLike(Input(GRAD));
      adagrad_update_output_effective_lr_and_update<Context>(
          Input(GRAD).numel(),
          Input(PARAM).template data<float>(),
          Input(GRAD).template data<float>(),
          Input(MOMENT_1).template data<float>(),
          Output(OUTPUT_PARAM)->template mutable_data<float>(),
          Output(OUTPUT_MOMENT_1)->template mutable_data<float>(),
          Output(OUTPUT_EFFECTIVE_LR)->template mutable_data<float>(),
          Output(OUTPUT_UPDATE)->template mutable_data<float>(),
          epsilon_,
          decay_,
          Input(LR).template data<float>(),
          &context_,
          weight_decay_);
    }

    return true;
  }

 protected:
  float epsilon_;
  float decay_;
  float weight_decay_;
  INPUT_TAGS(PARAM, MOMENT_1, GRAD, LR);
  OUTPUT_TAGS(
      OUTPUT_PARAM,
      OUTPUT_MOMENT_1,
      OUTPUT_EFFECTIVE_LR,
      OUTPUT_UPDATE);
};

class SparseAdagradOp final : public Operator<CPUContext> {
 public:
  SparseAdagradOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5f)),
        weight_decay_(
            this->template GetSingleArgument<float>("weight_decay", 0.f)) {
    VLOG(1) << "gradient optimization operator in use: "
            << "SparseAdagradOp"
            << " weight_decay_=" << weight_decay_;
    const float decay = this->template GetSingleArgument<float>("decay", 1.0);
    CAFFE_ENFORCE_EQ(
        decay, 1.0, "Decay is not supported for SparseSimdAdagradOp");
  }

  bool RunOnDevice() override {
    // Enforce shapes
    // input(embedding/momentum) == outputs(embedding/momentum)
    CAFFE_ENFORCE_EQ(
        Input(PARAM).numel(),
        Input(MOMENT_1).numel(),
        "Input Param size: ",
        Input(PARAM).numel(),
        " Input Moment size: ",
        Input(MOMENT_1).numel());
    CAFFE_ENFORCE_EQ(Input(LR).numel(), 1);
    CAFFE_ENFORCE_EQ(
        Input(PARAM).size_from_dim(1),
        Input(GRAD).size_from_dim(Input(INDICES).dim()));

    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename SIndex>
  bool DoRunWithType() {
    const auto* lr = Input(LR).template data<float>();

    auto n = Input(INDICES).numel();

    const auto* indices = Input(INDICES).template data<SIndex>();
    const auto* gradIn = Input(GRAD).template data<float>();
    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<float>();
    auto* momentOut = Output(OUTPUT_MOMENT_1)->template mutable_data<float>();

    if (n == 0) {
      return true;
    }
    auto block_size = Input(GRAD).numel() / n;

    // input(grad) is compatible with size of indexes
    CAFFE_ENFORCE_EQ(
        Input(GRAD).numel() % n,
        0,
        "Incorrect gradient size:",
        Input(GRAD).numel(),
        " size of indexes:",
        n);

#if defined(USE_FBGEMM) && !defined(__NVCC__)
    VLOG(1) << "using fbgemm::GenerateSparseAdaGrad in SparseAdagradOp";

    if (block_size != last_block_size_) {
      last_block_size_ = block_size;
      if (std::is_same<SIndex, std::int32_t>::value) {
        kernel_i32_ = fbgemm::GenerateSparseAdaGrad<std::int32_t>(
            block_size, /*rowwise=*/false, /*prefetch=*/16, weight_decay_);
      } else {
        CAFFE_ENFORCE((std::is_same<SIndex, std::int64_t>::value));
        kernel_i64_ = fbgemm::GenerateSparseAdaGrad<std::int64_t>(
            block_size, /*rowwise=*/false, /*prefetch=*/16, weight_decay_);
      }
    }

    int num_rows_processed;
    if (std::is_same<SIndex, std::int32_t>::value) {
      num_rows_processed = kernel_i32_(
          n,
          Input(PARAM).numel(),
          paramOut,
          gradIn,
          momentOut,
          reinterpret_cast<const std::int32_t*>(indices),
          epsilon_,
          lr[0]);
    } else {
      num_rows_processed = kernel_i64_(
          n,
          Input(PARAM).numel(),
          paramOut,
          gradIn,
          momentOut,
          reinterpret_cast<const std::int64_t*>(indices),
          epsilon_,
          lr[0]);
    }
    if (num_rows_processed < n) {
      CAFFE_ENFORCE_GE(
          Input(PARAM).numel(),
          (indices[num_rows_processed] + 1) * block_size,
          this->debug_def().input(PARAM),
          ", out of bound,  idx:",
          indices[num_rows_processed],
          " for input i:",
          num_rows_processed,
          " and block_size:",
          block_size,
          " max size:",
          Input(PARAM).numel());
      return false;
    } else {
      return true;
    }
#endif

    VLOG(1)
        << "using internal::adagrad_update_prefetch_inlined in SparseAdagradOp";

    const auto* paramIn = Input(PARAM).template data<float>();
    const auto* momentIn = Input(MOMENT_1).template data<float>();

    std::vector<float> grad(block_size);
    for (auto i = 0; i < n; ++i) {
      auto idx = indices[i];
      auto offsetI = i * block_size;
      auto offsetIdx = idx * block_size;

      // Enforce:
      // access within range
      // gradient access within range
      CAFFE_ENFORCE_GE(
          Input(PARAM).numel(),
          block_size + offsetIdx,
          this->debug_def().input(PARAM),
          ", out of bound,  idx:",
          idx,
          " for input i:",
          i,
          " and block size:",
          block_size,
          " max size:",
          Input(PARAM).numel());

      if (block_size == 1) {
        float gi = std::fma(weight_decay_, paramIn[idx], gradIn[i]);
        float hi = momentOut[idx] = momentIn[idx] + gi * gi;
        paramOut[idx] = paramIn[idx] + lr[0] * gi / (std::sqrt(hi) + epsilon_);
      } else {
        // prefetching
        const int prefdist_T0 = 16;
        int i_pref = (i < n - prefdist_T0) ? i + prefdist_T0 : i;
        std::size_t idx_pref = indices[i_pref];

        internal::adagrad_update_prefetch_inlined(
            block_size,
            paramIn + offsetIdx,
            &paramIn[idx_pref * block_size],
            gradIn + offsetI,
            momentIn + offsetIdx,
            &momentIn[idx_pref * block_size],
            paramOut + offsetIdx,
            &paramOut[idx_pref * block_size],
            momentOut + offsetIdx,
            &momentOut[idx_pref * block_size],
            epsilon_,
            lr[0],
            weight_decay_);
      }
    }
    return true;
  }

 protected:
  float epsilon_;
  float weight_decay_;
#if defined(USE_FBGEMM) && !defined(__NVCC__)
  fbgemm::SparseAdaGradSignature<std::int32_t>::Type kernel_i32_;
  fbgemm::SparseAdaGradSignature<std::int64_t>::Type kernel_i64_;
  std::int64_t last_block_size_{-1};
#endif

  INPUT_TAGS(PARAM, MOMENT_1, INDICES, GRAD, LR);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1);
};

template <class Context>
class RowWiseSparseAdagradOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  RowWiseSparseAdagradOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5f)),
        weight_decay_(
            this->template GetSingleArgument<float>("weight_decay", 0.f)) {
    VLOG(1) << "gradient optimization operator in use: "
            << "RowWiseSparseAdagradOp"
            << " weight_decay_=" << weight_decay_;
  }

  bool RunOnDevice() override {
    // Enforce shapes
    CAFFE_ENFORCE_EQ(Input(PARAM).sizes()[0], Input(MOMENT_1).numel());
    CAFFE_ENFORCE_EQ(Input(LR).numel(), 1);
    CAFFE_ENFORCE_EQ(
        Input(PARAM).size_from_dim(1),
        Input(GRAD).size_from_dim(Input(INDICES).dim()));

    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename SIndex>
  bool DoRunWithType() {
    const auto* lr = Input(LR).template data<float>();
    auto* param = Output(OUTPUT_PARAM)->template mutable_data<float>();
    auto* moment = Output(OUTPUT_MOMENT_1)->template mutable_data<float>();

    const auto* indices = Input(INDICES).template data<SIndex>();
    const auto* gradIn = Input(GRAD).template data<float>();

    auto n = Input(INDICES).numel();
    if (n == 0) {
      return true;
    }

    auto block_size = Input(GRAD).numel() / n;

    // Enforce:
    // Input(embedding/momentum) == outputs(embedding/momentum)
    CAFFE_ENFORCE_EQ(
        Input(PARAM).numel() / block_size,
        Input(MOMENT_1).numel(),
        "Input Param size: ",
        Input(PARAM).numel(),
        " Block size: ",
        block_size,
        " Input Moment size: ",
        Input(MOMENT_1).numel());

    // input(grad) is compatible with size of indexes
    CAFFE_ENFORCE_EQ(
        Input(GRAD).numel() % n,
        0,
        "Incorrect gradient size:",
        Input(GRAD).numel(),
        " size of indexes:",
        n);

#if defined(USE_FBGEMM) && !defined(__NVCC__)
    VLOG(1) << "using fbgemm::GenerateSparseAdaGrad in RowWiseSparseAdagradOp";

    if (block_size != last_block_size_) {
      last_block_size_ = block_size;
      if (std::is_same<SIndex, std::int32_t>::value) {
        kernel_i32_ = fbgemm::GenerateSparseAdaGrad<std::int32_t>(
            block_size, /*rowwise=*/true, /*prefetch=*/16, weight_decay_);
      } else {
        CAFFE_ENFORCE((std::is_same<SIndex, std::int64_t>::value));
        kernel_i64_ = fbgemm::GenerateSparseAdaGrad<std::int64_t>(
            block_size, /*rowwise=*/true, /*prefetch=*/16, weight_decay_);
      }
    }

    int num_rows_processed;
    if (std::is_same<SIndex, std::int32_t>::value) {
      num_rows_processed = kernel_i32_(
          n,
          Input(PARAM).numel(),
          param,
          gradIn,
          moment,
          reinterpret_cast<const std::int32_t*>(indices),
          epsilon_,
          lr[0]);
    } else {
      num_rows_processed = kernel_i64_(
          n,
          Input(PARAM).numel(),
          param,
          gradIn,
          moment,
          reinterpret_cast<const std::int64_t*>(indices),
          epsilon_,
          lr[0]);
    }

    if (num_rows_processed < n) {
      // Enforce:
      // access within range
      CAFFE_ENFORCE_GE(
          Input(PARAM).numel(),
          (indices[num_rows_processed] + 1) * block_size,
          this->debug_def().input(PARAM),
          ", out of bound,  idx:",
          indices[num_rows_processed],
          " for input i:",
          num_rows_processed,
          " and block size:",
          block_size,
          " max size:",
          Input(PARAM).numel());
      return false;
    } else {
      return true;
    }
#else
    VLOG(1) << "using plain adagrad updates in RowWiseSparseAdagradOp";

    for (auto i = 0; i < n; ++i) {
      auto idx = indices[i];
      if (block_size == 1) {
        float gi = std::fma(weight_decay_, param[idx], gradIn[i]);
        float hi = moment[idx] = moment[idx] + gi * gi;
        param[idx] = param[idx] + lr[0] * gi / (std::sqrt(hi) + epsilon_);
      } else {
        auto offsetI = i * block_size;
        auto offsetIdx = idx * block_size;

#ifndef NDEBUG
        CAFFE_ENFORCE_GE(
            Input(PARAM).numel(),
            block_size + offsetIdx,
            this->debug_def().input(PARAM),
            ", out of bound,  idx:",
            idx,
            " for input i:",
            i,
            " and block size:",
            block_size);
        CAFFE_ENFORCE_GE(
            Input(GRAD).numel(),
            block_size + offsetI,
            this->debug_def().input(GRAD),
            ", out of bound idx, idx:",
            idx,
            " for input i:",
            i);
#endif

        float* w = param + offsetIdx;
        const float* g = gradIn + offsetI;
        float* h = moment + idx;
        float hs = 0.;
        for (auto j = 0; j < block_size; ++j) {
          float gj = std::fma(weight_decay_, w[j], g[j]);
          hs += gj * gj;
        }
        float hi = h[0] = h[0] + hs / block_size;
        float step = lr[0] / (std::sqrt(hi) + epsilon_);
        for (auto j = 0; j < block_size; ++j) {
          float gj = std::fma(weight_decay_, w[j], g[j]);
          w[j] = w[j] + gj * step;
        }
      }
    }
    return true;
#endif // !USE_FBGEMM
  }

 protected:
  float epsilon_;
  float weight_decay_;
#if defined(USE_FBGEMM) && !defined(__NVCC__)
  fbgemm::SparseAdaGradSignature<std::int32_t>::Type kernel_i32_;
  fbgemm::SparseAdaGradSignature<std::int64_t>::Type kernel_i64_;
  std::int64_t last_block_size_{-1};
#endif

  INPUT_TAGS(PARAM, MOMENT_1, INDICES, GRAD, LR);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1);
};
} // namespace caffe2
