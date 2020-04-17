#pragma once

#include "caffe2/core/operator.h"
#include "caffe2/sgd/math_lp.h"

namespace caffe2 {

namespace internal {
inline float compute_square_average_inlined_(const float* a, int len) {
  float sum = 0.0f;

  int i = 0;
#ifdef __AVX__
  constexpr int kSize = 8;
  __m256 partial_sum = _mm256_setzero_ps();
  for (; i + kSize <= len; i += kSize) {
    __m256 ai = _mm256_loadu_ps(a + i);
    partial_sum = _mm256_add_ps(partial_sum, _mm256_mul_ps(ai, ai));
  }
  // Reduce sum to 1 value
  __m256 partial_sum_2 = _mm256_hadd_ps(partial_sum, partial_sum);
  __m256 partial_sum_3 = _mm256_hadd_ps(partial_sum_2, partial_sum_2);
  sum = _mm_cvtss_f32(_mm256_castps256_ps128(partial_sum_3)) +
      _mm_cvtss_f32(_mm256_extractf128_ps(partial_sum_3, 1));
#endif

  for (; i < len; ++i) {
    sum = std::fma(a[i], a[i], sum);
  }

  return sum / len;
}
} // namespace internal

/**
 * Fused operator of
 * SparseLengthsIndicesInGradientSumGradient (gradient of SparseLengthsSum) +
 * RowWiseSparseAdagrad.
 *
 * BW saving analysis for numSegments B, L_avg = avg(lengths), block_size D,
 * assuming T = float and SIndex = int64_t:
 * Before fusion, SparseLengthsIndicesInGradientSumGradient reads B*D*4 and
 * writes B*L_avg*D*4. RowWiseSparseAdagrad reads B*2*L_avg*D*4 and writes
 * B*L_avg*D*4. So, the total memory traffic is B*(1+4*L_avg)*D*4 .
 * After fusion, we read B*(1+L_avg)*D*4 and write B*L_avg*D*4 with total
 * memory traffic B*(1+2*L_avg)*D*4.
 * Assuming L_avg >> 1, the memory BW is saving is about 2x .
 *
 * See https://fb.quip.com/ldG7A55Ur5wM for more details on BW saving analysis
 * and evaluation results.
 */
template <
    typename Tdata, // embedding and momentum types
    typename T, // everything else
    typename TLengths,
    typename rowWiseAdagradT>
class RowWiseSparseAdagradFusedWithSparseLengthsSumGradientOp final
    : public Operator<CPUContext> {
 public:
  RowWiseSparseAdagradFusedWithSparseLengthsSumGradientOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5)) {
    const T decay = this->template GetSingleArgument<T>("decay", 1.0);
    CAFFE_ENFORCE_EQ(
        decay, 1.0, "Decay is not supported for SparseSimdAdagradOp");
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
    const auto* lr = Input(LR).template data<T>();
    Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
    Output(OUTPUT_MOMENT_1)->ResizeLike(Input(MOMENT_1));

    auto& segmentGradsInput = Input(GRAD);
    auto& lengthsInput = Input(LENGTHS);

    CAFFE_ENFORCE_EQ(lengthsInput.dim(), 1, "LENGTHS must be a vector");
    auto numSegments = lengthsInput.size(0);
    CAFFE_ENFORCE_GT(segmentGradsInput.dim(), 0);
    CAFFE_ENFORCE_EQ(numSegments, segmentGradsInput.size(0));
    const auto* lengths = lengthsInput.template data<TLengths>();

    auto n = Input(INDICES).numel();
    auto numParams = Input(PARAM).numel();

    const auto* indices = Input(INDICES).template data<SIndex>();
    const auto* gradIn = segmentGradsInput.template data<T>();
    const auto* paramIn = Input(PARAM).template data<Tdata>();
    const auto* momentIn = Input(MOMENT_1).template data<T>();

    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<Tdata>();
    auto* momentOut = Output(OUTPUT_MOMENT_1)->template mutable_data<T>();

    if (numSegments == 0) {
      return true;
    }

    auto block_size = segmentGradsInput.size_from_dim(1);

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

    compute<SIndex>(
        block_size,
        indices,
        n,
        lengths,
        numSegments,
        gradIn,
        paramIn,
        numParams,
        momentIn,
        paramOut,
        momentOut,
        epsilon_,
        lr[0],
        kernel_);

    return true;
  }

  template <typename SIndex>
  static void compute(
      int64_t block_size,
      const SIndex* indices,
      int64_t n,
      const TLengths* lengths,
      int64_t numSegments,
      const T* gradIn,
      const Tdata* paramIn,
      int64_t numParams,
      const T* momentIn,
      Tdata* paramOut,
      T* momentOut,
      float epsilon,
      T lr,
      rowWiseAdagradT& kernel) {
    int dataIndex = 0;
    for (auto rangeIndex = 0; rangeIndex < numSegments; ++rangeIndex) {
      auto offsetI = rangeIndex * block_size;
      const float* g = gradIn + offsetI;

      float g_sq_avg = 0;
      if (block_size > 1) {
        g_sq_avg = internal::compute_square_average_inlined_(g, block_size);
      }

      for (auto start = dataIndex; dataIndex < start + lengths[rangeIndex];
           ++dataIndex) {
        std::size_t idx = indices[dataIndex];
        auto offsetIdx = idx * block_size;

        // Enforce:
        // access within range
        // gradient access within range
        CAFFE_ENFORCE_GE(
            numParams,
            block_size + offsetIdx,
            "Accessing params out of bound,  idx:",
            idx,
            " for input dataIndex:",
            dataIndex,
            " and block size:",
            block_size,
            " max size:",
            numParams);

        if (block_size == 1) {
          float gi = *g;
          float hi = momentOut[idx] = momentIn[idx] + gi * gi;
          paramOut[idx] = paramIn[idx] + lr / (std::sqrt(hi) + epsilon) * gi;
        } else {
          // prefetching
          const int prefdist_T0 = 16;
          int i_pref = (dataIndex < n - prefdist_T0) ? dataIndex + prefdist_T0
                                                     : dataIndex;
          std::size_t idx_pref = indices[i_pref];

          kernel(
              block_size,

              paramOut + offsetIdx,
              &paramOut[idx_pref * block_size],

              g,
              g_sq_avg,

              momentOut + idx,
              momentOut + idx_pref,

              epsilon,
              lr);
        }
      }
    }
    CAFFE_ENFORCE_EQ(dataIndex, n);
  }

 protected:
  T epsilon_;
  rowWiseAdagradT kernel_;

  INPUT_TAGS(PARAM, MOMENT_1, INDICES, GRAD, LR, LENGTHS);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1);
};

template <
    typename Tdata, // embedding and momentum types
    typename T, // everything else
    typename TLengths,
    typename rowWiseAdagradT>
class RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientOp final
    : public Operator<CPUContext> {
 public:
  RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5)) {}

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
    const auto* lr = Input(LR).template data<T>();
    Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
    Output(OUTPUT_MOMENT_1)->ResizeLike(Input(MOMENT_1));

    auto& segmentGradsInput = Input(GRAD);
    auto& lengthsInput = Input(LENGTHS);

    CAFFE_ENFORCE_EQ(lengthsInput.dim(), 1, "LENGTHS must be a vector");
    auto numSegments = lengthsInput.size(0);
    CAFFE_ENFORCE_GT(segmentGradsInput.dim(), 0);
    CAFFE_ENFORCE_EQ(numSegments, segmentGradsInput.size(0));
    const auto* lengths = lengthsInput.template data<TLengths>();

    auto n = Input(INDICES).numel();
    auto numParams = Input(PARAM).numel();

    const auto* indices = Input(INDICES).template data<SIndex>();
    const auto* gradIn = segmentGradsInput.template data<T>();
    const auto* paramIn = Input(PARAM).template data<Tdata>();
    const auto* momentIn = Input(MOMENT_1).template data<T>();
    const auto* auxParamIn = Input(AUX_PARAM).template data<T>();

    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<Tdata>();
    auto* momentOut = Output(OUTPUT_MOMENT_1)->template mutable_data<T>();
    Output(AUX_GRAD)->Resize(n);
    auto* auxGrad = Output(AUX_GRAD)->template mutable_data<T>();

    CAFFE_ENFORCE_EQ(
        paramIn, paramOut, "RowWiseSparseAdagrad must use inplace param");
    CAFFE_ENFORCE_EQ(
        momentIn, momentOut, "RowWiseSparseAdagrad must use inplace momentum");

    if (numSegments == 0) {
      return true;
    }

    auto block_size = segmentGradsInput.size_from_dim(1);

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

    compute<SIndex>(
        block_size,
        indices,
        n,
        lengths,
        numSegments,
        gradIn,
        paramIn,
        numParams,
        momentIn,
        auxParamIn,
        paramOut,
        momentOut,
        auxGrad,
        epsilon_,
        lr[0],
        kernel_,
        &context_);

    return true;
  }

  template <typename SIndex>
  static void compute(
      int64_t block_size,
      const SIndex* indices,
      int64_t n,
      const TLengths* lengths,
      int64_t numSegments,
      const T* gradIn,
      const Tdata* paramIn,
      int64_t numParams,
      const T* momentIn,
      const T* auxParamIn,
      Tdata* paramOut,
      T* momentOut,
      T* auxGrad,
      float epsilon,
      T lr,
      rowWiseAdagradT& kernel,
      CPUContext* context) {
    // Cannot fuse this loop with the loop below because paramIn is updated
    // by the second loop. Specifically, there could be dataIndex1 != dataIndex2
    // s.t. indices[dataIndex1] == indices[dataIndex2], and fusing these two
    // loops would violate dependencies w.r.t.
    // paramIn[indices[dataIndex1]:block_size] The approximate version.
    // (RowWiseSparseSimdAdagradFusedWithSparseLengthsWeightedSumGradientApproxOp)
    // ignores this dependency and fuses these two loops.
    std::vector<T> temp_grad(block_size);
    int dataIndex = 0;
    for (auto rangeIndex = 0; rangeIndex < numSegments; ++rangeIndex) {
      for (auto start = dataIndex; dataIndex < start + lengths[rangeIndex];
           ++dataIndex) {
        std::size_t idx = indices[dataIndex];
        auto offsetI = rangeIndex * block_size;
        auto offsetIdx = idx * block_size;

        // Enforce:
        // access within range
        // gradient access within range
        CAFFE_ENFORCE_GE(
            numParams,
            block_size + offsetIdx,
            "Accessing params out of bound,  idx:",
            idx,
            " for input dataIndex:",
            dataIndex,
            " and block size:",
            block_size,
            " max size:",
            numParams);

        // temp_aux_grad[dataIndex] = gradIn[offsetI] dot paramIn[offsetIdx]
        internal::dot<T, Tdata, T>(
            block_size,
            gradIn + offsetI,
            paramIn + offsetIdx,
            auxGrad + dataIndex,
            context);
      }
    }
    CAFFE_ENFORCE_EQ(dataIndex, n);

    dataIndex = 0;
    for (auto rangeIndex = 0; rangeIndex < numSegments; ++rangeIndex) {
      auto offsetI = rangeIndex * block_size;
      const float* g = gradIn + offsetI;

      float g_sq_avg;
      if (block_size > 1) {
        g_sq_avg = internal::compute_square_average_inlined_(g, block_size);
      }

      for (auto start = dataIndex; dataIndex < start + lengths[rangeIndex];
           ++dataIndex) {
        auto idx = indices[dataIndex];
        auto offsetIdx = idx * block_size;
        auto localOffset = dataIndex - start;

        for (int i = 0; i < block_size; ++i) {
          temp_grad[i] = auxParamIn[localOffset] * g[i];
        }

        if (block_size == 1) {
          float gi = temp_grad[0];
          float hi = momentOut[idx] = momentIn[idx] + gi * gi;
          paramOut[idx] = paramIn[idx] + lr / (std::sqrt(hi) + epsilon) * gi;
        } else {
          // prefetching
          const int prefdist_T0 = 16;
          int i_pref = (dataIndex < n - prefdist_T0) ? dataIndex + prefdist_T0
                                                     : dataIndex;
          std::size_t idx_pref = indices[i_pref];
          kernel(
              block_size,

              paramOut + offsetIdx,
              &paramOut[idx_pref * block_size],

              temp_grad.data(),
              g_sq_avg * auxParamIn[localOffset] * auxParamIn[localOffset],

              momentOut + idx,
              momentOut + idx_pref,

              epsilon,
              lr);
        }
      }
    }
  }

 protected:
  T epsilon_;
  rowWiseAdagradT kernel_;

  INPUT_TAGS(PARAM, MOMENT_1, AUX_PARAM, INDICES, GRAD, LR, LENGTHS);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1, AUX_GRAD);
};

template <
    typename Tdata, // embedding and momentum types
    typename T, // everything else
    typename TLengths,
    typename rowWiseAdagradT>
class RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientApproxOp
    final : public Operator<CPUContext> {
 public:
  RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientApproxOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5)) {
    const T decay = this->template GetSingleArgument<T>("decay", 1.0);
    CAFFE_ENFORCE_EQ(
        decay, 1.0, "Decay is not supported for SparseSimdAdagradOp");
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
    const auto* lr = Input(LR).template data<T>();
    Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
    Output(OUTPUT_MOMENT_1)->ResizeLike(Input(MOMENT_1));

    auto& segmentGradsInput = Input(GRAD);
    auto& lengthsInput = Input(LENGTHS);

    CAFFE_ENFORCE_EQ(lengthsInput.dim(), 1, "LENGTHS must be a vector");
    auto numSegments = lengthsInput.size(0);
    CAFFE_ENFORCE_GT(segmentGradsInput.dim(), 0);
    CAFFE_ENFORCE_EQ(numSegments, segmentGradsInput.size(0));
    const auto* lengths = lengthsInput.template data<TLengths>();

    auto n = Input(INDICES).numel();

    const auto* indices = Input(INDICES).template data<SIndex>();
    const auto* gradIn = segmentGradsInput.template data<T>();
    const auto* paramIn = Input(PARAM).template data<Tdata>();
    const auto* momentIn = Input(MOMENT_1).template data<T>();
    const auto* auxParamIn = Input(AUX_PARAM).template data<T>();

    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<Tdata>();
    auto* momentOut = Output(OUTPUT_MOMENT_1)->template mutable_data<T>();
    Output(AUX_GRAD)->Resize(n);
    auto* auxGrad = Output(AUX_GRAD)->template mutable_data<T>();

    CAFFE_ENFORCE_EQ(
        paramIn, paramOut, "RowWiseSparseAdagrad must use inplace param");
    CAFFE_ENFORCE_EQ(
        momentIn, momentOut, "RowWiseSparseAdagrad must use inplace momentum");

    if (numSegments == 0) {
      return true;
    }

    auto block_size = segmentGradsInput.size_from_dim(1);

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

    std::vector<T> temp_grad(block_size);
    int dataIndex = 0;
    for (auto rangeIndex = 0; rangeIndex < numSegments; ++rangeIndex) {
      auto offsetI = rangeIndex * block_size;
      const float* g = gradIn + offsetI;

      float g_sq_avg;
      if (block_size > 1) {
        g_sq_avg = internal::compute_square_average_inlined_(g, block_size);
      }

      for (auto start = dataIndex; dataIndex < start + lengths[rangeIndex];
           ++dataIndex) {
        std::size_t idx = indices[dataIndex];
        auto offsetIdx = idx * block_size;
        auto localOffset = dataIndex - start;

        // Enforce:
        // access within range
        // gradient access within range
        CAFFE_ENFORCE_GE(
            Input(PARAM).numel(),
            block_size + offsetIdx,
            this->debug_def().input(PARAM),
            ", out of bound,  idx:",
            idx,
            " for input dataIndex:",
            dataIndex,
            " and block size:",
            block_size,
            " max size:",
            Input(PARAM).numel());

        int i = 0;
        float acc = 0.0f;

#ifdef __AVX__
        constexpr int VLEN = 8;
        __m256 acc_v = _mm256_setzero_ps();
        __m256 scalar_v = _mm256_set1_ps(auxParamIn[localOffset]);

        if (std::is_same<Tdata, float>::value) {
          for (; i < block_size / VLEN * VLEN; i += VLEN) {
            __m256 a_v = _mm256_loadu_ps(g + i);
            __m256 b_v = _mm256_loadu_ps(
                reinterpret_cast<const float*>(paramIn + offsetIdx + i));
            __m256 c_v = _mm256_mul_ps(a_v, b_v);
            acc_v = _mm256_add_ps(acc_v, c_v);
            _mm256_storeu_ps(&temp_grad[i], _mm256_mul_ps(a_v, scalar_v));
          }
        } else if (std::is_same<Tdata, at::Half>::value) {
          for (; i < block_size / VLEN * VLEN; i += VLEN) {
            __m256 a_v = _mm256_loadu_ps(g + i);
            __m256 b_v = _mm256_cvtph_ps(
                _mm_load_si128((__m128i*)(paramIn + offsetIdx + i)));
            __m256 c_v = _mm256_mul_ps(a_v, b_v);
            acc_v = _mm256_add_ps(acc_v, c_v);
            _mm256_storeu_ps(&temp_grad[i], _mm256_mul_ps(a_v, scalar_v));
          }
        } else {
          CAFFE_THROW("Unsupported type for Embedding");
        }

        alignas(64) float temp[VLEN];
        _mm256_store_ps(temp, acc_v);
        for (int j = 0; j < VLEN; ++j) {
          acc += temp[j];
        }
#endif

        for (; i < block_size; ++i) {
          float a = g[i];
          acc += a * paramIn[offsetIdx + i];
          temp_grad[i] = a * auxParamIn[localOffset];
        }
        auxGrad[dataIndex] = acc;

        if (block_size == 1) {
          float gi = temp_grad[0];
          float hi = momentOut[idx] = momentIn[idx] + gi * gi;
          paramOut[idx] =
              paramIn[idx] + lr[0] / (std::sqrt(hi) + epsilon_) * gi;
        } else {
          // prefetching
          const int prefdist_T0 = 16;
          int i_pref = (dataIndex < n - prefdist_T0) ? dataIndex + prefdist_T0
                                                     : dataIndex;
          std::size_t idx_pref = indices[i_pref];
          kernel_(
              block_size,

              paramOut + offsetIdx,
              &paramOut[idx_pref * block_size],

              temp_grad.data(),
              g_sq_avg * auxParamIn[localOffset] * auxParamIn[localOffset],

              momentOut + idx,
              momentOut + idx_pref,

              epsilon_,
              lr[0]);
        }
      }
    }
    CAFFE_ENFORCE_EQ(dataIndex, n);

    return true;
  }

 protected:
  T epsilon_;
  rowWiseAdagradT kernel_;

  INPUT_TAGS(PARAM, MOMENT_1, AUX_PARAM, INDICES, GRAD, LR, LENGTHS);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1, AUX_GRAD);
};

struct rowwise_adagrad_update_inlined {
  void operator()(
      int N,
      float* w,
      float* w_n, // prefetch ptr
      const float* g,
      float g_sq_avg,
      float* h,
      float* h_n, // prefetch ptr
      float epsilon,
      float lr) {
#ifdef __AVX__
    constexpr int kSize = 8;
    _mm_prefetch(reinterpret_cast<const char*>(h_n), _MM_HINT_T0);
#endif
    float hi = *h = *h + g_sq_avg;
    float float_step = lr / (std::sqrt(hi) + epsilon);

    int i = 0;

#ifdef __AVX__
    __m256 step = _mm256_set1_ps(float_step);

    for (i = 0; i + kSize <= N; i += kSize) {
      _mm_prefetch(reinterpret_cast<const char*>(&w_n[i]), _MM_HINT_T0);

      __m256 gi = _mm256_loadu_ps(g + i);
      __m256 wi = _mm256_loadu_ps(w + i);

      _mm256_storeu_ps(w + i, _mm256_add_ps(wi, _mm256_mul_ps(gi, step)));
    }
#endif

    for (; i < N; ++i) {
      float gi = g[i];
      w[i] = w[i] + gi * float_step;
    }
  }
};

} // namespace caffe2
