#include "caffe2/operators/fused_rowwise_nbitfake_conversion_ops.h"
#ifdef __AVX__
#include <immintrin.h>
#endif
#include "c10/util/Registry.h"

namespace caffe2 {

namespace {
float compress_uniform_simplified_(
    const float* X,
    int N,
    float xmin,
    float xmax,
    float* Xq,
    int bit_rate) {
  xmin = static_cast<at::Half>(xmin);
  float data_range = xmax - xmin;
  float qmax = (1 << bit_rate) - 1;
  float scale = data_range == 0
      ? 1.0
      : static_cast<float>(static_cast<at::Half>(data_range / qmax));
  float inverse_scale = 1.0f / scale;

  float norm = 0.0f;
  constexpr int VLEN = 8;
  int i = 0;

#ifdef __AVX__
  // vectorized loop
  __m256 norm_v = _mm256_setzero_ps();
  for (; i < N / VLEN * VLEN; i += VLEN) {
    __m256 X_v = _mm256_loadu_ps(X + i);
    // Affine
    __m256 Xq_v = _mm256_mul_ps(
        _mm256_sub_ps(X_v, _mm256_set1_ps(xmin)),
        _mm256_set1_ps(inverse_scale));
    // Round
    // Use _MM_FROUND_CUR_DIRECTION to match the behavior with the remainder
    // code. In most cases, the rounding mode is round-to-nearest-even.
    Xq_v = _mm256_round_ps(Xq_v, _MM_FROUND_CUR_DIRECTION);
    // Clip
    Xq_v = _mm256_max_ps(
        _mm256_setzero_ps(), _mm256_min_ps(Xq_v, _mm256_set1_ps(qmax)));
    // Inverse affine
    Xq_v = _mm256_add_ps(
        _mm256_mul_ps(Xq_v, _mm256_set1_ps(scale)), _mm256_set1_ps(xmin));
    __m256 err_v = _mm256_sub_ps(X_v, Xq_v);
    norm_v = _mm256_add_ps(_mm256_mul_ps(err_v, err_v), norm_v);
  }
  alignas(64) float temp[VLEN];
  _mm256_store_ps(temp, norm_v);
  for (int j = 0; j < VLEN; ++j) {
    norm += temp[j];
  }
#endif // __AVX__

  // remainder loop
  for (; i < N; i++) {
    Xq[i] = std::max(
        0.0f, std::min<float>(nearbyint((X[i] - xmin) * inverse_scale), qmax));
    Xq[i] = Xq[i] * scale + xmin;
    norm += (X[i] - Xq[i]) * (X[i] - Xq[i]);
  }

  return std::sqrt(norm);
}
} // namespace

namespace internal {
void convertfp32fp32(float* dst, const float* src, size_t N) {
  memcpy(dst, src, sizeof(float) * N);
}

void convertfp16fp32(float* dst, const at::Half* src, size_t N) {
  for (size_t i = 0; i < N; i++) {
    dst[i] = src[i];
  }
}

void param_search_greedy(
    const float* X,
    int N,
    const int n_bins, // = 200,
    const float ratio, // = 0.16,
    float& Xmin,
    float& Xmax,
    int bit_rate) {
  float stepsize = (Xmax - Xmin) / n_bins;
  int min_bins = n_bins * (1 - ratio);

  vector<float> Xq(N);

  float loss =
      compress_uniform_simplified_(X, N, Xmin, Xmax, Xq.data(), bit_rate);
  float best_loss = loss;

  float cur_min = Xmin;
  float cur_max = Xmax;
  float cur_loss = loss;

  float thr = min_bins * stepsize;
  while (cur_min + thr < cur_max) {
    // move left
    float loss1 = compress_uniform_simplified_(
        X, N, cur_min + stepsize, cur_max, Xq.data(), bit_rate);
    // move right
    float loss2 = compress_uniform_simplified_(
        X, N, cur_min, cur_max - stepsize, Xq.data(), bit_rate);
    if (cur_loss < loss1 && cur_loss < loss2 && cur_loss < best_loss) {
      // found a local optima
      best_loss = cur_loss;
      Xmin = cur_min;
      Xmax = cur_max;
    }
    if (loss1 < loss2) {
      cur_min = cur_min + stepsize;
      cur_loss = loss1;
    } else {
      cur_max = cur_max - stepsize;
      cur_loss = loss2;
    }
  }
}
} // namespace internal

REGISTER_CPU_OPERATOR(
    FloatToFused4BitFakeRowwiseQuantized,
    FloatToFusedNBitFakeRowwiseQuantizedOp<
        4,
        float,
        internal::convertfp32fp32>);
OPERATOR_SCHEMA(FloatToFused4BitFakeRowwiseQuantized)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(1, X.dims(1) + 8);
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_UINT8);
      return out;
    })
    .SetDoc(R"DOC(
Applies 4-bit row-wise fake quantization to a tensor of floats.
The output looks like an int8 rowwise quantized blob with
scale and biases in half float.
)DOC")
    .Input(0, "input", "Float32 input data")
    .Output(0, "output", "Fused scale, bias and quantized data");
NO_GRADIENT(FloatToFused4BitFakeRowwiseQuantized);

REGISTER_CPU_OPERATOR(
    HalfToFused4BitFakeRowwiseQuantized,
    FloatToFusedNBitFakeRowwiseQuantizedOp<
        4,
        at::Half,
        internal::convertfp16fp32>);
OPERATOR_SCHEMA(HalfToFused4BitFakeRowwiseQuantized)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(1, X.dims(1) + 8);
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_UINT8);
      return out;
    })
    .SetDoc(R"DOC(
Applies 4-bit row-wise fake quantization to a tensor of half floats.
The output looks like an int8 rowwise quantized blob with
scale and biases in half float.
)DOC")
    .Input(0, "input", "Float16 input data")
    .Output(0, "output", "Fused scale, bias and quantized data");
NO_GRADIENT(HalfToFused4BitFakeRowwiseQuantized);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    FloatToFused4BitFakeRowwiseQuantized,
    GREEDY,
    FloatToFusedNBitFakeRowwiseQuantizedOp<
        4,
        float,
        internal::convertfp32fp32,
        true /* GREEDY */>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    HalfToFused4BitFakeRowwiseQuantized,
    GREEDY,
    FloatToFusedNBitFakeRowwiseQuantizedOp<
        4,
        at::Half,
        internal::convertfp16fp32,
        true /* GREEDY */>);

REGISTER_CPU_OPERATOR(
    FloatToFused2BitFakeRowwiseQuantized,
    FloatToFusedNBitFakeRowwiseQuantizedOp<
        2,
        float,
        internal::convertfp32fp32>);
OPERATOR_SCHEMA(FloatToFused2BitFakeRowwiseQuantized)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(1, X.dims(1) + 8);
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_UINT8);
      return out;
    })
    .SetDoc(R"DOC(
Applies 2-bit row-wise fake quantization to a tensor of floats.
The output looks like an int8 rowwise quantized blob with
scale and biases in half float.
)DOC")
    .Input(0, "input", "Float32 input data")
    .Output(0, "output", "Fused scale, bias and quantized data");
NO_GRADIENT(FloatToFused2BitFakeRowwiseQuantized);

REGISTER_CPU_OPERATOR(
    HalfToFused2BitFakeRowwiseQuantized,
    FloatToFusedNBitFakeRowwiseQuantizedOp<
        2,
        at::Half,
        internal::convertfp16fp32>);
OPERATOR_SCHEMA(HalfToFused2BitFakeRowwiseQuantized)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(1, X.dims(1) + 8);
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_UINT8);
      return out;
    })
    .SetDoc(R"DOC(
Applies 2-bit row-wise fake quantization to a tensor of half floats.
The output looks like an int8 rowwise quantized blob with
scale and biases in half float.
)DOC")
    .Input(0, "input", "Float16 input data")
    .Output(0, "output", "Fused scale, bias and quantized data");
NO_GRADIENT(HalfToFused2BitFakeRowwiseQuantized);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    FloatToFused2BitFakeRowwiseQuantized,
    GREEDY,
    FloatToFusedNBitFakeRowwiseQuantizedOp<
        2,
        float,
        internal::convertfp32fp32,
        true /* GREEDY */>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    HalfToFused2BitFakeRowwiseQuantized,
    GREEDY,
    FloatToFusedNBitFakeRowwiseQuantizedOp<
        2,
        at::Half,
        internal::convertfp16fp32,
        true /* GREEDY */>);

} // namespace caffe2
