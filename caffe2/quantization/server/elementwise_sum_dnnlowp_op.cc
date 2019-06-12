#include "utility_dnnlowp_ops.h"

// #define DNNLOWP_MEASURE_TIME_BREAKDOWN
#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
#include <chrono>
#endif

#include "caffe2/utils/cpuid.h"
#include "dnnlowp_partition.h"

namespace caffe2 {

using namespace std;

template <typename T, bool ReluFused>
SumDNNLowPOp<T, ReluFused>::SumDNNLowPOp(
    const OperatorDef& operator_def, Workspace* ws)
  : BaseType(operator_def, ws) {}

template <typename T, bool ReluFused>
bool SumDNNLowPOp<T, ReluFused>::RunOnDevice() {
  if (!this->arguments_parsed_) {
    dnnlowp::ParseDNNLowPOperatorArguments(
        this,
        &dequantize_output_,
        &measure_quantization_error_,
        &followed_by_);

    if (ReluFused) {
      // It's actually fused with Relu not followed by but setting this to make
      // sure quantization error is correctly measured in
      // this->MeasureQuantizationError_
      followed_by_ = "Relu";
      dnnlowp::AdjustOutputTensorQuantizationParamsWithFollowedBy(
          this, followed_by_);
    }
    this->arguments_parsed_ = true;
  }

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
  chrono::time_point<chrono::system_clock> t_begin, t_end;

  t_begin = chrono::system_clock::now();
#endif

  if (!GetQuantizationParameters_()) {
    return false;
  }

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
  t_end = chrono::system_clock::now();
  double dt = chrono::duration<double>(t_end - t_begin).count();
  LOG(INFO) << "this=" << this << " get_quant_params: " << dt * 1e3 << " ms";

  t_begin = chrono::system_clock::now();
#endif

  using namespace dnnlowp;
  // Quantize inputs
  int len = InputTensorCPU_(0).size();

  // Element-wise sum
  int32_t intermediate_zero_point =
    intermediate_qparams_.zero_point * InputSize();

  auto* output = OutputTensorCPU_(0);
  output->ResizeLike(InputTensorCPU_(0));

  T *output_data = GetQuantizedOutputData_();

  if (InputTensorCPU_(0).template IsType<T>()) {
    if (InputSize() == 2 && is_same<T, uint8_t>::value &&
        GetCpuId().avx2() && GetCpuId().fma()) {
      // fast path when we have 2 uint8_t inputs with AVX2 / FMA support
      array<const T*, 2> input_data;
      for (int i = 0; i < 2; ++i) {
        input_data[i] = InputTensorCPU_(i).template data<T>();
      }

      // TODO: this intrinsic code is replicated in dnnlowp.cc,
      // fbgemm_i8i8_acc32.cc, conv_dnnlowp_op.cc, and here.
      // We need to somehow refactor this.
      __m256 min_v = _mm256_set1_ps(numeric_limits<uint8_t>::min());
      __m256 max_v = _mm256_set1_ps(numeric_limits<uint8_t>::max());

      __m256i shuffle_mask_v = _mm256_set_epi8(
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0x0c, 0x08, 0x04, 0x00,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0x0c, 0x08, 0x04, 0x00);
      __m256i permute_mask_v = _mm256_set_epi32(
          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00);

#ifdef _OPENMP
#pragma omp parallel
#endif
      {
        constexpr int VLEN = 8;

        int j_begin, j_end;
        tie(j_begin, j_end) = Get1DPartition(
            len,
            dnnlowp_get_num_threads(),
            dnnlowp_get_thread_num(),
            VLEN);

        int j = j_begin;
        int j_end_aligned = j_begin + (j_end - j_begin) / VLEN * VLEN;
        for ( ; j < j_end_aligned; j += VLEN) {
          // Input is uint8_t but cvtepi8_epi32 assumes the input is int8_t,
          // so we subtract 0x80, cvtepi8_epi32, and then add 0x80
          __m256 in_v = _mm256_cvtepi32_ps(_mm256_add_epi32(
              _mm256_cvtepi8_epi32(_mm_sub_epi8(
                  _mm_loadl_epi64(
                      reinterpret_cast<const __m128i*>(input_data[0] + j)),
                  _mm_set1_epi8(0x80))),
              _mm256_set1_epi32(0x80)));
          in_v = _mm256_fmadd_ps(
              in_v,
              _mm256_set1_ps(in_qparams_[0].scale),
              _mm256_set1_ps(
                  -in_qparams_[0].zero_point * in_qparams_[0].scale));
          __m256 acc_v = in_v;

          in_v = _mm256_cvtepi32_ps(_mm256_add_epi32(
              _mm256_cvtepi8_epi32(_mm_sub_epi8(
                  _mm_loadl_epi64(
                      reinterpret_cast<const __m128i*>(input_data[1] + j)),
                  _mm_set1_epi8(0x80))),
              _mm256_set1_epi32(0x80)));
          in_v = _mm256_fmadd_ps(
              in_v,
              _mm256_set1_ps(in_qparams_[1].scale),
              _mm256_set1_ps(
                  -in_qparams_[1].zero_point * in_qparams_[1].scale));
          acc_v = _mm256_add_ps(acc_v, in_v);

          __m256 transformed_v = _mm256_fmadd_ps(
            acc_v, _mm256_set1_ps(1.0 / out_qparams_.scale),
            _mm256_set1_ps(out_qparams_.zero_point));
          __m256 clipped_v = _mm256_min_ps(
              _mm256_max_ps(
                  transformed_v,
                  ReluFused ? _mm256_set1_ps(out_qparams_.zero_point) : min_v),
              max_v);
          __m256i rounded_v = _mm256_cvtps_epi32(clipped_v);
          rounded_v = _mm256_shuffle_epi8(rounded_v, shuffle_mask_v);
          rounded_v = _mm256_permutevar8x32_epi32(rounded_v, permute_mask_v);
          *reinterpret_cast<int64_t*>(output_data + j) =
              _mm256_extract_epi64(rounded_v, 0);
        }
        for ( ; j < j_end; ++j) {
          float acc = 0;
          for (int i = 0; i < 2; ++i) {
            acc += (input_data[i][j] - in_qparams_[i].zero_point) *
                   in_qparams_[i].scale;
          }
          float transformed_val =
            out_qparams_.zero_point + acc / out_qparams_.scale;
          output_data[j] = std::max(
              ReluFused ? out_qparams_.zero_point : 0.0f,
              std::min(255.0f, nearbyint(transformed_val)));
        }
      } // omp parallel
    }
    else {
      RequantizationParams in_requantization_params[InputSize()];
      const T *input_data[InputSize()];
      for (int i = 0; i < InputSize(); ++i) {
        float real_multiplier =
          in_qparams_[i].scale / intermediate_qparams_.scale;
        in_requantization_params[i] =
          qfactory_->ChooseRequantizationMultiplier(
            real_multiplier, intermediate_qparams_);
        input_data[i] = InputTensorCPU_(i).template data<T>();
      }

#ifdef _OPENMP
#pragma omp parallel
#endif
      {
        int j_begin, j_end;
        tie(j_begin, j_end) = Get1DPartition(
            len, dnnlowp_get_num_threads(), dnnlowp_get_thread_num());

        for (int j = j_begin; j < j_end; ++j) {
          int32_t acc = 0;
          for (int i = 0; i < InputSize(); ++i) {
            acc += Requantize<int32_t>(
              input_data[i][j] - in_qparams_[i].zero_point,
              in_requantization_params[i]);
          }
          int32_t raw = acc - intermediate_zero_point;
          if (ReluFused) {
            raw = std::max(0, raw);
          }
          output_data[j] = Requantize<T>(raw, out_requantization_params_);
        }
      }
    }
  } // InputTensorCPU_(0).template IsType<T>()
  else {
    const float *input_data[InputSize()];
    for (int i = 0; i < InputSize(); ++i) {
      input_data[i] = InputTensorCPU_(i).template data<float>();
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int j_begin, j_end;
      tie(j_begin, j_end) = Get1DPartition(
          len, dnnlowp_get_num_threads(), dnnlowp_get_thread_num());

      for (int j = j_begin; j < j_end; ++j) {
        int32_t acc = 0;
        for (int i = 0; i < InputSize(); ++i) {
          acc += Quantize<int32_t>(
            ((const float *)input_data[i])[j],
            intermediate_qparams_.zero_point, intermediate_qparams_.scale,
            qfactory_->GetEltwiseQuantizePrecision());
        }
        int32_t raw = acc - intermediate_zero_point;
        if (ReluFused) {
          raw = std::max(0, raw);
        }
        output_data[j] = Requantize<T>(raw, out_requantization_params_);
      }
    }
  } // !InputTensorCPU_(0).template IsType<T>()

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
  t_end = chrono::system_clock::now();
  dt = chrono::duration<double>(t_end - t_begin).count();
  LOG(INFO) << "this=" << this << " requantize inputs: " << dt * 1e3 << " ms";

  t_begin = chrono::system_clock::now();
#endif

  RunOnDeviceEpilogue_();

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
  t_end = chrono::system_clock::now();
  dt = chrono::duration<double>(t_end - t_begin).count();
  LOG(INFO) << "this=" << this << " prologue: " << dt * 1e3 << " ms";

  t_begin = chrono::system_clock::now();
#endif

  return true;
} // DoRunQuantizedWithType_

template <typename T, bool ReluFused>
bool SumDNNLowPOp<T, ReluFused>::GetQuantizationParameters_() {
  using namespace dnnlowp;

  // Find global min and max of all inputs
  float
    global_min = numeric_limits<float>::max(),
    global_max = numeric_limits<float>::lowest();

  for (int i = 0; i < InputSize(); ++i) {
    in_qparams_[i] =
      GetInputTensorQuantizationParamsOf(this, i, qfactory_.get());

    global_min = std::min(global_min, in_qparams_[i].Min());
    global_max = std::max(global_max, in_qparams_[i].Max());
  }

  intermediate_qparams_ =
    qfactory_->ChooseQuantizationParams(
      global_min, global_max,
      qfactory_->GetEltwiseQuantizePrecision(),
      qfactory_->GetPreserveActivationSparsity());

  GetOutputQuantizationParams_();

  // requantize from the intermediate precision to the final precision
  float real_multiplier = intermediate_qparams_.scale / out_qparams_.scale;
  out_requantization_params_ = qfactory_->ChooseRequantizationMultiplier(
    real_multiplier, out_qparams_);

  return true;
}

OPERATOR_SCHEMA(SumRelu)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .InputsCanCrossDevices()
    .IdenticalTypeAndShapeOfInput(0)
    .Input(0, "data_0", "First of the input tensors. Can be inplace.")
    .Output(0, "sum", "Output tensor. Same dimension as inputs.");

REGISTER_CPU_OPERATOR_WITH_ENGINE(
  Sum, DNNLOWP, SumDNNLowPOp<uint8_t, false>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
  SumRelu, DNNLOWP, SumDNNLowPOp<uint8_t, true>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
  Int8Sum, DNNLOWP, SumDNNLowPOp<uint8_t, false>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
  Int8SumRelu, DNNLOWP, SumDNNLowPOp<uint8_t, true>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
  Sum, DNNLOWP_16, SumDNNLowPOp<uint16_t, false>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
  SumRelu, DNNLOWP_16, SumDNNLowPOp<uint16_t, true>);

} // namespace caffe2
