#include "fully_connected_dnnlowp_op.h"

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
#include <chrono>
#endif

#include "caffe2/core/flags.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/fc_inference.h"
#include "caffe2/utils/cpuid.h"
#include "fbgemm_pack_matrix_cache.h"
#include "fbgemm_pack_op.h"
#include "mmio.h"

C10_DEFINE_bool(
    caffe2_dnnlowp_enforce_default_operators,
    false,
    "When true, enforce to use the default Caffe2 operators inside DNNLOWP"
    "instead of using its own implementation that uses AVX2 instructions"
    "(currently only honored by FC)");

C10_DECLARE_bool(caffe2_dnnlowp_dump_tensors);
C10_DECLARE_bool(caffe2_dnnlowp_force_slow_path);

namespace caffe2 {

using namespace std;

template <typename T, bool ReluFused>
FullyConnectedDNNLowPOp<T, ReluFused>::FullyConnectedDNNLowPOp(
    const OperatorDef& operator_def,
    Workspace* ws)
    : BaseType(operator_def, ws),
      axis_(this->template GetSingleArgument<int32_t>("axis", 1)),
      axis_w_(this->template GetSingleArgument<int32_t>("axis_w", 1)),
      quantize_channelwise_(this->template GetSingleArgument<bool>(
          "quantize_channelwise",
          false)),
      b_quantized_(make_shared<vector<int32_t>>()),
      column_offsets_(make_shared<vector<int32_t>>()),
      is_weight_constant_(
          this->template GetSingleArgument<bool>("constant_weight", true)) {
  if (!is_weight_constant_) {
    LOG(INFO) << operator_def.output(0) << " is_weight_constant "
              << is_weight_constant_;
  }
  if (this->debug_def().engine() == "DNNLOWP_ROWWISE" ||
      this->debug_def().engine() == "DNNLOWP_ROWWISE_16") {
    quantize_channelwise_ = true;
  }

  VLOG(2) << "DNNLOWP FC with output " << operator_def.output(0);
}

template <typename T, bool ReluFused>
bool FullyConnectedDNNLowPOp<T, ReluFused>::RunOnDevice() {
  using namespace std;
  using namespace dnnlowp;

  bool first_invocation = !this->arguments_parsed_;
  this->ParseDNNLowPOperatorArguments_();
  if (first_invocation && ReluFused) {
    followed_by_ = "Relu";
    AdjustOutputTensorQuantizationParamsWithFollowedBy(this, followed_by_);
  }

  if ((!GetCpuId().avx2() || FLAGS_caffe2_dnnlowp_enforce_default_operators) &&
      dequantize_output_) {
    if (!GetCpuId().avx2()) {
      static int log_occurences = 0;
      if (log_occurences < 32) {
        ++log_occurences;
        LOG(WARNING)
            << "Falling back to the default Caffe2 operator because AVX2 "
               "instruction is not available";
      }
    } else {
      static int log_occurences = 0;
      if (log_occurences < 32) {
        ++log_occurences;
        LOG(WARNING) << "Falling back to the default Caffe2 operator because "
                        "dnnlowp_enforce_default_caffe2_operators option is on";
      }
    }

    Fp32Op_()->DequantizeInput();
    FullyConnectedOp<CPUContext>* fp32_op = Fp32Op_()->Get();
    if (!fp32_op->RunOnDevice()) {
      return false;
    }

    auto* Y_ref = fp32_op->Output(0);
    auto* Y = OutputTensorCPU_(0);
    Y->ResizeLike(*Y_ref);
    fp32_op->context_.CopyItemsSameDevice(
        Y_ref->dtype(),
        Y_ref->size(),
        Y_ref->raw_data(),
        Y->raw_mutable_data(Y_ref->dtype()));
    return true;
  }

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
  chrono::time_point<chrono::system_clock> t_very_begin, t_begin, t_end;
  /* if (VLOG_IS_ON(3)) */
  {
    t_begin = chrono::system_clock::now();
    t_very_begin = t_begin;
  }
#endif

  // Get quantization parameters
  if (!GetQuantizationParameters_()) {
    return false;
  }

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
  /* if (VLOG_IS_ON(3)) */
  {
    t_end = chrono::system_clock::now();
    double dt = chrono::duration<double>(t_end - t_begin).count();
    LOG(INFO) << "@PERF this=" << this << " get_quant_params: " << dt * 1e3
              << " ms";
    t_begin = chrono::system_clock::now();
  }
#endif

  const auto& X = InputTensorCPU_(0);
  const auto& W = InputTensorCPU_(1);
  auto* Y = OutputTensorCPU_(0);
  const auto canonical_axis = X.canonical_axis_index(axis_);
  const auto M = X.size_to_dim(canonical_axis);
  const auto K = X.size_from_dim(canonical_axis);
  const auto canonical_axis_w = W.canonical_axis_index(axis_w_);
  const int N = W.size_to_dim(canonical_axis_w);

  if (M == 0) {
    LOG(WARNING) << "The batch size is 0!";
    return true;
  }

  const T_signed* Wdata = W_quantized_.data();

  Y_shape_cache_ = X.sizes().vec();
  Y_shape_cache_.resize(canonical_axis + 1);
  Y_shape_cache_[canonical_axis] = N;
  Y->Resize(Y_shape_cache_);

  const T* Xdata = nullptr;
  vector<T> X_temp;

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
  /* if (VLOG_IS_ON(1)) */
  {
    t_end = chrono::system_clock::now();
    double dt = chrono::duration<double>(t_end - t_begin).count();
    LOG(INFO) << "@PERF this=" << this << " initialize parameters: " << dt * 1e3
              << " ms";
    t_begin = chrono::system_clock::now();
  }
#endif

  if (Wq_packed_) {
    // fast path to use fbgemm
    using namespace fbgemm;

    if (X.template IsType<T>() || !dequantize_output_) {
      // Only when input and output are float, we don't need input to be
      // quantized.

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
      /* if (VLOG_IS_ON(1)) */
      { t_begin = chrono::system_clock::now(); }
#endif

      Xdata = QuantizeInputIfNeeded<T>(this, 0, in_qparams_[0], X_temp);

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
      /* if (VLOG_IS_ON(1)) */
      {
        t_end = chrono::system_clock::now();
        double dt = chrono::duration<double>(t_end - t_begin).count();
        LOG(INFO) << "@PERF this=" << this
                  << " input quantization: " << dt * 1e3 << " ms";
        t_begin = chrono::system_clock::now();
      }
#endif
    }

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
    /* if (VLOG_IS_ON(1)) */
    { t_begin = chrono::system_clock::now(); }
#endif

    if (!dequantize_output_) {
      Y_int32_.resize(Y->size());
      DoNothing<> doNothingObj{};

      if (quantize_channelwise_ || filter_qparams_[0].zero_point) {
        row_offsets_.resize(PackAWithRowOffset<uint8_t>::rowOffsetBufferSize());
        X_pack_buf_.resize(PackAWithRowOffset<uint8_t>::packedBufferSize());

        PackAWithRowOffset<uint8_t> packA(
            matrix_op_t::NoTranspose,
            M,
            K,
            reinterpret_cast<const uint8_t*>(Xdata),
            K,
            X_pack_buf_.data(), // buffer for packed matrix
            1, // group
            row_offsets_.data());

        if (quantize_channelwise_) {
          ReQuantizeOutput<ReluFused, QuantizationGranularity::OUT_CHANNEL>
              outputProcObj(
                  doNothingObj,
                  requantization_multipliers_.data(),
                  out_qparams_.zero_point,
                  column_offsets_->empty() ? 0 : in_qparams_[0].zero_point,
                  filter_zero_points_.data(),
                  packA.getRowOffsetBuffer(),
                  column_offsets_->empty() ? nullptr : column_offsets_->data(),
                  b_quantized_data_,
                  N);

          fbgemmPacked(
              packA,
              *Wq_packed_,
              reinterpret_cast<uint8_t*>(
                  OutputTensorCPU_(0)->template mutable_data<T>()),
              Y_int32_.data(),
              N,
              outputProcObj,
              0, // thread_id
              1); // num_threads
        } else {
          ReQuantizeOutput<ReluFused> outputProcObj(
              doNothingObj,
              requantization_multipliers_.data(),
              out_qparams_.zero_point,
              column_offsets_->empty() ? 0 : in_qparams_[0].zero_point,
              filter_zero_points_.data(),
              packA.getRowOffsetBuffer(),
              column_offsets_->empty() ? nullptr : column_offsets_->data(),
              b_quantized_data_,
              N);

          fbgemmPacked(
              packA,
              *Wq_packed_,
              reinterpret_cast<uint8_t*>(
                  OutputTensorCPU_(0)->template mutable_data<T>()),
              Y_int32_.data(),
              N,
              outputProcObj,
              0, // thread_id
              1); // num_threads
        }
      } else {
        X_pack_buf_.resize(PackAMatrix<uint8_t>::packedBufferSize());

        PackAMatrix<uint8_t> packA(
            matrix_op_t::NoTranspose,
            M,
            K,
            reinterpret_cast<const uint8_t*>(Xdata),
            K,
            X_pack_buf_.data(), // buffer for packed matrix
            1); // group

        ReQuantizeOutput<ReluFused> outputProcObj(
            doNothingObj,
            requantization_multipliers_.data(),
            out_qparams_.zero_point,
            column_offsets_->empty() ? 0 : in_qparams_[0].zero_point,
            filter_zero_points_.data(),
            nullptr,
            column_offsets_->empty() ? nullptr : column_offsets_->data(),
            b_quantized_data_,
            N);

        fbgemmPacked(
            packA,
            *Wq_packed_,
            reinterpret_cast<uint8_t*>(
                OutputTensorCPU_(0)->template mutable_data<T>()),
            Y_int32_.data(),
            N,
            outputProcObj,
            0, // thread_id
            1); // num_threads
      }
    } else {
      // dequantize_output
      float* Y_data = OutputTensorCPU_(0)->template mutable_data<float>();

      if (!X.template IsType<T>()) {
        // Both input and output are float
        row_offsets_.resize(
            PackAWithQuantRowOffset<uint8_t>::rowOffsetBufferSize());
        X_pack_buf_.resize(
            PackAWithQuantRowOffset<uint8_t>::packedBufferSize());
        PackAWithQuantRowOffset<uint8_t> packA(
            matrix_op_t::NoTranspose,
            M,
            K,
            X.template data<float>(),
            K,
            X_pack_buf_.data(), // buffer for packed matrix
            in_qparams_[0].scale,
            in_qparams_[0].zero_point,
            1, // groups
            row_offsets_.data());

        DoNothing<float, float> doNothingObj{};

        if (quantize_channelwise_) {
          ReQuantizeForFloat<ReluFused, QuantizationGranularity::OUT_CHANNEL>
              outputProcObj(
                  doNothingObj,
                  in_qparams_[0].scale,
                  filter_scales_.data(),
                  column_offsets_->empty() ? 0 : in_qparams_[0].zero_point,
                  filter_zero_points_.data(),
                  packA.getRowOffsetBuffer(),
                  column_offsets_->empty() ? nullptr : column_offsets_->data(),
                  b_dequantized_data_, // bias
                  N);

          fbgemmPacked(
              packA,
              *Wq_packed_,
              Y_data,
              reinterpret_cast<int32_t*>(Y_data),
              N,
              outputProcObj,
              0, // thread_id
              1); // num_threads
        } else {
          ReQuantizeForFloat<ReluFused> outputProcObj(
              doNothingObj,
              in_qparams_[0].scale,
              filter_scales_.data(),
              column_offsets_->empty() ? 0 : in_qparams_[0].zero_point,
              filter_zero_points_.data(),
              packA.getRowOffsetBuffer(),
              column_offsets_->empty() ? nullptr : column_offsets_->data(),
              b_dequantized_data_, // bias
              N);

          fbgemmPacked(
              packA,
              *Wq_packed_,
              Y_data,
              reinterpret_cast<int32_t*>(Y_data),
              N,
              outputProcObj,
              0, // thread_id
              1); // num_threads
        }
      } else {
        // Input quantized and output float
        row_offsets_.resize(PackAWithRowOffset<uint8_t>::rowOffsetBufferSize());
        X_pack_buf_.resize(PackAWithRowOffset<uint8_t>::packedBufferSize());
        PackAWithRowOffset<uint8_t> packA(
            matrix_op_t::NoTranspose,
            M,
            K,
            reinterpret_cast<const uint8_t*>(Xdata),
            K,
            X_pack_buf_.data(), // buffer for packed matrix
            1, // group
            row_offsets_.data());

        DoNothing<float, float> doNothingObj{};

        if (quantize_channelwise_) {
          ReQuantizeForFloat<ReluFused, QuantizationGranularity::OUT_CHANNEL>
              outputProcObj(
                  doNothingObj,
                  in_qparams_[0].scale,
                  filter_scales_.data(),
                  column_offsets_->empty() ? 0 : in_qparams_[0].zero_point,
                  filter_zero_points_.data(),
                  packA.getRowOffsetBuffer(),
                  column_offsets_->empty() ? nullptr : column_offsets_->data(),
                  b_dequantized_data_, // bias
                  N);

          fbgemmPacked(
              packA,
              *Wq_packed_,
              Y_data,
              reinterpret_cast<int32_t*>(Y_data),
              N,
              outputProcObj,
              0, // thread_id
              1); // num_threads
        } else {
          ReQuantizeForFloat<ReluFused> outputProcObj(
              doNothingObj,
              in_qparams_[0].scale,
              filter_scales_.data(),
              column_offsets_->empty() ? 0 : in_qparams_[0].zero_point,
              filter_zero_points_.data(),
              packA.getRowOffsetBuffer(),
              column_offsets_->empty() ? nullptr : column_offsets_->data(),
              b_dequantized_data_, // bias
              N);

          fbgemmPacked(
              packA,
              *Wq_packed_,
              Y_data,
              reinterpret_cast<int32_t*>(Y_data),
              N,
              outputProcObj,
              0, // thread_id
              1); // num_threads
        }
      }
    } // dequantize_output
  } else {
    // Quantize X

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
    /* if (VLOG_IS_ON(1)) */
    { t_begin = chrono::system_clock::now(); }
#endif

    Xdata = QuantizeInputIfNeeded<T>(this, 0, in_qparams_[0], X_temp);

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
    /* if (VLOG_IS_ON(1)) */
    {
      t_end = chrono::system_clock::now();
      double dt = chrono::duration<double>(t_end - t_begin).count();
      LOG(INFO) << "@PERF this=" << this << " input quantization: " << dt * 1e3
                << " ms";
      t_begin = chrono::system_clock::now();
    }
#endif

// #define DNNLOWP_DETAILED_LOG_IN_SLOW_PATH
#ifdef DNNLOWP_DETAILED_LOG_IN_SLOW_PATH
    int overflow_cnt = 0, underflow_cnt = 0;
#endif

    Y_int32_.resize(Y->size());
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        int32_t sum = 0;
        for (int k = 0; k < K; ++k) {
          int x = Xdata[i * K + k];
          int w = Wdata[j * K + k];
          sum += x * w;
#ifdef DNNLOWP_DETAILED_LOG_IN_SLOW_PATH
          if (k < K - 1) {
            int x2 = Xdata[i * K + k + 1];
            int w2 = Wdata[j * K + k + 1];
            int sum2 = x * w + x2 * w2;
            bool overflowed = false, underflowed = false;
            if (sum2 > numeric_limits<int16_t>::max()) {
              overflowed = true;
              ++overflow_cnt;
            } else if (sum2 < numeric_limits<int16_t>::min()) {
              underflowed = true;
              ++underflow_cnt;
            }
            if (overflowed || underflowed) {
              LOG(INFO) << "i " << i << " j " << j << " k " << k << " " << x
                        << " * " << w << " + " << x2 << " * " << w2 << " = "
                        << sum2;
            }
          }
#endif
        }
        Y_int32_[i * N + j] = sum;
      } // for each output element
    } // for each row

    // Expose the quantized X, W and Y for debugging if debug outputs are
    // attached to the operator and caffe2_dnnlowp_force_slow_path flag is set
    if (OutputSize() == 4) {
      auto* X_q = OutputTensorCPU_(1);
      auto* W_q = OutputTensorCPU_(2);
      auto* Y_q = OutputTensorCPU_(3);

      X_q->Resize(std::vector<std::int64_t>{M, K});
      W_q->Resize(std::vector<std::int64_t>{N, K});
      Y_q->Resize(std::vector<std::int64_t>{M, N});

      float* X_q_data = X_q->template mutable_data<float>();
      float* W_q_data = W_q->template mutable_data<float>();
      float* Y_q_data = Y_q->template mutable_data<float>();

      size_t X_size = M * K;
      size_t W_size = N * K;
      size_t Y_size = M * N;
      for (int i = 0; i < X_size; i++) {
        X_q_data[i] = Xdata[i];
      }
      for (int i = 0; i < W_size; i++) {
        W_q_data[i] = Wdata[i];
      }
      for (int i = 0; i < Y_size; i++) {
        Y_q_data[i] = Y_int32_[i];
      }
    }

#ifdef DNNLOWP_DETAILED_LOG_IN_SLOW_PATH
    LOG(INFO) << "underflow_cnt " << underflow_cnt << " ("
              << static_cast<float>(underflow_cnt) / (M * N * K) * 100
              << ") overflow_cnt " << overflow_cnt << " ("
              << static_cast<float>(overflow_cnt) / (M * N * K) * 100 << ")";
#endif
  }

  if (FLAGS_caffe2_dnnlowp_dump_tensors) {
    // Dump input activation
    StoreMatrixInMatrixMarketFormat(M, K, Xdata, this->debug_def().input(0));

    // Dump weight
    StoreMatrixInMatrixMarketFormat(N, K, Wdata, this->debug_def().input(1));
  }

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
  /* if (VLOG_IS_ON(1)) */
  {
    t_end = chrono::system_clock::now();
    double dt = chrono::duration<double>(t_end - t_begin).count();
    LOG(INFO) << "@PERF this=" << this << " gemm: " << dt * 1e3 << " ms";

    t_begin = chrono::system_clock::now();
  }
#endif

  // Adjust with bias and zero_point and then requantize
  // See batch_matmul_dnnlowp_op.cc to why we compute column_offsets,
  // row_offset, and const_offset in this way.
  if (dequantize_output_) {
    if (!Wq_packed_) {
      float* Ydata = OutputTensorCPU_(0)->template mutable_data<float>();

      for (int i = 0; i < M; ++i) {
        int32_t row_offset = 0;
        for (int k = 0; k < K; ++k) {
          row_offset += Xdata[i * K + k];
        }

        for (int j = 0; j < N; ++j) {
          if (!column_offsets_->empty()) {
            Y_int32_[i * N + j] -=
                in_qparams_[0].zero_point * (*column_offsets_)[j];
          }
          int quant_group = quantize_channelwise_ ? j : 0;
          Y_int32_[i * N + j] -=
              row_offset * filter_qparams_[quant_group].zero_point;
          Ydata[i * N + j] = Y_int32_[i * N + j] * in_qparams_[0].scale *
                  filter_qparams_[quant_group].scale +
              b_dequantized_data_[j];
          if (ReluFused) {
            Ydata[i * N + j] = std::max(Ydata[i * N + j], 0.0f);
          }
        }
      }
    }
  } else {
    if (!Wq_packed_) {
      T* Ydata = GetQuantizedOutputData_();
      for (int i = 0; i < M; ++i) {
        int32_t row_offset = 0;
        for (int k = 0; k < K; ++k) {
          row_offset += Xdata[i * K + k];
        }

        for (int j = 0; j < N; ++j) {
          if (!column_offsets_->empty()) {
            // empty column offset means it's folded into bias
            Y_int32_[i * N + j] -=
                in_qparams_[0].zero_point * (*column_offsets_)[j];
          }
          int quant_group = quantize_channelwise_ ? j : 0;
          Y_int32_[i * N + j] -=
              row_offset * filter_qparams_[quant_group].zero_point;
          Y_int32_[i * N + j] += b_quantized_data_[j];

          Ydata[i * N + j] = fbgemm::Requantize<T>(
              Y_int32_[i * N + j], requantization_params_[quant_group]);
          if (ReluFused) {
            Ydata[i * N + j] =
                std::max<T>(out_qparams_.zero_point, Ydata[i * N + j]);
          }
        }
      }
    }

    PropagateOutputTensorQuantizationParams(this, 0, out_qparams_);
  }

  MeasureQuantizationError_();

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
  /* if (VLOG_IS_ON(1)) */
  {
    t_end = chrono::system_clock::now();
    double dt = chrono::duration<double>(t_end - t_begin).count();
    LOG(INFO) << "@PERF this=" << this
              << " bias-offset-requantization: " << dt * 1e3 << " ms";

    t_end = chrono::system_clock::now();
    double ops = 2. * M * N * K;
    dt = chrono::duration<double>(t_end - t_very_begin).count();
    double gops = ops / dt / 1e9;
    LOG(INFO) << "@PERF this=" << this
              << " output=" << this->debug_def().output(0) << " " << M << "x"
              << N << "x" << K << ": " << dt * 1e3 << " ms " << gops << " gops";
  }
#endif

  return true;
}

template <typename T, bool ReluFused>
bool FullyConnectedDNNLowPOp<T, ReluFused>::GetQuantizationParameters_() {
  using namespace dnnlowp;

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
  chrono::time_point<chrono::system_clock> t_begin, t_end;
  /* if (VLOG_IS_ON(1)) */
  { t_begin = chrono::system_clock::now(); }
#endif

  // Choose quantization for X
  in_qparams_[0] = GetInputTensorQuantizationParamsOf(this, 0, qfactory_.get());

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
  /* if (VLOG_IS_ON(1)) */
  {
    t_end = chrono::system_clock::now();
    double dt = chrono::duration<double>(t_end - t_begin).count();
    LOG(INFO) << "@PERF this=" << this << " GetInputTensorQuantizationParamsOf "
              << dt * 1e3 << " ms";
    t_begin = chrono::system_clock::now();
  }
#endif

  // Quantize W
  const auto& X = InputTensorCPU_(0);
  const auto& W = InputTensorCPU_(1);
  const auto canonical_axis = X.canonical_axis_index(axis_);
  const auto K = X.size_from_dim(canonical_axis);
  const auto canonical_axis_w = W.canonical_axis_index(axis_w_);
  const int N = W.size_to_dim(canonical_axis_w);

  int signed_min = -(1 << (qfactory_->GetWeightPrecision() - 1));
  if (is_weight_constant_) {
    bool fast_path = is_same<T, uint8_t>::value && GetCpuId().avx2() &&
        this->debug_def().engine() != "DNNLOWP_ACC16" &&
        !FLAGS_caffe2_dnnlowp_force_slow_path;

    if ((fast_path && !Wq_packed_) || (!fast_path && W_quantized_.empty())) {
      if (this->template InputIsType<Int8FCDNNLowPPackedWeightBlob>(1)) {
        const auto& packed_filter =
            this->template Input<Int8FCDNNLowPPackedWeightBlob>(1);
        filter_qparams_ = packed_filter.qparams;
        if (quantize_channelwise_) {
          CAFFE_ENFORCE_EQ(filter_qparams_.size(), N);
        } else {
          CAFFE_ENFORCE_EQ(filter_qparams_.size(), 1);
        }
      } else {
        filter_qparams_.resize(quantize_channelwise_ ? N : 1);
        QuantizeWeight<T>(
            InputBlob(1), K, N, filter_qparams_, W_quantized_, qfactory_.get());
      }

      filter_scales_.resize(filter_qparams_.size());
      filter_zero_points_.resize(filter_qparams_.size());
      requantization_params_.resize(filter_qparams_.size());
      requantization_multipliers_.resize(filter_qparams_.size());
      for (int i = 0; i < filter_qparams_.size(); ++i) {
        filter_scales_[i] = filter_qparams_[i].scale;
        filter_zero_points_[i] = filter_qparams_[i].zero_point;
      }

      if (fast_path) {
        // fast path using fbgemm
        if (this->template InputIsType<Int8FCDNNLowPPackedWeightBlob>(1)) {
          const auto& packed_filter =
              this->template Input<Int8FCDNNLowPPackedWeightBlob>(1);
          Wq_packed_ = packed_filter.W;
        } else {
          Wq_packed_ = GetOrCreateFbgemmPackBMatrix<int32_t>(
              fbgemm::matrix_op_t::Transpose,
              K,
              N,
              W.raw_data(),
              reinterpret_cast<const int8_t*>(W_quantized_.data()),
              K); // ld
        }
      } else {
        string reason;
        if (!is_same<T, uint8_t>::value) {
          reason = "fbgemm only supports 8-bit integers";
        } else if (!GetCpuId().avx2()) {
          reason = "fbgemm only supports AVX2";
        } else if (this->debug_def().engine() == "DNNLOWP_ACC16") {
          reason = "";
        } else if (FLAGS_caffe2_dnnlowp_force_slow_path) {
          reason = "slow path enforced";
        } else {
          assert(false);
        }
        if (!reason.empty()) {
          LOG(WARNING) << "Conv with weight " << this->debug_def().input(1)
                       << " falls back to slow path because " << reason;
        }
      }
    }
  } // is_weight_constant_
  else {
    // !is_weight_constant_
    filter_qparams_.resize(1);
    filter_qparams_[0] = GetInputTensorQuantizationParamsOf(
        this, 1, qfactory_.get(), true /*weight*/);
    filter_qparams_[0].zero_point += signed_min;

    W_quantized_.resize(W.size());
    fbgemm::Quantize<T_signed>(
        W.template data<float>(),
        W_quantized_.data(),
        W_quantized_.size(),
        filter_qparams_[0]);
  }

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
  /* if (VLOG_IS_ON(1)) */
  {
    t_end = chrono::system_clock::now();
    double dt = chrono::duration<double>(t_end - t_begin).count();
    LOG(INFO) << "@PERF this=" << this << " Quantize W " << dt * 1e3 << " ms";
    t_begin = chrono::system_clock::now();
  }
#endif

  // Pre-compute column_offset
  // If input tensor doesn't use dynamic quantization, we fold column_offsets_
  // into bias.
  bool first_invocation = !b_quantized_data_ && !b_dequantized_data_;
  bool fold_col_offset_into_bias =
      this->template InputIsType<int8::Int8TensorCPU>(0) && !dequantize_output_;
  if (!is_weight_constant_ ||
      (first_invocation && !fold_col_offset_into_bias)) {
    if (this->template InputIsType<Int8FCDNNLowPPackedWeightBlob>(1)) {
      const auto& packed_filter =
          this->template Input<Int8FCDNNLowPPackedWeightBlob>(1);
      column_offsets_ = packed_filter.column_offsets;
    } else {
      ComputeColumnOffsets<T_signed>(
          K, N, W_quantized_.data(), filter_qparams_, *column_offsets_);
    }
  }

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
  /* if (VLOG_IS_ON(1)) */
  {
    t_end = chrono::system_clock::now();
    double dt = chrono::duration<double>(t_end - t_begin).count();
    LOG(INFO) << "@PERF this=" << this << " Calculate column offset "
              << dt * 1e3 << " ms";
    t_begin = chrono::system_clock::now();
  }
#endif

  // Quantize bias
  if (!is_weight_constant_ || (!b_quantized_data_ && !b_dequantized_data_) ||
      in_qparams_[0].scale != in_qparams0_scale_old_ ||
      in_qparams_[0].zero_point != in_qparams0_zero_point_old_) {
    if (this->template InputIsType<Int8FCDNNLowPPackedWeightBlob>(1) &&
        this->template Input<Int8FCDNNLowPPackedWeightBlob>(1).bias.get()) {
      const auto& packed_filter =
          this->template Input<Int8FCDNNLowPPackedWeightBlob>(1);
      CAFFE_ENFORCE(!dequantize_output_);
      b_quantized_data_ = packed_filter.bias->data();
    } else {
      const auto& bias = InputTensorCPU_(2);
      if (this->template InputIsType<int8::Int8TensorCPU>(2)) {
        TensorQuantizationParams bias_qparams;
        bias_qparams.scale = this->template Input<int8::Int8TensorCPU>(2).scale;
        bias_qparams.zero_point =
            this->template Input<int8::Int8TensorCPU>(2).zero_point;
        CAFFE_ENFORCE_LE(
            std::abs(
                bias_qparams.scale -
                in_qparams_[0].scale * filter_qparams_[0].scale),
            1e-4);
        CAFFE_ENFORCE_EQ(bias_qparams.zero_point, 0);
        b_quantized_data_ = bias.template data<int32_t>();
        if (dequantize_output_) {
          b_dequantized_.resize(N);
          for (int j = 0; j < N; ++j) {
            b_dequantized_[j] = fbgemm::Dequantize<int32_t>(
                b_quantized_data_[j], in_qparams_[2]);
          }
          b_dequantized_data_ = b_dequantized_.data();
        }
      } else {
        b_dequantized_data_ = bias.template data<float>();
        if (!dequantize_output_) {
          b_quantized_->resize(N);
          for (int j = 0; j < N; ++j) {
            (*b_quantized_)[j] = fbgemm::Quantize<int32_t>(
                b_dequantized_data_[j],
                0,
                in_qparams_[0].scale * filter_qparams_[0].scale,
                32);
          }
          b_quantized_data_ = b_quantized_->data();
        }
      }
    }
    in_qparams0_scale_old_ = in_qparams_[0].scale;
    in_qparams0_zero_point_old_ = in_qparams_[0].zero_point;

    // If column_offsets_ is empty even when we need column_offsets (asymmetric
    // quantization in input), it means we need to fuse column_offsets to bias.
    if (in_qparams_[0].zero_point && column_offsets_->empty() &&
        b_quantized_data_) {
      if (b_quantized_->empty()) {
        // When b_quantized_data_ is from pre-packed bias or Int8TensorCPU,
        // we can't inplace modify so copy to internal b_quantized_ vector.
        b_quantized_->assign(b_quantized_data_, b_quantized_data_ + N);
        b_quantized_data_ = b_quantized_->data();
      }
      vector<int32_t>* column_offset_ptr;
      vector<int32_t> column_offset_temp;
      if (this->template InputIsType<Int8FCDNNLowPPackedWeightBlob>(1)) {
        const auto& packed_filter =
            this->template Input<Int8FCDNNLowPPackedWeightBlob>(1);
        column_offset_ptr = packed_filter.column_offsets.get();
      } else {
        column_offset_temp.resize(N);
        ComputeColumnOffsets<T_signed>(
            K, N, W_quantized_.data(), filter_qparams_, column_offset_temp);
        column_offset_ptr = &column_offset_temp;
      }
      for (int i = 0; i < N; ++i) {
        (*b_quantized_)[i] -=
            in_qparams_[0].zero_point * (*column_offset_ptr)[i];
      }
    }

    CAFFE_ENFORCE(
        (dequantize_output_ && b_dequantized_data_) ||
        (!dequantize_output_ && b_quantized_data_));
  }

  if (Wq_packed_ && !FLAGS_caffe2_dnnlowp_dump_tensors) {
    // From here, W_quantized_ is not used anymore when we have Wq_packed_
    vector<T_signed>().swap(W_quantized_);
  }

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
  /* if (VLOG_IS_ON(1)) */
  {
    t_end = chrono::system_clock::now();
    double dt = chrono::duration<double>(t_end - t_begin).count();
    LOG(INFO) << "@PERF this=" << this << " Quantize bias " << dt * 1e3
              << " ms";
    t_begin = chrono::system_clock::now();
  }
#endif

  if (!dequantize_output_ && !requantization_param_selected_) {
    GetOutputQuantizationParams_();

    for (int i = 0; i < filter_qparams_.size(); ++i) {
      float real_multiplier =
          in_qparams_[0].scale * filter_qparams_[i].scale / out_qparams_.scale;
      requantization_params_[i] = qfactory_->ChooseRequantizationMultiplier(
          real_multiplier, out_qparams_);
      requantization_multipliers_[i] =
          requantization_params_[i].real_multiplier;
    }
    requantization_param_selected_ = true;
  } else {
    if (measure_quantization_error_) {
      // to measure quantization error, run ref impl.
      Fp32Op_()->DequantizeInput();
      Fp32Op_()->Get()->RunOnDevice();
    }
  }

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
  /* if (VLOG_IS_ON(1)) */
  {
    t_end = chrono::system_clock::now();
    double dt = chrono::duration<double>(t_end - t_begin).count();
    LOG(INFO) << "@PERF this=" << this << " GetOutputQuantizationParams "
              << dt * 1e3 << " ms";
    t_begin = chrono::system_clock::now();
  }
#endif

  return true;
}

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    FC,
    DNNLOWP,
    FullyConnectedDNNLowPOp<uint8_t>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    FC,
    DNNLOWP_16,
    FullyConnectedDNNLowPOp<uint16_t>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8FC,
    DNNLOWP,
    FullyConnectedDNNLowPOp<uint8_t>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    FC,
    DNNLOWP_ROWWISE,
    FullyConnectedDNNLowPOp<uint8_t>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    FC,
    DNNLOWP_ROWWISE_16,
    FullyConnectedDNNLowPOp<uint16_t>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8FC,
    DNNLOWP_ROWWISE,
    FullyConnectedDNNLowPOp<uint8_t>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8FCRelu,
    DNNLOWP,
    FullyConnectedDNNLowPOp<uint8_t, true>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8FCRelu,
    DNNLOWP_ROWWISE,
    FullyConnectedDNNLowPOp<uint8_t, true>);

using namespace std::placeholders;
OPERATOR_SCHEMA(Int8FCRelu)
    .NumInputs(3)
    .NumOutputs(1)
    .TensorInferenceFunction(std::bind(FCShapeInference, _1, _2, false))
    .CostInferenceFunction(std::bind(CostInferenceForFC, _1, _2, false));

} // namespace caffe2
