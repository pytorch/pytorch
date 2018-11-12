#include "fully_connected_rowwise_dnnlowp_op.h"
#include <chrono>
#include <fbgemm/src/RefImplementations.h>

namespace caffe2 {

using namespace std;

template <typename T>
FullyConnectedRowWiseDNNLowPOp<T>::FullyConnectedRowWiseDNNLowPOp(
    const OperatorDef& operator_def,
    Workspace* ws)
    : BaseType(operator_def, ws),
      axis_(OperatorBase::GetSingleArgument<int32_t>("axis", 1)),
      axis_w_(OperatorBase::GetSingleArgument<int32_t>("axis_w", 1)),
      is_weight_constant_(
          OperatorBase::GetSingleArgument<bool>("constant_weight", true)) {

  using namespace dnnlowp;
  LOG(INFO) << "Using Rowwise Quantization!";
  if (!is_weight_constant_) {
    LOG(INFO) << operator_def.output(0) << " is_weight_constant "
              << is_weight_constant_;
    LOG(FATAL) << "rowwise quantization doesn't support nonconstant weights";
  }
}

template <typename T>
bool FullyConnectedRowWiseDNNLowPOp<T>::RunOnDevice() {
  using namespace std;
  using namespace dnnlowp;

  this->ParseDNNLowPOperatorArguments_();

  chrono::time_point<chrono::system_clock> t_very_begin, t_begin, t_end;

  if (VLOG_IS_ON(3)) {
    t_begin = chrono::system_clock::now();
    t_very_begin = t_begin;
  }

  // Get quantization parameters
  if (!GetQuantizationParameters_()) {
    return false;
  }

  if (VLOG_IS_ON(3)) {
    t_end = chrono::system_clock::now();
    double dt = chrono::duration<double>(t_end - t_begin).count();
    VLOG(3) << "@PERF this=" << this << " get_quant_params: " << dt * 1e3
            << " ms";
  }
  const auto& X = InputTensorCPU_(0);
  const auto& W = InputTensorCPU_(1);
  auto* Y = OutputTensorCPU_(0);
  const auto canonical_axis = X.canonical_axis_index(axis_);
  const auto M = X.size_to_dim(canonical_axis);
  const auto K = X.size_from_dim(canonical_axis);
  const auto canonical_axis_w = W.canonical_axis_index(axis_w_);
  const int N = W.size_to_dim(canonical_axis_w);
  const auto& b = InputTensorCPU_(2);

  Y_shape_cache_ = X.sizes().vec();
  Y_shape_cache_.resize(canonical_axis + 1);
  Y_shape_cache_[canonical_axis] = N;
  Y->Resize(Y_shape_cache_);
  Y_int32_.resize(Y->size());
  // Quantize X
  vector<T> X_temp;
  if (VLOG_IS_ON(3)) {
    t_begin = chrono::system_clock::now();
  }
  const T* Xdata = QuantizeInputIfNeeded<T>(
      this, 0, in_qparams_[0], X_temp, qfactory_.get());
  if (VLOG_IS_ON(3)) {
    t_end = chrono::system_clock::now();
    double dt = chrono::duration<double>(t_end - t_begin).count();
    VLOG(3) << "@PERF this=" << this << " input quantization: " << dt * 1e3
            << " ms";
  }

  const T_signed* Wdata = W_quantized_.data();

  if (VLOG_IS_ON(3)) {
    t_begin = chrono::system_clock::now();
  }
  if (Wq_packed_) {
    // fast path using fbgemm
    using namespace fbgemm;
    int row_offset_size_per_thread = M;
    int x_pack_buf_size_per_thread =
        PackAMatrix<uint8_t>::packedBufferSize();
    row_offsets_.resize(row_offset_size_per_thread);
    X_pack_buf_.resize(x_pack_buf_size_per_thread);

    DoNothing<int32_t, int32_t> doNothingObj{};
    memCopy<> memCopyObj(doNothingObj);

    PackAMatrix<uint8_t> packA(
        matrix_op_t::NoTranspose,
        M,
        K,
        reinterpret_cast<const uint8_t*>(Xdata),
        K,
        X_pack_buf_.data(),
        1, // group
        in_qparams_[0].zero_point);

    fbgemmPacked(
        packA,
        *Wq_packed_,
        Y_int32_.data(),
        Y_int32_.data(),
        N,
        memCopyObj,
        0, // thread_id
        1); // num_threads

    if (VLOG_IS_ON(3)) {
      t_end = chrono::system_clock::now();
      double dt = chrono::duration<double>(t_end - t_begin).count();
      VLOG(3) << "@PERF this=" << this << " gemm: " << dt * 1e3 << " ms";

      t_begin = chrono::system_clock::now();
    }

    row_offsets_u8acc32_ref(
        M,
        K,
        K,
        reinterpret_cast<const uint8_t*>(Xdata),
        row_offsets_.data());

    // Requantization
    // TODO: implement row-wise requantization output pipeline
    if (dequantize_output_) {
      const float* b_data = b.template data<float>();
      float* Ydata = OutputTensorCPU_(0)->template mutable_data<float>();
      for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
          Y_int32_[i * N + j] -=
              in_qparams_[0].zero_point * column_offsets_[j] +
              rowwise_qparams_[j].zero_point * row_offsets_[i];
          Y_int32_[i * N + j] += b_quantized_[j];
          Ydata[i * N + j] = Y_int32_[i * N + j] * rowwise_qparams_[j].scale *
                  in_qparams_[0].scale +
              b_data[j];
        }
      }
    } else {
      T* Ydata = GetQuantizedOutputData_();
      for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
          Y_int32_[i * N + j] -=
              in_qparams_[0].zero_point * column_offsets_[j] +
              rowwise_qparams_[j].zero_point * row_offsets_[i];
          Y_int32_[i * N + j] += b_quantized_[j];
          Ydata[i * N + j] = Requantize<T>(
              Y_int32_[i * N + j], rowwise_requantization_params_[j]);
        }
      }
    }
  } else {
    // slow path
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        int32_t sum = 0;
        for (int k = 0; k < K; ++k) {
          int w = (int)Wdata[j * K + k];
          sum += Xdata[i * K + k] * w;
        }
        Y_int32_[i * N + j] = sum;
      } // for each column
    } // for each row

    if (VLOG_IS_ON(3)) {
      t_end = chrono::system_clock::now();
      double dt = chrono::duration<double>(t_end - t_begin).count();
      VLOG(3) << "@PERF this=" << this << " gemm: " << dt * 1e3 << " ms";

      t_begin = chrono::system_clock::now();
    }

    // Requantization
    if (dequantize_output_) {
      const float* b_data = b.template data<float>();
      float* Ydata = OutputTensorCPU_(0)->template mutable_data<float>();
      for (int i = 0; i < M; ++i) {
        int32_t row_offset = 0;
        for (int k = 0; k < K; ++k) {
          row_offset += (int)Xdata[i * K + k];
        }
        for (int j = 0; j < N; ++j) {
          Y_int32_[i * N + j] -=
              in_qparams_[0].zero_point * column_offsets_[j] +
              rowwise_qparams_[j].zero_point * row_offset;
          Ydata[i * N + j] = Y_int32_[i * N + j] * rowwise_qparams_[j].scale *
                  in_qparams_[0].scale +
              b_data[j];
        }
      }
    } else {
      T* Ydata = GetQuantizedOutputData_();
      for (int i = 0; i < M; ++i) {
        int32_t row_offset = 0;
        for (int k = 0; k < K; ++k) {
          row_offset += (int)Xdata[i * K + k];
        }
        for (int j = 0; j < N; ++j) {
          Y_int32_[i * N + j] -=
              in_qparams_[0].zero_point * column_offsets_[j] +
              rowwise_qparams_[j].zero_point * row_offset;
          Y_int32_[i * N + j] += b_quantized_[j];
          Ydata[i * N + j] = Requantize<T>(
              Y_int32_[i * N + j], rowwise_requantization_params_[j]);
        }
      }
    }
  }

  if (!dequantize_output_) {
    RunOnDeviceEpilogue_();
  } else {
    this->MeasureQuantizationError_();
  }

  if (VLOG_IS_ON(3)) {
    t_end = chrono::system_clock::now();
    double dt = chrono::duration<double>(t_end - t_begin).count();
    VLOG(3) << "@PERF this=" << this
            << " bias-offset-requantization: " << dt * 1e3 << " ms";

    t_end = chrono::system_clock::now();
    double ops = 2. * M * N * K;
    dt = chrono::duration<double>(t_end - t_very_begin).count();
    double gops = ops / dt / 1e9;
    VLOG(3) << "@PERF this=" << this <<
               " output=" << OperatorBase::debug_def().output(0) << " " <<
               M << "x" << N << "x" << K << ": " <<
               dt * 1e3 << " ms " << gops << " gops";
  }

  return true;
}

template <typename T>
bool FullyConnectedRowWiseDNNLowPOp<T>::GetQuantizationParameters_() {
  using namespace dnnlowp;

  in_qparams_[0] = GetInputTensorQuantizationParamsOf(this, 0, qfactory_.get());

  const auto& W = InputTensorCPU_(1);
  const auto canonical_axis_w = W.canonical_axis_index(axis_w_);
  const auto N = W.size_to_dim(canonical_axis_w);
  const auto K = W.size_from_dim(canonical_axis_w);
  bool fast_path = is_same<T, uint8_t>::value && GetCpuId().avx2();
  if (is_weight_constant_) {
    if ((fast_path && !Wq_packed_) || (!fast_path && W_quantized_.empty())) {
      LOG(INFO) << "Choose rowwise quantization params";
      if (rowwise_qparams_.empty()) {
        // choose rowwise quantization params
        rowwise_qparams_.resize(N);
        W_quantized_.resize(W.size());
        for (int i = 0; i < N; ++i) {
          rowwise_qparams_[i] = qfactory_->ChooseQuantizationParams(
              W.template data<float>() + K * i,
              K,
              true /*weight*/);
          rowwise_qparams_[i].zero_point -=
              (1 << (qfactory_->GetWeightPrecision() - 1));
          Quantize<T_signed>(
              W.template data<float>() + K * i,
              W_quantized_.data() + K * i,
              K,
              rowwise_qparams_[i]);
        }
      }
      if (fast_path) {
        // fast path using fbgemm
        LOG(INFO)
            << "Using fast path with int8 fbgemm and generating Wq_packed_";
        Wq_packed_.reset(new fbgemm::PackBMatrix<int8_t>(
            fbgemm::matrix_op_t::Transpose,
            K,
            N,
            reinterpret_cast<const int8_t*>(W_quantized_.data()),
            K, // ld
            nullptr, // pmat
            1, // groups
            in_qparams_[1].zero_point));
      } else {
        LOG(WARNING)
            << "Falling back to slow path because fbgemm doesn't support "
               "this type or shape";
      }
    }
  } else {
    // !is_weigtht_constant
    LOG(WARNING) << "Not supporting nonconstant weights";
    in_qparams_[1] =
        GetInputTensorQuantizationParamsOf(this, 1, qfactory_.get());
    Quantize<T_signed>(
        W.template data<float>(),
        W_quantized_.data(),
        W_quantized_.size(),
        in_qparams_[1]);
    if (rowwise_qparams_.empty()) {
      rowwise_qparams_.resize(N);
      for (int i = 0; i < N; ++i) {
        rowwise_qparams_[i] = in_qparams_[1];
      }
    }
  }

  if (!is_weight_constant_ || column_offsets_.empty()) {
    // pre-compute column_offsets_
    column_offsets_.resize(N);
    for (int j = 0; j < N; ++j) {
      int32_t sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += W_quantized_[j * K + k];
      }
      column_offsets_[j] = sum - rowwise_qparams_[j].zero_point * K;
    }
  }

  if (Wq_packed_) {
    vector<T_signed>().swap(W_quantized_);
  }
  if (!is_weight_constant_ || b_quantized_.empty()) {
      // Quantize bias
      b_quantized_.resize(N);
      const auto& b = InputTensorCPU_(2);
      const float* b_data = b.template data<float>();
      for (int j = 0; j < N; ++j) {
        b_quantized_[j] = Quantize<int32_t>(
            b_data[j], 0, in_qparams_[0].scale * rowwise_qparams_[j].scale, 32);
      }
    }
  if (!dequantize_output_) {
    GetOutputQuantizationParams_();

    if (rowwise_requantization_params_.empty()) {
      // Choose requantization params
      rowwise_requantization_params_.resize(N);
      for (int i = 0; i < N; ++i) {
        float real_multiplier = in_qparams_[0].scale *
            rowwise_qparams_[i].scale / out_qparams_.scale;
        rowwise_requantization_params_[i] =
            qfactory_->ChooseRequantizationMultiplier(
                real_multiplier, out_qparams_);
      }
    }
  } else {
    if (measure_quantization_error_) {
      // to measure quantization error, run ref impl.
      Fp32Op_()->DequantizeInput();
      Fp32Op_()->Get()->RunOnDevice();
    }
  }

  return true;
}

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    FC,
    DNNLOWP_ROWWISE,
    FullyConnectedRowWiseDNNLowPOp<uint8_t>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    FC,
    DNNLOWP_ROWWISE_16,
    FullyConnectedRowWiseDNNLowPOp<uint16_t>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8FC,
    DNNLOWP_ROWWISE,
    FullyConnectedRowWiseDNNLowPOp<uint8_t>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8FCRowWise,
    DNNLOWP,
    FullyConnectedRowWiseDNNLowPOp<uint8_t>);

} // namespace caffe2
