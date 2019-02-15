#include "fully_connected_dnnlowp_op.h"

#include <chrono>

#include "caffe2/core/flags.h"
#include "caffe2/core/tensor_int8.h"
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

namespace caffe2 {

using namespace std;

template <typename T>
FullyConnectedDNNLowPOp<T>::FullyConnectedDNNLowPOp(
    const OperatorDef& operator_def,
    Workspace* ws)
    : BaseType(operator_def, ws),
      axis_(this->template GetSingleArgument<int32_t>("axis", 1)),
      axis_w_(this->template GetSingleArgument<int32_t>("axis_w", 1)),
      b_quantized_(make_shared<vector<int32_t>>()),
      column_offsets_(make_shared<vector<int32_t>>()),
      is_weight_constant_(
          this->template GetSingleArgument<bool>("constant_weight", true)) {
  if (!is_weight_constant_) {
    LOG(INFO) << operator_def.output(0) << " is_weight_constant "
              << is_weight_constant_;
  }

  VLOG(2) << "DNNLOWP FC with output " << operator_def.output(0);
}

template <typename T>
bool FullyConnectedDNNLowPOp<T>::RunOnDevice() {
  using namespace std;
  using namespace dnnlowp;

  this->ParseDNNLowPOperatorArguments_();

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
    t_begin = chrono::system_clock::now();
  }

  const auto& X = InputTensorCPU_(0);
  const auto& W = InputTensorCPU_(1);
  auto* Y = OutputTensorCPU_(0);
  const auto canonical_axis = X.canonical_axis_index(axis_);
  const auto M = X.size_to_dim(canonical_axis);
  const auto K = X.size_from_dim(canonical_axis);
  const auto canonical_axis_w = W.canonical_axis_index(axis_w_);
  const int N = W.size_to_dim(canonical_axis_w);

  const T_signed* Wdata = W_quantized_.data();

  Y_shape_cache_ = X.sizes().vec();
  Y_shape_cache_.resize(canonical_axis + 1);
  Y_shape_cache_[canonical_axis] = N;
  Y->Resize(Y_shape_cache_);

  const T* Xdata = nullptr;
  vector<T> X_temp;

  if (VLOG_IS_ON(3)) {
    t_end = chrono::system_clock::now();
    double dt = chrono::duration<double>(t_end - t_begin).count();
    VLOG(3) << "@PERF this=" << this << " initialize parameters: " << dt * 1e3
            << " ms";
    t_begin = chrono::system_clock::now();
  }
  if (Wq_packed_) {
    // fast path to use fbgemm
    using namespace fbgemm;

    if (X.template IsType<T>() || !dequantize_output_) {
      // Only when input and output are float, we don't need input to be
      // quantized.
      if (VLOG_IS_ON(3)) {
        t_begin = chrono::system_clock::now();
      }
      Xdata = QuantizeInputIfNeeded<T>(this, 0, in_qparams_[0], X_temp);
      if (VLOG_IS_ON(3)) {
        t_end = chrono::system_clock::now();
        double dt = chrono::duration<double>(t_end - t_begin).count();
        VLOG(3) << "@PERF this=" << this << " input quantization: " << dt * 1e3
                << " ms";
        t_begin = chrono::system_clock::now();
      }
    }

    if (VLOG_IS_ON(3)) {
      t_begin = chrono::system_clock::now();
    }
    if (!dequantize_output_) {
      Y_int32_.resize(Y->size());
      DoNothing<> doNothingObj{};

      if (in_qparams_[1].zero_point) {
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

        ReQuantizeOutput<false /* FUSE_RELU */> outputProcObj(
            doNothingObj,
            &requantization_params_.real_multiplier,
            out_qparams_.zero_point,
            column_offsets_->empty() ? 0 : in_qparams_[0].zero_point,
            &in_qparams_[1].zero_point,
            packA.getRowOffsetBuffer(),
            column_offsets_->empty() ? nullptr : column_offsets_->data(),
            b_quantized_data_,
            N); // ncols per quant group

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
        X_pack_buf_.resize(PackAMatrix<uint8_t>::packedBufferSize());

        PackAMatrix<uint8_t> packA(
            matrix_op_t::NoTranspose,
            M,
            K,
            reinterpret_cast<const uint8_t*>(Xdata),
            K,
            X_pack_buf_.data(), // buffer for packed matrix
            1); // group

        ReQuantizeOutput<false /* FUSE_RELU */> outputProcObj(
            doNothingObj,
            &requantization_params_.real_multiplier,
            out_qparams_.zero_point,
            column_offsets_->empty() ? 0 : in_qparams_[0].zero_point,
            &in_qparams_[1].zero_point,
            nullptr,
            column_offsets_->empty() ? nullptr : column_offsets_->data(),
            b_quantized_data_,
            N); // ncols per quant group

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
        ReQuantizeForFloat<false /* FUSE_RELU*/> outputProcObj(
            doNothingObj,
            in_qparams_[0].scale,
            &in_qparams_[1].scale,
            column_offsets_->empty() ? 0 : in_qparams_[0].zero_point,
            &in_qparams_[1].zero_point,
            packA.getRowOffsetBuffer(),
            column_offsets_->empty() ? nullptr : column_offsets_->data(),
            b_dequantized_data_, // bias
            N); // ncols per quant group

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
        ReQuantizeForFloat<false /* FUSE_RELU*/> outputProcObj(
            doNothingObj,
            in_qparams_[0].scale,
            &in_qparams_[1].scale,
            column_offsets_->empty() ? 0 : in_qparams_[0].zero_point,
            &in_qparams_[1].zero_point,
            packA.getRowOffsetBuffer(),
            column_offsets_->empty() ? nullptr : column_offsets_->data(),
            b_dequantized_data_, // bias
            N); // ncols per quant group

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
    } // dequantize_output
  } else {
    // Quantize X
    if (VLOG_IS_ON(3)) {
      t_begin = chrono::system_clock::now();
    }
    Xdata = QuantizeInputIfNeeded<T>(this, 0, in_qparams_[0], X_temp);
    if (VLOG_IS_ON(3)) {
      t_end = chrono::system_clock::now();
      double dt = chrono::duration<double>(t_end - t_begin).count();
      VLOG(3) << "@PERF this=" << this << " input quantization: " << dt * 1e3
              << " ms";
      t_begin = chrono::system_clock::now();
    }

    Y_int32_.resize(Y->size());
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        int32_t sum = 0;
        for (int k = 0; k < K; ++k) {
          int w = Wdata[j * K + k];
          sum += Xdata[i * K + k] * w;
        }
        Y_int32_[i * N + j] = sum;
      } // for each output element
    } // for each row
  }

  if (FLAGS_caffe2_dnnlowp_dump_tensors) {
    // Dump input activation
    StoreMatrixInMatrixMarketFormat(M, K, Xdata, this->debug_def().input(0));

    // Dump weight
    StoreMatrixInMatrixMarketFormat(N, K, Wdata, this->debug_def().input(1));
  }

  if (VLOG_IS_ON(3)) {
    t_end = chrono::system_clock::now();
    double dt = chrono::duration<double>(t_end - t_begin).count();
    VLOG(3) << "@PERF this=" << this << " gemm: " << dt * 1e3 << " ms";

    t_begin = chrono::system_clock::now();
  }

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
        row_offset *= in_qparams_[1].zero_point;

        for (int j = 0; j < N; ++j) {
          if (!column_offsets_->empty()) {
            Y_int32_[i * N + j] -=
                in_qparams_[0].zero_point * (*column_offsets_)[j];
          }
          Y_int32_[i * N + j] -= row_offset;
          Ydata[i * N + j] = Y_int32_[i * N + j] * in_qparams_[0].scale *
                  in_qparams_[1].scale +
              b_dequantized_data_[j];
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
        row_offset *= in_qparams_[1].zero_point;

        for (int j = 0; j < N; ++j) {
          if (!column_offsets_->empty()) {
            // empty column offset means it's folded into bias
            Y_int32_[i * N + j] -=
                in_qparams_[0].zero_point * (*column_offsets_)[j];
          }
          Y_int32_[i * N + j] -= row_offset;
          Y_int32_[i * N + j] += b_quantized_data_[j];

          Ydata[i * N + j] = fbgemm::Requantize<T>(
              Y_int32_[i * N + j], requantization_params_);
        }
      }
    }

    PropagateOutputTensorQuantizationParams(this, 0, out_qparams_);
  }

  MeasureQuantizationError_();

  if (VLOG_IS_ON(3)) {
    t_end = chrono::system_clock::now();
    double dt = chrono::duration<double>(t_end - t_begin).count();
    VLOG(3) << "@PERF this=" << this
            << " bias-offset-requantization: " << dt * 1e3 << " ms";

    t_end = chrono::system_clock::now();
    double ops = 2. * M * N * K;
    dt = chrono::duration<double>(t_end - t_very_begin).count();
    double gops = ops / dt / 1e9;
    VLOG(3) << "@PERF this=" << this
            << " output=" << this->debug_def().output(0) << " " << M << "x" << N
            << "x" << K << ": " << dt * 1e3 << " ms " << gops << " gops";
  }

  return true;
}

template <typename T>
bool FullyConnectedDNNLowPOp<T>::GetQuantizationParameters_() {
  using namespace dnnlowp;

  chrono::time_point<chrono::system_clock> t_begin, t_end;
  if (VLOG_IS_ON(3)) {
    t_begin = chrono::system_clock::now();
  }
  // Choose quantization for X
  in_qparams_[0] = GetInputTensorQuantizationParamsOf(this, 0, qfactory_.get());

  if (VLOG_IS_ON(3)) {
    t_end = chrono::system_clock::now();
    double dt = chrono::duration<double>(t_end - t_begin).count();
    VLOG(3) << "@PERF this=" << this << " GetInputTensorQuantizationParamsOf "
            << dt * 1e3 << " ms";
    t_begin = chrono::system_clock::now();
  }

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
        this->debug_def().engine() != "DNNLOWP_ACC16";

    if ((fast_path && !Wq_packed_) || (!fast_path && W_quantized_.empty())) {
      if (this->template InputIsType<Int8FCDNNLowPPackedWeightBlob>(1)) {
        const auto& packed_filter =
            this->template Input<Int8FCDNNLowPPackedWeightBlob>(1);
        CAFFE_ENFORCE_EQ(packed_filter.qparams.size(), 1);
        in_qparams_[1] = packed_filter.qparams[0];
      } else {
        vector<TensorQuantizationParams> temp_qparams(1);
        QuantizeWeight<T>(
            InputBlob(1), K, N, temp_qparams, W_quantized_, qfactory_.get());
        in_qparams_[1] = temp_qparams[0];
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
              K, // ld
              in_qparams_[1].zero_point);
        }
      } else {
        string reason;
        if (!is_same<T, uint8_t>::value) {
          reason = "fbgemm only supports 8-bit integers";
        } else if (!GetCpuId().avx2()) {
          reason = "fbgemm only supports AVX2";
        } else if (this->debug_def().engine() == "DNNLOWP_ACC16") {
          reason = "";
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
    in_qparams_[1] = GetInputTensorQuantizationParamsOf(
        this, 1, qfactory_.get(), true /*weight*/);
    in_qparams_[1].zero_point += signed_min;

    W_quantized_.resize(W.size());
    fbgemm::Quantize<T_signed>(
        W.template data<float>(),
        W_quantized_.data(),
        W_quantized_.size(),
        in_qparams_[1]);
  }

  if (VLOG_IS_ON(3)) {
    t_end = chrono::system_clock::now();
    double dt = chrono::duration<double>(t_end - t_begin).count();
    VLOG(3) << "@PERF this=" << this << " Quantize W " << dt * 1e3 << " ms";
    t_begin = chrono::system_clock::now();
  }
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
      vector<TensorQuantizationParams> temp_qparams;
      temp_qparams.push_back(in_qparams_[1]);
      ComputeColumnOffsets<T_signed>(
          K, N, W_quantized_.data(), temp_qparams, *column_offsets_);
    }
  }
  if (VLOG_IS_ON(3)) {
    t_end = chrono::system_clock::now();
    double dt = chrono::duration<double>(t_end - t_begin).count();
    VLOG(3) << "@PERF this=" << this << " Calculate column offset " << dt * 1e3
            << " ms";
    t_begin = chrono::system_clock::now();
  }

  // Quantize bias
  if (!is_weight_constant_ || (!b_quantized_data_ && !b_dequantized_data_) ||
      in_qparams_[0].scale != in_qparams0_scale_old_ ||
      in_qparams_[0].zero_point != in_qparams0_zero_point_old_) {
    if (this->template InputIsType<Int8FCDNNLowPPackedWeightBlob>(2) &&
        this->template Input<Int8FCDNNLowPPackedWeightBlob>(2).bias.get()) {
      const auto& packed_filter =
          this->template Input<Int8FCDNNLowPPackedWeightBlob>(2);
      CAFFE_ENFORCE(!dequantize_output_);
      b_quantized_ = packed_filter.bias;
      b_quantized_data_ = b_quantized_->data();
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
                in_qparams_[0].scale * in_qparams_[1].scale),
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
                in_qparams_[0].scale * in_qparams_[1].scale,
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
        vector<TensorQuantizationParams> temp_qparams;
        temp_qparams.push_back(in_qparams_[1]);
        column_offset_temp.resize(N);
        ComputeColumnOffsets<T_signed>(
            K, N, W_quantized_.data(), temp_qparams, column_offset_temp);
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

  if (VLOG_IS_ON(3)) {
    t_end = chrono::system_clock::now();
    double dt = chrono::duration<double>(t_end - t_begin).count();
    VLOG(3) << "@PERF this=" << this << " Quantize bias " << dt * 1e3 << " ms";
    t_begin = chrono::system_clock::now();
  }

  if (!dequantize_output_ && !requantization_param_selected_) {
    GetOutputQuantizationParams_();

    float real_multiplier =
        in_qparams_[0].scale * in_qparams_[1].scale / out_qparams_.scale;
    requantization_params_ = qfactory_->ChooseRequantizationMultiplier(
        real_multiplier, out_qparams_);
    requantization_param_selected_ = true;
  } else {
    if (measure_quantization_error_) {
      // to measure quantization error, run ref impl.
      Fp32Op_()->DequantizeInput();
      Fp32Op_()->Get()->RunOnDevice();
    }
  }
  if (VLOG_IS_ON(3)) {
    t_end = chrono::system_clock::now();
    double dt = chrono::duration<double>(t_end - t_begin).count();
    VLOG(3) << "@PERF this=" << this << " GetOutputQuantizationParams "
            << dt * 1e3 << " ms";
    t_begin = chrono::system_clock::now();
  }
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

} // namespace caffe2
