#include "fully_connected_dnnlowp_acc16_op.h"

#include <fbgemm/src/RefImplementations.h>

C10_DECLARE_int32(dnnlowp_nbits_in_non_outlier);
C10_DECLARE_int32(dnnlowp_copy_to_32bit_frequency);

namespace caffe2 {

FullyConnectedDNNLowPAcc16Op::FullyConnectedDNNLowPAcc16Op(
    const OperatorDef& operator_def,
    Workspace* ws)
    : FullyConnectedDNNLowPOp<uint8_t>(operator_def, ws),
      nbits_in_non_outlier_(OperatorBase::GetSingleArgument<int>(
          "nbits_in_non_outlier",
          FLAGS_dnnlowp_nbits_in_non_outlier)),
      copy_to_32bit_frequency_(OperatorBase::GetSingleArgument<int>(
          "copy_to_32bit_frequency",
          FLAGS_dnnlowp_copy_to_32bit_frequency)) {}

bool FullyConnectedDNNLowPAcc16Op::RunOnDevice() {
  using namespace std;
  using namespace dnnlowp;

  this->ParseDNNLowPOperatorArguments_();

  // Get quantization parameters
  if (!GetQuantizationParameters_()) {
    return false;
  }

  const auto& X = InputTensorCPU_(0);
  const auto& W = InputTensorCPU_(1);
  auto *Y = OutputTensorCPU_(0);
  const auto canonical_axis = X.canonical_axis_index(axis_);
  const auto M = X.size_to_dim(canonical_axis);
  const auto K = X.size_from_dim(canonical_axis);
  const auto canonical_axis_w = W.canonical_axis_index(axis_w_);
  const int N = W.size_to_dim(canonical_axis_w);

  // Quantize X
  vector<uint8_t> X_temp;
  const uint8_t* Xdata = QuantizeInputIfNeeded<uint8_t>(
      this, 0, in_qparams_[0], X_temp, qfactory_.get());

  // Pack W if needed
  if (!Wq_acc16_packed_ || !is_weight_constant_) {
    if (!Wq_acc16_packed_ && nbits_in_non_outlier_ < 8) {
      static int log_occurences = 0;
      if (log_occurences < 32) {
        ++log_occurences;
        LOG(WARNING)
            << "FC DNNLOWP_ACC16 using outlier-aware quantization";
      }

      // Separate out outliers
      CAFFE_ENFORCE(!W_quantized_.empty());

      int32_t outlier_cnt = 0;
      for (int i = 0; i < W_quantized_.size(); ++i) {
        int8_t w = W_quantized_[i];
        bool is_outlier = nbits_in_non_outlier_ == 0 ||
            w < -(1 << (nbits_in_non_outlier_ - 1)) ||
            w >= (1 << (nbits_in_non_outlier_ - 1));
        if (is_outlier) {
          ++outlier_cnt;
        }
      }

      Wq_outlier_.reset(new fbgemm::CompressedSparseColumn(K, N));
      Wq_outlier_->RowIdx().resize(outlier_cnt);
      Wq_outlier_->Values().resize(outlier_cnt);

      outlier_cnt = 0;
      for (int j = 0; j < N; ++j) {
        Wq_outlier_->ColPtr()[j] = outlier_cnt;
        for (int16_t k = 0; k < K; ++k) {
          int8_t w = W_quantized_[j * K + k];
          bool is_outlier = nbits_in_non_outlier_ == 0 ||
              w < -(1 << (nbits_in_non_outlier_ - 1)) ||
              w >= (1 << (nbits_in_non_outlier_ - 1));
          if (is_outlier) {
            CAFFE_ENFORCE_LE(k, numeric_limits<int16_t>::max());
            Wq_outlier_->RowIdx()[outlier_cnt] = k;
            Wq_outlier_->Values()[outlier_cnt] = w;
            ++outlier_cnt;
            W_quantized_[j * K + k] = 0;
          }
        }
      }
      Wq_outlier_->ColPtr()[N] = outlier_cnt;

      LOG(INFO) << "Proportion of outlier for FC layer with weight blob "
                << OperatorBase::debug_def().input(1) << " is "
                << (float)outlier_cnt / W_quantized_.size();

      LOG(INFO) << "copy_to_32bit_frequency " << copy_to_32bit_frequency_;
    }

    Wq_acc16_packed_.reset(new fbgemm::PackBMatrix<int8_t, int16_t>(
        fbgemm::matrix_op_t::Transpose,
        K,
        N,
        reinterpret_cast<const int8_t*>(W_quantized_.data()),
        K));

    if (is_weight_constant_) {
      vector<T_signed>().swap(W_quantized_);
    }
  }

  Y_shape_cache_ = X.sizes().vec();
  Y_shape_cache_.resize(canonical_axis + 1);
  Y_shape_cache_[canonical_axis] = N;
  Y->Resize(Y_shape_cache_);

  using namespace fbgemm;
  // main GEMM
  // TODO : omp parallelization
  Y_int32_.resize(Y->size());
  uint8_t* Ydata = GetQuantizedOutputData_();
  if (nbits_in_non_outlier_ > 0) {
    int row_offset_size_per_thread =
        PackAWithRowOffset<uint8_t, int16_t>::rowOffsetBufferSize();
    int x_pack_buf_size_per_thread =
        PackAWithRowOffset<uint8_t, int16_t>::packedBufferSize();
    this->row_offsets_.resize(row_offset_size_per_thread);
    this->X_pack_buf_.resize(x_pack_buf_size_per_thread);

    PackAWithRowOffset<uint8_t, int16_t> packA(
        matrix_op_t::NoTranspose,
        M,
        K,
        Xdata,
        K,
        X_pack_buf_.data(),
        1, // group
        in_qparams_[0].zero_point,
        row_offsets_.data());

    if (!dequantize_output_) {
      DoNothing<> doNothingObj{};
      ReQuantizeOutput<false /* fuse relu */> reqObj(
          doNothingObj,
          requantization_params_.real_multiplier,
          out_qparams_.zero_point,
          in_qparams_[0].zero_point,
          in_qparams_[1].zero_point,
          packA.getRowOffsetBuffer(),
          column_offsets_.data(),
          this->b_quantized_data_);

      if (nbits_in_non_outlier_ < 8) {
        DoSpmdmOnInpBuffer<
            typename ReQuantizeOutput<false /* fuse relu */>::outType,
            int32_t,
            ReQuantizeOutput<false /* fuse relu */>>
            spmdmObj(reqObj, Xdata, K, *Wq_outlier_);

        fbgemmPacked(
            packA,
            *Wq_acc16_packed_,
            Ydata,
            Y_int32_.data(),
            N,
            spmdmObj,
            0, // thread_id
            1); // num_threads
      } else {
        fbgemmPacked(
            packA,
            *Wq_acc16_packed_,
            Ydata,
            Y_int32_.data(),
            N,
            reqObj,
            0, // thread_id
            1); // num_threads
      }
    } else {
      DoNothing<float, float> doNothingObj{};
      ReQuantizeForFloat<false /* FUSE_RELU*/> reqObj(
          doNothingObj,
          in_qparams_[0].scale,
          in_qparams_[1].scale,
          in_qparams_[0].zero_point,
          in_qparams_[1].zero_point,
          packA.getRowOffsetBuffer(),
          column_offsets_.data(),
          this->b_dequantized_data_);

      if (nbits_in_non_outlier_ < 8) {
        DoSpmdmOnInpBuffer<
            typename ReQuantizeForFloat<false /* fuse relu */>::outType,
            int32_t,
            ReQuantizeForFloat<false /* fuse relu */>>
            spmdmObj(reqObj, Xdata, K, *Wq_outlier_);

        fbgemmPacked(
            packA,
            *Wq_acc16_packed_,
            Y->mutable_data<float>(),
            Y_int32_.data(),
            N,
            spmdmObj,
            0, // thread_id
            1); // num_threads
      } else {
        fbgemmPacked(
            packA,
            *Wq_acc16_packed_,
            Y->mutable_data<float>(),
            Y_int32_.data(),
            N,
            reqObj,
            0, // thread_id
            1); // num_threads
      }
    }
  } else {
    block_type_t block{0, static_cast<int>(M), 0, static_cast<int>(N)};
    Wq_outlier_->SpMDM(
        block, Xdata, K, false /* accumulate */, Y_int32_.data(), N);

    if (dequantize_output_) {
      float* Ydata_float = Output(0)->template mutable_data<float>();

#pragma omp parallel for
      for (int i = 0; i < M; ++i) {
        int32_t row_offset = 0;
        for (int k = 0; k < K; ++k) {
          row_offset += Xdata[i * K + k];
        }
        row_offset *= in_qparams_[1].zero_point;

        for (int j = 0; j < N; ++j) {
          Y_int32_[i * N + j] -=
            in_qparams_[0].zero_point * column_offsets_[j] + row_offset;
          Ydata_float[i * N + j] = Y_int32_[i * N + j] * in_qparams_[0].scale *
                  in_qparams_[1].scale +
              b_dequantized_data_[j];
        }
      }
    } else {
      // Add offsets/bias, and requantize
#pragma omp parallel for
      for (int i = 0; i < M; ++i) {
        int32_t row_offset = 0;
        for (int k = 0; k < K; ++k) {
          row_offset += Xdata[i * K + k];
        }

        requantize_u8acc32_ref(
          1,
          N,
          N,
          Y_int32_.data() + i * N,
          Ydata + i * N,
          requantization_params_.real_multiplier,
          out_qparams_.zero_point,
          in_qparams_[0].zero_point,
          in_qparams_[1].zero_point,
          &row_offset,
          column_offsets_.data(),
          b_quantized_.data());
      }
    }
  }

  if (!dequantize_output_) {
    PropagateOutputTensorQuantizationParams(this, 0, out_qparams_);
  }
  MeasureQuantizationError_();

  return true;
}

REGISTER_CPU_OPERATOR_WITH_ENGINE(
  FC, DNNLOWP_ACC16, FullyConnectedDNNLowPAcc16Op);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
  Int8FC, DNNLOWP_ACC16, FullyConnectedDNNLowPAcc16Op);

} // namespace caffe2
