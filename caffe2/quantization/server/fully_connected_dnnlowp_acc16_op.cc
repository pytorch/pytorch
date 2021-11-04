#include "fully_connected_dnnlowp_acc16_op.h"

#include <fbgemm/src/RefImplementations.h>

#include "fbgemm_pack_op.h"

C10_DECLARE_int32(caffe2_dnnlowp_nbits_in_non_outlier);
C10_DECLARE_int32(caffe2_dnnlowp_copy_to_32bit_frequency);

namespace caffe2 {

FullyConnectedDNNLowPAcc16Op::FullyConnectedDNNLowPAcc16Op(
    const OperatorDef& operator_def,
    Workspace* ws)
    : FullyConnectedDNNLowPOp<uint8_t>(operator_def, ws),
      nbits_in_non_outlier_(this->template GetSingleArgument<int>(
          "nbits_in_non_outlier",
          FLAGS_caffe2_dnnlowp_nbits_in_non_outlier)),
      copy_to_32bit_frequency_(this->template GetSingleArgument<int>(
          "copy_to_32bit_frequency",
          FLAGS_caffe2_dnnlowp_copy_to_32bit_frequency)) {}

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
  auto* Y = OutputTensorCPU_(0);
  const auto canonical_axis = X.canonical_axis_index(axis_);
  const auto M = X.size_to_dim(canonical_axis);
  const auto K = X.size_from_dim(canonical_axis);
  const auto canonical_axis_w = W.canonical_axis_index(axis_w_);
  const int N = W.size_to_dim(canonical_axis_w);

  // Quantize X
  vector<uint8_t> X_temp;
  const uint8_t* Xdata =
      QuantizeInputIfNeeded<uint8_t>(this, 0, in_qparams_[0], X_temp);

  if (this->quantize_channelwise_) {
    LOG(WARNING) << "FC with 16-bit accumulation doesn't work with per-channel "
                    "quantization yet.";
  }

  // Pack W if needed
  if (!Wq_acc16_packed_ || !is_weight_constant_) {
    if (this->template InputIsType<Int8FCDNNLowPPackedWeightBlob>(1)) {
      // If the input is already packed
      const auto& packed_filter =
          this->template Input<Int8FCDNNLowPPackedWeightBlob>(1);
      Wq_outlier_ = packed_filter.W_outlier;
      Wq_acc16_packed_ = packed_filter.W_acc16;

      if (nbits_in_non_outlier_ != packed_filter.nbits_in_non_outlier) {
        LOG(WARNING)
            << "nbits_in_non_outlier in packed weight "
            << packed_filter.nbits_in_non_outlier
            << " doesn't match with nbits_in_non_outlier specified in operator "
            << nbits_in_non_outlier_;
      }
    } else {
      if (!Wq_acc16_packed_ && nbits_in_non_outlier_ < 8) {
        static int log_occurences = 0;
        if (log_occurences < 32) {
          ++log_occurences;
          LOG(WARNING) << "FC DNNLOWP_ACC16 using outlier-aware quantization";
        }

        // Separate out outliers
        CAFFE_ENFORCE(!W_quantized_.empty());

        Wq_outlier_.reset(
            ExtractOutlierMatrix(1, K, N, nbits_in_non_outlier_, W_quantized_));
        int outlier_cnt = Wq_outlier_->ColPtr()[N];

        LOG(INFO) << "Proportion of outlier for FC layer with weight blob "
                  << this->debug_def().input(1) << " is "
                  << (float)outlier_cnt / W_quantized_.size();

        LOG(INFO) << "copy_to_32bit_frequency " << copy_to_32bit_frequency_;
      }

      // NOLINTNEXTLINE(modernize-make-shared)
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

    // TODO: use PackAMatrix if filter_qparams_[0].zero_point == 0
    PackAWithRowOffset<uint8_t, int16_t> packA(
        matrix_op_t::NoTranspose,
        M,
        K,
        Xdata,
        K,
        X_pack_buf_.data(),
        1, // group
        row_offsets_.data());

    if (!dequantize_output_) {
      DoNothing<> doNothingObj{};
      ReQuantizeOutput<false /* fuse relu */> reqObj(
          doNothingObj,
          this->requantization_multipliers_.data(),
          out_qparams_.zero_point,
          column_offsets_->empty() ? 0 : in_qparams_[0].zero_point,
          this->filter_zero_points_.data(),
          packA.getRowOffsetBuffer(),
          column_offsets_->empty() ? nullptr : column_offsets_->data(),
          this->b_quantized_data_,
          N); // ncols per quant group

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
          this->filter_scales_.data(),
          column_offsets_->empty() ? 0 : in_qparams_[0].zero_point,
          this->filter_zero_points_.data(),
          packA.getRowOffsetBuffer(),
          column_offsets_->empty() ? nullptr : column_offsets_->data(),
          this->b_dequantized_data_,
          N); // ncols per quant group

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

        for (int j = 0; j < N; ++j) {
          int quant_group = this->quantize_channelwise_ ? j : 0;
          Y_int32_[i * N + j] -=
              row_offset * this->filter_qparams_[quant_group].zero_point;
          if (!column_offsets_->empty()) {
            Y_int32_[i * N + j] -=
                in_qparams_[0].zero_point * (*column_offsets_)[j];
          }
          Ydata_float[i * N + j] = Y_int32_[i * N + j] * in_qparams_[0].scale *
                  in_qparams_[quant_group].scale +
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
            this->requantization_multipliers_.data(),
            out_qparams_.zero_point,
            column_offsets_->empty() ? 0 : in_qparams_[0].zero_point,
            this->filter_zero_points_.data(),
            &row_offset,
            column_offsets_->empty() ? nullptr : column_offsets_->data(),
            b_quantized_->data(),
            N); // ncols per quant group
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
    FC,
    DNNLOWP_ACC16,
    FullyConnectedDNNLowPAcc16Op);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8FC,
    DNNLOWP_ACC16,
    FullyConnectedDNNLowPAcc16Op);

} // namespace caffe2
