#include "conv_dnnlowp_op.h"

// #define DNNLOWP_MEASURE_TIME_BREAKDOWN
#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
#include <chrono>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/conv_op_impl.h"
#include "caffe2/utils/cpuid.h"

#include <fbgemm/src/RefImplementations.h>

#include "dnnlowp_op.h"
#include "dnnlowp_partition.h"
#include "fbgemm_pack_op.h"
#include "im2col_dnnlowp.h"
#include "mmio.h"

C10_DEFINE_bool(
    caffe2_dnnlowp_shared_int32_buffer,
    false,
    "Share intermediate int32 buffer across DNNLOWP Conv ops");

C10_DEFINE_bool(
    caffe2_dnnlowp_dump_tensors,
    false,
    "Dump quantized input and weight tensors used in Conv and FC operators "
    "during the first iteration");

C10_DECLARE_bool(caffe2_dnnlowp_force_slow_path);

namespace caffe2 {

using namespace std;

template <typename T, bool ReluFused>
ConvDNNLowPOp<T, ReluFused>::ConvDNNLowPOp(
    const OperatorDef& operator_def,
    Workspace* ws)
    : BaseType(operator_def, ws),
      column_offsets_(make_shared<vector<int32_t>>()),
      b_quantized_(make_shared<vector<int32_t>>()) {
  in_qparams_.resize(1);

  // Create shared buffer mutex in the constructor
  // to avoid race-condition in DAGNet.
  if (FLAGS_caffe2_force_shared_col_buffer || shared_buffer_) {
    createSharedBuffer<CPUContext>(ws_);
  }

  if (FLAGS_caffe2_dnnlowp_shared_int32_buffer) {
    this->CreateSharedInt32Buffer_();
  }

  quantize_groupwise_ =
      this->template GetSingleArgument<bool>("quantize_groupwise", false);
}

template <typename T, bool ReluFused>
// NOLINTNEXTLINE(modernize-use-equals-default)
ConvDNNLowPOp<T, ReluFused>::~ConvDNNLowPOp() {}

template <typename T, bool ReluFused>
dnnlowp::TensorQuantizationParams&
ConvDNNLowPOp<T, ReluFused>::FilterQuantizationParams(int group_id) {
  return filter_qparams_[quantize_groupwise_ ? group_id : 0];
}

template <typename T, bool ReluFused>
dnnlowp::RequantizationParams&
ConvDNNLowPOp<T, ReluFused>::RequantizationParams(int group_id) {
  return requantization_params_[quantize_groupwise_ ? group_id : 0];
}

// FIXME : code duplication with
// ConvDNNLowPPackWeightOp::TakeDepthWise3x3FastPath_
template <typename T, bool ReluFused>
bool ConvDNNLowPOp<T, ReluFused>::TakeDepthWise3x3FastPath_() {
  const Tensor& X = InputTensorCPU_(INPUT);
  return this->order_ == StorageOrder::NHWC && is_same<T, uint8_t>::value &&
      !Acc16() && group_ == X.dim32(X.dim() - 1) && group_ % 8 == 0 &&
      this->kernel_.size() == 2 && kernel_h() == 3 && kernel_w() == 3 &&
      stride_h() == stride_w() && (stride_h() == 1 || stride_h() == 2) &&
      dilation_h() == 1 && dilation_w() == 1 && pad_t() == 1 && pad_b() == 1 &&
      pad_l() == 1 && pad_r() == 1 && GetCpuId().avx2();
}

// FIXME : code duplication with
// ConvDNNLowPPackWeightOp::TakeDepthWise3x3x3FastPath_
template <typename T, bool ReluFused>
bool ConvDNNLowPOp<T, ReluFused>::TakeDepthWise3x3x3FastPath_() {
  const Tensor& X = InputTensorCPU_(INPUT);
  bool ret = this->order_ == StorageOrder::NHWC && is_same<T, uint8_t>::value &&
      !Acc16() && group_ == X.dim32(X.dim() - 1) && group_ % 8 == 0 &&
      this->kernel_.size() == 3 && this->kernel_[0] == 3 &&
      this->kernel_[1] == 3 && this->kernel_[2] == 3 &&
      (this->stride_[0] == 1 || this->stride_[0] == 2) &&
      (this->stride_[1] == 1 || this->stride_[1] == 2) &&
      (this->stride_[2] == 1 || this->stride_[2] == 2) &&
      this->dilation_[0] == 1 && this->dilation_[1] == 1 &&
      this->dilation_[2] == 1 &&
      accumulate(
          this->pads_.begin(), this->pads_.end(), 1, multiplies<int>()) == 1 &&
      GetCpuId().avx2();
  return ret;
}

template <typename T, bool ReluFused>
fbgemm::conv_param_t<> ConvDNNLowPOp<T, ReluFused>::GetConvParam_() {
  CAFFE_ENFORCE_EQ(this->kernel_.size(), 2);
  CAFFE_ENFORCE_EQ(this->order_, StorageOrder::NHWC);

  const Tensor& X = InputTensorCPU_(INPUT);
  auto& filter = InputTensorCPU_(FILTER);
  const int N = X.dim32(0), C = X.dim32(X.dim() - 1);
  const int M = filter.dim32(0);

  return fbgemm::conv_param_t<>(
      N,
      C,
      M,
      {X.dim32(1), X.dim32(2)},
      group_,
      {this->kernel_[0], this->kernel_[1]},
      {this->stride_[0], this->stride_[1]},
      {this->pads_[0], this->pads_[1], this->pads_[2], this->pads_[3]});
}

template <typename T, bool ReluFused>
fbgemm::conv_param_t<3> ConvDNNLowPOp<T, ReluFused>::GetConv3DParam_() {
  CAFFE_ENFORCE_EQ(this->kernel_.size(), 3);
  CAFFE_ENFORCE_EQ(this->order_, StorageOrder::NHWC);

  const Tensor& X = InputTensorCPU_(INPUT);
  auto& filter = InputTensorCPU_(FILTER);
  const int N = X.dim32(0), C = X.dim32(X.dim() - 1);
  const int M = filter.dim32(0);

  return fbgemm::conv_param_t<3>(
      N,
      C,
      M,
      {X.dim32(1), X.dim32(2), X.dim32(3)},
      group_,
      {this->kernel_[0], this->kernel_[1], this->kernel_[2]},
      {this->stride_[0], this->stride_[1], this->stride_[2]},
      {this->pads_[0],
       this->pads_[1],
       this->pads_[2],
       this->pads_[3],
       this->pads_[4],
       this->pads_[5]});
}

// FIXME : code duplication with
// ConvDNNLowPPackWeightOp::TakeGConvFastPath_
template <typename T, bool ReluFused>
bool ConvDNNLowPOp<T, ReluFused>::TakeGConvFastPath_() {
  const Tensor& X = InputTensorCPU_(INPUT);
  if (this->order_ != StorageOrder::NHWC || !is_same<T, uint8_t>::value ||
      !X.template IsType<T>() ||
      (this->kernel_.size() != 2 && this->kernel_.size() != 3) || Acc16()) {
    return false;
  }

  if (this->kernel_.size() == 2) {
    return fbgemm::fbgemmOptimizedGConv(GetConvParam_());
  } else {
    CAFFE_ENFORCE_EQ(this->kernel_.size(), 3);
    return fbgemm::fbgemmOptimizedGConv(GetConv3DParam_());
  }
}

template <typename T, bool ReluFused>
int ConvDNNLowPOp<T, ReluFused>::KernelDim_() {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int kernel_dim;
  const Tensor& X = InputTensorCPU_(INPUT);
  const auto& filter = InputTensorCPU_(FILTER);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int C;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int filter_offset;
  if (ConvPoolOpBase<CPUContext>::order_ == StorageOrder::NCHW) {
    C = X.dim32(1);
    filter_offset = 2;
  } else {
    C = X.dim32(X.dim() - 1);
    filter_offset = 1;
  }

  int kernel_dims_size = 1;
  // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
  for (int i = 0; i < this->kernel_.size(); ++i) {
    CAFFE_ENFORCE_EQ(filter.dim32(i + filter_offset), kernel_[i]);
    kernel_dims_size *= kernel_[i];
  }
  kernel_dim = C / group_ * kernel_dims_size;

  return kernel_dim;
}

template <typename T, bool ReluFused>
bool ConvDNNLowPOp<T, ReluFused>::IsConvGEMM_() const {
  return accumulate(
             this->kernel_.begin(),
             this->kernel_.end(),
             1,
             multiplies<int>()) == 1 &&
      accumulate(
          this->stride_.begin(), this->stride_.end(), 1, multiplies<int>()) ==
      1 &&
      accumulate(
          this->dilation_.begin(),
          this->dilation_.end(),
          1,
          multiplies<int>()) == 1 &&
      accumulate(this->pads_.begin(), this->pads_.end(), 0) == 0;
}

template <typename T, bool ReluFused>
bool ConvDNNLowPOp<T, ReluFused>::NoIm2ColNHWC_() {
  if (TakeDepthWise3x3FastPath_() || TakeDepthWise3x3x3FastPath_() ||
      TakeGConvFastPath_()) {
    return true;
  }

  if (Wq_packed_ &&
      accumulate(
          this->dilation_.begin(),
          this->dilation_.end(),
          1,
          multiplies<int>()) == 1) {
    // im2col fusion
    return true;
  }

  return IsConvGEMM_();
}

template <typename T, bool ReluFused>
void ConvDNNLowPOp<T, ReluFused>::PreComputeRowColumnOffsets_() {
  if (this->order_ == StorageOrder::NHWC &&
      this->template InputIsType<int8::Int8TensorCPU>(INPUT)) {
    // If input tensor doesn't use dynamic quantization, we fold column_offsets_
    // into bias.
    return;
  }

  const auto& filter = InputTensorCPU_(FILTER);
  int kernel_dim = KernelDim_();
  int M = filter.dim32(0);

  // Pre-compute row_offset / column_offset
  vector<int>& offsets =
      this->order_ == StorageOrder::NCHW ? row_offsets_ : *column_offsets_;

  if (offsets.empty()) {
    if (this->template InputIsType<Int8ConvDNNLowPPackedWeightBlob>(FILTER)) {
      const auto& packed_filter =
          this->template Input<Int8ConvDNNLowPPackedWeightBlob>(FILTER);
      column_offsets_ = packed_filter.column_offsets;
    } else {
      ComputeColumnOffsets<T_signed>(
          kernel_dim, M, W_quantized_.data(), filter_qparams_, offsets);
    }
  }
}

template <typename T, bool ReluFused>
void ConvDNNLowPOp<T, ReluFused>::QuantizeBias_() {
  using namespace dnnlowp;

  const auto& filter = InputTensorCPU_(FILTER);
  int M = filter.dim32(0);

  bool has_packed_bias =
      this->template InputIsType<Int8ConvDNNLowPPackedWeightBlob>(FILTER) &&
      this->template Input<Int8ConvDNNLowPPackedWeightBlob>(FILTER).bias.get();
  bool has_bias = InputSize() == 3 || has_packed_bias;

  // Quantize bias
  if (has_bias &&
      (!b_quantized_data_ ||
       in_qparams_[INPUT].scale != in_qparams_scale_old_ ||
       in_qparams_[INPUT].zero_point != in_qparams_zero_point_old_)) {
    if (has_packed_bias) {
      const auto& packed_filter =
          this->template Input<Int8ConvDNNLowPPackedWeightBlob>(FILTER);
      b_quantized_data_ = packed_filter.bias->data();
    } else {
      const auto& bias = InputTensorCPU_(BIAS);
      if (this->template InputIsType<int8::Int8TensorCPU>(BIAS)) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
        TensorQuantizationParams bias_qparams;
        bias_qparams.scale =
            this->template Input<int8::Int8TensorCPU>(BIAS).scale;
        bias_qparams.zero_point =
            this->template Input<int8::Int8TensorCPU>(BIAS).zero_point;
        if (InputTensorCPU_(INPUT).dim32(0) > 0) {
          CAFFE_ENFORCE_LE(
              std::abs(
                  bias_qparams.scale -
                  in_qparams_[INPUT].scale * FilterQuantizationParams(0).scale),
              1e-4);
        }
        CAFFE_ENFORCE_EQ(bias_qparams.zero_point, 0);
        b_quantized_data_ = bias.template data<int32_t>();
      } else {
        const float* b_data = bias.template data<float>();
        b_quantized_->resize(bias.numel());
        // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
        for (int g = 0; g < filter_qparams_.size(); ++g) {
          int i_begin = g * (M / filter_qparams_.size());
          int i_end = i_begin + (M / filter_qparams_.size());
          for (int i = i_begin; i < i_end; ++i) {
            (*b_quantized_)[i] = fbgemm::Quantize<int32_t>(
                b_data[i],
                0,
                in_qparams_[INPUT].scale * FilterQuantizationParams(g).scale,
                32,
                true /* signed */);
          }
        }
        b_quantized_data_ = b_quantized_->data();
      }
    }
    in_qparams_scale_old_ = in_qparams_[INPUT].scale;
    in_qparams_zero_point_old_ = in_qparams_[INPUT].zero_point;

    CAFFE_ENFORCE(b_quantized_data_);

    // If column_offsets_ is empty even when we need column_offsets (asymmetric
    // quantization in input), it means we need to fuse column_offsets to bias.
    if (this->order_ == StorageOrder::NHWC && in_qparams_[INPUT].zero_point &&
        column_offsets_->empty()) {
      if (b_quantized_->empty()) {
        // When b_quantized_data_ is from pre-packed bias or Int8TensorCPU,
        // we can't inplace modify so copy to internal b_quantized_ vector.
        b_quantized_->assign(b_quantized_data_, b_quantized_data_ + M);
        b_quantized_data_ = b_quantized_->data();
      }
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      vector<int32_t>* column_offset_ptr;
      vector<int32_t> column_offset_temp;
      if (this->template InputIsType<Int8ConvDNNLowPPackedWeightBlob>(FILTER)) {
        const auto& packed_filter =
            this->template Input<Int8ConvDNNLowPPackedWeightBlob>(FILTER);
        column_offset_ptr = packed_filter.column_offsets.get();
      } else {
        column_offset_temp.resize(M);
        ComputeColumnOffsets<T_signed>(
            KernelDim_(),
            M,
            W_quantized_.data(),
            filter_qparams_,
            column_offset_temp);
        column_offset_ptr = &column_offset_temp;
      }
      for (int i = 0; i < M; ++i) {
        (*b_quantized_)[i] -=
            in_qparams_[INPUT].zero_point * (*column_offset_ptr)[i];
      }
    }
  }

  if (!has_bias && this->order_ == StorageOrder::NHWC &&
      in_qparams_[INPUT].zero_point && column_offsets_->empty() &&
      !b_quantized_data_) {
    // no bias but create one filling with column offset values
    b_quantized_->resize(M, 0);
    b_quantized_data_ = b_quantized_->data();

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    vector<int32_t>* column_offset_ptr;
    vector<int32_t> column_offset_temp;
    if (this->template InputIsType<Int8ConvDNNLowPPackedWeightBlob>(FILTER)) {
      const auto& packed_filter =
          this->template Input<Int8ConvDNNLowPPackedWeightBlob>(FILTER);
      column_offset_ptr = packed_filter.column_offsets.get();
    } else {
      column_offset_temp.resize(M);
      ComputeColumnOffsets<T_signed>(
          KernelDim_(),
          M,
          W_quantized_.data(),
          filter_qparams_,
          column_offset_temp);
      column_offset_ptr = &column_offset_temp;
    }
    for (int i = 0; i < M; ++i) {
      (*b_quantized_)[i] -= in_qparams_[INPUT].zero_point * (*column_offset_ptr)[i];
    }
  }
}

template <typename T, bool ReluFused>
void ConvDNNLowPOp<T, ReluFused>::QuantizeWeight_() {
  using namespace dnnlowp;

  // Quantize W if not done already
  int kernel_dim = KernelDim_();
  const auto& filter = InputTensorCPU_(FILTER);
  int M = filter.dim32(0);

  bool packW = ConvPoolOpBase<CPUContext>::order_ == StorageOrder::NHWC &&
      !Acc16() && is_same<T, uint8_t>::value && GetCpuId().avx2() &&
      !FLAGS_caffe2_dnnlowp_force_slow_path;

  bool depthwise_3x3_fast_path = false, depthwise_3x3x3_fast_path = false,
       gconv_fast_path = false;
  if (TakeDepthWise3x3FastPath_()) {
    depthwise_3x3_fast_path = true;
    packW = false;
  } else if (TakeDepthWise3x3x3FastPath_()) {
    depthwise_3x3x3_fast_path = true;
    packW = false;
  } else if (TakeGConvFastPath_()) {
    gconv_fast_path = true;
    packW = false;
  }

  if ((depthwise_3x3_fast_path && !Wq_depthwise_packed_) ||
      (depthwise_3x3x3_fast_path && !Wq_depthwise_packed_) ||
      (gconv_fast_path && !Wq_gconv_packed_ && !Wq_gconv3d_packed_) ||
      (packW && !Wq_packed_) || (!packW && W_quantized_.empty())) {
    if (this->template InputIsType<Int8ConvDNNLowPPackedWeightBlob>(FILTER)) {
      CAFFE_ENFORCE_EQ(
          ConvPoolOpBase<CPUContext>::order_,
          StorageOrder::NHWC,
          "Pre-packed weight only works with NHWC layout");

      const auto& packed_filter =
          this->template Input<Int8ConvDNNLowPPackedWeightBlob>(FILTER);
      filter_qparams_ = packed_filter.qparams;
    } else {
      filter_qparams_.resize(quantize_groupwise_ ? group_ : 1);
      QuantizeWeight<T>(
          InputBlob(FILTER),
          kernel_dim,
          M,
          filter_qparams_,
          W_quantized_,
          qfactory_.get());

      if (this->template InputIsType<int8::Int8TensorCPU>(FILTER) &&
          quantize_groupwise_) {
        static int log_occurences = 0;
        if (log_occurences < 32) {
          ++log_occurences;
          LOG(WARNING) << "Cannot do group-wise quantization for "
                          "pre-quantized weight "
                       << this->debug_def().input(FILTER);
        }
      }
    }

    filter_zero_points_.resize(filter_qparams_.size());
    requantization_params_.resize(filter_qparams_.size());
    requantization_multipliers_.resize(filter_qparams_.size());
    // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
    for (int i = 0; i < filter_qparams_.size(); ++i) {
      filter_zero_points_[i] = filter_qparams_[i].zero_point;
    }

    if (depthwise_3x3_fast_path) {
      if (this->template InputIsType<Int8ConvDNNLowPPackedWeightBlob>(FILTER)) {
        const auto& packed_filter =
            this->template Input<Int8ConvDNNLowPPackedWeightBlob>(FILTER);
        Wq_depthwise_packed_ = packed_filter.W_depthwise;
      } else {
        // NOLINTNEXTLINE(modernize-make-shared)
        Wq_depthwise_packed_.reset(new fbgemm::PackedDepthWiseConvMatrix(
            group_,
            3 * 3,
            reinterpret_cast<const int8_t*>(W_quantized_.data())));
      }
    } else if (depthwise_3x3x3_fast_path) {
      if (this->template InputIsType<Int8ConvDNNLowPPackedWeightBlob>(FILTER)) {
        const auto& packed_filter =
            this->template Input<Int8ConvDNNLowPPackedWeightBlob>(FILTER);
        Wq_depthwise_packed_ = packed_filter.W_depthwise;
      } else {
        // NOLINTNEXTLINE(modernize-make-shared)
        Wq_depthwise_packed_.reset(new fbgemm::PackedDepthWiseConvMatrix(
            group_,
            3 * 3 * 3,
            reinterpret_cast<const int8_t*>(W_quantized_.data())));
      }
    } else if (gconv_fast_path) {
      if (this->template InputIsType<Int8ConvDNNLowPPackedWeightBlob>(FILTER)) {
        const auto& packed_filter =
            this->template Input<Int8ConvDNNLowPPackedWeightBlob>(FILTER);
        if (packed_filter.W_gconv) {
          CAFFE_ENFORCE_EQ(this->kernel_.size(), 2);
          Wq_gconv_packed_ = packed_filter.W_gconv;
        } else {
          CAFFE_ENFORCE(packed_filter.W_gconv3d);
          CAFFE_ENFORCE_EQ(this->kernel_.size(), 3);
          Wq_gconv3d_packed_ = packed_filter.W_gconv3d;
        }
      } else {
        if (this->kernel_.size() == 2) {
          fbgemm::conv_param_t<> conv_p(GetConvParam_());
          // NOLINTNEXTLINE(modernize-make-shared)
          Wq_gconv_packed_.reset(new fbgemm::PackWeightMatrixForGConv<int8_t>(
              fbgemm::matrix_op_t::Transpose,
              conv_p,
              reinterpret_cast<const int8_t*>(W_quantized_.data())));
        } else {
          CAFFE_ENFORCE_EQ(this->kernel_.size(), 3);
          fbgemm::conv_param_t<3> conv_p(GetConv3DParam_());
          // NOLINTNEXTLINE(modernize-make-shared)
          Wq_gconv3d_packed_.reset(
              new fbgemm::PackWeightMatrixForGConv<int8_t, int32_t, 3>(
                  fbgemm::matrix_op_t::Transpose,
                  conv_p,
                  reinterpret_cast<const int8_t*>(W_quantized_.data())));
        }
      }
    } else if (packW) {
      if (this->template InputIsType<Int8ConvDNNLowPPackedWeightBlob>(FILTER)) {
        const auto& packed_filter =
            this->template Input<Int8ConvDNNLowPPackedWeightBlob>(FILTER);
        Wq_packed_ = packed_filter.W;
      } else {
        // fast path using fbgemm
        // NOLINTNEXTLINE(modernize-make-shared)
        Wq_packed_.reset(new fbgemm::PackBMatrix<int8_t>(
            fbgemm::matrix_op_t::Transpose,
            group_ * kernel_dim,
            M / group_,
            reinterpret_cast<const int8_t*>(W_quantized_.data()),
            kernel_dim, // ld
            nullptr, // pmat
            group_));
      }
    } else {
      string reason;
      if (ConvPoolOpBase<CPUContext>::order_ != StorageOrder::NHWC) {
        reason = "fbgemm only supports NHWC layout";
      } else if (!is_same<T, uint8_t>::value) {
        reason = "fbgemm only supports 8-bit integers";
      } else if (!GetCpuId().avx2()) {
        reason = "fbgemm only supports AVX2+";
      } else if (Acc16()) {
        reason = "";
      } else if (FLAGS_caffe2_dnnlowp_force_slow_path) {
        reason = "slow path enforced";
      } else {
        assert(false);
      }
      if (!reason.empty()) {
        static int log_occurences = 0;
        if (log_occurences < 32) {
          ++log_occurences;
          LOG(WARNING) << "Conv with weight " << this->debug_def().input(FILTER)
                       << " falls back to slow path because " << reason;
        }
      }
    }
  }
}

/**
 * @return false if something goes wrong
 */
template <typename T, bool ReluFused>
bool ConvDNNLowPOp<T, ReluFused>::GetQuantizationParameters_() {
  using namespace dnnlowp;

  if (!this->arguments_parsed_) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    bool dequantize_output;
    ParseDNNLowPOperatorArguments(
        this, &dequantize_output, &measure_quantization_error_, &followed_by_);
    CAFFE_ENFORCE_EQ(
        dequantize_output,
        false,
        "Conv DNNLOWP operators don't support dequantize_output");

    if (ReluFused) {
      // It's actually fused with Relu not followed by but setting this to make
      // sure quantization error is correctly measured in
      // this->MeasureQuantizationError_
      followed_by_ = "Relu";
      AdjustOutputTensorQuantizationParamsWithFollowedBy(this, followed_by_);
    }
    this->arguments_parsed_ = true;
  }

  // Choose quantization for X
  in_qparams_[INPUT] =
      GetInputTensorQuantizationParamsOf(this, INPUT, qfactory_.get());

  QuantizeWeight_();
  PreComputeRowColumnOffsets_();
  QuantizeBias_();

  if (Wq_packed_ && !FLAGS_caffe2_dnnlowp_dump_tensors) {
    // From here, W_quantized_ is not used anymore when we have Wq_packed_
    vector<T_signed>().swap(W_quantized_);
  }

  bool fp32_executed = false;
  if (HasStaticQuantization(this)) {
    out_qparams_ = GetStaticQuantizationParamsOf(this, 0);
  } else {
    // If quantization parameters are not chosen beforehand, run reference
    // Conv op in fp32 to choose quantization for Y.
    Fp32Op_()->DequantizeInput();
    Fp32Op_()->Get()->RunOnDevice();
    out_qparams_ = Fp32Op_()->GetOutputQuantizationParams(qfactory_.get());
    fp32_executed = true;
  }

  // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
  for (int g = 0; g < filter_qparams_.size(); ++g) {
    float real_multiplier = in_qparams_[INPUT].scale *
        FilterQuantizationParams(g).scale / out_qparams_.scale;
    requantization_params_[g] = qfactory_->ChooseRequantizationMultiplier(
        real_multiplier, out_qparams_);
    requantization_multipliers_[g] = requantization_params_[g].real_multiplier;
  }

  if (measure_quantization_error_ && Fp32Op_() && !fp32_executed) {
    // to measure quantization error, run ref impl.
    Fp32Op_()->DequantizeInput();
    Fp32Op_()->Get()->RunOnDevice();
  }

  return true;
}

template <typename T, bool ReluFused>
void ConvDNNLowPOp<T, ReluFused>::RunOnDeviceEpilogueNCHW_(
    const T* col_buffer_data,
    int32_t* Y_int32,
    T* Y_data,
    size_t i_offset,
    int group_id) {
  auto& filter = InputTensorCPU_(FILTER);
  const int M = filter.dim32(0);
  int kernel_dim = KernelDim_();

  Tensor* Y = OutputTensorCPU_(0);
  const int Y_HxW = this->GetDimsSize(*Y);

  // See batch_matmul_dnnlowp_op.cc to why we compute column_offsets,
  // row_offset, and const_offset in this way.
  int tid = dnnlowp_get_thread_num();
  int32_t* column_offsets = column_offsets_->data() + tid * Y_HxW;

  const dnnlowp::TensorQuantizationParams& filter_qparams =
      FilterQuantizationParams(group_id);
  for (int j = 0; j < Y_HxW; ++j) {
    int sum = 0;
    for (int k = 0; k < kernel_dim; ++k) {
      sum += col_buffer_data[k * Y_HxW + j];
    }
    column_offsets[j] = sum * filter_qparams.zero_point;
  }

  for (int i = 0; i < M / group_; ++i) {
    int32_t row_offset = row_offsets_[i_offset + i];
    row_offset *= -in_qparams_[INPUT].zero_point;
    if (b_quantized_data_) {
      row_offset += b_quantized_data_[i_offset + i];
    }
    for (int j = 0; j < Y_HxW; ++j) {
      int32_t raw = Y_int32[i * Y_HxW + j] + row_offset - column_offsets[j];
      if (ReluFused) {
        raw = std::max(0, raw);
      }
      Y_data[i * Y_HxW + j] =
          fbgemm::Requantize<T>(raw, RequantizationParams(group_id));
    }
  }
}

template <typename T, bool ReluFused>
bool ConvDNNLowPOp<T, ReluFused>::RunOnDeviceWithOrderNCHW() {
  VLOG(2) << "Running DNNLOWP Conv";

  using namespace dnnlowp;

  // Get quantization parameters
  if (!GetQuantizationParameters_()) {
    return false;
  }

  const Tensor& X = InputTensorCPU_(INPUT);
  auto& filter = InputTensorCPU_(FILTER);
  const int N = X.dim32(0), C = X.dim32(1);
  CAFFE_ENFORCE_EQ(X.dim(), filter.dim());
  const int M = filter.dim32(0);
  CAFFE_ENFORCE_EQ(
      C,
      filter.dim32(1) * group_,
      "Convolution op: input channels does not match: # of input channels ",
      C,
      " is not equal to kernel channels * group:",
      filter.dim32(1),
      "*",
      group_);
  CAFFE_ENFORCE_EQ(
      M % group_,
      0,
      "The number of output channels is not divisible by group.");

  auto sizes = ConvPoolOpBase<CPUContext>::GetOutputSize(X, filter.dim32(0));
  Tensor* Y = OutputTensorCPU_(0, sizes, at::dtype<T>());

  const vector<int> input_dims = GetDims(X);
  const vector<int> output_dims = GetDims(*Y);
  const int X_HxW = this->GetDimsSize(X);
  const int Y_HxW = this->GetDimsSize(*Y);

  // The dimension of each kernel
  const int kernel_dim = KernelDim_();

  vector<int> img_shape;
  img_shape.assign(X.sizes().begin() + 1, X.sizes().end());

  vector<int> buffer_shape;
  buffer_shape.push_back(kernel_dim);
  buffer_shape.insert(
      buffer_shape.end(), output_dims.begin(), output_dims.end());
  buffer_shape.insert(buffer_shape.begin(), dnnlowp_get_max_threads());

  if (this->kernel_.size() != 2) {
    SetDeviceTensor(img_shape, &img_shape_device_);
    SetDeviceTensor(buffer_shape, &col_buffer_shape_device_);
  }

  const int col_buffer_size = kernel_dim * Y_HxW;

  // The offset corresponding to a single input image, and a single output
  // image.
  const int input_offset = C / group_ * X_HxW;

  // The col buffer is stored in CHW order as well - kernel_dim, and the
  // height and width.
  const T* Xdata = X.template data<T>();

  // We must not call mutable_data inside omp region
  T* Y_data_T = Y->template mutable_data<T>();
  column_offsets_->resize(Y_HxW * dnnlowp_get_max_threads());

  auto f = [&](Tensor* col_buffer, vector<int32_t>* Y_int32) {
    col_buffer->Resize(buffer_shape);
    vector<int> buffer_shape_per_thread(
        buffer_shape.begin() + 1, buffer_shape.end());
    T* col_buffer_data = col_buffer->template mutable_data<T>();

    Y_int32->resize(M * Y_HxW * dnnlowp_get_max_threads());

    // Im2Col, followed by gemm.
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int image_id = 0; image_id < N; ++image_id) {
      int tid = dnnlowp_get_thread_num();
      for (int group_id = 0; group_id < group_; ++group_id) {
        if (this->kernel_.size() == 2) {
          math::Im2ColNCHW<T>(
              C / group_,
              input_dims[0],
              input_dims[1],
              kernel_h(),
              kernel_w(),
              dilation_h(),
              dilation_w(),
              pad_t(),
              pad_l(),
              pad_b(),
              pad_r(),
              stride_h(),
              stride_w(),
              Xdata + (group_ * image_id + group_id) * input_offset,
              col_buffer_data + tid * col_buffer_size,
              &context_,
              in_qparams_[INPUT].zero_point);
        } else {
          math::Im2ColNdNCHW<T>(
              this->kernel_.size(),
              C * X_HxW,
              col_buffer_size,
              img_shape.data(),
              buffer_shape_per_thread.data(),
              this->kernel_.data(),
              this->stride_.data(),
              this->dilation_.data(),
              this->pads_.data(),
              Xdata + (group_ * image_id + group_id) * input_offset,
              col_buffer_data + tid * col_buffer_size,
              &context_,
              in_qparams_[INPUT].zero_point);
        }

        // quantize col_buffer
        T* col_buffer_private = col_buffer_data + tid * col_buffer_size;

        int32_t* Y_int32_temp =
            Y_int32->data() + ((M / group_) * group_id + M * tid) * Y_HxW;
        T_signed* W_quantized_group =
            W_quantized_.data() + (M / group_) * group_id * kernel_dim;

        for (int i = 0; i < M / group_; ++i) {
          for (int j = 0; j < Y_HxW; ++j) {
            int32_t sum = 0;
            for (int k = 0; k < kernel_dim; ++k) {
              // NOLINTNEXTLINE(bugprone-signed-char-misuse)
              int w = W_quantized_group[i * kernel_dim + k];
              int x = col_buffer_private[k * Y_HxW + j];
              sum += w * x;
            }
            Y_int32_temp[i * Y_HxW + j] = sum;
          } // j
        } // i

        RunOnDeviceEpilogueNCHW_(
            col_buffer_private,
            Y_int32_temp,
            Y_data_T + (M * image_id + M / group_ * group_id) * Y_HxW,
            M / group_ * group_id,
            group_id);
      } // for each group
    } // for each image_id

    PropagateOutputTensorQuantizationParams(this, 0, out_qparams_);
    MeasureQuantizationError_();
  }; // f

  this->RunWithSharedBuffer_(&col_buffer_, &Y_int32_, f);

  return true;
} // RunOnDeviceWithOrderNCHW

template <typename T, bool ReluFused>
void ConvDNNLowPOp<T, ReluFused>::RunOnDeviceEpilogueNHWC_(
    const T* col_buffer_data,
    int32_t* Y_int32) {
  const Tensor& X = InputTensorCPU_(INPUT);
  auto& filter = InputTensorCPU_(FILTER);
  Tensor* Y = OutputTensorCPU_(0);
  const int N = X.dim32(0);
  const int M = filter.dim32(0);
  int kernel_dim = KernelDim_();
  const int Y_HxW = this->GetDimsSize(*Y);

  // Adjust with bias and zero_point and then requantize
  // See batch_matmul_dnnlowp_op.cc to why we compute column_offsets,
  // row_offset, and const_offset in this way.
  int32_t A_zero_point = in_qparams_[INPUT].zero_point;

  if (!dnnlowp::HasStaticQuantization(this)) {
    if (quantize_groupwise_) {
      static int log_occurences = 0;
      if (log_occurences < 32) {
        ++log_occurences;
        LOG(WARNING) << "Cannot do group-wise quantization without "
                        "static quantization of activations for "
                     << this->debug_def().output(0);
      }
    }

    int32_t Y_min = numeric_limits<int32_t>::max();
    int32_t Y_max = numeric_limits<int32_t>::min();

#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp parallel for reduction(min : Y_min), reduction(max : Y_max)
#endif
    for (int i = 0; i < N * Y_HxW; ++i) {
      for (int group_id = 0; group_id < group_; ++group_id) {
        int32_t row_offset = 0;
        for (int k = 0; k < kernel_dim; ++k) {
          row_offset +=
              col_buffer_data[(i * group_ + group_id) * kernel_dim + k];
        }
        row_offset *= FilterQuantizationParams(0).zero_point;

        for (int j = group_id * (M / group_); j < (group_id + 1) * (M / group_);
             ++j) {
          int32_t raw = Y_int32[i * M + j] - row_offset;
          if (!column_offsets_->empty()) {
            raw -= A_zero_point * (*column_offsets_)[j];
          }
          if (b_quantized_data_) {
            raw += b_quantized_data_[j];
          }
          Y_min = std::min(Y_min, raw);
          Y_max = std::max(Y_max, raw);
        }
      } // for each group
    } // for each row i

    if (ReluFused) {
      Y_min = std::max(0, Y_min);
      Y_max = std::max(0, Y_max);
    }

    float Y_scale =
        in_qparams_[INPUT].scale * FilterQuantizationParams(0).scale;
    out_qparams_ =
        qfactory_->ChooseQuantizationParams(Y_scale * Y_min, Y_scale * Y_max);

    float real_multiplier = Y_scale / out_qparams_.scale;
    requantization_params_[0] = qfactory_->ChooseRequantizationMultiplier(
        real_multiplier, out_qparams_);
    requantization_multipliers_[0] = requantization_params_[0].real_multiplier;
  }

  int32_t C_zero_point = out_qparams_.zero_point;

  T* Ydata = Y->template mutable_data<T>();

  using namespace fbgemm;
  if (is_same<T, uint8_t>::value && GetCpuId().avx2()) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < N * Y_HxW; ++i) {
      for (int group_id = 0; group_id < group_; ++group_id) {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        int32_t row_offset;
        row_offsets_u8acc32_ref(
            1,
            kernel_dim,
            group_ * kernel_dim,
            reinterpret_cast<const uint8_t*>(
                col_buffer_data + (i * group_ + group_id) * kernel_dim),
            &row_offset);

        int32_t B_zero_point = FilterQuantizationParams(group_id).zero_point;
        float C_multiplier = RequantizationParams(group_id).real_multiplier;

        requantize_u8acc32_ref(
            1,
            M / group_,
            M,
            Y_int32 + i * M + group_id * (M / group_),
            reinterpret_cast<uint8_t*>(Ydata + i * M + group_id * (M / group_)),
            &C_multiplier,
            C_zero_point,
            column_offsets_->empty() ? 0 : A_zero_point,
            &B_zero_point,
            &row_offset,
            column_offsets_->empty()
                ? nullptr
                : column_offsets_->data() + group_id * (M / group_),
            b_quantized_data_ ? b_quantized_data_ + group_id * (M / group_)
                              : nullptr,
            M / group_,
            ReluFused);
      } // for each group
    } // for each row i
  } else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < N * Y_HxW; ++i) {
      for (int group_id = 0; group_id < group_; ++group_id) {
        int32_t B_zero_point = FilterQuantizationParams(group_id).zero_point;
        int32_t row_offset = 0;
        for (int k = 0; k < kernel_dim; ++k) {
          row_offset +=
              col_buffer_data[(i * group_ + group_id) * kernel_dim + k];
        }
        row_offset *= B_zero_point;

        for (int j = group_id * (M / group_); j < (group_id + 1) * (M / group_);
             ++j) {
          int32_t raw = Y_int32[i * M + j] - row_offset;
          if (!column_offsets_->empty()) {
            raw -= A_zero_point * (*column_offsets_)[j];
          }
          if (b_quantized_data_) {
            raw += b_quantized_data_[j];
          }

          Ydata[i * M + j] =
              fbgemm::Requantize<T>(raw, RequantizationParams(group_id));
          if (ReluFused) { // static if
            Ydata[i * M + j] =
                std::max<int32_t>(C_zero_point, Ydata[i * M + j]);
          }
        }
      } // for each group
    } // for each row i
  } // !__AVX2__

  dnnlowp::PropagateOutputTensorQuantizationParams(this, 0, out_qparams_);
}

template <typename T, bool ReluFused>
void ConvDNNLowPOp<T, ReluFused>::PartitionGroupedNHWCConv_(
    int* group_begin,
    int* group_end,
    int* i_begin,
    int* i_end,
    int num_groups,
    int m,
    int nthreads,
    int thread_id) {
  // Make sure i_per_thread is a multiple of 32 because
  // cblas_gemm_compute_u8s8s32_acc16 performs the best when M is a multiple
  // of 32.
  Get1DPartitionOf2D(
      num_groups,
      m,
      nthreads,
      thread_id,
      group_begin,
      group_end,
      i_begin,
      i_end,
      32);
}

template <typename T, bool ReluFused>
const T* ConvDNNLowPOp<T, ReluFused>::Im2ColNHWC_(Tensor* col_buffer) {
  const Tensor& X = InputTensorCPU_(INPUT);
  Tensor* Y = OutputTensorCPU_(0);
  int ndim = X.dim();
  const int N = X.dim32(0), C = X.dim32(ndim - 1);

  const int kernel_dim = KernelDim_();
  // The offset corresponding to a single input image, and a single output
  // image.
  const int X_HxW = this->GetDimsSize(X);
  const int input_offset = X_HxW * C;
  const int Y_HxW = this->GetDimsSize(*Y);

  const T* Xdata = X.template data<T>();

  vector<int> buffer_shape(ndim);
  for (auto i = 0; i < ndim - 1; ++i) {
    buffer_shape[i] = Y->dim32(i);
  }
  buffer_shape[ndim - 1] = kernel_dim * group_;

  col_buffer->Resize(buffer_shape);

  T* col_buffer_data = col_buffer->template mutable_data<T>();

#ifdef _OPENMP
#pragma omp parallel for if (N > 1)
#endif
  for (int image_id = 0; image_id < N; ++image_id) {
    if (this->kernel_.size() <= 2) {
      math::Im2ColNHWC<T>(
          C,
          X.dim32(1),
          this->kernel_.size() == 2 ? X.dim32(2) : 1,
          kernel_h(),
          this->kernel_.size() == 2 ? kernel_w() : 1,
          dilation_h(),
          this->kernel_.size() == 2 ? dilation_w() : 1,
          pad_t(),
          this->kernel_.size() == 2 ? pad_l() : 0,
          this->kernel_.size() == 2 ? pad_b() : pad_l(),
          this->kernel_.size() == 2 ? pad_r() : 0,
          stride_h(),
          this->kernel_.size() == 2 ? stride_w() : 1,
          Xdata + image_id * input_offset,
          col_buffer_data + image_id * group_ * kernel_dim * Y_HxW,
          &context_,
          group_,
          in_qparams_[INPUT].zero_point);
    } else {
      math::Im2Col3DNHWC<T>(
          C,
          X.dim32(1), // num_frames
          X.dim32(2), // H
          X.dim32(3), // W
          this->kernel_[0],
          this->kernel_[1],
          this->kernel_[2],
          this->dilation_[0],
          this->dilation_[1],
          this->dilation_[2],
          this->pads_[0],
          this->pads_[1],
          this->pads_[2],
          this->pads_[3],
          this->pads_[4],
          this->pads_[5],
          this->stride_[0],
          this->stride_[1],
          this->stride_[2],
          Xdata + image_id * input_offset,
          col_buffer_data + image_id * group_ * kernel_dim * Y_HxW,
          &context_,
          group_,
          in_qparams_[INPUT].zero_point);
    }
  }

  return col_buffer->template data<T>();
}

template <typename T, typename T_signed>
static void conv_nhwc_ref_(
    int group_id,
    int num_groups,
    int i_begin,
    int i_end,
    int M,
    int kernel_dim,
    const T* col_buffer,
    const T_signed* W,
    int32_t* Y) {
  for (int i = i_begin; i < i_end; ++i) {
    for (int j = group_id * (M / num_groups);
         j < (group_id + 1) * (M / num_groups);
         ++j) {
      int32_t sum = 0;
      for (int k = 0; k < kernel_dim; ++k) {
        // NOLINTNEXTLINE(bugprone-signed-char-misuse)
        int w = W[j * kernel_dim + k];
        int x = col_buffer[(i * num_groups + group_id) * kernel_dim + k];
        sum += w * x;
      }
      Y[i * M + j] = sum;
    }
  }
}

template <typename T, bool ReluFused>
template <typename PackAMatrix, fbgemm::QuantizationGranularity Q_GRAN>
void ConvDNNLowPOp<T, ReluFused>::DispatchFBGEMM_(
    PackAMatrix& packA,
    vector<int32_t>* Y_int32,
    uint8_t* Y_uint8_data) {
  // This function is called within an OpenMP region
  auto& filter = InputTensorCPU_(FILTER);
  const int M = filter.dim32(0);

  int nthreads = dnnlowp_get_num_threads();
  int tid = dnnlowp_get_thread_num();

  using namespace fbgemm;
  DoNothing<> doNothingObj{};
  ReQuantizeOutput<ReluFused, Q_GRAN> outputProcObj(
      doNothingObj,
      requantization_multipliers_.data(),
      out_qparams_.zero_point,
      // column_offsets_ empty means column_offsets_ are folded into bias
      column_offsets_->empty() ? 0 : in_qparams_[INPUT].zero_point,
      filter_zero_points_.data(),
      packA.getRowOffsetBuffer(),
      column_offsets_->empty() ? nullptr : column_offsets_->data(),
      b_quantized_data_,
      M,
      group_);

  fbgemmPacked(
      packA,
      *Wq_packed_,
      Y_uint8_data,
      Y_int32->data(),
      M,
      outputProcObj,
      tid,
      nthreads);
}

template <typename T, bool ReluFused>
void ConvDNNLowPOp<T, ReluFused>::ConvNHWCCore_(
    const T* col_buffer_data,
    vector<int32_t>* Y_int32) {
  const Tensor& X = InputTensorCPU_(INPUT);
  auto& filter = InputTensorCPU_(FILTER);
  Tensor* Y = OutputTensorCPU_(0);
  const int N = X.dim32(0), C = X.dim32(X.dim() - 1);
  const int M = filter.dim32(0);
  const int kernel_dim = KernelDim_();
  const int Y_HxW = this->GetDimsSize(*Y);
  const int X_HxW = this->GetDimsSize(X);

  if (N == 0) {
    LOG(WARNING) << "The batch size is 0 in ConvNHWCCore_ function!";
  }

  if (FLAGS_caffe2_dnnlowp_dump_tensors) {
    // Dump input activation
    std::string input_name = this->debug_def().input(INPUT);
    std::string input_filename = input_name;
    while (input_filename.find('/') != std::string::npos) {
      input_filename.replace(input_filename.find('/'), 1, "_");
    }
    StoreMatrixInMatrixMarketFormat(
        N * X_HxW * C / kernel_dim,
        kernel_dim,
        col_buffer_data,
        input_filename);

    // Dump weight
    std::string weight_name = this->debug_def().input(FILTER);
    std::string weight_filename = weight_name;
    while (weight_filename.find('/') != std::string::npos) {
      weight_filename.replace(weight_name.find('/'), 1, "_");
    }
    StoreMatrixInMatrixMarketFormat(
        M, kernel_dim, W_quantized_.data(), weight_filename);
  }

  using namespace fbgemm;

  if (TakeDepthWise3x3x3FastPath_()) {
    const T* Xdata = X.template data<T>();
    uint8_t* Y_uint8_data =
        OutputTensorCPU_(0)->template mutable_data<uint8_t>();

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      conv_param_t<3> conv_p(
          N,
          C,
          C,
          {X.dim32(1), X.dim32(2), X.dim32(3)},
          C,
          {3, 3, 3},
          {this->stride_[0], this->stride_[1], this->stride_[2]},
          {1, 1, 1, 1, 1, 1});
      if (quantize_groupwise_) {
        depthwise_3d_same_pad<QuantizationGranularity::OUT_CHANNEL>(
            conv_p,
            // Shouldn't pass 0 if column_offsets_ is empty here because we
            // need zero_point for padding
            in_qparams_[INPUT].zero_point,
            reinterpret_cast<const uint8_t*>(Xdata),
            filter_zero_points_.data(),
            *Wq_depthwise_packed_,
            requantization_multipliers_.data(),
            out_qparams_.zero_point,
            Y_uint8_data,
            // column_offsets_ empty means column_offsets_ are folded into bias
            column_offsets_->empty() ? nullptr : column_offsets_->data(),
            b_quantized_data_,
            ReluFused,
            nullptr, /*act_times_w_scale*/
            dnnlowp_get_thread_num(),
            dnnlowp_get_num_threads());
      } else {
        depthwise_3d_same_pad<QuantizationGranularity::TENSOR>(
            conv_p,
            // Shouldn't pass 0 if column_offsets_ is empty here because we
            // need zero_point for padding
            in_qparams_[INPUT].zero_point,
            reinterpret_cast<const uint8_t*>(Xdata),
            &FilterQuantizationParams(0).zero_point,
            *Wq_depthwise_packed_,
            &requantization_params_[0].real_multiplier,
            out_qparams_.zero_point,
            Y_uint8_data,
            // column_offsets_ empty means column_offsets_ are folded into bias
            column_offsets_->empty() ? nullptr : column_offsets_->data(),
            b_quantized_data_,
            ReluFused,
            nullptr, /*act_times_w_scale*/
            dnnlowp_get_thread_num(),
            dnnlowp_get_num_threads());
      }
    } // omp parallel

    return;
  } else if (TakeDepthWise3x3FastPath_()) {
    const int H = X.dim32(1), W = X.dim32(2);
    const T* Xdata = X.template data<T>();
    uint8_t* Y_uint8_data =
        OutputTensorCPU_(0)->template mutable_data<uint8_t>();

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      if (quantize_groupwise_) {
        depthwise_2d_same_pad<QuantizationGranularity::OUT_CHANNEL>(
            N,
            H,
            W,
            C,
            C,
            stride_h(),
            stride_w(),
            // Shouldn't pass 0 if column_offsets_ is empty here because we
            // need zero_point for padding
            in_qparams_[INPUT].zero_point,
            reinterpret_cast<const uint8_t*>(Xdata),
            filter_zero_points_.data(),
            *Wq_depthwise_packed_,
            requantization_multipliers_.data(),
            out_qparams_.zero_point,
            Y_uint8_data,
            // column_offsets_ empty means column_offsets_ are folded into bias
            column_offsets_->empty() ? nullptr : column_offsets_->data(),
            b_quantized_data_,
            ReluFused,
            nullptr, /*act_times_w_scale*/
            dnnlowp_get_thread_num(),
            dnnlowp_get_num_threads());
      } else {
        depthwise_2d_same_pad<QuantizationGranularity::TENSOR>(
            N,
            H,
            W,
            C,
            C,
            stride_h(),
            stride_w(),
            // Shouldn't pass 0 if column_offsets_ is empty here because we
            // need zero_point for padding
            in_qparams_[INPUT].zero_point,
            reinterpret_cast<const uint8_t*>(Xdata),
            &FilterQuantizationParams(0).zero_point,
            *Wq_depthwise_packed_,
            &requantization_params_[0].real_multiplier,
            out_qparams_.zero_point,
            Y_uint8_data,
            // column_offsets_ empty means column_offsets_ are folded into bias
            column_offsets_->empty() ? nullptr : column_offsets_->data(),
            b_quantized_data_,
            ReluFused,
            nullptr, /*act_times_w_scale*/
            dnnlowp_get_thread_num(),
            dnnlowp_get_num_threads());
      }
    } // omp parallel

    return;
  } else if (TakeGConvFastPath_()) {
    const T* Xdata = X.template data<T>();
    uint8_t* Y_uint8_data =
        OutputTensorCPU_(0)->template mutable_data<uint8_t>();

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int row_offset_size_per_thread;
    if (this->kernel_.size() == 2) {
      row_offset_size_per_thread = rowOffsetBufferSizeGConv(GetConvParam_());
    } else {
      CAFFE_ENFORCE_EQ(this->kernel_.size(), 3);
      row_offset_size_per_thread = rowOffsetBufferSizeGConv(GetConv3DParam_());
    }
    row_offsets_.resize(dnnlowp_get_max_threads() * row_offset_size_per_thread);

#ifdef _OPENMP
// TODO: add parallelization once fbgemmGroupwiseConv supports multi-threading
// #pragma omp parallel
#endif
    {
      int tid = 0; // dnnlowp_get_thread_num();
      int nthreads = 1; // dnnlowp_get_num_threads();

      DoNothing<> doNothingObj{};
      if (quantize_groupwise_) {
        ReQuantizeOutput<false, QuantizationGranularity::GROUP> reqObj(
            doNothingObj,
            requantization_multipliers_.data(),
            out_qparams_.zero_point,
            // column_offsets_ empty means column_offsets_ are folded into bias
            column_offsets_->empty() ? 0 : in_qparams_[INPUT].zero_point,
            filter_zero_points_.data(),
            row_offsets_.data() + tid * row_offset_size_per_thread,
            column_offsets_->empty() ? nullptr : column_offsets_->data(),
            b_quantized_data_,
            M,
            group_);

        if (this->kernel_.size() == 2) {
          fbgemmGroupwiseConv(
              GetConvParam_(),
              reinterpret_cast<const uint8_t*>(Xdata),
              // Shouldn't pass 0 if column_offsets_ is empty here because we
              // need zero_point for padding
              in_qparams_[INPUT].zero_point,
              row_offsets_.data() + tid * row_offset_size_per_thread,
              *Wq_gconv_packed_,
              Y_uint8_data,
              Y_int32->data(),
              reqObj,
              tid,
              nthreads);
        } else {
          CAFFE_ENFORCE_EQ(this->kernel_.size(), 3);
          fbgemmGroupwiseConv(
              GetConv3DParam_(),
              reinterpret_cast<const uint8_t*>(Xdata),
              // Shouldn't pass 0 if column_offsets_ is empty here because we
              // need zero_point for padding
              in_qparams_[INPUT].zero_point,
              row_offsets_.data() + tid * row_offset_size_per_thread,
              *Wq_gconv3d_packed_,
              Y_uint8_data,
              Y_int32->data(),
              reqObj,
              tid,
              nthreads);
        }
      } else {
        ReQuantizeOutput<false, QuantizationGranularity::TENSOR> reqObj(
            doNothingObj,
            requantization_multipliers_.data(),
            out_qparams_.zero_point,
            // column_offsets_ empty means column_offsets_ are folded into bias
            column_offsets_->empty() ? 0 : in_qparams_[INPUT].zero_point,
            filter_zero_points_.data(),
            filter_zero_points_[0]
                ? row_offsets_.data() + tid * row_offset_size_per_thread
                : nullptr,
            column_offsets_->empty() ? nullptr : column_offsets_->data(),
            b_quantized_data_,
            M,
            group_);

        if (this->kernel_.size() == 2) {
          fbgemmGroupwiseConv(
              GetConvParam_(),
              reinterpret_cast<const uint8_t*>(Xdata),
              // Shouldn't pass 0 if column_offsets_ is empty here because we
              // need zero_point for padding
              in_qparams_[INPUT].zero_point,
              filter_zero_points_[0]
                  ? row_offsets_.data() + tid * row_offset_size_per_thread
                  : nullptr,
              *Wq_gconv_packed_,
              Y_uint8_data,
              Y_int32->data(),
              reqObj,
              tid,
              nthreads);
        } else {
          fbgemmGroupwiseConv(
              GetConv3DParam_(),
              reinterpret_cast<const uint8_t*>(Xdata),
              // Shouldn't pass 0 if column_offsets_ is empty here because we
              // need zero_point for padding
              in_qparams_[INPUT].zero_point,
              filter_zero_points_[0]
                  ? row_offsets_.data() + tid * row_offset_size_per_thread
                  : nullptr,
              *Wq_gconv3d_packed_,
              Y_uint8_data,
              Y_int32->data(),
              reqObj,
              tid,
              nthreads);
        }
      }
    } // omp parallel

    return;
  }

  // Normal path for non-special (e.g., no depth-wise) convolutions.
  int row_offset_size_per_thread = -1;
  int x_pack_buf_size_per_thread = -1;
  bool fuse_im2col =
      Wq_packed_ && X.template data<T>() == col_buffer_data && !IsConvGEMM_();
  if (Wq_packed_) {
    if (fuse_im2col) {
      row_offset_size_per_thread =
          PackAWithIm2Col<uint8_t>::rowOffsetBufferSize();
      x_pack_buf_size_per_thread = PackAWithIm2Col<uint8_t>::packedBufferSize();
    } else if (!quantize_groupwise_ && filter_zero_points_[0] == 0) {
      row_offset_size_per_thread = 0;
      x_pack_buf_size_per_thread = PackAMatrix<uint8_t>::packedBufferSize();
    } else {
      row_offset_size_per_thread =
          PackAWithRowOffset<uint8_t>::rowOffsetBufferSize();
      x_pack_buf_size_per_thread =
          PackAWithRowOffset<uint8_t>::packedBufferSize();
    }
    row_offsets_.resize(dnnlowp_get_max_threads() * row_offset_size_per_thread);
    X_pack_buf_.resize(dnnlowp_get_max_threads() * x_pack_buf_size_per_thread);
  }

  uint8_t* Y_uint8_data = Y->template mutable_data<uint8_t>();

  if (Wq_packed_)
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    int tid = dnnlowp_get_thread_num();

    // fast path to use fbgemm
    if (fuse_im2col) {
      if (this->kernel_.size() <= 2) {
        conv_param_t<> conv_p(
            N,
            C,
            M,
            {X.dim32(1), this->kernel_.size() == 2 ? X.dim32(2) : 1},
            group_,
            {this->kernel_[0],
             this->kernel_.size() == 2 ? this->kernel_[1] : 1},
            {this->stride_[0],
             this->kernel_.size() == 2 ? this->stride_[1] : 1},
            {this->pads_[0],
             this->kernel_.size() == 2 ? this->pads_[1] : 0,
             this->kernel_.size() == 2 ? this->pads_[2] : this->pads_[1],
             this->kernel_.size() == 2 ? this->pads_[3] : 0});

        PackAWithIm2Col<uint8_t> packA(
            conv_p,
            reinterpret_cast<const uint8_t*>(col_buffer_data),
            // buffer for packed matrix
            X_pack_buf_.data() + tid * x_pack_buf_size_per_thread,
            // Shouldn't pass 0 if column_offsets_ is empty here because we
            // need zero_point for padding
            in_qparams_[INPUT].zero_point,
            row_offsets_.data() + tid * row_offset_size_per_thread);

        if (quantize_groupwise_) {
          DispatchFBGEMM_<
              PackAWithIm2Col<uint8_t>,
              QuantizationGranularity::GROUP>(packA, Y_int32, Y_uint8_data);
        } else {
          DispatchFBGEMM_<
              PackAWithIm2Col<uint8_t>,
              QuantizationGranularity::TENSOR>(packA, Y_int32, Y_uint8_data);
        }
      } else {
        // 3D
        conv_param_t<3> conv_p(
            N,
            C,
            M,
            {X.dim32(1), X.dim32(2), X.dim32(3)},
            group_,
            {this->kernel_[0], this->kernel_[1], this->kernel_[2]},
            {this->stride_[0], this->stride_[1], this->stride_[2]},
            {this->pads_[0],
             this->pads_[1],
             this->pads_[2],
             this->pads_[3],
             this->pads_[4],
             this->pads_[5]});

        PackAWithIm2Col<uint8_t, int32_t, 3> packA(
            conv_p,
            reinterpret_cast<const uint8_t*>(col_buffer_data),
            // buffer for packed matrix
            X_pack_buf_.data() + tid * x_pack_buf_size_per_thread,
            // Shouldn't pass 0 if column_offsets_ is empty here because we
            // need zero_point for padding
            in_qparams_[INPUT].zero_point,
            row_offsets_.data() + tid * row_offset_size_per_thread);

        if (quantize_groupwise_) {
          DispatchFBGEMM_<
              PackAWithIm2Col<uint8_t, int32_t, 3>,
              QuantizationGranularity::GROUP>(packA, Y_int32, Y_uint8_data);
        } else {
          DispatchFBGEMM_<
              PackAWithIm2Col<uint8_t, int32_t, 3>,
              QuantizationGranularity::TENSOR>(packA, Y_int32, Y_uint8_data);
        }
      } // 3D
    } else if (!quantize_groupwise_ && filter_zero_points_[0] == 0) {
      // no im2col fusion
      PackAMatrix<uint8_t> packA(
          matrix_op_t::NoTranspose,
          N * Y_HxW,
          group_ * kernel_dim,
          reinterpret_cast<const uint8_t*>(col_buffer_data),
          group_ * kernel_dim,
          // buffer for packed matrix
          X_pack_buf_.data() + tid * x_pack_buf_size_per_thread,
          group_);

      DispatchFBGEMM_<PackAMatrix<uint8_t>, QuantizationGranularity::TENSOR>(
          packA, Y_int32, Y_uint8_data);
    } else {
      // no im2col fusion
      PackAWithRowOffset<uint8_t> packA(
          matrix_op_t::NoTranspose,
          N * Y_HxW,
          group_ * kernel_dim,
          reinterpret_cast<const uint8_t*>(col_buffer_data),
          group_ * kernel_dim,
          // buffer for packed matrix
          X_pack_buf_.data() + tid * x_pack_buf_size_per_thread,
          group_,
          row_offsets_.data() + tid * row_offset_size_per_thread);

      if (quantize_groupwise_) {
        DispatchFBGEMM_<
            PackAWithRowOffset<uint8_t>,
            QuantizationGranularity::GROUP>(packA, Y_int32, Y_uint8_data);
      } else {
        DispatchFBGEMM_<
            PackAWithRowOffset<uint8_t>,
            QuantizationGranularity::TENSOR>(packA, Y_int32, Y_uint8_data);
      }
    } // no im2col fusion
  } else {
    for (int group_id = 0; group_id < group_; ++group_id) {
      // Wq_packed_.empty()
      conv_nhwc_ref_(
          group_id,
          group_,
          0,
          N * Y_HxW,
          M,
          kernel_dim,
          col_buffer_data,
          W_quantized_.data(),
          Y_int32->data());
    }
  }
}

template <typename T, bool ReluFused>
bool ConvDNNLowPOp<T, ReluFused>::RunOnDeviceWithOrderNHWC() {
  CAFFE_ENFORCE_LE(
      this->kernel_.size(),
      3,
      "Only 1-3d convolutions are supported for NHWC storage type");

  using namespace dnnlowp;

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
  chrono::time_point<chrono::system_clock> t_very_begin, t_begin, t_end;
  /*if (VLOG_IS_ON(3))*/ {
    t_begin = chrono::system_clock::now();
    t_very_begin = t_begin;
  }
#endif

  // Get quantization parameters
  if (!GetQuantizationParameters_()) {
    return false;
  }

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
  /*if (VLOG_IS_ON(3))*/ {
    t_end = chrono::system_clock::now();
    double dt = chrono::duration<double>(t_end - t_begin).count();
    LOG(INFO) << "this=" << this << " get_quant_params: " << dt * 1e3 << " ms";
  }
#endif

  const Tensor& X = InputTensorCPU_(INPUT);
  auto& filter = InputTensorCPU_(FILTER);
  const int C = X.dim32(X.dim() - 1);
  const int G = group_;
  CAFFE_ENFORCE_EQ(X.dim(), filter.dim());
  const int M = filter.dim32(0);
  CAFFE_ENFORCE_EQ(
      C,
      filter.dim32(filter.dim() - 1) * G,
      "Convolution op: input channels does not match: # of input channels ",
      C,
      " is not equal to kernel channels * group: ",
      filter.dim32(filter.dim() - 1),
      "*",
      G);
  CAFFE_ENFORCE_EQ(
      M % G, 0, "The number of output channels is not divisible by group.");

  auto sizes = ConvPoolOpBase<CPUContext>::GetOutputSize(X, filter.dim32(0));
  Tensor* Y = OutputTensorCPU_(0, sizes, at::dtype<T>());

  // The col buffer is stored in HWC order as well - kernel_dim, and the height
  // and width.

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
  /*if (VLOG_IS_ON(3)) */ { t_begin = chrono::system_clock::now(); }
#endif

  bool no_im2col = NoIm2ColNHWC_();
  auto f = [&](Tensor* col_buffer, vector<int32_t>* Y_int32) {
    if (!TakeDepthWise3x3FastPath_() && !TakeDepthWise3x3x3FastPath_()) {
      Y_int32->resize(Y->numel());
    }

    // Im2col, followed by gemm.
    const T* Xdata = X.template data<T>();
    const T* col_buffer_data = no_im2col ? Xdata : Im2ColNHWC_(col_buffer);

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
    /*if (VLOG_IS_ON(3)) */ {
      t_end = chrono::system_clock::now();
      double dt = chrono::duration<double>(t_end - t_begin).count();
      LOG(INFO) << "this=" << this << " im2col: " << dt * 1e3 << " ms";
      t_begin = chrono::system_clock::now();
    }
#endif

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
    /*if (VLOG_IS_ON(3)) */ {
      t_end = chrono::system_clock::now();
      double dt = chrono::duration<double>(t_end - t_begin).count();
      LOG(INFO) << "this=" << this << " quantize col_buf: " << dt * 1e3
                << " ms";
      t_begin = chrono::system_clock::now();
    }
#endif

    ConvNHWCCore_(col_buffer_data, Y_int32);

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
    /*if (VLOG_IS_ON(3)) */ {
      t_end = chrono::system_clock::now();
      double dt = chrono::duration<double>(t_end - t_begin).count();
      LOG(INFO) << "this=" << this << " GEMM: " << dt * 1e3 << " ms";
      t_begin = chrono::system_clock::now();
    }
#endif

    if (Wq_packed_ || Wq_depthwise_packed_ || Wq_gconv_packed_ ||
        Wq_gconv3d_packed_) {
      // In fast path with fbgemm except when
      // rescaling quantized numbers should've been already done.
      PropagateOutputTensorQuantizationParams(this, 0, out_qparams_);
    } else {
      RunOnDeviceEpilogueNHWC_(col_buffer_data, Y_int32->data());
    }
  }; // f

  this->RunWithSharedBuffer_(&col_buffer_, &Y_int32_, f);

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
  /*if (VLOG_IS_ON(3)) */ {
    const int N = X.dim32(0);
    // The dimension of each kernel
    const int kernel_dim = KernelDim_();
    // The output image size is the spatial size of the output.
    const int Y_HxW = this->GetDimsSize(*Y);

    t_end = chrono::system_clock::now();
    double dt = chrono::duration<double>(t_end - t_begin).count();
    LOG(INFO) << "this=" << this << " prologue: " << dt * 1e3 << " ms";
    t_begin = chrono::system_clock::now();

    t_end = chrono::system_clock::now();
    const int M = filter.dim32(0);
    double ops = 2. * N * Y_HxW * M * kernel_dim;
    dt = chrono::duration<double>(t_end - t_very_begin).count();
    double gops = ops / dt / 1e9;
    LOG(INFO) << "this=" << this << " " << this->debug_def().type()
              << " output=" << this->debug_def().output(0) << " " << N * Y_HxW
              << "x" << M << "x" << kernel_dim << " G=" << group_
              << " C/G=" << C / group_ << " K/G=" << M / group_
              << " R=" << kernel_h() << " S=" << kernel_w() << " : " << dt * 1e3
              << " ms " << gops << " gops";
  }
#endif

  MeasureQuantizationError_();

  return true;
}

template class ConvDNNLowPOp<uint8_t, false>;
template class ConvDNNLowPOp<uint8_t, true>;

template class ConvDNNLowPOp<uint16_t, false>;
template class ConvDNNLowPOp<uint16_t, true>;

REGISTER_CPU_OPERATOR_WITH_ENGINE(Conv, DNNLOWP, ConvDNNLowPOp<uint8_t, false>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    ConvRelu,
    DNNLOWP,
    ConvDNNLowPOp<uint8_t, true>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8Conv,
    DNNLOWP,
    ConvDNNLowPOp<uint8_t, false>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8ConvRelu,
    DNNLOWP,
    ConvDNNLowPOp<uint8_t, true>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Conv,
    DNNLOWP_16,
    ConvDNNLowPOp<uint16_t, false>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    ConvRelu,
    DNNLOWP_16,
    ConvDNNLowPOp<uint16_t, true>);

} // namespace caffe2
