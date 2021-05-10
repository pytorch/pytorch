#include "conv_dnnlowp_acc16_op.h"

// #define DNNLOWP_ACC16_IN_SLOW_PATH
// #define DNNLOWP_MEASURE_TIME_BREAKDOWN
#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
#include <chrono>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#include "caffe2/core/logging.h"
#include "dnnlowp_op.h"
#include "dnnlowp_partition.h"
#include "fbgemm_pack_op.h"
#include "im2col_dnnlowp.h"

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DECLARE_int32(caffe2_dnnlowp_nbits_in_non_outlier);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DECLARE_int32(caffe2_dnnlowp_copy_to_32bit_frequency);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DECLARE_bool(caffe2_dnnlowp_shared_int32_buffer);
// Thresholds to fallback to 32-bit accumulation when 16-bit accumulation
// doesn't provide performance benefits.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_double(
    caffe2_dnnlowp_acc16_density_threshold,
    0.05,
    "If density of outlier is higher than this, fallback to 32-bit accumulation");
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_int32(
    caffe2_dnnlowp_acc16_m_threshold,
    0,
    "If m is smaller than this, fallback to 32-bit accumulation");
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_int32(
    caffe2_dnnlowp_acc16_n_threshold,
    0,
    "If n is smaller than this, fallback to 32-bit accumulation");
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_int32(
    caffe2_dnnlowp_acc16_k_threshold,
    0,
    "If k is smaller than this, fallback to 32-bit accumulation");

namespace caffe2 {

using namespace std;

template <bool ReluFused>
ConvDNNLowPAcc16Op<ReluFused>::ConvDNNLowPAcc16Op(
    const OperatorDef& operator_def,
    Workspace* ws)
    : ConvDNNLowPOp<uint8_t, ReluFused>(operator_def, ws),
      nbits_in_non_outlier_(this->template GetSingleArgument<int>(
          "nbits_in_non_outlier",
          FLAGS_caffe2_dnnlowp_nbits_in_non_outlier)),
      copy_to_32bit_frequency_(this->template GetSingleArgument<int>(
          "copy_to_32bit_frequency",
          FLAGS_caffe2_dnnlowp_copy_to_32bit_frequency)) {
  if (nbits_in_non_outlier_ == 0) {
    LOG(INFO) << "nbits_in_non_outlier == 0 means everything is outlier so we "
                 "fallback to acc32";
    fallback_to_32_bit_accumulation_ = true;
  }
}

template <bool ReluFused>
bool ConvDNNLowPAcc16Op<ReluFused>::GetQuantizationParameters_() {
  if (fallback_to_32_bit_accumulation_) {
    // Short cut if we already know we are falling back to acc32
    return BaseType::GetQuantizationParameters_();
  }

  int kernel_dim = this->KernelDim_();
  const auto& filter = InputTensorCPU_(FILTER);
  int num_out_channels = filter.dim32(0);

  // Check if we should fallback to 32-bit accumulation
  // We should do this before GetQuantizationParameters_ to make sure
  // GetQuantizationParameters_ initialize things like Wq_packed_ for acc32
  // properly.

  // We can't fallback if layout is not NHWC or
  // if weight is prepacked and the prepacked weight doesn't have acc32.
  bool can_fallback_to_32_bit_accumulation =
      this->order_ == StorageOrder::NHWC &&
      (!this->template InputIsType<Int8ConvDNNLowPPackedWeightBlob>(FILTER) ||
       this->template Input<Int8ConvDNNLowPPackedWeightBlob>(FILTER).W);
  if (can_fallback_to_32_bit_accumulation) {
    const Tensor& X = InputTensorCPU_(INPUT);
    int N = X.dim32(0);

    auto sizes = this->GetOutputSize(X, filter.dim32(0));
    Tensor* Y = OutputTensorCPU_(0, sizes, at::dtype<uint8_t>());
    const int output_image_size = this->GetDimsSize(*Y);

    // In Skylake, acc16 is not faster when N or K is smaller than 128
    constexpr int SKYLAKE_ACC16_N_THRESHOLD_MIN = 128,
                  SKYLAKE_ACC16_K_THRESHOLD_MIN = 128;
    int acc16_n_threshold = FLAGS_caffe2_dnnlowp_acc16_n_threshold;
    if (caffe2::GetCpuId().avx512f() &&
        acc16_n_threshold < SKYLAKE_ACC16_N_THRESHOLD_MIN) {
      acc16_n_threshold = SKYLAKE_ACC16_N_THRESHOLD_MIN;
    }
    int acc16_k_threshold = FLAGS_caffe2_dnnlowp_acc16_k_threshold;
    if (caffe2::GetCpuId().avx512f() &&
        acc16_k_threshold < SKYLAKE_ACC16_K_THRESHOLD_MIN) {
      acc16_k_threshold = SKYLAKE_ACC16_K_THRESHOLD_MIN;
    }

    if (N * output_image_size < FLAGS_caffe2_dnnlowp_acc16_m_threshold) {
      C10_LOG_FIRST_N(INFO, 10)
          << "M " << N * output_image_size << " of Conv layer with weight blob "
          << this->debug_def().input(FILTER) << " is smaller than threshold "
          << FLAGS_caffe2_dnnlowp_acc16_m_threshold
          << " . Falling back to acc32";
      fallback_to_32_bit_accumulation_ = true;
    }
    if (!fallback_to_32_bit_accumulation_ &&
        num_out_channels / group_ < acc16_n_threshold) {
      C10_LOG_FIRST_N(INFO, 10)
          << "N " << num_out_channels / group_
          << " of Conv layer with weight blob "
          << this->debug_def().input(FILTER) << " is smaller than threshold "
          << acc16_n_threshold << " . Falling back to acc32";
      fallback_to_32_bit_accumulation_ = true;
    }
    if (!fallback_to_32_bit_accumulation_ && kernel_dim < acc16_k_threshold) {
      C10_LOG_FIRST_N(INFO, 10)
          << "K " << kernel_dim << " of Conv layer with weight blob "
          << this->debug_def().input(FILTER) << " is smaller than threshold "
          << acc16_k_threshold << " . Falling back to acc32";
      fallback_to_32_bit_accumulation_ = true;
    }
    if (!fallback_to_32_bit_accumulation_ &&
        this->template InputIsType<Int8ConvDNNLowPPackedWeightBlob>(FILTER) &&
        !this->template Input<Int8ConvDNNLowPPackedWeightBlob>(FILTER)
             .W_acc16) {
      C10_LOG_FIRST_N(INFO, 10)
          << "Falling back to acc32 because packed weight for acc16 is not "
             "available";
      fallback_to_32_bit_accumulation_ = true;
    }
  }

  if (!BaseType::GetQuantizationParameters_()) {
    return false;
  }

  if (fallback_to_32_bit_accumulation_) {
    return true;
  }

  if (!Wq_acc16_packed_ &&
      this->template InputIsType<Int8ConvDNNLowPPackedWeightBlob>(FILTER)) {
    CAFFE_ENFORCE_EQ(
        this->order_,
        StorageOrder::NHWC,
        "Pre-packed weight only works with NHWC layout");
    // If the input is already packed
    const auto& packed_filter =
        this->template Input<Int8ConvDNNLowPPackedWeightBlob>(FILTER);
    Wq_outlier_ = packed_filter.W_outlier;
    Wq_acc16_packed_ = packed_filter.W_acc16;

    if (nbits_in_non_outlier_ != packed_filter.nbits_in_non_outlier) {
      C10_LOG_FIRST_N(WARNING, 10)
          << "nbits_in_non_outlier in packed weight "
          << packed_filter.nbits_in_non_outlier
          << " doesn't match with nbits_in_non_outlier specified in operator "
          << nbits_in_non_outlier_;
    }
    first_invocation_ = false;
    return true;
  }

  // Separate out outliers
  if (!Wq_outlier_ && this->order_ == StorageOrder::NHWC &&
      nbits_in_non_outlier_ < 8) {
    CAFFE_ENFORCE(!W_quantized_.empty());

    int outlier_cnt = CountOutliers(
        group_,
        kernel_dim,
        num_out_channels,
        nbits_in_non_outlier_,
        W_quantized_);

    C10_LOG_FIRST_N(INFO, 10)
        << "Proportion of outlier for Conv layer with weight blob "
        << this->debug_def().input(FILTER) << " is "
        << static_cast<float>(outlier_cnt) / W_quantized_.size();
    C10_LOG_FIRST_N(INFO, 10)
        << "nbits_in_non_outlier " << nbits_in_non_outlier_
        << " copy_to_32bit_frequency " << copy_to_32bit_frequency_;

    if (can_fallback_to_32_bit_accumulation &&
        static_cast<float>(outlier_cnt) / W_quantized_.size() >
            FLAGS_caffe2_dnnlowp_acc16_density_threshold) {
      C10_LOG_FIRST_N(INFO, 10)
          << "Density of outliers is higher than threshold "
          << FLAGS_caffe2_dnnlowp_acc16_density_threshold
          << " . Falling back to acc32";
      fallback_to_32_bit_accumulation_ = true;
      Wq_outlier_.reset();
      // We need to call GetQuantizationParameters_ again to pack for acc32
      return BaseType::GetQuantizationParameters_();
    }

    Wq_outlier_.reset(ExtractOutlierMatrix(
        group_,
        kernel_dim,
        num_out_channels,
        nbits_in_non_outlier_,
        W_quantized_));
  }

  bool packW = this->order_ == StorageOrder::NHWC && GetCpuId().avx2();

  if (first_invocation_) {
    if (!packW) {
      string reason;
      if (this->order_ != StorageOrder::NHWC) {
        reason = "fbgemm only supports NHWC layout";
      } else if (!GetCpuId().avx2()) {
        reason = "fbgemm only supports AVX2+";
      } else {
        assert(false);
      }

      if (!reason.empty()) {
        static int log_occurences = 0;
        if (log_occurences < 32) {
          ++log_occurences;
          C10_LOG_FIRST_N(WARNING, 10)
              << "Conv with weight " << this->debug_def().input(FILTER)
              << " falls back to slow path because " << reason;
        }
      }
    }
    if (nbits_in_non_outlier_ < 8 && this->order_ != StorageOrder::NHWC) {
      static int log_occurences = 0;
      if (log_occurences < 32) {
        ++log_occurences;
        C10_LOG_FIRST_N(WARNING, 10)
            << "Outlier-aware quantization only supports "
               "NHWC layout";
      }
    }
    first_invocation_ = false;
  }

  if (packW && !Wq_acc16_packed_) {
    // NOLINTNEXTLINE(modernize-make-shared)
    Wq_acc16_packed_.reset(new fbgemm::PackBMatrix<int8_t, int16_t>(
        fbgemm::matrix_op_t::Transpose,
        group_ * kernel_dim,
        num_out_channels / group_,
        W_quantized_.data(),
        kernel_dim, // ld
        nullptr, // pmat
        group_));
    vector<int8_t>().swap(W_quantized_);
  }

  return true;
}

template <bool ReluFused>
bool ConvDNNLowPAcc16Op<ReluFused>::RunOnDeviceWithOrderNCHW() {
  VLOG(2) << "Running DNNLOWP_ACC16 Conv";

  using namespace dnnlowp;

  // Get quantization parameters
  if (!GetQuantizationParameters_()) {
    return false;
  }
  if (fallback_to_32_bit_accumulation_) {
    return BaseType::RunOnDeviceWithOrderNCHW();
  }

  const Tensor& X = InputTensorCPU_(INPUT);
  auto& filter = InputTensorCPU_(FILTER);
  const int N = X.dim32(0), C = X.dim32(1);
  CAFFE_ENFORCE_EQ(X.ndim(), filter.ndim());
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

  auto sizes = this->GetOutputSize(X, filter.dim32(0));
  Tensor* Y = OutputTensorCPU_(0, sizes, at::dtype<uint8_t>());

  const vector<int> input_dims = GetDims(X);
  const vector<int> output_dims = GetDims(*Y);
  const int input_image_size = this->GetDimsSize(X);
  const int output_image_size = this->GetDimsSize(*Y);

  // The dimension of each kernel
  const int kernel_dim = this->KernelDim_();

  vector<int> img_shape;
  img_shape.assign(X.sizes().begin() + 1, X.sizes().end());

  vector<int> buffer_shape;
  buffer_shape.push_back(kernel_dim);
  buffer_shape.insert(
      buffer_shape.end(), output_dims.begin(), output_dims.end());
  buffer_shape.insert(buffer_shape.begin(), dnnlowp_get_max_threads());

  if (this->kernel_.size() != 2) {
    SetDeviceTensor(img_shape, &(this->img_shape_device_));
    SetDeviceTensor(buffer_shape, &(this->col_buffer_shape_device_));
  }

  const int col_buffer_size = kernel_dim * output_image_size;

  // The offset corresponding to a single input image, and a single output
  // image.
  const int input_offset = C / group_ * input_image_size;

  // The col buffer is stored in CHW order as well - kernel_dim, and the
  // height and width.
  const uint8_t* Xdata = X.template data<uint8_t>();

  auto f = [&](Tensor* col_buffer, vector<int32_t>* Y_int32) {
    col_buffer->Resize(buffer_shape);
    uint8_t* col_buffer_data = col_buffer->template mutable_data<uint8_t>();

    Y_int32->resize(M * output_image_size * dnnlowp_get_max_threads());
    vector<int> buffer_shape_per_thread(
        buffer_shape.begin() + 1, buffer_shape.end());

    // Im2Col, followed by gemm.
    uint8_t* Y_data = Y->template mutable_data<uint8_t>();
    this->column_offsets_->resize(
        output_image_size * dnnlowp_get_max_threads());

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int image_id = 0; image_id < N; ++image_id) {
      int tid = dnnlowp_get_thread_num();
      for (int group_id = 0; group_id < group_; ++group_id) {
        if (this->kernel_.size() == 2) {
          math::Im2ColNCHW<uint8_t>(
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
          math::Im2ColNdNCHW<uint8_t>(
              this->kernel_.size(),
              C * input_image_size,
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
        uint8_t* col_buffer_private = col_buffer_data + tid * col_buffer_size;

        // main GEMM
        int32_t* Y_int32_temp = Y_int32->data() +
            ((M / group_) * group_id + M * tid) * output_image_size;
        int8_t* W_quantized_group =
            W_quantized_.data() + (M / group_) * group_id * kernel_dim;

        static int log_occurences = 0;
        if (log_occurences < 32) {
          ++log_occurences;
          C10_LOG_FIRST_N(WARNING, 10)
              << "Consider using DNNLOWP instead of DNNLOWP_ACC16 engine since "
                 "we're falling back to a slow path because of NCHW layout";
        }

        for (int i = 0; i < M / group_; ++i) {
          for (int j = 0; j < output_image_size; ++j) {
            int32_t int32_sum = 0;
            int16_t int16_sum = 0;
            for (int k = 0; k < kernel_dim; ++k) {
              // NOLINTNEXTLINE(bugprone-signed-char-misuse)
              int32_t w = W_quantized_group[i * kernel_dim + k];
              int32_t x = col_buffer_private[k * output_image_size + j];
#ifdef DNNLOWP_ACC16_IN_SLOW_PATH
              int16_sum = std::max<int32_t>(
                  numeric_limits<int16_t>::min(),
                  std::min<int32_t>(
                      numeric_limits<int16_t>::max(), int16_sum + x * w));
              if (k % copy_to_32bit_frequency_ ==
                  copy_to_32bit_frequency_ - 1) {
                int32_sum += int16_sum;
                int16_sum = 0;
              }
#else
              int32_sum += w * x;
#endif
            }
            Y_int32_temp[i * output_image_size + j] = int32_sum + int16_sum;
          }
        }

        this->RunOnDeviceEpilogueNCHW_(
            col_buffer_private,
            Y_int32_temp,
            Y_data + (M * image_id + M / group_ * group_id) * output_image_size,
            M / group_ * group_id,
            group_id);
      } // for each group
    } // for each image_id
  }; // f

  this->RunWithSharedBuffer_(&col_buffer_, &(this->Y_int32_), f);

  PropagateOutputTensorQuantizationParams(this, 0, out_qparams_);

  this->MeasureQuantizationError_();

  return true;
} // RunOnDeviceWithOrderNCHWAndType_

static void conv_nhwc_acc16_ref_(
    int num_groups,
    int N,
    int output_image_size,
    int M,
    int kernel_dim,
    const uint8_t* col_buffer,
    const int8_t* W,
    int32_t* Y
#ifdef DNNLOWP_ACC16_IN_SLOW_PATH
    ,
    OperatorBase* op
#endif
) {
#ifdef DNNLOWP_ACC16_IN_SLOW_PATH
  uint64_t underflow_cnt = 0, overflow_cnt = 0;
#endif
  for (int group_id = 0; group_id < num_groups; ++group_id) {
    for (int i = 0; i < N * output_image_size; ++i) {
      for (int j = 0; j < M / num_groups; ++j) {
        int32_t int32_sum = 0;
        int16_t int16_sum = 0;
#ifdef DNNLOWP_ACC16_IN_SLOW_PATH
        bool overflowed = false, underflowed = false;
#endif
        for (int k = 0; k < kernel_dim; ++k) {
          int32_t x = col_buffer[(i * num_groups + group_id) * kernel_dim + k];
          // NOLINTNEXTLINE(bugprone-signed-char-misuse)
          int32_t w = W[(group_id * (M / num_groups) + j) * kernel_dim + k];
#ifdef DNNLOWP_ACC16_IN_SLOW_PATH
          if (!overflowed && !underflowed) {
            if (int16_sum + x * w > numeric_limits<int16_t>::max()) {
              overflowed = true;
            } else if (int16_sum + x * w < numeric_limits<int16_t>::min()) {
              underflowed = true;
            }
          }

          int16_sum = std::max<int32_t>(
              numeric_limits<int16_t>::min(),
              std::min<int32_t>(
                  numeric_limits<int16_t>::max(), int16_sum + x * w));
          if (k % copy_to_32bit_frequency_ == copy_to_32bit_frequency_ - 1) {
            int32_sum += int16_sum;
            int16_sum = 0;
          }
#else
          int32_sum += x * w;
#endif
        }
        Y[i * M + group_id * (M / num_groups) + j] = int32_sum + int16_sum;
#ifdef DNNLOWP_ACC16_IN_SLOW_PATH
        if (overflowed) {
          ++overflow_cnt;
        } else if (underflowed) {
          ++underflow_cnt;
        }
#ifdef DNNLOWP_DETAILED_LOG_IN_ACC16_SLOW_PATH
        if (overflowed || underflowed) {
          int32_t sum = 0;
          for (int k = 0; k < kernel_dim; ++k) {
            int32_t x =
                col_buffer[(i * num_groups + group_id) * kernel_dim + k];
            int32_t w = W[k * M + group_id * (M / num_groups) + j];
            LOG(INFO) << k << ": " << sum << " + " << x << " * " << w << " = "
                      << sum + x * w;
            sum += x * w;
          }
        }
#endif
#endif
      }
    }
  } // for each group

#ifdef DNNLOWP_ACC16_IN_SLOW_PATH
  LOG(INFO) << op->debug_def().input(1) << " underflow_cnt " << underflow_cnt
            << " (" << (float)underflow_cnt / (N * output_image_size * M) * 100
            << ") overflow_cnt " << overflow_cnt << " ("
            << (float)overflow_cnt / (N * output_image_size * M) * 100 << ")";
#endif
}

template <bool ReluFused>
template <typename PackAMatrix, fbgemm::QuantizationGranularity Q_GRAN>
void ConvDNNLowPAcc16Op<ReluFused>::DispatchFBGEMM_(
    PackAMatrix& packA,
    const uint8_t* col_buffer_data,
    vector<int32_t>* Y_int32,
    uint8_t* Y_uint8_data) {
  // This function is called within an OpenMP region
  auto& filter = InputTensorCPU_(FILTER);
  const int M = filter.dim32(0);

  assert(Wq_acc16_packed_.get());
  int kernel_dim = this->KernelDim_();

  int nthreads = dnnlowp_get_num_threads();
  int tid = dnnlowp_get_thread_num();

  using namespace fbgemm;
  DoNothing<> doNothingObj{};
  ReQuantizeOutput<ReluFused, Q_GRAN> reqObj(
      doNothingObj,
      this->requantization_multipliers_.data(),
      out_qparams_.zero_point,
      // column_offsets_ empty means column_offsets_ are folded into bias
      this->column_offsets_->empty() ? 0 : in_qparams_[INPUT].zero_point,
      this->filter_zero_points_.data(),
      packA.getRowOffsetBuffer(),
      this->column_offsets_->empty() ? nullptr : this->column_offsets_->data(),
      InputSize() == 3 ? this->b_quantized_data_ : nullptr,
      M,
      group_);

  if (nbits_in_non_outlier_ < 8) {
    DoSpmdmOnInpBuffer<
        typename ReQuantizeOutput<ReluFused>::outType,
        int32_t,
        ReQuantizeOutput<ReluFused, Q_GRAN>>
        spmdmObj(
            reqObj, col_buffer_data, group_ * kernel_dim, *Wq_outlier_, group_);

    fbgemmPacked(
        packA,
        *Wq_acc16_packed_,
        Y_uint8_data,
        Y_int32->data(),
        M,
        spmdmObj,
        tid,
        nthreads);
  } else {
    fbgemmPacked(
        packA,
        *Wq_acc16_packed_,
        Y_uint8_data,
        Y_int32->data(),
        M,
        reqObj,
        tid,
        nthreads);
  }
}

template <bool ReluFused>
void ConvDNNLowPAcc16Op<ReluFused>::ConvOutlier_(
    const uint8_t* col_buffer,
    vector<int32_t>* Y_int32) {
  if (nbits_in_non_outlier_ < 8) {
    const Tensor& X = InputTensorCPU_(INPUT);
    auto& filter = InputTensorCPU_(FILTER);
    Tensor* Y = OutputTensorCPU_(0);
    const int N = X.dim32(0);
    const int M = filter.dim32(0);

    const int kernel_dim = this->KernelDim_();
    const int output_image_size = this->GetDimsSize(*Y);

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      int group_begin, group_end, i_begin, i_end;
      this->PartitionGroupedNHWCConv_(
          &group_begin,
          &group_end,
          &i_begin,
          &i_end,
          group_,
          N * output_image_size,
          dnnlowp_get_num_threads(),
          dnnlowp_get_thread_num());

      for (int group_id = group_begin; group_id < group_end; ++group_id) {
        CAFFE_ENFORCE_EQ(Wq_outlier_->NumOfRows(), kernel_dim);
        // Dense-matrix times sparse-matrix multiplication for outlier
        fbgemm::block_type_t block = {
            0, i_end - i_begin, group_id * (M / group_), M / group_};
        Wq_outlier_->SpMDM(
            block,
            col_buffer + (i_begin * group_ + group_id) * kernel_dim,
            group_ * kernel_dim,
            true /* accumulate */,
            Y_int32->data() + i_begin * M + group_id * (M / group_),
            M);
      }
    }
  }
}

template <bool ReluFused>
bool ConvDNNLowPAcc16Op<ReluFused>::RunOnDeviceWithOrderNHWC() {
  CAFFE_ENFORCE_LE(
      this->kernel_.size(),
      3,
      "Only 1-3d convolution is supported for NHWC storage type");

  using namespace dnnlowp;

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
  chrono::time_point<chrono::system_clock> t_very_begin, t_begin, t_end;

  t_begin = chrono::system_clock::now();
  t_very_begin = t_begin;
#endif

  // Get quantization parameters
  if (!GetQuantizationParameters_()) {
    return false;
  }

  if (fallback_to_32_bit_accumulation_) {
    return BaseType::RunOnDeviceWithOrderNHWC();
  }

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
  t_end = chrono::system_clock::now();
  double dt = chrono::duration<double>(t_end - t_begin).count();
  LOG(INFO) << "this=" << this << " get_quant_params: " << dt * 1e3 << " ms";
#endif

  const Tensor& X = InputTensorCPU_(INPUT);
  auto& filter = InputTensorCPU_(FILTER);
  const int N = X.dim32(0), C = X.dim32(X.ndim() - 1);

  CAFFE_ENFORCE_EQ(X.ndim(), filter.ndim());
  const int M = filter.dim32(0);
  CAFFE_ENFORCE_EQ(filter.dim32(filter.ndim() - 1), C / group_);

  auto sizes = this->GetOutputSize(X, filter.dim32(0));
  Tensor* Y = OutputTensorCPU_(0, sizes, at::dtype<uint8_t>());
  // The dimension of each kernel
  const int kernel_dim = this->KernelDim_();
  // The output image size is the spatial size of the output.
  const int output_image_size = this->GetDimsSize(*Y);
  // The col buffer is stored in HWC order as well - kernel_dim, and the height
  // and width.

  auto f = [&](Tensor* col_buffer, vector<int32_t>* Y_int32) {
    Y_int32->resize(Y->numel());

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
    t_begin = chrono::system_clock::now();
#endif

    bool no_im2col = this->NoIm2ColNHWC_();

    // Im2Col, followed by gemm.
    const uint8_t* Xdata = X.template data<uint8_t>();
    const uint8_t* col_buffer_data =
        no_im2col ? Xdata : this->Im2ColNHWC_(col_buffer);

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
    t_end = chrono::system_clock::now();
    dt = chrono::duration<double>(t_end - t_begin).count();
    LOG(INFO) << "this=" << this << " im2col: " << dt * 1e3 << " ms";
    t_begin = chrono::system_clock::now();
#endif

    using namespace fbgemm;
    int row_offset_size_per_thread = -1;
    int x_pack_buf_size_per_thread = -1;
    if (Wq_acc16_packed_) {
      if (!this->quantize_groupwise_ && this->filter_zero_points_[0] == 0) {
        x_pack_buf_size_per_thread =
            PackAMatrix<uint8_t, int16_t>::packedBufferSize();
        X_pack_buf_.resize(
            dnnlowp_get_max_threads() * x_pack_buf_size_per_thread);
      } else {
        row_offset_size_per_thread =
            PackAWithRowOffset<uint8_t, int16_t>::rowOffsetBufferSize();
        x_pack_buf_size_per_thread =
            PackAWithRowOffset<uint8_t, int16_t>::packedBufferSize();
        row_offsets_.resize(
            dnnlowp_get_max_threads() * row_offset_size_per_thread);
        X_pack_buf_.resize(
            dnnlowp_get_max_threads() * x_pack_buf_size_per_thread);
      }
    }

    uint8_t* Y_uint8_data = Y->template mutable_data<uint8_t>();

    // Main GEMM for non-outlier
    if (Wq_acc16_packed_)
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      // fast path
      int tid = dnnlowp_get_thread_num();

      // no im2col fusion
      if (!this->quantize_groupwise_ && this->filter_zero_points_[0] == 0) {
        PackAMatrix<uint8_t, int16_t> packA(
            matrix_op_t::NoTranspose,
            N * output_image_size,
            group_ * kernel_dim,
            col_buffer_data,
            group_ * kernel_dim,
            X_pack_buf_.data() + tid * x_pack_buf_size_per_thread,
            group_);

        if (this->quantize_groupwise_) {
          DispatchFBGEMM_<
              PackAMatrix<uint8_t, int16_t>,
              QuantizationGranularity::GROUP>(
              packA, col_buffer_data, Y_int32, Y_uint8_data);
        } else {
          DispatchFBGEMM_<
              PackAMatrix<uint8_t, int16_t>,
              QuantizationGranularity::TENSOR>(
              packA, col_buffer_data, Y_int32, Y_uint8_data);
        }
      } else {
        // no im2col fusion
        PackAWithRowOffset<uint8_t, int16_t> packA(
            matrix_op_t::NoTranspose,
            N * output_image_size,
            group_ * kernel_dim,
            col_buffer_data,
            group_ * kernel_dim,
            X_pack_buf_.data() + tid * x_pack_buf_size_per_thread,
            group_,
            row_offsets_.data() + tid * row_offset_size_per_thread);

        if (this->quantize_groupwise_) {
          DispatchFBGEMM_<
              PackAWithRowOffset<uint8_t, int16_t>,
              QuantizationGranularity::GROUP>(
              packA, col_buffer_data, Y_int32, Y_uint8_data);
        } else {
          DispatchFBGEMM_<
              PackAWithRowOffset<uint8_t, int16_t>,
              QuantizationGranularity::TENSOR>(
              packA, col_buffer_data, Y_int32, Y_uint8_data);
        }
      }
    } else {
      // slow path
      conv_nhwc_acc16_ref_(
          group_,
          N,
          output_image_size,
          M,
          kernel_dim,
          col_buffer_data,
          W_quantized_.data(),
          Y_int32->data()
#ifdef DNNLOWP_ACC16_IN_SLOW_PATH
              ,
          this
#endif
      );
    } // slow path

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
    t_end = chrono::system_clock::now();
    dt = chrono::duration<double>(t_end - t_begin).count();
    double ops = 2. * N * output_image_size * M * kernel_dim;
    double gops = ops / dt / 1e9;
    LOG(INFO) << "this=" << this << " GEMM: " << dt * 1e3 << " ms " << gops
              << " gops";
    t_begin = chrono::system_clock::now();
#endif

    if (!Wq_acc16_packed_) {
      ConvOutlier_(col_buffer_data, Y_int32);
    }

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
    t_end = chrono::system_clock::now();
    dt = chrono::duration<double>(t_end - t_begin).count();
    LOG(INFO) << "this=" << this << " out-lier: " << dt * 1e3 << " ms";
    t_begin = chrono::system_clock::now();
#endif

    if (!Wq_acc16_packed_) {
      this->RunOnDeviceEpilogueNHWC_(col_buffer_data, Y_int32->data());
    } else {
      PropagateOutputTensorQuantizationParams(this, 0, out_qparams_);
    }
  }; // f

  this->RunWithSharedBuffer_(&col_buffer_, &(this->Y_int32_), f);

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
  t_end = chrono::system_clock::now();
  dt = chrono::duration<double>(t_end - t_begin).count();
  LOG(INFO) << "this=" << this << " prologue: " << dt * 1e3 << " ms";
  t_begin = chrono::system_clock::now();

  t_end = chrono::system_clock::now();
  dt = chrono::duration<double>(t_end - t_very_begin).count();
  double ops = 2. * N * output_image_size * M * kernel_dim;
  double gops = ops / dt / 1e9;
  LOG(INFO) << "this=" << this << " " << this->debug_def().type()
            << " output=" << this->debug_def().output(0) << " "
            << N * output_image_size << "x" << M << "x" << kernel_dim
            << " G=" << group_ << " C/G=" << C / group_ << " K/G=" << M / group_
            << " R=" << kernel_h() << " S=" << kernel_w() << " : " << dt * 1e3
            << " ms " << gops << " gops";
#endif

  this->MeasureQuantizationError_();

  return true;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Conv,
    DNNLOWP_ACC16,
    ConvDNNLowPAcc16Op<false>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    ConvRelu,
    DNNLOWP_ACC16,
    ConvDNNLowPAcc16Op<true>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8Conv,
    DNNLOWP_ACC16,
    ConvDNNLowPAcc16Op<false>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8ConvRelu,
    DNNLOWP_ACC16,
    ConvDNNLowPAcc16Op<true>);

} // namespace caffe2
