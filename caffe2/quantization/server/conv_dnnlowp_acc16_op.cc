#include "conv_dnnlowp_acc16_op.h"
#include "dnnlowp_op.h"

// #define DNNLOWP_ACC16_IN_SLOW_PATH
// #define DNNLOWP_MEASURE_TIME_BREAKDOWN
#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
#include <chrono>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#include "dnnlowp_partition.h"
#include "im2col_dnnlowp.h"

C10_DECLARE_int32(dnnlowp_nbits_in_non_outlier);
C10_DECLARE_int32(dnnlowp_copy_to_32bit_frequency);
C10_DECLARE_bool(caffe2_dnnlowp_shared_int32_buffer);

namespace caffe2 {

using namespace std;

template <bool ReluFused>
ConvDNNLowPAcc16Op<ReluFused>::ConvDNNLowPAcc16Op(
    const OperatorDef& operator_def,
    Workspace* ws)
    : ConvDNNLowPOp<uint8_t, ReluFused>(operator_def, ws),
      nbits_in_non_outlier_(OperatorBase::GetSingleArgument<int>(
          "nbits_in_non_outlier",
          FLAGS_dnnlowp_nbits_in_non_outlier)),
      copy_to_32bit_frequency_(OperatorBase::GetSingleArgument<int>(
          "copy_to_32bit_frequency",
          FLAGS_dnnlowp_copy_to_32bit_frequency)) {}

template <bool ReluFused>
bool ConvDNNLowPAcc16Op<ReluFused>::RunOnDeviceWithOrderNCHW() {
  const Tensor& X = InputTensorCPU_(INPUT);
  if (X.template IsType<uint8_t>()) {
    return RunOnDeviceWithOrderNCHWAndType_<uint8_t>();
  } else {
    assert(X.template IsType<float>());
    return RunOnDeviceWithOrderNCHWAndType_<float>();
  }
}

template <bool ReluFused>
bool ConvDNNLowPAcc16Op<ReluFused>::RunOnDeviceWithOrderNHWC() {
  const Tensor& X = InputTensorCPU_(INPUT);
  if (X.template IsType<uint8_t>()) {
    return RunOnDeviceWithOrderNHWCAndType_<uint8_t>();
  } else {
    assert(X.template IsType<float>());
    return RunOnDeviceWithOrderNHWCAndType_<float>();
  }
}

template <bool ReluFused>
bool ConvDNNLowPAcc16Op<ReluFused>::GetQuantizationParameters_() {
  if (!BaseType::GetQuantizationParameters_()) {
    return false;
  }

  int kernel_dim = this->KernelDim_();
  const auto& filter = InputTensorCPU_(FILTER);
  int M = filter.dim32(0);

  // Separate out outliers
  if (Wq_outlier_.empty() &&
      ConvPoolOpBase<CPUContext>::order_ == StorageOrder::NHWC &&
      nbits_in_non_outlier_ < 8) {
    CAFFE_ENFORCE(!W_quantized_.empty());

    int total_outlier_cnt = 0;
    for (int group_id = 0; group_id < group_; ++group_id) {
      int outlier_cnt = 0;
      for (int i = 0; i < (M / group_) * kernel_dim; ++i) {
        int8_t w = W_quantized_[group_id * (M / group_) * kernel_dim + i];
        bool is_outlier = nbits_in_non_outlier_ == 0 ||
            w < -(1 << (nbits_in_non_outlier_ - 1)) ||
            w >= (1 << (nbits_in_non_outlier_ - 1));
        if (is_outlier) {
          ++outlier_cnt;
        }
      }
      total_outlier_cnt += outlier_cnt;

      Wq_outlier_.emplace_back(kernel_dim, M / group_);
      Wq_outlier_.back().RowIdx().resize(outlier_cnt);
      Wq_outlier_.back().Values().resize(outlier_cnt);

      outlier_cnt = 0;
      for (int j = 0; j < M / group_; ++j) {
        Wq_outlier_.back().ColPtr()[j] = outlier_cnt;
        for (int k = 0; k < kernel_dim; ++k) {
          int8_t w =
              W_quantized_[(group_id * (M / group_) + j) * kernel_dim + k];
          bool is_outlier = nbits_in_non_outlier_ == 0 ||
              w < -(1 << (nbits_in_non_outlier_ - 1)) ||
              w >= (1 << (nbits_in_non_outlier_ - 1));
          if (is_outlier) {
            CAFFE_ENFORCE_LE(k, numeric_limits<int16_t>::max());
            Wq_outlier_.back().RowIdx()[outlier_cnt] = k;
            Wq_outlier_.back().Values()[outlier_cnt] = w;
            ++outlier_cnt;

            W_quantized_[(group_id * (M / group_) + j) * kernel_dim + k] = 0;
          }
        }
      }
      Wq_outlier_.back().ColPtr()[M / group_] = outlier_cnt;
    } // for each group

    LOG(INFO) << "Proportion of outlier for Conv layer with weight blob "
              << OperatorBase::debug_def().input(1) << " is "
              << (float)total_outlier_cnt / W_quantized_.size();
    LOG(INFO) << "nbits_in_non_outlier " << nbits_in_non_outlier_
              << " copy_to_32bit_frequency " << copy_to_32bit_frequency_;
  }

  bool packW = ConvPoolOpBase<CPUContext>::order_ == StorageOrder::NHWC &&
      GetCpuId().avx2();

  if (first_invocation_) {
    if (!packW) {
      string reason;
      if (ConvPoolOpBase<CPUContext>::order_ != StorageOrder::NHWC) {
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
          LOG(WARNING) << "Conv with weight "
                       << OperatorBase::debug_def().input(FILTER)
                       << " falls back to slow path because " << reason;
        }
      }
    }
    if (nbits_in_non_outlier_ < 8 &&
        ConvPoolOpBase<CPUContext>::order_ != StorageOrder::NHWC) {
        static int log_occurences = 0;
        if (log_occurences < 32) {
          ++log_occurences;
          LOG(WARNING) << "Outlier-aware quantization only supports "
                          "NHWC layout";
        }
    }
    first_invocation_ = false;
  }

  if (packW && Wq_acc16_packed_.empty()) {
    Wq_acc16_packed_.resize(group_);
    for (int group_id = 0; group_id < group_; ++group_id) {
      Wq_acc16_packed_[group_id].reset(new fbgemm::PackBMatrix<int8_t, int16_t>(
          fbgemm::matrix_op_t::Transpose,
          kernel_dim,
          M / group_,
          W_quantized_.data() + group_id * (M / group_) * kernel_dim,
          kernel_dim));
    }
    vector<int8_t>().swap(W_quantized_);
  }

  return true;
}

template <bool ReluFused>
template <typename InType>
bool ConvDNNLowPAcc16Op<ReluFused>::RunOnDeviceWithOrderNCHWAndType_() {
  VLOG(2) << "Running DNNLOWP_ACC16 Conv";

  using namespace dnnlowp;

  // Get quantization parameters
  if (!GetQuantizationParameters_()) {
    return false;
  }

  const Tensor& X = InputTensorCPU_(INPUT);
  auto& filter = InputTensorCPU_(FILTER);
  Tensor* Y = OutputTensorCPU_(0);
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

  ConvPoolOpBase<CPUContext>::SetOutputSize(X, Y, filter.dim32(0));

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
  const InType* Xdata = X.template data<InType>();

  col_buffer_.Resize(buffer_shape);
  InType* col_buffer_data = col_buffer_.template mutable_data<InType>();

  auto f = [&](vector<int32_t>* Y_int32) {
    Y_int32->resize(M * output_image_size * dnnlowp_get_max_threads());
    vector<int> buffer_shape_per_thread(
        buffer_shape.begin() + 1, buffer_shape.end());

    // Im2Col, followed by gemm.
    vector<uint8_t> Y_temp;
    uint8_t* Y_data;
    float* Y_data_float = nullptr;
    if (dequantize_output_) {
      Y_temp.resize(Y->numel());
      Y_data = Y_temp.data();
      Y_data_float = Y->template mutable_data<float>();
    } else {
      Y_data = Y->template mutable_data<uint8_t>();
    }
    this->column_offsets_.resize(output_image_size * dnnlowp_get_max_threads());

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int image_id = 0; image_id < N; ++image_id) {
      int tid = dnnlowp_get_thread_num();
      for (int group_id = 0; group_id < group_; ++group_id) {
        if (this->kernel_.size() == 2) {
          math::Im2ColNCHW<InType>(
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
              X.IsType<uint8_t>() ? in_qparams_[INPUT].zero_point : 0);
        } else {
          math::Im2ColNdNCHW<InType>(
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
              X.IsType<uint8_t>() ? in_qparams_[INPUT].zero_point : 0);
        }

        // quantize col_buffer
        uint8_t* col_buffer_quantized_data = nullptr;
        vector<uint8_t> col_buffer_quantized;
        if (X.template IsType<uint8_t>()) {
          col_buffer_quantized_data =
              (uint8_t*)col_buffer_data + tid * col_buffer_size;
        } else {
          col_buffer_quantized.resize(kernel_dim * output_image_size);
          Quantize<uint8_t>(
              (const float*)col_buffer_data + tid * col_buffer_size,
              col_buffer_quantized.data(),
              col_buffer_quantized.size(),
              in_qparams_[INPUT]);
          col_buffer_quantized_data = col_buffer_quantized.data();
        }

        // main GEMM
        int32_t* Y_int32_temp = Y_int32->data() +
            ((M / group_) * group_id + M * tid) * output_image_size;
        int8_t* W_quantized_group =
            W_quantized_.data() + (M / group_) * group_id * kernel_dim;

        static int log_occurences = 0;
        if (log_occurences < 32) {
          ++log_occurences;
          LOG(WARNING)
              << "Consider using DNNLOWP instead of DNNLOWP_ACC16 engine since "
                 "we're falling back to a slow path because of NCHW layout";
        }

        for (int i = 0; i < M / group_; ++i) {
          for (int j = 0; j < output_image_size; ++j) {
            int32_t int32_sum = 0;
            int16_t int16_sum = 0;
            for (int k = 0; k < kernel_dim; ++k) {
              int32_t w = W_quantized_group[i * kernel_dim + k];
              int32_t x = col_buffer_quantized_data[k * output_image_size + j];
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

        if (dequantize_output_) {
          this->RunOnDeviceEpilogueNCHW_(
              col_buffer_quantized_data,
              Y_int32_temp,
              Y_data_float +
                  (M * image_id + M / group_ * group_id) * output_image_size,
              M / group_ * group_id,
              group_id);
        } else {
          this->RunOnDeviceEpilogueNCHW_(
              col_buffer_quantized_data,
              Y_int32_temp,
              Y_data +
                  (M * image_id + M / group_ * group_id) * output_image_size,
              M / group_ * group_id,
              group_id);
        }
      } // for each group
    } // for each image_id
  }; // f

  if (FLAGS_caffe2_dnnlowp_shared_int32_buffer) {
    this->RunWithSharedInt32Buffer_(f);
  } else {
    f(&(this->Y_int32_));
  }

  if (!dequantize_output_) {
    PropagateOutputTensorQuantizationParams(this, 0, out_qparams_);
  }

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

    if (nbits_in_non_outlier_ == 0) {
      memset(Y_int32->data(), 0, sizeof((*Y_int32)[0]) * M * N);
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
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
        assert(Wq_outlier_[group_id].NumOfRows() == kernel_dim);
        // Dense-matrix times sparse-matrix multiplication for outlier
        fbgemm::block_type_t block = {0, i_end - i_begin, 0, M / group_};
        Wq_outlier_[group_id].SpMDM(
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
template <typename InType>
bool ConvDNNLowPAcc16Op<ReluFused>::RunOnDeviceWithOrderNHWCAndType_() {
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

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
  t_end = chrono::system_clock::now();
  double dt = chrono::duration<double>(t_end - t_begin).count();
  LOG(INFO) << "this=" << this << " get_quant_params: " << dt * 1e3 << " ms";
#endif

  const Tensor& X = InputTensorCPU_(INPUT);
  auto& filter = InputTensorCPU_(FILTER);
  Tensor* Y = OutputTensorCPU_(0);
  const int N = X.dim32(0), C = X.dim32(X.ndim() - 1);

  CAFFE_ENFORCE_EQ(X.ndim(), filter.ndim());
  const int M = filter.dim32(0);
  CAFFE_ENFORCE_EQ(filter.dim32(filter.ndim() - 1), C / group_);

  ConvPoolOpBase<CPUContext>::SetOutputSize(X, Y, filter.dim32(0));
  // The dimension of each kernel
  const int kernel_dim = this->KernelDim_();
  // The output image size is the spatial size of the output.
  const int output_image_size = this->GetDimsSize(*Y);
  // The col buffer is stored in HWC order as well - kernel_dim, and the height
  // and width.

  auto f = [&](vector<int32_t>* Y_int32) {
    Y_int32->resize(Y->numel());

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
    t_begin = chrono::system_clock::now();
#endif

    bool no_im2col = this->NoIm2ColNHWC_();

    // Im2Col, followed by gemm.
    auto f2 = [&](Tensor* col_buffer_) {
      const InType* Xdata = X.template data<InType>();
      const InType* col_buffer_data =
          no_im2col ? Xdata : this->template Im2ColNHWC_<InType>(col_buffer_);

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
      t_end = chrono::system_clock::now();
      dt = chrono::duration<double>(t_end - t_begin).count();
      LOG(INFO) << "this=" << this << " im2col: " << dt * 1e3 << " ms";
      t_begin = chrono::system_clock::now();
#endif

      // quantize col_buffer
      uint8_t* col_buffer_quantized_data = nullptr;
      vector<uint8_t> col_buffer_quantized;
      if (X.template IsType<uint8_t>()) {
        col_buffer_quantized_data = (uint8_t*)col_buffer_data;
      } else {
        col_buffer_quantized.resize(
            group_ * kernel_dim * output_image_size * N);
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
          size_t begin, end;
          std::tie(begin, end) = Get1DPartition(
              col_buffer_quantized.size(),
              dnnlowp_get_num_threads(),
              dnnlowp_get_thread_num());
          Quantize<uint8_t>(
              (const float*)col_buffer_data + begin,
              col_buffer_quantized.data() + begin,
              end - begin,
              in_qparams_[INPUT]);
        }
        col_buffer_quantized_data = col_buffer_quantized.data();
      }

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
      t_end = chrono::system_clock::now();
      dt = chrono::duration<double>(t_end - t_begin).count();
      LOG(INFO) << "this=" << this << " quantize col_buf: " << dt * 1e3
                << " ms";
      t_begin = chrono::system_clock::now();
#endif

      bool fuse_output_pipeline = !Wq_acc16_packed_.empty() &&
          nbits_in_non_outlier_ > 0 && !dequantize_output_;

      using namespace fbgemm;
      int row_offset_size_per_thread = -1;
      int x_pack_buf_size_per_thread = -1;
      if (!Wq_acc16_packed_.empty()) {
        if (fuse_output_pipeline) {
          row_offset_size_per_thread =
              PackAWithRowOffset<uint8_t, int16_t>::rowOffsetBufferSize();
          x_pack_buf_size_per_thread =
              PackAWithRowOffset<uint8_t, int16_t>::packedBufferSize();
          row_offsets_.resize(
              dnnlowp_get_max_threads() * row_offset_size_per_thread);
        } else {
          x_pack_buf_size_per_thread =
              PackAMatrix<uint8_t, int16_t>::packedBufferSize();
        }
        X_pack_buf_.resize(
            dnnlowp_get_max_threads() * x_pack_buf_size_per_thread);
      }

      if (nbits_in_non_outlier_ > 0) {
        // Main GEMM for non-outlier
        if (!Wq_acc16_packed_.empty()) {
          // fast path
          uint8_t* Y_uint8_data =
              OutputTensorCPU_(0)->template mutable_data<uint8_t>();
#ifdef _OPENMP
#pragma omp parallel
#endif
          {
            int tid = dnnlowp_get_thread_num();
            int group_begin, group_end;
            int i_begin, i_end;

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
              if (fuse_output_pipeline) {
                PackAWithRowOffset<uint8_t, int16_t> packA(
                    matrix_op_t::NoTranspose,
                    i_end - i_begin,
                    kernel_dim,
                    col_buffer_quantized_data +
                        (i_begin * group_ + group_id) * kernel_dim,
                    group_ * kernel_dim,
                    X_pack_buf_.data() + tid * x_pack_buf_size_per_thread,
                    1, // group
                    in_qparams_[INPUT].zero_point,
                    row_offsets_.data() + tid * row_offset_size_per_thread);

                DoNothing<> doNothingObj{};
                ReQuantizeOutput<ReluFused> reqObj(
                    doNothingObj,
                    this->RequantizationParams(group_id).real_multiplier,
                    out_qparams_.zero_point,
                    in_qparams_[INPUT].zero_point,
                    this->FilterQuantizationParams(group_id).zero_point,
                    packA.getRowOffsetBuffer(),
                    this->column_offsets_.data() + group_id * (M / group_),
                    InputSize() == 3
                        ? this->b_quantized_data_ + group_id * (M / group_)
                        : nullptr);

                if (nbits_in_non_outlier_ < 8) {
                  DoSpmdmOnInpBuffer<
                      typename ReQuantizeOutput<ReluFused>::outType,
                      int32_t,
                      ReQuantizeOutput<ReluFused>>
                      spmdmObj(
                          reqObj,
                          col_buffer_quantized_data +
                              (i_begin * group_ + group_id) * kernel_dim,
                          group_ * kernel_dim,
                          Wq_outlier_[group_id]);

                  fbgemmPacked(
                      packA,
                      *Wq_acc16_packed_[group_id],
                      Y_uint8_data + i_begin * M + group_id * (M / group_),
                      // Y_int32 is a temporal storage so it's OK to reuse
                      // group_begin
                      Y_int32->data() + i_begin * M +
                          group_begin * (M / group_),
                      M,
                      spmdmObj,
                      0, // thread_id
                      1); // num_threads
                } else {
                  fbgemmPacked(
                      packA,
                      *Wq_acc16_packed_[group_id],
                      Y_uint8_data + i_begin * M + group_id * (M / group_),
                      // Y_int32 is a temporal storage so it's OK to reuse
                      // group_begin
                      Y_int32->data() + i_begin * M +
                          group_begin * (M / group_),
                      M,
                      reqObj,
                      0, // thread_id
                      1); // num_threads
                }
              } else {
                // !fuse_output_pipeline
                PackAMatrix<uint8_t, int16_t> packA(
                    matrix_op_t::NoTranspose,
                    i_end - i_begin,
                    kernel_dim,
                    col_buffer_quantized_data +
                        (i_begin * group_ + group_id) * kernel_dim,
                    group_ * kernel_dim,
                    X_pack_buf_.data() + tid * x_pack_buf_size_per_thread,
                    1, // group
                    in_qparams_[INPUT].zero_point);

                DoNothing<int32_t, int32_t> doNothingObj{};
                memCopy<> memCopyObj(doNothingObj);
                fbgemmPacked(
                    packA,
                    *Wq_acc16_packed_[group_id],
                    Y_int32->data() + i_begin * M + group_id * (M / group_),
                    Y_int32->data() + i_begin * M + group_id * (M / group_),
                    M,
                    memCopyObj,
                    0, // thread_id
                    1); // num_threads
              }
            } // for each group
          } // omp parallel
        } else {
          // slow path
          conv_nhwc_acc16_ref_(
              group_,
              N,
              output_image_size,
              M,
              kernel_dim,
              col_buffer_quantized_data,
              W_quantized_.data(),
              Y_int32->data()
#ifdef DNNLOWP_ACC16_IN_SLOW_PATH
                  ,
              this
#endif
          );
        } // slow path
      } // nbits_in_non_outlier_ > 0

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
      t_end = chrono::system_clock::now();
      dt = chrono::duration<double>(t_end - t_begin).count();
      double ops = 2. * N * output_image_size * M * kernel_dim;
      double gops = ops / dt / 1e9;
      LOG(INFO) << "this=" << this << " GEMM: " << dt * 1e3 << " ms " << gops
                << " gops";
      t_begin = chrono::system_clock::now();
#endif

      if (!fuse_output_pipeline) {
        ConvOutlier_(col_buffer_quantized_data, Y_int32);
      }

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
      t_end = chrono::system_clock::now();
      dt = chrono::duration<double>(t_end - t_begin).count();
      LOG(INFO) << "this=" << this << " out-lier: " << dt * 1e3 << " ms";
      t_begin = chrono::system_clock::now();
#endif

      if (!fuse_output_pipeline) {
        this->RunOnDeviceEpilogueNHWC_(
            col_buffer_quantized_data, Y_int32->data());
      } else {
        if (!dequantize_output_) {
          PropagateOutputTensorQuantizationParams(this, 0, out_qparams_);
        }
      }
    }; // f2

    if (FLAGS_caffe2_force_shared_col_buffer || this->shared_buffer_) {
      runWithSharedBuffer<CPUContext>(this->ws_, f2);
    } else {
      f2(&(this->col_buffer_));
    }
  }; // f

  if (FLAGS_caffe2_dnnlowp_shared_int32_buffer) {
    this->RunWithSharedInt32Buffer_(f);
  } else {
    f(&(this->Y_int32_));
  }

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
  t_end = chrono::system_clock::now();
  dt = chrono::duration<double>(t_end - t_begin).count();
  LOG(INFO) << "this=" << this << " prologue: " << dt * 1e3 << " ms";
  t_begin = chrono::system_clock::now();

  t_end = chrono::system_clock::now();
  dt = chrono::duration<double>(t_end - t_very_begin).count();
  double ops = 2. * N * output_image_size * M * kernel_dim;
  double gops = ops / dt / 1e9;
  LOG(INFO) << "this=" << this << " " << OperatorBase::debug_def().type()
            << " output=" << OperatorBase::debug_def().output(0) << " "
            << N * output_image_size << "x" << M << "x" << kernel_dim
            << " G=" << group_ << " C/G=" << C / group_ << " K/G=" << M / group_
            << " R=" << kernel_h() << " S=" << kernel_w() << " : " << dt * 1e3
            << " ms " << gops << " gops";
#endif

  this->MeasureQuantizationError_();

  return true;
}

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Conv,
    DNNLOWP_ACC16,
    ConvDNNLowPAcc16Op<false>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    ConvRelu,
    DNNLOWP_ACC16,
    ConvDNNLowPAcc16Op<true>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8Conv,
    DNNLOWP_ACC16,
    ConvDNNLowPAcc16Op<false>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8ConvRelu,
    DNNLOWP_ACC16,
    ConvDNNLowPAcc16Op<true>);

} // namespace caffe2
