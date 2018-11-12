#include "conv_dnnlowp_op.h"
#include "dnnlowp_op.h"

// #define DNNLOWP_MEASURE_TIME_BREAKDOWN
#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
#include <chrono>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "caffe2/core/tensor_int8.h"
#include "caffe2/utils/cpuid.h"

#include <fbgemm/src/RefImplementations.h>

#include "dnnlowp_partition.h"
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

namespace caffe2 {

using namespace std;

template <typename T, bool ReluFused>
ConvDNNLowPOp<T, ReluFused>::ConvDNNLowPOp(
    const OperatorDef& operator_def,
    Workspace* ws)
    : BaseType(operator_def, ws) {
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
      OperatorBase::GetSingleArgument<bool>("quantize_groupwise", false);
}

template <typename T, bool ReluFused>
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

template <typename T, bool ReluFused>
bool ConvDNNLowPOp<T, ReluFused>::RunOnDeviceWithOrderNCHW() {
  const Tensor& X = InputTensorCPU_(INPUT);
  if (X.template IsType<T>()) {
    return RunOnDeviceWithOrderNCHWAndType_<T>();
  } else {
    assert(X.template IsType<float>());
    return RunOnDeviceWithOrderNCHWAndType_<float>();
  }
}

template <typename T, bool ReluFused>
bool ConvDNNLowPOp<T, ReluFused>::RunOnDeviceWithOrderNHWC() {
  const Tensor& X = InputTensorCPU_(INPUT);
  if (X.template IsType<T>()) {
    return RunOnDeviceWithOrderNHWCAndType_<T>();
  } else {
    assert(X.template IsType<float>());
    return RunOnDeviceWithOrderNHWCAndType_<float>();
  }
}

template <typename T, bool ReluFused>
bool ConvDNNLowPOp<T, ReluFused>::TakeDepthWise3x3FastPath_() {
  const Tensor& X = InputTensorCPU_(INPUT);
  return StorageOrder::NHWC == ConvPoolOpBase<CPUContext>::order_ &&
      is_same<T, uint8_t>::value && X.template IsType<T>() &&
      OperatorBase::debug_def().engine() != "DNNLOWP_ACC16" &&
      group_ == X.dim32(X.dim() - 1) && group_ % 8 == 0 &&
      this->kernel_.size() == 2 && kernel_h() == 3 && kernel_w() == 3 &&
      stride_h() == stride_w() && (stride_h() == 1 || stride_h() == 2) &&
      pad_t() == 1 && pad_b() == 1 && pad_l() == 1 && pad_r() == 1 &&
      !dequantize_output_ && GetCpuId().avx2() && !quantize_groupwise_;
}

template <typename T, bool ReluFused>
bool ConvDNNLowPOp<T, ReluFused>::TakeDepthWise3x3x3FastPath_() {
  const Tensor& X = InputTensorCPU_(INPUT);
  bool ret = StorageOrder::NHWC == ConvPoolOpBase<CPUContext>::order_ &&
      is_same<T, uint8_t>::value && X.template IsType<T>() &&
      OperatorBase::debug_def().engine() != "DNNLOWP_ACC16" &&
      group_ == X.dim32(X.dim() - 1) && group_ % 8 == 0 &&
      this->kernel_.size() == 3 && this->kernel_[0] == 3 &&
      this->kernel_[1] == 3 && this->kernel_[2] == 3 &&
      this->stride_[0] == this->stride_[1] &&
      this->stride_[0] == this->stride_[2] &&
      (this->stride_[0] == 1 || this->stride_[0] == 2) && this->pads_[0] == 1 &&
      this->pads_[1] == 1 && this->pads_[2] == 1 && this->pads_[3] == 1 &&
      this->pads_[4] == 1 && this->pads_[5] == 1 && !dequantize_output_ &&
      GetCpuId().avx2() && !quantize_groupwise_;
  return ret;
}

template <typename T, bool ReluFused>
int ConvDNNLowPOp<T, ReluFused>::KernelDim_() {
  int kernel_dim;
  const Tensor& X = InputTensorCPU_(INPUT);
  const auto& filter = InputTensorCPU_(FILTER);

  int C;
  int filter_offset;
  if (ConvPoolOpBase<CPUContext>::order_ == StorageOrder::NCHW) {
    C = X.dim32(1);
    filter_offset = 2;
  } else {
    C = X.dim32(X.dim() - 1);
    filter_offset = 1;
  }

  int kernel_dims_size = 1;
  for (int i = 0; i < this->kernel_.size(); ++i) {
    CAFFE_ENFORCE_EQ(filter.dim32(i + filter_offset), kernel_[i]);
    kernel_dims_size *= kernel_[i];
  }
  kernel_dim = C / group_ * kernel_dims_size;

  return kernel_dim;
}

template <typename T, bool ReluFused>
bool ConvDNNLowPOp<T, ReluFused>::NoIm2ColNHWC_() {
  if (TakeDepthWise3x3FastPath_() || TakeDepthWise3x3x3FastPath_()) {
    return true;
  }

  const Tensor& X = InputTensorCPU_(INPUT);
  Tensor* Y = OutputTensorCPU_(0);
  const int C = X.dim32(X.dim() - 1);
  int kernel_dim = KernelDim_();
  if (kernel_dim != (C / group_)) {
    return false;
  }

  for (auto i = 0; i < this->kernel_.size(); ++i) {
    if (Y->dim32(i + 1) != X.dim32(i + 1) || this->stride_[i] != 1 ||
        pads_[2 * i] != 0 || pads_[2 * i + 1] != 0) {
      return false;
    }
  }
  return true;
}

template <typename T, bool ReluFused>
void ConvDNNLowPOp<T, ReluFused>::PreComputeRowColumnOffsets_() {
  const auto& filter = InputTensorCPU_(FILTER);
  int kernel_dim = KernelDim_();
  int M = filter.dim32(0);

  // Pre-compute row_offset / column_offset
  vector<int>& offsets =
      StorageOrder::NCHW == ConvPoolOpBase<CPUContext>::order_
      ? row_offsets_
      : column_offsets_;

  if (offsets.empty()) {
    offsets.resize(M);
    for (int g = 0; g < filter_qparams_.size(); ++g) {
      int i_begin = g * (M / filter_qparams_.size());
      int i_end = i_begin + (M / filter_qparams_.size());
      for (int i = i_begin; i < i_end; ++i) {
        int32_t sum = 0;
        for (int k = 0; k < kernel_dim; ++k) {
          sum += W_quantized_[i * kernel_dim + k];
        }
        offsets[i] = sum - FilterQuantizationParams(g).zero_point * kernel_dim;
      }
    }
  }
}

template <typename T, bool ReluFused>
void ConvDNNLowPOp<T, ReluFused>::QuantizeBias_() {
  using namespace dnnlowp;

  const auto& filter = InputTensorCPU_(FILTER);
  int M = filter.dim32(0);

  // Quantize bias
  if (InputSize() == 3 &&
      ((!b_quantized_data_ && !b_dequantized_data_) ||
       in_qparams_[INPUT].scale != in_qparams_scale_old_)) {
    const auto& bias = InputTensorCPU_(BIAS);
    if (OperatorBase::InputIsType<int8::Int8TensorCPU>(BIAS)) {
      TensorQuantizationParams bias_qparams;
      bias_qparams.scale = OperatorBase::Input<int8::Int8TensorCPU>(BIAS).scale;
      bias_qparams.zero_point =
          OperatorBase::Input<int8::Int8TensorCPU>(BIAS).zero_point;
      CAFFE_ENFORCE_LE(
          std::abs(
              bias_qparams.scale -
              in_qparams_[INPUT].scale * FilterQuantizationParams(0).scale),
          1e-4);
      CAFFE_ENFORCE_EQ(bias_qparams.zero_point, 0);
      b_quantized_data_ = bias.template data<int32_t>();
      if (dequantize_output_) {
        b_dequantized_.resize(bias.numel());
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < b_dequantized_.size(); ++i) {
          b_dequantized_[i] =
              Dequantize<int32_t>(b_quantized_data_[i], bias_qparams);
        }
        b_dequantized_data_ = b_dequantized_.data();
      }
    } else {
      b_dequantized_data_ = bias.template data<float>();
      if (!dequantize_output_) {
        b_quantized_.resize(bias.numel());
        for (int g = 0; g < filter_qparams_.size(); ++g) {
          int i_begin = g * (M / filter_qparams_.size());
          int i_end = i_begin + (M / filter_qparams_.size());
          for (int i = i_begin; i < i_end; ++i) {
            b_quantized_[i] = Quantize<int32_t>(
                b_dequantized_data_[i],
                0,
                in_qparams_[INPUT].scale * FilterQuantizationParams(g).scale,
                32,
                true /* signed */);
          }
        }
        b_quantized_data_ = b_quantized_.data();
      }
    }
    in_qparams_scale_old_ = in_qparams_[INPUT].scale;

    CAFFE_ENFORCE(
        (dequantize_output_ && b_dequantized_data_) ||
        (!dequantize_output_ && b_quantized_data_));
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
      OperatorBase::debug_def().engine() != "DNNLOWP_ACC16" &&
      is_same<T, uint8_t>::value && GetCpuId().avx2();

  bool depthwise_3x3_fast_path = false, depthwise_3x3x3_fast_path = false;
  if (TakeDepthWise3x3FastPath_()) {
    depthwise_3x3_fast_path = true;
    packW = false;
  } else if (TakeDepthWise3x3x3FastPath_()) {
    depthwise_3x3x3_fast_path = true;
    packW = false;
  }

  if ((depthwise_3x3_fast_path && !Wq_depthwise_3x3_packed_) ||
      (depthwise_3x3x3_fast_path && !Wq_depthwise_3x3x3_packed_) ||
      (packW && Wq_packed_.empty()) || (!packW && W_quantized_.empty())) {
    W_quantized_.resize(filter.numel());
    if (quantize_groupwise_) {
      filter_qparams_.resize(group_);
      requantization_params_.resize(group_);
    } else {
      filter_qparams_.resize(1);
      requantization_params_.resize(1);
    }

    int signed_min = 1 << (qfactory_->GetWeightPrecision() - 1);
    if (OperatorBase::InputIsType<int8::Int8TensorCPU>(FILTER)) {
      if (quantize_groupwise_) {
        static int log_occurences = 0;
        if (log_occurences < 32) {
          ++log_occurences;
          LOG(WARNING) << "Cannot do group-wise quantization for "
                          "pre-quantized weight "
                       << OperatorBase::debug_def().input(FILTER);
        }
      }
      FilterQuantizationParams(0).scale =
          OperatorBase::Input<int8::Int8TensorCPU>(FILTER).scale;
      FilterQuantizationParams(0).zero_point =
          OperatorBase::Input<int8::Int8TensorCPU>(FILTER).zero_point -
          signed_min;

      const auto& W = InputTensorCPU_(FILTER);
      const T* W_data = W.template data<T>();
      for (auto i = 0; i < W.numel(); ++i) {
        W_quantized_[i] = W_data[i] - signed_min;
      }
    } else {
      for (int g = 0; g < filter_qparams_.size(); ++g) {
        size_t offset = g * (M / filter_qparams_.size()) * kernel_dim;
        filter_qparams_[g] = qfactory_->ChooseQuantizationParams(
            filter.template data<float>() + offset,
            (M / filter_qparams_.size()) * kernel_dim,
            true /*weight*/);

        // filter_qparams_[g] is computed for unsigned type.
        // Adjust for the fact that weight will actually use signed.
        FilterQuantizationParams(g).zero_point -= signed_min;

        Quantize<T_signed>(
            filter.template data<float>() + offset,
            W_quantized_.data() + offset,
            (M / filter_qparams_.size()) * kernel_dim,
            FilterQuantizationParams(g));
      }
    }

    if (depthwise_3x3_fast_path) {
      Wq_depthwise_3x3_packed_.reset(new fbgemm::Packed3x3ConvMatrix(
          group_, reinterpret_cast<const int8_t*>(W_quantized_.data())));
    } else if (depthwise_3x3x3_fast_path) {
      Wq_depthwise_3x3x3_packed_.reset(new fbgemm::Packed3x3x3ConvMatrix(
          group_, reinterpret_cast<const int8_t*>(W_quantized_.data())));
    } else if (packW) {
      // fast path using fbgemm
      Wq_packed_.resize(group_);
      for (int group_id = 0; group_id < group_; ++group_id) {
        Wq_packed_[group_id].reset(new fbgemm::PackBMatrix<int8_t>(
            fbgemm::matrix_op_t::Transpose,
            kernel_dim,
            M / group_,
            reinterpret_cast<const int8_t*>(W_quantized_.data()) +
                group_id * (M / group_) * kernel_dim,
            kernel_dim, // ld
            nullptr, // pmat
            1, // groups
            FilterQuantizationParams(group_id).zero_point));
      }
    } else {
      string reason;
      if (ConvPoolOpBase<CPUContext>::order_ != StorageOrder::NHWC) {
        reason = "fbgemm only supports NHWC layout";
      } else if (!is_same<T, uint8_t>::value) {
        reason = "fbgemm only supports 8-bit integers";
      } else if (!GetCpuId().avx2()) {
        reason = "fbgemm only supports AVX2+";
      } else if (
          OperatorBase::debug_def().engine() == "DNNLOWP_ACC16" ||
          depthwise_3x3_fast_path) {
        reason = "";
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
  }
}

/**
 * @return false if something goes wrong
 */
template <typename T, bool ReluFused>
bool ConvDNNLowPOp<T, ReluFused>::GetQuantizationParameters_() {
  using namespace dnnlowp;

  if (!this->arguments_parsed_) {
    ParseDNNLowPOperatorArguments(
        this, &dequantize_output_, &measure_quantization_error_, &followed_by_);

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
  if (!Wq_packed_.empty() && !FLAGS_caffe2_dnnlowp_dump_tensors) {
    // From here, W_quantized_ is not used anymore when we have Wq_packed_
    vector<T_signed>().swap(W_quantized_);
  }

  QuantizeBias_();

  bool fp32_executed = false;
  if (!dequantize_output_) {
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

    for (int g = 0; g < filter_qparams_.size(); ++g) {
      float real_multiplier = in_qparams_[INPUT].scale *
          FilterQuantizationParams(g).scale / out_qparams_.scale;
      requantization_params_[g] = qfactory_->ChooseRequantizationMultiplier(
          real_multiplier, out_qparams_);
    }
  }

  if (measure_quantization_error_ && Fp32Op_() && !fp32_executed) {
    // to measure quantization error, run ref impl.
    Fp32Op_()->DequantizeInput();
    Fp32Op_()->Get()->RunOnDevice();
  }

  return true;
}

template <typename T, bool ReluFused>
template <typename OutType>
void ConvDNNLowPOp<T, ReluFused>::RunOnDeviceEpilogueNCHW_(
    const T* col_buffer_quantized_data,
    int32_t* Y_int32,
    OutType* Y_data,
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
  int32_t *column_offsets = column_offsets_.data() + tid * Y_HxW;

  const dnnlowp::TensorQuantizationParams& filter_qparams =
      FilterQuantizationParams(group_id);
  for (int j = 0; j < Y_HxW; ++j) {
    int sum = 0;
    for (int k = 0; k < kernel_dim; ++k) {
      sum += col_buffer_quantized_data[k * Y_HxW + j];
    }
    column_offsets[j] = sum * filter_qparams.zero_point;
  }

  if (dequantize_output_) {
    for (int i = 0; i < M / group_; ++i) {
      int32_t row_offset = row_offsets_[i_offset + i];
      row_offset *= -in_qparams_[INPUT].zero_point;
      for (int j = 0; j < Y_HxW; ++j) {
        int32_t raw = Y_int32[i * Y_HxW + j] + row_offset - column_offsets[j];
        float dequantized =
            raw * in_qparams_[INPUT].scale * filter_qparams.scale;
        if (InputSize() == 3) {
          dequantized += b_dequantized_data_[i_offset + i];
        }
        if (ReluFused) {
          dequantized = std::max(0.f, dequantized);
        }
        Y_data[i * Y_HxW + j] = dequantized;
      }
    }
  } // dequantize_output_
  else {
    for (int i = 0; i < M / group_; ++i) {
      int32_t row_offset = row_offsets_[i_offset + i];
      row_offset *= -in_qparams_[INPUT].zero_point;
      if (InputSize() == 3) {
        row_offset += b_quantized_data_[i_offset + i];
      }
      for (int j = 0; j < Y_HxW; ++j) {
        int32_t raw = Y_int32[i * Y_HxW + j] + row_offset - column_offsets[j];
        if (ReluFused) {
          raw = std::max(0, raw);
        }
        Y_data[i * Y_HxW + j] =
            dnnlowp::Requantize<T>(raw, RequantizationParams(group_id));
      }
    }
  } // !dequantize_output_
}

template <typename T, bool ReluFused>
template <typename InType>
bool ConvDNNLowPOp<T, ReluFused>::RunOnDeviceWithOrderNCHWAndType_() {
  VLOG(2) << "Running DNNLOWP Conv";

  using namespace dnnlowp;

  // Get quantization parameters
  if (!GetQuantizationParameters_()) {
    return false;
  }

  const Tensor& X = InputTensorCPU_(INPUT);
  auto& filter = InputTensorCPU_(FILTER);
  Tensor* Y = OutputTensorCPU_(0);
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

  ConvPoolOpBase<CPUContext>::SetOutputSize(X, Y, filter.dim32(0));

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
  const InType* Xdata = X.template data<InType>();

  // We must not call mutable_data inside omp region
  float* Y_data_float = nullptr;
  T* Y_data_T = nullptr;
  if (dequantize_output_) {
    Y_data_float = Y->template mutable_data<float>();
  } else {
    Y_data_T = Y->template mutable_data<T>();
  }
  column_offsets_.resize(Y_HxW * dnnlowp_get_max_threads());

  auto f = [&](Tensor* col_buffer) {
    col_buffer->Resize(buffer_shape);
    vector<int> buffer_shape_per_thread(
        buffer_shape.begin() + 1, buffer_shape.end());
    InType* col_buffer_data = col_buffer->template mutable_data<InType>();

    auto f2 = [&](vector<int32_t>* Y_int32) {
      Y_int32->resize(M * Y_HxW * dnnlowp_get_max_threads());

    // Im2Col, followed by gemm.
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
                X.IsType<T>() ? in_qparams_[INPUT].zero_point : 0);
          } else {
            math::Im2ColNdNCHW<InType>(
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
                X.IsType<T>() ? in_qparams_[INPUT].zero_point : 0);
          }

          // quantize col_buffer
          T* col_buffer_quantized_data = nullptr;
          vector<T> col_buffer_quantized;
          if (X.template IsType<T>()) {
            col_buffer_quantized_data =
                (T*)col_buffer_data + tid * col_buffer_size;
          } else {
            col_buffer_quantized.resize(kernel_dim * Y_HxW);
            Quantize<T>(
                (const float*)col_buffer_data + tid * col_buffer_size,
                col_buffer_quantized.data(),
                col_buffer_quantized.size(),
                in_qparams_[INPUT]);
            col_buffer_quantized_data = col_buffer_quantized.data();
          }

          int32_t* Y_int32_temp =
              Y_int32->data() + ((M / group_) * group_id + M * tid) * Y_HxW;
          T_signed* W_quantized_group =
              W_quantized_.data() + (M / group_) * group_id * kernel_dim;

          for (int i = 0; i < M / group_; ++i) {
            for (int j = 0; j < Y_HxW; ++j) {
              int32_t sum = 0;
              for (int k = 0; k < kernel_dim; ++k) {
                int w = W_quantized_group[i * kernel_dim + k];
                int x = col_buffer_quantized_data[k * Y_HxW + j];
                sum += w * x;
              }
              Y_int32_temp[i * Y_HxW + j] = sum;
            } // j
          } // i

          if (dequantize_output_) {
            RunOnDeviceEpilogueNCHW_(
                col_buffer_quantized_data,
                Y_int32_temp,
                Y_data_float + (M * image_id + M / group_ * group_id) * Y_HxW,
                M / group_ * group_id,
                group_id);
          } else {
            RunOnDeviceEpilogueNCHW_(
                col_buffer_quantized_data,
                Y_int32_temp,
                Y_data_T + (M * image_id + M / group_ * group_id) * Y_HxW,
                M / group_ * group_id,
                group_id);
          }
        } // for each group
      } // for each image_id

      if (!dequantize_output_) {
        PropagateOutputTensorQuantizationParams(this, 0, out_qparams_);
      }
      MeasureQuantizationError_();
    }; // f2

    if (FLAGS_caffe2_dnnlowp_shared_int32_buffer) {
      this->RunWithSharedInt32Buffer_(f2);
    } else {
      f2(&Y_int32_);
    }
  }; // f

  if (FLAGS_caffe2_force_shared_col_buffer || this->shared_buffer_) {
    runWithSharedBuffer<CPUContext>(this->ws_, f);
  } else {
    f(&col_buffer_);
  }

  return true;
} // RunOnDeviceWithOrderNCHWAndType_

template <typename T, bool ReluFused>
void ConvDNNLowPOp<T, ReluFused>::RunOnDeviceEpilogueNHWC_(
    const T* col_buffer_quantized_data,
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
  if (dequantize_output_) {
    float* Ydata = Y->template mutable_data<float>();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < N * Y_HxW; ++i) {
      for (int group_id = 0; group_id < group_; ++group_id) {
        int32_t row_offset = 0;
        for (int k = 0; k < kernel_dim; ++k) {
          row_offset += col_buffer_quantized_data
              [(i * group_ + group_id) * kernel_dim + k];
        }
        row_offset *= FilterQuantizationParams(group_id).zero_point;

        for (int j = group_id * (M / group_); j < (group_id + 1) * (M / group_);
             ++j) {
          Y_int32[i * M + j] -=
              in_qparams_[INPUT].zero_point * column_offsets_[j] + row_offset;
          Ydata[i * M + j] = Y_int32[i * M + j] * in_qparams_[INPUT].scale *
                  FilterQuantizationParams(group_id).scale +
              ((InputSize() == 3) ? b_dequantized_data_[j] : 0.f);
          if (ReluFused) {
            Ydata[i * M + j] = std::max(Ydata[i * M + j], 0.f);
          }
        }
      } // for each group
    } // for each i
  } else {
    int32_t A_zero_point = in_qparams_[INPUT].zero_point;

    if (!dnnlowp::HasStaticQuantization(this)) {
      if (quantize_groupwise_) {
        static int log_occurences = 0;
        if (log_occurences < 32) {
          ++log_occurences;
          LOG(WARNING) << "Cannot do group-wise quantization without "
                          "static quantization of activations for "
                       << OperatorBase::debug_def().output(0);
        }
      }

      int32_t Y_int32_min = numeric_limits<int32_t>::max();
      int32_t Y_int32_max = numeric_limits<int32_t>::min();

#ifdef _OPENMP
#pragma omp parallel for reduction(min:Y_int32_min), reduction(max:Y_int32_max)
#endif
      for (int i = 0; i < N * Y_HxW; ++i) {
        for (int group_id = 0; group_id < group_; ++group_id) {
          int32_t row_offset = 0;
          for (int k = 0; k < kernel_dim; ++k) {
            row_offset += col_buffer_quantized_data
                [(i * group_ + group_id) * kernel_dim + k];
          }
          row_offset *= FilterQuantizationParams(0).zero_point;

          for (int j = group_id * (M / group_);
               j < (group_id + 1) * (M / group_);
               ++j) {
            int32_t raw = Y_int32[i * M + j] -
                A_zero_point * column_offsets_[j] - row_offset;
            if (b_quantized_data_) {
              raw += b_quantized_data_[j];
            }
            Y_int32_min = std::min(Y_int32_min, raw);
            Y_int32_max = std::max(Y_int32_max, raw);
          }
        } // for each group
      } // for each row i

      if (ReluFused) {
        Y_int32_min = std::max(0, Y_int32_min);
        Y_int32_max = std::max(0, Y_int32_max);
      }

      float Y_int32_scale =
          in_qparams_[INPUT].scale * FilterQuantizationParams(0).scale;
      out_qparams_ = qfactory_->ChooseQuantizationParams(
          Y_int32_scale * Y_int32_min, Y_int32_scale * Y_int32_max);

      float real_multiplier = Y_int32_scale / out_qparams_.scale;
      requantization_params_[0] = qfactory_->ChooseRequantizationMultiplier(
          real_multiplier, out_qparams_);
    }

    int32_t C_zero_point = out_qparams_.zero_point;

    T* Ydata = Y->template mutable_data<T>();

    using namespace fbgemm;
#ifdef __AVX2__
    if (is_same<T, uint8_t>::value && GetCpuId().avx2()) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int i = 0; i < N * Y_HxW; ++i) {
        for (int group_id = 0; group_id < group_; ++group_id) {
          int32_t row_offset;
          row_offsets_u8acc32_ref(
              1,
              kernel_dim,
              group_ * kernel_dim,
              reinterpret_cast<const uint8_t*>(
                  col_buffer_quantized_data +
                  (i * group_ + group_id) * kernel_dim),
              &row_offset);

          int32_t B_zero_point = FilterQuantizationParams(group_id).zero_point;
          float C_multiplier = RequantizationParams(group_id).real_multiplier;

          DoNothing<> doNothingObj{};
          ReQuantizeOutput<ReluFused> requantizationObj(
              doNothingObj,
              C_multiplier,
              C_zero_point,
              A_zero_point,
              B_zero_point,
              &row_offset,
              column_offsets_.data() + group_id * (M / group_),
              b_quantized_data_ ? b_quantized_data_ + group_id * (M / group_)
                                : nullptr);

          block_type_t block{0, 1, 0, M / group_};
          requantizationObj.template f<inst_set_t::avx2>(
              reinterpret_cast<uint8_t*>(
                  Ydata + i * M + group_id * (M / group_)),
              Y_int32 + i * M + group_id * (M / group_),
              block,
              M,
              M);
        } // for each group
      } // for each row i
    } else
#endif // __AVX2__
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int i = 0; i < N * Y_HxW; ++i) {
        for (int group_id = 0; group_id < group_; ++group_id) {
          int32_t B_zero_point = FilterQuantizationParams(group_id).zero_point;
          int32_t row_offset = 0;
          for (int k = 0; k < kernel_dim; ++k) {
            row_offset += col_buffer_quantized_data
                [(i * group_ + group_id) * kernel_dim + k];
          }
          row_offset *= B_zero_point;

          for (int j = group_id * (M / group_);
               j < (group_id + 1) * (M / group_);
               ++j) {
            int32_t raw = Y_int32[i * M + j] -
                A_zero_point * column_offsets_[j] - row_offset;
            if (b_quantized_data_) {
              raw += b_quantized_data_[j];
            }

            Ydata[i * M + j] =
                dnnlowp::Requantize<T>(raw, RequantizationParams(group_id));
            if (ReluFused) { // static if
              Ydata[i * M + j] =
                  std::max<int32_t>(C_zero_point, Ydata[i * M + j]);
            }
          }
        } // for each group
      } // for each row i
    } // !__AVX2__

    PropagateOutputTensorQuantizationParams(this, 0, out_qparams_);
  }
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
template <typename InType>
const InType* ConvDNNLowPOp<T, ReluFused>::Im2ColNHWC_(Tensor* col_buffer) {
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

  const InType* Xdata = X.template data<InType>();

  vector<int> buffer_shape(ndim);
  for (auto i = 0; i < ndim - 1; ++i) {
    buffer_shape[i] = Y->dim32(i);
  }
  buffer_shape[ndim - 1] = kernel_dim * group_;

  col_buffer->Resize(buffer_shape);

  InType* col_buffer_data = col_buffer->template mutable_data<InType>();

#ifdef _OPENMP
#pragma omp parallel for if (N > 1)
#endif
  for (int image_id = 0; image_id < N; ++image_id) {
    if (this->kernel_.size() <= 2) {
      math::Im2ColNHWC<InType>(
          C,
          X.dim32(1),
          this->kernel_.size() == 2 ? X.dim32(2) : 1,
          kernel_h(),
          this->kernel_.size() == 2 ? kernel_w() : 1,
          dilation_h(),
          this->kernel_.size() == 2 ? dilation_w() : 1,
          pad_t(),
          this->kernel_.size() == 2 ? pad_l() : 1,
          pad_b(),
          this->kernel_.size() == 2 ? pad_r() : 1,
          stride_h(),
          this->kernel_.size() == 2 ? stride_w() : 1,
          Xdata + image_id * input_offset,
          col_buffer_data + image_id * group_ * kernel_dim * Y_HxW,
          &context_,
          group_,
          X.IsType<T>() ? in_qparams_[INPUT].zero_point : 0);
    } else {
      math::Im2Col3DNHWC<InType>(
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
          X.IsType<T>() ? in_qparams_[INPUT].zero_point : 0);
    }
  }

  return col_buffer->template data<InType>();
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
        int w = W[j * kernel_dim + k];
        int x = col_buffer[(i * num_groups + group_id) * kernel_dim + k];
        sum += w * x;
      }
      Y[i * M + j] = sum;
    }
  }
}

template <typename T, bool ReluFused>
template <typename InType>
void ConvDNNLowPOp<T, ReluFused>::ConvNHWCCore_(
    const InType* col_buffer_data,
    const T* col_buffer_quantized_data,
    vector<int32_t>* Y_int32) {
  const Tensor& X = InputTensorCPU_(INPUT);
  auto& filter = InputTensorCPU_(FILTER);
  Tensor* Y = OutputTensorCPU_(0);
  const int N = X.dim32(0), C = X.dim32(X.dim() - 1);
  const int M = filter.dim32(0);
  const int kernel_dim = KernelDim_();
  const int Y_HxW = this->GetDimsSize(*Y);

  if (FLAGS_caffe2_dnnlowp_dump_tensors) {
    // Dump input activation
    StoreMatrixInMatrixMarketFormat(
        N * Y_HxW * group_,
        kernel_dim,
        col_buffer_quantized_data,
        OperatorBase::debug_def().input(INPUT));

    // Dump weight
    StoreMatrixInMatrixMarketFormat(
        group_ * M,
        kernel_dim,
        W_quantized_.data(),
        OperatorBase::debug_def().input(FILTER));
  }

  uint8_t* Y_uint8_data = nullptr;
  float* Y_float_data = nullptr;
  if (!Wq_packed_.empty()) {
    // fast path to use fbgemm
    if (dequantize_output_) {
      // Output is float
      Y_float_data = Y->template mutable_data<float>();
    } else {
      // Output is uint8_t
      Y_uint8_data = Y->template mutable_data<uint8_t>();
    }
  }

  if (TakeDepthWise3x3x3FastPath_()) {
    const InType* Xdata = X.template data<InType>();
    Y_uint8_data = OutputTensorCPU_(0)->template mutable_data<uint8_t>();

#ifdef _OPENMP
#pragma omp parallel
#endif
    fbgemm::depthwise_3x3x3_pad_1(
        N,
        X.dim32(1),
        X.dim32(2),
        X.dim32(3),
        C,
        this->stride_[0],
        this->stride_[1],
        this->stride_[2],
        in_qparams_[INPUT].zero_point,
        reinterpret_cast<const uint8_t*>(Xdata),
        FilterQuantizationParams(0).zero_point,
        *Wq_depthwise_3x3x3_packed_,
        requantization_params_[0].real_multiplier,
        out_qparams_.zero_point,
        Y_uint8_data,
        column_offsets_.data(),
        b_quantized_data_,
        ReluFused,
        dnnlowp_get_thread_num(),
        dnnlowp_get_num_threads());

    return;
  } else if (TakeDepthWise3x3FastPath_()) {
    const int H = X.dim32(1), W = X.dim32(2);
    const InType* Xdata = X.template data<InType>();
    Y_uint8_data = OutputTensorCPU_(0)->template mutable_data<uint8_t>();

#ifdef _OPENMP
#pragma omp parallel
#endif
    fbgemm::depthwise_3x3_pad_1(
        N,
        H,
        W,
        C,
        stride_h(),
        stride_w(),
        in_qparams_[INPUT].zero_point,
        reinterpret_cast<const uint8_t*>(Xdata),
        FilterQuantizationParams(0).zero_point,
        *Wq_depthwise_3x3_packed_,
        requantization_params_[0].real_multiplier,
        out_qparams_.zero_point,
        Y_uint8_data,
        column_offsets_.data(),
        b_quantized_data_,
        dnnlowp_get_thread_num(),
        dnnlowp_get_num_threads(),
        ReluFused);

    return;
  }

  using namespace fbgemm;
  int row_offset_size_per_thread = -1;
  int x_pack_buf_size_per_thread = -1;
  if (!Wq_packed_.empty()) {
    if (!Y_uint8_data && !X.template IsType<T>()) {
      row_offset_size_per_thread =
          PackAWithQuantRowOffset<uint8_t>::rowOffsetBufferSize();
      x_pack_buf_size_per_thread =
          PackAWithQuantRowOffset<uint8_t>::packedBufferSize();
    } else {
      row_offset_size_per_thread =
          PackAWithRowOffset<uint8_t>::rowOffsetBufferSize();
      x_pack_buf_size_per_thread =
          PackAWithRowOffset<uint8_t>::packedBufferSize();
    }
    row_offsets_.resize(dnnlowp_get_max_threads() * row_offset_size_per_thread);
    X_pack_buf_.resize(dnnlowp_get_max_threads() * x_pack_buf_size_per_thread);
  }

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    int tid = dnnlowp_get_thread_num();
    int group_begin, group_end;
    int i_begin, i_end;

    PartitionGroupedNHWCConv_(
        &group_begin,
        &group_end,
        &i_begin,
        &i_end,
        group_,
        N * Y_HxW,
        dnnlowp_get_num_threads(),
        tid);

    for (int group_id = group_begin; group_id < group_end; ++group_id) {
      if (!Wq_packed_.empty()) {
        // fast path to use fbgemm
        if (Y_uint8_data) {
          // Output is uint8_t
          PackAWithRowOffset<uint8_t> packA(
              matrix_op_t::NoTranspose,
              i_end - i_begin,
              kernel_dim,
              reinterpret_cast<const uint8_t*>(col_buffer_quantized_data) +
                  (i_begin * group_ + group_id) * kernel_dim,
              group_ * kernel_dim,
              // buffer for packed matrix
              X_pack_buf_.data() + tid * x_pack_buf_size_per_thread,
              1, // group
              in_qparams_[INPUT].zero_point,
              row_offsets_.data() + tid * row_offset_size_per_thread);

          DoNothing<> doNothingObj{};
          ReQuantizeOutput<ReluFused> outputProcObj(
              doNothingObj,
              RequantizationParams(group_id).real_multiplier,
              out_qparams_.zero_point,
              in_qparams_[INPUT].zero_point,
              FilterQuantizationParams(group_id).zero_point,
              packA.getRowOffsetBuffer(),
              column_offsets_.data() + group_id * (M / group_),
              InputSize() == 3 ? b_quantized_data_ + group_id * (M / group_)
                               : nullptr);

          fbgemmPacked(
              packA,
              *Wq_packed_[group_id],
              Y_uint8_data + i_begin * M + group_id * (M / group_),
              // Y_int32 is a temporal storage so it's OK to reuse group_begin
              Y_int32->data() + i_begin * M + group_begin * (M / group_),
              M,
              outputProcObj,
              0, // thread_id
              1); // num_threads
        } else {
          if (!X.template IsType<T>()) {
            // Both input and output are float
            PackAWithQuantRowOffset<uint8_t> packA(
                matrix_op_t::NoTranspose,
                i_end - i_begin,
                kernel_dim,
                reinterpret_cast<const float*>(col_buffer_data) +
                    (i_begin * group_ + group_id) * kernel_dim,
                group_ * kernel_dim,
                // buffer for packed matrix
                X_pack_buf_.data() + tid * x_pack_buf_size_per_thread,
                in_qparams_[INPUT].scale,
                in_qparams_[INPUT].zero_point,
                1, // groups
                row_offsets_.data() + tid * row_offset_size_per_thread);

            DoNothing<float, float> doNothingObj{};
            ReQuantizeForFloat<ReluFused> outputProcObj(
                doNothingObj,
                in_qparams_[INPUT].scale,
                FilterQuantizationParams(group_id).scale,
                in_qparams_[INPUT].zero_point,
                FilterQuantizationParams(group_id).zero_point,
                packA.getRowOffsetBuffer(),
                column_offsets_.data() + group_id * (M / group_),
                InputSize() == 3 ? b_dequantized_data_ + group_id * (M / group_)
                                 : nullptr);

            fbgemmPacked(
                packA,
                *Wq_packed_[group_id],
                Y_float_data + i_begin * M + group_id * (M / group_),
                reinterpret_cast<int32_t*>(Y_float_data) + i_begin * M +
                    group_id * (M / group_),
                M,
                outputProcObj,
                0, // thread_id
                1); // num_threads
          } else {
            PackAWithRowOffset<uint8_t> packA(
                matrix_op_t::NoTranspose,
                i_end - i_begin,
                kernel_dim,
                reinterpret_cast<const uint8_t*>(col_buffer_quantized_data) +
                    (i_begin * group_ + group_id) * kernel_dim,
                group_ * kernel_dim,
                // buffer for packed matrix
                X_pack_buf_.data() + tid * x_pack_buf_size_per_thread,
                1, // group
                in_qparams_[INPUT].zero_point,
                row_offsets_.data() + tid * row_offset_size_per_thread);

            DoNothing<float, float> doNothingObj{};
            ReQuantizeForFloat<ReluFused> outputProcObj(
                doNothingObj,
                in_qparams_[INPUT].scale,
                FilterQuantizationParams(group_id).scale,
                in_qparams_[INPUT].zero_point,
                FilterQuantizationParams(group_id).zero_point,
                packA.getRowOffsetBuffer(),
                column_offsets_.data() + group_id * (M / group_),
                InputSize() == 3 ? b_dequantized_data_ + group_id * (M / group_)
                                 : nullptr);

            fbgemmPacked(
                packA,
                *Wq_packed_[group_id],
                Y_float_data + i_begin * M + group_id * (M / group_),
                reinterpret_cast<int32_t*>(Y_float_data) + i_begin * M +
                    group_id * (M / group_),
                M,
                outputProcObj,
                0, // thread_id
                1); // num_threads
          }
        }
      } else {
        // Wq_packed_.empty()
        conv_nhwc_ref_(
            group_id,
            group_,
            i_begin,
            i_end,
            M,
            kernel_dim,
            col_buffer_quantized_data,
            W_quantized_.data(),
            Y_int32->data());
      }
    } // for each group
  } // omp parallel
}

template <typename T, bool ReluFused>
template <typename InType>
bool ConvDNNLowPOp<T, ReluFused>::RunOnDeviceWithOrderNHWCAndType_() {
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
  Tensor* Y = OutputTensorCPU_(0);
  const int N = X.dim32(0), C = X.dim32(X.dim() - 1);
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

  ConvPoolOpBase<CPUContext>::SetOutputSize(X, Y, filter.dim32(0));
  // The dimension of each kernel
  const int kernel_dim = KernelDim_();
  // The output image size is the spatial size of the output.
  const int Y_HxW = this->GetDimsSize(*Y);
  // The col buffer is stored in HWC order as well - kernel_dim, and the height
  // and width.

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
  /*if (VLOG_IS_ON(3)) */ { t_begin = chrono::system_clock::now(); }
#endif

  bool no_im2col = NoIm2ColNHWC_();
  auto f = [&](vector<int32_t>* Y_int32) {
    if (!TakeDepthWise3x3FastPath_() && !TakeDepthWise3x3x3FastPath_()) {
      Y_int32->resize(Y->numel());
    }

    // Im2col, followed by gemm.
    auto f2 = [&](Tensor* col_buffer_) {
      const InType* Xdata = X.template data<InType>();
      const InType* col_buffer_data =
          no_im2col ? Xdata : Im2ColNHWC_<InType>(col_buffer_);

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
      /*if (VLOG_IS_ON(3)) */ {
        t_end = chrono::system_clock::now();
        double dt = chrono::duration<double>(t_end - t_begin).count();
        LOG(INFO) << "this=" << this << " im2col: " << dt * 1e3 << " ms";
        t_begin = chrono::system_clock::now();
      }
#endif

      // quantize col_buffer
      const T* col_buffer_quantized_data = nullptr;
      vector<T> col_buffer_quantized;
      if (Wq_packed_.empty() || X.template IsType<T>() || !dequantize_output_) {
        if (X.template IsType<T>()) {
          col_buffer_quantized_data =
              reinterpret_cast<const T*>(col_buffer_data);
        } else {
          col_buffer_quantized.resize(G * kernel_dim * Y_HxW * N);
          Quantize<T>(
              reinterpret_cast<const float*>(col_buffer_data),
              col_buffer_quantized.data(),
              col_buffer_quantized.size(),
              in_qparams_[INPUT]);
          col_buffer_quantized_data = col_buffer_quantized.data();
        }
      }

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
      /*if (VLOG_IS_ON(3)) */ {
        t_end = chrono::system_clock::now();
        double dt = chrono::duration<double>(t_end - t_begin).count();
        LOG(INFO) << "this=" << this << " quantize col_buf: " << dt * 1e3
                  << " ms";
        t_begin = chrono::system_clock::now();
      }
#endif

      ConvNHWCCore_(col_buffer_data, col_buffer_quantized_data, Y_int32);

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
      /*if (VLOG_IS_ON(3)) */ {
        t_end = chrono::system_clock::now();
        double dt = chrono::duration<double>(t_end - t_begin).count();
        LOG(INFO) << "this=" << this << " GEMM: " << dt * 1e3 << " ms";
        t_begin = chrono::system_clock::now();
      }
#endif

      if (!Wq_packed_.empty() || Wq_depthwise_3x3_packed_ ||
          Wq_depthwise_3x3x3_packed_) {
        // In fast path with fbgemm except when
        // rescaling quantized numbers should've been already done.
        if (!dequantize_output_) {
          PropagateOutputTensorQuantizationParams(this, 0, out_qparams_);
        }
      } else {
        RunOnDeviceEpilogueNHWC_(col_buffer_quantized_data, Y_int32->data());
      }
    }; // f2

    if (FLAGS_caffe2_force_shared_col_buffer || this->shared_buffer_) {
      runWithSharedBuffer<CPUContext>(this->ws_, f2);
    } else {
      f2(&col_buffer_);
    }
  }; // f

  if (FLAGS_caffe2_dnnlowp_shared_int32_buffer) {
    this->RunWithSharedInt32Buffer_(f);
  } else {
    f(&Y_int32_);
  }

#ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
  /*if (VLOG_IS_ON(3)) */ {
    t_end = chrono::system_clock::now();
    double dt = chrono::duration<double>(t_end - t_begin).count();
    LOG(INFO) << "this=" << this << " prologue: " << dt * 1e3 << " ms";
    t_begin = chrono::system_clock::now();

    t_end = chrono::system_clock::now();
    const int M = filter.dim32(0);
    double ops = 2. * N * Y_HxW * M * kernel_dim;
    dt = chrono::duration<double>(t_end - t_very_begin).count();
    double gops = ops / dt / 1e9;
    LOG(INFO) << "this=" << this << " " << OperatorBase::debug_def().type()
              << " output=" << OperatorBase::debug_def().output(0) << " "
              << N * Y_HxW << "x" << M << "x" << kernel_dim << " G=" << group_
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

OPERATOR_SCHEMA(ConvRelu).NumInputs(2, 3).NumOutputs(1).TensorInferenceFunction(
    ConvPoolOpBase<CPUContext>::TensorInferenceForConv);

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
