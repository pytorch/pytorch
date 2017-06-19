#ifdef CAFFE2_USE_MKL
#include <mkl_service.h>
#endif
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/operators/leaky_relu_op.h"
#include "caffe2/utils/math.h"
#include "nnpack.h"

CAFFE2_DEFINE_int(
    caffe2_nnpack_num_threads, 1,
    "The number of nnpack pthreadpool threads.");
CAFFE2_DEFINE_bool(
    caffe2_nnpack_use_mkl_num_threads, true,
    "If MKL is built, this sets nnpack to use the same number of threads as "
    "MKL does. This overrides caffe2_nnpack_num_threads if set.");

namespace caffe2 {
////////////////////////////////////////////////////////////////////////////////
// Helper Functions
////////////////////////////////////////////////////////////////////////////////

namespace {

nnp_convolution_algorithm get_nnp_convolution_algorithm(
    const std::string& algo) {
  if (algo == "AUTO") {
    return nnp_convolution_algorithm_auto;
  }
  if (algo == "WINOGRAD") {
    return nnp_convolution_algorithm_wt8x8;
  }
  if (algo == "FT16") {
    return nnp_convolution_algorithm_ft16x16;
  }
  if (algo == "FT8") {
    return nnp_convolution_algorithm_ft8x8;
  }
  return nnp_convolution_algorithm_auto;
}

nnp_convolution_transform_strategy get_nnp_convolution_transform_strategy(
    const std::string& kts) {
  if (kts == "BLOCK") {
    return nnp_convolution_transform_strategy_block_based;
  }
  if (kts == "TUPLE") {
    return nnp_convolution_transform_strategy_tuple_based;
  }
  return nnp_convolution_transform_strategy_block_based;
}

////////////////////////////////////////////////////////////////////////////////
// Thread Pool
////////////////////////////////////////////////////////////////////////////////

static pthreadpool_t nnpack_threadpool_ = nullptr;

pthreadpool_t nnpack_threadpool() {
  if (nnpack_threadpool_ == nullptr) {
    enum nnp_status nnpack_status = nnp_initialize();
    CAFFE_ENFORCE(
        nnpack_status == nnp_status_success, "NNPack is not supported here!");
    int num_threads = FLAGS_caffe2_nnpack_num_threads;
    if (FLAGS_caffe2_nnpack_use_mkl_num_threads) {
#ifdef CAFFE2_USE_MKL
      num_threads = mkl_get_max_threads();
#else
      VLOG(1) << "I am asked to use MKL num of threads for NNPACK but this "
                 "Caffe2 is not built with MKL. Skipping.";
#endif
    }
    nnpack_threadpool_ = pthreadpool_create(num_threads);
  }
  return nnpack_threadpool_;
}
}

////////////////////////////////////////////////////////////////////////////////
// NNPACK Ops
////////////////////////////////////////////////////////////////////////////////

class NNPACKConvOp final : public ConvPoolOpBase<CPUContext> {
 public:
  NNPACKConvOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CPUContext>(operator_def, ws),
        algo_(get_nnp_convolution_algorithm(
            OperatorBase::GetSingleArgument<std::string>("algo", "AUTO"))),
        kts_(get_nnp_convolution_transform_strategy(
            OperatorBase::GetSingleArgument<std::string>("kts", "TUPLE"))) {
    OPERATOR_NEEDS_FEATURE(
        this->order_ == StorageOrder::NCHW,
        "NNPack only supports NCHW order. Please consider adding "
        "TransposeOp with axes=[0, 3, 1, 2] before NNPack Conv.");
#ifdef CAFFE2_USE_FBCODE
    // Facebook's nnpack build assumes existence of avx2, so we explicitly
    // check if the machine has avx2 support.
    OPERATOR_NEEDS_FEATURE(
        __builtin_cpu_supports("avx2"), "NNPack requires AVX2");
#endif
  }

  bool RunOnDeviceWithOrderNCHW() override {
    auto& X = Input(0);
    auto& filter = Input(1);
    auto& bias = Input(2);
    auto* Y = Output(0);

    const int N = X.dim32(0), C = X.dim32(1), H = X.dim32(2), W = X.dim32(3);
    const int M = filter.dim32(0);

    CAFFE_ENFORCE(X.ndim() == 4, "Input dim should be 4");
    CAFFE_ENFORCE(filter.ndim(), 4);
    CAFFE_ENFORCE(C % this->group_ == 0, "");
    CAFFE_ENFORCE(M % this->group_ == 0, "");
    CAFFE_ENFORCE(filter.dim32(1) == C / this->group_, "");
    CAFFE_ENFORCE(filter.dim32(2) == this->kernel_h(), "");
    CAFFE_ENFORCE(filter.dim32(3) == this->kernel_w(), "");
    CAFFE_ENFORCE(bias.size() == M, "");

    ConvPoolOpBase<CPUContext>::SetOutputSize(X, Y, filter.dim32(0));
    const int oH = Y->dim32(2), oW = Y->dim32(3);

    if (N > 1) {
      // NNPack only supports stride = 1 when doing batch feedforward
      CAFFE_ENFORCE(this->stride_h() == 1, "");
      CAFFE_ENFORCE(this->stride_w() == 1, "");
    }
    std::vector<int> pads(
        {this->pad_t(), this->pad_b(), this->pad_l(), this->pad_r()});
    std::vector<int> stride({this->stride_h(), this->stride_w()});

    const size_t input_channels = X.dim32(1);
    const size_t output_channels = Y->dim32(1);
    const nnp_size input_size = {.width = static_cast<size_t>(X.dim32(3)),
                                 .height = static_cast<size_t>(X.dim32(2))};
    // filter is MCHW
    const nnp_size kernel_size = {
        .width = static_cast<size_t>(filter.dim32(3)),
        .height = static_cast<size_t>(filter.dim32(2))};
    // pad is tblr
    const nnp_padding padding = {.top = static_cast<size_t>(pads[0]),
                                 .right = static_cast<size_t>(pads[3]),
                                 .bottom = static_cast<size_t>(pads[1]),
                                 .left = static_cast<size_t>(pads[2])};

    const nnp_size output_subsample = {
        .width = static_cast<size_t>(stride[1]),
        .height = static_cast<size_t>(stride[0])};
    if (N == 1) {
      VLOG(1) << "Running inference mode";
      for (auto g = 0; g < group_; ++g) {
        const auto status = nnp_convolution_inference(
            algo_,
            kts_,
            C / group_,
            M / group_,
            input_size,
            padding,
            kernel_size,
            output_subsample,
            X.template data<float>() + g * H * W * (C / group_),
            filter.template data<float>() + filter.size() / group_ * g,
            bias.template data<float>() + bias.size() / group_ * g,
            Y->template mutable_data<float>() + g * oH * oW * (M / group_),
            nnpack_threadpool(),
            nullptr);
        CAFFE_ENFORCE(nnp_status_success == status, "");
      }
    } else {
      VLOG(1) << "Running batched mode";
      for (auto g = 0; g < group_; ++g) {
        const auto status = nnp_convolution_output(
            algo_,
            N,
            C / group_,
            M / group_,
            input_size,
            padding,
            kernel_size,
            X.template data<float>() + g * H * W * (C / group_),
            filter.template data<float>() + filter.size() / group_ * g,
            bias.template data<float>() + bias.size() / group_ * g,
            Y->template mutable_data<float>() + g * oH * oW * (M / group_),
            nnpack_threadpool(),
            nullptr);
        CAFFE_ENFORCE(nnp_status_success == status, "");
      }
    }
    return true;
  }

 private:
  const nnp_convolution_algorithm algo_;
  const nnp_convolution_transform_strategy kts_;
};

class NNPACKMaxPoolOp final : public ConvPoolOpBase<CPUContext> {
 public:
  NNPACKMaxPoolOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CPUContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(
        this->order_ == StorageOrder::NCHW,
        "NNPack only supports NCHW order. Please consider add "
        "TransposeOp with axes=[0, 3, 1, 2] before NNPack Conv.");
    OPERATOR_NEEDS_FEATURE(
        this->kernel_h() == 2, "NNPack only supports MaxPool kernel size 2*2!");
    OPERATOR_NEEDS_FEATURE(
        this->kernel_w() == 2, "NNPack only supports MaxPool kernel size 2*2!");
    OPERATOR_NEEDS_FEATURE(
        this->stride_h() == 2, "NNPack only supports MaxPool stride size 2*2!");
    OPERATOR_NEEDS_FEATURE(
        this->stride_w() == 2, "NNPack only supports MaxPool stride size 2*2!");
    OPERATOR_NEEDS_FEATURE(
        this->pad_t() == 0,
        "NNPack Pooling differs from Caffe2 Pooling when pad > 0!");
    OPERATOR_NEEDS_FEATURE(
        this->pad_l() == 0,
        "NNPack Pooling differs from Caffe2 Pooling when pad > 0!");
    OPERATOR_NEEDS_FEATURE(
        this->pad_r() == 0,
        "NNPack Pooling differs from Caffe2 Pooling when pad > 0!");
    OPERATOR_NEEDS_FEATURE(
        this->pad_b() == 0,
        "NNPack Pooling differs from Caffe2 Pooling when pad > 0!");
#ifdef CAFFE2_USE_FBCODE
    // Facebook's nnpack build assumes existence of avx2, so we explicitly
    // check if the machine has avx2 support.
    OPERATOR_NEEDS_FEATURE(
        __builtin_cpu_supports("avx2"), "NNPack requires AVX2");
#endif
  }

  bool RunOnDeviceWithOrderNCHW() override {
    auto& X = Input(0);
    auto* Y = Output(0);
    CAFFE_ENFORCE(X.ndim() == 4, "");
    const int H = X.dim32(2), W = X.dim32(3);
    ConvPoolOpBase<CPUContext>::SetOutputSize(X, Y, X.dim32(1));
    std::vector<int> pads(
        {this->pad_t(), this->pad_b(), this->pad_l(), this->pad_r()});
    std::vector<int> stride({this->stride_h(), this->stride_w()});
    std::vector<int> pooling({this->kernel_h(), this->kernel_w()});

    // Input X is in NCHW order
    const size_t batch_size = X.dim32(0);
    const size_t input_channels = X.dim32(1);
    const nnp_size input_size = {.width = static_cast<size_t>(X.dim32(3)),
                                 .height = static_cast<size_t>(X.dim32(2))};
    // pooling kernel
    const nnp_size pooling_size = {.width = static_cast<size_t>(pooling[1]),
                                   .height = static_cast<size_t>(pooling[0])};
    // pad is tblr
    const nnp_padding padding = {.top = static_cast<size_t>(pads[0]),
                                 .right = static_cast<size_t>(pads[3]),
                                 .bottom = static_cast<size_t>(pads[1]),
                                 .left = static_cast<size_t>(pads[2])};

    const nnp_size pooling_stride = {.width = static_cast<size_t>(stride[1]),
                                     .height = static_cast<size_t>(stride[0])};
    const auto status = nnp_max_pooling_output(
        batch_size,
        input_channels,
        input_size,
        padding,
        pooling_size,
        pooling_stride,
        X.template data<float>(),
        Y->template mutable_data<float>(),
        nnpack_threadpool());
    CAFFE_ENFORCE(nnp_status_success == status, "");
    return true;
  }

 private:
};

class NNPACKReluOp final : public Operator<CPUContext> {
 public:
  NNPACKReluOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {
#ifdef CAFFE2_USE_FBCODE
    // Facebook's nnpack build assumes existence of avx2, so we explicitly
    // check if the machine has avx2 support.
    OPERATOR_NEEDS_FEATURE(
        __builtin_cpu_supports("avx2"), "NNPack requires AVX2");
#endif
  }

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto* Y = Output(0);
    const auto status = nnp_relu_output(
        1,
        X.size(),
        X.template data<float>(),
        Y->template mutable_data<float>(),
        0.0,
        nnpack_threadpool());
    CAFFE_ENFORCE(nnp_status_success == status, "");
    return true;
  }

 private:
};

class NNPACKLeakyReluOp final : public LeakyReluOp<float, CPUContext> {
 public:
  NNPACKLeakyReluOp(const OperatorDef& operator_def, Workspace* ws)
      : LeakyReluOp<float, CPUContext>(operator_def, ws) {
#ifdef CAFFE2_USE_FBCODE
    // Facebook's nnpack build assumes existence of avx2, so we explicitly
    // check if the machine has avx2 support.
    OPERATOR_NEEDS_FEATURE(
        __builtin_cpu_supports("avx2"), "NNPack requires AVX2");
#endif
  }

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto* Y = Output(0);
    const auto status = nnp_relu_output(
        1,
        X.size(),
        X.template data<float>(),
        Y->template mutable_data<float>(),
        alpha_,
        nnpack_threadpool());
    CAFFE_ENFORCE(nnp_status_success == status, "");
    return true;
  }

 private:
};

REGISTER_CPU_OPERATOR_WITH_ENGINE(Conv, NNPACK, NNPACKConvOp);
REGISTER_CPU_OPERATOR_WITH_ENGINE(MaxPool, NNPACK, NNPACKMaxPoolOp);
REGISTER_CPU_OPERATOR_WITH_ENGINE(Relu, NNPACK, NNPACKReluOp);
REGISTER_CPU_OPERATOR_WITH_ENGINE(LeakyRelu, NNPACK, NNPACKLeakyReluOp);

} // namespace caffe2
