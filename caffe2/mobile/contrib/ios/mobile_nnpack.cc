#include "caffe2/core/common.h"

#ifndef CAFFE2_MOBILE
#error "Caffe2 mobile state not defined"
#endif

#if CAFFE2_MOBILE

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_op_shared.h"
#include "caffe2/operators/conv_pool_op_base.h"

#include "caffe2/utils/math.h"
#include "caffe2/utils/threadpool/pthreadpool_impl.h"
#include "nnpack.h"

CAFFE2_DEFINE_bool(caffe2_profile_nnpack, false, "");
namespace caffe2 {

void initNNPACK() {
  static std::once_flag once;
  std::call_once(once, []() {
    enum nnp_status nnpack_status = nnp_initialize();
    CAFFE_ENFORCE(nnpack_status == nnp_status_success, "NNPack is not supported here!");
  });
}

////////////////////////////////////////////////////////////////////////////////
// Definitions
////////////////////////////////////////////////////////////////////////////////

class NNPACKConvOp final : public ConvPoolOpBase<CPUContext> {
 public:
  NNPACKConvOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CPUContext>(operator_def, ws),
        algo_(getConvolutionAlgorithm()),
        kts_(getConvolutionTransformStrategy()),
        ws_(ws) {

    OPERATOR_NEEDS_FEATURE(this->order_ == StorageOrder::NCHW,
                           "NNPack only supports NCHW order. Please consider add \
            TransposeOp with axes=[0, 3, 1, 2] before NNPack Conv.");
    OPERATOR_NEEDS_FEATURE(pad_t() < kernel_h(), "NNPACK only supports pad < kernel size");
    OPERATOR_NEEDS_FEATURE(pad_b() < kernel_h(), "NNPACK only supports pad < kernel size");
    OPERATOR_NEEDS_FEATURE(pad_l() < kernel_w(), "NNPACK only supports pad < kernel size");
    OPERATOR_NEEDS_FEATURE(pad_r() < kernel_w(), "NNPACK only supports pad < kernel size");

    createSharedBuffer<CPUContext>(ws);
  }

  bool RunOnDeviceWithOrderNCHW() override;

 private:
  nnp_convolution_algorithm getConvolutionAlgorithm() const;
  nnp_convolution_transform_strategy getConvolutionTransformStrategy() const;

  const nnp_convolution_algorithm algo_;
  // Modified after precomputing the kernels. State transitions are:
  // - precompute -> (first call to Run()) -> reuse (on successful precompute)
  //                                       -> compute (on failing precompute)
  // - compute
  nnp_convolution_transform_strategy kts_;
  Workspace* ws_;
  // Per-group transformed filters
  std::vector<TensorCPU*> transformedFilters_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////

nnp_convolution_algorithm NNPACKConvOp::getConvolutionAlgorithm() const {
  if (!OperatorBase::HasSingleArgumentOfType<std::string>("algo")) {
    // No preference is stated. Heuristics for the best mobile device
    // algorithm are different than NNPACK's version, as Winograd
    // tends to be a lot faster. Use Winograd if the convolution
    // is 3x3d1s1.
    if (kernel_h() == 3 && kernel_w() == 3 && dilation_h() == 1 && dilation_w() == 1 &&
        stride_h() == 1 && stride_w() == 1) {
      // use Winograd
      return nnp_convolution_algorithm_wt8x8;
    }

    return nnp_convolution_algorithm_auto;
  }

  // Otherwise, there is a preference.
  auto algo = OperatorBase::GetSingleArgument<std::string>("algo", "AUTO");
  if (algo == "AUTO") {
    return nnp_convolution_algorithm_auto;
  }
  if (algo == "WINOGRAD") {
    return nnp_convolution_algorithm_wt8x8;
  }
  if (algo == "WINOGRAD_FP16") {
    return nnp_convolution_algorithm_wt8x8_fp16;
  }
  if (algo == "FT16") {
    return nnp_convolution_algorithm_ft16x16;
  }
  if (algo == "FT8") {
    return nnp_convolution_algorithm_ft8x8;
  }
  if (algo == "IMPLICIT_GEMM") {
    return nnp_convolution_algorithm_implicit_gemm;
  }
  if (algo == "DIRECT") {
    return nnp_convolution_algorithm_direct;
  }
  return nnp_convolution_algorithm_auto;
}

nnp_convolution_transform_strategy NNPACKConvOp::getConvolutionTransformStrategy() const {
  auto kts =
      OperatorBase::GetSingleArgument<std::string>("convolution_transform_strategy", "COMPUTE");
  if (kts == "PRECOMPUTE") {
    return nnp_convolution_transform_strategy_precompute;
  }
  // Default to computing each time.
  return nnp_convolution_transform_strategy_compute;
}

bool NNPACKConvOp::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto& filter = Input(1);
  auto& bias = Input(2);
  auto* Y = Output(0);
  CAFFE_ENFORCE(X.ndim() == 4, "Input dim should be 4");
  const int N = X.dim32(0), C = X.dim32(1), H = X.dim32(2), W = X.dim32(3);
  CAFFE_ENFORCE(filter.ndim(), 4);
  const int M = filter.dim32(0);
  CAFFE_ENFORCE(C % this->group_ == 0, "");
  CAFFE_ENFORCE(M % this->group_ == 0, "");
  CAFFE_ENFORCE(filter.dim32(1) == C / this->group_, "");
  CAFFE_ENFORCE(filter.dim32(2) == kernel_h(), "");
  CAFFE_ENFORCE(filter.dim32(3) == kernel_w(), "");
  CAFFE_ENFORCE(bias.ndim() == 1, "");
  CAFFE_ENFORCE(bias.dim32(0) == M, "");
  ConvPoolOpBase<CPUContext>::SetOutputSize(X, Y, filter.dim32(0));
  const int oH = Y->dim32(2), oW = Y->dim32(3);
  if (N > 1) {
    // NNPack only supports stride = 1 when doing batch feedforward
    CAFFE_ENFORCE(stride_h() == 1, "");
    CAFFE_ENFORCE(stride_w() == 1, "");
  }

  const size_t batch_size = X.dim32(0);
  const size_t input_channels = X.dim32(1);
  const size_t output_channels = Y->dim32(1);
  const nnp_size input_size = {.width = static_cast<size_t>(X.dim32(3)),
                               .height = static_cast<size_t>(X.dim32(2))};
  // filter is MCHW
  const nnp_size kernel_size = {.width = static_cast<size_t>(filter.dim32(3)),
                                .height = static_cast<size_t>(filter.dim32(2))};
  // pad is tblr
  const nnp_padding padding = {.top = static_cast<size_t>(pad_t()),
                               .right = static_cast<size_t>(pad_r()),
                               .bottom = static_cast<size_t>(pad_b()),
                               .left = static_cast<size_t>(pad_l())};

  const nnp_size output_subsample = {.width = static_cast<size_t>(stride_w()),
                                     .height = static_cast<size_t>(stride_h())};
  initNNPACK();
  pthreadpool pool(ws_->GetThreadPool());

  runWithSharedBuffer<CPUContext>(ws_, [&](Tensor<CPUContext>* buffer) {
    struct nnp_allocator allocator;
    allocator.ctx = buffer;
    allocator.allocator = [](void* ctx, size_t bytes) -> void* {
      Tensor<CPUContext>* b = (Tensor<CPUContext>*)ctx;
      // Uses float in ConvPoolOpBase (TODO: fix), so use float here.
      // Just in case bytes are odd, but shouldn't be.
      const size_t num_elements = (bytes + sizeof(float) - 1) / sizeof(float);
      b->Resize(num_elements);
      return (void*)b->template mutable_data<float>();
    };

    if (kts_ == nnp_convolution_transform_strategy_precompute) {
      transformedFilters_.resize(group_);

      for (auto g = 0; g < group_; ++g) {
        nnp_profile profile;
        const auto status =
            nnp_convolution_inference(algo_,
                                      nnp_convolution_transform_strategy_precompute,
                                      C / group_,
                                      M / group_,
                                      input_size,
                                      padding,
                                      kernel_size,
                                      output_subsample,
                                      nullptr,
                                      filter.template data<float>() + filter.size() / group_ * g,
                                      nullptr,
                                      nullptr,
                                      &pool,
                                      FLAGS_caffe2_profile_nnpack ? &profile : nullptr,
                                      &allocator);
        if (status != nnp_status_success) {
          // e.g. unsupported algorithm - i.e. passing precompute to a 1x1 direct conv.
          LOG(ERROR) << "Failed to precompute kernels, falling back to compute";
          kts_ = nnp_convolution_transform_strategy_compute;
          break;
        }

        auto* trnsFilter = ws_->CreateBlob(debug_def().name() + "_transformed_" + to_string(g))
                               ->GetMutable<TensorCPU>();
        trnsFilter->CopyFrom<CPUContext>(*buffer);
        transformedFilters_[g] = trnsFilter;
      }

      // Now, we've precomputed all our filters. Switch to reuse so we know to
      // reuse these in the future.
      if (kts_ == nnp_convolution_transform_strategy_precompute) {
        CAFFE_ENFORCE_EQ(transformedFilters_.size(), group_);
        kts_ = nnp_convolution_transform_strategy_reuse;
      }

      // Enforce when we leave this block that we have transitioned out of the
      // precompute state.
      CAFFE_ENFORCE(kts_ != nnp_convolution_transform_strategy_precompute);
    }

    CAFFE_ENFORCE(kts_ == nnp_convolution_transform_strategy_reuse ||
                  kts_ == nnp_convolution_transform_strategy_compute);
    const auto N = X.dim32(0);
    for (auto n = 0; n < N; ++n) {
      for (auto g = 0; g < group_; ++g) {
        nnp_profile profile;
        const auto status = nnp_convolution_inference(
            algo_,
            kts_,
            C / group_,
            M / group_,
            input_size,
            padding,
            kernel_size,
            output_subsample,
            X.template data<float>() + n * H * W * C + g * H * W * (C / group_),
            kts_ == nnp_convolution_transform_strategy_reuse
                ? transformedFilters_[g]->template data<float>()
                : filter.template data<float>() + filter.size() / group_ * g,
            bias.template data<float>() + bias.size() / group_ * g,
            Y->template mutable_data<float>() + n * oH * oW * M + g * oH * oW * (M / group_),
            &pool,
            FLAGS_caffe2_profile_nnpack ? &profile : nullptr,
            &allocator);
        VLOG(1) << "NNPACK buffer size: " << buffer->size();
        CAFFE_ENFORCE(nnp_status_success == status, "");
        if (FLAGS_caffe2_profile_nnpack) {
          char buffer[1024];
          const double gmacs = double(Y->dim32(2) * Y->dim32(3) * Y->dim32(1) * X.dim32(1) *
                                      kernel_size.width * kernel_size.height / group_ / group_) /
                               1.0E9;
          const double gflops = 2 * gmacs / profile.total;
          auto ret =
              snprintf(buffer,
                       sizeof(buffer),
                       "H: %3zu, W: %3zu, iC: %3zu, oC: %3zu, K: %1zu, S: %1zu, P: %1zu, GMACs: "
                       "%4.2f, totalT: %6.3f, inputT: %6.3f, "
                       "kernelT: %6.3f, blockT: %6.3f, outputT: %6.3f, GFLOPS: %6.3f",
                       size_t(X.dim(2)),
                       size_t(X.dim(3)),
                       size_t(X.dim(1)),
                       size_t(Y->dim(1)),
                       size_t(kernel_size.width),
                       size_t(output_subsample.width),
                       size_t(padding.top),
                       gmacs,
                       profile.total * 1E3,
                       profile.input_transform * 1E3,
                       profile.kernel_transform * 1E3,
                       profile.block_multiplication * 1E3,
                       profile.output_transform * 1E3,
                       gflops);
          CAFFE_ENFORCE(ret > 0);
          LOG(INFO) << buffer;
        }
      }
    }
  });
  return true;
}

////////////////////////////////////////////////////////////////////////////////
// Definitions
////////////////////////////////////////////////////////////////////////////////

class NNPACKMaxPoolOp final : public ConvPoolOpBase<CPUContext> {
 public:
  NNPACKMaxPoolOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CPUContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(this->order_ == StorageOrder::NCHW,
                           "NNPack only supports NCHW order. Please consider add \
            TransposeOp with axes=[0, 3, 1, 2] before NNPack Conv.");
    OPERATOR_NEEDS_FEATURE(kernel_h() == 2, "NNPack only supports MaxPool kernel size 2*2!");
    OPERATOR_NEEDS_FEATURE(kernel_w() == 2, "NNPack only supports MaxPool kernel size 2*2!");
    OPERATOR_NEEDS_FEATURE(stride_h() == 2, "NNPack only supports MaxPool stride size 2*2!");
    OPERATOR_NEEDS_FEATURE(stride_w() == 2, "NNPack only supports MaxPool stride size 2*2!");
    OPERATOR_NEEDS_FEATURE(pad_t() == 0,
                           "NNPack Pooling differs from Caffe2 Pooling when pad > 0!");
    OPERATOR_NEEDS_FEATURE(pad_l() == 0,
                           "NNPack Pooling differs from Caffe2 Pooling when pad > 0!");
    OPERATOR_NEEDS_FEATURE(pad_r() == 0,
                           "NNPack Pooling differs from Caffe2 Pooling when pad > 0!");
    OPERATOR_NEEDS_FEATURE(pad_b() == 0,
                           "NNPack Pooling differs from Caffe2 Pooling when pad > 0!");
  }
  bool RunOnDeviceWithOrderNCHW() override;

 private:
};

////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////

bool NNPACKMaxPoolOp::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto* Y = Output(0);
  CAFFE_ENFORCE(X.ndim() == 4, "");
  const int H = X.dim32(2), W = X.dim32(3);
  CAFFE_ENFORCE(H % 2 == 0, "NNPack MaxPool differs from Caffe2 when Input Size is not even!");
  CAFFE_ENFORCE(W % 2 == 0, "NNPack MaxPool differs from Caffe2 when Input Size is not even!");
  ConvPoolOpBase<CPUContext>::SetOutputSize(X, Y, X.dim32(1));
  std::vector<int> pads({pad_t(), pad_b(), pad_l(), pad_r()});
  std::vector<int> stride({stride_h(), stride_w()});
  std::vector<int> pooling({kernel_h(), kernel_w()});

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
  initNNPACK();
  pthreadpool pool(ws_->GetThreadPool());

  const auto status = nnp_max_pooling_output(batch_size,
                                             input_channels,
                                             input_size,
                                             padding,
                                             pooling_size,
                                             pooling_stride,
                                             X.template data<float>(),
                                             Y->template mutable_data<float>(),
                                             &pool);
  CAFFE_ENFORCE(nnp_status_success == status, "");
  return true;
}

REGISTER_CPU_OPERATOR_WITH_ENGINE(Conv, NNPACK, NNPACKConvOp);
REGISTER_CPU_OPERATOR_WITH_ENGINE(MaxPool, NNPACK, NNPACKMaxPoolOp);

} // namespace caffe2

#endif // CAFFE2_MOBILE
