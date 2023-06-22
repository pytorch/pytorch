
#include <iostream>

#include "caffe2/core/common.h"

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_op_shared.h"
#include "caffe2/operators/conv_pool_op_base.h"

#include "caffe2/utils/math.h"
#include "nnpack.h"

C10_DEFINE_bool(caffe2_profile_nnpack, false, "");
namespace caffe2 {

static void initNNPACK() {
  static std::once_flag once;
  std::call_once(once, []() {
    enum nnp_status nnpack_status = nnp_initialize();
    CAFFE_ENFORCE(
        nnpack_status == nnp_status_success, "NNPack is not supported here!");
  });
}

////////////////////////////////////////////////////////////////////////////////
// Definitions
////////////////////////////////////////////////////////////////////////////////

class NNPACKConvOp final : public ConvPoolOpBase<CPUContext> {
 public:
  NNPACKConvOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CPUContext>(operator_def, ws),
        algorithm_(getConvolutionAlgorithm()),
        activation_(getActivationType()),
        transformStrategy_(getConvolutionTransformStrategy()),
        ws_(ws) {
    OPERATOR_NEEDS_FEATURE(
        this->order_ == StorageOrder::NCHW,
        "NNPack only supports NCHW order. Please consider add \
            TransposeOp with axes=[0, 3, 1, 2] before NNPack Conv.");
    OPERATOR_NEEDS_FEATURE(
        pad_t() < kernel_h(), "NNPACK only supports pad < kernel size");
    OPERATOR_NEEDS_FEATURE(
        pad_b() < kernel_h(), "NNPACK only supports pad < kernel size");
    OPERATOR_NEEDS_FEATURE(
        pad_l() < kernel_w(), "NNPACK only supports pad < kernel size");
    OPERATOR_NEEDS_FEATURE(
        pad_r() < kernel_w(), "NNPACK only supports pad < kernel size");

    createSharedBuffer<CPUContext>(ws);
  }

  bool RunOnDeviceWithOrderNCHW() override;

 private:
  nnp_convolution_algorithm getConvolutionAlgorithm() const;
  nnp_convolution_transform_strategy getConvolutionTransformStrategy() const;
  nnp_activation getActivationType() const;

  const nnp_convolution_algorithm algorithm_;
  const nnp_activation activation_;
  // Modified after precomputing the kernels. State transitions are:
  // - precompute -> (first call to Run()) -> reuse (on successful precompute)
  //                                       -> compute (on failing precompute)
  // - compute
  nnp_convolution_transform_strategy transformStrategy_;
  Workspace* ws_;
  // Per-group transformed filters
  std::vector<TensorCPU*> transformedFilters_;
  // Zero-filled bias for convolutions without bias
  // This may be needed because NNPACK interface always expects conv with bias
  std::vector<float> dummyBias_;
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
    if (kernel_h() == 3 && kernel_w() == 3 && dilation_h() == 1 &&
        dilation_w() == 1 && stride_h() == 1 && stride_w() == 1) {
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

nnp_convolution_transform_strategy
NNPACKConvOp::getConvolutionTransformStrategy() const {
  auto kts = OperatorBase::GetSingleArgument<std::string>(
      "convolution_transform_strategy", "COMPUTE");
  if (kts == "PRECOMPUTE") {
    return nnp_convolution_transform_strategy_precompute;
  }
  // Default to computing each time.
  return nnp_convolution_transform_strategy_compute;
}

nnp_activation NNPACKConvOp::getActivationType() const {
  auto activation =
      OperatorBase::GetSingleArgument<std::string>("activation", "identity");
  if (activation == "identity") {
    return nnp_activation_identity;
  } else if (activation == "Relu") {
    return nnp_activation_relu;
  } else {
    CAFFE_THROW("unsupported activation type \"", activation, "\"");
  }
}

bool NNPACKConvOp::RunOnDeviceWithOrderNCHW() {
  /* Global variable with a unique ID of the pre-transformed kernel blob */
  volatile static uint32_t precomputed_transform_id = 0;

  auto& X = Input(0);
  auto& filter = Input(1);
  auto* Y = Output(0);
  CAFFE_ENFORCE(X.ndim() == 4, "Input dim should be 4");
  const int C = X.dim32(1), H = X.dim32(2), W = X.dim32(3);
  CAFFE_ENFORCE(filter.ndim() == 4, "");
  const int M = filter.dim32(0);
  CAFFE_ENFORCE(C % this->group_ == 0, "");
  CAFFE_ENFORCE(M % this->group_ == 0, "");
  CAFFE_ENFORCE(filter.dim32(1) == C / this->group_, "");
  CAFFE_ENFORCE(filter.dim32(2) == kernel_h(), "");
  CAFFE_ENFORCE(filter.dim32(3) == kernel_w(), "");
  ConvPoolOpBase<CPUContext>::SetOutputSize(X, Y, filter.dim32(0));
  const int oH = Y->dim32(2), oW = Y->dim32(3);

  const float* biasData = nullptr;
  if (InputSize() == 3) {
    /* Convolution with bias */
    auto& bias = Input(2);
    CAFFE_ENFORCE(bias.ndim() == 1, "");
    CAFFE_ENFORCE(bias.dim32(0) == M, "");
    biasData = bias.template data<float>();
  } else {
    /* NNPACK interface requires bias. Use a dummy zero-filled vector. */
    // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
    if (dummyBias_.size() != M) {
      dummyBias_.resize(M);
    }
    biasData = dummyBias_.data();
  }

  const nnp_size input_size = {
      .width = static_cast<size_t>(X.dim32(3)),
      .height = static_cast<size_t>(X.dim32(2))};
  // filter is MCHW
  const nnp_size kernel_size = {
      .width = static_cast<size_t>(filter.dim32(3)),
      .height = static_cast<size_t>(filter.dim32(2))};
  // pad is tblr
  const nnp_padding padding = {
      .top = static_cast<size_t>(pad_t()),
      .right = static_cast<size_t>(pad_r()),
      .bottom = static_cast<size_t>(pad_b()),
      .left = static_cast<size_t>(pad_l())};

  const nnp_size output_subsample = {
      .width = static_cast<size_t>(stride_w()),
      .height = static_cast<size_t>(stride_h())};
  initNNPACK();

#if !defined(USE_INTERNAL_PTHREADPOOL_IMPL)
  pthreadpool_t pool = nullptr;
#else
  pthreadpool_t pool = reinterpret_cast<pthreadpool_t>(ws_->GetThreadPool());
#endif

  runWithSharedBuffer<CPUContext>(ws_, [&](Tensor* buffer) {
    if (transformStrategy_ == nnp_convolution_transform_strategy_precompute) {
      transformedFilters_.resize(group_);

      size_t transformedFilterSize = 0;
      nnp_status status = nnp_convolution_inference(
          algorithm_,
          nnp_convolution_transform_strategy_precompute,
          C / group_,
          M / group_,
          input_size,
          padding,
          kernel_size,
          output_subsample,
          nullptr /* input */,
          nullptr /* filters */,
          nullptr /* bias */,
          nullptr /* output */,
          nullptr /* workspace buffer = transformed filter */,
          &transformedFilterSize,
          nnp_activation_identity,
          nullptr /* activation parameter */,
          pool,
          nullptr /* profile */);
      if (status == nnp_status_success) {
        /* For these convolution parameters filter transforms can be
         * pre-computed */

        /* Division with rounding up, in case size is not multiple of
         * sizeof(float) */
        const size_t transformedFilterElements =
            (transformedFilterSize + sizeof(float) - 1) / sizeof(float);

        for (auto g = 0; g < group_; g++) {
          transformedFilters_[g] = BlobGetMutableTensor(
              ws_->CreateBlob(
                  "__transformed_kernel_" +
                  to_string(
                      __sync_fetch_and_add(&precomputed_transform_id, 1))),
              CPU);
          transformedFilters_[g]->Resize(transformedFilterElements);

          status = nnp_convolution_inference(
              algorithm_,
              nnp_convolution_transform_strategy_precompute,
              C / group_,
              M / group_,
              input_size,
              padding,
              kernel_size,
              output_subsample,
              nullptr /* input */,
              filter.template data<float>() + filter.size() / group_ * g,
              nullptr /* bias */,
              nullptr /* output */,
              static_cast<void*>(
                  transformedFilters_[g]->template mutable_data<float>()),
              &transformedFilterSize,
              nnp_activation_identity,
              nullptr /* activation parameter */,
              pool,
              nullptr /* profile */);
          CAFFE_ENFORCE(
              nnp_status_success == status,
              "NNPACK convolution filter pre-transformation return error");
        }

        /*
         * Now, we've precomputed all our filter transformations.
         * Switch to reuse strategy to avoid doing transformation again on next
         * iteration.
         */
        if (transformStrategy_ ==
            nnp_convolution_transform_strategy_precompute) {
          CAFFE_ENFORCE_EQ(transformedFilters_.size(), group_);
          transformStrategy_ = nnp_convolution_transform_strategy_reuse;
        }
      } else {
        LOG(WARNING)
            << "Failed to query workspace size to precompute kernels, falling back to re-compute strategy";
        transformStrategy_ = nnp_convolution_transform_strategy_compute;
      }

      // Enforce when we leave this block that we have transitioned out of the
      // precompute state.
      CAFFE_ENFORCE(
          transformStrategy_ != nnp_convolution_transform_strategy_precompute);
    }

    CAFFE_ENFORCE(
        transformStrategy_ == nnp_convolution_transform_strategy_reuse ||
        transformStrategy_ == nnp_convolution_transform_strategy_compute);
    const auto N = X.dim32(0);
    for (auto n = 0; n < N; ++n) {
      for (auto g = 0; g < group_; ++g) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
        nnp_profile profile;
        size_t workspaceSize = buffer->nbytes();
        if (workspaceSize == 0) {
          /* Allocate some memory to ensure buffer pointer is not NULL. This
           * simplifies further logic. */
          buffer->Resize(1);
          workspaceSize = buffer->nbytes();
        }
        nnp_status status = nnp_convolution_inference(
            algorithm_,
            transformStrategy_,
            C / group_,
            M / group_,
            input_size,
            padding,
            kernel_size,
            output_subsample,
            X.template data<float>() + n * C * H * W + g * H * W * (C / group_),
            transformStrategy_ == nnp_convolution_transform_strategy_reuse
                ? transformedFilters_[g]->template data<float>()
                : filter.template data<float>() + filter.size() / group_ * g,
            biasData + M / group_ * g,
            Y->template mutable_data<float>() + n * oH * oW * M +
                g * oH * oW * (M / group_),
            static_cast<void*>(buffer->template mutable_data<float>()),
            &workspaceSize,
            activation_,
            nullptr /* activation parameter */,
            pool,
            FLAGS_caffe2_profile_nnpack ? &profile : nullptr);
        if (status == nnp_status_insufficient_buffer) {
          /* Query required workspace size, increase buffer, and try again */
          status = nnp_convolution_inference(
              algorithm_,
              transformStrategy_,
              C / group_,
              M / group_,
              input_size,
              padding,
              kernel_size,
              output_subsample,
              nullptr /* input */,
              nullptr,
              nullptr /* bias */,
              nullptr /* output */,
              nullptr /* workspace buffer */,
              &workspaceSize,
              activation_,
              nullptr /* activation parameter */,
              pool,
              nullptr /* profile */);
          if (status == nnp_status_success) {
            /* Division with rounding up, in case size is not multiple of
             * sizeof(float) */
            const size_t workspace_elements =
                (workspaceSize + sizeof(float) - 1) / sizeof(float);
            buffer->Resize(workspace_elements);

            /* Try convolution_inference again. If this time it fails, it is
             * fatal. */
            status = nnp_convolution_inference(
                algorithm_,
                transformStrategy_,
                C / group_,
                M / group_,
                input_size,
                padding,
                kernel_size,
                output_subsample,
                X.template data<float>() + n * C * H * W +
                    g * H * W * (C / group_),
                transformStrategy_ == nnp_convolution_transform_strategy_reuse
                    ? transformedFilters_[g]->template data<float>()
                    : filter.template data<float>() +
                        filter.size() / group_ * g,
                biasData + M / group_ * g,
                Y->template mutable_data<float>() + n * oH * oW * M +
                    g * oH * oW * (M / group_),
                static_cast<void*>(buffer->template mutable_data<float>()),
                &workspaceSize,
                activation_,
                nullptr /* activation parameter */,
                pool,
                FLAGS_caffe2_profile_nnpack ? &profile : nullptr);
          }
        }

        VLOG(1) << "NNPACK buffer size: " << buffer->nbytes();
        CAFFE_ENFORCE(
            nnp_status_success == status,
            "NNPACK convolution computation returned error");
        if (FLAGS_caffe2_profile_nnpack) {
          // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays)
          char buffer[1024];
          const double gmacs =
              double(
                  // NOLINTNEXTLINE(bugprone-integer-division)
                  Y->dim32(2) * Y->dim32(3) * Y->dim32(1) * X.dim32(1) *
                  kernel_size.width * kernel_size.height / group_ / group_) /
              1.0E9;
          // NOLINTNEXTLINE(clang-analyzer-core.UndefinedBinaryOperatorResult)
          const double gflops = 2 * gmacs / profile.total;
          auto ret = snprintf(
              buffer,
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
          std::cout << buffer << std::endl;
        }
      }
    }
  });
  return true;
}

REGISTER_CPU_OPERATOR_WITH_ENGINE(Conv, NNPACK, NNPACKConvOp);

} // namespace caffe2
