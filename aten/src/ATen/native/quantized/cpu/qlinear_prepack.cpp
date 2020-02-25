#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <ATen/quantized/Quantizer.h>
#include <algorithm>
#include <vector>

namespace caffe2 {
#ifdef USE_FBGEMM
// Required for cpp_custom_type_hack to work
CAFFE_KNOWN_TYPE(PackedLinearWeight);
CAFFE_KNOWN_TYPE(PackedLinearWeightFp16);
#endif // USE_FBGEMM
#ifdef USE_PYTORCH_QNNPACK
// Required for cpp_custom_type_hack to work
CAFFE_KNOWN_TYPE(PackedLinearWeightsQnnp);
#endif // USE_PYTORCH_QNNPACK
} // namespace caffe2

namespace at {
namespace native {
namespace {

class QLinearPackWeightInt8 final : public c10::OperatorKernel {
 public:
#ifdef USE_FBGEMM
  // Calculate the column offsets.
  // Note this includes the sum of the columns as well as the scalar term
  // B_zero_point * K, whereas the row_offsets created by
  // PackAWithQuantRowOffset is only the sum of the A rows.
  void calc_col_offsets_transpose(
      int K,
      int N,
      const int8_t* Bint8,
      int32_t* B_zero_point,
      int32_t* col_offsets,
      c10::QScheme qtype) {
    for (size_t i = 0; i < N; ++i) {
      int32_t sum = 0;
      for (size_t j = 0; j < K; ++j) {
        sum += Bint8[i * K + j];
      }
      if (qtype == kPerTensorAffine) {
        col_offsets[i] = sum - B_zero_point[0] * K;
      } else {
        col_offsets[i] = sum - B_zero_point[i] * K;
      }
    }
  }
  at::Tensor fbgemm_linear_prepack(
      at::Tensor weight,
      c10::optional<Tensor> bias) {
    TORCH_CHECK(
        weight.dim() == 2,
        "The weight tensor for quantized::linear_prepack (fbgemm) should"
        " be 2-dimensional.");

    auto N = weight.size(0);
    auto K = weight.size(1);

    // TODO: contiguous is called for further JIT optimizations.
    auto weight_contig = weight.contiguous();
    const auto qtype = weight.qscheme();
    std::vector<int32_t> weight_zero_points_int32(1, 0);
    if (qtype == kPerTensorAffine) {
      weight_zero_points_int32[0] = weight.q_zero_point();
    } else if (qtype == kPerChannelAffine) {
      weight_zero_points_int32.resize(N, 0);
      for (int i = 0; i < N; ++i) {
        weight_zero_points_int32[i] =
            weight.q_per_channel_zero_points()[i].item<int32_t>();
      }
    }
    std::vector<float> weight_scales_float(1, 0.0);
    if (qtype == kPerTensorAffine) {
      weight_scales_float[0] = weight.q_scale();
    } else if (qtype == kPerChannelAffine) {
      weight_scales_float.resize(N, 0.0);
      for (int i = 0; i < N; ++i) {
        weight_scales_float[i] = weight.q_per_channel_scales()[i].item<float>();
      }
    }

    int8_t* weight_ptr_int8 =
        reinterpret_cast<int8_t*>(weight_contig.data_ptr<c10::qint8>());

    std::vector<int32_t> col_offsets(N);
    calc_col_offsets_transpose(
        /*K=*/K,
        /*N=*/N,
        /*Bint8=*/weight_ptr_int8,
        /*B_zero_point=*/weight_zero_points_int32.data(),
        /*col_offsets=*/col_offsets.data(),
        /*qtype=*/qtype);

    c10::optional<at::Tensor> bias_contig;
    if (bias.has_value()) {
      Tensor bias_vec = bias.value();
      TORCH_CHECK(bias_vec.dim() == 1, "bias should be a vector (1D Tensor)");
      TORCH_CHECK(
          bias_vec.size(0) == N,
          "bias should have N elements: " + std::to_string(N));
      bias_contig = bias->contiguous();
    }
    auto ret_ptr = std::make_unique<PackedLinearWeight>(PackedLinearWeight{
        std::make_unique<fbgemm::PackBMatrix<int8_t>>(
            /*trans=*/fbgemm::matrix_op_t::Transpose,
            /*nRow=*/K,
            /*nCol=*/N,
            /*smat=*/weight_ptr_int8,
            /*ld=*/K,
            /*pmat=*/nullptr, // PackBMatrix manages ownership of pmat
            /*groups=*/1),
        bias_contig,
        col_offsets,
        weight_scales_float,
        weight_zero_points_int32,
        qtype});

    // TODO: we will need to replace this with torchscript classes at a later
    // point.
    return cpp_custom_type_hack::create(std::move(ret_ptr), weight.options());
  }
#endif
#ifdef USE_PYTORCH_QNNPACK
  at::Tensor qnnpack_linear_prepack(
      at::Tensor weight,
      c10::optional<Tensor> bias_in) {
    TORCH_CHECK(
        weight.dim() == 2,
        "quantized::linear_prepack (qnnpack): Weight tensor rank should be == 2");
    TORCH_CHECK(
        weight.qscheme() == kPerTensorAffine,
        "quantized::linear_prepack (qnnpack) only supports Per Tensor Quantization Scheme")

    int64_t rows_w = weight.size(0);
    Tensor bias_fp32;
    if (bias_in.has_value()) {
      bias_fp32 = bias_in.value();
    } else {
      bias_fp32 = at::zeros(rows_w, weight.options().dtype(at::kFloat));
    }
    TORCH_CHECK(
        !bias_fp32.defined() || (bias_fp32.ndimension() == 1 && bias_fp32.size(0) == rows_w),
        "quantized::linear_prepack (qnnpack): Given weight of size ",
        weight.sizes(),
        ", expected bias to be 1-dimensional with ",
        rows_w,
        " elements",
        ", but got bias of size ",
        bias_fp32.sizes(),
        " instead");

    Tensor weight_contig = weight.contiguous();
    auto weight_zp = weight.q_zero_point();

    initQNNPACK();

    // We set the pre-packed linear weights to nullptr below as we call pre-pack
    // during the first invocation of operator run. Refer to qlinear.cpp for more
    // details. TODO Update to actually call pre-pack here once bias is removed
    // from pre-packing step.
    auto wt_ptr = std::make_unique<PackedLinearWeightsQnnp>(
        PackedLinearWeightsQnnp{nullptr,
                                weight_contig, /* int8_t weight */
                                bias_fp32.contiguous(), /* fp32 bias */
                                c10::nullopt, /* input_scale */
                                weight.q_scale(),
                                weight_zp});
    return cpp_custom_type_hack::create(std::move(wt_ptr), weight.options());
  }
#endif
  at::Tensor operator()(at::Tensor weight, c10::optional<Tensor> bias) {
    auto& ctx = at::globalContext();

#ifdef USE_FBGEMM
    if (ctx.qEngine() == at::QEngine::FBGEMM) {
      return fbgemm_linear_prepack(weight, bias);
    }
#endif
#ifdef USE_PYTORCH_QNNPACK
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      return qnnpack_linear_prepack(weight, bias);
    }
#endif
    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::linear_prepack ",
        toString(ctx.qEngine()));
  }
};

class QLinearPackWeightFp16 final : public c10::OperatorKernel {
 public:
#ifdef USE_FBGEMM
  at::Tensor fbgemm_linear_prepack_fp16(
      at::Tensor weight,
      c10::optional<Tensor> bias) {
    const int64_t K = weight.size(1);
    const int64_t N = weight.size(0);
    Tensor weight_contig = weight.contiguous();
    float* weight_contig_ptr = weight_contig.data_ptr<float>();
    HandleWeightsSaturation(K * N, weight_contig_ptr);

    // TODO(mingzhe09088):
    // Consider using a functor here in PackedGemmMatrixFP16
    // Comments from (XQ): Not entirely sure this make_unique is safe.
    // make_unique is created with regular "new", and freed through
    // TypeMetaData::deleteFn in this function. This is perfectly fine if the
    // tensors are created and freed within this translation unit. It might be
    // very problematic if that tensor flows across dll boundaries.
    auto ptr = std::make_unique<PackedLinearWeightFp16>(PackedLinearWeightFp16{
        std::make_unique<fbgemm::PackedGemmMatrixFP16>(
            fbgemm::matrix_op_t::Transpose, K, N, 1, weight_contig_ptr),
        bias});
    return cpp_custom_type_hack::create(std::move(ptr), weight.options());
  }
#endif
#ifdef USE_PYTORCH_QNNPACK
  at::Tensor qnnpack_linear_prepack_fp16(
      at::Tensor weight,
      c10::optional<Tensor> bias_in) {
    TORCH_CHECK(
        false,
        "quantized::linear_prepack_fp16 is currently "
        "not supported by QNNPACK");
  }
#endif // USE_PYTORCH_QNNPACK
  at::Tensor operator()(at::Tensor weight, c10::optional<Tensor> bias) {
    auto& ctx = at::globalContext();
#ifdef USE_FBGEMM
    if (ctx.qEngine() == at::QEngine::FBGEMM) {
      return fbgemm_linear_prepack_fp16(weight, bias);
    }
#endif // USE_FBGEMM
#ifdef USE_PYTORCH_QNNPACK
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      return qnnpack_linear_prepack_fp16(weight, bias);
    }
#endif // USE_PYTORCH_QNNPACK
    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::linear_prepack_fp16 ",
        toString(ctx.qEngine()));
  }

 private:
#ifdef USE_FBGEMM
  float RawUint16ToFp16(unsigned short value) {
    // Convert raw 16 bits half precision floating point number
    // to single precision floating point number.
    const unsigned short sign_bits = value >> 15;
    const unsigned short exponent_bits = value >> 10 & 0x1f;
    const unsigned short significand_bits = value & 0x3ff;

    const float sign = sign_bits ? -1 : 1;
    const float significand = 1 +
        significand_bits * 0.0009765625f; // 0.0009765625f = 0x1p-10 = 2^-10;
    const float exponent = exponent_bits - 0xf;

    return sign * std::ldexp(significand, exponent);
  }

  template <typename T>
  bool CheckAndSaturate(T max_val, T* element) {
    if (*element > max_val) {
      *element = max_val;
      return true;
    }
    if (*element < -max_val) {
      *element = -max_val;
      return true;
    }
    return false;
  }

  // The range for using FP16 quantization of weights requires that the elements
  // should be in the range of [5.96e-8, 65504]. If it is out of range, then the
  // number will be saturated to max or min representable values by FP16.
  void HandleWeightsSaturation(int64_t N, float* weight) {
    const float kFp16Max = RawUint16ToFp16(0x7BFF);
    bool found_out_of_range = false;
    for (int64_t i = 0; i < N; ++i) {
      if (CheckAndSaturate<float>(kFp16Max, weight + i)) {
        found_out_of_range = true;
      }
    }
    if (found_out_of_range) {
      TORCH_WARN("FOUND weight out of range ");
    }
  }
#endif // USE_FBGEMM
};

static auto registry =
    c10::RegisterOperators()
        .op("quantized::linear_prepack(Tensor W, Tensor? B=None) -> Tensor W_prepack",
            c10::RegisterOperators::options().kernel<QLinearPackWeightInt8>(
                DispatchKey::QuantizedCPUTensorId))
        .op("quantized::linear_prepack_fp16(Tensor W, Tensor? B=None) -> Tensor W_prepack",
            c10::RegisterOperators::options().kernel<QLinearPackWeightFp16>(
                DispatchKey::CPUTensorId))
        .op("_quantized::linear_prepack(Tensor W, Tensor? B=None) -> Tensor W_prepack",
            c10::RegisterOperators::options().kernel<QLinearPackWeightInt8>(
                DispatchKey::QuantizedCPUTensorId))
        .op("_quantized::linear_prepack_fp16(Tensor W, Tensor? B=None) -> Tensor W_prepack",
            c10::RegisterOperators::options().kernel<QLinearPackWeightFp16>(
                DispatchKey::CPUTensorId));

} // namespace
} // namespace native
} // namespace at
