#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/quantized/Quantizer.h>

#include <algorithm>
#include <vector>

namespace caffe2 {
#ifdef USE_FBGEMM
// Required for cpp_custom_type_hack to work
CAFFE_KNOWN_TYPE(PackedLinearWeight);
#endif // USE_FBGEMM
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

  at::Tensor operator()(at::Tensor weight) {
    TORCH_CHECK(
        weight.dim() == 2,
        "The weight tensor for quantized::fbgemm_linear_prepack should be 2-dimensional.");

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

    auto ret_ptr = guts::make_unique<PackedLinearWeight>(PackedLinearWeight{
        guts::make_unique<fbgemm::PackBMatrix<int8_t>>(
            /*trans=*/fbgemm::matrix_op_t::Transpose,
            /*nRow=*/K,
            /*nCol=*/N,
            /*smat=*/weight_ptr_int8,
            /*ld=*/K,
            /*pmat=*/nullptr, // PackBMatrix manages ownership of pmat
            /*groups=*/1),
        col_offsets,
        weight_scales_float,
        weight_zero_points_int32,
        qtype});

    // TODO: we will need to replace this with torchscript classes at a later
    // point.
    return cpp_custom_type_hack::create(std::move(ret_ptr), weight.options());
  }
#else // USE_FBGEMM
  at::Tensor operator()(at::Tensor /* weight */
  ) {
    // We make a strong guarantee that models using these operators will have
    // the same numerics across different machines. Therefore, we do not provide
    // a fallback path and rather fail loudly if we cannot run FBGEMM.
    TORCH_CHECK(
        false, "This PyTorch installation was not built with FBGEMM operators");
  }
#endif // USE_FBGEMM
};

static auto registry = c10::RegisterOperators().op(
    "quantized::fbgemm_linear_prepack(Tensor W) -> Tensor W_prepack",
    c10::RegisterOperators::options().kernel<QLinearPackWeightInt8>(
        TensorTypeId::QuantizedCPUTensorId));
} // namespace
} // namespace native
} // namespace at
