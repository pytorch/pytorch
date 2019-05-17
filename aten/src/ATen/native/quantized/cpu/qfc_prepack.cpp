#include <ATen/ATen.h>
#include <ATen/core/Type.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/quantized/Quantizer.h>

#include <algorithm>
#include <vector>

namespace caffe2 {
#ifdef USE_FBGEMM
// Required for cpp_custom_type_hack to work
CAFFE_KNOWN_TYPE(PackedFCWeight);
#endif // USE_FBGEMM
} // namespace caffe2

namespace at {
namespace native {
namespace {

class QFCPackWeightInt8 final : public c10::OperatorKernel {
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
      int32_t B_zero_point,
      int32_t* col_offsets) {
    for (size_t i = 0; i < N; ++i) {
      int32_t sum = 0;
      for (size_t j = 0; j < K; ++j) {
        sum += Bint8[i * K + j];
      }
      col_offsets[i] = sum - B_zero_point * K;
    }
  }

  at::Tensor operator()(at::Tensor weight) {
    auto N = weight.size(0);
    auto K = weight.size(1);

    int32_t weight_zero_point_int32 = weight.q_zero_point().toInt() - 128;

    // TODO: contiguous is called for further JIT optimizations.
    auto weight_contig = weight.contiguous();

    std::vector<int8_t> weight_int8(K * N);
    int8_t* weight_ptr_int8 = weight_int8.data();
    uint8_t* weight_ptr_uint8 =
        reinterpret_cast<uint8_t*>(weight_contig.data<c10::quint8>());
    convert_uint8_int8(K, N, weight_ptr_uint8, weight_ptr_int8);

    std::vector<int32_t> col_offsets(N);
    calc_col_offsets_transpose(
        /*K=*/K,
        /*N=*/N,
        /*Bint8=*/weight_ptr_int8,
        /*B_zero_point=*/weight_zero_point_int32,
        /*col_offsets=*/col_offsets.data());

    auto ret_ptr = guts::make_unique<PackedFCWeight>(PackedFCWeight{
        guts::make_unique<fbgemm::PackBMatrix<int8_t>>(
            /*trans=*/fbgemm::matrix_op_t::Transpose,
            /*nRow=*/K,
            /*nCol=*/N,
            /*smat=*/weight_ptr_int8,
            /*ld=*/K,
            /*pmat=*/nullptr, // PackBMatrix manages ownership of pmat
            /*groups=*/1),
        col_offsets,
        weight.q_scale().toFloat(),
        weight_zero_point_int32});

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
    AT_ASSERTM(
        false, "This PyTorch installation was not built with FBGEMM operators");
  }
#endif // USE_FBGEMM
};

static auto registry = c10::RegisterOperators().op(
    "quantized::fbgemm_linear_prepack(Tensor W) -> Tensor W_prepack",
    c10::kernel<QFCPackWeightInt8>(),
    c10::dispatchKey(QuantizedCPUTensorId()));
} // namespace
} // namespace native
} // namespace at
