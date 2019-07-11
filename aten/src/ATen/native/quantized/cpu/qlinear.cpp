#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>

#include <algorithm>

namespace at {
namespace native {
namespace {

template <bool ReluFused>
class QLinearInt8 final : public c10::OperatorKernel {
 public:
#ifdef USE_FBGEMM
  at::Tensor operator()(
      at::Tensor input,
      at::Tensor packed_weight,
      at::Tensor bias,
      double output_scale,
      int64_t output_zero_point) {
    // uint8 * int8 -> uint8 (no quantization/dequantization)

    // We make a strong guarantee that models using these operators will have
    // the same numerics across different machines. Therefore, we do not provide
    // a fallback path and rather fail loudly if we cannot run FBGEMM.
    TORCH_CHECK(
        fbgemm::fbgemmSupportedCPU(), "Your CPU does not support FBGEMM.");

    // TODO: contiguous is called for further jit optimizations.
    auto input_contig = input.contiguous();
    const auto* input_ptr =
        reinterpret_cast<uint8_t*>(input_contig.data<c10::quint8>());

    AT_ASSERT(input.dim() >= 2);
    // C(output) = A(input) x B(weight), where C, A, B are M x N, M x K, K x N
    // matrices, respectively.
    int64_t M = 1;
    for (size_t i = 0; i < input.dim() - 1; ++i) {
      M *= input.size(i);
    }

    // Pull out the PackBMatrix and col_offsets instance from the owning tensor.
    auto& pack_ptr =
        cpp_custom_type_hack::cast<PackedLinearWeight>(packed_weight);
    auto packB = pack_ptr.w.get();
    // packB->printPackedMatrix("packedB inside fbgemm_linear (QLinearInt8): ");
    auto& col_offsets = pack_ptr.col_offsets;

    int64_t N = static_cast<int64_t>(packB->numCols());
    int64_t K = input.size(input.dim() - 1);
    AT_ASSERT(K == static_cast<int64_t>(packB->numRows()));
    AT_ASSERT(bias.size(0) == N);
    AT_ASSERT(bias.dim() == 1);

    float input_scale_float = input.q_scale();
    int32_t input_zero_point_int32 = input.q_zero_point();

    float weight_scale_float = pack_ptr.w_scale;
    int32_t weight_zero_point_int32 = pack_ptr.w_zp;

    float output_multiplier_float = (input_scale_float * weight_scale_float) /
        static_cast<float>(output_scale);
    int32_t output_zero_point_int32 = static_cast<int32_t>(output_zero_point);

    // This operation does the following:
    // 1) Creates a "row buffer" vector with offset values that must be added
    //    to the integer matrix multiplication operation to ensure correctness.
    //    This "row buffer" is also called the row offset, and it is needed when
    //    we use affine quantization for weights.
    // 2) Packs the resulting quantized matrix into vector-register and cache
    //    friendly tiles.
    //
    //  Note this is not executed eagerly, but rather within the fbgemmPacked
    //  call below.
    fbgemm::PackAWithRowOffset<uint8_t> packA(
        /*trans=*/fbgemm::matrix_op_t::NoTranspose,
        /*nRow=*/M,
        /*nCol=*/K,
        /*smat=*/input_ptr,
        /*ld=*/K,
        /*pmat=*/nullptr); // Currently, packA manages ownership of `pmat`.
                           // TODO: Consider a way to pre-allocate and reuse
                           // pmat buffer.

    // ReQuantizeOutput requires pointers to the zero point values,
    // since in the case of rowwise quantization these will be arrays rather
    // than scalars. But in this case, we're doing whole-tensor quantization so
    // we just pass a pointer to the scale values (and internally
    // ReQuantizeOutput won't index past 0.

    // This is the end of the pipeline, pass the resulting matrix through.
    fbgemm::DoNothing<> doNothingObj{};

    // TODO: contiguous is called for further jit optimizations.
    auto bias_contig = bias.contiguous();
    const auto* bias_ptr =
        reinterpret_cast<int32_t*>(bias_contig.data<c10::qint32>());

    // After the uint8 * int8 matrix multiplication is performed, this operation
    // does:
    //  1) Add in row and column offsets to the rows and columns, respectively.
    //  2) Add in the bias term.
    fbgemm::ReQuantizeOutput<ReluFused> outputProcObj(
        /*nextop=*/doNothingObj,
        /*C_multiplier=*/&output_multiplier_float,
        /*C_zero_point=*/output_zero_point_int32,
        /*Aq_zero_point=*/input_zero_point_int32,
        /*Bq_zero_point=*/&weight_zero_point_int32,
        /*row_offsets=*/packA.getRowOffsetBuffer(),
        /*col_offsets=*/col_offsets.data(),
        /*bias=*/bias_ptr,
        /*nCol=*/N);

    // Allocate output Tensor and a buffer for fbgemmPacked to use
    auto output = _empty_affine_quantized(
        {M, N},
        at::device(kCPU).dtype(kQUInt8),
        output_scale,
        output_zero_point);

    auto buffer = at::zeros_like(output, output.options().dtype(at::kInt));

    // Do the GEMM
    fbgemm::fbgemmPacked(
        /*packA=*/packA,
        /*packB=*/*packB,
        /*C=*/reinterpret_cast<uint8_t*>(output.data<c10::quint8>()),
        /*C_buffer=*/buffer.data<int32_t>(),
        /*ldc=*/N,
        /*outProcess=*/outputProcObj,
        /*thread_id=*/0,
        /*num_threads=*/1);

    return output;
  }
#else // USE_FBGEMM
  at::Tensor operator()(
      at::Tensor /* input */,
      at::Tensor /* packed_weight */,
      at::Tensor /* bias */,
      double /* output_scale */,
      int64_t /* output_zero_point */) {
    // We make a strong guarantee that models using these operators will have
    // the same numerics across different machines. Therefore, we do not provide
    // a fallback path and rather fail loudly if we cannot run FBGEMM.
    TORCH_CHECK(
        false, "This PyTorch installation was not built with FBGEMM operators");
  }
#endif // USE_FBGEMM
};

static auto registry =
    c10::RegisterOperators()
        .op("quantized::fbgemm_linear(Tensor X, Tensor W_prepack, Tensor b, float Y_scale_i, int Y_zero_point_i) -> Tensor Y",
            c10::RegisterOperators::options()
              .kernel<QLinearInt8<false>>(QuantizedCPUTensorId()))
        .op("quantized::fbgemm_linear_relu(Tensor X, Tensor W_prepack, Tensor b, float Y_scale_i, int Y_zero_point_i) -> Tensor Y",
            c10::RegisterOperators::options()
              .kernel<QLinearInt8<true>>(QuantizedCPUTensorId()));
} // namespace
} // namespace native
} // namespace at
