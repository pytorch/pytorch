#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <torch/library.h>

#include <c10/util/irange.h>

#include <algorithm>
#include <string>

#ifdef USE_FBGEMM
at::Tensor PackedLinearWeight::apply_fused_skip_requant(
  at::Tensor input,
  double input_scale,
  int64_t input_zero_point) {
  TORCH_CHECK(!input.is_quantized(), "Input tensor for apply_fused_skip_requant is quantized; "
  "Expected input tensor in PackedLinearWeight::apply_fused_skip_requant to be full precision.");

  return apply_fused_skip_requant_impl<false>(input, input_scale, input_zero_point);
}

at::Tensor PackedLinearWeight::apply_fused_skip_requant_relu(
  at::Tensor input,
  double input_scale,
  int64_t input_zero_point) {
  TORCH_CHECK(!input.is_quantized(), "Input tensor for apply_fused_skip_requant is quantized; "
  "Expected input tensor in PackedLinearWeight::apply_fused_skip_requant to be full precision.");

  return apply_fused_skip_requant_impl<true>(input, input_scale, input_zero_point);
}

template <bool ReluFused>
at::Tensor PackedLinearWeight::apply_fused_skip_requant_impl(
    const at::Tensor& input,
    double input_scale,
    int64_t input_zero_point) {
  auto start = std::chrono::steady_clock::now();

  TORCH_CHECK(
      fbgemm::fbgemmSupportedCPU(), "Your CPU does not support FBGEMM.");

  auto input_contig = input.expect_contiguous();
  const auto* input_ptr = input_contig->data_ptr<float>();

  TORCH_CHECK(
      input.dim() >= 2,
      "The dimension of input tensor should be larger than or equal to 2");
  int64_t M = size_to_dim_(input.dim() - 1, input.sizes());

  auto packB = w.get();

  int64_t N = static_cast<int64_t>(packB->numCols());
  int64_t K = input.sizes()[input.dim() - 1];
  TORCH_CHECK(
      K == static_cast<int64_t>(packB->numRows()),
      "The number of rows in the packB should be equal to K: " +
          std::to_string(K));

  TORCH_CHECK(
      w_scale.size() == w_zp.size(),
      "Weight scales and zero points vectors should have the same size.");

  const float* bias_ptr = nullptr;
  c10::MaybeOwned<at::Tensor> bias_contig;
  if (this->bias_.has_value()) {
    auto& bias = this->bias_.value();
    bias_contig = bias.expect_contiguous();
    TORCH_CHECK(bias_contig->dim() == 1, "bias should be a vector (1D Tensor)");
    TORCH_CHECK(
        bias_contig->sizes()[0] == N, "bias should have N elements: " + std::to_string(N));
    bias_ptr = bias_contig->data_ptr<float>();
  }

  std::vector<int64_t> out_sizes = input.sizes().vec();
  out_sizes.back() = N;
  // Allocate output Tensor and a buffer for fbgemmPacked to use
  auto output = at::empty(out_sizes, input.options().dtype(at::kFloat));
  auto buffer = at::empty_like(
      output,
      output.options().dtype(at::kInt),
      LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  int num_tasks = at::get_num_threads();
  at::parallel_for(0, num_tasks, 1, [&](int64_t begin, int64_t end) {
    fbgemm::PackAWithQuantRowOffset<uint8_t> packA(
        /*trans=*/fbgemm::matrix_op_t::NoTranspose,
        /*nRow=*/M,
        /*nCol=*/K,
        /*smat=*/input_ptr,
        /*ld=*/K,
        /*pmat=*/nullptr,
        /*scale=*/input_scale,
        /*zero_pt=*/input_zero_point);

    fbgemm::DoNothing<float, float> doNothingObj{};
    for (const auto task_id : c10::irange(begin, end)) {
      if (q_scheme == c10::kPerTensorAffine) {
        fbgemm::ReQuantizeForFloat<ReluFused> outputProcObj(
                doNothingObj,
                input_scale,
                w_scale.data(),
                input_zero_point,
                w_zp.data(),
                packA.getRowOffsetBuffer(),
                col_offsets.data(),
                bias_ptr,
                N);

        fbgemm::fbgemmPacked(
            /*packA=*/packA,
            /*packB=*/*packB,
            /*C=*/output.data_ptr<float>(),
            /*C_buffer=*/buffer.data_ptr<int32_t>(),
            /*ldc=*/N,
            /*outProcess=*/outputProcObj,
            /*thread_id=*/task_id,
            /*num_threads=*/num_tasks);
      } else if (q_scheme == c10::kPerChannelAffine) {
  //       // Process the per channel quantization.
  //       //
  //       // After the uint8 * int8 matrix multiplication is performed, this
  //       // operation does:
  //       //  1) Add in row and column offsets to the rows and columns,
  //       //  respectively.
  //       //  2) Add in the bias term.
  //       fbgemm::ReQuantizeOutput<
  //           ReluFused,
  //           fbgemm::QuantizationGranularity::OUT_CHANNEL,
  //           float>
  //           outputProcObj(
  //               doNothingObj,
  //               output_multiplier_float.data(),
  //               output_zero_point,
  //               input_zero_point,
  //               w_zp.data(),
  //               packA.getRowOffsetBuffer(),
  //               col_offsets.data(),
  //               bias_ptr,
  //               // NOLINTNEXTLINE(bugprone-argument-comment)
  //               N, /*nCol=*/
  //               1, /* groups*/
  //               act_times_w_scale.data());

  //       // Do the GEMM
  //       fbgemm::fbgemmPacked(
  //           /*packA=*/packA,
  //           /*packB=*/*packB,
  //           /*C=*/reinterpret_cast<uint8_t*>(output.data_ptr<c10::quint8>()),
  //           /*C_buffer=*/buffer.data_ptr<int32_t>(),
  //           /*ldc=*/N,
  //           /*outProcess=*/outputProcObj,
  //           /*thread_id=*/task_id,
  //           /*num_threads=*/num_tasks);
      }
    }
  });
  static int64_t iter = 0;
  static double elapsed_time = 0.0;
  auto end = std::chrono::steady_clock::now();
  if (iter >= 100) {
    elapsed_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }
  if (iter == 10000 + 99) {
    std::cout << "elapsed_time: " << elapsed_time / 1000000.0 << "ms" <<std::endl;
  }
  ++iter;
  return output;
}

#endif // USE_FBGEMM

namespace at {
namespace native {
namespace {

template <bool ReluFused>
class QLinearFusedSkipRequantInt8 final {
 public:
  static at::Tensor run(
      at::Tensor input,
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight,
      double input_scale,
      int64_t input_zero_point) {
    if (ReluFused) {
      return packed_weight->apply_fused_skip_requant_relu(std::move(input), input_scale, input_zero_point);
    } else {
      return packed_weight->apply_fused_skip_requant(std::move(input), input_scale, input_zero_point);
    }
  }
};

TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::linear_fused_skip_requant"),
      TORCH_FN(QLinearFusedSkipRequantInt8<false>::run));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::linear_fused_skip_requant_relu"),
      TORCH_FN(QLinearFusedSkipRequantInt8<true>::run));
}

} // anonymous
} // native
} // at
