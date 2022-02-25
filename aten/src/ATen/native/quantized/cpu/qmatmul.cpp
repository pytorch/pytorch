#include <ATen/ATen.h>
#include <torch/library.h>

#ifdef USE_RUY_QMATMUL
#include <ruy/ruy.h>
#endif

namespace at {
namespace native {

namespace {

inline void check_inputs(const Tensor& qa, const Tensor& qb) {
  TORCH_CHECK(
      qa.scalar_type() == c10::kQInt8 || qa.scalar_type() == c10::kQUInt8,
      "MatMul operands should use QInt8 or QUInt8 data types.");
  TORCH_CHECK(
      qa.scalar_type() == qb.scalar_type(),
      "MatMul operands should have same data type.");
  TORCH_CHECK(
      qa.qscheme() == kPerTensorAffine || qa.qscheme() == kPerTensorSymmetric,
      "Only per-tensor quantization is suported in Matmul.");
  TORCH_CHECK(
      qa.qscheme() == qb.qscheme(),
      "Both inputs to Matmul must have the same quantization scheme.");
}

#ifdef USE_RUY_QMATMUL

Tensor qmatmul(
    const Tensor& qa,
    const Tensor& qb,
    const double output_scale,
    const int64_t output_zero_point) {
  check_inputs(qa, qb);

  const size_t num_dims = qa.dim();
  const size_t b_num_dims = qb.dim();

  TORCH_CHECK(
      num_dims == b_num_dims,
      "MatMul operands should have the same dimensionality. (", num_dims,
      " and ", b_num_dims, " provided)");
  TORCH_CHECK(
      num_dims == 2,
      "Quantized Matmul currently only suports operands which are 2-dimensional. (",
      num_dims, " provided)");

  const int64_t m = qa.size(0);
  const int64_t k = qa.size(1);
  const int64_t b_k = qb.size(0);
  const int64_t n = qb.size(1);

  TORCH_CHECK(
      b_k == k,
      "For Quantized Matmul, the size of tensor a (", k,
      ") at dimension ", 1, " must match the size of tensor b (",
      b_k, ") at dimension ", 0, ".");

  Tensor out = at::empty(
      {m, n},
      at::device(kCPU).dtype(c10::kInt).memory_format(
          qa.suggest_memory_format()));

  const Tensor& qa_contig = qa.contiguous();
  const Tensor& qb_contig = qb.contiguous();

  ruy::Context context;

  AT_DISPATCH_QINT_BYTE_TYPES(qa.scalar_type(), "qmatmul", [&] {
    using underlying_t = typename scalar_t::underlying;

    const underlying_t* qa_data = reinterpret_cast<const underlying_t*>(
        qa_contig.data_ptr<scalar_t>());
    const underlying_t* qb_data = reinterpret_cast<const underlying_t*>(
        qb_contig.data_ptr<scalar_t>());
    int32_t* out_data = out.data_ptr<int32_t>();

    ruy::Matrix<underlying_t> qa_matrix;
    ruy::MakeSimpleLayout(
        m, k, ruy::Order::kRowMajor, qa_matrix.mutable_layout());
    qa_matrix.set_data(qa_data);
    qa_matrix.set_zero_point(qa.q_zero_point());

    ruy::Matrix<underlying_t> qb_matrix;
    ruy::MakeSimpleLayout(
        k, n, ruy::Order::kRowMajor, qb_matrix.mutable_layout());
    qb_matrix.set_data(qb_data);
    qb_matrix.set_zero_point(qb.q_zero_point());

    ruy::Matrix<int32_t> out_matrix;
    ruy::MakeSimpleLayout(
        m, n, ruy::Order::kRowMajor, out_matrix.mutable_layout());
    out_matrix.set_data(out_data);

    ruy::MulParams<int32_t, int32_t> mul_params;
    ruy::Mul(qa_matrix, qb_matrix, mul_params, &context, &out_matrix);
  });

  // Requantization explanation:
  // https://github.com/google/gemmlowp/blob/e844ffd17118c1e17d94e1ba4354c075a4577b88/doc/quantization.md
  const double requantization_scale =
      output_scale / (qa.q_scale() * qb.q_scale());

  return at::quantize_per_tensor(
      out.toType(c10::kFloat),
      requantization_scale,
      output_zero_point,
      qa.scalar_type());
}

#else // ifdef USE_RUY_QMATMUL

Tensor qmatmul(
    const Tensor& qa,
    const Tensor& qb,
    const double output_scale,
    const int64_t output_zero_point) {
  check_inputs(qa, qb);
  Tensor ra = at::dequantize(qa);
  Tensor rb = at::dequantize(qb);
  Tensor rc = at::matmul(ra, rb);
  return at::quantize_per_tensor(
      rc, output_scale, output_zero_point, qa.scalar_type());
}

#endif // ifdef USE_RUY_QMATMUL

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::matmul"), TORCH_FN(qmatmul));
}

} // namespace

} // namespace native
} // namespace at
