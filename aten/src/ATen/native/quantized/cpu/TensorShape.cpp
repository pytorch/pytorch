#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>
#include <ATen/Dispatch.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/IListRef.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorShape.h>
#include <c10/util/irange.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/cat.h>
#include <ATen/ops/cat_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/quantize_per_tensor.h>
#include <ATen/ops/zeros_like_ops.h>
#endif

#include <algorithm>
#include <vector>

namespace at::native {

DEFINE_DISPATCH(qcat_nhwc_stub);
DEFINE_DISPATCH(qcat_relu_nhwc_stub);

namespace {

bool is_cat_nhwc_fast_path(const MaterializedITensorListRef& qxs, int64_t dim) {
  TORCH_CHECK(!qxs.empty());
  bool is_fast_path = dim == 1;
  // NOLINTNEXTLINE(performance-implicit-conversion-in-loop)
  for (const at::Tensor& qx : qxs) {
    is_fast_path &= qx.dim() == 4;
    is_fast_path &= qx.is_contiguous(c10::MemoryFormat::ChannelsLast);
  }
  return is_fast_path;
}

bool is_valid_quantization_scheme(const Tensor& t) {
  const auto qtype = t.qscheme();
  return (qtype == kPerTensorAffine) || (qtype == kPerTensorSymmetric);
}

#define QPARAM_THRESHOLD 1e-04

bool all_inputs_sharing_qparams(const MaterializedITensorListRef& qxs) {
  bool is_valid = true;
  for (const auto i : c10::irange(1, qxs.size())) {
    is_valid &= qxs[0].get().is_quantized();
    is_valid &= qxs[i].get().is_quantized() == qxs[0].get().is_quantized();
    is_valid &= qxs[i].get().qscheme() == qxs[0].get().qscheme();
    is_valid &= qxs[i].get().dtype() == qxs[0].get().dtype();
    if (qxs[0].get().qscheme() == kPerTensorAffine) {
        is_valid &= fabs(qxs[i].get().q_scale() - qxs[0].get().q_scale()) < QPARAM_THRESHOLD;
      is_valid &= qxs[i].get().q_zero_point() == qxs[0].get().q_zero_point();
    } else if (qxs[0].get().qscheme() == kPerChannelAffine) {
        is_valid &= qxs[i].get().q_per_channel_scales().isclose(qxs[0].get().q_per_channel_scales(), 0, QPARAM_THRESHOLD, false).all().item().to<bool>();
      is_valid &= qxs[i].get().q_per_channel_zero_points().equal(qxs[0].get().q_per_channel_zero_points());
    } else {
        TORCH_CHECK(false, "Unrecognized qscheme:", toString(qxs[0].get().qscheme()));
    }
  }
  return is_valid;
}

/* Quantized concatenation.
 *
 * Note: This function uses a dequantization.
 */
template <bool ReLUFused>
Tensor quantized_cat_impl(
    const MaterializedITensorListRef& qxs,
    int64_t dim,
    double scale,
    int64_t zero_point) {
  if (is_cat_nhwc_fast_path(qxs, dim)) {
    if (ReLUFused) {
      return qcat_relu_nhwc_stub(at::kCPU, qxs, dim, scale, zero_point);
    } else {
      return qcat_nhwc_stub(at::kCPU, qxs, dim, scale, zero_point);
    }
  }

  const auto x_dtype = qxs[0].get().scalar_type();
  const auto x_qscheme = qxs[0].get().qscheme();
  std::vector<Tensor> xs;
  xs.reserve(qxs.size());
  // NOLINTNEXTLINE(performance-implicit-conversion-in-loop)
  for (const at::Tensor& qx : qxs) {
    TORCH_CHECK(x_dtype == qx.scalar_type(), "All dtypes must be the same.");
    TORCH_CHECK(
        x_qscheme == qx.qscheme(), "Quantization schemes must be the same.");
    xs.push_back(qx.dequantize());
  }
  const Tensor y = at::cat(xs, dim);
  Tensor qy;
  AT_DISPATCH_QINT_TYPES(x_dtype, "qcat", [&]() {
    // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
    qy = at::quantize_per_tensor(y, scale, zero_point, SCALAR_TYPE);
    if (ReLUFused) {
      auto iter = TensorIterator::unary_op(qy, qy);
      cpu_kernel(iter, [&](scalar_t value) -> scalar_t {
        return scalar_t(std::max<underlying_t>(value.val_, zero_point));
      });
    }
  });
  return qy;
}

template <bool ReLUFused>
Tensor quantized_cat_impl(
    ITensorListRef qxs,
    int64_t dim,
    double scale,
    int64_t zero_point) {
  return quantized_cat_impl<ReLUFused>(qxs.materialize(), dim, scale, zero_point);
}

template <bool ReLUFused = false>
Tensor qcat(
    const c10::List<Tensor>& qxs,
    int64_t dim,
    std::optional<double> scale,
    std::optional<int64_t> zero_point) {
  TORCH_CHECK(is_valid_quantization_scheme(qxs[0]),
              "Only per-tensor quantization is supported in 'cat'!")
  double _scale = scale.has_value() ? scale.value() : qxs.get(0).q_scale();
  int64_t _zero_point =
      zero_point.has_value() ? zero_point.value() : qxs.get(0).q_zero_point();
  return quantized_cat_impl<ReLUFused>(qxs, dim, _scale, _zero_point);
}

template <bool ReLUFused = false>
Tensor qcat_out(const c10::List<Tensor>& qxs, int64_t dim, Tensor out) {
  TORCH_CHECK(is_valid_quantization_scheme(qxs[0]),
              "Only per-tensor quantization is supported in 'cat'!")
  TORCH_CHECK(is_valid_quantization_scheme(out),
              "Only per-tensor quantization is supported in 'cat'!")
  auto out_ =
      quantized_cat_impl<ReLUFused>(qxs, dim, out.q_scale(), out.q_zero_point());
  at::native::copy_(out, out_, /*non_blocking=*/false);
  return out;
}

} // namespace

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::cat"), TORCH_FN(qcat<false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::cat_relu"), TORCH_FN(qcat<true>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::cat_out"), TORCH_FN(qcat_out<false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::cat_relu_out"), TORCH_FN(qcat_out<true>));
}

Tensor cat_quantized_cpu(const ITensorListRef& qxs, int64_t dim) {
  auto materialized = qxs.materialize();
  TORCH_CHECK(is_valid_quantization_scheme(materialized[0]),
              "Only per-tensor quantization is supported in 'cat'!");

  if (!all_inputs_sharing_qparams(materialized)) {
      // TODO: if possible change this warning to an error T194501002
      TORCH_WARN("All inputs of this cat operator must share the same quantization parameters. Otherwise large numerical inaccuracies may occur.");
  }
  check_cat_no_zero_dim(materialized);
  dim = legacy_cat_wrap_dim(dim, materialized);
  double _scale = materialized[0].get().q_scale();
  int64_t _zero_point = materialized[0].get().q_zero_point();
  return quantized_cat_impl<false>(materialized, dim, _scale, _zero_point);
}

Tensor& cat_out_quantized_cpu(const ITensorListRef& qxs, int64_t dim, Tensor& out) {
  auto materialized = qxs.materialize();
  TORCH_CHECK(is_valid_quantization_scheme(materialized[0]),
              "Only per-tensor quantization is supported in 'cat'!")
  TORCH_CHECK(is_valid_quantization_scheme(out),
              "Only per-tensor quantization is supported in 'cat'!")
  check_cat_no_zero_dim(materialized);
  dim = legacy_cat_wrap_dim(dim, materialized);
  auto out_ = quantized_cat_impl<false>(qxs, dim, out.q_scale(), out.q_zero_point());
  at::native::copy_(out, out_, /*non_blocking=*/false);
  return out;
}

}  // namespace at::native
