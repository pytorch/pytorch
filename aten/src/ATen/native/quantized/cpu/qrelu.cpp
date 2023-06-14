#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>
#include <c10/util/irange.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/_prelu_kernel_native.h>
#include <ATen/ops/hardtanh_native.h>
#include <ATen/ops/leaky_relu_native.h>
#include <ATen/ops/prelu.h>
#include <ATen/ops/prelu_native.h>
#include <ATen/ops/quantize_per_tensor.h>
#include <ATen/ops/relu_native.h>
#endif

#include <algorithm>

namespace at {
namespace native {

DEFINE_DISPATCH(qrelu_stub);
DEFINE_DISPATCH(qrelu_leaky_stub);
DEFINE_DISPATCH(qprelu_stub);

#ifdef USE_PYTORCH_QNNPACK
static Tensor qnnpack_relu(Tensor input) {
  Tensor qy;
  TORCH_CHECK(
      input.ndimension() > 0, "qnnpack_relu(): Got empty input tensor");
  TORCH_CHECK(input.scalar_type() == c10::kQUInt8,
               "qnnpack_relu(): Expected input data type ",
               toString(c10::kQUInt8),
               " but got ",
               toString(input.scalar_type()));

  Tensor input_contig = input.contiguous(input.suggest_memory_format());

  const auto zero_point = input_contig.q_zero_point();

  initQNNPACK();

  size_t num_elems = 1;
  for (const auto i : c10::irange(1, input_contig.ndimension())) {
    num_elems *= input_contig.size(i);
  }

  pytorch_qnnp_operator_t qnnpack_operator{nullptr};

  const pytorch_qnnp_status createStatus = pytorch_qnnp_create_clamp_nc_u8(
      num_elems /* channels */,
      zero_point /* output min */,
      std::numeric_limits<uint8_t>::max() /* output max */,
      0 /* flags */,
      &qnnpack_operator);

  std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
      qnnpack_uniq_ptr(qnnpack_operator);

  TORCH_INTERNAL_ASSERT(
      createStatus == pytorch_qnnp_status_success,
      "failed to create QNNPACK Relu operator");

  qy = at::_empty_affine_quantized(
      input_contig.sizes(),
      at::device(kCPU).dtype(input.scalar_type()),
      input_contig.q_scale(),
      input_contig.q_zero_point(),
      input.suggest_memory_format());

  const pytorch_qnnp_status setupStatus = pytorch_qnnp_setup_clamp_nc_u8(
      qnnpack_operator, /* clamp */
      input_contig.size(0) /* batch size */,
      (uint8_t*)input_contig.data_ptr<c10::quint8>() /* input data */,
      num_elems /* input stride */,
      (uint8_t*)qy.data_ptr<c10::quint8>() /* output data */,
      num_elems /* output stride */);
  TORCH_INTERNAL_ASSERT(
      setupStatus == pytorch_qnnp_status_success,
      "failed to setup QNNPACK Relu operator");

  pthreadpool_t threadpool = caffe2::pthreadpool_();

  const pytorch_qnnp_status runStatus =
      pytorch_qnnp_run_operator(qnnpack_operator, threadpool);

  TORCH_INTERNAL_ASSERT(
      runStatus == pytorch_qnnp_status_success,
      "failed to run QNNPACK Relu operator");
  return qy;
}
#endif

Tensor relu_quantized_cpu(const Tensor& qx) {
  #ifdef USE_PYTORCH_QNNPACK
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK && qx.scalar_type() == kQUInt8) {
    return qnnpack_relu(qx);
  }
  #endif
  Tensor qy;
  qrelu_stub(qx.device().type(), qx, qy);
  return qy;
}
Tensor& relu_quantized_cpu_(Tensor& qx) {
  const auto zero_point = qx.q_zero_point();
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qrelu", [&]() {
    using Vec = Vectorized<scalar_t>;
    auto iter = TensorIterator::unary_op(qx, qx);
    auto zero_point_vec = Vec(scalar_t(zero_point));
    cpu_kernel_vec(
        iter,
        [&](scalar_t value) -> scalar_t {
          return scalar_t(std::max<underlying_t>(value.val_, zero_point));
        },
        [&](Vec value) -> Vec { return value.relu(zero_point_vec); });
  });
  return qx;
}

Tensor& leaky_relu_out_quantized_cpu(const Tensor& self,
                                 const Scalar& negval, Tensor& result) {
  qrelu_leaky_stub(self.device().type(), result, self, negval);
  return result;
}

Tensor leaky_relu_quantized_cpu(const Tensor& self, const Scalar& negval) {
  const auto qx = self.contiguous(self.suggest_memory_format());
  auto qy = at::_empty_affine_quantized(qx.sizes(),
      at::device(kCPU).dtype(self.scalar_type()),
      qx.q_scale(),
      qx.q_zero_point(),
      self.suggest_memory_format());
  qrelu_leaky_stub(self.device().type(), qy, qx, negval);
  return qy;
}

Tensor& leaky_relu_quantized_cpu_(Tensor& self, const Scalar& negval) {
  qrelu_leaky_stub(self.device().type(), self, self, negval);
  return self;
}

static Tensor _prelu_kernel_quantized_cpu_impl(const Tensor& self, const Tensor& weight,
                                double output_scale, int64_t output_zero_point) {
  auto ndim = self.dim();
  // for ndim < 1 or > 5, go to reference path
  if (ndim > 5 || ndim < 1) {
    auto x = self.dequantize();
    auto y = at::prelu(x, weight);
    return at::quantize_per_tensor(y, output_scale, output_zero_point, c10::kQUInt8);
  }

  auto qy = at::_empty_affine_quantized(self.sizes(),
      at::device(kCPU)
        .dtype(self.scalar_type()),
      output_scale,
      output_zero_point,
      self.suggest_memory_format());

  qprelu_stub(self.device().type(), qy, self, weight);

  return qy;
}

Tensor _prelu_kernel_quantized_cpu(const Tensor& self, const Tensor& weight) {
  return _prelu_kernel_quantized_cpu_impl(self, weight, self.q_scale(), self.q_zero_point());
}

namespace {
Tensor quantized_relu6(const Tensor& qx) {
  Tensor qy;
  qy = hardtanh_quantized_cpu(qx, 0.0f, 6.0f);
  return qy;
}

Tensor quantized_relu6_(Tensor& qx) {
  hardtanh_quantized_cpu_(qx, 0.0f, 6.0f);
  return qx;
}

class QRelu6 final {
 public:
  static Tensor run(Tensor qx, bool inplace) {
    if (inplace) {
      return quantized_relu6_(qx);
    } else {
      return quantized_relu6(qx);
    }
  }
};

class QLeakyRelu final {
 public:
  static Tensor run(Tensor self, const Scalar& negative_slope, bool inplace, double output_scale, int64_t output_zero_point) {
    // inplace argument is ignored now, TODO:support inplace
    if (inplace) {
      TORCH_WARN("inplace=True is not supported for quantized::leaky_relu yet");
    }
    const auto qx = self.contiguous(self.suggest_memory_format());
    auto qy = at::_empty_affine_quantized(qx.sizes(),
      at::device(kCPU).dtype(self.scalar_type()),
      output_scale,
      output_zero_point,
      self.suggest_memory_format());
    qrelu_leaky_stub(self.device().type(), qy, qx, negative_slope);
    return qy;
  }
};

class QPRelu final {
 public:
  static Tensor run(Tensor self, const Tensor& weight, double output_scale, int64_t output_zero_point) {
  return _prelu_kernel_quantized_cpu_impl(self, weight, output_scale, output_zero_point);
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::relu6"), TORCH_FN(QRelu6::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::leaky_relu"), TORCH_FN(QLeakyRelu::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::prelu"), TORCH_FN(QPRelu::run));
}

} // namespace

}}  // namespace at::native
