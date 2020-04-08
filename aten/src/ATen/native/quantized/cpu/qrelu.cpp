#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <caffe2/utils/threadpool/ThreadPoolMobile.h>

#include <algorithm>

namespace at {
namespace native {

DEFINE_DISPATCH(qrelu_stub);
DEFINE_DISPATCH(qrelu6_stub);
DEFINE_DISPATCH(qrelu_leaky_stub);

#ifdef USE_PYTORCH_QNNPACK
Tensor qnnpack_relu(Tensor input) {
  Tensor qy;
  TORCH_CHECK(
      input.ndimension() > 0, "qnnpack_relu(): Got empty input tensor");

  Tensor input_contig = input.contiguous();

  const auto zero_point = input_contig.q_zero_point();

  initQNNPACK();

  size_t volume = input_contig.numel();

  size_t num_elems_x = 1;
  for (int i = 1; i < input_contig.ndimension(); ++i) {
    num_elems_x *= input_contig.size(i);
  }

  pytorch_qnnp_operator_t qnnpack_operator{nullptr};

  const pytorch_qnnp_status createStatus = pytorch_qnnp_create_clamp_nc_u8(
      num_elems_x /* channels */,
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
      input.options(),
      input_contig.q_scale(),
      input_contig.q_zero_point());

  size_t num_elems_y = volume / qy.size(0);

  const pytorch_qnnp_status setupStatus = pytorch_qnnp_setup_clamp_nc_u8(
      qnnpack_operator, /* clamp */
      input_contig.size(0) /* batch size */,
      (uint8_t*)input_contig.data_ptr<c10::quint8>() /* input data */,
      num_elems_x /* input stride */,
      (uint8_t*)qy.data_ptr<c10::quint8>() /* output data */,
      num_elems_y /* output stride */);
  TORCH_INTERNAL_ASSERT(
      setupStatus == pytorch_qnnp_status_success,
      "failed to setup QNNPACK Relu operator");

  pthreadpool_t threadpool = caffe2::mobile_pthreadpool();

  const pytorch_qnnp_status runStatus =
      pytorch_qnnp_run_operator(qnnpack_operator, threadpool);

  TORCH_INTERNAL_ASSERT(
      runStatus == pytorch_qnnp_status_success,
      "failed to run QNNPACK Relu operator");
  return qy;
}
#endif

Tensor quantized_relu(const Tensor& qx) {
  #ifdef USE_PYTORCH_QNNPACK
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK && qx.scalar_type() == kQUInt8) {
    return qnnpack_relu(qx);
  }
  #endif
  Tensor qy;
  qrelu_stub(qx.device().type(), qx, qy);
  return qy;
}
Tensor& quantized_relu_(Tensor& qx) {
  const auto zero_point = qx.q_zero_point();
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qrelu", [&]() {
    using Vec = Vec256<scalar_t>;
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

Tensor& quantized_leaky_relu_out(Tensor& result, const Tensor& self,
                                 Scalar negval) {
  qrelu_leaky_stub(self.device().type(), result, self, negval);
  return result;
}

Tensor quantized_leaky_relu(const Tensor& self, Scalar negval) {
  const auto qx = self.contiguous();
  auto qy = at::_empty_affine_quantized(qx.sizes(), self.options(),
                                        qx.q_scale(), qx.q_zero_point());
  qrelu_leaky_stub(self.device().type(), qy, qx, negval);
  return qy;
}

Tensor& quantized_leaky_relu_(Tensor& self, Scalar negval) {
  qrelu_leaky_stub(self.device().type(), self, self, negval);
  return self;
}

namespace {
Tensor quantized_relu6(const Tensor& qx) {
  Tensor qy;
  qrelu6_stub(qx.device().type(), qx, qy);
  return qy;
}

Tensor quantized_relu6_(Tensor& qx) {
  const auto zero_point = qx.q_zero_point();
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qrelu6_", [&]() {
    using Vec = Vec256<scalar_t>;
    auto iter = TensorIterator::unary_op(qx, qx);
    auto zero_point_vec = Vec(scalar_t(zero_point));
    scalar_t six = at::quantize_val<scalar_t>(qx.q_scale(), qx.q_zero_point(),
                                              /*value=*/6.0);
    auto six_vec = Vec(six);
    cpu_kernel_vec(
        iter,
        [&](scalar_t value) -> scalar_t {
          underlying_t relu_val = std::max<underlying_t>(value.val_,
                                                         zero_point);
          return scalar_t(std::min<underlying_t>(relu_val, six.val_));
        },
        [&](Vec value) -> Vec { return value.relu6(zero_point_vec, six_vec); });
  });
  return qx;
}

class QRelu6 final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor qx, bool inplace) {
    if (inplace) {
      return quantized_relu6_(qx);
    } else {
      return quantized_relu6(qx);
    }
  }
};

static auto registry = c10::RegisterOperators()
.op("quantized::relu6(Tensor qx, bool inplace=False) -> Tensor",
    c10::RegisterOperators::options().kernel<QRelu6>(DispatchKey::QuantizedCPU));
} // namespace

}}  // namespace at::native
