#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <caffe2/utils/threadpool/ThreadPoolMobile.h>

#include "init_qnnpack.h"
#include "qnnpack_utils.h"

namespace at {
namespace native {
namespace {

class QNNPACKRelu final : public torch::OperatorKernel {
 public:
#ifdef USE_PYTORCH_QNNPACK
  Tensor operator()(Tensor input) {
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
    TORCH_INTERNAL_ASSERT(qnnpack_operator != nullptr);

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

    pthreadpool_t threadpool = caffe2::mobile_threadpool();

    const pytorch_qnnp_status runStatus =
        pytorch_qnnp_run_operator(qnnpack_operator, threadpool);

    TORCH_INTERNAL_ASSERT(
        runStatus == pytorch_qnnp_status_success,
        "failed to run QNNPACK Relu operator");

    return qy;
  }
#else
  Tensor operator()(Tensor /* input */) {
    TORCH_CHECK(
        false,
        "This PyTorch installation was not built "
        "with QNNPACK operators");
  }
#endif
};

static auto registry = torch::RegisterOperators().op(
    "quantized::qnnpack_relu(Tensor input) -> Tensor",
    torch::RegisterOperators::options().kernel<QNNPACKRelu>(
        TensorTypeId::QuantizedCPUTensorId));
} // namespace
} // namespace native
} // namespace at
