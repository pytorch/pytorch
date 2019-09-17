#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/quantized/Quantizer.h>
#include <caffe2/utils/threadpool/ThreadPoolMobile.h>

#include "init_qnnpack.h"
#include "qnnpack_utils.h"

namespace at {
namespace native {
namespace {

class QNNPACKLinear final : public torch::OperatorKernel {
 public:
#ifdef USE_PYTORCH_QNNPACK
  Tensor operator()(
      at::Tensor input,
      at::Tensor weight,
      at::Tensor bias,
      double output_scale,
      int64_t output_zero_point) {
    TORCH_CHECK(input.dim() >= 2, "Input tensor rank should be >= 2");
    TORCH_CHECK(weight.dim() == 2, "Weight tensor rank should be == 2");

    Tensor input_contig = input.contiguous();

    // Y(output) = X(input_contig) x W(weight)
    int64_t rows_x = 1;
    int64_t cols_x = input_contig.size(input_contig.dim() - 1);
    for (size_t i = 0; i < input_contig.dim() - 1; ++i) {
      rows_x *= input_contig.size(i);
    }

    int64_t rows_y = weight.size(0);

    TORCH_CHECK(
        cols_x == weight.size(1),
        "qnnpack_linear(): input size does not match weight dimension 1 size: got ",
        cols_x,
        " but expected ",
        weight.size(1));

    TORCH_CHECK(
        !bias.defined() || (bias.ndimension() == 1 && bias.size(0) == rows_y),
        "qnnpack_linear(): Given weight of size ",
        weight.sizes(),
        ", expected bias to be 1-dimensional with ",
        rows_y,
        " elements",
        ", but got bias of size ",
        bias.sizes(),
        " instead");

    initQNNPACK();

    // Allocate output Tensor and a buffer for QNNPACK to use
    Tensor output = at::_empty_affine_quantized(
        {rows_x, rows_y}, input.options(), output_scale, output_zero_point);

    pytorch_qnnp_operator_t qnnpack_operator{nullptr};

    // QNNPACK expects both weights and inputs to be uint8
    const pytorch_qnnp_status createStatus =
        pytorch_qnnp_create_fully_connected_nc_q8(
            cols_x /* input channels */,
            rows_y /* output channels */,
            input_contig.q_zero_point() /* input zero_point */,
            input_contig.q_scale() /* input scale */,
            weight.q_zero_point() /* kernel zero_point */,
            weight.q_scale() /* kernel scale */,
            (uint8_t*)weight.data_ptr<c10::quint8>() /* kernel data */,
            (int32_t*)bias.data_ptr<c10::qint32>() /* bias data */,
            output.q_zero_point() /* output zero_point */,
            output.q_scale() /* output scale */,
            std::numeric_limits<uint8_t>::min() /* output_min */,
            std::numeric_limits<uint8_t>::max() /* output_max */,
            0 /* flags */,
            &qnnpack_operator);

    std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
        qnnpack_uniq_ptr(qnnpack_operator);

    TORCH_INTERNAL_ASSERT(
        createStatus == pytorch_qnnp_status_success,
        "failed to create QNNPACK Linear operator");
    TORCH_INTERNAL_ASSERT(qnnpack_operator != nullptr);

    const pytorch_qnnp_status setupStatus =
        pytorch_qnnp_setup_fully_connected_nc_q8(
            qnnpack_operator /* fully_connected */,
            rows_x /* batch_size */,
            (uint8_t*)input_contig.data_ptr<c10::quint8>() /* input */,
            cols_x /* input stride */,
            (uint8_t*)output.data_ptr<c10::quint8>() /* output */,
            rows_y /* output stride */);

    TORCH_INTERNAL_ASSERT(
        setupStatus == pytorch_qnnp_status_success,
        "failed to setup QNNPACK Linear operator");
    pthreadpool_t threadpool = caffe2::mobile_threadpool();

    const pytorch_qnnp_status runStatus =
        pytorch_qnnp_run_operator(qnnpack_operator, threadpool);

    TORCH_INTERNAL_ASSERT(
        runStatus == pytorch_qnnp_status_success,
        "failed to run QNNPACK operator");

    return output;
  }
#else
  Tensor operator()(
      at::Tensor /* input */,
      at::Tensor /* weight */,
      at::Tensor /* bias */,
      double /* output_scale */,
      int64_t /* output_zero_point */) {
    TORCH_CHECK(
        false,
        "This PyTorch installation was not built "
        "with QNNPACK operators");
  }
#endif
};

static auto registry = torch::RegisterOperators().op(
    "quantized::qnnpack_linear(Tensor X, Tensor W, Tensor b, float Y_scale, int Y_zero_point) -> Tensor",
    torch::RegisterOperators::options().kernel<QNNPACKLinear>(
        TensorTypeId::QuantizedCPUTensorId));
} // namespace
} // namespace native
} // namespace at
