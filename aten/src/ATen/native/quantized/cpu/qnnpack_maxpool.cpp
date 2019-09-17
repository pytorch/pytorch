#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/Pool.h>
#include <caffe2/utils/threadpool/ThreadPoolMobile.h>

#include "init_qnnpack.h"
#include "qnnpack_utils.h"

namespace at {
namespace native {
namespace {

class QNNPACKMaxPool2D final : public torch::OperatorKernel {
 public:
#ifdef USE_PYTORCH_QNNPACK
  Tensor operator()(
      Tensor input,
      const torch::List<int64_t>& kernel_size,
      const torch::List<int64_t>& stride,
      const torch::List<int64_t>& padding,
      const torch::List<int64_t>& dilation) {
    Tensor qy;

    TORCH_CHECK(
        input.ndimension() == 4,
        "qnnpack_maxpool(): Expected input to be 4-dimensional: got ",
        input.ndimension());
    TORCH_CHECK(
        kernel_size.size() == 2,
        "qnnpack_maxpool(): Expected kernel_size to be 2-dimensional: got ",
        kernel_size.size());
    TORCH_CHECK(
        stride.size() == 2,
        "qnnpack_maxpool(): Expected stride to be 2-dimensional: got ",
        stride.size());
    TORCH_CHECK(
        dilation.size() == 2,
        "qnnpack_maxpool(): Expected dilation to be 2-dimensional: got ",
        dilation.size());
    TORCH_CHECK(
        padding.size() == 2,
        "qnnpack_maxpool(): Expected padding to be 2-dimensional: got ",
        padding.size());

    Tensor input_contig = input.contiguous();

    initQNNPACK();
    const auto scale = input_contig.q_scale();
    const auto zero_point = input_contig.q_zero_point();
    pytorch_qnnp_operator_t qnnpack_operator{nullptr};

    int64_t padH = padding[0];
    int64_t padW = padding[1];
    int64_t kH = kernel_size[0];
    int64_t kW = kernel_size[1];
    int64_t strideH = stride[0];
    int64_t strideW = stride[1];
    int64_t dilationH = dilation[0];
    int64_t dilationW = dilation[1];

    TORCH_CHECK(
        kH > 0 && kW > 0,
        "qnnpack_maxpool(): kernel_size should be greater than zero.");
    TORCH_CHECK(
        strideH > 0 && strideW > 0,
        "qnnpack_maxpool(): strides should be greater than zero.");

    // Input is in NHWC format
    int64_t batch_size = input_contig.size(0);
    int64_t inH = input_contig.size(1);
    int64_t inW = input_contig.size(2);
    int64_t inC = input_contig.size(3);

    const pytorch_qnnp_status createStatus =
        pytorch_qnnp_create_max_pooling2d_nhwc_u8(
            padH /* input_padding_top */,
            padW /* input_padding_right */,
            padH /* input_padding_bottom */,
            padW /* input_padding_left */,
            kH /* pooling height */,
            kW /* pooling width */,
            strideH /* stride height */,
            strideW /* stride width */,
            dilationH /* dilation height */,
            dilationW /* dilation width */,
            inC /* input channels */,
            std::numeric_limits<uint8_t>::min() /* output min */,
            std::numeric_limits<uint8_t>::max() /* output max */,
            0 /* flags */,
            &qnnpack_operator);
    TORCH_INTERNAL_ASSERT(
        createStatus == pytorch_qnnp_status_success,
        "failed to create QNNPACK MaxPool operator");
    TORCH_INTERNAL_ASSERT(qnnpack_operator != nullptr);

    int64_t outC = inC;
    int64_t outH =
        pooling_output_shape(inH, kH, padH, strideH, dilationH, false);
    int64_t outW =
        pooling_output_shape(inW, kW, padW, strideW, dilationW, false);

    TORCH_CHECK(
        outH > 0 && outW > 0,
        "qnnpack_maxpool(): the resulting output Tensor size should be >= 0");

    std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
        qnnpack_uniq_ptr(qnnpack_operator);

    // NHWC output
    qy = at::_empty_affine_quantized(
        {batch_size, outH, outW, outC},
        at::device(kCPU).dtype(kQUInt8),
        scale,
        zero_point);

    const pytorch_qnnp_status setupStatus =
        pytorch_qnnp_setup_max_pooling2d_nhwc_u8(
            qnnpack_operator /* max pooling */,
            batch_size /* batch size */,
            inH /* input height */,
            inW /* input width */,
            (uint8_t*)input_contig.data_ptr<c10::quint8>() /* input */,
            inC /* input_pixel_stride */,
            (uint8_t*)qy.data_ptr<c10::quint8>() /* output data */,
            outC /* output_pixel_stride */,
            nullptr /* thread pool */);
    TORCH_INTERNAL_ASSERT(
        setupStatus == pytorch_qnnp_status_success,
        "failed to setup QNNPACK MaxPool operator");

    pthreadpool_t threadpool = caffe2::mobile_threadpool();
    const pytorch_qnnp_status runStatus =
        pytorch_qnnp_run_operator(qnnpack_operator, threadpool);
    TORCH_INTERNAL_ASSERT(
        runStatus == pytorch_qnnp_status_success,
        "failed to run QNNPACK MaxPool operator");
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
    "quantized::qnnpack_maxpool2d",
    torch::RegisterOperators::options().kernel<QNNPACKMaxPool2D>(
        TensorTypeId::QuantizedCPUTensorId));
} // namespace
} // namespace native
} // namespace at
