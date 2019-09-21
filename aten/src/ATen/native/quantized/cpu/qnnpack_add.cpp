#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <caffe2/utils/threadpool/ThreadPoolMobile.h>

#include "init_qnnpack.h"
#include "qnnpack_utils.h"

namespace at {
namespace native {
namespace {

class QNNPACKAdd final : public torch::OperatorKernel {
 public:
#ifdef USE_PYTORCH_QNNPACK
  Tensor operator()(Tensor qa, Tensor qb, double scale, int64_t zero_point) {
    TORCH_CHECK(qa.ndimension() > 0, "qnnpack_add(): Got empty input tensor.");
    TORCH_CHECK(
        qa.numel() == qb.numel(),
        "qnnpack_add(): Add operands must be the same size!");
    TORCH_CHECK(
        qa.scalar_type() == qb.scalar_type(),
        "qnnpack_add(): Add operands should have same data type.");

    Tensor qa_contig = qa.contiguous();
    Tensor qb_contig = qb.contiguous();

    const auto a_zero_point = qa_contig.q_zero_point();
    const auto b_zero_point = qb_contig.q_zero_point();
    const auto a_scale = qa_contig.q_scale();
    const auto b_scale = qb_contig.q_scale();

    Tensor qy = at::_empty_affine_quantized(
        qa_contig.sizes(), at::device(kCPU).dtype(kQUInt8), scale, zero_point);

    if (qa_contig.size(0) == 0) {
      return qy;
    }

    initQNNPACK();

    pytorch_qnnp_operator_t qnnpack_operator{nullptr};

    size_t num_elems = qa_contig.numel() / qa_contig.size(0);

    const pytorch_qnnp_status createStatus = pytorch_qnnp_create_add_nc_q8(
        num_elems /* input size */,
        a_zero_point /* a zero_point */,
        a_scale /* a scale */,
        b_zero_point /* b zero_point */,
        b_scale /* b scale */,
        static_cast<uint8_t>(zero_point) /* sum zero_point */,
        scale /* sum scale */,
        std::numeric_limits<uint8_t>::min() /* output min */,
        std::numeric_limits<uint8_t>::max() /* output max */,
        0 /* flags */,
        &qnnpack_operator);

    TORCH_INTERNAL_ASSERT(
        createStatus == pytorch_qnnp_status_success,
        "failed to create QNNPACK Add operator");
    TORCH_INTERNAL_ASSERT(qnnpack_operator != nullptr);

    std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
        qnnpack_uniq_ptr(qnnpack_operator);

    const pytorch_qnnp_status setupStatus = pytorch_qnnp_setup_add_nc_q8(
        qnnpack_operator /* add op */,
        qa_contig.size(0) /* batch size */,
        (uint8_t*)qa_contig.data_ptr<c10::quint8>() /* a data */,
        num_elems /* A stride */,
        (uint8_t*)qb_contig.data_ptr<c10::quint8>() /* b data */,
        num_elems /* B stride */,
        (uint8_t*)qy.data_ptr<c10::quint8>() /* output data */,
        num_elems /* sum stride */);
    TORCH_INTERNAL_ASSERT(
        setupStatus == pytorch_qnnp_status_success,
        "failed to setup QNNPACK Add operator");

    pthreadpool_t threadpool = caffe2::mobile_threadpool();
    const pytorch_qnnp_status runStatus =
        pytorch_qnnp_run_operator(qnnpack_operator, threadpool);

    TORCH_INTERNAL_ASSERT(
        runStatus == pytorch_qnnp_status_success,
        "failed to run QNNPACK Add operator");

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
    "quantized::qnnpack_add",
    torch::RegisterOperators::options().kernel<QNNPACKAdd>(
        TensorTypeId::QuantizedCPUTensorId));
} // namespace
} // namespace native
} // namespace at
