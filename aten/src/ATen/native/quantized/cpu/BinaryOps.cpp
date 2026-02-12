#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <torch/library.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/quantized/cpu/ACLUtils.h>
#include <ATen/native/quantized/cpu/BinaryOps.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/cpu/XnnpackUtils.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/_empty_affine_quantized_native.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/relu_native.h>
#endif

#include <algorithm>
#include <utility>

namespace at::native {

DEFINE_DISPATCH(qadd_relu_stub);
DEFINE_DISPATCH(qadd_stub);
DEFINE_DISPATCH(qadd_scalar_relu_stub);
DEFINE_DISPATCH(qadd_scalar_stub);

namespace {

inline void check_inputs(const Tensor& qa, const Tensor& qb) {
  TORCH_CHECK(
      qa.qscheme() == kPerTensorAffine,
      "Only per tensor quantization is supported in Add.");
  TORCH_CHECK(
      qa.qscheme() == qb.qscheme(),
      "Both inputs to Add must have the same quantization scheme.");
  TORCH_CHECK(
      qa.scalar_type() == qb.scalar_type(),
      "Add operands should have same data type.");
}

// Note: out is assumed to be the same size as self and other.
// Note: Addition is only supported when self, other, out are of the same dtype.
template <bool ReLUFused = false>
Tensor _add_out(Tensor& out, const Tensor& self, const Tensor& other) {
  if (ReLUFused) {
    qadd_relu_stub(self.device().type(), out, self, other);
  } else {
    qadd_stub(self.device().type(), out, self, other);
  }
  return out;
}

template <bool ReLUFused = false>
Tensor _add_scalar_out(Tensor& out, const Tensor& self, const Scalar& other) {
  TORCH_CHECK(
      self.qscheme() == kPerTensorAffine,
      "Only per tensor affine is supported for now!!");
  // To implement tensor-scalar addition in quantized space, we simply
  // adjust the quantization parameters based on the following rules:
  //
  // Let s = scale, z = zero point, c = other.toFloat(), c_q = round(c/s)
  // q_min = lowest representable value of scalar type
  // q_max = highest representable value of scalar type
  //
  // Let s' = the calculated scale or the output
  // z' = the calculated zero-point for the output
  //
  // If q_min > z - c_q
  //   s' = [(q_max - (z - c_q)]/[q_max - q_min] * s
  //   z' = q_min
  //   Xq' = at::requantize_from_int(Xq - z + c_q, s/s', z')
  // If q_max < z - c_q
  //   s' = [z - c_q -q_min]/[q_max - q_min] * s
  //   z' = q_max
  //   Xq' = at::requantize_from_int(Xq - z + c_q, s/s', z')
  // Else
  //   s' = s
  //   z' = z - c_q

  AT_DISPATCH_QINT_TYPES(self.scalar_type(), "qadd_scalar", [&]() {
    double s = self.q_scale();
    int64_t z = self.q_zero_point();
    double c = other.toDouble();
    // NOLINTNEXTLINE(bugprone-signed-char-misuse)
    int64_t q_min = std::numeric_limits<underlying_t>::min();
    int64_t q_max = std::numeric_limits<underlying_t>::max();

    int64_t c_q = std::nearbyint(c / s);

    double s_prime;
    int64_t z_prime;

    if (q_min > z - c_q) {
      s_prime = (((double)q_max - (z - c_q))) / ((double)q_max - q_min) * s;
      z_prime = q_min;
      set_quantizer_(out, make_per_tensor_affine_quantizer(
          s_prime, z_prime, self.scalar_type()));
      if (ReLUFused) {
        qadd_scalar_relu_stub(self.device().type(), out, self, c_q);
      } else {
        qadd_scalar_stub(self.device().type(), out, self, c_q);
      }
    } else if (q_max < z - c_q) {
      s_prime = ((double)(z - c_q) - q_min) / ((double)q_max - q_min) * s;
      z_prime = q_max;
      set_quantizer_(out, make_per_tensor_affine_quantizer(
          s_prime, z_prime, self.scalar_type()));
      if (ReLUFused) {
        qadd_scalar_relu_stub(self.device().type(), out, self, c_q);
      } else {
        qadd_scalar_stub(self.device().type(), out, self, c_q);
      }
    } else {
      s_prime = s;
      z_prime = z - c_q;
      out.copy_(self);
      set_quantizer_(out, make_per_tensor_affine_quantizer(
          s_prime, z_prime, self.scalar_type()));
      if (ReLUFused) {
        at::native::relu_quantized_cpu_(out);
      }
    }
  });
  return out;
}


#ifdef USE_PYTORCH_QNNPACK
template <bool ReLUFused = false>
Tensor qnnpack_add(Tensor qa, Tensor qb, double scale, int64_t zero_point) {
  TORCH_CHECK(qa.ndimension() > 0, "qnnpack_add(): Got empty input tensor.");
  TORCH_CHECK(qa.scalar_type() == c10::kQUInt8 && qb.scalar_type() == c10::kQUInt8,
                "qnnpack_add(): Expected both input data types to be ",
                toString(c10::kQUInt8),
                " but got ",
                toString(qa.scalar_type()),
                " and ",
                toString(qb.scalar_type()));
  Tensor qa_contig = qa.contiguous(qa.suggest_memory_format());
  // Reason for use qa's memory format for qb is that for the underlying
  // kernel can flatten all the dims and iterate over both the tensors.
  // In most cases, both qa and qb are in same memory format.
  // When they are not there is a copy overhead to make it contiguous
  // in qa's memory format.
  Tensor qb_contig = qb.contiguous(qa.suggest_memory_format());

  const auto a_zero_point = qa_contig.q_zero_point();
  const auto b_zero_point = qb_contig.q_zero_point();
  const auto a_scale = qa_contig.q_scale();
  const auto b_scale = qb_contig.q_scale();

  Tensor qy = at::native::empty_affine_quantized(
      qa_contig.sizes(),
      kQUInt8,
      std::nullopt /* layout */,
      kCPU,
      std::nullopt /* pin_memory */,
      scale,
      zero_point,
      qa.suggest_memory_format());

  if (qa_contig.size(0) == 0) {
    return qy;
  }

  initQNNPACK();

  pytorch_qnnp_operator_t qnnpack_operator{nullptr};

  size_t num_elems = qa_contig.numel() / qa_contig.size(0);
  auto output_min = ReLUFused
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      ? activationLimits<uint8_t>(scale, zero_point, Activation::RELU)
            .first
      : std::numeric_limits<uint8_t>::min();
  auto output_max = ReLUFused
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      ? activationLimits<uint8_t>(scale, zero_point, Activation::RELU)
            .second
      : std::numeric_limits<uint8_t>::max();
  const pytorch_qnnp_status createStatus = pytorch_qnnp_create_add_nc_q8(
      num_elems /* input size */,
      a_zero_point /* a zero_point */,
      a_scale /* a scale */,
      b_zero_point /* b zero_point */,
      b_scale /* b scale */,
      static_cast<uint8_t>(zero_point) /* sum zero_point */,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      scale /* sum scale */,
      output_min /* output min */,
      output_max /* output max */,
      0 /* flags */,
      &qnnpack_operator);

  TORCH_INTERNAL_ASSERT(
      createStatus == pytorch_qnnp_status_success,
      "failed to create QNNPACK Add operator");

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

  pthreadpool_t threadpool = caffe2::pthreadpool_();
  const pytorch_qnnp_status runStatus =
      pytorch_qnnp_run_operator(qnnpack_operator, threadpool);

  TORCH_INTERNAL_ASSERT(
      runStatus == pytorch_qnnp_status_success,
      "failed to run QNNPACK Add operator");

  return qy;
}
#endif // USE_PYTORCH_QNNPACK

#ifdef USE_XNNPACK
C10_ALWAYS_INLINE
enum xnn_status xnnp_define_q_tensor(const Tensor& tensor, MemoryFormat format, uint32_t& id, xnn_subgraph_t subgraph_ptr, uint32_t external_id, uint32_t flags){
  Tensor contig_tensor = tensor.contiguous(format);
  const auto tensor_shape = xnnp_utils::get_mem_format_aware_shape(contig_tensor);
  const int32_t zero_point = static_cast<int32_t>(contig_tensor.q_zero_point());
  const float scale = static_cast<float>(contig_tensor.q_scale());

  return xnn_define_quantized_tensor_value(
    subgraph_ptr,
    xnn_datatype_qint8,
    zero_point,
    scale,
    tensor.ndimension(),
    tensor_shape.data(),
    nullptr,
    external_id,
    flags,
    &id);
}

template <typename scalar_t, bool ReLUFused = false>
Tensor xnnp_add(Tensor qa, Tensor qb, double scale, int64_t zero_point) {
  TORCH_CHECK(qa.ndimension() > 0, __func__, ": Got empty input tensor.");
  TORCH_CHECK(at::native::xnnpack::available(), __func__, ": XNNPACK is not available")

  // using qa memory format for qb to allow xnnpack kernel to flatten all the
  // dims
  auto qa_mem_format = qa.suggest_memory_format();
  Tensor qa_contig = qa.contiguous(qa_mem_format);
  Tensor qb_contig = qb.contiguous(qa_mem_format);
  Tensor qy = at::native::empty_affine_quantized(
      at::infer_size_dimvector(qa_contig.sizes(), qb_contig.sizes()),
      qa.scalar_type(),
      std::nullopt /* layout */,
      kCPU,
      std::nullopt /* pin_memory */,
      scale,
      zero_point,
      qa_mem_format);

  if (qa_contig.size(0) == 0) {
    return qy;
  }


  auto output_max = std::numeric_limits<float>::infinity();
  auto output_min = -std::numeric_limits<float>::infinity();
  if (ReLUFused) {
    output_min = 0;
  }

  // Create XNNPACK Subgraph
  xnn_subgraph_t subgraph_ptr = nullptr;
  auto status = xnn_create_subgraph(
    /*external_value_ids=*/3,
    /*flags=*/0,
    &subgraph_ptr);
  TORCH_CHECK(
      status == xnn_status_success,
      __func__, ": xnn create subgraph failed(", status,")!");
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> subgraph(
      subgraph_ptr, &xnn_delete_subgraph);

  uint32_t input0_id = XNN_INVALID_VALUE_ID, input1_id = XNN_INVALID_VALUE_ID, output_id = XNN_INVALID_VALUE_ID;

  // Defining the quantized input 0
  status = xnnp_define_q_tensor(
    qa,
    qa_mem_format,
    input0_id,
    subgraph_ptr,
    0,
    XNN_VALUE_FLAG_EXTERNAL_INPUT
  );
  TORCH_CHECK(
      status == xnn_status_success && input0_id != XNN_INVALID_VALUE_ID,
      __func__, ": xnn define input 0 failed(", status,")!");

  // Defining the quantized input 1
  status = xnnp_define_q_tensor(
    qb,
    qa_mem_format,
    input1_id,
    subgraph_ptr,
    1,
    XNN_VALUE_FLAG_EXTERNAL_INPUT
  );
  TORCH_CHECK(
      status == xnn_status_success && input1_id != XNN_INVALID_VALUE_ID,
      __func__, ": xnn define input 1 failed(", status,")!");

  // Defining the quantized output
  status = xnnp_define_q_tensor(
    qy,
    qa_mem_format,
    output_id,
    subgraph_ptr,
    2,
    XNN_VALUE_FLAG_EXTERNAL_OUTPUT
  );
  TORCH_CHECK(
      status == xnn_status_success && output_id != XNN_INVALID_VALUE_ID,
      __func__, ": xnn define output failed(", status,")!");

  const struct xnn_binary_params binary_params = {output_min, output_max};
  status = xnn_define_binary(
    subgraph_ptr,
    xnn_binary_add,
    &binary_params,
    input0_id,
    input1_id,
    output_id,
    0);
  TORCH_CHECK(
      status == xnn_status_success,
      __func__, ": xnn define binary add failed(", status,")!");

  // create runtime
  xnn_runtime_t runtime_ptr = nullptr;
  status = xnn_create_runtime_v2(subgraph_ptr, caffe2::pthreadpool_(), 0, &runtime_ptr);
  TORCH_CHECK(
      status == xnn_status_success,
      __func__, ": xnn create runtime failed(", status,")!");
  TORCH_CHECK(
      runtime_ptr != nullptr,
      __func__, ": xnn create runtime failed because runtime_ptr is null");
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime_ptr, &xnn_delete_runtime);

  std::array<xnn_external_value, 3> external = {
    xnn_external_value{input0_id, reinterpret_cast<void*>(qa_contig.data_ptr<scalar_t>())},
    xnn_external_value{input1_id, reinterpret_cast<void*>(qb_contig.data_ptr<scalar_t>())},
    xnn_external_value{output_id, reinterpret_cast<void*>(qy.data_ptr<scalar_t>())}};

  status = xnn_setup_runtime(
    runtime_ptr,
    external.size(),
    external.data());
  TORCH_CHECK(
      status == xnn_status_success,
      __func__, ": xnn setup runtime failed(", status,")!");
  status = xnn_invoke_runtime(runtime_ptr);
  TORCH_CHECK(
      status == xnn_status_success,
      __func__, ": xnn invoke runtime failed(", status,")!");

  return qy;
}
#endif // USE_XNNPACK

#if AT_MKLDNN_ACL_ENABLED()
Tensor acl_qadd(Tensor qa, Tensor qb, double scale, int64_t zero_point) {
  TORCH_CHECK(
      qa.qscheme() == kPerTensorAffine || qa.qscheme() == kPerTensorSymmetric,
      "Only per tensor quantization is supported in ACL quantized add.");

  Tensor qa_contig = qa.contiguous(qa.suggest_memory_format());
  Tensor qb_contig = qb.contiguous(qa.suggest_memory_format());
  auto qa_mem_format = qa_contig.suggest_memory_format();
  Tensor dst = at::native::empty_affine_quantized(
      at::infer_size_dimvector(qa_contig.sizes(), qb_contig.sizes()),
      qa_contig.scalar_type(),
      std::nullopt /* layout */,
      kCPU,
      std::nullopt /* pin_memory */,
      scale,
      zero_point,
      qa_mem_format);

  if (qb_contig.size(0) == 0) {
    return dst;
  }

  auto input_dims = qa_contig.sizes().vec();
  auto acl_dtype = dst.scalar_type() == kQInt8
      ? arm_compute::DataType::QASYMM8_SIGNED
      : arm_compute::DataType::QASYMM8;
  auto acl_add = std::make_shared<acl_utils::QuantAdd>(
      acl_dtype,
      input_dims,
      qa_contig.q_scale(),
      qa_contig.q_zero_point(),
      qb_contig.q_scale(),
      qb_contig.q_zero_point(),
      dst.q_scale(),
      dst.q_zero_point());

  auto status = acl_add->validate();
  TORCH_CHECK(
      status.error_code() == arm_compute::ErrorCode::OK,
      "Arm Compute Library's Quantized Matmul Validation Failed: " +
          status.error_description());

  acl_add->configure();

  acl_add->qa_tensor.allocator()->import_memory(qa_contig.data_ptr());
  acl_add->qb_tensor.allocator()->import_memory(qb_contig.data_ptr());
  acl_add->qdst_tensor.allocator()->import_memory(dst.data_ptr());

  acl_add->q_add.run();

  // this will not free memory, it will just tell ACL that we're no longer
  // using the pointer
  acl_add->qa_tensor.allocator()->free();
  acl_add->qb_tensor.allocator()->free();
  acl_add->qdst_tensor.allocator()->free();

  return dst;
}
#endif // AT_MKLDNN_ACL_ENABLED()

template <bool ReLUFused = false>
Tensor qadd(Tensor qa, Tensor qb, double scale, int64_t zero_point) {
  check_inputs(qa, qb);

  if (at::globalContext().qEngine() == at::QEngine::QNNPACK) {
    TORCH_CHECK(
        qa.scalar_type() == qb.scalar_type(),
        "Both inputs to qadd must have same type");

#ifdef USE_XNNPACK
    if (qa.scalar_type() == kQInt8) {
          return xnnp_add<c10::qint8, ReLUFused>(qa, qb, scale, zero_point);
    }
#endif // USE_XNNPACK

#ifdef USE_PYTORCH_QNNPACK
    if(qa.sizes() == qb.sizes() && /* qnnpack does not support broadcasting */
      qa.scalar_type() == kQUInt8) {
    return qnnpack_add<ReLUFused>(qa, qb, scale, zero_point);
    }
#endif // USE_PYTORCH_QNNPACK
  }

#if AT_MKLDNN_ACL_ENABLED()
  if (!ReLUFused && qa.ndimension() > 0 && qa.sizes() == qb.sizes() &&
      qa.scalar_type() == qb.scalar_type() &&
      (qa.scalar_type() == kQInt8 || qa.scalar_type() == kQUInt8)) {
    return acl_qadd(qa, qb, scale, zero_point);
  }
#endif // AT_MKLDNN_ACL_ENABLED()

  auto qc = at::_empty_affine_quantized(
      qa.sizes(),
      at::device(kCPU)
         .dtype(qa.scalar_type())
         .memory_format(qa.suggest_memory_format()),
      scale,
      zero_point,
      std::nullopt);
  return _add_out<ReLUFused>(qc, qa, qb);
}

template <bool ReLUFused = false>
Tensor qadd_out(Tensor qa, Tensor qb, Tensor out) {
  check_inputs(qa, qb);
  check_inputs(qa, out);
  return _add_out<ReLUFused>(out, qa, qb);
}


template <bool ReLUFused = false>
Tensor qadd_scalar(Tensor qa, const Scalar& b) {
  TORCH_CHECK(qa.qscheme() == kPerTensorAffine ||
              qa.qscheme() == kPerTensorSymmetric,
              "Only per tensor quantization is supported in Add.");
  auto qc = at::empty_like(qa, qa.suggest_memory_format());
  return _add_scalar_out<ReLUFused>(qc, qa, b);
}

template <bool ReLUFused = false>
Tensor qadd_scalar2(Scalar b, Tensor qa) {
  TORCH_CHECK(qa.qscheme() == kPerTensorAffine ||
              qa.qscheme() == kPerTensorSymmetric,
              "Only per tensor quantization is supported in Add.");
  auto qc = at::empty_like(qa, qa.suggest_memory_format());
  return _add_scalar_out<ReLUFused>(qc, qa, b);
}

template <bool ReLUFused = false>
Tensor qadd_scalar_out(Tensor qa, const Scalar& b, Tensor out) {
  check_inputs(qa, out);
  return _add_scalar_out<ReLUFused>(out, qa, b);
}

// `torch.jit.trace` will trace Scalar as Tensor
// This can be removed after broadcast is supported and
// all variations of `quantized::add` is merged into `quantized::add`
template <bool ReLUFused = false>
Tensor qadd_scalar_tensor(Tensor qa, Tensor b) {
  return qadd_scalar(std::move(qa), b.item());
}

// `torch.jit.trace` will trace Scalar as Tensor
// This can be removed after broadcast is supported and
// all variations of `quantized::add` is merged into `quantized::add`
template <bool ReLUFused = false>
Tensor qadd_scalar_tensor_out(Tensor qa, Tensor b, Tensor out) {
  return qadd_scalar_out(std::move(qa), b.item(), std::move(out));
}

DEFINE_DISPATCH(qadd_tensor_cpu_stub);
DEFINE_DISPATCH(qadd_relu_tensor_cpu_stub);
template <bool ReLUFused = false>
Tensor int8_add_tensor_onednn(
    const Tensor& self, double self_scale, int64_t self_zero_point,
    const Tensor& other, double other_scale, int64_t other_zero_point,
    double output_scale, int64_t output_zero_point, c10::ScalarType output_dtype) {
  // Both inputs should have the same shape and both in uint8 dtype.
  // If output_dtype is uint8, output is requantized with output scale/zero point.
  // Otherwise, output scale should be 1 and zero point 0.
  TORCH_CHECK(self.sizes() == other.sizes(),
              "Quantized add operands should have the same size.");
  TORCH_CHECK(self.scalar_type() == at::kByte && other.scalar_type() == at::kByte,
              "Quantized add operands should be of type uint8, but got ",
              self.scalar_type(), " and ", other.scalar_type());
  TORCH_CHECK(output_dtype == at::kByte || output_dtype == at::kFloat || output_dtype == at::kBFloat16 || output_dtype == at::kHalf,
              "Quantized add output should be of type uint8, float, bfloat16 or float16, but got ",
              output_dtype);
  if (output_dtype != at::kByte) {
    TORCH_CHECK(output_scale == 1.0 && output_zero_point == 0,
                "Quantized add output scale and zero point should be 1 and 0 for "
                "output_dtype ", output_dtype, ", but got scale = ",
                output_scale, " and zero point = ", output_zero_point);
  }
  at::Tensor out = at::empty_like(self, self.options().dtype(output_dtype));


  if constexpr (ReLUFused) {
    qadd_relu_tensor_cpu_stub(
        self.device().type(), out, self, self_scale, self_zero_point,
        other, other_scale, other_zero_point,
        output_scale, output_zero_point);
  } else {
    qadd_tensor_cpu_stub(
        self.device().type(), out, self, self_scale, self_zero_point,
        other, other_scale, other_zero_point,
        output_scale, output_zero_point);
  }

  return out;
}

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::add"),                 TORCH_FN(qadd</*ReLUFused=*/false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add.out"),             TORCH_FN(qadd_out</*ReLUFused=*/false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add.Scalar"),          TORCH_FN(qadd_scalar</*ReLUFused=*/false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add.Scalar2"),          TORCH_FN(qadd_scalar2</*ReLUFused=*/false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add.Scalar_out"),      TORCH_FN(qadd_scalar_out</*ReLUFused=*/false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_relu"),            TORCH_FN(qadd</*ReLUFused=*/true>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_relu.out"),        TORCH_FN(qadd_out</*ReLUFused=*/true>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_relu.Scalar"),     TORCH_FN(qadd_scalar</*ReLUFused=*/true>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_relu.Scalar2"),     TORCH_FN(qadd_scalar2</*ReLUFused=*/true>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_relu.Scalar_out"), TORCH_FN(qadd_scalar_out</*ReLUFused=*/true>));
  // deprecated functions, kept for backward compatibility
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_out"),             TORCH_FN(qadd_out</*ReLUFused=*/false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_relu_out"),        TORCH_FN(qadd_out</*ReLUFused=*/true>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_scalar"),          TORCH_FN(qadd_scalar</*ReLUFused=*/false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_scalar_relu"),     TORCH_FN(qadd_scalar</*ReLUFused=*/true>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_scalar_out"),      TORCH_FN(qadd_scalar_out</*ReLUFused=*/false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_scalar_relu_out"), TORCH_FN(qadd_scalar_out</*ReLUFused=*/true>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_scalar.Tensor"),   TORCH_FN(qadd_scalar_tensor</*ReLUFused=*/false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_scalar_relu.Tensor"), TORCH_FN(qadd_scalar_tensor</*ReLUFused=*/true>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_scalar_out.Tensor"), TORCH_FN(qadd_scalar_tensor_out</*ReLUFused=*/false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_scalar_relu_out.Tensor"), TORCH_FN(qadd_scalar_tensor_out</*ReLUFused=*/true>));
}

TORCH_LIBRARY_IMPL(_quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("_quantized::add"), TORCH_FN(qadd</*ReLUFused=*/false>));
}

TORCH_LIBRARY_IMPL(onednn, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("onednn::qadd.tensor"), TORCH_FN(int8_add_tensor_onednn<false>));
  m.impl(TORCH_SELECTIVE_NAME("onednn::qadd_relu.tensor"), TORCH_FN(int8_add_tensor_onednn<true>));
}

}  // namespace

Tensor quantized_add(Tensor qa, Tensor qb, double scale, int64_t zero_point){
  return qadd<false>(std::move(qa), std::move(qb), scale, zero_point);
}

}  // namespace at::native
