#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/native/Activation.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/gelu_native.h>
#include <ATen/ops/gelu_backward_native.h>
#endif

#if !AT_MKLDNN_ENABLED()

namespace at::native {

Tensor mkldnn_gelu(const Tensor& input, std::string_view approximate) {
  TORCH_CHECK(false, "mkldnn_gelu: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_gelu_backward(const Tensor& grad_output, const Tensor& input, std::string_view approximate) {
  TORCH_CHECK(false, "mkldnn_gelu_backward: ATen not compiled with MKLDNN support");
}

}

#else // AT_MKLDNN_ENABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>
#include <oneapi/dnnl/dnnl.hpp>

namespace at::native {

Tensor mkldnn_gelu(const Tensor& input, std::string_view approximate) {
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_gelu: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }
  TORCH_CHECK(get_gelutype_enum(approximate) == GeluType::None,
                  "mkldnn_gelu: fast, approximate gelu is not supported");

  auto src_desc = get_mkldnn_memory_desc(input);
  auto& engine = get_mkldnn_cpu_engine();

  auto src_mem = mkldnn_memory_from_tensor(input, src_desc, engine);

  auto op_attr = dnnl::primitive_attr();
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  auto pd = dnnl::eltwise_forward::primitive_desc(
      engine,
      dnnl::prop_kind::forward_training,
      dnnl::algorithm::eltwise_gelu_erf,
      src_desc,
      src_desc,
      /*alpha*/ 0.0f,
      /*beta*/ 0.0f,
      op_attr);

  // Create output memory
  dnnl::memory dst_mem(pd.dst_desc(), engine);
  dnnl::memory scratchpad(pd.scratchpad_desc(), engine);

  dnnl::eltwise_forward(pd).execute(
      get_mkldnn_default_stream(),
      {{DNNL_ARG_SRC, src_mem},
       {DNNL_ARG_DST, dst_mem},
       {DNNL_ARG_SCRATCHPAD, scratchpad}});

  // Convert dnnl::memory to ideep::tensor for output wrapper
  ideep::tensor y;
  y.init(pd.dst_desc(), dst_mem.get_data_handle());

  return new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(input.options().dtype_opt()),
                                 input.options().device_opt());
}

Tensor mkldnn_gelu_backward(const Tensor& grad_output, const Tensor& input, std::string_view approximate) {
  TORCH_CHECK(get_gelutype_enum(approximate) == GeluType::None,
                  "mkldnn_gelu_backward: fast, approximate gelu is not supported");

  auto src_desc = get_mkldnn_memory_desc(input);
  auto& engine = get_mkldnn_cpu_engine();

  auto src_mem = mkldnn_memory_from_tensor(input, src_desc, engine);

  auto forward_hints = dnnl::eltwise_forward::primitive_desc(
      engine,
      dnnl::prop_kind::forward,
      dnnl::algorithm::eltwise_gelu_erf,
      src_desc,
      src_desc,
      /*alpha*/ 0.0f,
      /*beta*/ 0.0f);

  auto op_attr = dnnl::primitive_attr();
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  auto pd = dnnl::eltwise_backward::primitive_desc(
      engine,
      dnnl::algorithm::eltwise_gelu_erf,
      forward_hints.src_desc(),
      forward_hints.dst_desc(),
      src_desc,
      /*alpha*/ 0.0f,
      /*beta*/ 0.0f,
      forward_hints,
      op_attr);

  auto diff_dst_desc = pd.diff_dst_desc();
  auto grady_mem = mkldnn_memory_from_tensor(grad_output, get_mkldnn_memory_desc(grad_output), engine);

  // Reorder if needed
  dnnl::memory diff_dst_mem = grady_mem;
  if (diff_dst_desc != grady_mem.get_desc()) {
    diff_dst_mem = dnnl::memory(diff_dst_desc, engine);
    dnnl::reorder(grady_mem, diff_dst_mem)
        .execute(get_mkldnn_default_stream(), grady_mem, diff_dst_mem);
  }

  // Create output gradient memory
  dnnl::memory diff_src_mem(pd.diff_src_desc(), engine);

  // Reorder src if needed
  dnnl::memory expected_src_mem = src_mem;
  if (pd.src_desc() != src_mem.get_desc()) {
    expected_src_mem = dnnl::memory(pd.src_desc(), engine);
    dnnl::reorder(src_mem, expected_src_mem)
        .execute(get_mkldnn_default_stream(), src_mem, expected_src_mem);
  }

  dnnl::memory scratchpad(pd.scratchpad_desc(), engine);

  dnnl::eltwise_backward(pd).execute(
      get_mkldnn_default_stream(),
      {{DNNL_ARG_DIFF_DST, diff_dst_mem},
       {DNNL_ARG_SRC, expected_src_mem},
       {DNNL_ARG_DIFF_SRC, diff_src_mem},
       {DNNL_ARG_SCRATCHPAD, scratchpad}});

  // Convert dnnl::memory to ideep::tensor for output wrapper
  ideep::tensor gradx;
  gradx.init(pd.diff_src_desc(), diff_src_mem.get_data_handle());

  return new_with_itensor_mkldnn(std::move(gradx),
                                 optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                 grad_output.options().device_opt());
}

}

#endif // AT_MKLDNN_ENABLED
