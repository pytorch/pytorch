
#include <c10/xpu/XPUFunctions.h>

#include <ATen/ATen.h>
#include <ATen/record_function.h>

#include <ATen/native/mkldnn/xpu/detail/Attr.h>
#include <ATen/native/mkldnn/xpu/detail/Utils.h>

#include <ATen/native/mkldnn/xpu/detail/MatmulHelpers.h>

#include <c10/core/ScalarType.h>
#include <oneapi/dnnl/dnnl.hpp>

namespace at::native::onednn {

sycl::event matmul(
    at::Tensor& result,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Tensor& b_raw,
    bool m2_trans,
    Attr attr,
    const std::vector<sycl::event>& deps) {
  // m2_trans means mat2 is transposed from the nn.Linear perspective.
  // m2_trans==true means mat2 is [k, n] layout.
  // m2_trans==false means mat2 is [n, k] layout, aka, the default layout in
  // nn.Linear.
  int64_t dims = result.dim();
  TORCH_CHECK(
      dims == 2 || dims == 3,
      "oneDNN matmul only works with 2D or 3D, got ",
      dims);
  TORCH_CHECK(
      dims == mat1.dim() && dims == mat2.dim(),
      "oneDNN input matrixes must have the same ranks");
  TORCH_CHECK(result.defined(), "oneDNN matmul result should be defined");

  auto& engine = GpuEngineManager::Instance().get_engine();
  auto& stream = GpuStreamManager::Instance().get_stream();

  at::Tensor m1 = mat1;
  at::Tensor m2 = mat2;
  undo_broadcast_on_batch(m1, m2);

  auto matmul_stride_handler = [](const at::Tensor& t) -> at::Tensor {
    return is_onednn_matmul_strides(t) ? t : t.contiguous();
  };
  m1 = matmul_stride_handler(m1);
  m2 = matmul_stride_handler(m2);
  at::Tensor dst = matmul_stride_handler(result);

  int64_t m = dst.size(-2);
  int64_t n = dst.size(-1);
  int64_t k = m1.size(-1);
  int64_t mb = dims == 3 ? dst.size(0) : 1;
  if (dims == 3) {
    TORCH_CHECK(
        mb == mat1.size(0) && mb == mat2.size(0),
        "batch size mismatch, dst mb: ",
        mb,
        "m1 mb",
        mat1.size(0),
        " m2 mb: ",
        mat2.size(0));
  }

  // validate bias and make it compatible with oneDNN implementation
  // Step1: handle shape, and create mds
  at::Tensor b = b_raw;
  BiasHandler bias_handler(b, mb, m, k, n, dims);
  bias_handler.handle();
  bool with_bias = bias_handler.is_with_bias();
  b = b.contiguous();
  GEMMMemoryCreator memory_creator(
      dims, m, k, n, mb, m1.size(0), m2.size(0), m2_trans, with_bias);
  memory_creator.initialize(m1, m2, dst, b);
  auto [m1_md, m2_md, dst_md, bias_md] = memory_creator.query_md();

  dnnl::post_ops po = attr.extract_post_ops(dst);
  dnnl::primitive_attr pattr;
  pattr.set_post_ops(po);

  // Step2: primitive attributes
  dnnl::memory::data_type m1_dt = get_onednn_dtype_include_double(m1);
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
  if (m1_dt == dnnl::memory::data_type::f32) {
    bool allow_tf32 = at::globalContext().allowTF32OneDNN();
    if (allow_tf32) {
      pattr.set_fpmath_mode(dnnl::fpmath_mode::tf32);
    } else {
      pattr.set_fpmath_mode(dnnl::fpmath_mode::strict);
    }
  }

#if ONEDNN_SUPPORT_DETERMINISTIC
  if (at::globalContext().deterministicAlgorithms() ||
      at::globalContext().deterministicMkldnn())
    pattr.set_deterministic(true);
#endif

  // Step3: pd, primitive creation
  dnnl::matmul matmul_p;
  dnnl::matmul::primitive_desc matmul_pd;
  if (with_bias) {
    matmul_pd = dnnl::matmul::primitive_desc(
        engine, m1_md, m2_md, bias_md, dst_md, pattr);
  } else {
    matmul_pd =
        dnnl::matmul::primitive_desc(engine, m1_md, m2_md, dst_md, pattr);
  }
  matmul_p = dnnl::matmul(matmul_pd);

  // Step4: create memory args and execution
  std::unordered_map<int, dnnl::memory> args =
      memory_creator.create_memory(m1, m2, dst, b);
  if (attr.with_binary())
    attr.construct_post_binary(matmul_pd, args);

  size_t scratchpad_size = matmul_pd.scratchpad_desc().get_size();
  at::Tensor scratchpad_tensor = at::empty(
      {static_cast<int64_t>(scratchpad_size)},
      m1.options().dtype(at::kByte),
      std::nullopt);
  auto scratchpad_memory = make_onednn_memory(
      matmul_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_memory});

  sycl::event matmul_event =
      dnnl::sycl_interop::execute(matmul_p, stream, args, deps);

  if (!dst.is_same(result))
    result.copy_(dst);

  return matmul_event;
}

} // namespace at::native::onednn
