#include <ATen/native/mkldnn/detail/Softmax.h>

#if AT_MKLDNN_ENABLED()

#include <ATen/Context.h>
#include <c10/util/env.h>

namespace at::native::mkldnn {

void softmax_forward(
    const MkldnnMemory& src,
    MkldnnMemory& dst,
    int softmax_axis,
    dnnl::prop_kind prop_kind) {
  const auto& engine = Engine::cpu_engine();
  auto src_desc = src.get_desc();
  // May replace dst's buffer and layout when they differ from src's.
  dst.reinit_if_possible(src_desc);

  dnnl::primitive_attr op_attr;
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  auto pd = dnnl::softmax_forward::primitive_desc(
      engine,
      prop_kind,
      dnnl::algorithm::softmax_accurate,
      src_desc,
      src_desc,
      softmax_axis,
      op_attr);
  MkldnnMemory scratchpad(pd.scratchpad_desc());
  dnnl::softmax_forward(pd).execute(
      IStream::default_stream(),
      {{DNNL_ARG_SRC, src},
       {DNNL_ARG_DST, dst},
       {DNNL_ARG_SCRATCHPAD, scratchpad}});
}

namespace {

int64_t get_mkldnn_softmax_min_dim_size() {
  static auto value = [] {
    // Minimum last-dimension size for dense oneDNN softmax on AArch64;
    // derived from benchmarks across batch sizes and thread counts.
    constexpr int64_t default_min_dim_size = 8192;
    const auto value = c10::utils::get_env("TORCH_MKLDNN_SOFTMAX_MIN_DIM_SIZE");
    return value.has_value() ? std::stoi(value.value()) : default_min_dim_size;
  }();
  return value;
}

bool use_mkldnn_dense_softmax(
    const Tensor& src,
    const Tensor& dst,
    int softmax_axis) {
#if defined(__aarch64__)
  return at::globalContext().userEnabledMkldnn() &&
      src.is_cpu() && dst.is_cpu() &&
      src.layout() == at::kStrided && dst.layout() == at::kStrided &&
      src.scalar_type() == at::kFloat && dst.scalar_type() == at::kFloat &&
      src.is_contiguous() && dst.is_contiguous() &&
      src.dim() > 0 && softmax_axis == src.dim() - 1 &&
      src.size(softmax_axis) >= get_mkldnn_softmax_min_dim_size() &&
      src.sizes() == dst.sizes();
#else
  return false;
#endif
}

} // namespace

bool dense_softmax_forward(
    const Tensor& src,
    const Tensor& dst,
    int softmax_axis,
    bool log_softmax) {
  if (!use_mkldnn_dense_softmax(src, dst, softmax_axis)) {
    return false;
  }

  try {
    auto src_mem = at::native::itensor_view_from_const_dense(src);
    auto dst_mem = at::native::itensor_view_from_dense(dst);
    auto algorithm = log_softmax ? dnnl::algorithm::softmax_log
                                 : dnnl::algorithm::softmax_accurate;
    const auto& engine = Engine::cpu_engine();
    auto src_desc = src_mem.get_desc();
    auto dst_desc = dst_mem.get_desc();

    dnnl::primitive_attr op_attr;
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd = dnnl::softmax_forward::primitive_desc(
        engine,
        dnnl::prop_kind::forward_inference,
        algorithm,
        src_desc,
        dst_desc,
        softmax_axis,
        op_attr);
    MkldnnMemory scratchpad(pd.scratchpad_desc());
    dnnl::softmax_forward(pd).execute(
        IStream::default_stream(),
        {{DNNL_ARG_SRC, src_mem},
         {DNNL_ARG_DST, dst_mem},
         {DNNL_ARG_SCRATCHPAD, scratchpad}});
    return true;
  } catch (const dnnl::error&) {
    return false;
  }
}

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
