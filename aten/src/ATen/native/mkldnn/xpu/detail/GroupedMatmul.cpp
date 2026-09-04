#include <ATen/Tensor.h>
#include <ATen/core/Tensor.h>
#include <c10/core/ScalarType.h>

#include <ATen/native/mkldnn/xpu/detail/Attr.h>
#include <ATen/native/mkldnn/xpu/detail/Utils.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNNContext.h>

#include <oneapi/dnnl/dnnl.hpp>

namespace at::native::onednn {

// Identifies which matmul operand a GroupedScaleSpec is being built for.
enum class GroupedGemmArgType { Src, Wei };

// Describes how to configure oneDNN scales for a given role/ScalingType.
// When use_mask_only is true, apply via set_scales_mask(arg, mask).
// Otherwise, apply via set_scales(arg, mask, groups, dtype).
struct GroupedScaleSpec {
  int mask;
  dnnl::memory::dims groups;
  dnnl::memory::data_type dtype;
  bool use_mask_only;

  void apply(dnnl::primitive_attr& attr, int arg) const {
    if (use_mask_only) {
      attr.set_scales_mask(arg, mask);
    } else {
      attr.set_scales(arg, mask, groups, dtype);
    }
  }
};

inline GroupedScaleSpec make_grouped_scale_spec(
    at::blas::ScalingType scaling_type,
    GroupedGemmArgType arg_type,
    bool a_is_2d,
    bool b_is_2d) {
  TORCH_CHECK(
      a_is_2d && !b_is_2d,
      "Currently only 2d x 3d grouped matmul with offsets is supported");
  bool is_src = (arg_type == GroupedGemmArgType::Src);

  switch (scaling_type) {
    case at::blas::ScalingType::RowWise:
      // For FP8 rowwise with grouped matmul, use set_scales_mask (no groups)
      // following the official grouped matmul example:
      //   src: mask=(1<<0) -> per-token scale, shape [M]
      //   wei: mask=(1<<0)|(1<<2) -> per-group-per-N scale, shape [G, N]
      return {
          is_src ? (1 << 0) : (1 << 0) | (1 << 2),
          {},
          dnnl::memory::data_type::f32,
          /*use_mask_only=*/true};
    case at::blas::ScalingType::BlockWise1x32:
      return {
          is_src ? (1 << 0) | (1 << 1) : (1 << 0) | (1 << 1) | (1 << 2),
          is_src ? dnnl::memory::dims{1, 32} : dnnl::memory::dims{32, 1},
          dnnl::memory::data_type::e8m0,
          /*use_mask_only=*/false};
    case at::blas::ScalingType::BlockWise1x16:
      return {
          is_src ? (1 << 0) | (1 << 1) : (1 << 0) | (1 << 1) | (1 << 2),
          is_src ? dnnl::memory::dims{1, 16} : dnnl::memory::dims{16, 1},
          dnnl::memory::data_type::f8_e4m3,
          /*use_mask_only=*/false};
    default:
      TORCH_CHECK(
          false, "Unknown scaling_type: ", static_cast<int>(scaling_type));
  }
}

// Prepare input tensors for grouped matmul: ensure proper contiguity and
// 64-byte alignment. 2D x 3D (inference): mat_a needs K-contiguous, mat_b needs
// K-contiguous or N-contiguous 2D x 2D (training grad_B): mat_a needs col-major
// (dim0 stride=1), mat_b needs row-major (N-contiguous)
std::tuple<Tensor, Tensor, Tensor> prepare_grouped_matmul_inputs(
    const Tensor& mat_a,
    const Tensor& mat_b,
    Tensor& out,
    bool a_is_2d,
    bool b_is_2d) {
  Tensor prepared_mat_a, prepared_mat_b, prepared_out;
  if (a_is_2d && !b_is_2d) {
    // 2D x 3D: A[M,K] needs K-contiguous (row-major), B[E,K,N] should be
    // K-contiguous or N-contiguous
    prepared_mat_a = make_contiguous_and_aligned(mat_a);
    prepared_mat_b = !is_64_bytes_aligned(mat_b) ||
            !(mat_b.is_contiguous() || mat_b.transpose(-2, -1).is_contiguous())
        ? make_contiguous_and_aligned(mat_b.transpose(-2, -1)).transpose(-2, -1)
        : mat_b;
  } else if (a_is_2d && b_is_2d) {
    // 2D x 2D: A[M,K] needs col-major (M-major, dim0 stride=1), B[K,N] needs
    // row-major (N-contiguous)
    prepared_mat_a =
        make_contiguous_and_aligned(mat_a.transpose(-2, -1)).transpose(-2, -1);
    prepared_mat_b = make_contiguous_and_aligned(mat_b);
  } else {
    TORCH_CHECK(false, "Unsupported grouped matmul input dimensions");
  }
  TORCH_CHECK(
      out.is_contiguous() && is_64_bytes_aligned(out),
      "Output tensor must be contiguous and 64-byte aligned for oneDNN");
  prepared_out = out;
  return {prepared_mat_a, prepared_mat_b, prepared_out};
}

std::tuple<dnnl::memory::desc, dnnl::memory::desc, dnnl::memory::desc>
get_grouped_gemm_md(
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t group_count,
    dnnl::memory::data_type dtype,
    dnnl::memory::data_type dst_dtype,
    const Tensor& mat_b,
    const Tensor& out,
    bool a_is_2d,
    bool b_is_2d) {
  dnnl::memory::desc src_md, weights_md, dst_md;
  if (a_is_2d && !b_is_2d) {
    src_md = dnnl::memory::desc::grouped({M, K}, dtype, 0, group_count);
    weights_md = get_onednn_md(mat_b);
    dst_md = dnnl::memory::desc::grouped({M, N}, dst_dtype, 0, group_count);
  } else if (a_is_2d && b_is_2d) {
    src_md = dnnl::memory::desc::grouped({M, K}, dtype, 1, group_count);
    weights_md = dnnl::memory::desc::grouped({K, N}, dtype, 0, group_count);
    dst_md = get_onednn_md(out);
  } else {
    TORCH_CHECK(
        false,
        "Unsupported grouped matmul dimensions: mat_a.dim() = ",
        a_is_2d ? 2 : 3,
        ", mat_b.dim() = ",
        b_is_2d ? 2 : 3);
  }
  return {src_md, weights_md, dst_md};
}

std::tuple<dnnl::memory, dnnl::memory, dnnl::memory> make_grouped_gemm_mem(
    const dnnl::memory::desc& src_md,
    const dnnl::memory::desc& weights_md,
    const dnnl::memory::desc& dst_md,
    const Tensor& src,
    const Tensor& weights,
    const Tensor& dst,
    const Tensor& offs,
    dnnl::engine& engine,
    bool a_is_2d,
    bool b_is_2d) {
  dnnl::memory src_mem, weights_mem, dst_mem;
  if (a_is_2d && !b_is_2d) {
    src_mem = make_onednn_grouped_memory(
        src_md, engine, src.data_ptr(), offs.data_ptr());
    weights_mem = make_onednn_memory(weights_md, engine, weights.data_ptr());
    dst_mem = make_onednn_grouped_memory(
        dst_md, engine, dst.data_ptr(), offs.data_ptr());
  } else if (a_is_2d && b_is_2d) {
    src_mem = make_onednn_grouped_memory(
        src_md, engine, src.data_ptr(), offs.data_ptr());
    weights_mem = make_onednn_grouped_memory(
        weights_md, engine, weights.data_ptr(), offs.data_ptr());
    dst_mem = make_onednn_memory(dst_md, engine, dst.data_ptr());
  } else {
    TORCH_CHECK(
        false,
        "Unsupported grouped matmul dimensions: mat_a.dim() = ",
        src.dim(),
        ", mat_b.dim() = ",
        weights.dim());
  }
  return {src_mem, weights_mem, dst_mem};
}

sycl::event scaled_grouped_matmul(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const std::optional<Tensor> scale_a,
    const std::optional<Tensor> scale_b,
    const std::optional<at::blas::ScalingType> scaling_choice_a,
    const std::optional<at::blas::ScalingType> scaling_choice_b,
    const Tensor& offs,
    Tensor& out,
    const std::optional<Tensor> alpha) {
  bool a_is_2d = mat_a.dim() == 2;
  bool b_is_2d = mat_b.dim() == 2;

  TORCH_CHECK(
      (a_is_2d && !b_is_2d),
      "Currently only 2d x 3d grouped matmul with offsets is supported");

  bool is_fp4 = mat_a.scalar_type() == at::kFloat4_e2m1fn_x2;
  int64_t M, N, K, group_count;
  M = mat_a.size(-2);
  K = is_fp4 ? mat_a.size(-1) * 2 : mat_a.size(-1);
  N = mat_b.size(-1);
  group_count = offs.size(0);

  auto [prepared_mat_a, prepared_mat_b, prepared_out] =
      prepare_grouped_matmul_inputs(mat_a, mat_b, out, a_is_2d, b_is_2d);

  auto& engine = GpuEngineManager::Instance().get_engine();
  auto& stream = GpuStreamManager::Instance().get_stream();

  // 1.1 Create memory descriptor
  dnnl::memory::data_type dtype = get_onednn_dtype(mat_a);
  dnnl::memory::data_type dst_dtype = get_onednn_dtype(out);
  auto [src_md, weights_md, dst_md] = get_grouped_gemm_md(
      M,
      N,
      K,
      group_count,
      dtype,
      dst_dtype,
      prepared_mat_b,
      prepared_out,
      a_is_2d,
      b_is_2d);

  // 1.2 Create the matmul primitive descriptor
  dnnl::primitive_attr op_attr = dnnl::primitive_attr();

  if (scaling_choice_a.has_value() && scaling_choice_b.has_value()) {
    const GroupedScaleSpec src_spec = make_grouped_scale_spec(
        scaling_choice_a.value(), GroupedGemmArgType::Src, a_is_2d, b_is_2d);
    const GroupedScaleSpec wei_spec = make_grouped_scale_spec(
        scaling_choice_b.value(), GroupedGemmArgType::Wei, a_is_2d, b_is_2d);
    src_spec.apply(op_attr, DNNL_ARG_SRC);
    wei_spec.apply(op_attr, DNNL_ARG_WEIGHTS);
  }

  dnnl::memory::desc alpha_md;
  if (alpha.has_value()) {
    // set alpha in post-op attr, this is used for NVFP4 case where we need to
    // combine two global fp32 scales into a single alpha value.
    // For grouped_matmul, alpha is per-group, so it should have one scale
    // per group.
    TORCH_CHECK(
        alpha.value().dim() == 1 && alpha.value().size(0) == group_count,
        "Expected alpha to be a 1D tensor of size ",
        group_count,
        ", but got shape: ",
        alpha.value().sizes());
    alpha_md = get_onednn_md(alpha.value().view({group_count, 1}));
    dnnl::post_ops post_ops;
    post_ops.append_binary(dnnl::algorithm::binary_mul, alpha_md);
    op_attr.set_post_ops(post_ops);
  }

#if ONEDNN_SUPPORT_DETERMINISTIC
  if (at::globalContext().deterministicAlgorithms() ||
      at::globalContext().deterministicMkldnn())
    op_attr.set_deterministic(true);
#endif

  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  // 1.3 Create the matmul primitive descriptor
  dnnl::matmul::primitive_desc matmul_pd =
      dnnl::matmul::primitive_desc(engine, src_md, weights_md, dst_md, op_attr);

  // 2. Prepare memory
  auto [src_mem, weights_mem, dst_mem] = make_grouped_gemm_mem(
      src_md,
      weights_md,
      dst_md,
      prepared_mat_a,
      prepared_mat_b,
      prepared_out,
      offs,
      engine,
      a_is_2d,
      b_is_2d);
  size_t scratchpad_size = matmul_pd.scratchpad_desc().get_size();
  at::Tensor scratchpad_tensor = at::empty(
      {static_cast<int64_t>(scratchpad_size)},
      mat_a.options().dtype(at::kByte),
      std::nullopt);
  dnnl::memory scratchpad = make_onednn_memory(
      matmul_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());

  // 3. Setup Args for exec
  std::unordered_map<int, dnnl::memory> args;
  args.insert({DNNL_ARG_SRC, src_mem});
  args.insert({DNNL_ARG_WEIGHTS, weights_mem});
  args.insert({DNNL_ARG_DST, dst_mem});
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});

  if (scaling_choice_a.has_value() && scaling_choice_b.has_value()) {
    at::Tensor src_scale = make_contiguous_and_aligned(scale_a.value());
    at::Tensor wei_scale = make_contiguous_and_aligned(scale_b.value());
    dnnl::memory::desc src_scale_md = get_onednn_md(src_scale);
    dnnl::memory::desc wei_scale_md = get_onednn_md(wei_scale);
    auto src_scale_mem =
        make_onednn_memory(src_scale_md, engine, src_scale.data_ptr());
    auto wei_scale_mem =
        make_onednn_memory(wei_scale_md, engine, wei_scale.data_ptr());

    args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scale_mem});
    args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_scale_mem});
  }
  if (alpha.has_value()) {
    // set alpha in args for post-op binary
    dnnl::memory alpha_mem =
        make_onednn_memory(alpha_md, engine, alpha.value().data_ptr());
    args.insert(
        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, alpha_mem});
  }

  dnnl::matmul matmul_p = dnnl::matmul(matmul_pd);
  sycl::event matmul_fwd_event =
      dnnl::sycl_interop::execute(matmul_p, stream, args);
  return matmul_fwd_event;
}

} // namespace at::native::onednn
