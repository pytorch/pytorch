#include "mha_fwd.h"
#include "mha_common.h"

// batch, numhead_qo,numhead_kv,seqlen_qo,seqlen_kv,headsize_qk,headsize_vo
using ProblemShapeRegular = cute::tuple<int, int, int, int, int, int, int>;

namespace cute {

template <class...>
class MhaName;

template <class FMHAPrefillKernel>
struct FA2Runner {
  using StrideQ = typename FMHAPrefillKernel::StrideQ;
  using StrideK = typename FMHAPrefillKernel::StrideK;
  using StrideV = typename FMHAPrefillKernel::StrideV;
  using StrideO = typename FMHAPrefillKernel::StrideO;

  using ElementQ = typename FMHAPrefillKernel::ElementQ;
  using ElementK = typename FMHAPrefillKernel::ElementK;
  using ElementV = typename FMHAPrefillKernel::ElementV;
  using ElementAcc = typename FMHAPrefillKernel::ElementAccumulator;

  using CollectiveEpilogue = typename FMHAPrefillKernel::CollectiveEpilogue;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;
  using ElementAccumulator = typename CollectiveEpilogue::ElementAccumulator;

  using ProblemShapeType = typename FMHAPrefillKernel::ProblemShape;

  //
  // Data members
  //

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;

  //
  // Methods
  //

  // Note that the GemmUniversalAdapter currently doesn't support flash
  // attention, which is why this secondary `run` function is required to launch
  // the kernel.
  void run(sycl::queue& queue, typename FMHAPrefillKernel::Params params) {
    dim3 const block = FMHAPrefillKernel::get_block_shape();
    dim3 const grid = FMHAPrefillKernel::get_grid_shape(params);

    // configure smem size and carveout
    int smem_size = FMHAPrefillKernel::SharedStorageSize;

    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

// Launch parameters depend on whether SYCL compiler supports work-group scratch
// memory extension
#if !defined(SYCL_EXT_ONEAPI_WORK_GROUP_SCRATCH_MEMORY)
    using namespace compat::experimental;
    auto event = launch<
        cutlass::device_kernel<FMHAPrefillKernel>,
        MhaName<FMHAPrefillKernel>>(
        launch_policy{
            sycl_grid,
            sycl_block,
            local_mem_size{static_cast<std::size_t>(smem_size)},
            kernel_properties{sycl_exp::sub_group_size<
                FMHAPrefillKernel::DispatchPolicy::SubgroupSize>}},
        queue,
        params);
#else
    compat::experimental::launch_properties launch_props{
        sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
    };
    compat::experimental::kernel_properties kernel_props{
        sycl::ext::oneapi::experimental::sub_group_size<
            FMHAPrefillKernel::DispatchPolicy::SubgroupSize>};
    compat::experimental::launch_policy policy{
        sycl_grid, sycl_block, launch_props, kernel_props};
    auto event = compat::experimental::launch<
        cutlass::device_kernel<FMHAPrefillKernel>,
        MhaName<FMHAPrefillKernel>>(policy, queue, params);
#endif
  }

  void run(
      sycl::queue& queue,
      ProblemShapeType problem_size,
      const cutlass::KernelHardwareInfo& hw_info,
      const ElementQ* inputQ,
      const ElementK* inputK,
      const ElementV* inputV,
      ElementOutput* output,
      float* logsumexp,
      float softmax_scale) {
    auto
        [batch,
         num_heads_q,
         num_heads_kv,
         seq_len_qo,
         seq_len_kv,
         head_size_qk,
         head_size_vo] = problem_size;

    stride_Q = cutlass::make_cute_packed_stride(
        StrideQ{},
        cute::make_shape(seq_len_qo, head_size_qk, batch * num_heads_q));
    stride_K = cutlass::make_cute_packed_stride(
        StrideK{},
        cute::make_shape(seq_len_kv, head_size_qk, batch * num_heads_kv));
    stride_V = cutlass::make_cute_packed_stride(
        StrideV{},
        cute::make_shape(head_size_vo, seq_len_kv, batch * num_heads_kv));
    stride_O = cutlass::make_cute_packed_stride(
        StrideO{},
        cute::make_shape(seq_len_qo, head_size_vo, batch * num_heads_q));

    typename FMHAPrefillKernel::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        {inputQ, stride_Q, inputK, stride_K, inputV, stride_V},
        {softmax_scale},
        {output, stride_O, logsumexp},
        hw_info, softmax_scale};

    // Define device-global scratch memory
    size_t workspace_size = FMHAPrefillKernel::get_workspace_size(arguments);
    at::Tensor workspace_tensor = at::empty(
        {static_cast<int64_t>(workspace_size)},
        at::device(at::kXPU).dtype(at::kByte));

    if (!FMHAPrefillKernel::can_implement(arguments)) {
      TORCH_CHECK(
          false,
          "Invalid Problem Size",
          batch,
          "x",
          num_heads_q,
          "x",
          seq_len_qo,
          "x",
          seq_len_kv,
          "x",
          head_size_qk,
          "x",
          head_size_vo);
      return;
    }

    // Initialize the workspace
    CUTLASS_CHECK(FMHAPrefillKernel::initialize_workspace(
        arguments, workspace_tensor.data_ptr()));

    // Convert host-side arguments to device-side arguments to be passed to the
    // kernel
    auto params = FMHAPrefillKernel::to_underlying_arguments(
        arguments, workspace_tensor.data_ptr());

    // Launch a SYCL kernel using scratch/shared memory
    run(queue, params);
  }
};

template <
    typename T,
    typename ProblemShape,
    bool IS_CAUSAL,
    typename TileShapeQK,
    typename TileShapePV,
    typename TileShapeOutPut,
    typename SubgroupLayout,
    int PipelineStages>
void run_mha_fwd_(
    sycl::queue& queue,
    ProblemShape& problem_shape,
    const T* query,
    const T* key,
    const T* value,
    T* out,
    float* logsumexp,
    float scale) {
  cutlass::KernelHardwareInfo hw_info;

  using LayoutQ = cutlass::layout::RowMajor;
  using LayoutK = cutlass::layout::ColumnMajor;
  using LayoutV = cutlass::layout::RowMajor;
  using LayoutO = cutlass::layout::RowMajor;

  using ElementInputQ = T;
  using ElementInputKV = T;
  using ElementOutput = T;
  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;

  using MMAOperation = std::conditional_t<
      std::is_same_v<T, bfloat16_t>,
      XE_8x16x16_F32BF16BF16F32_TT,
      XE_8x16x16_F32F16F16F32_TT>;
  using GmemTiledCopyQ = XE_2D_U16x8x32_LD_N;
  using GmemTiledCopyK =
      XE_2D_U16x16x16_LD_T; // _T designates a transposed block load operation
  using GmemTiledCopyV = XE_2D_U16x16x32_LD_V;
  using GmemTiledCopyStore = XE_2D_U16x8x16_ST_N; // Change to output BF16

  using GEMMDispatchPolicy =
      cutlass::gemm::MainloopIntelXeXMX16<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;
  using CollectiveEpilogue =
      cutlass::flash_attention::collective::FlashPrefillEpilogue<
          EpilogueDispatchPolicy,
          MMAOperation,
          TileShapeOutPut,
          SubgroupLayout,
          ElementComputeEpilogue,
          ElementOutput,
          cutlass::gemm::TagToStrideC_t<LayoutO>,
          ElementOutput,
          GmemTiledCopyStore>;
  using CollectiveSoftmaxEpilogue =
      cutlass::flash_attention::collective::FlashPrefillSoftmaxEpilogue<
          IS_CAUSAL,
          EpilogueDispatchPolicy,
          ElementAccumulator>;

  using namespace cutlass::fmha::collective;

  using ProblemShapeType = ProblemShape;

  // Mainloop
  using CollectiveMainloop =
      cutlass::flash_attention::collective::FlashPrefillMma<
          GEMMDispatchPolicy,
          ProblemShapeType,
          ElementInputQ,
          cutlass::gemm::TagToStrideA_t<LayoutQ>,
          ElementInputKV,
          cutlass::gemm::TagToStrideB_t<LayoutK>,
          ElementInputKV,
          cutlass::gemm::TagToStrideB_t<LayoutV>,
          MMAOperation,
          TileShapeQK,
          TileShapePV,
          SubgroupLayout,
          GmemTiledCopyQ, // Q
          GmemTiledCopyK, // K
          GmemTiledCopyV, // V,
          IS_CAUSAL>;
  using FMHAPrefillKernel = cutlass::flash_attention::kernel::FMHAPrefill<
      ProblemShapeType,
      CollectiveMainloop,
      CollectiveSoftmaxEpilogue,
      CollectiveEpilogue,
      cutlass::flash_attention::IndividualScheduler>;

  FA2Runner<FMHAPrefillKernel> runner;
  runner.run(
      queue, problem_shape, hw_info, query, key, value, out, logsumexp, scale);
}

template <typename T, typename ProblemShape, bool IS_CAUSAL>
void run_mha_fwd_(
    sycl::queue& queue,
    ProblemShape& problem_shape,
    const T* query,
    const T* key,
    const T* value,
    T* out,
    float* logsumexp,
    float scale) {
  const int headdim = get<5>(problem_shape);

#define run_mha_fwd_specialized( \
    TileShapeQK_,                \
    TileShapePV_,                \
    TileShapeOutPut_,            \
    SubgroupLayout_,             \
    PipelineStages_)             \
  run_mha_fwd_<                  \
      T,                         \
      ProblemShape,              \
      IS_CAUSAL,                 \
      TileShapeQK_,              \
      TileShapePV_,              \
      TileShapeOutPut_,          \
      SubgroupLayout_,           \
      PipelineStages_>(          \
      queue, problem_shape, query, key, value, out, logsumexp, scale);

  constexpr int PipelineStages = 2;
  if (headdim == 64) {
    using TileShapeQK = Shape<_128, _64, _64>;
    using TileShapePV = Shape<_128, _32, _64>;
    using TileShapeOutPut = Shape<_128, _64, _64>;
    using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>;
    run_mha_fwd_specialized(
        TileShapeQK,
        TileShapePV,
        TileShapeOutPut,
        SubgroupLayout,
        PipelineStages);
  } else if (headdim == 96) {
    using TileShapeQK = Shape<_128, _64, _32>;
    using TileShapePV = Shape<_128, _32, _64>;
    using TileShapeOutPut = Shape<_128, _96, _64>;
    using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>;
    run_mha_fwd_specialized(
        TileShapeQK,
        TileShapePV,
        TileShapeOutPut,
        SubgroupLayout,
        PipelineStages);
  } else if (headdim == 128) {
    using TileShapeQK = Shape<_256, _32, _64>;
    using TileShapePV = Shape<_256, _32, _32>;
    using TileShapeOutPut = Shape<_256, _128, _32>;
    using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>;
    run_mha_fwd_specialized(
        TileShapeQK,
        TileShapePV,
        TileShapeOutPut,
        SubgroupLayout,
        PipelineStages);
  } else if (headdim == 192) {
    using TileShapeQK = Shape<_256, _64, _64>;
    using TileShapePV = Shape<_256, _32, _64>;
    using TileShapeOutPut = Shape<_256, _192, _64>;
    using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>;
    run_mha_fwd_specialized(
        TileShapeQK,
        TileShapePV,
        TileShapeOutPut,
        SubgroupLayout,
        PipelineStages);
  } else {
    TORCH_CHECK(
        false, "FlashAttentionForwardXPU only support headdim 64,96,128,192");
  }
}

template <typename ProblemShape>
void run_mha_fwd(
    sycl::queue& queue,
    ProblemShape& problem_shape,
    const void* query,
    const void* key,
    const void* value,
    void* out,
    void* logsumexp,
    bool is_causal,
    float scale,
    at::ScalarType dtype) {
  FP16_SWITCH(dtype == at::kHalf, [&] {
    BOOL_SWITCH(is_causal, IS_CAUSAL, [&] {
      run_mha_fwd_<elem_type, ProblemShape, IS_CAUSAL>(
          queue,
          problem_shape,
          static_cast<const elem_type*>(query),
          static_cast<const elem_type*>(key),
          static_cast<const elem_type*>(value),
          static_cast<elem_type*>(out),
          static_cast<float*>(logsumexp),
          scale);
    });
  });
}
} // namespace cute

namespace sycltla {

std::tuple<at::Tensor, at::Tensor> flash_attention_forward_sycltla(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const double dropout,
    const bool is_causal,
    const float scale) {
  TORCH_CHECK(
      dropout == 0.0,
      "FlashAttentionForwardXPU does not only support dropout > 0.0 yet");

  CHECK_DEVICE(query);
  CHECK_DEVICE(key);
  CHECK_DEVICE(value);

  TORCH_CHECK(
      !query.is_nested() && !key.is_nested() && !value.is_nested(),
      "FlashAttentionForwardXPU only support dense inputs");

  auto dtype = query.scalar_type();
  TORCH_CHECK(
      dtype == at::kHalf || dtype == at::kBFloat16,
      "FlashAttentionForwardXPU only support fp16 and bf16 data type");
  TORCH_CHECK(
      key.scalar_type() == dtype,
      "FlashAttentionForwardXPU: query and key must have the same dtype");
  TORCH_CHECK(
      value.scalar_type() == dtype,
      "FlashAttentionForwardXPU: query and value must have the same dtype");

  TORCH_CHECK(
      query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
      "FlashAttentionForwardXPU requires query, key, value to be 4 dimensional");

  const int batch_size = query.sizes()[0];
  const int numhead_qo = query.sizes()[1];
  const int numhead_kv = key.sizes()[1];
  const int seqlen_qo = query.sizes()[2];
  const int seqlen_kv = key.sizes()[2];
  const int headsize_qk = query.sizes()[3];
  const int headsize_vo = value.sizes()[3];

  CHECK_SHAPE(query, batch_size, numhead_qo, seqlen_qo, headsize_qk);
  CHECK_SHAPE(key, batch_size, numhead_kv, seqlen_kv, headsize_qk);
  CHECK_SHAPE(value, batch_size, numhead_kv, seqlen_kv, headsize_vo);

  TORCH_CHECK(
      numhead_qo % numhead_kv == 0,
      "FlashAttentionForwardXPU: numhead_qo must be divisible by numhead_kv");
  TORCH_CHECK(
      numhead_qo == numhead_kv,
      "FlashAttentionForwardXPU: currently only support numhead_qo == numhead_kv");

  TORCH_CHECK(
      query.stride(-1) == 1,
      "FlashAttentionForwardXPU: input tensor must have contiguous last dimension");
  TORCH_CHECK(
      key.stride(-1) == 1,
      "FlashAttentionForwardXPU: input tensor must have contiguous last dimension");
  TORCH_CHECK(
      value.stride(-1) == 1,
      "FlashAttentionForwardXPU: input tensor must have contiguous last dimension");

  ATTN_TENSOR_LAYOUT layout = get_attn_tensor_layout(query);
  if (layout == ATTN_TENSOR_LAYOUT::UNSUPPORTED) {
    TORCH_CHECK(
        false,
        "FlashAttentionForwardXPU: only support BHSD or BSHD layout, got query with shape ",
        query.sizes(),
        ", stride ",
        query.strides());
  }
  layout = fuse_attn_tensor_layout(layout, get_attn_tensor_layout(key));
  TORCH_CHECK(
      ATTN_TENSOR_LAYOUT::UNSUPPORTED != layout,
      "FlashAttentionBackwardXPU: query and key must have the same layout, got query with layout ",
      to_string(layout),
      ", key with layout ",
      to_string(get_attn_tensor_layout(key)));
  layout = fuse_attn_tensor_layout(layout, get_attn_tensor_layout(value));
  TORCH_CHECK(
      ATTN_TENSOR_LAYOUT::UNSUPPORTED != layout,
      "FlashAttentionBackwardXPU: query and value must have the same layout, got query with layout ",
      to_string(layout),
      ", value with layout ",
      to_string(get_attn_tensor_layout(value)));
  if (layout == ATTN_TENSOR_LAYOUT::BXD) {
    layout = ATTN_TENSOR_LAYOUT::BSHD;
  }
  TORCH_CHECK(
      layout == ATTN_TENSOR_LAYOUT::BSHD,
      "FlashAttentionBackwardXPU: currently only support BSHD layout");

  auto opts = query.options();
  at::Tensor out;
  if (layout == ATTN_TENSOR_LAYOUT::BHSD) {
    out = at::empty({batch_size, numhead_qo, seqlen_qo, headsize_vo}, opts);
  } else if (layout == ATTN_TENSOR_LAYOUT::BSHD) {
    out = at::empty({batch_size, seqlen_qo, numhead_qo, headsize_vo}, opts)
              .permute({0, 2, 1, 3});
  } else {
    TORCH_CHECK(
        false, "FlashAttentionForwardXPU: only support BHSD or BSHD layout");
  }

  at::Tensor logsumexp =
      at::empty({batch_size, numhead_qo, seqlen_qo}, opts.dtype(at::kFloat));

  auto sycl_queue = at::xpu::getCurrentXPUStream().queue();
  auto device_architecture =
      sycl_queue.get_device()
          .get_info<
              sycl::ext::oneapi::experimental::info::device::architecture>();
  constexpr auto supported_architectures =
      std::array<sycl::ext::oneapi::experimental::architecture, 4>{
          sycl::ext::oneapi::experimental::architecture::intel_gpu_pvc,
          sycl::ext::oneapi::experimental::architecture::intel_gpu_pvc_vg,
          sycl::ext::oneapi::experimental::architecture::intel_gpu_bmg_g21,
          sycl::ext::oneapi::experimental::architecture::intel_gpu_bmg_g31};
  if (std::find(
          supported_architectures.begin(),
          supported_architectures.end(),
          device_architecture) == supported_architectures.end()) {
    TORCH_CHECK(
        false,
        "XPU device architecture does not support flash attention. Supported architectures are: intel_gpu_pvc, intel_gpu_pvc_vg, intel_gpu_bmg_g21, intel_gpu_bmg_g31.");
  }

  auto problem_shape = ProblemShapeRegular(
      batch_size,
      numhead_qo,
      numhead_kv,
      seqlen_qo,
      seqlen_kv,
      headsize_qk,
      headsize_vo);

  cute::run_mha_fwd<decltype(problem_shape)>(
      sycl_queue,
      problem_shape,
      query.data_ptr(),
      key.data_ptr(),
      value.data_ptr(),
      out.data_ptr(),
      logsumexp.data_ptr(),
      is_causal,
      scale,
      dtype);

  return std::tuple<at::Tensor, at::Tensor>{out, logsumexp};
}

} // namespace sycltla