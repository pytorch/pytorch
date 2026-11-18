This directory contains files copied from
`src/fastertransformer/cutlass_extensions/include/cutlass_extensions`
directory of the source tree of
[*FasterTransformer*](https://github.com/NVIDIA/FasterTransformer)
project.  These are intended for supporting mixed datatypes GEMM
implementation, in `aten/src/ATen/native/cuda/MixedDTypesLinear.cu`
file of *PyTorch* source tree.  Not all files from given directory of
*FasterTransformer* project are here, only ones necessary to support
mentioned functionality.

The original copy of these files is made from commit `f8e42aa` of
*FasterTransformer project*.  The changes from original files are
minimal, just to support *CUTLASS* 3.x (*FasterTransfomer* project
was, as of mentioned commit, based on *CUTLASS* 2.10).  However, the
copies of files in the *PyTorch* source tree are linted using
*PyTorch* lint rules, so at this stage they differ quite a bit from
the original files.  Thus, for keeping track of the original changes,
here is the diff between the two sets of files, before linting:

```
Only in FasterTransformer/src/fastertransformer/cutlass_extensions/include/cutlass_extensions: compute_occupancy.h
Only in FasterTransformer/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/epilogue: epilogue_quant_helper.h
Only in FasterTransformer/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/epilogue: threadblock
diff -r FasterTransformer/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/gemm/kernel/fpA_intB_gemm.h pytorch/aten/src/ATen/native/cuda/cutlass_extensions/gemm/kernel/fpA_intB_gemm.h
157c157,158
<     struct Params {
---
>     struct Params
>     {
183d183
<         CUTLASS_HOST_DEVICE
186d185
<         CUTLASS_HOST_DEVICE
188,190c187,188
<                cutlass::gemm::GemmCoord const& grid_tiled_shape,
<                const int                       gemm_k_size,
<                void*                           workspace = nullptr):
---
>                int                             device_sms,
>                int                             sm_occupancy):
192d189
<             grid_tiled_shape(grid_tiled_shape),
205,206d201
<             semaphore(static_cast<int*>(workspace)),
<             gemm_k_size(gemm_k_size),
210a206,227
>             ThreadblockSwizzle swizzle;
>             grid_tiled_shape = swizzle.get_tiled_shape(
>                 args.problem_size,
>                 {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
>                 args.batch_count);
>
>             gemm_k_size = args.problem_size.k();
>         }
>
>         size_t get_workspace_size() const
>         {
>             return 0;
>         }
>
>         Status init_workspace(void *workspace,cudaStream_t stream = nullptr)
>         {
>             return Status::kSuccess;
>         }
>
>         dim3 get_grid_dims() const
>         {
>             return ThreadblockSwizzle().get_grid_shape(grid_tiled_shape);
278,283d294
<     static size_t get_extra_workspace_size(Arguments const& args, cutlass::gemm::GemmCoord const& grid_tiled_shape)
<     {
<
<         return 0;
<     }
<
464a476,482
>     CUTLASS_DEVICE
>     static void invoke(Params const &params, SharedStorage &shared_storage)
>     {
>         GemmFpAIntB op;
>         op(params, shared_storage);
>     }
>
492c510
< }  // namespace cutlass
\ No newline at end of file
---
> }  // namespace cutlass
Only in FasterTransformer/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/gemm/kernel: gemm_moe_problem_visitor.h
Only in FasterTransformer/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/gemm/kernel: gemm_with_epilogue_visitor.h
Only in FasterTransformer/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/gemm/kernel: moe_cutlass_kernel.h
Only in FasterTransformer/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/gemm/kernel: moe_problem_visitor.h
diff -r FasterTransformer/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/gemm/warp/mma_tensorop_dequantizer.h pytorch/aten/src/ATen/native/cuda/cutlass_extensions/gemm/warp/mma_tensorop_dequantizer.h
55c55,58
< #include <src/fastertransformer/utils/cuda_bf16_wrapper.h>
---
> //#include <src/fastertransformer/utils/cuda_bf16_wrapper.h>
> //#ifdef ENABLE_BF16
> #include <cuda_bf16.h>
> //#endif
155c158,159
< #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && defined(ENABLE_BF16))
---
> //#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && defined(ENABLE_BF16))
> #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
470c474
< ////////////////////////////////////////////////////////////////////////////////
\ No newline at end of file
---
> ////////////////////////////////////////////////////////////////////////////////
```

As mentioned [here](https://github.com/NVIDIA/cutlass/discussions/911)
and [here](https://github.com/NVIDIA/cutlass/issues/1060), *CUTLASS*
itself is expected to include the functionality provided by these
extensions, so hopefully this whole directory will be removed from
*PyTorch* source tree at some later time.
