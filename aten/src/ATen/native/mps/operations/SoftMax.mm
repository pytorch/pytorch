//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/mps/OperationUtils.h>

#ifdef __OBJC__
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_softmax_backward_data_native.h>
#include <ATen/ops/_softmax_native.h>
#include <ATen/ops/empty_like.h>
#endif

namespace at::native {

using namespace mps;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Softmax_metallib.h>
#endif

namespace {

struct StridedAxes {
  std::vector<uint> sizes, in_strides, out_strides;
};
StridedAxes collect_strided_axes(const Tensor& in, const Tensor& out, int64_t dim) {
  StridedAxes a;
  for (int64_t i = 0; i < in.dim(); ++i) {
    if (i == dim)
      continue;
    a.sizes.push_back(static_cast<uint>(in.size(i)));
    a.in_strides.push_back(static_cast<uint>(in.stride(i)));
    a.out_strides.push_back(static_cast<uint>(out.stride(i)));
  }
  auto perm = std::vector<size_t>(a.sizes.size());
  std::iota(perm.begin(), perm.end(), size_t{0});
  std::sort(perm.begin(), perm.end(), [&](size_t x, size_t y) { return a.in_strides[x] > a.in_strides[y]; });
  StridedAxes s;
  for (auto p : perm) {
    s.sizes.push_back(a.sizes[p]);
    s.in_strides.push_back(a.in_strides[p]);
    s.out_strides.push_back(a.out_strides[p]);
  }
  if (s.sizes.empty()) {
    s.sizes.push_back(1);
    s.in_strides.push_back(0);
    s.out_strides.push_back(0);
  }
  return s;
}

// choose_softmax_kernel picks a variant, launch_softmax_kernel dispatches it.
// Variants describe what each kernel does differently and why it exists.
struct SoftmaxKernelChoice {
  enum class Variant {
    SplitK, // 2-pass online merge: K× more TGs in flight when num_rows is too small to fill the GPU.
    SingleRow, // one TG/row, in-shmem parallel reduce, dim_size fits within one TG.
    Looped, // one TG/row, loops within TG when dim_size exceeds one TG worth of threads.
    General, // last-dim-contig fallback when dim_size doesn't divide cleanly.
    StridedInnerVec, // stride[dim]==1, dim_size%4==0: vec4 loads for memory throughput.
    StridedInner, // stride[dim]==1: scalar loads.
    StridedOuter, // tiny dim_size (≤64): one thread per row, no shmem reduction.
    StridedChunked, // stride[dim]>1: 32-row TG, multi-warp split-row + shmem partial-reduce combine.
  };
  Variant variant;
  int n_reads = 0; // SingleRow
  int n_local = 0; // StridedInner{,Vec}
  int chunk_cap = 0; // StridedChunked
  uint tptg = 0; // SingleRow / Looped / General / StridedInner{,Vec}
};

SoftmaxKernelChoice choose_softmax_kernel(const Tensor& input, const Tensor& output, int64_t dim) {
  using V = SoftmaxKernelChoice::Variant;
  auto dim_size = input.size(dim);
  auto num_rows = input.numel() / dim_size;
  auto last_dim_contig = (dim == input.dim() - 1) && input.is_contiguous() && output.is_contiguous();
  if (last_dim_contig) {
    auto starves_general = (num_rows < 4 && dim_size > 16384) || (num_rows <= 8 && dim_size > 65536);
    if (starves_general) {
      return {V::SplitK};
    }
    if (dim_size > 4096 && dim_size % 4096 == 0) {
      auto c = SoftmaxKernelChoice{V::Looped};
      c.tptg = 1024;
      return c;
    }
    if (dim_size <= 4096 && dim_size % 128 == 0) {
      auto warps_per_group = dim_size / 128;
      if (warps_per_group >= 8 || num_rows >= 16) {
        auto c = SoftmaxKernelChoice{V::SingleRow};
        c.n_reads = 4;
        c.tptg = static_cast<uint>(dim_size / 4);
        return c;
      }
    }
    if (dim_size <= 1024 && dim_size % 32 == 0) {
      auto c = SoftmaxKernelChoice{V::SingleRow};
      c.n_reads = 1;
      c.tptg = static_cast<uint>(dim_size);
      return c;
    }
    auto c = SoftmaxKernelChoice{V::General};
    c.tptg = std::min<uint>(static_cast<uint>(((dim_size + 31) / 32) * 32), 1024);
    return c;
  }

  auto input_dim_stride = static_cast<uint>(input.stride(dim));
  auto output_dim_stride = static_cast<uint>(output.stride(dim));
  if (input_dim_stride == 1 && output_dim_stride == 1) {
    auto can_vec4 = (dim_size % 4 == 0) && (dim_size <= 16384);
    if (can_vec4) {
      uint elems_per_thread_vec = 4;
      auto inner_tptg = std::min<uint>(static_cast<uint>(((dim_size / elems_per_thread_vec + 31) / 32) * 32), 1024);
      auto n = (dim_size + inner_tptg * elems_per_thread_vec - 1) / (inner_tptg * elems_per_thread_vec);
      auto c = SoftmaxKernelChoice{V::StridedInnerVec};
      c.n_local = n <= 1 ? 1 : (n <= 2 ? 2 : 4);
      c.tptg = inner_tptg;
      return c;
    }
    auto inner_tptg = std::min<uint>(static_cast<uint>(((dim_size + 31) / 32) * 32), 1024);
    auto n = (dim_size + inner_tptg - 1) / inner_tptg;
    auto c = SoftmaxKernelChoice{V::StridedInner};
    c.n_local = n <= 1 ? 1 : (n <= 2 ? 2 : (n <= 4 ? 4 : (n <= 8 ? 8 : 16)));
    c.tptg = inner_tptg;
    return c;
  }
  if (dim_size <= 64) {
    return {V::StridedOuter};
  }
  auto num_chunks_dispatch = std::clamp<uint>((dim_size + 31) / 32, 1, 32);
  auto c = SoftmaxKernelChoice{V::StridedChunked};
  c.chunk_cap = (dim_size <= 1024 && num_chunks_dispatch <= 16) ? 32 : 0;
  return c;
}

void launch_softmax_kernel(const Tensor& input, int64_t dim, const Tensor& output, const SoftmaxKernelChoice& choice) {
  using V = SoftmaxKernelChoice::Variant;
  auto type_str = scalarToMetalTypeString(input);
  auto dim_size = static_cast<uint>(input.size(dim));
  auto num_rows = static_cast<uint>(input.numel() / dim_size);
  auto stream = getCurrentMPSStream();

  switch (choice.variant) {
    case V::SplitK: {
      constexpr uint num_chunks = 32;
      constexpr uint tptg = 256;
      auto partials = at::empty({static_cast<int64_t>(num_rows), static_cast<int64_t>(num_chunks), 2},
                                input.options().dtype(kFloat));
      auto p1 = fmt::format("softmax_split_k_pass1_{}", type_str);
      auto p2 = fmt::format("softmax_split_k_pass2_{}", type_str);
      dispatch_sync_with_rethrow(stream->queue(), ^() {
        @autoreleasepool {
          auto enc = stream->commandEncoder();
          auto pso1 = lib.getPipelineStateForFunc(p1);
          [enc setComputePipelineState:pso1];
          auto dim_size_num_chunks = std::array<uint32_t, 2>{dim_size, num_chunks};
          mtl_setArgs(enc, input, partials, dim_size_num_chunks);
          [enc dispatchThreadgroups:MTLSizeMake(num_chunks * num_rows, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tptg, 1, 1)];
          [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
          auto pso2 = lib.getPipelineStateForFunc(p2);
          [enc setComputePipelineState:pso2];
          mtl_setArgs(enc, input, output, partials, dim_size_num_chunks);
          [enc dispatchThreadgroups:MTLSizeMake(num_chunks * num_rows, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tptg, 1, 1)];
        }
      });
      return;
    }

    case V::SingleRow:
    case V::Looped:
    case V::General: {
      std::string name;
      if (choice.variant == V::SingleRow) {
        name = fmt::format("softmax_single_pass_{}_{}", type_str, choice.n_reads);
      } else if (choice.variant == V::Looped) {
        name = fmt::format("softmax_looped_{}", type_str);
      } else {
        name = fmt::format("softmax_general_{}", type_str);
      }
      dispatch_sync_with_rethrow(stream->queue(), ^() {
        @autoreleasepool {
          auto enc = stream->commandEncoder();
          auto pso = lib.getPipelineStateForFunc(name);
          [enc setComputePipelineState:pso];
          mtl_setArgs(enc, input, output, dim_size);
          [enc dispatchThreadgroups:MTLSizeMake(num_rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(choice.tptg, 1, 1)];
        }
      });
      return;
    }

    case V::StridedInnerVec:
    case V::StridedInner:
    case V::StridedOuter:
    case V::StridedChunked: {
      auto input_dim_stride = static_cast<uint>(input.stride(dim));
      auto output_dim_stride = static_cast<uint>(output.stride(dim));
      auto axes = collect_strided_axes(input, output, dim);
      auto ndim_other = static_cast<uint>(axes.sizes.size());

      std::string name;
      switch (choice.variant) {
        case V::StridedInnerVec:
          name = fmt::format("softmax_strided_inner_vec_{}_{}_4", type_str, choice.n_local);
          break;
        case V::StridedInner:
          name = fmt::format("softmax_strided_inner_{}_{}", type_str, choice.n_local);
          break;
        case V::StridedOuter:
          name = fmt::format("softmax_strided_outer_{}", type_str);
          break;
        case V::StridedChunked:
          name = fmt::format("softmax_strided_chunked_{}_{}", type_str, choice.chunk_cap);
          break;
        default:
          break;
      }

      dispatch_sync_with_rethrow(stream->queue(), ^() {
        @autoreleasepool {
          auto enc = stream->commandEncoder();
          auto pso = lib.getPipelineStateForFunc(name);
          [enc setComputePipelineState:pso];
          if (choice.variant == V::StridedInnerVec) {
            auto params = std::array<uint32_t, 2>{dim_size, ndim_other};
            mtl_setArgs(enc, input, output, params, axes.sizes, axes.in_strides, axes.out_strides);
            [enc dispatchThreadgroups:MTLSizeMake(num_rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(choice.tptg, 1, 1)];
          } else if (choice.variant == V::StridedInner) {
            auto params = std::array<uint32_t, 4>{dim_size, input_dim_stride, output_dim_stride, ndim_other};
            mtl_setArgs(enc, input, output, params, axes.sizes, axes.in_strides, axes.out_strides);
            [enc dispatchThreadgroups:MTLSizeMake(num_rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(choice.tptg, 1, 1)];
          } else if (choice.variant == V::StridedOuter) {
            auto params = std::array<uint32_t, 4>{dim_size, input_dim_stride, output_dim_stride, num_rows};
            mtl_setArgs(enc, input, output, params, ndim_other, axes.sizes, axes.in_strides, axes.out_strides);
            auto tptg = std::min<uint>(num_rows, 256);
            [enc dispatchThreads:MTLSizeMake(num_rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(tptg, 1, 1)];
          } else {
            auto num_chunks = std::clamp<uint>((dim_size + 31) / 32, 1, 32);
            auto tptg = num_chunks * 32;
            auto num_tgs = (num_rows + 31) / 32;
            auto params = std::array<uint32_t, 4>{dim_size, input_dim_stride, output_dim_stride, num_rows};
            mtl_setArgs(enc, input, output, params, ndim_other, axes.sizes, axes.in_strides, axes.out_strides);
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(tptg, 1, 1)];
          }
        }
      });
      return;
    }
  }
}

} // namespace

TORCH_IMPL_FUNC(softmax_mps_out)
(const Tensor& input_, const int64_t dim, const bool half_to_float, const Tensor& output) {
  TORCH_CHECK(!half_to_float, "softmax with half to float conversion is not supported on MPS");
  TORCH_CHECK(c10::isFloatingType(input_.scalar_type()), "softmax only supported for floating types");

  if (input_.numel() == 0) {
    return;
  }

  Tensor input = input_.dim() == 0 ? input_.view(1) : input_;
  Tensor out = output.dim() == 0 ? output.view(1) : output;

  int64_t dim_ = maybe_wrap_dim(dim, input.dim());
  TORCH_CHECK(dim_ >= 0 && dim_ < input.dim(), "Softmax:dim must be non-negative and less than input dimensions");

  auto choice = choose_softmax_kernel(input, out, dim_);
  launch_softmax_kernel(input, dim_, out, choice);
}

TORCH_IMPL_FUNC(softmax_backward_mps_out)
(const Tensor& grad_, const Tensor& output_, int64_t dim, ScalarType input_dtype, const Tensor& grad_input) {
  if (output_.numel() == 0) {
    return;
  }

  Tensor grad;
  if (grad_.dim() == 0) {
    grad = grad_.view(1);
  } else
    grad = grad_;

  Tensor output;
  if (output_.dim() == 0) {
    output = output_.view(1);
  } else
    output = output_;

  int64_t dim_ = maybe_wrap_dim(dim, grad.dim());
  TORCH_CHECK(dim_ >= 0 && dim_ < grad.dim(), "Grad:dim must be non-negative and less than input dimensions");

  using namespace mps;
  using CachedGraph = MPSUnaryGradCachedGraph;
  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    MPSShape* grad_shape = mps::getMPSShape(grad);
    NSString* ns_shape_key = [[grad_shape valueForKey:@"description"] componentsJoinedByString:@","];

    std::string key = "softmax_backward_mps_out:" + getMPSTypeString(output) + ":" + [ns_shape_key UTF8String] + ":" +
        std::to_string(dim_);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* softmaxTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(output), grad_shape);
      MPSGraphTensor* gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(grad), grad_shape);

      MPSGraphTensor* mulTensor = [mpsGraph multiplicationWithPrimaryTensor:softmaxTensor
                                                            secondaryTensor:gradOutputTensor
                                                                       name:nil];
      MPSGraphTensor* mulSumTensor = [mpsGraph reductionSumWithTensor:mulTensor axis:(NSInteger)dim_ name:nil];
      MPSGraphTensor* gradSubTensor = [mpsGraph subtractionWithPrimaryTensor:gradOutputTensor
                                                             secondaryTensor:mulSumTensor
                                                                        name:nil];
      MPSGraphTensor* gradInputTensor = [mpsGraph multiplicationWithPrimaryTensor:softmaxTensor
                                                                  secondaryTensor:gradSubTensor
                                                                             name:nil];

      newCachedGraph->outputTensor_ = softmaxTensor;
      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->gradInputTensor_ = gradInputTensor;
    });

    Placeholder softmaxPlaceholder = Placeholder(cachedGraph->outputTensor_, output, grad_shape);
    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad, grad_shape);
    Placeholder gradInputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, grad_input);

    auto feeds = dictionaryFromPlaceholders(softmaxPlaceholder, gradOutputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, gradInputPlaceholder);
  }
}

} // namespace at::native
