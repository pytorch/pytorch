//  Copyright (c) 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/mps/MetalShaderLibrary.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/ScatterGather.h>
#include <ATen/native/mps/ScatterGather_metallib.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/gather_native.h>
#include <ATen/ops/scatter_add_native.h>
#include <ATen/ops/scatter_native.h>
#include <ATen/ops/view_as_real.h>
#endif

namespace at::native {

static Tensor maybe_expand_0_dim(const Tensor& t) {
  return t.dim() == 0 ? t.view({1}) : t;
}

static Tensor expand_index_as_real(const Tensor& index) {
  auto index_view = maybe_expand_0_dim(index);
  std::vector<int64_t> index_expanded_sizes = index_view.sizes().vec();
  index_expanded_sizes.push_back(2);
  auto index_expanded = index_view.unsqueeze(-1).expand(index_expanded_sizes);
  return index_expanded;
}

static ScatterGatherParams<> build_scatter_params(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  ScatterGatherParams<> params;
  params.ndim = static_cast<int32_t>(self.dim());
  params.dim = static_cast<int32_t>(dim);
  for (int32_t i = 0; i < params.ndim; i++) {
    params.self_strides[i] = static_cast<uint32_t>(self.stride(i));
    params.self_sizes[i] = static_cast<uint32_t>(self.size(i));
    params.src_strides[i] = static_cast<uint32_t>(src.stride(i));
    params.src_sizes[i] = static_cast<uint32_t>(src.size(i));
    params.index_strides[i] = static_cast<uint32_t>(index.stride(i));
    params.index_sizes[i] = static_cast<uint32_t>(index.size(i));
  }
  return params;
}

static std::string metal_type_string(c10::ScalarType t) {
  if (t == kBool)
    t = kByte;
  return mps::scalarToMetalTypeString(t);
}

// ── gather ───────────────────────────────────────────────────────────────────

TORCH_IMPL_FUNC(gather_out_mps)
(const Tensor& self_arg,
 int64_t dim,
 const Tensor& index,
 bool sparse_grad,
 const Tensor& output) {
  using namespace mps;
  if (self_arg.numel() == 0 || index.numel() == 0) {
    return;
  }
  auto self = self_arg.dim() == 0 ? self_arg.view({1}) : self_arg;
  auto idx = index.dim() == 0 ? index.view({1}) : index;
  auto out = output.dim() == 0 ? const_cast<Tensor&>(output).view({1}) : output;
  dim = at::maybe_wrap_dim(dim, self.dim());
  TORCH_CHECK(!sparse_grad, "sparse_grad not supported in MPS yet");
  TORCH_CHECK(
      self.scalar_type() == output.scalar_type(),
      "gather(): self and output must have the same scalar type");
  TORCH_CHECK(
      dim >= 0 && dim < self.dim(),
      "gather(): Indexing dim ",
      dim,
      " is out of bounds of tensor");

  if (self.is_complex()) {
    auto self_real = at::view_as_real(self);
    auto index_expanded = expand_index_as_real(index);
    auto output_real = at::view_as_real(maybe_expand_0_dim(output));
    structured_gather_out_mps::impl(
        self_real, dim, index_expanded, sparse_grad, output_real);
    return;
  }

  // For gather: src_strides/sizes hold output strides/sizes (repurposed)
  ScatterGatherParams<> params;
  params.ndim = static_cast<int32_t>(self.dim());
  params.dim = static_cast<int32_t>(dim);
  for (int32_t i = 0; i < params.ndim; i++) {
    params.self_strides[i] = static_cast<uint32_t>(self.stride(i));
    params.self_sizes[i] = static_cast<uint32_t>(self.size(i));
    params.src_strides[i] = static_cast<uint32_t>(out.stride(i));
    params.src_sizes[i] = static_cast<uint32_t>(out.size(i));
    params.index_strides[i] = static_cast<uint32_t>(idx.stride(i));
    params.index_sizes[i] = static_cast<uint32_t>(idx.size(i));
  }

  auto kernel_name = std::string("gather_") +
      metal_type_string(self.scalar_type()) + "_" +
      metal_type_string(index.scalar_type());

  @autoreleasepool {
    
    auto cplState = lib.getPipelineStateForFunc(kernel_name);
    getMPSProfiler().beginProfileKernel(cplState, kernel_name, {out, self, idx});
    id<MTLComputeCommandEncoder> computeEncoder =
        getCurrentMPSStream()->commandEncoder();
    [computeEncoder setComputePipelineState:cplState];
    mtl_setBuffer(computeEncoder, out, 0);
    mtl_setBuffer(computeEncoder, self, 1);
    mtl_setBuffer(computeEncoder, idx, 2);
    [computeEncoder setBytes:&params length:sizeof(params) atIndex:3];
    mtl_dispatch1DJob(computeEncoder, cplState, idx.numel());
    getMPSProfiler().endProfileKernel(cplState);
  }
}

// ── scatter (all modes) ──────────────────────────────────────────────────────

static void scatter_mps_general(
    const Tensor& self_arg,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    const Tensor& output,
    const std::string& /*func_name*/,
    const std::string_view reduce) {
  using namespace mps;
  if (!self_arg.is_same(output)) {
    const_cast<Tensor&>(output).copy_(self_arg);
  }
  if (self_arg.numel() == 0 || index.numel() == 0 || src.numel() == 0) {
    return;
  }
  auto self = self_arg.dim() == 0 ? self_arg.view({1}) : self_arg;
  auto idx = index.dim() == 0 ? index.view({1}) : index;
  auto source = src.dim() == 0 ? src.view({1}) : src;
  auto out = output.dim() == 0 ? const_cast<Tensor&>(output).view({1}) : output;
  dim = at::maybe_wrap_dim(dim, self.dim());

  TORCH_CHECK(
      self.scalar_type() == out.scalar_type() &&
          out.scalar_type() == source.scalar_type(),
      "scatter(): self, src and output must have the same scalar type");
  TORCH_CHECK(
      dim >= 0 && dim < self.dim(),
      "scatter(): Indexing dim ",
      dim,
      " is out of bounds of tensor");

  if (self.is_complex()) {
    auto self_real = at::view_as_real(self);
    auto index_expanded = expand_index_as_real(index);
    auto src_real = at::view_as_real(maybe_expand_0_dim(src));
    auto output_real = at::view_as_real(maybe_expand_0_dim(output));
    scatter_mps_general(
        self_real, dim, index_expanded, src_real, output_real, "", reduce);
    return;
  }

  std::string mode;
  if (reduce == "set")
    mode = "set";
  else if (reduce == "add" || reduce == "sum")
    mode = "add";
  else if (reduce == "prod" || reduce == "multiply")
    mode = "prod";
  else if (reduce == "amax")
    mode = "amax";
  else if (reduce == "amin")
    mode = "amin";
  else
    TORCH_CHECK(false, "scatter(): Unsupported reduce mode '", reduce, "'");

  auto kernel_name = std::string("scatter_") + mode + "_" +
      metal_type_string(out.scalar_type()) + "_" +
      metal_type_string(idx.scalar_type());

  auto params = build_scatter_params(out, dim, idx, source);

  @autoreleasepool {
    
    auto cplState = lib.getPipelineStateForFunc(kernel_name);
    getMPSProfiler().beginProfileKernel(
        cplState, kernel_name, {out, idx, source});
    id<MTLComputeCommandEncoder> computeEncoder =
        getCurrentMPSStream()->commandEncoder();
    [computeEncoder setComputePipelineState:cplState];
    mtl_setBuffer(computeEncoder, out, 0);
    mtl_setBuffer(computeEncoder, idx, 1);
    mtl_setBuffer(computeEncoder, source, 2);
    [computeEncoder setBytes:&params length:sizeof(params) atIndex:3];
    mtl_dispatch1DJob(computeEncoder, cplState, idx.numel());
    getMPSProfiler().endProfileKernel(cplState);
  }
}

TORCH_IMPL_FUNC(scatter_src_out_mps)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Tensor& src,
 const Tensor& output) {
  scatter_mps_general(self, dim, index, src, output, "", "set");
}

TORCH_IMPL_FUNC(scatter_value_out_mps)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Scalar& value,
 const Tensor& output) {
  Tensor src = at::empty(
      index.sizes(),
      self.scalar_type(),
      std::nullopt,
      kMPS,
      std::nullopt,
      self.suggest_memory_format());
  src.fill_(value);
  scatter_mps_general(self, dim, index, src, output, "", "set");
}

TORCH_IMPL_FUNC(scatter_reduce_out_mps)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Tensor& src,
 const std::string_view reduce,
 const Tensor& output) {
  TORCH_CHECK(
      reduce != "mean",
      "scatter_reduce(): 'mean' reduce mode not yet supported in MPS");
  scatter_mps_general(self, dim, index, src, output, "", reduce);
}

TORCH_IMPL_FUNC(scatter_value_reduce_out_mps)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Scalar& value,
 const std::string_view reduce,
 const Tensor& output) {
  Tensor src = at::empty(
      index.sizes(),
      self.scalar_type(),
      std::nullopt,
      kMPS,
      std::nullopt,
      self.suggest_memory_format());
  src.fill_(value);
  scatter_mps_general(self, dim, index, src, output, "", reduce);
}

TORCH_IMPL_FUNC(scatter_add_mps_out)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Tensor& src,
 const Tensor& output) {
  scatter_mps_general(self, dim, index, src, output, "", "add");
}

// ── scatter_reduce (two) — dispatch stub ─────────────────────────────────────

static void scatter_reduce_two_mps_kernel(
    const Tensor& self,
    const int64_t dim,
    const Tensor& index,
    const Tensor& src,
    const ReductionType& reduce) {
  switch (reduce) {
    case ReductionType::MEAN:
    case ReductionType::SUM:
      return scatter_mps_general(self, dim, index, src, self, "", "add");
    case ReductionType::PROD:
      return scatter_mps_general(self, dim, index, src, self, "", "prod");
    case ReductionType::MIN:
      return scatter_mps_general(self, dim, index, src, self, "", "amin");
    case ReductionType::MAX:
      return scatter_mps_general(self, dim, index, src, self, "", "amax");
  }
  TORCH_CHECK(
      false, "Unsupported reduction type: ", static_cast<int>(reduce));
}

REGISTER_DISPATCH(scatter_reduce_two_stub, &scatter_reduce_two_mps_kernel);

} // namespace at::native
