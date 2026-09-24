#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/Eig.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_linalg_check_errors.h>
#include <ATen/ops/_linalg_eigvals_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/linalg_eig.h>
#include <ATen/ops/linalg_eig_native.h>
#include <ATen/ops/linalg_eigvals.h>
#include <ATen/ops/linalg_eigvals_native.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {
namespace mps {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Eig_metallib.h>
#endif

// Beyond kEigMaxDim the scratch matrices no longer fit in threadgroup
// memory, and float64 is not a Metal dtype at all. Both cases go to CPU, the
// same way the MPS eigh and svd kernels handle inputs outside their range.
static bool eig_needs_cpu(const Tensor& input, int64_t n) {
  const auto dtype = input.scalar_type();
  return n > kEigMaxDim || (dtype != kFloat && dtype != kComplexFloat);
}

static void eig_mps_impl(const Tensor& input, const Tensor& values, const Tensor& vectors, bool compute_vectors) {
  const auto n = input.size(-1);
  const auto batch = batchCount(input);
  if (n == 0 || batch == 0) {
    return;
  }

  if (eig_needs_cpu(input, n)) {
    if (n > kEigMaxDim) {
      TORCH_WARN_ONCE("linalg.eig: matrices larger than ",
                      kEigMaxDim,
                      "x",
                      kEigMaxDim,
                      " do not fit in MPS threadgroup memory; falling back to CPU.");
    }
    if (compute_vectors) {
      const auto [values_cpu, vectors_cpu] = at::linalg_eig(input.cpu());
      values.copy_(values_cpu);
      vectors.copy_(vectors_cpu);
    } else {
      values.copy_(at::linalg_eigvals(input.cpu()));
    }
    return;
  }

  const auto complex_dtype = toComplexType(input.scalar_type());
  const auto A = input.to(complex_dtype).contiguous();
  // The kernel writes dense complex64, so only a strided or differently typed
  // out= tensor needs a staging buffer.
  auto writable = [&](const Tensor& out) {
    return out.is_contiguous() && out.scalar_type() == complex_dtype
        ? out
        : at::empty(out.sizes(), out.options().dtype(complex_dtype));
  };
  const auto values_out = writable(values);
  const auto vectors_out = compute_vectors ? std::optional<Tensor>(writable(vectors)) : std::nullopt;
  auto infos = at::zeros({batch}, input.options().dtype(kInt));

  const EigParams params{
      .n = static_cast<int32_t>(n),
      .compute_vectors = compute_vectors,
  };

  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = stream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc("eig_qr");
      getMPSProfiler().beginProfileKernel(pso, "eig_qr", {input}, stream);
      [computeEncoder setComputePipelineState:pso];
      mtl_setArgs(computeEncoder, A, values_out, vectors_out, infos, params);
      // One SIMD group per matrix.
      [computeEncoder dispatchThreadgroups:MTLSizeMake(batch, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(kEigMaxDim, 1, 1)];
      getMPSProfiler().endProfileKernel(pso, stream);
    }
  });

  at::_linalg_check_errors(infos, "torch.linalg.eig", input.dim() == 2);

  if (!values_out.is_same(values)) {
    values.copy_(values_out);
  }
  if (vectors_out && !vectors_out->is_same(vectors)) {
    vectors.copy_(*vectors_out);
  }
}

} // namespace mps

std::tuple<Tensor&, Tensor&> linalg_eig_out_mps(const Tensor& input, Tensor& values, Tensor& vectors) {
  squareCheckInputs(input, "linalg.eig");
  const auto complex_dtype = toComplexType(input.scalar_type());
  checkLinalgCompatibleDtype("torch.linalg.eig", values.scalar_type(), complex_dtype, "eigenvalues");
  checkLinalgCompatibleDtype("torch.linalg.eig", vectors.scalar_type(), complex_dtype, "eigenvectors");
  checkSameDevice("torch.linalg.eig", values, input, "eigenvalues");
  checkSameDevice("torch.linalg.eig", vectors, input, "eigenvectors");

  at::native::resize_output(values, IntArrayRef(input.sizes().data(), input.dim() - 1));
  at::native::resize_output(vectors, input.sizes());
  mps::eig_mps_impl(input, values, vectors, /*compute_vectors=*/true);
  return std::tuple<Tensor&, Tensor&>(values, vectors);
}

std::tuple<Tensor, Tensor> linalg_eig_mps(const Tensor& input) {
  const auto complex_dtype = toComplexType(input.scalar_type());
  Tensor values = at::empty({0}, input.options().dtype(complex_dtype));
  Tensor vectors = at::empty({0}, input.options().dtype(complex_dtype));
  linalg_eig_out_mps(input, values, vectors);
  return std::tuple<Tensor, Tensor>(std::move(values), std::move(vectors));
}

Tensor& linalg_eigvals_out_mps(const Tensor& input, Tensor& values) {
  squareCheckInputs(input, "linalg.eigvals");
  const auto complex_dtype = toComplexType(input.scalar_type());
  checkLinalgCompatibleDtype("torch.linalg.eigvals", values.scalar_type(), complex_dtype, "eigenvalues");
  checkSameDevice("torch.linalg.eigvals", values, input, "eigenvalues");

  at::native::resize_output(values, IntArrayRef(input.sizes().data(), input.dim() - 1));
  Tensor vectors = at::empty({0}, input.options().dtype(complex_dtype));
  mps::eig_mps_impl(input, values, vectors, /*compute_vectors=*/false);
  return values;
}

Tensor _linalg_eigvals_mps(const Tensor& input) {
  const auto complex_dtype = toComplexType(input.scalar_type());
  Tensor values = at::empty({0}, input.options().dtype(complex_dtype));
  linalg_eigvals_out_mps(input, values);
  return values;
}

} // namespace at::native
