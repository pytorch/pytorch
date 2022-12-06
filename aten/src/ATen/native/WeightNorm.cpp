#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>
#include <ATen/TensorOperators.h>
#include <ATen/native/cpu/WeightNormKernel.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_weight_norm_differentiable_backward_native.h>
#include <ATen/ops/_weight_norm_interface.h>
#include <ATen/ops/_weight_norm_native.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/norm_except_dim.h>
#include <ATen/ops/norm_except_dim_native.h>
#endif

#include <vector>

namespace at {
namespace native {

DEFINE_DISPATCH(weight_norm_stub);
DEFINE_DISPATCH(weight_norm_backward_stub);

// Staying faithful to the Python for now for clarity, look for optimizations later
// (e.g., single return statement for RVO)
Tensor norm_except_dim(const Tensor & v, int64_t pow, int64_t dim)
{
  // I assume tensor.contiguous(), view(), norm(), etc. here will dispatch through VariableType.
  if (dim == -1) {
    return v.norm(pow);
  } else if (dim == 0) {
    std::vector<int64_t> output_size(v.dim(), 1);
    output_size[0] = v.size(0);
    return v.contiguous().view({v.size(0), -1}).norm(pow, 1).view(output_size);
  } else if (dim == v.dim() - 1) {
    std::vector<int64_t> output_size(v.dim(), 1);
    output_size[v.dim() - 1] = v.size(v.dim() - 1);
    return v.contiguous().view({-1, v.size(v.dim() - 1)}).norm(pow, 0).view(output_size);
  } else {
    // To consider: at::native::norm_except_dim is probably fine as well,
    // and would avoid an additional dynamic dispatch.
    return at::norm_except_dim(v.transpose(0, dim), pow, 0).transpose(0, dim); // optimize?
  }
}

std::tuple<Tensor,Tensor> weight_norm_cpu(
    const Tensor& v,
    const Tensor& g,
    int64_t dim) {
  auto w = at::empty_like(v, at::MemoryFormat::Contiguous);

  // align with cuda behavior, keep norm in 'Float' when g is 'BFloat16'
  const auto dtype = g.scalar_type() == at::ScalarType::BFloat16 ?
      at::ScalarType::Float : g.scalar_type();
  auto norm = at::empty_strided(g.sizes(), g.strides(), g.options().dtype(dtype));
  weight_norm_stub(kCPU, w, norm, v, g, dim);

  return std::tuple<Tensor, Tensor>{w, norm};
}

std::tuple<Tensor, Tensor> weight_norm_backward_cpu(
    const Tensor& grad_w,
    const Tensor& saved_v,
    const Tensor& saved_g,
    const Tensor& saved_norm,
    int64_t dim) {
  TORCH_CHECK(saved_v.is_contiguous(), "saved_v must be contiguous");
  TORCH_CHECK(saved_g.is_contiguous(), "saved_g must be contiguous");
  TORCH_CHECK(saved_norm.is_contiguous(), "saved_norm must be contiguous");

  auto grad_v = at::empty_like(saved_v, at::MemoryFormat::Contiguous);
  auto grad_g = at::empty_like(saved_g, at::MemoryFormat::Contiguous);
  weight_norm_backward_stub(kCPU, grad_v, grad_g, grad_w, saved_v, saved_g, saved_norm, dim);

  return std::tuple<Tensor, Tensor>{grad_v, grad_g};
}

Tensor _weight_norm
  (const Tensor & v_in,
   const Tensor & g_in,
   int64_t dim)
{

  TORCH_CHECK(
    v_in.device() == g_in.device(),
    "weight_norm: expected v_in and g_in to be on the same device, but v_in is "
    "on ", v_in.device(), " and g_in is on ", g_in.device());

  auto v = v_in.contiguous();
  auto g = g_in.contiguous();

  auto has_half_dtype = v.scalar_type() == at::ScalarType::Half
    || g.scalar_type() == at::ScalarType::Half;

  bool can_use_fused = !has_half_dtype && ((dim == 0) || (dim == v.dim() - 1));

  if (can_use_fused) {
    // weight_norm does not have a derivative defined for it, so this will route back through
    // VariableType.cpp, and construct a WeightNormFusedBackward object in the autograd graph.
    return std::get<0>(at::_weight_norm_interface(v, g, dim));
  } else {
    // Double-differentiable primitive ops
    // at::native::norm_except_dim would probably be fine as well.
    return v*(g/at::norm_except_dim(v, 2, dim));
  }
}

// Differentiable backward path, an alternative to weight_norm_backward, to be used
// when backward is itself creating a graph.
// The GradMode::is_enabled() check must be performed within Functions.cpp; that's why we
// define a separate function here, instead of inlining it in weight_norm_cuda_backward.
std::tuple<Tensor, Tensor> _weight_norm_differentiable_backward
  (const Tensor & grad_w,
   const Tensor & saved_v,
   const Tensor & saved_g,
   const Tensor & saved_norms,
   int64_t dim)
{
  // In Functions.cpp, the HardshrinkBackward object supplies "grad.contiguous()"
  // as the first argument, so grad_w should be contiguous here.
  // All these checks should succeed:
  TORCH_CHECK(grad_w.is_contiguous(), "grad_w must be contiguous");
  TORCH_CHECK(saved_v.is_contiguous(), "saved_v must be contiguous");
  TORCH_CHECK(saved_g.is_contiguous(), "saved_g must be contiguous");
  TORCH_CHECK(saved_norms.is_contiguous(), "saved_norms must be contiguous");

  int64_t last_dim = saved_v.dim() - 1;
  int64_t last_size = saved_v.size(last_dim);

  // Like weight_norm_fused_backward, weight_norm_differentiable_backward should only ever be called
  // through a WeightNormFusedBackward object, so we expect that dim == 0 || dim == saved_v.size(-1)
  TORCH_CHECK(dim == 0 || dim == last_dim, "Expected dim to be the first or last dimension");

  // saved_g and saved_norms are already shaped to broadcast over the correct dimensions

  // ...but saved_norms might be Float when saved_g and saved_v are half.
  // To consider:  saved_norms.to(..., True /*non_blocking*/);
  auto norms = saved_norms.to(saved_g.scalar_type());

  std::vector<int64_t> bcast_size(saved_v.dim(), 1);

  // Analytic backward path using differentiable primitive ops
  if (dim == 0) {
    bcast_size[0] = saved_v.size(0);
    auto per_dim_sums = (grad_w*saved_v).view({saved_v.size(0), -1}).sum(1).view(bcast_size);
    auto grad_v = (saved_g/norms)*(grad_w - saved_v*(per_dim_sums/(norms*norms)));
    auto grad_g = per_dim_sums/norms;
    return std::tuple<Tensor, Tensor>{grad_v, grad_g};
  } else { // dim == last_dim
    bcast_size[last_dim] = last_size;
    auto per_dim_sums = (grad_w*saved_v).view({-1, last_size}).sum(0).view(bcast_size);
    auto grad_v = (saved_g/norms)*(grad_w - saved_v*(per_dim_sums/(norms*norms)));
    auto grad_g = per_dim_sums/norms;
    return std::tuple<Tensor, Tensor>{grad_v, grad_g};
  }
}

} // namespace native
} // namespace at
