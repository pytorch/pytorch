#include "ATen/ATen.h"
#include "ATen/TensorUtils.h"
#include "ATen/NativeFunctions.h"

#include <cstring>
#include <memory>
#include <sstream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif


namespace at { 
namespace native {

// Staying faithful to the Python for now for clarity, look for optimizations later
// (eg single return statement for RVO)
Tensor norm_except_dim(const Tensor & v, int64_t pow, int64_t dim)
{
  // I assume tensor.contiguous(), view(), norm(), etc. here will dispatch through VariableType.
  if(dim == -1)
    return p.norm(pow);
  else if(dim == 0)
  {
    std::vector<int64_t> output_size(v.dim(), 1);
    output_size[0] = v.size(0);
    return v.contiguous().view(v.size(0), -1).norm(pow, 1).view(output_size);
  }
  else if(dim == v.dim() - 1)
  {
    std::vector<int64_t> output_size(v.dim(), 1);
    output_size[v.dim() - 1] = v.size(v.dim() - 1);
    return v.contiguous().view(-1, v.size(v.dim() - 1)).norm(pow, 0).view(output_size);
  }
  else 
    return norm_except_dim(v.transpose(0, dim), 0).transpose(0, dim); // optimize?
}

Tensor weight_norm
  (const Tensor & v_in, 
   const Tensor & g_in,
   int64_t dim) 
{

  AT_CHECK
    (v_in.type().is_cuda() == g_in.type().is_cuda(),
     "In weight_norm, v and g must both be on CPU, or both be on GPU") 

  auto v = v_in.contiguous();
  auto g = g_in.contiguous();
    
  bool can_use_fused = v.type().is_cuda() && (dim == 0 || dim == v.dim() - 1);

  if(can_use_fused) 
  {
    // weight_norm does not have a derivative defined for it, so this will route back through
    // VariableType.cpp, and construct a WeightNormFusedBackward object in the autograd graph.
    return std::get<0>(at::weight_norm_fused(v, g, dim));
  }
  else
  {
    // Double-differentiable primitive ops
    // at::native::norm_except_dim would probably be fine as well.
    return v*(g/at::norm_except_dim(v, 2, dim));
  }
}

} // namespace native
} // namespace at
