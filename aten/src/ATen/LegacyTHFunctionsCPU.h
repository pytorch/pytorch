#pragma once

#include <ATen/Context.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

namespace c10 {
class Scalar;
}
namespace at {
struct Generator;
class Tensor;
struct Type;
} // namespace at

namespace at {
namespace native {
namespace legacy {
namespace cpu {

Tensor & _th_masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source);
Tensor & _th_masked_scatter_bool_(Tensor & self, const Tensor & mask, const Tensor & source);
Tensor & _th_nonzero_out(Tensor & result, const Tensor & self);
Tensor _th_nonzero(const Tensor & self);
Tensor & _th_index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source);
Tensor & _th_put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate);
std::tuple<Tensor &,Tensor &> _th_mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim);
std::tuple<Tensor,Tensor> _th_mode(const Tensor & self, int64_t dim, bool keepdim);
Tensor _th_var(const Tensor & self, bool unbiased);
Tensor _th_std(const Tensor & self, bool unbiased);
Tensor & _th_renorm_out(Tensor & result, const Tensor & self, const Scalar& p, int64_t dim, const Scalar& maxnorm);
Tensor _th_renorm(const Tensor & self, const Scalar& p, int64_t dim, const Scalar& maxnorm);
Tensor & _th_renorm_(Tensor & self, const Scalar& p, int64_t dim, const Scalar& maxnorm);
Tensor & _th_histc_out(Tensor & result, const Tensor & self, int64_t bins, const Scalar& min, const Scalar& max);
Tensor _th_histc(const Tensor & self, int64_t bins, const Scalar& min, const Scalar& max);
std::tuple<Tensor &,Tensor &> _th_gels_out(Tensor & res1, Tensor & res2, const Tensor & self, const Tensor & A);
std::tuple<Tensor,Tensor> _th_gels(const Tensor & self, const Tensor & A);
std::tuple<Tensor &,Tensor &> _th_geqrf_out(Tensor & res1, Tensor & res2, const Tensor & self);
std::tuple<Tensor,Tensor> _th_geqrf(const Tensor & self);
Tensor & _th_ormqr_out(Tensor & result, const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose);
Tensor _th_ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose);

} // namespace th
} // namespace legacy
} // namespace native
} // namespace at
