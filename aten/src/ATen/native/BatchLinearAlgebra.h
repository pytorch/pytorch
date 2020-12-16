#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

#include <TH/TH.h> // for USE_LAPACK


namespace at { namespace native {

#ifdef USE_LAPACK
// Define per-batch functions to be used in the implementation of batched
// linear algebra operations

template<class scalar_t>
void lapackEig(char jobvl, char jobvr, int n, scalar_t *a, int lda, scalar_t *wr, scalar_t *wi, scalar_t* vl, int ldvl, scalar_t *vr, int ldvr, scalar_t *work, int lwork, int *info);

#endif

using eig_fn = std::tuple<Tensor, Tensor> (*)(const Tensor&, bool&);

DECLARE_DISPATCH(eig_fn, eig_stub);

}} // namespace at::native
