#pragma once

#include <ATen/Config.h>

#if AT_USE_EIGEN_SPARSE()
#ifndef EIGEN_MPL2_ONLY
#define EIGEN_MPL2_ONLY
#endif

#include <ATen/Tensor.h>

namespace at::native::sparse::impl::eigen {

void addmm_out_sparse(
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Tensor& result,
    const at::Scalar& alpha,
    const at::Scalar& beta);

void add_out_sparse(
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& alpha,
    const at::Tensor& result);

} // namespace at::native::sparse::impl::eigen

#endif
