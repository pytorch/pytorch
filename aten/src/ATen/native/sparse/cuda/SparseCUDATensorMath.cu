#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/sparse/SparseUtils.h>
#include <ATen/native/sparse/cuda/SparseCUDAApplyUtils.cuh>
#include <ATen/native/sparse/cuda/SparseCUDABlas.cuh>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>

#include <THC/THCTensorMathPointwise.cuh>
#include <THC/THCThrustAllocator.cuh>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/system/cuda/execution_policy.h>

#define I_INFO(tensor) cuda::detail::getTensorInfo<int64_t, uint64_t>(tensor)
#define V_INFO(tensor) cuda::detail::getTensorInfo<scalar_t, uint64_t>(tensor)

namespace at { namespace native {

// --------------------------------------------------------------------
// Utility functions
// --------------------------------------------------------------------

namespace {
  IntTensor _to_csr_int(const LongTensor& rowIndices, int64_t dim, int64_t nnz) {
    IntTensor csr = at::empty({dim+1}, CUDA(kInt));
    IntTensor rowIndicesInt = at::empty({rowIndices.size(0)}, CUDA(kInt));
    rowIndicesInt.copy_(rowIndices);
    sparse::cuda::Xcoo2csr(rowIndicesInt.data<int32_t>(), nnz, dim, csr.data<int32_t>());
    return csr;
  }
}

// NB: Deleted spaddcmul (aka addcmul_, but not actually wired up), spaddcdiv (not
// wired at all)

// --------------------------------------------------------------------
// addmm(Tensor, SparseTensorRef, Tensor, Scalar, Scalar)  [broadcasts]
// --------------------------------------------------------------------

Tensor& s_addmm_out_sparse_dense_cuda(Tensor& r_, const Tensor& t, const SparseTensor& sparse_, const Tensor& dense, Scalar beta, Scalar alpha) {
  AT_ASSERT(t.is_cuda()); // dispatch argument
  AT_CHECK(r_.is_cuda(), "addmm: expected 'out' to be CUDA, but got CPU");
  AT_CHECK(sparse_.is_cuda(), "addmm: expected 'mat1' to be CUDA, but got CPU");
  AT_CHECK(dense.is_cuda(), "addmm: expected 'mat2' to be CUDA, but got CPU");

  AT_CHECK(_check_device({sparse_, r_, t, dense}));

  // TODO: This error message seems awfully opaque
  AT_CHECK(sparse_._sparseDims() == 2, "addmm: 2D tensor expected, got ", sparse_._sparseDims(), "D tensor");
  AT_CHECK(sparse_._denseDims() == 0, "addmm: scalar values expected, got ", sparse_._denseDims(), "D values");
  AT_CHECK(dense.dim() == 2, "addmm: 2D tensor expected, got ", dense.dim(), "D tensor");

  // mxk * kxn = mxn
  int64_t m = sparse_.size(0);
  int64_t k = sparse_.size(1);
  int64_t n = dense.size(1);

  AT_CHECK(t.size(0) == m,
      "addmm: Argument #1 (t): Expected dim 0 size ", m, ", got ", t.size(0));
  AT_CHECK(t.size(1) == n,
      "addmm: Argument #1 (t): Expected dim 1 size ", n, ", got ", t.size(1));
  AT_CHECK(dense.size(0) == k,
      "addmm: Argument #3 (dense): Expected dim 0 size ", k, ", got ", dense.size(0));

  r_.resize_({m, n});

  SparseTensor sparse = sparse_.coalesce();

  int64_t nnz = sparse._nnz();
  LongTensor indices = sparse._indices();
  Tensor values = sparse._values();

  LongTensor rowIndices = indices.select(0, 0);
  LongTensor colIndices = indices.select(0, 1);
  IntTensor csr = _to_csr_int(rowIndices, m, nnz);
  IntTensor colIndicesInt = at::empty({colIndices.size(0)}, indices.options().dtype(kInt));
  colIndicesInt.copy_(colIndices);

  // No half support, so we don't have to use CUDATypeConversion
  Tensor r__;
  AT_DISPATCH_FLOATING_TYPES(
      values.type(), "addmm_sparse_cuda", [&] {
        scalar_t cast_beta = beta.to<scalar_t>();
        scalar_t cast_alpha = alpha.to<scalar_t>();
        if (cast_beta == 0) {
          r_.zero_();
        } else if (cast_beta == 1) {
          if (!isSameTensor(t, r_)) {
            r_.copy_(t);
          }
        } else {
          at::mul_out(r_, t, scalar_to_tensor(beta));
        }

        /* r_ */
        if(r_.stride(0) == 1 && r_.stride(1) == r_.size(0)) {
          r__ = r_;
        } else {
          // TODO: how... strange
          r__ = r_.transpose(0, 1).clone();
          r__.transpose_(0, 1);
        }

        if (nnz > 0) {
          /* dense */
          Tensor dense_;
          char transpose_dense;
          if(dense.stride(0) == 1 && dense.stride(1) == dense.size(0)) {
            transpose_dense = 'n';
            dense_ = dense;
          } else if(dense.stride(1) == 1 && dense.stride(0) != dense.size(1)) {
            transpose_dense = 't';
            dense_ = dense;
          } else {
            transpose_dense = 't';
            dense_ = dense.contiguous();
          }

          sparse::cuda::csrmm2(
            'n',
            transpose_dense,
            m,
            n,
            k,
            nnz,
            cast_alpha,
            values.data<scalar_t>(),
            csr.data<int32_t>(),
            colIndicesInt.data<int32_t>(),
            dense_.data<scalar_t>(),
            (transpose_dense == 'n' ? dense_.stride(1) : dense_.stride(0)),
            cast_beta,
            r__.data<scalar_t>(),
            r__.stride(1));
        }
      });

  r_.copy_(r__);
  return r_;
}

Tensor s_addmm_sparse_dense_cuda(
    const Tensor& t,
    const SparseTensor& sparse,
    const Tensor& dense,
    Scalar beta,
    Scalar alpha
) {
  Tensor r = at::empty({0}, t.options());
  s_addmm_out_sparse_dense_cuda(r, t, sparse, dense, beta, alpha);
  return r;
}

Tensor& s_addmm_sparse_dense_cuda_(
    Tensor& t,
    const SparseTensor& sparse,
    const Tensor& dense,
    Scalar beta,
    Scalar alpha
) {
  return s_addmm_out_sparse_dense_cuda(t, t, sparse, dense, beta, alpha);
}

// Deleted sspaddmm (sparse, dense) -> sparse

// --------------------------------------------------------------------
// hspmm(SparseTensor mat1, Tensor mat2)
// --------------------------------------------------------------------

SparseTensor& hspmm_out_sparse_cuda(SparseTensor& r_, const SparseTensor& sparse_, const Tensor& dense/* , Scalar alpha */) {
  AT_ASSERT(sparse_.is_cuda()); // dispatch argument
  AT_CHECK(r_.is_cuda(), "hspmm: expected 'out' to be CUDA, but got CPU");
  AT_CHECK(dense.is_cuda(), "hspmm: expected 'mat2' to be CUDA, but got CPU");

  AT_CHECK(_check_device({r_, sparse_, dense}));

  AT_CHECK(sparse_._sparseDims() == 2,
      "hspmm: Argument #2: 2D tensor expected, got ", sparse_._sparseDims(), "D tensor");
  AT_CHECK(sparse_._denseDims() == 0,
      "hspmm: Argument #2: scalar values expected, got ", sparse_._denseDims(), "D values");
  AT_CHECK(dense.dim() == 2,
      "hspmm: Argument #3: 2D tensor expected, got ", dense.dim(), "D tensor");

  int64_t m = sparse_.size(0);
  int64_t k = sparse_.size(1);
  int64_t n = dense.size(1);

  AT_CHECK(dense.size(0) == k,
      "hspmm: Argument #3: Expected dim 0 size ", k, ", got ", dense.size(0));

  _get_sparse_impl(r_)->resize_and_clear_(1, 1, {m, n});

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);

  SparseTensor sparse = sparse_.coalesce();

  int64_t nnz = sparse._nnz();

  LongTensor indices = at::empty({1, nnz}, CUDA(kLong));
  // create values in column-major format to avoid copying in spaddmm
  Tensor values = at::empty({n, nnz}, dense.options());
  values.transpose_(0, 1);

  // why does sparse need to be cloned? If this is really necessary maybe we
  // need to fuse this with newCoalesce
  SparseTensor newSparse = sparse.clone();
  LongTensor spIndices = newSparse._indices();
  LongTensor dstIndices = spIndices.select(0, 0);
  // Save destination indices to output hybrid tensor
  indices.copy_(dstIndices);
  // Replace destination indices with 0, 1, 2, 3, ... and compute output values
  // tensor with sparse * dense multiplication
  thrust::device_ptr<int64_t> indicesIter(dstIndices.data<int64_t>());
  thrust::sequence(policy, indicesIter, indicesIter + nnz);

  std::vector<int64_t> new_size = _get_sparse_impl(newSparse)->sizes().vec();
  new_size[0] = nnz;
  _get_sparse_impl(newSparse)->raw_resize_(_get_sparse_impl(newSparse)->sparseDims(), _get_sparse_impl(newSparse)->denseDims(), new_size);

  s_addmm_out_sparse_dense_cuda(values, values, newSparse, dense, 0, /*alpha*/ 1);
  _get_sparse_impl(r_)->set_indices_and_values_unsafe(indices, values);

  return r_;
}

SparseTensor hspmm_sparse_cuda(const SparseTensor& sparse, const Tensor& dense) {
  SparseTensor r = at::empty({0}, sparse.options());
  hspmm_out_sparse_cuda(r, sparse, dense);
  return r;
}

// --------------------------------------------------------------------
// add(Tensor, SparseTensorRef, Scalar)
//    formerly known as spcadd
// --------------------------------------------------------------------

Tensor& add_out_dense_sparse_cuda(Tensor& r_, const Tensor& dense, SparseTensorRef sparse_, at::Scalar value) {
  const SparseTensor& sparse = sparse_.tref;

  AT_ASSERT(dense.is_cuda()); // dispatch argument
  AT_CHECK(sparse.is_cuda(), "add: expected 'other' to be CUDA, but got CPU");
  AT_CHECK(r_.is_cuda(), "add: expected 'out' to be CUDA, but got CPU");

  AT_CHECK(_check_device({sparse, r_, dense}));

  AT_CHECK(dense.sizes().equals(sparse.sizes()), "add: expected 'self' and 'other' to have same size, but self has size ",
    dense.sizes(), " while other has size ", sparse.sizes(), " (FYI: dense-sparse addition does not currently support broadcasting)");

  const int64_t nnz = sparse._nnz();
  if (nnz == 0) {
    r_.resize_as_(dense);
    r_.copy_(dense);
    return r_;
  }

  Tensor r = r_;
  if (!isSameTensor(r, dense)) {
    r_.resize_as_(dense);
    r_.copy_(dense);
  } else {
    AT_CHECK(r_.is_contiguous(), "add: CUDA dense-sparse addition with a non-contiguous output tensor does not work; shout if you need it (see https://github.com/pytorch/pytorch/issues/1521 )");
    r = r_.contiguous();
  }

  LongTensor indices = sparse._indices();
  Tensor values = sparse._values();
  int64_t nDim = dense.dim();
  int64_t nDimI = sparse._sparseDims();

  if (sparse._values().numel() == 0) {
    return r_;
  }

  if (sparse.is_coalesced()) {
    // TODO benchmark to decide whether to remove this special case
    const dim3 block = cuda::getApplyBlock();
    dim3 grid;
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);
    if (sparse._denseDims() == 0) {
      AT_CHECK(cuda::getApplyGrid(nnz, grid, curDevice), "add: Argument #0: tensor too large or too many dimensions");

      AT_DISPATCH_ALL_TYPES_AND_HALF(
          values.type(), "add_out_dense_sparse_cuda", [&] {
            apply::sparseElementwiseKernelScalar<TensorCAddOp<scalar_t>, uint64_t, scalar_t>
              <<<grid, block, 0, stream>>>(
                TensorCAddOp<scalar_t>(value.to<scalar_t>()),
                V_INFO(r_), I_INFO(indices), V_INFO(values),
                static_cast<uint64_t>(nnz));
          });
    } else {
      AT_CHECK(cuda::getApplyGrid(nnz * block.x, grid, curDevice), "add: Argument #0: tensor too large or too many dimensions");

      AT_DISPATCH_ALL_TYPES_AND_HALF(
          values.type(), "add_out_dense_sparse_cuda", [&] {
            apply::sparseElementwiseKernel<TensorCAddOp<scalar_t>, uint64_t, scalar_t>
              <<<grid, block, 0, stream>>>(
                TensorCAddOp<scalar_t>(value.to<scalar_t>()),
                V_INFO(r_), I_INFO(indices), V_INFO(values),
                static_cast<uint64_t>(nnz));
          });
    }
  } else {
    LongTensor indices1D = _newFlattenedIndices(sparse, 0).squeeze_(0).narrow(0, 0, nnz);

    // FIXME: at some point we can wrap the scale into indexAdd
    // NB: Purposely not inplace!
    AT_DISPATCH_ALL_TYPES_AND_HALF(
        values.type(), "add_out_dense_sparse_cuda", [&] {
          if (value.to<scalar_t>() != static_cast<scalar_t>(1)) {
            values = values.mul(value);
          }
        });

    int64_t view_rows = 1;
    int64_t view_columns = 1;
    for (int i = 0; i < nDimI; i++) {
      view_rows *= r.size(i);
    }
    for (int i = nDimI; i < nDim; i++) {
      view_columns *= r.size(i);
    }

    Tensor r_view = r.view({view_rows, view_columns});
    values = values.narrow(0, 0, nnz).reshape({nnz, view_columns});
    r_view.index_add_(0, indices1D, values);
  }
  THCudaCheck(cudaGetLastError());

  return r_;
}

// --------------------------------------------------------------------
// add(SparseTensor, SparseTensor, Scalar)  [broadcasts]
// --------------------------------------------------------------------

SparseTensor& add_out_sparse_cuda(SparseTensor& r_, const SparseTensor& t, const SparseTensor& src, Scalar value) {
  AT_ASSERT(t.is_cuda()); // dispatch argument
  AT_CHECK(src.is_cuda(), "add: expected 'other' to be CUDA, but got CPU");
  AT_CHECK(r_.is_cuda(), "add: expected 'out' to be CUDA, but got CPU");

  AT_CHECK(_check_device({r_, t, src}));
  AT_CHECK(t.sizes().equals(src.sizes()), "add: expected 'self' and 'other' to have same size, but ", t.sizes(), " != ", src.sizes());

  if (src._nnz() == 0) {
    return copy_sparse_to_sparse_(r_, t);
  }
  if (t._nnz() == 0) {
    return mul_out_sparse_scalar(r_, src, value);
  }

  AT_CHECK(_is_same_density(t, src), "add: expected 'self' and 'other' to have same density, but 'self' has ", t._sparseDims(), " sparse dimensions while 'other' has ", src._sparseDims(), " sparse dimensions");

  // We deliberately choose to simply concat the indices and values tensors
  // rather than merging them. This removes the need to synchronously fetch nnz
  // at the end of the operation, at the cost of having a non-coalesced result.
  // This trade-off is preferable for the common use-case of gradient accumulation.
  LongTensor t_indices_ = t._indices();
  Tensor t_values_ = t._values();
  LongTensor s_indices_ = src._indices();
  Tensor s_values_ = src._values();

  AT_DISPATCH_ALL_TYPES_AND_HALF(
      s_values_.type(), "add_out_sparse_cuda", [&] {
        if (value.to<scalar_t>() != static_cast<scalar_t>(1)) {
          s_values_ = s_values_.mul(value);
        }
      });

  LongTensor r_indices_ = at::cat({t_indices_, s_indices_}, 1);
  Tensor r_values_ = at::cat({t_values_, s_values_}, 0);
  r_.resize_as_(src);
  _alias_into_sparse(r_, r_indices_, r_values_);

  // FIXME: add some heuristic about when to call coalesce() here, so that
  // tensors don't totally blow up in size by concatenation; e.g.
  //   r->minUnique = max(a->minUnique + b->minUnique);
  //   if (r->nnz / r->minUnique > COMPACTION_THRESHOLD) {
  //     THCSTensor_(contiguous)(r);
  //     r->minUnique = r->nnz;
  //   }

  return r_;
}

// --------------------------------------------------------------------
// mul(SparseTensor, SparseTensor)  [broadcasts]
// --------------------------------------------------------------------

SparseTensor& mul_out_sparse_cuda(SparseTensor& r_, const SparseTensor& t_, const SparseTensor& src_) {
  if (src_.dim() == 0) {
    return mul_out_sparse_zerodim(r_, t_, src_);
  } else if (t_.dim() == 0) {
    return mul_out_sparse_zerodim(r_, src_, t_);
  }

  AT_ASSERT(t_.is_cuda()); // dispatch argument
  AT_CHECK(src_.is_cuda(), "mul: expected 'other' to be CUDA, but got CPU");
  AT_CHECK(r_.is_cuda(), "mul: expected 'out' to be CUDA, but got CPU");
  AT_CHECK(_check_device({r_, t_, src_}));
  AT_CHECK(t_.sizes().equals(src_.sizes()), "mul: expected 'self' and 'other' to have same size, but ", t_.sizes(), " != ", src_.sizes());

  SparseTensor t = t_.coalesce();
  SparseTensor src = src_.coalesce();

  if (src_._nnz() == 0 || t_._nnz() == 0) {
    r_.resize_as_(src_);
    return r_.zero_();
  }

  // saving those because they can be overwritten when doing in-place operations
  int64_t t_nnz = t._nnz(), s_nnz = src._nnz();
  int64_t max_nnz = std::min(t_nnz, s_nnz);  // multiply by zero is zero, and can be dropped
  int64_t sparseDims = src._sparseDims();
  LongTensor t_indices_ = t._indices();
  Tensor t_values_ = t._values();
  LongTensor s_indices_ = src._indices();
  Tensor s_values_ = src._values();
  LongTensor r_indices_ = at::empty({sparseDims, max_nnz}, t_indices_.options());
  Tensor r_values_ = _new_values_with_size_of(t_values_, max_nnz).zero_();
  r_.resize_as_(src);
  _get_sparse_impl(r_)->set_indices_and_values_unsafe(r_indices_, r_values_);

  int64_t valueSize = t_values_.stride(0);
  const dim3 block = dim3(std::min(static_cast<int64_t>(cuda::getApplyBlock().x), valueSize));
  dim3 grid;
  int curDevice = -1;
  cudaGetDevice(&curDevice);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);
  AT_CHECK(cuda::getApplyGrid(valueSize, grid, curDevice), "mul: Argument #0: tensor too large or too many dimensions");

  LongTensor resultNnz = at::empty({1}, CUDA(kLong));
  AT_DISPATCH_ALL_TYPES_AND_HALF(
      t_values_.type(), "mul_out_sparse_cuda", [&] {
        apply::valueSparseIntersectionKernel<TensorMulOp<scalar_t>, uint64_t, scalar_t>
          <<<grid, block, 0, stream>>>(
            TensorMulOp<scalar_t>(),
            I_INFO(r_indices_), I_INFO(t_indices_), I_INFO(s_indices_),
            V_INFO(r_values_), V_INFO(t_values_), V_INFO(s_values_),
            static_cast<uint64_t>(t_nnz), static_cast<uint64_t>(s_nnz));
        THCudaCheck(cudaGetLastError());

        apply::indexSparseIntersectionKernel<uint64_t, scalar_t>
          <<<1, 1, 0, stream>>>(
            I_INFO(r_indices_), I_INFO(t_indices_), I_INFO(s_indices_),
            // reinterpret_cast shenanigans, because we don't actually have
            // unsigned tensors...
            static_cast<uint64_t>(t_nnz), static_cast<uint64_t>(s_nnz), reinterpret_cast<uint64_t*>(resultNnz.data_ptr()));
        THCudaCheck(cudaGetLastError());
      });

  // sync!  (surely there is a more idiomatic way to do this...)
  LongTensor cpu_resultNnz = at::empty({1}, CPU(kLong));
  cpu_resultNnz.copy_(resultNnz);
  _get_sparse_impl(r_)->set_nnz_and_narrow(cpu_resultNnz.accessor<int64_t, 1>()[0]);
  _get_sparse_impl(r_)->set_coalesced(true);

  return r_;
}

}} // namespace at::native
