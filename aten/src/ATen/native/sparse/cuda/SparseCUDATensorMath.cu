#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/NativeFunctions.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/native/sparse/SparseTensorMath.h>
#include <ATen/native/sparse/cuda/SparseCUDAApplyUtils.cuh>
#include <ATen/native/sparse/cuda/SparseCUDABlas.cuh>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/ExpandUtils.h>

#include <THC/THCTensorMathPointwise.cuh>
#include <THC/THCThrustAllocator.cuh>

#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/binary_search.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include <bitset>

#define I_INFO(tensor) cuda::detail::getTensorInfo<int64_t, uint64_t>(tensor)
#define V_INFO(tensor) cuda::detail::getTensorInfo<scalar_t, uint64_t>(tensor)

namespace at { namespace native {

using namespace at::sparse;
using at::cuda::detail::TensorInfo;
using at::cuda::detail::getTensorInfo;

// --------------------------------------------------------------------
// Utility functions
// --------------------------------------------------------------------

namespace {
  IntTensor _to_csr_int(const LongTensor& rowIndices, int64_t dim, int64_t nnz) {
    IntTensor csr = at::empty({dim+1}, CUDA(kInt));
    IntTensor rowIndicesInt = at::empty({rowIndices.size(0)}, CUDA(kInt));
    rowIndicesInt.copy_(rowIndices);
    sparse::cuda::Xcoo2csr(rowIndicesInt.data_ptr<int32_t>(), nnz, dim, csr.data_ptr<int32_t>());
    return csr;
  }
}

// NB: Deleted spaddcmul (aka addcmul_, but not actually wired up), spaddcdiv (not
// wired at all)

// --------------------------------------------------------------------
// addmm(Tensor, SparseTensor, Tensor, Scalar, Scalar)  [broadcasts]
// --------------------------------------------------------------------

Tensor& s_addmm_out_sparse_dense_cuda(Tensor& r_, const Tensor& t, const SparseTensor& sparse_, const Tensor& dense, Scalar beta, Scalar alpha) {
  TORCH_CHECK(t.is_cuda(), "addmm: expected 'self' to be CUDA, but got CPU");
  TORCH_CHECK(r_.is_cuda(), "addmm: expected 'out' to be CUDA, but got CPU");
  TORCH_CHECK(sparse_.is_cuda(), "addmm: expected 'mat1' to be CUDA, but got CPU");
  TORCH_CHECK(dense.is_cuda(), "addmm: expected 'mat2' to be CUDA, but got CPU");

  TORCH_CHECK(cuda::check_device({sparse_, r_, t, dense}));

  TORCH_CHECK(dense.dim() == 2, "addmm: 2D tensor expected, got ", dense.dim(), "D tensor");
  TORCH_CHECK(sparse_.sparse_dim() == 2, "addmm: expected first two dims to be sparse (indices has size 2 at first dim), but got ", sparse_.sparse_dim(), " spase dims");
  // no need to check dense_dim because dense_dim + sparse_dim = dim

  // mxk * kxn = mxn
  int64_t m = sparse_.size(0);
  int64_t k = sparse_.size(1);
  int64_t n = dense.size(1);

  TORCH_CHECK(t.size(0) == m,
      "addmm: Argument #1 (t): Expected dim 0 size ", m, ", got ", t.size(0));
  TORCH_CHECK(t.size(1) == n,
      "addmm: Argument #1 (t): Expected dim 1 size ", n, ", got ", t.size(1));
  TORCH_CHECK(dense.size(0) == k,
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
      values.scalar_type(), "addmm_sparse_cuda", [&] {
        scalar_t cast_beta = beta.to<scalar_t>();
        scalar_t cast_alpha = alpha.to<scalar_t>();
        if (cast_beta == 0) {
          r_.zero_();
        } else if (cast_beta == 1) {
          if (!is_same_tensor(t, r_)) {
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
          r__ = r_.transpose(0, 1).clone(at::MemoryFormat::Contiguous);
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
            values.data_ptr<scalar_t>(),
            csr.data_ptr<int32_t>(),
            colIndicesInt.data_ptr<int32_t>(),
            dense_.data_ptr<scalar_t>(),
            (transpose_dense == 'n' ? dense_.stride(1) : dense_.stride(0)),
            cast_beta,
            r__.data_ptr<scalar_t>(),
            r__.stride(1));
        }
      });

  r_.copy_(r__);
  return r_;
}

Tensor& addmm_out_sparse_dense_cuda(
    Tensor& result,
    const Tensor& self,
    const SparseTensor& mat1,
    const Tensor& mat2,
    Scalar beta,
    Scalar alpha
) {
  Tensor b_self;
  std::tie(b_self) = expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm_out");
  return s_addmm_out_sparse_dense_cuda(result, b_self, mat1, mat2, beta, alpha);
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

Tensor addmm_sparse_dense_cuda(
    const Tensor& self,
    const SparseTensor& mat1,
    const Tensor& mat2,
    Scalar beta,
    Scalar alpha
) {
  Tensor b_self;
  std::tie(b_self) = expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm_out");
  return s_addmm_sparse_dense_cuda(b_self, mat1, mat2, beta, alpha);
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

// NB: Purposely no broadcasting version of addmm inplace

// Deleted sspaddmm (sparse, dense) -> sparse

// --------------------------------------------------------------------
// hspmm(SparseTensor mat1, Tensor mat2)
// --------------------------------------------------------------------

SparseTensor& hspmm_out_sparse_cuda(SparseTensor& r_, const SparseTensor& sparse_, const Tensor& dense/* , Scalar alpha */) {
  TORCH_CHECK(sparse_.is_cuda(), "hspmm: expected 'self' to be CUDA, but got CPU");
  TORCH_CHECK(r_.is_cuda(), "hspmm: expected 'out' to be CUDA, but got CPU");
  TORCH_CHECK(dense.is_cuda(), "hspmm: expected 'mat2' to be CUDA, but got CPU");

  TORCH_CHECK(cuda::check_device({r_, sparse_, dense}));

  TORCH_CHECK(sparse_.sparse_dim() == 2,
      "hspmm: Argument #2: 2D tensor expected, got ", sparse_.sparse_dim(), "D tensor");
  TORCH_CHECK(sparse_.dense_dim() == 0,
      "hspmm: Argument #2: scalar values expected, got ", sparse_.dense_dim(), "D values");
  TORCH_CHECK(dense.dim() == 2,
      "hspmm: Argument #3: 2D tensor expected, got ", dense.dim(), "D tensor");

  int64_t m = sparse_.size(0);
  int64_t k = sparse_.size(1);
  int64_t n = dense.size(1);

  TORCH_CHECK(dense.size(0) == k,
      "hspmm: Argument #3: Expected dim 0 size ", k, ", got ", dense.size(0));

  get_sparse_impl(r_)->resize_and_clear_(1, 1, {m, n});

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
  thrust::device_ptr<int64_t> indicesIter(dstIndices.data_ptr<int64_t>());
  thrust::sequence(policy, indicesIter, indicesIter + nnz);

  std::vector<int64_t> new_size = get_sparse_impl(newSparse)->sizes().vec();
  new_size[0] = nnz;
  get_sparse_impl(newSparse)->raw_resize_(get_sparse_impl(newSparse)->sparse_dim(), get_sparse_impl(newSparse)->dense_dim(), new_size);

  s_addmm_out_sparse_dense_cuda(values, values, newSparse, dense, 0, /*alpha*/ 1);
  get_sparse_impl(r_)->set_indices_and_values_unsafe(indices, values);

  return r_;
}

SparseTensor hspmm_sparse_cuda(const SparseTensor& sparse, const Tensor& dense) {
  SparseTensor r = at::empty({0}, sparse.options());
  hspmm_out_sparse_cuda(r, sparse, dense);
  return r;
}

// --------------------------------------------------------------------
// add(Tensor, SparseTensor, Scalar)
//    formerly known as spcadd
// --------------------------------------------------------------------

Tensor& add_out_dense_sparse_cuda(Tensor& r_, const Tensor& dense, const SparseTensor& sparse, at::Scalar value) {
  TORCH_CHECK(dense.is_cuda(), "add: expected 'self' to be a CUDA tensor, but got a CPU tensor");
  TORCH_CHECK(sparse.is_cuda(), "add: expected 'other' to be a CUDA tensor, but got a CPU tensor");
  TORCH_CHECK(r_.is_cuda(), "add: expected 'out' to be a CUDA tensor, but got a CPU tensor");

  TORCH_CHECK(cuda::check_device({sparse, r_, dense}));

  TORCH_CHECK(dense.sizes().equals(sparse.sizes()), "add: expected 'self' and 'other' to have same size, but self has size ",
    dense.sizes(), " while other has size ", sparse.sizes(), " (FYI: dense-sparse addition does not currently support broadcasting)");

  const int64_t nnz = sparse._nnz();
  if (nnz == 0) {
    r_.resize_as_(dense);
    r_.copy_(dense);
    return r_;
  }

  Tensor r = r_;
  if (!is_same_tensor(r, dense)) {
    r_.resize_as_(dense);
    r_.copy_(dense);
  } else {
    TORCH_CHECK(r_.is_contiguous(), "add: CUDA dense-sparse addition with a non-contiguous output tensor does not work; shout if you need it (see https://github.com/pytorch/pytorch/issues/1521 )");
    r = r_.contiguous();
  }

  LongTensor indices = sparse._indices();
  Tensor values = sparse._values();
  int64_t nDim = dense.dim();
  int64_t nDimI = sparse.sparse_dim();

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
    if (sparse.dense_dim() == 0) {
      TORCH_CHECK(cuda::getApplyGrid(nnz, grid, curDevice), "add: Argument #0: tensor too large or too many dimensions");

      AT_DISPATCH_ALL_TYPES_AND(
        at::ScalarType::Half, values.scalar_type(), "add_out_dense_sparse_cuda", [&] {
            apply::sparseElementwiseKernelScalar<TensorCAddOp<scalar_t>, uint64_t, scalar_t>
              <<<grid, block, 0, stream>>>(
                TensorCAddOp<scalar_t>(value.to<scalar_t>()),
                V_INFO(r_), I_INFO(indices), V_INFO(values),
                static_cast<uint64_t>(nnz));
          });
    } else {
      TORCH_CHECK(cuda::getApplyGrid(nnz * block.x, grid, curDevice), "add: Argument #0: tensor too large or too many dimensions");

      // sparseElementwiseKernel needs values to be contiguous too
      values = values.contiguous();

      AT_DISPATCH_ALL_TYPES_AND(
        at::ScalarType::Half, values.scalar_type(), "add_out_dense_sparse_cuda", [&] {
            apply::sparseElementwiseKernel<TensorCAddOp<scalar_t>, uint64_t, scalar_t>
              <<<grid, block, 0, stream>>>(
                TensorCAddOp<scalar_t>(value.to<scalar_t>()),
                V_INFO(r_), I_INFO(indices), V_INFO(values),
                static_cast<uint64_t>(nnz));
          });
    }
  } else {
    LongTensor indices1D = flatten_indices(indices, sparse.sizes(), 0);

    // FIXME: at some point we can wrap the scale into indexAdd
    // NB: Purposely not inplace!
    AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Half, values.scalar_type(), "add_out_dense_sparse_cuda", [&] {
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
    values = values.reshape({nnz, view_columns});
    r_view.index_add_(0, indices1D, values);
  }
  THCudaCheck(cudaGetLastError());

  return r_;
}

// --------------------------------------------------------------------
// add(SparseTensor, SparseTensor, Scalar)  [broadcasts]
// --------------------------------------------------------------------

Tensor& add_out_dense_sparse_cuda(Tensor& r, const Tensor& dense, const SparseTensor& sparse_, Scalar value);

SparseTensor& add_out_sparse_cuda(SparseTensor& r_, const SparseTensor& t, const SparseTensor& src, Scalar value) {
  if (!t.is_sparse()) {
    return add_out_dense_sparse_cuda(r_, t, src, value);
  }

  // TODO: This test seems a bit goofy
  TORCH_CHECK(src.is_sparse(), "add(sparse, dense) is not supported. Use add(dense, sparse) instead.");

  TORCH_CHECK(t.is_cuda(), "add: expected 'self' to be CUDA, but got CPU");
  TORCH_CHECK(src.is_cuda(), "add: expected 'other' to be CUDA, but got CPU");
  TORCH_CHECK(r_.is_cuda(), "add: expected 'out' to be CUDA, but got CPU");

  TORCH_CHECK(cuda::check_device({r_, t, src}));
  TORCH_CHECK(t.sizes().equals(src.sizes()), "add: expected 'self' and 'other' to have same size, but ", t.sizes(), " != ", src.sizes());

  if (src._nnz() == 0) {
    return copy_sparse_to_sparse_(r_, t);
  }
  if (t._nnz() == 0) {
    return mul_out_sparse_scalar(r_, src, value);
  }

  TORCH_CHECK(is_same_density(t, src), "add: expected 'self' and 'other' to have same density, but 'self' has ", t.sparse_dim(), " sparse dimensions while 'other' has ", src.sparse_dim(), " sparse dimensions");

  // We deliberately choose to simply concat the indices and values tensors
  // rather than merging them. This removes the need to synchronously fetch nnz
  // at the end of the operation, at the cost of having a non-coalesced result.
  // This trade-off is preferable for the common use-case of gradient accumulation.
  LongTensor t_indices_ = t._indices();
  Tensor t_values_ = t._values();
  LongTensor s_indices_ = src._indices();
  Tensor s_values_ = src._values();

  AT_DISPATCH_ALL_TYPES_AND(
    at::ScalarType::Half, s_values_.scalar_type(), "add_out_sparse_cuda", [&] {
        if (value.to<scalar_t>() != static_cast<scalar_t>(1)) {
          s_values_ = s_values_.mul(value);
        }
      });

  LongTensor r_indices_ = at::cat({t_indices_, s_indices_}, 1);
  Tensor r_values_ = at::cat({t_values_, s_values_}, 0);
  r_.resize_as_(src);
  alias_into_sparse(r_, r_indices_, r_values_);

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

  TORCH_CHECK(t_.is_cuda(), "mul: expected 'self' to be CUDA, but got CPU");
  TORCH_CHECK(src_.is_cuda(), "mul: expected 'other' to be CUDA, but got CPU");
  TORCH_CHECK(r_.is_cuda(), "mul: expected 'out' to be CUDA, but got CPU");
  TORCH_CHECK(cuda::check_device({r_, t_, src_}));
  TORCH_CHECK(t_.sizes().equals(src_.sizes()), "mul: expected 'self' and 'other' to have same size, but ", t_.sizes(), " != ", src_.sizes());

  SparseTensor t = t_.coalesce();
  SparseTensor src = src_.coalesce();

  if (src_._nnz() == 0 || t_._nnz() == 0) {
    r_.resize_as_(src_);
    return r_.zero_();
  }

  // saving those because they can be overwritten when doing in-place operations
  int64_t t_nnz = t._nnz(), s_nnz = src._nnz();
  int64_t max_nnz = std::min(t_nnz, s_nnz);  // multiply by zero is zero, and can be dropped
  int64_t sparse_dim = src.sparse_dim();
  LongTensor t_indices_ = t._indices();
  Tensor t_values_ = t._values();
  LongTensor s_indices_ = src._indices();
  Tensor s_values_ = src._values();
  LongTensor r_indices_ = at::empty({sparse_dim, max_nnz}, t_indices_.options());
  Tensor r_values_ = new_values_with_size_of(t_values_, max_nnz).zero_();
  r_.resize_as_(src);
  get_sparse_impl(r_)->set_indices_and_values_unsafe(r_indices_, r_values_);

  int64_t valueSize = t_values_.stride(0);
  const dim3 block = dim3(std::min(static_cast<int64_t>(cuda::getApplyBlock().x), valueSize));
  dim3 grid;
  int curDevice = -1;
  cudaGetDevice(&curDevice);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);
  TORCH_CHECK(cuda::getApplyGrid(valueSize, grid, curDevice), "mul: Argument #0: tensor too large or too many dimensions");

  LongTensor resultNnz = at::empty({1}, CUDA(kLong));
  AT_DISPATCH_ALL_TYPES_AND(
    at::ScalarType::Half, t_values_.scalar_type(), "mul_out_sparse_cuda", [&] {
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
  get_sparse_impl(r_)->set_nnz_and_narrow(cpu_resultNnz.accessor<int64_t, 1>()[0]);

  return r_._coalesced_(true);
}

// --------------------------------------------------------------------
// sparse.sum() backward
//
// see NOTE [ sparse.sum() backward ]
// --------------------------------------------------------------------
template <typename scalar_t>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_1(512)
#endif
__global__ void _sparse_sum_backward_cuda_kernel(
  int64_t total_threads,
  const TensorInfo<int64_t, int64_t> grad_indices_ti,
  const TensorInfo<int64_t, int64_t> input_indices_ti,
  const TensorInfo<int64_t, int64_t> input_indices_pos_ti,
  const TensorInfo<scalar_t, int64_t> grad_values_expand_ti,
  TensorInfo<scalar_t, int64_t> grad_input_values_ti
) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total_threads) return;
  const int64_t j = input_indices_pos_ti.data[i];

  bool has_match = false;
  if (grad_indices_ti.data[j] == input_indices_ti.data[i]) {
    has_match = true;
  }

  int64_t grad_input_values_stride0 = grad_input_values_ti.strides[0];
  int64_t out_start = i * grad_input_values_stride0;
  int64_t out_end = (i + 1) * grad_input_values_stride0;
  int64_t in_start = j * grad_values_expand_ti.strides[0];

  if (has_match) {
    for (int64_t out_i = out_start, in_i = in_start; out_i < out_end; out_i++, in_i++) {
      grad_input_values_ti.data[out_i] = grad_values_expand_ti.data[in_i];
    }
  }
  else {
    for (int64_t out_i = out_start; out_i < out_end; out_i++) {
      grad_input_values_ti.data[out_i] = scalar_t(0);
    }
  }
}

Tensor _sparse_sum_backward_cuda(const Tensor& grad_, const SparseTensor& input_, IntArrayRef dims_to_sum) {
  TORCH_CHECK(grad_.is_cuda(), "_sparse_sum_backward_cuda: expected 'grad_' to be CUDA tensor, but got CPU tensor");
  TORCH_CHECK(input_.is_cuda(), "_sparse_sum_backward_cuda: expected 'input_' to be CUDA tensor, but got CPU tensor");

  auto input = input_.coalesce();
  const int64_t input_dim = input.dim();
  auto dims_to_sum_b = dim_list_to_bitset(dims_to_sum, input_dim);
  auto dims_to_sum_v = dims_to_sum.vec();
  maybe_wrap_dims(dims_to_sum_v, input_dim);

  LongTensor input_indices = input._indices();
  Tensor input_values = input._values();
  IntArrayRef input_sizes = input.sizes();
  const int64_t input_sparse_dim = input.sparse_dim();
  const int64_t input_dense_dim = input.dense_dim();
  const int64_t input_nnz = input._nnz();

  int64_t sparse_dims_to_sum_size = 0;
  auto sparse_dims_to_keep_v = std::vector<int64_t>();
  auto dense_dims_to_sum_v = std::vector<int64_t>();
  for (int64_t d = 0; d < input_dim; d++) {
    if (dims_to_sum_b[d]) {
      if (d < input_sparse_dim) sparse_dims_to_sum_size ++;
      else dense_dims_to_sum_v.emplace_back(d + 1 - input_sparse_dim);
    }
    else {
      if (d < input_sparse_dim) sparse_dims_to_keep_v.emplace_back(d);
    }
  }

  const bool sum_all_sparse_dim = (input_sparse_dim == sparse_dims_to_sum_size);
  const bool sum_dense_dim = (dense_dims_to_sum_v.size() > 0);
  const bool sum_sparse_dim = (sparse_dims_to_sum_size > 0);

  if (sum_all_sparse_dim) {
    TORCH_CHECK(!grad_.is_sparse(), "_sparse_sum_backward_cuda: expected grad Tensor to be dense since all sparse dims are summed");
    auto grad_input_values = grad_;
    auto expand_size = input_values.sizes().vec();
    if (sum_dense_dim) {
      auto dense_expand_size = std::vector<int64_t>(expand_size);
      dense_expand_size.erase(dense_expand_size.begin()); // remove nnz dim
      for (auto d : dense_dims_to_sum_v) grad_input_values = grad_input_values.unsqueeze(d - 1); // -1 since grad has no nnz dim
      grad_input_values = grad_input_values.expand(dense_expand_size);
    }
    grad_input_values = grad_input_values.expand(expand_size).clone();
    return at::_sparse_coo_tensor_with_dims_and_tensors(input_sparse_dim, input_dense_dim, input_sizes, input_indices.clone(), grad_input_values,  input.options().dtype(grad_.dtype())); // convert to grad dtype
  }
  else {
    TORCH_CHECK(grad_.is_sparse(), "_sparse_sum_backward_cuda: expected grad_ Tensor to be sparse, but got dense");
    auto grad = grad_.coalesce();
    LongTensor grad_indices = grad._indices();
    Tensor grad_values = grad._values();
    const int64_t grad_sparse_dim = grad.sparse_dim();
    const int64_t grad_nnz = grad._nnz();

    Tensor grad_values_expand = grad_values;
    if (sum_dense_dim) {
      auto expand_size = input_values.sizes().vec();
      if (sum_sparse_dim) expand_size[0] = grad_values.size(0); // update nnz
      for (auto d : dense_dims_to_sum_v) grad_values_expand = grad_values_expand.unsqueeze(d);
      grad_values_expand = grad_values_expand.expand(expand_size).clone();
    }

    Tensor grad_input_values;
    if (!sum_sparse_dim) {
      grad_input_values = grad_values_expand;
    }
    else {
      int curDevice = -1;
      cudaGetDevice(&curDevice);
      cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);
      auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
      auto policy = thrust::cuda::par(allocator).on(stream);
      typedef thrust::device_ptr<int64_t> thrust_ptr;

      grad_input_values = at::empty_like(input_values, grad_values.options(), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      AT_ASSERT(grad_input_values.is_cuda());

      // get 1D indices
      auto grad_sparse_dim_to_keep_v = std::vector<int64_t>(grad_sparse_dim);
      std::iota(grad_sparse_dim_to_keep_v.begin(), grad_sparse_dim_to_keep_v.end(), 0);

      auto grad_indices_1D = flatten_indices_by_dims(grad_indices, grad.sizes(), grad_sparse_dim_to_keep_v); // flatten indices on all sparse_dim of grad, output indices is coalesced and sorted
      auto input_indices_1D = flatten_indices_by_dims(input_indices, input_sizes, sparse_dims_to_keep_v);
      thrust_ptr grad_indices_iter(grad_indices_1D.data_ptr<int64_t>());
      thrust_ptr input_indices_iter(input_indices_1D.data_ptr<int64_t>());

      // store lower_bound of input indices at grad indices
      LongTensor input_indices_pos = at::empty_like(input_indices_1D, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      thrust_ptr input_indices_pos_iter(input_indices_pos.data_ptr<int64_t>());
      thrust::lower_bound(policy,
                          grad_indices_iter, grad_indices_iter + grad_nnz,
                          input_indices_iter, input_indices_iter + input_nnz,
                          input_indices_pos_iter);

      // config to run cuda kernel
      int64_t total_threads = input_nnz;
      const dim3 block = dim3(std::min(static_cast<int64_t>(cuda::getApplyBlock().x), total_threads));
      dim3 grid;
      TORCH_CHECK(cuda::getApplyGrid(total_threads, grid, curDevice), "_sparse_sum_backward_cuda: input too large or too many dimensions");

      auto grad_indices_ti = getTensorInfo<int64_t, int64_t>(grad_indices_1D);
      auto input_indices_ti = getTensorInfo<int64_t, int64_t>(input_indices_1D);
      auto input_indices_pos_ti = getTensorInfo<int64_t, int64_t>(input_indices_pos);

      AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_values.scalar_type(), "_sparse_sum_backward_cuda", [&] {
        auto grad_values_expand_ti = getTensorInfo<scalar_t, int64_t>(grad_values_expand);
        auto grad_input_values_ti = getTensorInfo<scalar_t, int64_t>(grad_input_values);

        _sparse_sum_backward_cuda_kernel<scalar_t><<<grid, block, 0, stream>>>(
          total_threads,
          grad_indices_ti,
          input_indices_ti,
          input_indices_pos_ti,
          grad_values_expand_ti,
          grad_input_values_ti
        );
      });
    }

    return at::_sparse_coo_tensor_with_dims_and_tensors(input_sparse_dim, input_dense_dim, input_sizes, input_indices.clone(), grad_input_values, grad.options());
  }
}

}} // namespace at::native
