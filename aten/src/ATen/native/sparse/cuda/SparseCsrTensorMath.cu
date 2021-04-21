#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/BinaryOps.h>
#include <algorithm>

#include <cuda_runtime.h>
#include <type_traits>

#include <THC/THCTensorMathPointwise.cuh>
#include <THC/THCThrustAllocator.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/native/sparse/cuda/SparseCUDABlas.cuh>
#include <c10/cuda/CUDACachingAllocator.h>

#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

namespace at {
namespace native {

using namespace at::sparse_csr;
// certain utiliy functions are usable from sparse COO.
using namespace at::sparse;

// // Functions for matrix multiplication.
// Tensor& addmm_out_sparse_csr_dense_cuda(
//     const Tensor& self,
//     const SparseCsrTensor& op1,
//     const Tensor& op2,
//     const Scalar& beta,
//     const Scalar& alpha,
//     Tensor& out) {
//   AT_ASSERT(op1.is_sparse_csr());
//   Tensor expand_self = *expand_size(self, {op1.size(0), op2.size(1)}, "addmm_out_sparse_csr");

//   AT_ASSERT(expand_self.device().type() == kCPU);
//   TORCH_CHECK(
//       output.device().type() == kCPU,
//       "addmm: expected 'out' to be CPU tensor, but got CUDA tensor");
//   TORCH_CHECK(
//       op1.device().type() == kCPU,
//       "addmm: expected 'mat1' to be a CPU tensor, but got a CUDA tensor");
//   TORCH_CHECK(
//       op2.device().type() == kCPU,
//       "addmm: expected 'mat2' to be a CPU tensor, but got a CUDA tensor");

//   TORCH_CHECK(
//       op1.dim() == 2,
//       "addmm: 2-D matrices expected, got ",
//       op1.dim(),
//       "D tensor");
//   TORCH_CHECK(
//       op2.dim() == 2,
//       "addmm: 2-D matrices expected, got ",
//       op2.dim(),
//       "D tensor");

//   TORCH_CHECK(
//       output.is_contiguous(),
//       "out argument must be contiguous, but got: ",
//       output.suggest_memory_format());

//   // ixk * kxj = ixj
//   int64_t dim_i = op1.size(0);
//   int64_t dim_j = op2.size(1);
//   int64_t dim_k = op1.size(1);

//   TORCH_CHECK(
//       op2.size(0) == dim_k,
//       "addmm: Expected dense matrix (op2) size(0)=",
//       dim_k,
//       ", got ",
//       op2.size(0));
//   TORCH_CHECK(
//       op1.size(1) == dim_k,
//       "addmm: Expected sparse matrix (op1) size(1)=",
//       dim_k,
//       ", got ",
//       op1.size(1));
//   output.resize_({dim_i, dim_j});

//   auto col_indices = op1.col_indices();
//   auto crow_indices = op1.crow_indices();
//   auto values = op1.values();

//   AT_DISPATCH_FLOATING_TYPES(
//       values.scalar_type(), "addmm_sparse_csr_dense", [&] {
//         scalar_t cast_beta = beta.to<scalar_t>();
//         if (!is_same_tensor(out, expand_self)) {
//           output.copy_(expand_self);
//         }
//         if (cast_beta == 0) {
//           output.zero_();
//         } else {
//           at::mul_out(out, expand_self, scalar_to_tensor(beta));
//         }
//       });

//   // Do not use MKL for Windows due to linking issues with sparse MKL routines.
//   if (at::hasMKL() && !is_msvc()) {
//     _sparse_mm_mkl_(out, op1, op2, expand_self, alpha, beta);
//   } else {
//     int64_t dense_stride0 = op1.stride(0);
//     int64_t dense_stride1 = op1.stride(1);
//     int64_t out_stride0 = output.stride(0);
//     int64_t out_stride1 = output.stride(1);

//     AT_DISPATCH_FLOATING_TYPES(
//         values.scalar_type(),
//         "sparse_csr_mm_cuda",
//         [&alpha,
//          &beta,
//          &op1,
//          &out,
//          &values,
//          &crow_indices,
//          &col_indices,
//          &dense_stride0,
//          &dense_stride1,
//          &out_stride0,
//          &out_stride1,
//          &dim_k]() {
//           AT_DISPATCH_INDEX_TYPES(
//               crow_indices.scalar_type(),
//               "csr_mm_crow_indices",
//               [&alpha,
//                &beta,
//                &op1,
//                &out,
//                &values,
//                &crow_indices,
//                &col_indices,
//                &dense_stride0,
//                &dense_stride1,
//                &out_stride0,
//                &out_stride1,
//                &dim_k]() {
//                 scalar_t cast_alpha = alpha.to<scalar_t>();
//                 scalar_t cast_beta = beta.to<scalar_t>();
//                 scalar_t* dense_ptr = op1.data_ptr<scalar_t>();
//                 scalar_t* out_ptr = output.data_ptr<scalar_t>();

//                 auto col_indices_accessor = col_indices.accessor<index_t, 1>();
//                 auto crow_indices_accessor =
//                     crow_indices.accessor<index_t, 1>();
//                 auto values_accessor = values.accessor<scalar_t, 1>();

//                 at::parallel_for(
//                     0,
//                     crow_indices.size(0) - 1,
//                     internal::GRAIN_SIZE,
//                     [&](int64_t irow_start, int64_t irow_end) {
//                       for (int irow = irow_start; irow < irow_end; ++irow) {
//                         int start_index = crow_indices_accessor[irow];
//                         int end_index = crow_indices_accessor[irow + 1];

//                         for (int i = start_index; i < end_index; ++i) {
//                           auto val = values_accessor[i];
//                           auto icol = col_indices_accessor[i];

//                           at::native::cpublas::axpy<scalar_t>(
//                               dim_k,
//                               cast_alpha * val,
//                               dense_ptr + icol * dense_stride0,
//                               dense_stride1,
//                               out_ptr + irow * out_stride0,
//                               out_stride1);
//                         }
//                       }
//                     });
//               });
//         });
//   }
//   return out;
// }

// Tensor addmm_sparse_csr_dense_cuda(
//     const Tensor& self,
//     const SparseCsrTensor& sparse,
//     const Tensor& dense,
//     const Scalar& beta,
//     const Scalar& alpha) {
//   Tensor r = at::empty({0}, self.options());
//   at::addmm_out(r, self, sparse, dense, beta, alpha);
//   return r;
// }

void add_out_dense_sparse_csr_cuda_kernel() {
    // for (int32_t irow = 0; irow < src_crow_indices.size(0) - 1; ++irow) {
    //     int32_t start_index = crow_indices_accessor[irow];
    //     int32_t end_index = crow_indices_accessor[irow + 1];

    //     for (int i = start_index; i < end_index; ++i) {
    //         auto icol = col_indices_accessor[i];
    //         auto index = output.storage_offset() + irow * out_strides0 +
    //             icol * out_strides1;
    //         out_ptr[index] += cast_value * values_accessor[i];
    //     }
    // }
}

// TODO: refactor common cpu/cuda source code
// TODO 1: checks and commong pre_processing like asinh!

Tensor& add_out_dense_sparse_csr_cuda(
    Tensor& output,
    const Tensor& dense,
    const SparseCsrTensor& src,
    const Scalar& alpha) {
  AT_ASSERT(dense.layout() == kStrided);
  AT_ASSERT(src.is_sparse_csr());
  std::cerr << "dense.device(): " << dense.device() << std::endl;
  std::cerr << "src.device(): " << src.is_cuda() << std::endl;
  AT_ASSERT(dense.is_cuda());

  TORCH_CHECK(
      output.is_contiguous(),
      "out argument must be contiguous, but got: ",
      output.suggest_memory_format());
  TORCH_CHECK(
      output.is_cuda(),
      "add: expected 'out' to be CUDA tensor, but got tensor on device: ",
      output.device());

/*TORCH_CHECK(
      src.is_cuda(),
      "add: expected 'other' to be a CUDA tensor, but got tensor on device: ",
      src.device());
*/
  TORCH_CHECK(
      dense.sizes().equals(src.sizes()),
      "add: expected 'self' and 'other' to have same size, but self has size ",
      dense.sizes(),
      " while other has size ",
      src.sizes(),
      " (FYI: op2-sparse addition does not currently support broadcasting)");

  auto commonDtype = promoteTypes(dense.scalar_type(), src.scalar_type());
  TORCH_CHECK(
      canCast(commonDtype, output.scalar_type()),
      "Can't convert result type ",
      commonDtype,
      " to output ",
      output.scalar_type(),
      " in add operation");

  Tensor src_values = src.values().to(commonDtype);
  Tensor src_crow_indices = src.crow_indices();
  Tensor src_col_indices = src.col_indices();

  output.resize_as_(dense);
  Tensor resultBuffer = output;
  Tensor valuesBuffer = src_values.to(commonDtype);

  if (output.scalar_type() != commonDtype) {
    resultBuffer = dense.to(commonDtype);
  } else if (!is_same_tensor(output, dense)) {
    resultBuffer.copy_(dense);
  }
  std::cerr << " ** src_values ** \n \t " << src_values.device() << std::endl;
  std::cerr << " ** src_crow_indices ** \n \t " << src_crow_indices.device() << std::endl;
  std::cerr << " ** src_col_indices ** \n \t " << src_col_indices.device() << std::endl;

  AT_DISPATCH_ALL_TYPES(
      commonDtype,
      "add_out_op2_sparse_csr",
      [&src_values, &output, &alpha, &src_crow_indices, &src_col_indices]() {
        AT_DISPATCH_INDEX_TYPES(
            src_crow_indices.scalar_type(),
            "csr_add_out_crow_indices",
              [&src_values, &output, &alpha, &src_crow_indices, &src_col_indices]() {
                scalar_t* values_accessor = src_values.data_ptr<scalar_t>();
                scalar_t* out_ptr = output.data_ptr<scalar_t>();
                scalar_t cast_value = alpha.to<scalar_t>();

                index_t* crow_indices_accessor = src_crow_indices.data_ptr<index_t>();
                index_t* col_indices_accessor = src_col_indices.data_ptr<index_t>();
                int64_t out_storage_offset = output.storage_offset();

                auto out_strides = output.strides();
                int64_t out_strides0 = out_strides[0];
                int64_t out_strides1 = out_strides[1];

                cudaStream_t stream = at::cuda::getCurrentCUDAStream();
                auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
                auto policy = thrust::cuda::par(allocator).on(stream);

               thrust::for_each(
                    policy,
                    thrust::make_counting_iterator(int64_t(0)),
                    thrust::make_counting_iterator(int64_t(src_crow_indices.size(0) - 1)),
                    [values_accessor,
                    crow_indices_accessor,
                    col_indices_accessor,
                    out_ptr,
                    out_storage_offset,
                    out_strides0,
                    cast_value,
                    out_strides1
                    ]__device__(int64_t irow) {
                        int32_t start_index = crow_indices_accessor[irow];
                        int32_t end_index = crow_indices_accessor[irow + 1];

                        for (int i = start_index; i < end_index; ++i) {
                            auto icol = col_indices_accessor[i];
                            auto index = out_storage_offset + irow * out_strides0 + icol * out_strides1;
                            out_ptr[index] += cast_value * values_accessor[i];
                        }
                    });
              });
      });
  std::cerr << " **** add_out_dense_sparse_csr_cuda ** \n \t " << output << std::endl;
  return output;
}

Tensor& add_out_sparse_csr_cuda(
    const Tensor& self,
    const SparseCsrTensor& other,
    const Scalar& alpha,
    SparseCsrTensor& out) {
  if (self.layout() == kStrided) {
    return add_out_dense_sparse_csr_cuda(out, self, other, alpha);
  } else {
    TORCH_CHECK(
        false,
        "NotImplementedError: Addition of sparse CSR tensors is not yet implemented.")
  }
  return out;
}

} // namespace native
} // namespace at
