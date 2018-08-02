#include "ATen/Context.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/Dispatch.h"
#include "ATen/NativeFunctions.h"
#include "ATen/cuda/PinnedMemoryAllocator.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"

#include "ATen/native/LinearAlgebraUtils.h"
#include "ATen/native/Gesv.h"

#include "THC.h" // for USE_MAGMA

#ifdef USE_MAGMA
#include <magma.h>
#include <magma_types.h>
#endif

namespace at {
namespace native {

#ifdef USE_MAGMA

template<class scalar_t>
void magmaGesv(
    int64_t n, int64_t nrhs, scalar_t* A_data, int64_t lda,
    int* ipiv, scalar_t* B_data, int64_t ldb, int* info) {
  AT_ERROR("magma: gesv only takes float or double Tensors");
}

template<>
void magmaGesv<float>(
    int64_t n, int64_t nrhs, float* A_data, int64_t lda,
    int* ipiv, float* B_data, int64_t ldb, int* info) {
  magma_sgesv_gpu(n, nrhs, A_data, lda, ipiv, B_data, ldb, info);
}

template<>
void magmaGesv<double>(
    int64_t n, int64_t nrhs, double* A_data, int64_t lda,
    int* ipiv, double* B_data, int64_t ldb, int* info) {
  magma_dgesv_gpu(n, nrhs, A_data, lda, ipiv, B_data, ldb, info);
}

template<class scalar_t>
void magmaGesvBatched(
    magma_int_t n, magma_int_t nrhs, scalar_t** dA_array, magma_int_t ldda,
    magma_int_t** dipiv_array, scalar_t** dB_array, magma_int_t lddb,
    magma_int_t* dinfo_array, magma_int_t batch_count, magma_queue_t queue) {
  AT_ERROR("gesv only takes float or double Tensors");
}

template<>
void magmaGesvBatched<float>(
    magma_int_t n, magma_int_t nrhs, float** dA_array, magma_int_t ldda,
    magma_int_t** dipiv_array, float** dB_array, magma_int_t lddb,
    magma_int_t* dinfo_array, magma_int_t batch_count, magma_queue_t queue) {
  magma_sgesv_batched(
      n, nrhs, dA_array, ldda, dipiv_array,
      dB_array, lddb, dinfo_array, batch_count, queue);
}

template<>
void magmaGesvBatched<double>(
    magma_int_t n, magma_int_t nrhs, double** dA_array, magma_int_t ldda,
    magma_int_t** dipiv_array, double** dB_array, magma_int_t lddb,
    magma_int_t* dinfo_array, magma_int_t batch_count, magma_queue_t queue) {
  magma_dgesv_batched(
      n, nrhs, dA_array, ldda, dipiv_array,
      dB_array, lddb, dinfo_array, batch_count, queue);
}

static magma_queue_t createMagmaQueue(const Tensor& tensor) {
  auto& context = tensor.type().get_context();
  magma_queue_t magma_queue;
  magma_queue_create_from_cuda(
      tensor.get_device(),
      at::cuda::getCurrentCUDAStream(),
      THCState_getCurrentBlasHandle(context.getTHCState()),
      THCState_getCurrentSparseHandle(context.getTHCState()),
      &magma_queue);
  return magma_queue;
}

static inline magma_int_t magma_int_cast(int64_t value, const char* varname) {
  auto result = static_cast<magma_int_t>(value);
  if (static_cast<int64_t>(result) != value) {
    AT_ERROR("magma: The value of %s (%lld) is too large to fit into a magma_int_t (%llu bytes)",
             varname, (long long)value, sizeof(magma_int_t));
  }
  return result;
}
#endif

// Creates an array of size elements of type T, backed by pinned memory
// wrapped in a Storage
template<class T>
static inline std::unique_ptr<Storage> pin_memory(int64_t size, Tensor dummy) {
  int64_t adjusted_size = size * sizeof(T);
  auto* allocator = cuda::getPinnedMemoryAllocator();
  auto& backend = dummy.type().toBackend(kCPU).toScalarType(kByte);
  return backend.storageWithAllocator(adjusted_size, allocator);
}

static inline bool isTransposeContiguous(Tensor& self) {
  return self.dim() == 2 &&
         self.stride(0) == 1 &&
         self.stride(1) == self.size(0);
}

#define ALLOCATE_ARRAY(name, type, size, dummy_tensor) \
  auto storage_##name = pin_memory<type>(size, dummy_tensor); \
  name = reinterpret_cast<type*>(storage_##name->data());

template <typename scalar_t>
static void applyGesv(Tensor& b, Tensor& A, std::vector<int64_t> infos) {
#ifndef USE_MAGMA
AT_ERROR("gesv: MAGMA library not found in "
    "compilation. Please rebuild with MAGMA.");
#else
  auto A_data = A.data<scalar_t>();
  auto b_data = b.data<scalar_t>();
  auto A_mat_stride = matrixStride(A);
  auto b_mat_stride = matrixStride(b);

  magma_int_t batch_size = magma_int_cast(batchCount(A), "batchCount");
  magma_int_t n = magma_int_cast(A.size(-2), "A.size(-2)");
  magma_int_t nrhs = magma_int_cast(b.size(-1), "b.size(-1)");

  magma_int_t* info_array;
  magma_int_t* ipiv_data;
  magma_int_t** ipiv_array;
  scalar_t** A_array;
  scalar_t** b_array;

  ALLOCATE_ARRAY(info_array, magma_int_t, batch_size, b);
  ALLOCATE_ARRAY(ipiv_data, magma_int_t, batch_size * n, b);
  ALLOCATE_ARRAY(ipiv_array, magma_int_t*, batch_size, b);
  ALLOCATE_ARRAY(A_array, scalar_t*, batch_size, b);
  ALLOCATE_ARRAY(b_array, scalar_t*, batch_size, b);

  // Set up the created arrays
  for (int64_t i = 0; i < batch_size; i++) {
    A_array[i] = &A_data[i * A_mat_stride];
    b_array[i] = &b_data[i * b_mat_stride];
    ipiv_array[i] = &ipiv_data[i * n];
  }

  magmaGesvBatched<scalar_t>(
      n, nrhs, A_array, n, ipiv_array, b_array, n,
      info_array, batch_size, createMagmaQueue(b));

  for (int64_t i = 0; i < batch_size; i++) {
    infos[i] = info_array[i];
  }
#endif
}

std::tuple<Tensor,Tensor> _gesv_helper_cuda(const Tensor& self, const Tensor& A) {
  std::vector<int64_t> infos(batchCount(A), 0);
  auto A_working_copy = cloneBatchedColumnMajor(A);
  auto b_working_copy = cloneBatchedColumnMajor(self);
  AT_DISPATCH_FLOATING_TYPES(self.type(), "gesv", [&]{
    applyGesv<scalar_t>(b_working_copy, A_working_copy, infos);
  });
  checkErrors(infos);
  return std::tuple<Tensor,Tensor>(b_working_copy, A_working_copy);
}

std::tuple<Tensor&,Tensor&> _gesv_single_out_cuda(Tensor& sol, Tensor& lu,
    const Tensor& self, const Tensor& A) {
#ifndef USE_MAGMA
AT_ERROR("gesv: MAGMA library not found in "
    "compilation. Please rebuild with MAGMA.");
#else
  int64_t bx = self.size(0);
  int64_t by = (self.dim() == 1) ? 1 : self.size(1);
  int64_t ax = A.size(0);
  int64_t ay = A.size(1);
  int info;
  int* ipiv;

  bool use_temp_sol = false;
  bool use_temp_lu = false;
  auto temp_sol = self.type().tensor();
  auto temp_lu = self.type().tensor();

  /* Init to column major format. See Gesv.cpp for more comments */

  bool tc = isTransposeContiguous(sol);
  if (tc) {
    sol.t_();
  } else if (sol.dim() == 2 &&
             sol.size(0) == bx &&
             sol.size(1) == by) {
    use_temp_sol = true;
  }

  if (use_temp_sol) {
    temp_sol.resize_({by, bx});
    temp_sol.copy_(self.view({bx, by}).t());
  } else {
    sol.resize_({by, bx});
    if (&self == &sol) {
      if (!tc) {
        sol.copy_(self.view({bx, by}).t().clone());
      }
    } else {
      sol.copy_(self.view({bx, by}).t());
    }
  }

  tc = isTransposeContiguous(lu);
  if (tc) {
    lu.t_();
  } else if (lu.dim() == 2 &&
             lu.size(0) == ax &&
             lu.size(1) == ay) {
    use_temp_lu = true;
  }

  if (use_temp_lu) {
    temp_lu.resize_({ay, ax});
    temp_lu.copy_(A.t());
  } else {
    lu.resize_({ay, ax});
    if (&A == &lu) {
      if (!tc) {
        lu.copy_(A.t().clone());
      }
    } else {
      lu.copy_(A.t());
    }
  }

  AT_DISPATCH_FLOATING_TYPES(self.type(), "gesv", [&]{
      auto A_ptr = use_temp_lu ? temp_lu.data<scalar_t>()
                               : lu.data<scalar_t>();
      auto b_ptr = use_temp_sol ? temp_sol.data<scalar_t>()
                                : sol.data<scalar_t>();
      ALLOCATE_ARRAY(ipiv, int, bx, sol);
      magmaGesv<scalar_t>(bx, by, A_ptr, bx, ipiv, b_ptr, bx, &info);
  });

  if (use_temp_sol) {
    sol.copy_(temp_sol.t_());
  } else {
    sol.t_();
  }

  if (use_temp_lu) {
    lu.copy_(temp_lu.t_());
  } else {
    lu.t_();
  }

  checkErrors({info});
  return std::tuple<Tensor&,Tensor&>(sol, lu);
#endif
}
}}  // namespace at::native

#undef ALLOCATE_ARRAY
