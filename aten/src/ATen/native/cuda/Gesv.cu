#include "ATen/Context.h"
#include "ATen/Dispatch.h"
#include "ATen/NativeFunctions.h"
#include "ATen/PinnedMemoryAllocator.h"
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
template<class real>
void magmaGesvBatched(
    magma_int_t n, magma_int_t nrhs, real** dA_array, magma_int_t ldda,
    magma_int_t** dipiv_array, real** dB_array, magma_int_t lddb,
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
      context.getCurrentCUDAStream(),
      THCState_getCurrentBlasHandle(context.thc_state),
      THCState_getCurrentSparseHandle(context.thc_state),
      &magma_queue);
  return magma_queue;
}
#endif

// Creates an array of size elements of type T, backed by pinned memory
// wrapped in a Storage
template<class T>
static inline std::unique_ptr<Storage> pin_memory(int64_t size, Tensor dummy) {
  int64_t adjusted_size = size * sizeof(T);
  auto allocator = std::unique_ptr<Allocator>(new PinnedMemoryAllocator());
  auto& backend = dummy.type().toBackend(kCPU).toScalarType(kByte);
  return backend.storageWithAllocator(adjusted_size, std::move(allocator));
}

#define ALLOCATE_ARRAY(name, type, size, dummy_tensor) \
  auto storage_##name = pin_memory<type>(size, dummy_tensor); \
  name = reinterpret_cast<type*>(storage_##name->data());

template <typename real>
static void applyGesv(Tensor& b, Tensor& A, std::vector<int64_t> infos) {
#ifndef USE_MAGMA
AT_ERROR("gesv: MAGMA library not found in "
    "compilation. Please rebuild with MAGMA.");
#else
  real* A_data = static_cast<real*>(A.data_ptr());
  real* b_data = static_cast<real*>(b.data_ptr());
  auto A_mat_stride = matrixStride(A);
  auto b_mat_stride = matrixStride(b);

  magma_int_t batch_size = batchCount(A);
  magma_int_t n = A.size(-2);
  magma_int_t nrhs = b.size(-1);

  magma_int_t* info_array;
  magma_int_t* ipiv_data;
  magma_int_t** ipiv_array;
  real** A_array;
  real** b_array;

  ALLOCATE_ARRAY(info_array, magma_int_t, batch_size, b);
  ALLOCATE_ARRAY(ipiv_data, magma_int_t, batch_size * n, b);
  ALLOCATE_ARRAY(ipiv_array, magma_int_t*, batch_size, b);
  ALLOCATE_ARRAY(A_array, real*, batch_size, b);
  ALLOCATE_ARRAY(b_array, real*, batch_size, b);

  // Set up the created arrays
  for (int64_t i = 0; i < batch_size; i++) {
    A_array[i] = &A_data[i * A_mat_stride];
    b_array[i] = &b_data[i * b_mat_stride];
    ipiv_array[i] = &ipiv_data[i * n];
  }

  magmaGesvBatched<real>(
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

}}  // namespace at::native

#undef ALLOCATE_ARRAY
