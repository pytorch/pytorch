#include "ATen/Context.h"
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
void magmaPotrfBatched(
  magma_uplo_t uplo, magma_int_t n, scalar_t **dA_array,
  magma_int_t ldda, magma_int_t *info_array, magma_int_t batchCount,
  magma_queue_t queue) {
  AT_ERROR("potrf only takes float or double Tensors");
}

template<>
void magmaPotrfBatched<float>(
    magma_uplo_t uplo, magma_int_t n, float **dA_array,
    magma_int_t ldda, magma_int_t *info_array, magma_int_t batchCount,
    magma_queue_t queue) {
  magma_spotrf_batched(uplo, n, dA_array,
                       ldda, info_array, batchCount, queue);
}

template<>
void magmaPotrfBatched<double>(
    magma_uplo_t uplo, magma_int_t n, double **dA_array,
    magma_int_t ldda, magma_int_t *info_array, magma_int_t batchCount,
    magma_queue_t queue) {
  magma_dpotrf_batched(uplo, n, dA_array,
                       ldda, info_array, batchCount, queue);
}


static magma_queue_t createMagmaQueue(const Tensor& tensor) {
  auto& context = tensor.type().get_context();
  magma_queue_t magma_queue;
  magma_queue_create_from_cuda(
      tensor.get_device(),
      context.getCurrentCUDAStream(),
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

#define ALLOCATE_ARRAY(name, type, size, dummy_tensor) \
  auto storage_##name = pin_memory<type>(size, dummy_tensor); \
  name = reinterpret_cast<type*>(storage_##name->data());

template <typename scalar_t>
static void applyPotrf(Tensor& A, bool upper, std::vector<int64_t> infos) {
#ifndef USE_MAGMA
AT_ERROR("potrf: MAGMA library not found in "
    "compilation. Please rebuild with MAGMA.");
#else
  auto A_data = A.data<scalar_t>();
  auto A_mat_stride = matrixStride(A);

  magma_int_t batch_size = magma_int_cast(batchCount(A), "batchCount");
  magma_int_t n = magma_int_cast(A.size(-2), "A.size(-2)");

  magma_int_t* info_array;
  scalar_t** A_array;

  ALLOCATE_ARRAY(info_array, magma_int_t, batch_size, A);
  ALLOCATE_ARRAY(A_array, scalar_t*, batch_size, A);

  // Set up the created arrays
  for (int64_t i = 0; i < batch_size; i++) {
    A_array[i] = &A_data[i * A_mat_stride];
  }

  magma_uplo_t uplo = magma_uplo_const(upper ? 'U' : 'L');
  magmaPotrfBatched<scalar_t>(uplo, n, A_array, n,
                              info_array, batch_size, createMagmaQueue(A));

  for (int64_t i = 0; i < batch_size; i++) {
    infos[i] = info_array[i];
  }
#endif
}

Tensor _potrf_helper_cuda(const Tensor& A, bool upper) {
  std::vector<int64_t> infos(batchCount(A), 0);
  auto A_working_copy = cloneBatchedColumnMajor(A);
  AT_DISPATCH_FLOATING_TYPES(A.type(), "potrf", [&]{
    applyPotrf<scalar_t>(A_working_copy, upper, infos);
  });
  checkErrors(infos);
  return A_working_copy;
}

}}  // namespace at::native

#undef ALLOCATE_ARRAY
