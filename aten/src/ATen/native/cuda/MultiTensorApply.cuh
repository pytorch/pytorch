#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>

namespace at { namespace native {

namespace {

static constexpr int64_t kILP = 4;
static constexpr int64_t kChunkSize = 65536;
static constexpr int64_t kBlockSize = 512;

template<typename T>
__device__ __forceinline__ bool is_aligned(T* p){
  return ((uint64_t)p) % (kILP * sizeof(T)) == 0;
}

template<typename T>
__device__ __forceinline__ void load_store(T* dst, T* src, int dst_offset, int src_offset){
  using LT = at::native::memory::aligned_vector<T, kILP>;
  ((LT*)dst)[dst_offset] = ((LT*)src)[src_offset];
}

// TensorListMetadata has to be < 4KB - the limit for kernel launch argument
static constexpr int depth_to_max_tensors[5] = {110, 64, 48, 36, 30};
static constexpr int depth_to_max_blocks[5] = {320, 320, 320, 320, 320};
static constexpr int depth_to_max_tensors_scalarlist[5] = {96, 64, 48, 36, 30};

template<int n> struct TensorListMetadata
{
  void* addresses[n][depth_to_max_tensors[n-1]];
  int numel_for_tensor[depth_to_max_tensors[n-1]];
  unsigned char block_to_tensor[depth_to_max_blocks[n-1]];
  int block_to_chunk[depth_to_max_blocks[n-1]];
};

template<typename scalar_vals_t, int n> struct TensorListScalarListMetadata
{
  void* addresses[n][depth_to_max_tensors_scalarlist[n-1]];
  int numel_for_tensor[depth_to_max_tensors_scalarlist[n-1]];
  scalar_vals_t scalar_vals[depth_to_max_tensors_scalarlist[n-1]];
  unsigned char block_to_tensor[depth_to_max_blocks[n-1]];
  int block_to_chunk[depth_to_max_blocks[n-1]];
};

template<typename T, typename U, typename... ArgTypes>
C10_LAUNCH_BOUNDS_1(kBlockSize)
__global__ void
multi_tensor_apply_kernel(
    T tensorListMeta,
    U callable,
    ArgTypes... args) {
  // Hand the chunk information to the user-supplied functor to process however it likes.
  callable(kChunkSize, tensorListMeta, args...);
}

template<int depth, typename scalar_T, typename T, typename... ArgTypes>
void multi_tensor_apply(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    at::ArrayRef<Scalar> scalars,
    T callable,
    ArgTypes... args) {
        TORCH_CHECK(tensor_lists.size() == depth, "Number of tensor lists has to match the depth.");
        size_t n_tensors = tensor_lists[0].size();
        using scalar_vals_t = typename T::opmath_t;
        TensorListScalarListMetadata<scalar_vals_t, depth> tensorListMeta;

        int loc_block_info = 0;
        int loc_tensor_info = 0;
        for(size_t t = 0; t < n_tensors; t++) {

            tensorListMeta.scalar_vals[loc_tensor_info] = scalars[t].to<scalar_T>();

            tensorListMeta.numel_for_tensor[loc_tensor_info] = tensor_lists[0][t].numel();
            for (int d = 0; d < depth; d++) {
                tensorListMeta.addresses[d][loc_tensor_info] = tensor_lists[d][t].data_ptr();
            }
            loc_tensor_info++;

            int chunks = (tensor_lists[0][t].numel() + kChunkSize - 1)/kChunkSize;
            for (int chunk = 0; chunk < chunks; chunk++) {
                tensorListMeta.block_to_tensor[loc_block_info] = loc_tensor_info - 1;
                tensorListMeta.block_to_chunk[loc_block_info] = chunk;
                loc_block_info++;

                bool tensors_full = (loc_tensor_info == depth_to_max_tensors_scalarlist[depth-1] &&
                    chunk == chunks - 1);
                bool blocks_full = (loc_block_info == depth_to_max_blocks[depth-1]);
                bool last_chunk = (t == n_tensors - 1 && chunk == chunks - 1);

                if (tensors_full || blocks_full || last_chunk) {
                    multi_tensor_apply_kernel<<<loc_block_info, kBlockSize, 0, at::cuda::getCurrentCUDAStream()>>>(
                        tensorListMeta,
                        callable,
                        args...);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();

                    // Reset.
                    loc_block_info = 0;
                    if(chunk == chunks - 1) {
                        loc_tensor_info = 0;
                    }
                    else {
                        tensorListMeta.numel_for_tensor[0] = tensorListMeta.numel_for_tensor[loc_tensor_info-1];
                        tensorListMeta.scalar_vals[0] = tensorListMeta.scalar_vals[loc_tensor_info-1];
                        for(int d = 0; d < depth; d++) {
                            tensorListMeta.addresses[d][0] = tensorListMeta.addresses[d][loc_tensor_info-1];
                        }
                        loc_tensor_info = 1;
                    }
                }
            }
        }
    }

template<int depth, typename T, typename... ArgTypes>
void multi_tensor_apply(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    T callable,
    ArgTypes... args) {
        TORCH_CHECK(tensor_lists.size() == depth, "Number of tensor lists has to match the depth.");
        size_t n_tensors = tensor_lists[0].size();
        TensorListMetadata<depth> tensorListMeta;

        int loc_block_info = 0;
        int loc_tensor_info = 0;
        for(size_t t = 0; t < n_tensors; t++) {
            tensorListMeta.numel_for_tensor[loc_tensor_info] = tensor_lists[0][t].numel();
            for (int d = 0; d < depth; d++) {
                tensorListMeta.addresses[d][loc_tensor_info] = tensor_lists[d][t].data_ptr();
            }
            loc_tensor_info++;

            int chunks = (tensor_lists[0][t].numel() + kChunkSize - 1)/kChunkSize;
            for (int chunk = 0; chunk < chunks; chunk++) {
                tensorListMeta.block_to_tensor[loc_block_info] = loc_tensor_info - 1;
                tensorListMeta.block_to_chunk[loc_block_info] = chunk;
                loc_block_info++;

                bool tensors_full = (loc_tensor_info == depth_to_max_tensors[depth-1] &&
                    chunk == chunks - 1);
                bool blocks_full = (loc_block_info == depth_to_max_blocks[depth-1]);
                bool last_chunk = (t == n_tensors - 1 && chunk == chunks - 1);

                if (tensors_full || blocks_full || last_chunk) {
                    multi_tensor_apply_kernel<<<loc_block_info, kBlockSize, 0, at::cuda::getCurrentCUDAStream()>>>(
                        tensorListMeta,
                        callable,
                        args...);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();

                    // Reset.
                    loc_block_info = 0;
                    if(chunk == chunks - 1) {
                        loc_tensor_info = 0;
                    }
                    else {
                        tensorListMeta.numel_for_tensor[0] = tensorListMeta.numel_for_tensor[loc_tensor_info-1];
                        for(int d = 0; d < depth; d++) {
                            tensorListMeta.addresses[d][0] = tensorListMeta.addresses[d][loc_tensor_info-1];
                        }
                        loc_tensor_info = 1;
                    }
                }
            }
        }
    }
} // namespace
}} // at::native
