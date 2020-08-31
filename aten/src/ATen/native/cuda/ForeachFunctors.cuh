#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/MultiTensorApply.cuh>

namespace at { namespace native {

namespace {

template<typename T>
struct AddScalarFunctor_ {
    __device__ void operator() (
        int chunk_size,
        TensorListMetadata<1>& tl,
        T scalar) {
            int tensor_loc = tl.block_to_tensor[blockIdx.x];
            int chunk_idx = tl.block_to_chunk[blockIdx.x];
            int n = tl.sizes[tensor_loc];

            T* x = (T*)tl.addresses[0][tensor_loc];
            x += chunk_idx * chunk_size;

            n -= chunk_idx * chunk_size;

            T r_x[kILP];

            // to make things simple, we put aligned case in a different code path
            if(n % kILP == 0 && chunk_size % kILP == 0 && is_aligned(x)) {
                for(int i_start = threadIdx.x; i_start * kILP < n && i_start * kILP < chunk_size; i_start += blockDim.x) {
                    // load
                    load_store(r_x, x, 0 , i_start);
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        r_x[ii] = static_cast<T>(r_x[ii]) + scalar;
                    }
                    // store
                    load_store(x, r_x, i_start, 0);
                }
            }
            else {
                for(int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * kILP) {
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        r_x[ii] = 0;
                        int i = i_start + threadIdx.x + ii * blockDim.x;
                        if(i < n && i < chunk_size) {
                            r_x[ii] = x[i];
                        }
                    }
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        r_x[ii] = static_cast<T>(r_x[ii]) + scalar;
                    }
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        int i = i_start + threadIdx.x + ii * blockDim.x;
                        if(i < n && i < chunk_size)
                            x[i] = r_x[ii];
                    }
                }
            }
        }
};

template<typename T>
struct AddScalarFunctor {
    __device__ void operator() (
        int chunk_size,
        TensorListMetadata<2>& tl,
        T scalar) {
            int tensor_loc = tl.block_to_tensor[blockIdx.x];
            int chunk_idx = tl.block_to_chunk[blockIdx.x];
            int n = tl.sizes[tensor_loc];

            T* x = (T*)tl.addresses[0][tensor_loc];
            x += chunk_idx * chunk_size;

            T* out = (T*)tl.addresses[1][tensor_loc];
            out += chunk_idx * chunk_size;

            n -= chunk_idx * chunk_size;

            T r_x[kILP];
            T r_out[kILP];

            // to make things simple, we put aligned case in a different code path
            if(n % kILP == 0 && chunk_size % kILP == 0 && is_aligned(x) && is_aligned(out)) {
                for(int i_start = threadIdx.x; i_start * kILP < n && i_start * kILP < chunk_size; i_start += blockDim.x) {
                    // load
                    load_store(r_x, x, 0 , i_start);
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        r_out[ii] = static_cast<T>(r_x[ii]) + scalar;
                    }
                    // store
                    load_store(out, r_out, i_start, 0);
                }
            }
            else {
                for(int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * kILP) {
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        r_x[ii] = 0;
                        int i = i_start + threadIdx.x + ii * blockDim.x;
                        if(i < n && i < chunk_size) {
                            r_x[ii] = x[i];
                        }
                    }
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        r_out[ii] = static_cast<T>(r_x[ii]) + scalar;
                    }
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        int i = i_start + threadIdx.x + ii * blockDim.x;
                        if(i < n && i < chunk_size)
                            out[i] = r_out[ii];
                    }
                }
            }
        }
};

template<typename T>
struct AddListFunctor_ {
    __device__ void operator() (
        int chunk_size,
        TensorListMetadata<2>& tl) {
            int tensor_loc = tl.block_to_tensor[blockIdx.x];
            int chunk_idx = tl.block_to_chunk[blockIdx.x];
            int n = tl.sizes[tensor_loc];

            T* x = (T*)tl.addresses[0][tensor_loc];
            x += chunk_idx * chunk_size;

            T* y = (T*)tl.addresses[1][tensor_loc];
            y += chunk_idx * chunk_size;

            n -= chunk_idx * chunk_size;

            T r_x[kILP];
            T r_y[kILP];

            // to make things simple, we put aligned case in a different code path
            if(n % kILP == 0 && chunk_size % kILP == 0 && is_aligned(x) && is_aligned(y)) {
                for(int i_start = threadIdx.x; i_start * kILP < n && i_start * kILP < chunk_size; i_start += blockDim.x) {
                    // load
                    load_store(r_x, x, 0 , i_start);
                    load_store(r_y, y, 0 , i_start);
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        r_x[ii] = static_cast<T>(r_x[ii]) + static_cast<T>(r_y[ii]);
                    }
                    // store
                    load_store(x, r_x, i_start , 0);
                }
            }
            else {
                for(int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * kILP) {
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        r_x[ii] = 0;
                        r_y[ii] = 0;
                        int i = i_start + threadIdx.x + ii * blockDim.x;
                        if(i < n && i < chunk_size) {
                            r_x[ii] = x[i];
                            r_y[ii] = y[i];
                        }
                    }
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        r_x[ii] = static_cast<T>(r_x[ii]) + static_cast<T>(r_y[ii]);
                    }
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        int i = i_start + threadIdx.x + ii * blockDim.x;
                        if(i < n && i < chunk_size)
                            x[i] = r_x[ii];
                    }
                }
            }
        }
};

template<typename T>
struct AddListFunctor {
    __device__ void operator() (
        int chunk_size,
        TensorListMetadata<3>& tl) {
            int tensor_loc = tl.block_to_tensor[blockIdx.x];
            int chunk_idx = tl.block_to_chunk[blockIdx.x];
            int n = tl.sizes[tensor_loc];

            T* x = (T*)tl.addresses[0][tensor_loc];
            x += chunk_idx * chunk_size;

            T* y = (T*)tl.addresses[1][tensor_loc];
            y += chunk_idx * chunk_size;

            T* out = (T*)tl.addresses[2][tensor_loc];
            out += chunk_idx * chunk_size;

            n -= chunk_idx * chunk_size;

            T r_x[kILP];
            T r_y[kILP];
            T r_out[kILP];

            // to make things simple, we put aligned case in a different code path
            if(n % kILP == 0 && chunk_size % kILP == 0 && is_aligned(x) && is_aligned(y) && is_aligned(out)) {
                for(int i_start = threadIdx.x; i_start * kILP < n && i_start * kILP < chunk_size; i_start += blockDim.x) {
                    // load
                    load_store(r_x, x, 0 , i_start);
                    load_store(r_y, y, 0 , i_start);
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        r_out[ii] = static_cast<T>(r_x[ii]) + static_cast<T>(r_y[ii]);
                    }
                    // store
                    load_store(out, r_out, i_start , 0);
                }
            }
            else {
                for(int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * kILP) {
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        r_x[ii] = 0;
                        r_y[ii] = 0;
                        int i = i_start + threadIdx.x + ii * blockDim.x;
                        if(i < n && i < chunk_size) {
                            r_x[ii] = x[i];
                            r_y[ii] = y[i];
                        }
                    }
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        r_out[ii] = static_cast<T>(r_x[ii]) + static_cast<T>(r_y[ii]);
                    }
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        int i = i_start + threadIdx.x + ii * blockDim.x;
                        if(i < n && i < chunk_size)
                            out[i] = r_out[ii];
                    }
                }
            }
        }
};

} // namespace

}} // namespace at::native
