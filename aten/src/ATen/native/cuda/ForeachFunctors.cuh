#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/MultiTensorApply.cuh>

namespace at { namespace native {

namespace {

// For FP16 or BFloat16 inputs, ops should perform internal math in FP32.
template<typename scalar_t> struct get_opmath_t { using opmath_t = scalar_t; };
template<> struct get_opmath_t<at::Half> { using opmath_t = float; };
template<> struct get_opmath_t<at::BFloat16> { using opmath_t = float; };

template<typename T>
struct BinaryOpScalarFunctor_ {
    using opmath_t = typename get_opmath_t<T>::opmath_t;
    template<typename Op> __device__ __forceinline__ void operator() (
        int chunk_size,
        TensorListMetadata<1>& tl,
        Op op,
        opmath_t scalar) {
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
                        r_x[ii] = static_cast<T>(op(static_cast<opmath_t>(r_x[ii]),
                                                    static_cast<opmath_t>(scalar)));
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
                        r_x[ii] = static_cast<T>(op(static_cast<opmath_t>(r_x[ii]),
                                                    static_cast<opmath_t>(scalar)));
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
struct BinaryOpScalarFunctor {
    using opmath_t = typename get_opmath_t<T>::opmath_t;
    template<typename Op> __device__ __forceinline__ void operator() (
        int chunk_size,
        TensorListMetadata<2>& tl,
        Op op,
        opmath_t scalar) {
            int tensor_loc = tl.block_to_tensor[blockIdx.x];
            int chunk_idx = tl.block_to_chunk[blockIdx.x];
            int n = tl.sizes[tensor_loc];

            T* x = (T*)tl.addresses[0][tensor_loc];
            x += chunk_idx * chunk_size;

            T* out = (T*)tl.addresses[1][tensor_loc];
            out += chunk_idx * chunk_size;

            n -= chunk_idx * chunk_size;

            T r_x[kILP];

            // to make things simple, we put aligned case in a different code path
            if(n % kILP == 0 && chunk_size % kILP == 0 && is_aligned(x) && is_aligned(out)) {
                for(int i_start = threadIdx.x; i_start * kILP < n && i_start * kILP < chunk_size; i_start += blockDim.x) {
                    // load
                    load_store(r_x, x, 0 , i_start);
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        r_x[ii] = static_cast<T>(op(static_cast<opmath_t>(r_x[ii]),
                                                    static_cast<opmath_t>(scalar)));
                    }
                    // store
                    load_store(out, r_x, i_start, 0);
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
                        r_x[ii] = static_cast<T>(op(static_cast<opmath_t>(r_x[ii]),
                                                    static_cast<opmath_t>(scalar)));
                    }
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        int i = i_start + threadIdx.x + ii * blockDim.x;
                        if(i < n && i < chunk_size)
                            out[i] = r_x[ii];
                    }
                }
            }
        }
};

template<typename T>
struct BinaryOpScalarListFunctor_ {
    using io_t = T;
    using opmath_t = typename get_opmath_t<T>::opmath_t;
    template<typename Op> __device__ __forceinline__ void operator() (
        int chunk_size,
        TensorListScalarListMetadata<opmath_t, 1>& tl,
        Op op) {
            int tensor_loc = tl.block_to_tensor[blockIdx.x];
            int chunk_idx = tl.block_to_chunk[blockIdx.x];
            int n = tl.sizes[tensor_loc];

            T* x = (T*)tl.addresses[0][tensor_loc];
            x += chunk_idx * chunk_size;

            opmath_t y = tl.scalar_vals[tensor_loc];

            n -= chunk_idx * chunk_size;

            T r_x[kILP];

            // to make things simple, we put aligned case in a different code path
            if(n % kILP == 0 && chunk_size % kILP == 0 && is_aligned(x)) {
                for(int i_start = threadIdx.x; i_start * kILP < n && i_start * kILP < chunk_size; i_start += blockDim.x) {
                    // load
                    load_store(r_x, x, 0 , i_start);
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        r_x[ii] = static_cast<T>(op(static_cast<opmath_t>(r_x[ii]), y));
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
                        r_x[ii] = static_cast<T>(op(static_cast<opmath_t>(r_x[ii]), y));
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
struct BinaryOpScalarListFunctor {
    using io_t = T;
    using opmath_t = typename get_opmath_t<T>::opmath_t;
    template<typename Op> __device__ __forceinline__ void operator() (
        int chunk_size,
        TensorListScalarListMetadata<opmath_t, 2>& tl,
        Op op) {
            int tensor_loc = tl.block_to_tensor[blockIdx.x];
            int chunk_idx = tl.block_to_chunk[blockIdx.x];
            int n = tl.sizes[tensor_loc];

            T* x = (T*)tl.addresses[0][tensor_loc];
            x += chunk_idx * chunk_size;

            T* out = (T*)tl.addresses[1][tensor_loc];
            out += chunk_idx * chunk_size;

            opmath_t y = tl.scalar_vals[tensor_loc];

            n -= chunk_idx * chunk_size;

            T r_x[kILP];

            // to make things simple, we put aligned case in a different code path
            if(n % kILP == 0 && chunk_size % kILP == 0 && is_aligned(x) && is_aligned(out)) {
                for(int i_start = threadIdx.x; i_start * kILP < n && i_start * kILP < chunk_size; i_start += blockDim.x) {
                    // load
                    load_store(r_x, x, 0 , i_start);
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        r_x[ii] = static_cast<T>(op(static_cast<opmath_t>(r_x[ii]), y));
                    }
                    // store
                    load_store(out, r_x, i_start, 0);
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
                        r_x[ii] = static_cast<T>(op(static_cast<opmath_t>(r_x[ii]), y));
                    }
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        int i = i_start + threadIdx.x + ii * blockDim.x;
                        if(i < n && i < chunk_size)
                            out[i] = r_x[ii];
                    }
                }
            }
        }
};

template<typename T>
struct BinaryOpListAlphaFunctor_ {
    using opmath_t = typename get_opmath_t<T>::opmath_t;
    template<typename Op> __device__ __forceinline__ void operator() (
        int chunk_size,
        TensorListMetadata<2>& tl,
        Op op,
        opmath_t alpha) {
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
                        r_x[ii] = static_cast<T>(op(static_cast<opmath_t>(r_x[ii]),
                                                    alpha * static_cast<opmath_t>(r_y[ii])));
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
                        r_x[ii] = static_cast<T>(op(static_cast<opmath_t>(r_x[ii]),
                                                    alpha * static_cast<opmath_t>(r_y[ii])));
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
struct BinaryOpListAlphaFunctor {
    using opmath_t = typename get_opmath_t<T>::opmath_t;
    template<typename Op> __device__ __forceinline__ void operator() (
        int chunk_size,
        TensorListMetadata<3>& tl,
        Op op,
        opmath_t alpha) {
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

            // to make things simple, we put aligned case in a different code path
            if(n % kILP == 0 && chunk_size % kILP == 0 && is_aligned(x) && is_aligned(y) && is_aligned(out)) {
                for(int i_start = threadIdx.x; i_start * kILP < n && i_start * kILP < chunk_size; i_start += blockDim.x) {
                    // load
                    load_store(r_x, x, 0 , i_start);
                    load_store(r_y, y, 0 , i_start);
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        r_x[ii] = static_cast<T>(op(static_cast<opmath_t>(r_x[ii]),
                                                    alpha * static_cast<opmath_t>(r_y[ii])));
                    }
                    // store
                    load_store(out, r_x, i_start , 0);
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
                        r_x[ii] = static_cast<T>(op(static_cast<opmath_t>(r_x[ii]),
                                                    alpha * static_cast<opmath_t>(r_y[ii])));
                    }
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        int i = i_start + threadIdx.x + ii * blockDim.x;
                        if(i < n && i < chunk_size)
                            out[i] = r_x[ii];
                    }
                }
            }
        }
};

template<typename T>
struct UnaryOpFunctor_ {
    using opmath_t = typename get_opmath_t<T>::opmath_t;
    template<typename Op> __device__ __forceinline__ void operator() (
        int chunk_size,
        TensorListMetadata<1>& tl,
        Op op) {
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
                        r_x[ii] = static_cast<T>(op(static_cast<opmath_t>(r_x[ii])));
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
                        r_x[ii] = static_cast<T>(op(static_cast<opmath_t>(r_x[ii])));
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
struct UnaryOpFunctor {
    using opmath_t = typename get_opmath_t<T>::opmath_t;
    template<typename Op> __device__ __forceinline__ void operator() (
        int chunk_size,
        TensorListMetadata<2>& tl,
        Op op) {
            int tensor_loc = tl.block_to_tensor[blockIdx.x];
            int chunk_idx = tl.block_to_chunk[blockIdx.x];
            int n = tl.sizes[tensor_loc];

            T* x = (T*)tl.addresses[0][tensor_loc];
            x += chunk_idx * chunk_size;

            T* out = (T*)tl.addresses[1][tensor_loc];
            out += chunk_idx * chunk_size;

            n -= chunk_idx * chunk_size;

            T r_x[kILP];

            // to make things simple, we put aligned case in a different code path
            if(n % kILP == 0 && chunk_size % kILP == 0 && is_aligned(x) && is_aligned(out)) {
                for(int i_start = threadIdx.x; i_start * kILP < n && i_start * kILP < chunk_size; i_start += blockDim.x) {
                    // load
                    load_store(r_x, x, 0 , i_start);
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        r_x[ii] = static_cast<T>(op(static_cast<opmath_t>(r_x[ii])));
                    }
                    // store
                    load_store(out, r_x, i_start, 0);
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
                        r_x[ii] = static_cast<T>(op(static_cast<opmath_t>(r_x[ii])));
                    }
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        int i = i_start + threadIdx.x + ii * blockDim.x;
                        if(i < n && i < chunk_size)
                            out[i] = r_x[ii];
                    }
                }
            }
        }
};

template<typename T>
struct PointwiseOpFunctor_ {
    using opmath_t = typename get_opmath_t<T>::opmath_t;
    template<typename Op> __device__ __forceinline__ void operator() (
        int chunk_size,
        TensorListMetadata<3>& tl,
        Op op,
        opmath_t scalar) {
            int tensor_loc = tl.block_to_tensor[blockIdx.x];
            int chunk_idx = tl.block_to_chunk[blockIdx.x];
            int n = tl.sizes[tensor_loc];

            T* x = (T*)tl.addresses[0][tensor_loc];
            x += chunk_idx * chunk_size;

            T* y = (T*)tl.addresses[1][tensor_loc];
            y += chunk_idx * chunk_size;

            T* z = (T*)tl.addresses[2][tensor_loc];
            z += chunk_idx * chunk_size;

            n -= chunk_idx * chunk_size;

            T r_x[kILP];
            T r_y[kILP];
            T r_z[kILP];

            // to make things simple, we put aligned case in a different code path
            if(n % kILP == 0 && chunk_size % kILP == 0 && is_aligned(x) && is_aligned(y) && is_aligned(z)) {
                for(int i_start = threadIdx.x; i_start * kILP < n && i_start * kILP < chunk_size; i_start += blockDim.x) {
                    // load
                    load_store(r_x, x, 0 , i_start);
                    load_store(r_y, y, 0 , i_start);
                    load_store(r_z, z, 0 , i_start);
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        r_x[ii] = static_cast<T>(static_cast<opmath_t>(r_x[ii]) +
                                                 scalar * op(static_cast<opmath_t>(r_y[ii]),
                                                             static_cast<opmath_t>(r_z[ii])));
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
                        r_y[ii] = 0;
                        r_z[ii] = 0;
                        int i = i_start + threadIdx.x + ii * blockDim.x;
                        if(i < n && i < chunk_size) {
                            r_x[ii] = x[i];
                            r_y[ii] = y[i];
                            r_z[ii] = z[i];
                        }
                    }
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        r_x[ii] = static_cast<T>(static_cast<opmath_t>(r_x[ii]) +
                                                 scalar * op(static_cast<opmath_t>(r_y[ii]),
                                                             static_cast<opmath_t>(r_z[ii])));
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
struct PointwiseOpFunctor {
    using opmath_t = typename get_opmath_t<T>::opmath_t;
    template<typename Op> __device__ __forceinline__ void operator() (
        int chunk_size,
        TensorListMetadata<4>& tl,
        Op op,
        opmath_t scalar) {
            int tensor_loc = tl.block_to_tensor[blockIdx.x];
            int chunk_idx = tl.block_to_chunk[blockIdx.x];
            int n = tl.sizes[tensor_loc];

            T* x = (T*)tl.addresses[0][tensor_loc];
            x += chunk_idx * chunk_size;

            T* y = (T*)tl.addresses[1][tensor_loc];
            y += chunk_idx * chunk_size;

            T* z = (T*)tl.addresses[2][tensor_loc];
            z += chunk_idx * chunk_size;

            T* out = (T*)tl.addresses[3][tensor_loc];
            out += chunk_idx * chunk_size;

            n -= chunk_idx * chunk_size;

            T r_x[kILP];
            T r_y[kILP];
            T r_z[kILP];

            // to make things simple, we put aligned case in a different code path
            if(n % kILP == 0 && chunk_size % kILP == 0 && is_aligned(x) && is_aligned(y) && is_aligned(z) && is_aligned(out)) {
                for(int i_start = threadIdx.x; i_start * kILP < n && i_start * kILP < chunk_size; i_start += blockDim.x) {
                    // load
                    load_store(r_x, x, 0 , i_start);
                    load_store(r_y, y, 0 , i_start);
                    load_store(r_z, z, 0 , i_start);
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        r_x[ii] = static_cast<T>(static_cast<opmath_t>(r_x[ii]) +
                                                 scalar * op(static_cast<opmath_t>(r_y[ii]),
                                                             static_cast<opmath_t>(r_z[ii])));
                    }
                    // store
                    load_store(out, r_x, i_start, 0);
                }
            }
            else {
                for(int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * kILP) {
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        r_x[ii] = 0;
                        r_y[ii] = 0;
                        r_z[ii] = 0;

                        int i = i_start + threadIdx.x + ii * blockDim.x;
                        if(i < n && i < chunk_size) {
                            r_x[ii] = x[i];
                            r_y[ii] = y[i];
                            r_z[ii] = z[i];
                        }
                    }
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        r_x[ii] = static_cast<T>(static_cast<opmath_t>(r_x[ii]) +
                                                 scalar * op(static_cast<opmath_t>(r_y[ii]),
                                                             static_cast<opmath_t>(r_z[ii])));
                    }
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        int i = i_start + threadIdx.x + ii * blockDim.x;
                        if(i < n && i < chunk_size)
                            out[i] = r_x[ii];
                    }
                }
            }
        }
};

} // namespace

}} // namespace at::native
