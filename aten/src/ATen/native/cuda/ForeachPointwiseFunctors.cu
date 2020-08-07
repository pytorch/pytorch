#include <ATen/Dispatch.h>
#include <ATen/native/cuda/ForeachUtils.cuh>
#include <ATen/native/cuda/MultiTensorApply.cuh>

namespace at { namespace native {

namespace {

template<typename x_t, template<class> class Op>
struct PointwiseOpFunctor_ {
    __device__ void operator() (
        int chunk_size,
        TensorListMetadata<3>& tl,
        x_t scalar) {
            int tensor_loc = tl.block_to_tensor[blockIdx.x];
            int chunk_idx = tl.block_to_chunk[blockIdx.x];
            int n = tl.sizes[tensor_loc];

            x_t* x = (x_t*)tl.addresses[0][tensor_loc];
            x += chunk_idx * chunk_size;
            
            x_t* y = (x_t*)tl.addresses[1][tensor_loc];
            y += chunk_idx * chunk_size;

            x_t* z = (x_t*)tl.addresses[2][tensor_loc];
            z += chunk_idx * chunk_size;

            n -= chunk_idx * chunk_size;

            x_t r_x[kILP];
            x_t r_y[kILP];
            x_t r_z[kILP];

            // to make things simple, we put aligned case in a different code path
            if(n % kILP == 0 && chunk_size % kILP == 0 && is_aligned(x) && is_aligned(y) && is_aligned(z)) {
                for(int i_start = threadIdx.x; i_start * kILP < n && i_start * kILP < chunk_size; i_start += blockDim.x) {
                    // load
                    load_store(r_x, x, 0 , i_start);
                    load_store(r_y, y, 0 , i_start);
                    load_store(r_z, z, 0 , i_start);
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        r_x[ii] = static_cast<x_t>(r_x[ii]) + scalar * Op<x_t>()(static_cast<x_t>(r_y[ii]), static_cast<x_t>(r_z[ii]));
                    }
                    // store
                    load_store(x, r_x, i_start, 0);
                    load_store(y, r_y, i_start, 0);
                    load_store(z, r_z, i_start, 0);
                }
            }
            else {
                // Non-divergent exit condition for __syncthreads, not necessary here
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
                        r_x[ii] = static_cast<x_t>(r_x[ii]) + scalar * Op<x_t>()(static_cast<x_t>(r_y[ii]), static_cast<x_t>(r_z[ii]));
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

template<typename x_t, template<class> class Op>
struct PointwiseOpFunctor {
    __device__ void operator() (
        int chunk_size,
        TensorListMetadata<4>& tl,
        x_t scalar) {
            int tensor_loc = tl.block_to_tensor[blockIdx.x];
            int chunk_idx = tl.block_to_chunk[blockIdx.x];
            int n = tl.sizes[tensor_loc];

            x_t* x = (x_t*)tl.addresses[0][tensor_loc];
            x += chunk_idx * chunk_size;

            x_t* y = (x_t*)tl.addresses[1][tensor_loc];
            y += chunk_idx * chunk_size;

            x_t* z = (x_t*)tl.addresses[2][tensor_loc];
            z += chunk_idx * chunk_size;

            x_t* out = (x_t*)tl.addresses[3][tensor_loc];
            out += chunk_idx * chunk_size;

            n -= chunk_idx * chunk_size;

            x_t r_x[kILP];
            x_t r_y[kILP];
            x_t r_z[kILP];
            x_t r_out[kILP];

            // to make things simple, we put aligned case in a different code path
            if(n % kILP == 0 && chunk_size % kILP == 0 && is_aligned(x) && is_aligned(y) && is_aligned(z) && is_aligned(out)) {
                for(int i_start = threadIdx.x; i_start * kILP < n && i_start * kILP < chunk_size; i_start += blockDim.x) {
                    // load
                    load_store(r_x, x, 0 , i_start);
                    load_store(r_y, y, 0 , i_start);
                    load_store(r_z, z, 0 , i_start);
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        r_out[ii] = static_cast<x_t>(r_x[ii]) + scalar * Op<x_t>()(static_cast<x_t>(r_y[ii]), static_cast<x_t>(r_z[ii]));
                    }
                    // store
                    load_store(out, r_out, i_start, 0);
                }
            }
            else {
                // Non-divergent exit condition for __syncthreads, not necessary here
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
                        r_out[ii] = static_cast<x_t>(r_x[ii]) + scalar * Op<x_t>()(static_cast<x_t>(r_y[ii]), static_cast<x_t>(r_z[ii]));
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

template<template<class> class Op>
std::vector<Tensor> foreach_pointwise_op(TensorList input, TensorList tensors1, TensorList tensors2, Scalar scalar) {
    std::vector<std::vector<at::Tensor>> tensor_lists; 
    std::vector<at::Tensor> vec_res;
    for (const auto& t: input) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    tensor_lists.emplace_back(std::move(input.vec()));
    tensor_lists.emplace_back(std::move(tensors1.vec()));
    tensor_lists.emplace_back(std::move(tensors2.vec()));
    tensor_lists.emplace_back(std::move(vec_res));

    AT_DISPATCH_ALL_TYPES_AND(kHalf, input[0].scalar_type(), "foreach_pointwise_op_cuda", [&]() {
        multi_tensor_apply<4>(tensor_lists, PointwiseOpFunctor<scalar_t, Op>(), scalar.to<scalar_t>());
    });

    return tensor_lists[3];
}

template<template<class> class Op>
std::vector<Tensor> foreach_pointwise_op_(TensorList input, TensorList tensors1, TensorList tensors2, Scalar scalar) {
    std::vector<std::vector<at::Tensor>> tensor_lists; 
    tensor_lists.emplace_back(std::move(input.vec()));
    tensor_lists.emplace_back(std::move(tensors1.vec()));
    tensor_lists.emplace_back(std::move(tensors2.vec()));

    AT_DISPATCH_ALL_TYPES_AND(kHalf, input[0].scalar_type(), "foreach_pointwise_op__cuda", [&]() {
        multi_tensor_apply<3>(tensor_lists, PointwiseOpFunctor_<scalar_t, Op>(), scalar.to<scalar_t>());
    });

    return tensor_lists[2];
}

std::vector<Tensor> foreach_tensor_addcdiv_cuda(TensorList input, TensorList tensors1, TensorList tensors2, Scalar scalar) {
    TORCH_CHECK(input.size() > 0, "Tensor list must have at least one tensor.");
    TORCH_CHECK(input.size() ==  tensors1.size(), "Tensor lists must be of the same length.");
    TORCH_CHECK(tensors1.size() ==  tensors2.size(), "Tensor lists must be of the same length.");

    if (!check_fast_route(input, scalar) ||
        !check_fast_route(tensors1, tensors2) ||
        !check_fast_route(input, tensors1)) {
        return at::native::foreach_addcdiv_fallback(input, tensors1, tensors2, scalar);
    }

    return foreach_pointwise_op<std::divides>(input, tensors1, tensors2, scalar);
}

std::vector<Tensor> foreach_tensor_addcdiv__cuda(TensorList input, TensorList tensors1, TensorList tensors2, Scalar scalar) {
    TORCH_CHECK(input.size() > 0, "Tensor list must have at least one tensor.");
    TORCH_CHECK(input.size() ==  tensors1.size(), "Tensor lists must be of the same length.");
    TORCH_CHECK(tensors1.size() ==  tensors2.size(), "Tensor lists must be of the same length.");

    if (!check_fast_route(input, scalar) ||
        !check_fast_route(tensors1, tensors2) ||
        !check_fast_route(input, tensors1)) {
        return at::native::foreach_addcdiv__fallback(input, tensors1, tensors2, scalar);
    }

    return foreach_pointwise_op_<std::divides>(input, tensors1, tensors2, scalar);
}

std::vector<Tensor> foreach_tensor_addcmul_cuda(TensorList input, TensorList tensors1, TensorList tensors2, Scalar scalar) {
    TORCH_CHECK(input.size() > 0, "Tensor list must have at least one tensor.");
    TORCH_CHECK(input.size() ==  tensors1.size(), "Tensor lists must be of the same length.");
    TORCH_CHECK(tensors1.size() ==  tensors2.size(), "Tensor lists must be of the same length.");

    if (!check_fast_route(input, scalar) ||
        !check_fast_route(tensors1, tensors2) ||
        !check_fast_route(input, tensors1)) {
        return at::native::foreach_addcmul_fallback(input, tensors1, tensors2, scalar);
    }

    return foreach_pointwise_op<std::multiplies>(input, tensors1, tensors2, scalar);
}

std::vector<Tensor> foreach_tensor_addcmul__cuda(TensorList input, TensorList tensors1, TensorList tensors2, Scalar scalar) {
    TORCH_CHECK(input.size() > 0, "Tensor list must have at least one tensor.");
    TORCH_CHECK(input.size() ==  tensors1.size(), "Tensor lists must be of the same length.");
    TORCH_CHECK(tensors1.size() ==  tensors2.size(), "Tensor lists must be of the same length.");

    if (!check_fast_route(input, scalar) ||
        !check_fast_route(tensors1, tensors2) ||
        !check_fast_route(input, tensors1)) {
        return at::native::foreach_addcmul__fallback(input, tensors1, tensors2, scalar);
    }

    return foreach_pointwise_op_<std::multiplies>(input, tensors1, tensors2, scalar);
}

}} // namespace at::native
