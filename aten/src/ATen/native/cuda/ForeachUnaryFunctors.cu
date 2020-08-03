#include <ATen/Dispatch.h>
#include <ATen/native/cuda/ForeachUtils.cuh>
#include <ATen/native/cuda/MultiTensorApply.cuh>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

namespace {

template<typename x_t, template<class> class Op>
struct UnaryOpFunctor_ {
    __device__ void operator() (
        int chunk_size,
        TensorListMetadata<1>& tl) {
            int tensor_loc = tl.block_to_tensor[blockIdx.x];
            int chunk_idx = tl.block_to_chunk[blockIdx.x];
            int n = tl.sizes[tensor_loc];

            x_t* x = (x_t*)tl.addresses[0][tensor_loc];
            x += chunk_idx * chunk_size;

            n -= chunk_idx * chunk_size;

            x_t r_x[kILP];

            // to make things simple, we put aligned case in a different code path
            if(n % kILP == 0 && chunk_size % kILP == 0 && is_aligned(x)) {
                for(int i_start = threadIdx.x; i_start * kILP < n && i_start * kILP < chunk_size; i_start += blockDim.x) {
                    // load
                    load_store(r_x, x, 0 , i_start);
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        r_x[ii] = Op<x_t>()(static_cast<x_t>(r_x[ii]));
                    }
                    // store
                    load_store(x, r_x, i_start, 0);
                }
            }
            else {
                // Non-divergent exit condition for __syncthreads, not necessary here
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
                        r_x[ii] = Op<x_t>()(static_cast<x_t>(r_x[ii]));
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

template<typename x_t, typename out_t, template<class> class Op>
struct UnaryOpFunctor {
    __device__ void operator() (
        int chunk_size,
        TensorListMetadata<2>& tl) {
            int tensor_loc = tl.block_to_tensor[blockIdx.x];
            int chunk_idx = tl.block_to_chunk[blockIdx.x];
            int n = tl.sizes[tensor_loc];

            x_t* x = (x_t*)tl.addresses[0][tensor_loc];
            x += chunk_idx * chunk_size;

            out_t* out = (out_t*)tl.addresses[1][tensor_loc];
            out += chunk_idx * chunk_size;

            n -= chunk_idx * chunk_size;

            x_t r_x[kILP];
            out_t r_out[kILP];

            // to make things simple, we put aligned case in a different code path
            if(n % kILP == 0 && chunk_size % kILP == 0 && is_aligned(x) && is_aligned(out)) {
                for(int i_start = threadIdx.x; i_start * kILP < n && i_start * kILP < chunk_size; i_start += blockDim.x) {
                    // load
                    load_store(r_x, x, 0 , i_start);
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        r_out[ii] = Op<x_t>()(static_cast<x_t>(r_x[ii]));
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
                        int i = i_start + threadIdx.x + ii * blockDim.x;
                        if(i < n && i < chunk_size) {
                            r_x[ii] = x[i];
                        }
                    }
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++) {
                        r_out[ii] = Op<x_t>()(static_cast<x_t>(r_x[ii]));
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
std::vector<Tensor> foreach_unary_op(TensorList tensors) {
    std::vector<std::vector<at::Tensor>> tensor_lists; 
    std::vector<at::Tensor> vec_res;
    for (int i = 0; i < tensors.size(); i++) {
        vec_res.emplace_back(at::native::empty_like(tensors[i]));
    }

    tensor_lists.emplace_back(std::move(tensors.vec()));
    tensor_lists.emplace_back(std::move(vec_res));

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, tensors[0].scalar_type(), "foreach_tensor_add_scalar_kernel_cuda", [&]() {
        multi_tensor_apply<2>(tensor_lists, UnaryOpFunctor<scalar_t, scalar_t, Op>());
    });
    return tensor_lists[1];
}

template<template<class> class Op>
std::vector<Tensor> foreach_unary_op_(TensorList tensors) {
    std::vector<std::vector<at::Tensor>> tensor_lists; 
    tensor_lists.emplace_back(std::move(tensors.vec()));

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, tensors[0].scalar_type(), "foreach_tensor_add_scalar__kernel_cuda", [&]() {
        multi_tensor_apply<1>(tensor_lists, UnaryOpFunctor_<scalar_t, Op>());
    });
    return tensor_lists[0];
}

std::vector<Tensor> foreach_tensor_exp_cuda(TensorList tensors) {
    TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");

    if (!check_fast_route(tensors)) {
        return at::native::foreach_tensor_exp_cpu(tensors);
    }

    return foreach_unary_op<std::exp>(tensors);
}

std::vector<Tensor> foreach_tensor_exp__cuda(TensorList tensors) {
    TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");

    if (!check_fast_route(tensors)) {
        return at::native::foreach_tensor_exp__cpu(tensors);
    }

    return foreach_unary_op_<std::exp>(tensors);
}

std::vector<Tensor> foreach_tensor_sqrt_cuda(TensorList tensors) {
    TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");

    if (!check_fast_route(tensors)) {
        return at::native::foreach_tensor_sqrt_cpu(tensors);
    }

    return foreach_unary_op<std::sqrt>(tensors);
}

std::vector<Tensor> foreach_tensor_sqrt__cuda(TensorList tensors) {
    TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");

    if (!check_fast_route(tensors)) {
        return at::native::foreach_tensor_sqrt__cpu(tensors);
    }

    return foreach_unary_op_<std::sqrt>(tensors);
}

}} // namespace at::native
