#include <ATen/Dispatch.h>
#include <ATen/native/cuda/ForeachUtils.cuh>
#include <ATen/native/cuda/MultiTensorApply.cuh>

namespace at { namespace native {

namespace {

template<typename x_t, typename y_t>
struct AddListFunctor_
{
    __device__ void operator() (
        int chunk_size,
        TensorListMetadata<2>& tl) 
        {
            int tensor_loc = tl.block_to_tensor[blockIdx.x];
            int chunk_idx = tl.block_to_chunk[blockIdx.x];
            int n = tl.sizes[tensor_loc];

            x_t* x = (x_t*)tl.addresses[0][tensor_loc];
            x += chunk_idx * chunk_size;

            y_t* y = (y_t*)tl.addresses[1][tensor_loc];
            y += chunk_idx * chunk_size;

            n -= chunk_idx * chunk_size;

            x_t r_x[kILP];
            y_t r_y[kILP];

            // to make things simple, we put aligned case in a different code path
            if(n % kILP == 0 && chunk_size % kILP == 0 && is_aligned(x) && is_aligned(y))
            {
                for(int i_start = threadIdx.x; i_start * kILP < n && i_start * kILP < chunk_size; i_start += blockDim.x)
                {
                    // load
                    load_store(r_x, x, 0 , i_start);
                    load_store(r_y, y, 0 , i_start);
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++)
                    {
                        r_x[ii] = static_cast<x_t>(r_x[ii]) + static_cast<y_t>(r_y[ii]);
                    }
                    // store
                    load_store(x, r_x, i_start , 0);
                }
            }
            else
            {
                // Non-divergent exit condition for __syncthreads, not necessary here
                for(int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * kILP)
                {
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++)
                    {
                        r_x[ii] = 0;
                        r_y[ii] = 0;
                        int i = i_start + threadIdx.x + ii * blockDim.x;
                        if(i < n && i < chunk_size)
                        {
                            r_x[ii] = x[i];
                            r_y[ii] = y[i];
                        }
                    }
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++)
                    {
                        r_x[ii] = static_cast<x_t>(r_x[ii]) + static_cast<y_t>(r_y[ii]);
                    }
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++)
                    {
                        int i = i_start + threadIdx.x + ii * blockDim.x;
                        if(i < n && i < chunk_size)
                            x[i] = r_x[ii];
                    }
                }
            }
        }
};

template<typename x_t, typename y_t, typename out_t>
struct AddListFunctor
{
    __device__ void operator() (
        int chunk_size,
        TensorListMetadata<3>& tl) 
        {
            int tensor_loc = tl.block_to_tensor[blockIdx.x];
            int chunk_idx = tl.block_to_chunk[blockIdx.x];
            int n = tl.sizes[tensor_loc];

            x_t* x = (x_t*)tl.addresses[0][tensor_loc];
            x += chunk_idx * chunk_size;

            y_t* y = (y_t*)tl.addresses[1][tensor_loc];
            y += chunk_idx * chunk_size;

            out_t* out = (out_t*)tl.addresses[2][tensor_loc];
            out += chunk_idx * chunk_size;

            n -= chunk_idx * chunk_size;

            x_t r_x[kILP];
            y_t r_y[kILP];
            out_t r_out[kILP];

            // to make things simple, we put aligned case in a different code path
            if(n % kILP == 0 && chunk_size % kILP == 0 && is_aligned(x) && is_aligned(y) && is_aligned(out))
            {
                for(int i_start = threadIdx.x; i_start * kILP < n && i_start * kILP < chunk_size; i_start += blockDim.x)
                {
                    // load
                    load_store(r_x, x, 0 , i_start);
                    load_store(r_y, y, 0 , i_start);
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++)
                    {
                        r_out[ii] = static_cast<x_t>(r_x[ii]) + static_cast<y_t>(r_y[ii]);
                    }
                    // store
                    load_store(out, r_out, i_start , 0);
                }
            }
            else
            {
                // Non-divergent exit condition for __syncthreads, not necessary here
                for(int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * kILP)
                {
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++)
                    {
                        r_x[ii] = 0;
                        r_y[ii] = 0;
                        int i = i_start + threadIdx.x + ii * blockDim.x;
                        if(i < n && i < chunk_size)
                        {
                            r_x[ii] = x[i];
                            r_y[ii] = y[i];
                        }
                    }
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++)
                    {
                        r_out[ii] = static_cast<x_t>(r_x[ii]) + static_cast<y_t>(r_y[ii]);
                    }
#pragma unroll
                    for(int ii = 0; ii < kILP; ii++)
                    {
                        int i = i_start + threadIdx.x + ii * blockDim.x;
                        if(i < n && i < chunk_size)
                            out[i] = r_out[ii];
                    }
                }
            }
        }
};

} // namespace

std::vector<Tensor> foreach_tensor_add_list_kernel_cuda(TensorList tensors1, TensorList tensors2) {
    TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
    TORCH_CHECK(tensors1.size() ==  tensors2.size(), "Tensor lists must be of the same length.");

    if (!check_fast_route(tensors1, tensors2)) {
        return at::native::foreach_add_list_kernel_fallback(tensors1, tensors2);
    }

    std::vector<std::vector<at::Tensor>> tensor_lists; 
    std::vector<at::Tensor> vec_res;
    for (const auto& t: tensors1) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    tensor_lists.emplace_back(std::move(tensors1.vec()));
    tensor_lists.emplace_back(std::move(tensors2.vec()));
    tensor_lists.emplace_back(std::move(vec_res));

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, tensors1[0].scalar_type(), "foreach_tensor_add_list_kernel_cuda", [&]() {
        multi_tensor_apply<3>(tensor_lists, AddListFunctor<scalar_t, scalar_t, scalar_t>());
    });

    return tensor_lists[2];
}

std::vector<Tensor> foreach_tensor_add_list__kernel_cuda(TensorList tensors1, TensorList tensors2) {
    TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
    TORCH_CHECK(tensors1.size() ==  tensors2.size(), "Tensor lists must be of the same length.");

    if (!check_fast_route(tensors1, tensors2)) {
        return at::native::foreach_add_list__kernel_fallback(tensors1, tensors2);
    }

    std::vector<std::vector<at::Tensor>> tensor_lists; 
    tensor_lists.emplace_back(std::move(tensors1.vec()));
    tensor_lists.emplace_back(std::move(tensors2.vec()));

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, tensors1[0].scalar_type(), "foreach_tensor_add_list__kernel_cuda", [&]() {
        multi_tensor_apply<2>(tensor_lists, AddListFunctor_<scalar_t, scalar_t>());
    });

    return tensor_lists[0];
}

}} // namespace at::native
