#include <ATen/Dispatch.h>
#include <c10/macros/Macros.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/foreach/Utils.cuh>
#include <ATen/native/cuda/foreach/MultiTensorApply.cuh>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

namespace {

template<typename x_t, typename out_t>
struct AddScalarFunctor {
    __device__ void operator() (
        int chunk_size,
        TensorListMetadata<2>& tl,
        x_t scalar) {
            int tensor_loc = tl.block_to_tensor[blockIdx.x];
            int chunk_idx = tl.block_to_chunk[blockIdx.x];
            int n = tl.sizes[tensor_loc];

            x_t* x = (x_t*)tl.addresses[0][tensor_loc];
            x += chunk_idx * chunk_size;

            out_t* out = (out_t*)tl.addresses[1][tensor_loc];
            out += chunk_idx * chunk_size;

            n -= chunk_idx * chunk_size;

            x_t r_x[ILP];
            out_t r_out[ILP];

            // to make things simple, we put aligned case in a different code path
            if(n % ILP == 0 && chunk_size % ILP == 0 && is_aligned(x) && is_aligned(out)) {
                for(int i_start = threadIdx.x; i_start * ILP < n && i_start * ILP < chunk_size; i_start += blockDim.x) {
                    // load
                    load_store(r_x, x, 0 , i_start);
#pragma unroll
                    for(int ii = 0; ii < ILP; ii++) {
                        r_out[ii] = static_cast<x_t>(r_x[ii]) + scalar;
                    }
                    // store
                    load_store(out, r_out, i_start, 0);
                }
            }
            else {
                // Non-divergent exit condition for __syncthreads, not necessary here
                for(int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * ILP) {
#pragma unroll
                    for(int ii = 0; ii < ILP; ii++) {
                        r_x[ii] = 0;
                        int i = i_start + threadIdx.x + ii * blockDim.x;
                        if(i < n && i < chunk_size) {
                            r_x[ii] = x[i];
                        }
                    }
#pragma unroll
                    for(int ii = 0; ii < ILP; ii++) {
                        r_out[ii] = static_cast<x_t>(r_x[ii]) + scalar;
                    }
#pragma unroll
                    for(int ii = 0; ii < ILP; ii++) {
                        int i = i_start + threadIdx.x + ii * blockDim.x;
                        if(i < n && i < chunk_size)
                            out[i] = r_out[ii];
                    }
                }
            }
        }
};

} // namespace

std::vector<Tensor> foreach_tensor_add_scalar_kernel_cuda(TensorList tensors, Scalar scalar) {
    if (tensors.size() == 0) {
        return std::move(tensors.vec());
    }

    TORCH_CHECK(std::all_of(tensors.begin(), tensors.end(), [] (const Tensor& t) {
        return t.layout() == at::kStrided;
    }), "Only tensors with strided layouts are supported.");

    TORCH_CHECK(std::all_of(tensors.begin(), tensors.end(), [] (const Tensor& t) {
        return t.is_non_overlapping_and_dense();
    }), "Only non overlapping and dense tensors are supported.");

    std::vector<std::vector<at::Tensor>> tensor_lists; 
    std::vector<at::Tensor> vec_res;
    for (int i = 0; i < tensors.size(); i++) {
        vec_res.emplace_back(torch::empty_like(tensors[i]));
    }

    tensor_lists.emplace_back(std::move(tensors.vec()));
    tensor_lists.emplace_back(std::move(vec_res));

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, tensors[0].scalar_type(), "foreach_tensor_add_scalar_kernel_cuda", [&]() {
        multi_tensor_apply<2>(tensor_lists, AddScalarFunctor<scalar_t, scalar_t>(), scalar.to<scalar_t>());
    });
    return tensor_lists[1];
}

}} // namespace at::native
