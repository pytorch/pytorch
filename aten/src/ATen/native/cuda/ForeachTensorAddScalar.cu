#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>

namespace at { namespace native {

std::vector<Tensor> foreach_tensor_add_scalar_kernel_cuda(TensorList tensors, Scalar scalar) {
    check_foreach_api_restrictions(tensors);

    if (!can_use_fast_route(tensors, scalar)) {
        return at::native::foreach_tensor_add_scalar_kernel_slow(tensors, scalar);
    }

    std::vector<std::vector<at::Tensor>> tensor_lists; 
    std::vector<at::Tensor> vec_res;
    for (const auto& t: tensors) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    tensor_lists.emplace_back(tensors.vec());
    tensor_lists.emplace_back(vec_res);

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, tensors[0].scalar_type(), "foreach_tensor_add_scalar_kernel_cuda", [&]() {
        multi_tensor_apply<2>(tensor_lists, AddScalarFunctor<scalar_t>(), scalar.to<scalar_t>());
    });
    return tensor_lists[1];
}

void foreach_tensor_add_scalar_kernel_cuda_(TensorList tensors, Scalar scalar) {
    check_foreach_api_restrictions(tensors);

    if (!can_use_fast_route(tensors, scalar)) {
        return at::native::foreach_tensor_add_scalar_kernel_slow_(tensors, scalar);
    }

    std::vector<std::vector<at::Tensor>> tensor_lists; 
    tensor_lists.emplace_back(tensors.vec());

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, tensors[0].scalar_type(), "foreach_tensor_add_scalar_kernel_cuda_", [&]() {
        multi_tensor_apply<1>(tensor_lists, AddScalarFunctor_<scalar_t>(), scalar.to<scalar_t>());
    });
}

}} // namespace at::native
