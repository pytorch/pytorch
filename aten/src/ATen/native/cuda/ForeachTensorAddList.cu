#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>

namespace at { namespace native {

std::vector<Tensor> foreach_tensor_add_list_kernel_cuda(TensorList tensors1, TensorList tensors2) {
    check_foreach_api_restrictions(tensors1, tensors2);

    if (!can_use_fast_route(tensors1, tensors2)) {
        return at::native::foreach_tensor_add_list_kernel_slow(tensors1, tensors2);
    }

    std::vector<std::vector<at::Tensor>> tensor_lists; 
    std::vector<at::Tensor> vec_res;
    vec_res.reserve(tensors1.size());

    for (const auto& t: tensors1) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    tensor_lists.emplace_back(tensors1.vec());
    tensor_lists.emplace_back(tensors2.vec());
    tensor_lists.emplace_back(vec_res);

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, tensors1[0].scalar_type(), "foreach_tensor_add_list_kernel_cuda", [&]() {
        multi_tensor_apply<3>(tensor_lists, AddListFunctor<scalar_t>());
    });

    return tensor_lists[2];
}

void foreach_tensor_add_list_kernel_cuda_(TensorList tensors1, TensorList tensors2) {
    check_foreach_api_restrictions(tensors1, tensors2);

    if (!can_use_fast_route(tensors1, tensors2)) {
        return at::native::foreach_tensor_add_list_kernel_slow_(tensors1, tensors2);
    }

    std::vector<std::vector<at::Tensor>> tensor_lists; 
    tensor_lists.emplace_back(tensors1.vec());
    tensor_lists.emplace_back(tensors2.vec());

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, tensors1[0].scalar_type(), "foreach_tensor_add_list_kernel_cuda_", [&]() {
        multi_tensor_apply<2>(tensor_lists, AddListFunctor_<scalar_t>());
    });
}

}} // namespace at::native
