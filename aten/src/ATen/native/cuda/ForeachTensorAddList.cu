#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>

namespace at { namespace native {

std::vector<Tensor> foreach_tensor_add_list_kernel_cuda(TensorList tensors1, TensorList tensors2) {
    verify_list(tensors1, tensors2);

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
        multi_tensor_apply<3>(tensor_lists, AddListFunctor<scalar_t>());
    });

    return tensor_lists[2];
}

std::vector<Tensor> foreach_tensor_add_list_kernel_cuda_(TensorList tensors1, TensorList tensors2) {
    verify_list(tensors1, tensors2);

    if (!check_fast_route(tensors1, tensors2)) {
        return at::native::foreach_add_list_kernel_fallback_(tensors1, tensors2);
    }

    std::vector<std::vector<at::Tensor>> tensor_lists; 
    tensor_lists.emplace_back(std::move(tensors1.vec()));
    tensor_lists.emplace_back(std::move(tensors2.vec()));

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, tensors1[0].scalar_type(), "foreach_tensor_add_list__kernel_cuda", [&]() {
        multi_tensor_apply<2>(tensor_lists, AddListFunctor_<scalar_t>());
    });

    return tensor_lists[0];
}

}} // namespace at::native
