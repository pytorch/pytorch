#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>

namespace at { namespace native {

template <template<class> class Op>
std::vector<Tensor> foreach_unary_op(TensorList tensors) {
    std::vector<std::vector<at::Tensor>> tensor_lists; 
    std::vector<at::Tensor> vec_res;
    for (const auto& t: tensors) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    tensor_lists.emplace_back(std::move(tensors.vec()));
    tensor_lists.emplace_back(std::move(vec_res));

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half,  tensors[0].scalar_type(), "foreach_unary_op_cuda", [&]() {
        multi_tensor_apply<2>(tensor_lists, UnaryOpFunctor<scalar_t, Op>());
    });
    return tensor_lists[1];
}

template <template<class> class Op>
void foreach_unary_op_(TensorList tensors) {
    std::vector<std::vector<at::Tensor>> tensor_lists; 
    tensor_lists.emplace_back(std::move(tensors.vec()));

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, tensors[0].scalar_type(), "foreach_unary_op_cuda_", [&]() {
        multi_tensor_apply<1>(tensor_lists, UnaryOpFunctor_<scalar_t, Op>());
    });
}

template<typename T>
struct Sqrt {
    __device__ T operator()(T t) const { return std::sqrt(t); }
};

template<typename T>
struct Exp {
    __device__ T operator()(T t) const { return std::exp(t); }
};

std::vector<Tensor> foreach_tensor_exp_cuda(TensorList tensors) {
    check_foreach_api_restrictions(tensors);

    if (!check_fast_route(tensors)) {
        return at::native::foreach_tensor_exp_slow(tensors);
    }
    
    return foreach_unary_op<Exp>(tensors);
}

void foreach_tensor_exp_cuda_(TensorList tensors) {
    check_foreach_api_restrictions(tensors);

    if (!check_fast_route(tensors)) {
        return at::native::foreach_tensor_exp_slow_(tensors);
    }

    foreach_unary_op_<Exp>(tensors);
}

std::vector<Tensor> foreach_tensor_sqrt_cuda(TensorList tensors) {
    check_foreach_api_restrictions(tensors);

    if (!check_fast_route(tensors)) {
        return at::native::foreach_tensor_sqrt_slow(tensors);
    }

    return foreach_unary_op<Sqrt>(tensors);

}

void foreach_tensor_sqrt_cuda_(TensorList tensors) {
    check_foreach_api_restrictions(tensors);

    if (!check_fast_route(tensors)) {
        return at::native::foreach_tensor_sqrt_slow_(tensors);
    }

    foreach_unary_op_<Sqrt>(tensors);
}

}} // namespace at::native
