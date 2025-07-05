#include <ATen/core/Tensor.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/tanh.h>
#include <ATen/autocast_mode.h>

namespace at {
namespace native {

std::tuple<Tensor, Tensor> attn(const Tensor& Q, const Tensor& K, const Tensor& V) {
    TORCH_CHECK(Q.dim() == 2, "Expected Q of dimension 2. Got ", Q.dim(), "-D tensor");
    TORCH_CHECK(K.dim() == 2, "Expected K of dimension 2. Got ", K.dim(), "-D tensor");
    TORCH_CHECK(V.dim() == 2, "Expected V of dimension 2. Got ", V.dim(), "-D tensor");
    TORCH_CHECK(
        Q.sym_sizes()[0] == K.sym_sizes()[0] && K.sym_sizes()[0] == Q.sym_sizes()[0],
        "Expected Q, K, V to all have the same first dimension. Got Q: ", Q.sym_sizes()[0],
        " K: ", K.sym_sizes()[0],
        " V: ", V.sym_sizes()[0]
    );
    TORCH_CHECK(
        Q.sym_sizes()[1] == K.sym_sizes()[1],
        "Expected Q, K to all have the same second dimension. Got Q: ", Q.sym_sizes()[1],
        " K: ", K.sym_sizes()[1],
        " V: ", V.sym_sizes()[1]
    );

    at::autocast::set_autocast_enabled(at::kCPU, false);

    at::Tensor B = at::matmul(Q, K.transpose(0,1));
    at::Tensor A = at::tanh(B);
    at::Tensor O = at::matmul(A, V);

    return std::tuple<at::Tensor, at::Tensor>(O, A);
}
}
}
