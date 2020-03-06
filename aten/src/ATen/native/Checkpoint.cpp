#include <ATen/ATen.h>
#include <ATen/CheckpointTensorImpl.h>

namespace at { namespace native {

Tensor checkpoint_add(const Tensor& a, const Tensor& b, c10::Scalar c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::add(vec.at(0), vec.at(1), c)};
    };
  return CheckpointTensorImpl::make("add", rt, {a, b})[0];
}

Tensor checkpoint_t(at::Tensor const& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::t(vec.at(0))};
    };
  return CheckpointTensorImpl::make("t", rt, {a})[0];
}

Tensor checkpoint_add(at::Tensor const& a, c10::Scalar b, c10::Scalar c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::add(vec.at(0), b, c)};
    };
  return CheckpointTensorImpl::make("add", rt, {a})[0];
}

Tensor& checkpoint_add_(Tensor& a, const Tensor& b, Scalar c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).add_(vec.at(1), c);
    };
  CheckpointTensorImpl::mutate("add_", mt, {a, b}, {0});
  return a;
}

Tensor checkpoint_mul(at::Tensor const& a, at::Tensor const& b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::mul(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("mul", rt, {a, b})[0];
}

Tensor& checkpoint_mul_(at::Tensor& a, at::Tensor const& b) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).mul_(vec.at(1));
    };
  CheckpointTensorImpl::mutate("mul_", mt, {a, b}, {0});
  return a;
}

Tensor& checkpoint_mul_(at::Tensor& a, c10::Scalar b) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).mul_(b);
    };
  CheckpointTensorImpl::mutate("mul_", mt, {a}, {0});
  return a;
}

Tensor checkpoint_zeros_like(at::Tensor const& a, c10::TensorOptions const& b, c10::optional<c10::MemoryFormat> c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::zeros_like(vec.at(0), b, c)};
    };
  return CheckpointTensorImpl::make("zeros_like", rt, {a})[0];
}

Tensor checkpoint_ones_like(at::Tensor const& a, c10::TensorOptions const& b, c10::optional<c10::MemoryFormat> c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::ones_like(vec.at(0), b, c)};
    };
  return CheckpointTensorImpl::make("ones_like", rt, {a})[0];
}

Tensor checkpoint_addcmul(at::Tensor const& a, at::Tensor const& b, at::Tensor const& c, c10::Scalar d) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::addcmul(vec.at(0), vec.at(1), vec.at(2), d)};
    };
  return CheckpointTensorImpl::make("addcmul", rt, {a, b, c})[0];
}

Tensor& checkpoint_addcmul_(at::Tensor& a, at::Tensor const& b, at::Tensor const& c, c10::Scalar d) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).addcmul_(vec.at(1), vec.at(2), d);
    };
  CheckpointTensorImpl::mutate("addcmul_", mt, {a, b, c}, {0});
  return a;
}

Tensor checkpoint_abs(const Tensor& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::abs(vec.at(0))};
    };
  return CheckpointTensorImpl::make("abs", rt, {a})[0];
}

Tensor checkpoint_sqrt(const Tensor& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sqrt(vec.at(0))};
    };
  return CheckpointTensorImpl::make("sqrt", rt, {a})[0];
}

Tensor& checkpoint_addcdiv_(at::Tensor& a, at::Tensor const& b, at::Tensor const& c, c10::Scalar d) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).addcdiv_(vec.at(1), vec.at(2), d);
    };
  CheckpointTensorImpl::mutate("addcdiv_", mt, {a, b, c}, {0});
  return a;
}

Tensor checkpoint_addcdiv(at::Tensor const& a, at::Tensor const& b, at::Tensor const& c, c10::Scalar d) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::addcdiv(vec.at(0), vec.at(1), vec.at(2), d)};
    };
  return CheckpointTensorImpl::make("addcdiv", rt, {a, b, c})[0];
}

Tensor checkpoint_to(at::Tensor const& a, c10::TensorOptions const& b, bool c, bool d, c10::optional<c10::MemoryFormat> e) {
  c10::TensorOptions b_ = b;
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {vec.at(0).to(b_, c, d, e)};
    };
  return CheckpointTensorImpl::make("to", rt, {a})[0];
}

Tensor checkpoint_div(const Tensor& a, const Tensor& b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::div(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("div", rt, {a, b})[0];
}

Tensor& checkpoint_div_(Tensor& a, const Tensor& b) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).div_(vec.at(1));
    };
  CheckpointTensorImpl::mutate("div_", mt, {a, b}, {0});
  return a;
}

Tensor checkpoint_clone(at::Tensor const& a, c10::optional<c10::MemoryFormat> b) {
  if (b) {
    rematerialize_function_t rt =
      [=](const Tensors& vec) -> Tensors {
        return {at::clone(vec.at(0), b)};
      };
    return CheckpointTensorImpl::make("clone", rt, {a})[0];
  }
  else {
    return a;
  }
}

Tensor checkpoint_where(at::Tensor const& a, at::Tensor const& b, at::Tensor const& c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::where(vec.at(0), vec.at(1), vec.at(2))};
    };
  return CheckpointTensorImpl::make("where", rt, {a, b, c})[0];
}

Tensor checkpoint_constant_pad_nd(Tensor const& a, c10::ArrayRef<long> b, c10::Scalar c) {
  std::vector<long> b_ = b.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::constant_pad_nd(vec.at(0), b_, c)};
    };
  return CheckpointTensorImpl::make("constant_pad_nd", rt, {a})[0];
}

Tensor checkpoint_binary_cross_entropy(const Tensor& a, const Tensor& b, const Tensor& c, long d) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::binary_cross_entropy(vec.at(0), vec.at(1), vec.at(2), d)};
    };
  return CheckpointTensorImpl::make("binary_cross_entropy", rt, {a, b, c})[0];
}

Tensor& checkpoint_binary_cross_entropy_out(Tensor& a, const Tensor& b, const Tensor& c, const Tensor& d, long e) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor self = vec.at(0);
      at::binary_cross_entropy_out(self, vec.at(1), vec.at(2), vec.at(3), e);
    };
  CheckpointTensorImpl::mutate("binary_cross_entropy_out", mt, {a, b, c, d}, {0});
  return a;
}

Tensor checkpoint_binary_cross_entropy_backward(const Tensor& a, const Tensor& b, const Tensor& c, const Tensor& d, long e) { 
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::binary_cross_entropy_backward(vec.at(0), vec.at(1), vec.at(2), vec.at(3), e)};
    };
  return CheckpointTensorImpl::make("binary_cross_entropy_backward", rt, {a, b, c, d})[0];
}

Tensor& checkpoint_binary_cross_entropy_backward_out(Tensor& a, const Tensor& b, const Tensor& c, const Tensor& d, const Tensor& e, long f) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor self = vec.at(0);
      at::binary_cross_entropy_backward_out(self, vec.at(1), vec.at(2), vec.at(3), vec.at(4), f);
    };
  CheckpointTensorImpl::mutate("binary_cross_entropy_backward_out", mt, {a, b, c, d, e}, {0});
  return a;
}

Tensor checkpoint_embedding(const Tensor& a, const Tensor& b, long c, bool d, bool e) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::embedding(vec.at(0), vec.at(1), c, d, e)};
    };
  return CheckpointTensorImpl::make("embedding", rt, {a, b})[0];
}

Tensor checkpoint_embedding_backward(const Tensor& a, const Tensor& b, long c, long d, bool e, bool f) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::embedding_backward(vec.at(0), vec.at(1), c, d, e, f)};
    };
  return CheckpointTensorImpl::make("embedding", rt, {a, b})[0];
}

std::tuple<Tensor, Tensor, Tensor, Tensor>
checkpoint_cudnn_batch_norm(const Tensor& a, const Tensor& b, const Tensor& c, const Tensor& d, const Tensor& e, bool f, double g, double h) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::cudnn_batch_norm(vec.at(0), vec.at(1), vec.at(2), vec.at(3), vec.at(4), f, g, h);
      return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret), std::get<3>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("cudnn_batch_norm", rt, {a, b, c, d, e});
  return {ret[0], ret[1], ret[2], ret[3]};
}

std::tuple<Tensor, Tensor, Tensor> checkpoint_cudnn_batch_norm_backward(at::Tensor const& a, at::Tensor const& b, at::Tensor const& c, at::Tensor const& d, at::Tensor const& e, at::Tensor const& f, at::Tensor const& g, double h, at::Tensor const& i) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::cudnn_batch_norm_backward(vec.at(0), vec.at(1), vec.at(2), vec.at(3), vec.at(4), vec.at(5), vec.at(6), h, vec.at(7));
      return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("cudnn_batch_norm_backward", rt, {a, b, c, d, e, f, g, i});
  return {ret[0], ret[1], ret[2]};
}

Tensor checkpoint_as_strided(const Tensor& a, c10::ArrayRef<long> b, c10::ArrayRef<long> c, c10::optional<long> d) {
  std::vector<long> b_ = b.vec(), c_ = c.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::as_strided(vec.at(0), b_, c_, d)};
    };
  return CheckpointTensorImpl::make("as_strided", rt, {a})[0];
}

Tensor checkpoint__masked_scale(const Tensor& a, const Tensor& b, double c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::_masked_scale(vec.at(0), vec.at(1), c)};
    };
  return CheckpointTensorImpl::make("_masked_scale", rt, {a, b})[0];
}

Tensor checkpoint_cudnn_convolution(const Tensor& a, const Tensor& b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, long f, bool g, bool h) {
  std::vector<long> c_ = c.vec(), d_ = d.vec(), e_ = e.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::cudnn_convolution(vec.at(0), vec.at(1), c_, d_, e_, f, g, h)};
    };
  return CheckpointTensorImpl::make("cudnn_convolution", rt, {a, b})[0];
}

Tensor checkpoint_cudnn_convolution_transpose(const Tensor& a, const Tensor& b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, long g, bool h, bool i) {
  std::vector<long> c_ = c.vec(), d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::cudnn_convolution_transpose(vec.at(0), vec.at(1), c_, d_, e_, f_, g, h, i)};
    };
  return CheckpointTensorImpl::make("cudnn_convolution_transpose", rt, {a, b})[0];
}

std::tuple<Tensor, Tensor> checkpoint_cudnn_convolution_backward(const Tensor& a, const Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, long g, bool h, bool i, std::array<bool, 2ul> j) {
  std::vector<long> d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::cudnn_convolution_backward(vec.at(0), vec.at(1), vec.at(2), d_, e_, f_, g, h, i, j);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("cudnn_convolution_backward", rt, {a, b, c});
  return {ret[0], ret[1]};
}

std::tuple<Tensor, Tensor> checkpoint_cudnn_convolution_transpose_backward(const Tensor& a, const Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, c10::ArrayRef<long> g, long h, bool i, bool j, std::array<bool, 2ul> k) {
  std::vector<long> d_ = d.vec(), e_ = e.vec(), f_ = f.vec(), g_ = g.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::cudnn_convolution_transpose_backward(vec.at(0), vec.at(1), vec.at(2), d_, e_, f_, g_, h, i, j, k);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("cudnn_convolution_transpose_backward", rt, {a, b, c});
  return {ret[0], ret[1]};
}

Tensor checkpoint_cudnn_convolution_backward_input(c10::ArrayRef<long> a, const Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, long g, bool h, bool i) {
  std::vector<long> a_ = a.vec(), d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::cudnn_convolution_backward_input(a_, vec.at(0), vec.at(1), d_, e_, f_, g, h, i)};
    };
  return CheckpointTensorImpl::make("cudnn_convolution_backward_input", rt, {b, c})[0];
}

Tensor checkpoint_cudnn_convolution_transpose_backward_input(const Tensor& a, const Tensor& b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, long f, bool g, bool h) {
  std::vector<long> c_ = c.vec(), d_ = d.vec(), e_ = e.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::cudnn_convolution_transpose_backward_input(vec.at(0), vec.at(1), c_, d_, e_, f, g, h)};
    };
  return CheckpointTensorImpl::make("cudnn_convolution_transpose_backward_input", rt, {a, b})[0];
}

Tensor checkpoint_cudnn_convolution_backward_weight(c10::ArrayRef<long> a, const Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, long g, bool h, bool i) {
  std::vector<long> a_ = a.vec(), d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::cudnn_convolution_backward_weight(a_, vec.at(0), vec.at(1), d_, e_, f_, g, h, i)};
    };
  return CheckpointTensorImpl::make("cudnn_convolution_backward_weight", rt, {b, c})[0];
}

Tensor checkpoint_cudnn_convolution_transpose_backward_weight(c10::ArrayRef<long> a, const Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, long g, bool h, bool i) {
  std::vector<long> a_ = a.vec(), d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::cudnn_convolution_transpose_backward_weight(a_, vec.at(0), vec.at(1), d_, e_, f_, g, h, i)};
    };
  return CheckpointTensorImpl::make("cudnn_convolution_transpose_backward_weight", rt, {b, c})[0];
}

Tensor checkpoint_relu(const Tensor& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::relu(vec.at(0))};
    };
  return CheckpointTensorImpl::make("relu", rt, {a})[0];
}

Tensor& checkpoint_relu_(Tensor& a) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).relu_();
    };
  CheckpointTensorImpl::mutate("relu_", mt, {a}, {0});
  return a;
}

Tensor checkpoint_log(at::Tensor const& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::log(vec.at(0))};
    };
  return CheckpointTensorImpl::make("log", rt, {a})[0];
}

Tensor& checkpoint_log_out(at::Tensor& a, at::Tensor const& b) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::log_out(a_, vec.at(1));
    };
  CheckpointTensorImpl::mutate("log_out", mt, {a, b}, {0});
  return a;
}

Tensor checkpoint_rsub(at::Tensor const& a, at::Tensor const& b, c10::Scalar c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::rsub(vec.at(0), vec.at(1), c)};
    };
  return CheckpointTensorImpl::make("rsub", rt, {a, b})[0];
}

Tensor checkpoint_rsub(at::Tensor const& a, c10::Scalar b, c10::Scalar c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::rsub(vec.at(0), b, c)};
    };
  return CheckpointTensorImpl::make("rsub", rt, {a})[0];
}

Tensor checkpoint_mul(at::Tensor const& a, c10::Scalar b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::mul(vec.at(0), b)};
    };
  return CheckpointTensorImpl::make("mul", rt, {a})[0];
}

std::tuple<Tensor&, Tensor&> checkpoint_max_pool2d_with_indices_out(Tensor& a, Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, c10::ArrayRef<long> g, bool h) {
  std::vector<long> d_ = d.vec(), e_ = e.vec(), f_ = f.vec(), g_ = g.vec();
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0), b_ = vec.at(1);
      at::max_pool2d_with_indices_out(a_, b_, vec.at(2), d_, e_, f_, g_, h);
    };
  CheckpointTensorImpl::mutate("max_pool2d_with_indices_out", mt, {a, b, c}, {0, 1});
  return {a, b};
}

Tensor checkpoint_avg_pool2d(const Tensor& a, c10::ArrayRef<long> b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, bool e, bool f, c10::optional<long> g) {
  std::vector<long> b_ = b.vec(), c_ = c.vec(), d_ = d.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::avg_pool2d(vec.at(0), b_, c_, d_, e, f, g)};
    };
  return CheckpointTensorImpl::make("avg_pool2d", rt, {a})[0];
}

Tensor checkpoint_avg_pool2d_backward(const Tensor& a, const Tensor& b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, bool f, bool g, c10::optional<long> h) {
  std::vector<long> c_ = c.vec(), d_ = d.vec(), e_ = e.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::avg_pool2d_backward(vec.at(0), vec.at(1), c_, d_, e_, f, g, h)};
    };
  return CheckpointTensorImpl::make("avg_pool2d_backward", rt, {a, b})[0];
}

Tensor& checkpoint_avg_pool2d_out(Tensor& a, const Tensor& b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, bool f, bool g, c10::optional<long> h) {
  std::vector<long> c_ = c.vec(), d_ = d.vec(), e_ = e.vec();
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::avg_pool2d_out(a_, vec.at(1), c_, d_, e_, f, g, h);
    };
  CheckpointTensorImpl::mutate("avg_pool2d_out", mt, {a, b}, {0});
  return a;
}

Tensor& checkpoint_avg_pool2d_backward_grad_input(Tensor& a, const Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, bool g, bool h, c10::optional<long> i) {
  std::vector<long> d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::avg_pool2d_backward_out(a_, vec.at(1), vec.at(2), d_, e_, f_, g, h, i);
    };
  CheckpointTensorImpl::mutate("avg_pool2d_backward_grad_input", mt, {a, b, c}, {0});
  return a;
}

std::tuple<Tensor, Tensor> checkpoint_max_pool2d_with_indices(const Tensor& a, c10::ArrayRef<long> b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, bool f) {
  std::vector<long> b_ = b.vec(), c_ = c.vec(), d_ = d.vec(), e_ = e.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::max_pool2d_with_indices(vec.at(0), b_, c_, d_, e_, f);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("max_pool2d_backward", rt, {a});
  return {ret[0], ret[1]};
}

Tensor& checkpoint_max_pool2d_with_indices_backward_grad_input(Tensor& a, const Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, c10::ArrayRef<long> g, bool h, const Tensor& i) {
  std::vector<long> d_ = d.vec(), e_ = e.vec(), f_ = f.vec(), g_ = g.vec();
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::max_pool2d_with_indices_backward_out(a_, vec.at(1), vec.at(2), d_, e_, f_, g, h, vec.at(3));
    };
  CheckpointTensorImpl::mutate("max_pool2d_with_indices_backward_grad_input", mt, {a, b, c, i}, {0});
  return a;
}

Tensor checkpoint_max_pool2d_with_indices_backward(const Tensor& a, const Tensor& b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, bool g, const Tensor& h) {
  std::vector<long> c_ = c.vec(), d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::max_pool2d_with_indices_backward(vec.at(0), vec.at(1), c_, d_, e_, f_, g, vec.at(2))};
    };
  return CheckpointTensorImpl::make("max_pool2d_with_indices_backward", rt, {a, b, h})[0];
}

Tensor checkpoint_view(const Tensor& a, c10::ArrayRef<long> b) {
  std::vector<long> b_ = b.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {vec.at(0).view(b_)};
    };
  return CheckpointTensorImpl::make("view", rt, {a})[0];
}

Tensor checkpoint_ne_Scalar(const Tensor& a, c10::Scalar b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::ne(vec.at(0), b)};
    };
  return CheckpointTensorImpl::make("ne_Scalar", rt, {a})[0];
}

Tensor& checkpoint_ne_Scalar_out(Tensor& a, const Tensor& b, c10::Scalar c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::ne_out(a_, vec.at(1), c);
    };
  CheckpointTensorImpl::mutate("ne_Scalar_out", mt, {a, b}, {0});
  return a;
}

Tensor checkpoint_ne_Tensor(const Tensor& a, const Tensor& b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::ne(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("ne_Tensor", rt, {a, b})[0];
}

Tensor& checkpoint_ne_Tensor_out(Tensor& a, const Tensor& b, const Tensor& c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::ne_out(a_, vec.at(1), vec.at(2));
    };
  CheckpointTensorImpl::mutate("ne_Tensor_out", mt, {a, b, c}, {0});
  return a;
}

Tensor checkpoint_eq_Scalar(const Tensor& a, c10::Scalar b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::eq(vec.at(0), b)};
    };
  return CheckpointTensorImpl::make("eq_Scalar", rt, {a})[0];
}

Tensor& checkpoint_eq_Scalar_out(Tensor& a, const Tensor& b, c10::Scalar c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::eq_out(a_, vec.at(1), c);
    };
  CheckpointTensorImpl::mutate("eq_Scalar_out", mt, {a, b}, {0});
  return a;
}

Tensor checkpoint_eq_Tensor(const Tensor& a, const Tensor& b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::eq(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("eq_Tensor", rt, {a, b})[0];
}

Tensor& checkpoint_eq_Tensor_out(Tensor& a, const Tensor& b, const Tensor& c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::eq_out(a_, vec.at(1), vec.at(2));
    };
  CheckpointTensorImpl::mutate("eq_Tensor_out", mt, {a, b, c}, {0});
  return a;
}

Tensor checkpoint_addmm(const Tensor& a, const Tensor& b, const Tensor& c, c10::Scalar d, c10::Scalar e) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::addmm(vec.at(0), vec.at(1), vec.at(2), d, e)};
    };
  return CheckpointTensorImpl::make("addmm", rt, {a, b, c})[0];
}

Tensor& checkpoint_addmm_out(Tensor& a, const Tensor& b, const Tensor& c, const Tensor& d, c10::Scalar e, c10::Scalar f) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::addmm_out(a_, vec.at(1), vec.at(2), d, e, f);
    };
  CheckpointTensorImpl::mutate("addmm_out", mt, {a, b, c}, {0});
  return a;
}

Tensor& checkpoint_addmm_(Tensor& a, const Tensor& b, const Tensor& c, c10::Scalar d, c10::Scalar e) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      a.addmm_(vec.at(1), vec.at(2), d, e);
    };
  CheckpointTensorImpl::mutate("addmm_", mt, {a, b, c}, {0});
  return a;
}

Tensor checkpoint_sigmoid(const Tensor& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sigmoid(vec.at(0))};
    };
  return CheckpointTensorImpl::make("sigmoid", rt, {a})[0];
}

Tensor& checkpoint_sigmoid_(Tensor& a) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      a.sigmoid_();
    };
  CheckpointTensorImpl::mutate("sigmoid_", mt, {a}, {0});
  return a;
}

Tensor checkpoint__log_softmax(const Tensor& a, long b, bool c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::_log_softmax(vec.at(0), b, c)};
    };
  return CheckpointTensorImpl::make("_log_softmax", rt, {a})[0];
}

Tensor checkpoint__log_softmax_backward_data(const Tensor& a, const Tensor& b, long c, const Tensor& d) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::_log_softmax_backward_data(vec.at(0), vec.at(1), c, vec.at(2))};
    };
  return CheckpointTensorImpl::make("_log_softmax_backward_data", rt, {a, b, d})[0];
}

std::tuple<Tensor, Tensor> checkpoint_nll_loss_forward(const Tensor& a, const Tensor& b, const Tensor& c, long d, long e) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::nll_loss_forward(vec.at(0), vec.at(1), vec.at(2), d, e);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("nll_loss_forward", rt, {a, b, c});
  return {ret[0], ret[1]};
}

std::tuple<Tensor&, Tensor&> checkpoint_nll_loss_forward_out(Tensor& a, Tensor& b, const Tensor& c, const Tensor& d, const Tensor& e, long f, long g) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      Tensor b_ = vec.at(1);
      at::nll_loss_forward_out(a_, b_, vec.at(2), vec.at(3), vec.at(4), f, g);
    };
  CheckpointTensorImpl::mutate("nll_loss_forward_out", mt, {a, b, c, d, e}, {0, 1});
  return {a, b};
}

Tensor checkpoint_nll_loss_backward(const Tensor& a, const Tensor& b, const Tensor& c, const Tensor& d, long e, long f, const Tensor& g) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::nll_loss_backward(vec.at(0), vec.at(1), vec.at(2), vec.at(3), e, f, vec.at(4))};
    };
  return CheckpointTensorImpl::make("nll_loss_backward", rt, {a, b, c, d, g})[0];
}

Tensor& checkpoint_nll_loss_backward_grad_input(Tensor& a, const Tensor& b, const Tensor& c, const Tensor& d, const Tensor& e, long f, long g, const Tensor& h) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::nll_loss_backward_out(a_, vec.at(1), vec.at(2), vec.at(3), vec.at(4), f, g, vec.at(5));
    };
  CheckpointTensorImpl::mutate("nll_loss_backward_grad_input", mt, {a, b, c, d, e, h}, {0});
  return a;
}

Tensor checkpoint_mm(const Tensor& a, const Tensor& b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::mm(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("mm", rt, {a, b})[0];
}

Tensor& checkpoint_mm_out(Tensor& a, const Tensor& b, const Tensor& c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::mm_out(a_, vec.at(1), vec.at(2));
    };
  CheckpointTensorImpl::mutate("mm_out", mt, {a, b, c}, {0});
  return a;
}

Tensor checkpoint_sum(const Tensor& a, c10::optional<c10::ScalarType> b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sum(vec.at(0), b)};
    };
  return CheckpointTensorImpl::make("sum", rt, {a})[0];
}

Tensor checkpoint_sum_dim_IntList(const Tensor& a, c10::ArrayRef<long> b, bool c, c10::optional<c10::ScalarType> d) {
  std::vector<long> b_ = b.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sum(vec.at(0), b_, c, d)};
    };
  return CheckpointTensorImpl::make("sum_dim_IntList", rt, {a})[0];
}

Tensor checkpoint_threshold(const Tensor& a, c10::Scalar b, c10::Scalar c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::threshold(vec.at(0), b, c)};
    };
  return CheckpointTensorImpl::make("threshold", rt, {a})[0];
}

Tensor& checkpoint_threshold_(Tensor& a, c10::Scalar b, c10::Scalar c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::threshold_(a_, b, c);
    };
  CheckpointTensorImpl::mutate("threshold_", mt, {a}, {0});
  return a;
}

Tensor& checkpoint_threshold_out(Tensor& a, const Tensor& b, c10::Scalar c, c10::Scalar d) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::threshold_out(a_, b, c, d);
    };
  CheckpointTensorImpl::mutate("threshold_out", mt, {a}, {0});
  return a;
}

Tensor checkpoint_threshold_backward(const Tensor& a, const Tensor& b, c10::Scalar c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::threshold_backward(vec.at(0), vec.at(1), c)};
    };
  return CheckpointTensorImpl::make("threshold_backward", rt, {a, b})[0];
}

Tensor checkpoint_select(const Tensor& a, long b, long c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::select(vec.at(0), b, c)};
    };
  return CheckpointTensorImpl::make("select", rt, {a})[0];
}

Tensor checkpoint_select_backward(const Tensor& a, c10::ArrayRef<long> b, long c, long d) {
  std::vector<long> b_ = b.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::select_backward(vec.at(0), b_, c, d)};
    };
  return CheckpointTensorImpl::make("select_backward", rt, {a})[0];
}

Tensor checkpoint_slice(const Tensor& a, long b, long c, long d, long e) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::slice(vec.at(0), b, c, d, e)};
    };
  return CheckpointTensorImpl::make("slice", rt, {a})[0];
}

Tensor checkpoint_slice_backward(const Tensor& a, c10::ArrayRef<long> b, long c, long d, long e, long f) {
  std::vector<long> b_ = b.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::slice_backward(vec.at(0), b_, c, d, e, f)};
    };
  return CheckpointTensorImpl::make("slice_backward", rt, {a})[0];
}

Tensor& checkpoint_zero_(Tensor& a) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).zero_();
    };
  CheckpointTensorImpl::mutate("zero_", mt, {a}, {0});
  return a;
}

Tensor& checkpoint_squeeze_(at::Tensor& a, at::Dimname b) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).squeeze_(b);
    };
  CheckpointTensorImpl::mutate("squeeze_", mt, {a}, {0});
  return a;
}

Tensor& checkpoint_squeeze_(at::Tensor& a) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).squeeze_();
    };
  CheckpointTensorImpl::mutate("squeeze_", mt, {a}, {0});
  return a;
}

Tensor& checkpoint_squeeze_(at::Tensor& a, long b) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).squeeze_(b);
    };
  CheckpointTensorImpl::mutate("squeeze_", mt, {a}, {0});
  return a;
}

Tensor checkpoint_sigmoid_backward(at::Tensor const& a, at::Tensor const& b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sigmoid_backward(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("sigmoid_backward", rt, {a, b})[0];
}

Tensor& checkpoint_sigmoid_backward_out(at::Tensor& a, at::Tensor const& b, at::Tensor const& c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::sigmoid_backward_out(a_, vec.at(1), vec.at(2));
    };
  CheckpointTensorImpl::mutate("sigmoid_backward_out", mt, {a, b, c}, {0});
  return a;
}

Tensor& checkpoint_sign_out(at::Tensor& a, at::Tensor const& b) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::sign_out(a_, vec.at(1));
    };
  CheckpointTensorImpl::mutate("sign_out", mt, {a, b}, {0});
  return a;
}

Tensor checkpoint_sign(const Tensor& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sign(vec.at(0))};
    };
  return CheckpointTensorImpl::make("sign", rt, {a})[0];
}

Tensor checkpoint_tanh(const Tensor& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::tanh(vec.at(0))};
    };
  return CheckpointTensorImpl::make("tanh", rt, {a})[0];
}

Tensor checkpoint_tanh_backward(at::Tensor const& a, at::Tensor const& b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::tanh_backward(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("tanh_backward", rt, {a, b})[0];
}

Tensor& checkpoint_tanh_backward_out(at::Tensor& a, at::Tensor const& b, at::Tensor const& c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::tanh_backward_out(a_, vec.at(1), vec.at(2));
    };
  CheckpointTensorImpl::mutate("tanh_backward_out", mt, {a, b, c}, {0});
  return a;
}

Tensor checkpoint_neg(at::Tensor const& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::neg(vec.at(0))};
    };
  return CheckpointTensorImpl::make("neg", rt, {a})[0];
}

Tensor checkpoint_sub(at::Tensor const& a, at::Tensor const& b, c10::Scalar c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sub(vec.at(0), vec.at(1), c)};
    };
  return CheckpointTensorImpl::make("sub", rt, {a, b})[0];
}

Tensor& checkpoint_sub_(at::Tensor& a, at::Tensor const& b, c10::Scalar c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor self = vec.at(0);
    self.sub_(vec.at(1), c);
  };
  CheckpointTensorImpl::mutate("sub_", mt, {a, b}, {0});
  return a;
}

Tensor checkpoint_repeat(const at::Tensor& a, c10::ArrayRef<long> b) {
  std::vector<long> b_ = b.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {vec.at(0).repeat(b_)};
    };
  return CheckpointTensorImpl::make("repeat", rt, {a})[0];
}

Tensor checkpoint_mean(const Tensor& self, c10::optional<c10::ScalarType> dtype) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::native::mean_cpu_gpu(vec[0], dtype)};
  };
  return CheckpointTensorImpl::make("mean", rt, {self})[0];
}

Tensor checkpoint_mean(const Tensor& self, IntArrayRef dim, bool keepdim, c10::optional<c10::ScalarType> dtype) {
  std::vector<long> dim_ = dim.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::native::mean_cpu_gpu(vec[0], dim_, keepdim, dtype)};
  };
  return CheckpointTensorImpl::make("mean.dim", rt, {self})[0];
}

Tensor checkpoint__cat(c10::ArrayRef<Tensor> a, long b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::cat(vec, b)};
    };
  std::vector<Tensor> s;
  for (const Tensor& t : a) {
    s.push_back(t);
  }
  return CheckpointTensorImpl::make("_cat", rt, s)[0];
}

Tensor& checkpoint__cat_out(Tensor& a, c10::ArrayRef<Tensor> b, long c) {
  std::vector<Tensor> args;
  args.push_back(a);
  for (const Tensor& t : b) {
    args.push_back(t);
  }
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor t = vec[0];
      at::cat_out(t, ArrayRef<Tensor>(vec.data() + 1, vec.size() - 1), c);
    };
  CheckpointTensorImpl::mutate("_cat_out", mt, args, {0});
  return a;
}

Tensor checkpoint_kl_div(at::Tensor const& a, at::Tensor const& b, long c, bool d) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::kl_div(vec.at(0), vec.at(1), c, d)};
    };
  return CheckpointTensorImpl::make("kl_div", rt, {a, b})[0];
}

Tensor checkpoint_kl_div_backward(at::Tensor const& a, at::Tensor const& b, at::Tensor const& c, long d, bool e) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::kl_div_backward(vec.at(0), vec.at(1), vec.at(2), d, e)};
    };
  return CheckpointTensorImpl::make("kl_div_backward", rt, {a, b, c})[0];
}

Tensor checkpoint_upsample_bilinear2d(at::Tensor const& self, c10::ArrayRef<long> output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  std::vector<long> output_size_ = output_size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::upsample_bilinear2d(vec.at(0), output_size_, align_corners, scales_h, scales_w)};
  };
  return CheckpointTensorImpl::make("upsample_bilinear2d", rt, {self})[0];
}

Tensor& checkpoint_upsample_bilinear2d_out(at::Tensor& out, const at::Tensor& self, c10::ArrayRef<long> output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  std::vector<long> output_size_ = output_size.vec();
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor out = vec.at(0);
    at::upsample_bilinear2d_out(out, vec.at(1), output_size_, align_corners, scales_h, scales_w);
  };
  CheckpointTensorImpl::mutate("binary_cross_entropy_out", mt, {out, self}, {0});
  return out;
}

Tensor& checkpoint_upsample_bilinear2d_backward_out(at::Tensor& grad_input, const at::Tensor& grad_output, c10::ArrayRef<long> output_size, c10::ArrayRef<long> input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  std::vector<long> output_size_ = output_size.vec();
  std::vector<long> input_size_ = input_size.vec();
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor grad_input = vec.at(0);
    at::upsample_bilinear2d_backward_out(grad_input, vec.at(1), output_size_, input_size_, align_corners, scales_h, scales_w);
  };
  CheckpointTensorImpl::mutate("upsample_bilinear2d_backward_out", mt, {grad_input, grad_output}, {0});
  return grad_input;
}

Tensor checkpoint_upsample_bilinear2d_backward(at::Tensor const& grad_output, c10::ArrayRef<long> output_size, c10::ArrayRef<long> input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  std::vector<long> output_size_ = output_size.vec();
  std::vector<long> input_size_ = input_size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::upsample_bilinear2d_backward(vec.at(0), output_size_, input_size_, align_corners, scales_h, scales_w)};
  };
  return CheckpointTensorImpl::make("upsample_bilinear2d_backward", rt, {grad_output})[0];
}

Tensor& checkpoint_clamp_min_(Tensor& a, Scalar min) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor self = vec.at(0);
    at::clamp_min_(self, min);
  };
  CheckpointTensorImpl::mutate("clamp_min_", mt, {a}, {0});
  return a;
}

Tensor& checkpoint_clamp_min_out(Tensor& out, const Tensor& self, Scalar min) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor out = vec.at(0);
    at::clamp_min_out(out, vec.at(1), min);
  };
  CheckpointTensorImpl::mutate("clamp_min__out", mt, {out, self}, {0});
  return out;
}

Tensor checkpoint_binary_cross_entropy_with_logits(const Tensor& input, const Tensor& target, const Tensor& weight, const Tensor& pos_weight, int64_t reduction) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::binary_cross_entropy_with_logits(vec.at(0), vec.at(1), vec.at(2), vec.at(3), reduction)};
  };
  return CheckpointTensorImpl::make("binary_cross_entropy_with_logits", rt, {input, target, weight, pos_weight})[0];
}

Tensor checkpoint_binary_cross_entropy_with_logits_backward(const Tensor& grad, const Tensor& input, const Tensor& target, const Tensor& weight, const Tensor& pos_weight, int64_t reduction) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::binary_cross_entropy_with_logits_backward(vec.at(0), vec.at(1), vec.at(2), vec.at(3), vec.at(4), reduction)};
  };
  return CheckpointTensorImpl::make("binary_cross_entropy_with_logits_backward", rt, {grad, input, target, weight, pos_weight})[0];
}

std::tuple<Tensor, Tensor> checkpoint__fused_dropout(const Tensor & self, double p, c10::optional<Generator> g) {
  // TODO: Figure out how to properly duplicate the generator;
  // note that the commented-out code below results in a segfault!
  // Ref<std::shared_ptr<Generator>> gen;
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    // Generator* cur = gen.t ? gen.t.get() : g;
    // auto newG = cur->clone();
    // auto res = at::_fused_dropout(vec.at(0), p, cur);
    // gen.t = newG;
    auto res = at::_fused_dropout(vec.at(0), p, g);
    return {std::get<0>(res), std::get<1>(res)};
  };
  auto res = CheckpointTensorImpl::make("_fused_droupout_", rt, {self});
  return {res[0], res[1]};
}

std::tuple<Tensor, Tensor, Tensor> checkpoint__thnn_fused_lstm_cell(const Tensor& input_gates, const Tensor& hidden_gates, const Tensor& cx, const Tensor& input_bias, const Tensor& hidden_bias) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto res = at::_thnn_fused_lstm_cell(vec.at(0), vec.at(1), vec.at(2),
                                         vec.at(3), vec.at(4));
    return {std::get<0>(res), std::get<1>(res), std::get<2>(res)};
  };
  auto res = CheckpointTensorImpl::make("_thnn_fused_lstm_cell", rt,
                                        {input_gates, hidden_gates, cx, input_bias, hidden_bias});
  return {res[0], res[1], res[2]};
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> checkpoint__thnn_fused_lstm_cell_backward(const Tensor& grad_hy, const Tensor& grad_cy, const Tensor& cx, const Tensor& cy, const Tensor& workspace, bool has_bias) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto res = at::_thnn_fused_lstm_cell_backward(vec.at(0), vec.at(1), vec.at(2), vec.at(3), vec.at(4), has_bias);
    return {std::get<0>(res), std::get<1>(res),
        std::get<2>(res), std::get<3>(res), std::get<4>(res)};
  };
  auto res = CheckpointTensorImpl::make("_thnn_fused_lstm_cell_backward", rt,
                                        {grad_hy, grad_cy, cx, cy, workspace});
  return {res[0], res[1], res[2], res[3], res[4]};
}

std::tuple<Tensor, Tensor> checkpoint__thnn_fused_gru_cell(const Tensor& input_gates, const Tensor& hidden_gates, const Tensor& hx, const Tensor& input_bias, const Tensor& hidden_bias) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto res = at::_thnn_fused_gru_cell(vec.at(0), vec.at(1), vec.at(2), vec.at(3), vec.at(4));
    return {std::get<0>(res), std::get<1>(res)};
  };
  auto res = CheckpointTensorImpl::make("_thnn_fused_gru_cell", rt,
                                        {input_gates, hidden_gates, hx, input_bias, hidden_bias});
  return {res[0], res[1]};
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> checkpoint__thnn_fused_gru_cell_backward(const Tensor& grad_hy, const Tensor& workspace, bool has_bias) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto res = at::_thnn_fused_gru_cell_backward(vec.at(0), vec.at(1), has_bias);
    return {std::get<0>(res), std::get<1>(res),
        std::get<2>(res), std::get<3>(res), std::get<4>(res)};
  };
  auto res = CheckpointTensorImpl::make("_thnn_fused_gru_cell_backward", rt,
                                        {grad_hy, workspace});
  return {res[0], res[1], res[2], res[3], res[4]};
}

Tensor& checkpoint_bitwise_and_out(Tensor& self, const Tensor& other, const Tensor& out) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor self = vec.at(0);
    at::bitwise_and_out(self, vec.at(1), vec.at(2));
  };
  CheckpointTensorImpl::mutate("bitwise_and_out", mt, {self, other, out}, {0});
  return self;
}

Tensor& checkpoint_bitwise_and_out(Tensor& self, const Tensor& out, Scalar other) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor self = vec.at(0);
    at::bitwise_and_out(self, vec.at(1), other);
  };
  CheckpointTensorImpl::mutate("bitwise_and_out", mt, {self, out}, {0});
  return self;
}

Tensor& checkpoint_fill_(Tensor& self, const Tensor& value) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor self = vec.at(0);
    at::fill_(self, vec.at(1));
  };
  CheckpointTensorImpl::mutate("fill_", mt, {self, value}, {0});
  return self;
}

Tensor& checkpoint_fill_(Tensor& self, Scalar value) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor self = vec.at(0);
    at::fill_(self, value);
  };
  CheckpointTensorImpl::mutate("fill_", mt, {self}, {0});
  return self;
}

Tensor& checkpoint_masked_select_out(Tensor& self, const Tensor& mask, const Tensor& out) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor self = vec.at(0);
    at::masked_select_out(self, vec.at(1), vec.at(2));
  };
  CheckpointTensorImpl::mutate("masked_select_out", mt, {self, mask, out}, {0});
  return self;
}

Tensor checkpoint_masked_select(const Tensor& self, const Tensor& mask) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::masked_select(vec.at(0), vec.at(1))};
  };
  return CheckpointTensorImpl::make("masked_select", rt, {self, mask})[0];
}

Tensor checkpoint_index(const Tensor& self, ArrayRef<Tensor> indices) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto self = vec.at(0);
    auto indices = std::vector<Tensor>(vec.begin() + 1, vec.end());
    return {at::index(self, indices)};
  };

  std::vector<Tensor> s = {self};
  for (const Tensor& t: indices) {
    s.push_back(t);
  }
  return CheckpointTensorImpl::make("index", rt, s)[0];
}

Tensor& checkpoint_index_put_(Tensor& self, ArrayRef<Tensor> indices, const Tensor& values, const bool accumulate) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor self = vec.at(0);
    auto values = vec.at(1);
    auto indices = std::vector<Tensor>(vec.begin() + 2, vec.end());
    at::index_put_(self, indices, values, accumulate);
  };
  std::vector<Tensor> s = {self, values};
  for (const Tensor& t: indices) {
    s.push_back(t);
  }
  CheckpointTensorImpl::mutate("index_put_", mt, s, {0});
  return self;
}

Tensor checkpoint_bmm(const Tensor& self, const Tensor& mat2) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::bmm(vec.at(0), vec.at(1))};
  };
  return CheckpointTensorImpl::make("bmm", rt, {self, mat2})[0];
}

Tensor checkpoint__softmax(const Tensor& self, long dim, bool half_to_float) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::_softmax(vec.at(0), dim, half_to_float)};
  };
  return CheckpointTensorImpl::make("_softmax", rt, {self})[0];
}

Tensor checkpoint__softmax_backward_data(const Tensor& grad_output, const Tensor& output, long dim, const Tensor& self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::_softmax_backward_data(vec.at(0), vec.at(1), dim, vec.at(2))};
  };
  return CheckpointTensorImpl::make("_softmax_backward_data", rt, {grad_output, output, self})[0];
}

std::tuple<Tensor, Tensor, Tensor>
checkpoint_layer_norm(const Tensor& input, const Tensor& weight, const Tensor& bias, long M, long N, double eps) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto ret = at::native_layer_norm(vec.at(0), vec.at(1), vec.at(2), M, N, eps);
    return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
  };
  auto ret = CheckpointTensorImpl::make("native_layer_norm", rt, {input, weight, bias});
  return {ret[0], ret[1], ret[2]};
}

std::tuple<Tensor, Tensor, Tensor>
checkpoint_layer_norm_backward(const Tensor& grad_out, const Tensor& input, const Tensor& mean, const Tensor& rstd, const Tensor& weight, long M, long N, std::array<bool, 3ul> output_mask) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto ret = at::native_layer_norm_backward(vec.at(0), vec.at(1), vec.at(2), vec.at(3), vec.at(4), M, N, output_mask);
    return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
  };
  auto ret = CheckpointTensorImpl::make("native_layer_norm_backward", rt, {grad_out, input, mean, rstd, weight});
  return {ret[0], ret[1], ret[2]};
}

std::tuple<Tensor, Tensor>
checkpoint_topk(const Tensor& self, long k, long dim, bool largest, bool sorted) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto ret = at::topk(vec.at(0), k, dim, largest, sorted);
    return {std::get<0>(ret), std::get<1>(ret)};
  };
  auto ret = CheckpointTensorImpl::make("topk", rt, {self});
  return {ret[0], ret[1]};
}

std::tuple<Tensor&, Tensor&>
checkpoint_topk_values(Tensor& values, Tensor& indices, const Tensor& self, long k, long dim, bool largest, bool sorted) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor values_ = vec.at(0);
    Tensor indices_ = vec.at(1);
    at::topk_out(values_, indices_, vec.at(2), k, dim, largest, sorted);
  };
  CheckpointTensorImpl::mutate("topk_values", mt, {values, indices, self}, {0, 1});
  return {values, indices};
}

Tensor& checkpoint_masked_fill_(Tensor& self, const Tensor& mask, Scalar value) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor self_ = vec.at(0);
    self_.masked_fill_(vec.at(1), value);
  };
  CheckpointTensorImpl::mutate("masked_fill_Scalar", mt, {self, mask}, {0});
  return {self};
}

Tensor& checkpoint_masked_fill_(Tensor& self, const Tensor& mask, const Tensor& value) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor self_ = vec.at(0);
    self_.masked_fill_(vec.at(1), vec.at(2));
  };
  CheckpointTensorImpl::mutate("masked_fill_Tensor", mt, {self, mask, value}, {0});
  return {self};
}

Tensor checkpoint_clamp(const Tensor& self, c10::optional<Scalar> min, c10::optional<Scalar> max) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::clamp(vec.at(0), min, max)};
  };
  return CheckpointTensorImpl::make("clamp", rt, {self})[0];
}

Tensor& checkpoint_clamp_(Tensor& self, c10::optional<Scalar> min, c10::optional<Scalar> max) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor self_ = vec.at(0);
    at::clamp_(self_, min, max);
  };
  CheckpointTensorImpl::mutate("clamp_", mt, {self}, {0});
  return {self};
}

Tensor& checkpoint_clamp_out(Tensor& out, const Tensor& self, c10::optional<Scalar> min, c10::optional<Scalar> max) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor out_ = vec.at(0);
    at::clamp_out(out_, vec.at(1), min, max);
  };
  CheckpointTensorImpl::mutate("clamp_out", mt, {out, self}, {0});
  return {out};
}

std::tuple<Tensor&, Tensor&, Tensor&> checkpoint_thnn_conv2d_forward_out(Tensor& output, Tensor& finput, Tensor& fgrad_input, const Tensor& self, const Tensor& weight, c10::ArrayRef<long> kernel_size, const Tensor& bias, c10::ArrayRef<long> stride, c10::ArrayRef<long> padding) {
  auto kernel_size_ = kernel_size.vec();
  auto stride_ = stride.vec();
  auto padding_ = padding.vec();
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor output_ = vec.at(0);
    Tensor finput_ = vec.at(1);
    Tensor fgrad_input_ = vec.at(2);
    at::thnn_conv2d_forward_out(output_, finput_, fgrad_input_, vec.at(3), vec.at(4), kernel_size_, vec.at(5), stride_, padding_);
  };
  CheckpointTensorImpl::mutate("thnn_conv2d_forward_out", mt, {output, finput, fgrad_input, self, weight, bias}, {0, 1, 2});
  return {output, finput, fgrad_input};
}

std::tuple<Tensor, Tensor, Tensor> checkpoint_thnn_conv2d_forward(const Tensor& self, const Tensor& weight, c10::ArrayRef<long> kernel_size, const Tensor& bias, c10::ArrayRef<long> stride, c10::ArrayRef<long> padding) {
  auto kernel_size_ = kernel_size.vec();
  auto stride_ = stride.vec();
  auto padding_ = padding.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto ret = at::thnn_conv2d_forward(vec.at(0), vec.at(1), kernel_size_, vec.at(2), stride_, padding_);
    return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
  };
  auto ret = CheckpointTensorImpl::make("thnn_conv2d_forward", rt, {self, weight, bias});
  return {ret[0], ret[1], ret[2]};
}

std::tuple<Tensor&, Tensor&, Tensor&> checkpoint_thnn_conv2d_backward_out(Tensor& grad_input, Tensor& grad_weight, Tensor& grad_bias, const Tensor& grad_output, const Tensor& self, const Tensor& weight, c10::ArrayRef<long> kernel_size, c10::ArrayRef<long> stride, c10::ArrayRef<long> padding, const Tensor& finput, const Tensor& fgrad_input) {
  auto kernel_size_ = kernel_size.vec();
  auto stride_ = stride.vec();
  auto padding_ = padding.vec();
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor grad_input_ = vec.at(0);
    Tensor grad_weight_ = vec.at(1);
    Tensor grad_bias_ = vec.at(2);
    at::thnn_conv2d_backward_out(grad_input_, grad_weight_, grad_bias_, vec.at(3), vec.at(4), vec.at(5), kernel_size_, stride_, padding_, vec.at(6), vec.at(7));
  };
  CheckpointTensorImpl::mutate("thnn_conv2d_backward_out", mt, {grad_input, grad_weight, grad_bias, grad_output, self, weight, finput, fgrad_input}, {0, 1, 2});
  return {grad_input, grad_weight, grad_bias};
}

std::tuple<Tensor, Tensor, Tensor> checkpoint_thnn_conv2d_backward(const Tensor& grad_output, const Tensor& self, const Tensor& weight, c10::ArrayRef<long> kernel_size, c10::ArrayRef<long> stride, c10::ArrayRef<long> padding, const Tensor& finput, const Tensor& fgrad_input, std::array<bool, 3ul> output_mask) {
  auto kernel_size_ = kernel_size.vec();
  auto stride_ = stride.vec();
  auto padding_ = padding.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto ret = at::thnn_conv2d_backward(vec.at(0), vec.at(1), vec.at(2), kernel_size_, stride_, padding_, vec.at(3), vec.at(4), output_mask);
    return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
  };
  auto ret = CheckpointTensorImpl::make("thnn_conv2d_backward", rt, {grad_output, self, weight, finput, fgrad_input});
  return {ret[0], ret[1], ret[2]};
}

std::tuple<Tensor, Tensor, Tensor> checkpoint_native_batch_norm(const Tensor& input, const Tensor& weight, const Tensor& bias, const Tensor& running_mean, const Tensor& running_var, bool training, double momentum, double eps) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto ret = at::native_batch_norm(vec.at(0), vec.at(1), vec.at(2), vec.at(3), vec.at(4), training, momentum, eps);
    return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
  };
  auto ret = CheckpointTensorImpl::make("native_batch_norm", rt, {input, weight, bias, running_mean, running_var});
  return {ret[0], ret[1], ret[2]};
}

std::tuple<Tensor&, Tensor&, Tensor&> checkpoint_native_batch_norm_out(Tensor& out, Tensor& save_mean, Tensor& save_invstd, const Tensor& input, const Tensor& weight, const Tensor& bias, const Tensor& running_mean, const Tensor& running_var, bool training, double momentum, double eps) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor out_ = vec.at(0);
    Tensor save_mean_ = vec.at(1);
    Tensor save_invstd_ = vec.at(2);
    at::native_batch_norm_out(out_, save_mean_, save_invstd_, vec.at(3), vec.at(4), vec.at(5), vec.at(6), vec.at(7), training, momentum, eps);
  };
  CheckpointTensorImpl::mutate("native_batch_norm_out", mt, {out, save_mean, save_invstd, input, weight, bias, running_mean, running_var}, {0, 1, 2});
  return {out, save_mean, save_invstd};
}

std::tuple<Tensor, Tensor, Tensor> checkpoint_native_batch_norm_backward(const Tensor& grad_out, const Tensor& input, const Tensor& weight, const Tensor& running_mean, const Tensor& running_var, const Tensor& save_mean, const Tensor& save_invstd, bool train, double eps, std::array<bool, 3ul> output_mask) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto ret = at::native_batch_norm_backward(vec.at(0), vec.at(1), vec.at(2), vec.at(3), vec.at(4), vec.at(5), vec.at(6), train, eps, output_mask);
    return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
  };
  auto ret = CheckpointTensorImpl::make("native_batch_norm_backward", rt, {grad_out, input, weight, running_mean, running_var, save_mean, save_invstd});
  return {ret[0], ret[1], ret[2]};
}

std::tuple<Tensor, Tensor> checkpoint__cudnn_ctc_loss(const Tensor& log_probs, const Tensor& targets, ArrayRef<long> input_lengths, ArrayRef<long> target_lengths, long blank, bool deterministic, bool zero_infinity) {
  auto input_lengths_ = input_lengths.vec();
  auto target_lengths_ = target_lengths.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto ret = at::_cudnn_ctc_loss(vec.at(0), vec.at(1), input_lengths_, target_lengths_, blank, deterministic, zero_infinity);
    return {std::get<0>(ret), std::get<1>(ret)};
  };
  auto ret = CheckpointTensorImpl::make("_cudnn_ctc_loss", rt, {log_probs, targets});
  return {ret[0], ret[1]};
}

std::tuple<Tensor, Tensor> checkpoint__ctc_loss(const Tensor& log_probs, const Tensor& targets, ArrayRef<long> input_lengths, ArrayRef<long> target_lengths,  long blank, bool zero_infinity) {
  auto input_lengths_ = input_lengths.vec();
  auto target_lengths_ = target_lengths.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto ret = at::_ctc_loss(vec.at(0), vec.at(1), input_lengths_, target_lengths_, blank, zero_infinity);
    return {std::get<0>(ret), std::get<1>(ret)};
  };
  auto ret = CheckpointTensorImpl::make("_ctc_loss", rt, {log_probs, targets});
  return {ret[0], ret[1]};
}

Tensor checkpoint__ctc_loss_backward(const Tensor& grad, const Tensor& log_probs, const Tensor& targets, ArrayRef<long> input_lengths, ArrayRef<long> target_lengths, const Tensor& neg_log_likelihood, const Tensor& log_alpha, long blank, bool zero_infinity) {
  auto input_lengths_ = input_lengths.vec();
  auto target_lengths_ = target_lengths.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::_ctc_loss_backward(vec.at(0), vec.at(1), vec.at(2), input_lengths_, target_lengths_, vec.at(3), vec.at(4), blank, zero_infinity)};
  };
  return CheckpointTensorImpl::make("_ctc_loss_backward", rt, {grad, log_probs, targets, neg_log_likelihood, log_alpha})[0];
}

Tensor& checkpoint_hardtanh_backward_out(Tensor& grad_input, const Tensor& grad_output, const Tensor& self, Scalar min_val, Scalar max_val) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor grad_input_ = vec.at(0);
    at::hardtanh_backward_out(grad_input_, vec.at(1), vec.at(2), min_val, max_val);
  };
  CheckpointTensorImpl::mutate("hardtanh_backward_out", mt, {grad_input, grad_output, self}, {0});
  return {grad_input};
}

Tensor checkpoint_hardtanh_backward(const Tensor& grad_output, const Tensor& self, Scalar min_val, Scalar max_val) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::hardtanh_backward(vec.at(0), vec.at(1), min_val, max_val)};
  };
  return CheckpointTensorImpl::make("hardtanh_backward", rt, {grad_output, self})[0];
}

Tensor checkpoint_nonzero(const Tensor& self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::nonzero(vec.at(0))};
  };
  return CheckpointTensorImpl::make("nonzero", rt, {self})[0];
}

Tensor& checkpoint_nonzero_out(Tensor& out, const Tensor& self) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor out_ = vec.at(0);
    at::nonzero_out(out_, vec.at(1));
  };
  CheckpointTensorImpl::mutate("nonzero_out", mt, {out, self}, {0});
  return {out};
}

Tensor checkpoint_lt(const Tensor& self, Scalar other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::lt(vec.at(0), other)};
  };
  return CheckpointTensorImpl::make("lt_Scalar", rt, {self})[0];
}

Tensor& checkpoint_lt_out(Tensor& out, const Tensor& self, Scalar other) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor out_ = vec.at(0);
    at::lt_out(out_, vec.at(1), other);
  };
  CheckpointTensorImpl::mutate("lt_Scalar_out", mt, {out, self}, {0});
  return {out};
}

Tensor checkpoint_lt(const Tensor& self, const Tensor& other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::lt(vec.at(0), vec.at(1))};
  };
  return CheckpointTensorImpl::make("lt_Tensor", rt, {self, other})[0];
}

Tensor& checkpoint_lt_out(Tensor& out, const Tensor& self, const Tensor& other) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor out_ = vec.at(0);
    at::lt_out(out_, vec.at(1), vec.at(2));
  };
  CheckpointTensorImpl::mutate("lt_Tensor_out", mt, {out, self, other}, {0});
  return {out};
}

Tensor checkpoint_any(const Tensor& self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::any(vec.at(0))};
  };
  return CheckpointTensorImpl::make("any", rt, {self})[0];
}

bool checkpoint__use_cudnn_ctc_loss(const Tensor& log_probs, const Tensor& targets, ArrayRef<long> input_lengths, ArrayRef<long> target_lengths, long blank) {
  return at::_use_cudnn_ctc_loss(decheckpoint(log_probs), decheckpoint(targets), input_lengths, target_lengths, blank);
}

bool checkpoint_equal(const Tensor& self, const Tensor& other) {
  // there can't possibly be a reason to rematerialize
  // a single bool so we'll just compute it now
  return at::equal(decheckpoint(self), decheckpoint(other));
}

Scalar checkpoint__local_scalar_dense(at::Tensor const& a) {
  return at::_local_scalar_dense(decheckpoint(a));
}

Tensor checkpoint_split_with_sizes_backward(c10::ArrayRef<at::Tensor> a, c10::ArrayRef<long> b, long c, c10::ArrayRef<long> d, c10::TensorOptions const& e) {
  std::vector<Tensor> a_ = a.vec();
  std::vector<long> d_ = d.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::split_with_sizes_backward(vec, b, c, d_, e)};
    };
  return CheckpointTensorImpl::make("split_with_sizes_backward", rt, a_)[0];
}

std::vector<Tensor> checkpoint_split_with_sizes(at::Tensor const& a, c10::ArrayRef<long> b, long c) {
  std::vector<long> b_ = b.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return at::split_with_sizes(vec.at(0), b_, c);
    };
  return CheckpointTensorImpl::make("split_with_sizes", rt, {a});
}

Tensor checkpoint_split_backward(c10::ArrayRef<at::Tensor> a, long b, long c, c10::ArrayRef<long> d, const c10::TensorOptions& e) {
  std::vector<Tensor> a_ = a.vec();
  std::vector<long> d_ = d.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::split_backward(vec, b, c, d_, e)};
    };
  return CheckpointTensorImpl::make("split_backward", rt, a_)[0];
}

std::vector<Tensor> checkpoint_split(const at::Tensor& a, long b, long c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return at::split(vec.at(0), b, c);
    };
  return CheckpointTensorImpl::make("split", rt, {a});
}

Tensor checkpoint_expand(at::Tensor const& a, c10::ArrayRef<long> b, bool c) {
  std::vector<long> b_ = b.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {vec.at(0).expand(b_, c)};
    };
  return CheckpointTensorImpl::make("expand", rt, {a})[0];
}

Tensor checkpoint_diag(at::Tensor const& self, long diagonal) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::diag(vec.at(0), diagonal)};
    };
  return CheckpointTensorImpl::make("diag", rt, {self})[0];
}

Tensor& checkpoint_diag_out(at::Tensor& out, const Tensor& self, long diagonal) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor out_ = vec.at(0);
      at::diag_out(out_, vec.at(1), diagonal);
    };
  CheckpointTensorImpl::mutate("diag_out", mt, {out, self}, {0});
  return {out};
}

Tensor checkpoint_mv(at::Tensor const& self, at::Tensor const& vec) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::mv(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("mv", rt, {self, vec})[0];
}

Tensor& checkpoint_mv_out(at::Tensor& out, const Tensor& self, const Tensor& vec) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor out_ = vec.at(0);
      at::mv_out(out_, vec.at(1), vec.at(2));
    };
  CheckpointTensorImpl::mutate("mv_out", mt, {out, self, vec}, {0});
  return {out};
}

}}
