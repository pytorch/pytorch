#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Exception.h>

#include <optional>

void inline sgd_math(
  float* param_ptr,
  float* grad_ptr,
  float* out_ptr,
  const float weight_decay,
  const double lr,
  const bool maximize,
  int64_t size
){
  int64_t d = 0;
  for (; d < size; d++) {
    float grad_val = grad_ptr[d];
    if (maximize) grad_val = -grad_val;
    if (weight_decay != 0.0){
      grad_val += param_ptr[d] * weight_decay;
    }
    out_ptr[d] = param_ptr[d] - grad_val * float(lr);
  }
}

using torch::stable::Tensor;

Tensor sgd_out_of_place(
    const Tensor param,
    const Tensor grad,
    const float weight_decay,
    const double lr,
    const bool maximize) {
  STD_TORCH_CHECK(param.dim() == 1, "param must be 1D");

  int64_t *param_sizes;
  int64_t *param_strides;
  aoti_torch_get_sizes(param.get(), &param_sizes);
  aoti_torch_get_strides(param.get(), &param_strides);

  int32_t param_dtype;
  aoti_torch_get_dtype(param.get(), &param_dtype);

  int32_t param_device_type;
  aoti_torch_get_device_type(param.get(), &param_device_type);

  AtenTensorHandle out_ath;
  aoti_torch_empty_strided(param.dim(), param_sizes, param_strides, param_dtype, param_device_type, param.get_device(), &out_ath);
  auto out = Tensor(out_ath);

  sgd_math(
    reinterpret_cast<float*>(param.data_ptr()),
    reinterpret_cast<float*>(grad.data_ptr()),
    reinterpret_cast<float*>(out.data_ptr()),
    weight_decay,
    lr,
    maximize,
    param.numel()
  );

  return out;
}

void boxed_sgd_out_of_place(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  Tensor res = sgd_out_of_place(
    to<Tensor>(stack[0]),
    to<Tensor>(stack[1]),
    float(to<double>(stack[2])),
    to<double>(stack[3]),
    to<bool>(stack[4]));

  stack[0] = from(res);
}

STABLE_TORCH_LIBRARY(libtorch_agnostic, m) {
  m.def("sgd_out_of_place(Tensor param, Tensor grad, float weight_decay, float lr, bool maximize) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CPU, m) {
  m.impl("sgd_out_of_place", &boxed_sgd_out_of_place);
}

Tensor identity(Tensor t) {
  return t;
}

void boxed_identity(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  Tensor res = identity(to<Tensor>(stack[0]));
  stack[0] = from(res);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic, m) {
  m.def("identity(Tensor t) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CUDA, m) {
  m.impl("identity", &boxed_identity);
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CPU, m) {
  m.impl("identity", &boxed_identity);
}

Tensor my_abs(Tensor t) {
  const auto num_args = 1;
  StableIValue stack[num_args];
  stack[0] = from(t);
  aoti_torch_call_dispatcher("aten::abs", "", stack);
  return to<Tensor>(stack[0]);
}

void boxed_my_abs(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  Tensor tensor_res = my_abs(to<Tensor>(stack[0]));
  stack[0] = from(tensor_res);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic, m) {
  m.def("my_abs(Tensor t) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CompositeExplicitAutograd, m) {
  m.impl("my_abs", &boxed_my_abs);
}

Tensor my_ones_like(Tensor t, StableIValue device) {
  const auto num_args = 6;
  StableIValue stack[num_args];

  int32_t t_dtype;
  aoti_torch_get_dtype(t.get(), &t_dtype);
  auto mf = aoti_torch_memory_format_contiguous_format();

  stack[0] = from(t);
  stack[1] = from(std::optional(t_dtype));    // dtype
  stack[2] = from(std::nullopt);              // layout
  stack[3] = from(std::optional(device));     // device
  stack[4] = from(std::optional(false));      // pin_memory
  stack[5] = from(std::optional(mf));         // memory_format

  aoti_torch_call_dispatcher("aten::ones_like", "", stack);

  return to<Tensor>(stack[0]);
}

void boxed_my_ones_like(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  Tensor res = my_ones_like(to<Tensor>(stack[0]), stack[1]);
  stack[0] = from(res);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic, m) {
  m.def("my_ones_like(Tensor t, Device d) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CompositeExplicitAutograd, m) {
  m.impl("my_ones_like", &boxed_my_ones_like);
}

std::tuple<Tensor, Tensor, bool> exp_neg_is_leaf(Tensor t1, Tensor t2, Tensor t3) {
  StableIValue stack_exp[1];
  stack_exp[0] = from(t1);
  aoti_torch_call_dispatcher("aten::exp", "", stack_exp);

  StableIValue stack_neg[1];
  stack_neg[0] = from(t2);
  aoti_torch_call_dispatcher("aten::neg", "", stack_neg);

  StableIValue stack_is_leaf[1];
  stack_is_leaf[0] = from(t3);
  aoti_torch_call_dispatcher("aten::is_leaf", "", stack_is_leaf);

  return std::make_tuple(
    to<Tensor>(stack_exp[0]),
    to<Tensor>(stack_neg[0]),
    to<bool>(stack_is_leaf[0]));
}

void boxed_exp_neg_is_leaf(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  auto tuple = exp_neg_is_leaf(to<Tensor>(stack[0]), to<Tensor>(stack[1]), to<Tensor>(stack[2]));
  stack[0] = from(std::get<0>(tuple));
  stack[1] = from(std::get<1>(tuple));
  stack[2] = from(std::get<2>(tuple));
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic, m) {
  m.def("exp_neg_is_leaf(Tensor t1, Tensor t2, Tensor t3) -> (Tensor, Tensor, bool)");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CompositeExplicitAutograd, m) {
  m.impl("exp_neg_is_leaf", &boxed_exp_neg_is_leaf);
}

Tensor neg_exp(Tensor t) {
  StableIValue stack[1];
  stack[0] = from(t);
  aoti_torch_call_dispatcher("aten::exp", "", stack);
  aoti_torch_call_dispatcher("aten::neg", "", stack);
  return to<Tensor>(stack[0]);
}

void boxed_neg_exp(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  Tensor res = neg_exp(to<Tensor>(stack[0]));
  stack[0] = from(res);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic, m) {
  m.def("neg_exp(Tensor t) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CompositeExplicitAutograd, m) {
  m.impl("neg_exp", &boxed_neg_exp);
}

Tensor divide_neg_exp(Tensor t) {
  StableIValue stack_neg[1];
  stack_neg[0] = from(t);

  StableIValue stack_exp[1];
  stack_exp[0] = from(t);
  aoti_torch_call_dispatcher("aten::exp", "", stack_exp);
  aoti_torch_call_dispatcher("aten::neg", "", stack_neg);

  StableIValue stack_div[2];
  stack_div[0] = stack_neg[0];
  stack_div[1] = stack_exp[0];
  aoti_torch_call_dispatcher("aten::divide", "Tensor", stack_div);
  return to<Tensor>(stack_div[0]);
}

void boxed_divide_neg_exp(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  Tensor res = divide_neg_exp(to<Tensor>(stack[0]));
  stack[0] = from(res);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic, m) {
  m.def("divide_neg_exp(Tensor t) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CompositeExplicitAutograd, m) {
  m.impl("divide_neg_exp", &boxed_divide_neg_exp);
}

bool is_contiguous(Tensor t) {
  return t.is_contiguous();
}

void boxed_is_contiguous(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  bool res = is_contiguous(to<Tensor>(stack[0]));
  stack[0] = from(res);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic, m) {
  m.def("is_contiguous(Tensor t) -> bool");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CompositeExplicitAutograd, m) {
  m.impl("is_contiguous", &boxed_is_contiguous);
}

Tensor my_transpose(Tensor t, int64_t dim0, int64_t dim1) {
  return transpose(t, dim0, dim1);
}

void boxed_my_transpose(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  auto res = my_transpose(to<Tensor>(stack[0]), to<int64_t>(stack[1]), to<int64_t>(stack[2]));

  stack[0] = from(res);
}

Tensor my_empty_like(Tensor t) {
  return empty_like(t);
}

void boxed_empty_like(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  auto res = my_empty_like(to<Tensor>(stack[0]));
  stack[0] = from(res);
}

Tensor fill_infinity(Tensor t) {
  auto value = std::numeric_limits<float>::infinity();
  return fill_(t, value);
}

void boxed_fill_infinity(
    StableIValue* stack,
    uint64_t num_args,
    uint64_t num_outputs) {
  auto res = fill_infinity(to<Tensor>(stack[0]));
  stack[0] = from(res);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic, m) {
  m.def("my_transpose(Tensor t, int dim0, int dim1) -> Tensor");
  m.def("my_empty_like(Tensor t) -> Tensor");
  m.def("fill_infinity(Tensor(a!) t) -> Tensor(a!)");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CompositeExplicitAutograd, m) {
  m.impl("my_transpose", &boxed_my_transpose);
  m.impl("my_empty_like", &boxed_empty_like);
  m.impl("fill_infinity", &boxed_fill_infinity);
}


Tensor my_zero_(Tensor t) {
  return zero_(t);
}

void boxed_my_zero_(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  auto res = my_zero_(to<Tensor>(stack[0]));
  stack[0] = from(res);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic, m) {
  m.def("my_zero_(Tensor(a!) t) -> Tensor(a!)");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CPU, m) {
  m.impl("my_zero_", &boxed_my_zero_);
}
