#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/core/ScalarType.h>

#ifdef LAE_USE_CUDA
#include <cuda_runtime.h>
#endif

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
    const double weight_decay,
    const double lr,
    const bool maximize) {
  STD_TORCH_CHECK(param.dim() == 1, "param must be 1D");

  // these test the get_device() and get_device_index() methods
  // while ascertaining that we are still on CPU
  STD_TORCH_CHECK(param.get_device() == -1, "CPU device index = -1");
  STD_TORCH_CHECK(param.get_device_index() == -1, "CPU device index = -1");

  // testing Tensor strides + stride
  STD_TORCH_CHECK(param.strides()[0] == param.stride(0));

  auto out = new_empty(param, param.sizes());

  sgd_math(
    reinterpret_cast<float*>(param.data_ptr()),
    reinterpret_cast<float*>(grad.data_ptr()),
    reinterpret_cast<float*>(out.data_ptr()),
    float(weight_decay),
    lr,
    maximize,
    param.numel()
  );

  return out;
}

STABLE_TORCH_LIBRARY(libtorch_agnostic, m) {
  m.def("sgd_out_of_place(Tensor param, Tensor grad, float weight_decay, float lr, bool maximize) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CPU, m) {
  m.impl("sgd_out_of_place", TORCH_BOX(&sgd_out_of_place));
}

Tensor identity(Tensor t) {
  return t;
}


STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic, m) {
  m.def("identity(Tensor t) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CUDA, m) {
  m.impl("identity", TORCH_BOX(&identity));
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CPU, m) {
  m.impl("identity", TORCH_BOX(&identity));
}

Tensor my_abs(Tensor t) {
  const auto num_args = 1;
  StableIValue stack[num_args];
  stack[0] = torch::stable::detail::from(t);
  aoti_torch_call_dispatcher("aten::abs", "", stack);
  return torch::stable::detail::to<Tensor>(stack[0]);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic, m) {
  m.def("my_abs(Tensor t) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CompositeExplicitAutograd, m) {
  m.impl("my_abs", TORCH_BOX(&my_abs));
}

Tensor my_ones_like(Tensor t, StableIValue device) {
  const auto num_args = 6;
  StableIValue stack[num_args];

  auto mf = aoti_torch_memory_format_contiguous_format();

  stack[0] = torch::stable::detail::from(t);
  stack[1] = torch::stable::detail::from(std::optional(t.scalar_type()));    // dtype
  stack[2] = torch::stable::detail::from(std::nullopt);              // layout
  stack[3] = torch::stable::detail::from(std::optional(device));     // device
  stack[4] = torch::stable::detail::from(std::optional(false));      // pin_memory
  stack[5] = torch::stable::detail::from(std::optional(mf));         // memory_format

  aoti_torch_call_dispatcher("aten::ones_like", "", stack);

  return torch::stable::detail::to<Tensor>(stack[0]);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic, m) {
  m.def("my_ones_like(Tensor t, Device d) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CompositeExplicitAutograd, m) {
  m.impl("my_ones_like", TORCH_BOX(&my_ones_like));
}

std::tuple<Tensor, Tensor, bool> exp_neg_is_leaf(Tensor t1, Tensor t2, Tensor t3) {
  StableIValue stack_exp[1];
  stack_exp[0] = torch::stable::detail::from(t1);
  aoti_torch_call_dispatcher("aten::exp", "", stack_exp);

  StableIValue stack_neg[1];
  stack_neg[0] = torch::stable::detail::from(t2);
  aoti_torch_call_dispatcher("aten::neg", "", stack_neg);

  StableIValue stack_is_leaf[1];
  stack_is_leaf[0] = torch::stable::detail::from(t3);
  aoti_torch_call_dispatcher("aten::is_leaf", "", stack_is_leaf);

  return std::make_tuple(
    torch::stable::detail::to<Tensor>(stack_exp[0]),
    torch::stable::detail::to<Tensor>(stack_neg[0]),
    torch::stable::detail::to<bool>(stack_is_leaf[0]));
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic, m) {
  m.def("exp_neg_is_leaf(Tensor t1, Tensor t2, Tensor t3) -> (Tensor, Tensor, bool)");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CompositeExplicitAutograd, m) {
  m.impl("exp_neg_is_leaf", TORCH_BOX(&exp_neg_is_leaf));
}

Tensor neg_exp(Tensor t) {
  StableIValue stack[1];
  stack[0] = torch::stable::detail::from(t);
  aoti_torch_call_dispatcher("aten::exp", "", stack);
  aoti_torch_call_dispatcher("aten::neg", "", stack);
  return torch::stable::detail::to<Tensor>(stack[0]);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic, m) {
  m.def("neg_exp(Tensor t) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CompositeExplicitAutograd, m) {
  m.impl("neg_exp", TORCH_BOX(&neg_exp));
}

Tensor divide_neg_exp(Tensor t) {
  StableIValue stack_neg[1];
  stack_neg[0] = torch::stable::detail::from(t);

  StableIValue stack_exp[1];
  stack_exp[0] = torch::stable::detail::from(t);
  aoti_torch_call_dispatcher("aten::exp", "", stack_exp);
  aoti_torch_call_dispatcher("aten::neg", "", stack_neg);

  StableIValue stack_div[2];
  stack_div[0] = stack_neg[0];
  stack_div[1] = stack_exp[0];
  aoti_torch_call_dispatcher("aten::divide", "Tensor", stack_div);
  return torch::stable::detail::to<Tensor>(stack_div[0]);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic, m) {
  m.def("divide_neg_exp(Tensor t) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CompositeExplicitAutograd, m) {
  m.impl("divide_neg_exp", TORCH_BOX(&divide_neg_exp));
}

bool is_contiguous(Tensor t) {
  return t.is_contiguous();
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic, m) {
  m.def("is_contiguous(Tensor t) -> bool");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CompositeExplicitAutograd, m) {
  m.impl("is_contiguous", TORCH_BOX(&is_contiguous));
}

Tensor my_transpose(Tensor t, int64_t dim0, int64_t dim1) {
  return transpose(t, dim0, dim1);
}

Tensor my_empty_like(Tensor t) {
  return empty_like(t);
}

bool my_is_cpu(Tensor t) {
  return t.is_cpu();
}

Tensor fill_infinity(Tensor t) {
  auto value = std::numeric_limits<float>::infinity();
  return fill_(t, value);
}

Tensor my_pad(Tensor t) {
  std::string mode = "constant";
  double value = 0.0;
  return pad(t, {1, 2, 2, 1}, mode, value);
}

Tensor my_narrow(Tensor t, int64_t dim, int64_t start, int64_t length) {
  return narrow(t, dim, start, length);
}

Tensor my_new_empty_dtype_variant(Tensor t) {
  // Still using a std::vector below even though people can just pass in an
  // initializer list (which will be implicitly converted to an HeaderOnlyArrayRef)
  // directly.
  // This is to test that passing in a std::vector works for BC. (It gets
  // implicitly converted to HeaderOnlyArrayRef too!)
  std::vector<int64_t> sizes = {2, 5};
  auto dtype = std::make_optional(torch::headeronly::ScalarType::BFloat16);
  return new_empty(t, sizes, dtype);
}

Tensor my_new_zeros_dtype_variant(Tensor t) {
  auto dtype = std::make_optional(at::ScalarType::Float);
  return new_zeros(t, {2, 5}, dtype);
}

Tensor my_copy_(Tensor dst, Tensor src, bool non_blocking) {
  return copy_(dst, src, non_blocking);
}

Tensor my_clone(Tensor t) {
  return clone(t);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic, m) {
  m.def("my_transpose(Tensor t, int dim0, int dim1) -> Tensor");
  m.def("my_empty_like(Tensor t) -> Tensor");
  m.def("fill_infinity(Tensor(a!) t) -> Tensor(a!)");
  m.def("my_pad(Tensor t) -> Tensor");
  m.def("my_narrow(Tensor t, int dim, int start, int length) -> Tensor");
  m.def("my_new_empty_dtype_variant(Tensor t) -> Tensor");
  m.def("my_new_zeros_dtype_variant(Tensor t) -> Tensor");
  m.def("my_copy_(Tensor dst, Tensor src, bool non_blocking) -> Tensor");
  m.def("my_clone(Tensor t) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CompositeExplicitAutograd, m) {
  m.impl("my_transpose", TORCH_BOX(&my_transpose));
  m.impl("my_empty_like", TORCH_BOX(&my_empty_like));
  m.impl("fill_infinity", TORCH_BOX(&fill_infinity));
  m.impl("my_is_cpu", TORCH_BOX(&my_is_cpu));
  m.impl("my_new_empty_dtype_variant", TORCH_BOX(&my_new_empty_dtype_variant));
  m.impl("my_new_zeros_dtype_variant", TORCH_BOX(&my_new_zeros_dtype_variant));
  m.impl("my_copy_", TORCH_BOX(&my_copy_));
  m.impl("my_clone", TORCH_BOX(&my_clone));
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CompositeImplicitAutograd, m) {
  m.impl("my_pad", TORCH_BOX(&my_pad));
  m.impl("my_narrow", TORCH_BOX(&my_narrow));
}

Tensor my_zero_(Tensor t) {
  return zero_(t);
}

Tensor my_amax(Tensor t) {
  return amax(t, 0, false);
}

Tensor my_amax_vec(Tensor t) {
  return amax(t, {0,1}, false);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic, m) {
  m.def("my_zero_(Tensor(a!) t) -> Tensor(a!)");
  m.def("my_amax(Tensor a) -> Tensor");
  m.def("my_amax_vec(Tensor a) -> Tensor");
  m.def("my_is_cpu(Tensor t) -> bool");
   m.def("test_default_constructor(bool undefined) -> bool");
}

bool test_default_constructor(bool defined) {
  Tensor out;
  if (defined) {
    AtenTensorHandle defined_ath;
    int64_t sizes[] = {2, 3};
    int64_t strides[] = {3, 1};
    aoti_torch_empty_strided(
        2,
        sizes,
        strides,
        aoti_torch_dtype_float32(),
        aoti_torch_device_type_cpu(),
        0,
        &defined_ath);
    out = Tensor(defined_ath);
  }
  return out.defined();
}


STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CompositeExplicitAutograd, m) {
  m.impl("my_zero_", TORCH_BOX(&my_zero_));
  m.impl("my_amax", TORCH_BOX(&my_amax));
  m.impl("my_amax_vec", TORCH_BOX(&my_amax_vec));
  m.impl("test_default_constructor", TORCH_BOX(&test_default_constructor));
}

std::vector<Tensor> my__foreach_mul(torch::headeronly::HeaderOnlyArrayRef<Tensor> self, torch::headeronly::HeaderOnlyArrayRef<Tensor> other) {
  std::array<StableIValue, 2> stack = {torch::stable::detail::from(self), torch::stable::detail::from(other)};
  aoti_torch_call_dispatcher("aten::_foreach_mul", "List", stack.data());
  return torch::stable::detail::to<std::vector<Tensor>>(stack[0]);
}

// void boxed_my__foreach_mul(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
//   // Why is the following NOT torch::stable::detail::to<HeaderOnlyArrayRef<Tensor>>(stack[0])? Because calling `to`
//   // on a StableIValue means that the result is owning its underlying data now! HeaderOnlyArrayRef
//   // is not owning, so it cannot safely steward the result of the torch::stable::detail::to<>.
//   auto res = my__foreach_mul(torch::stable::detail::to<std::vector<Tensor>>(stack[0]), torch::stable::detail::to<std::vector<Tensor>>(stack[1]));
//   stack[0] = torch::stable::detail::from(res);
// }

void my__foreach_mul_(torch::headeronly::HeaderOnlyArrayRef<Tensor> self, torch::headeronly::HeaderOnlyArrayRef<Tensor> other) {
  std::array<StableIValue, 2> stack = {torch::stable::detail::from(self), torch::stable::detail::from(other)};
  aoti_torch_call_dispatcher("aten::_foreach_mul_", "List", stack.data());
}

std::vector<Tensor> make_tensor_clones_and_call_foreach(Tensor t1, Tensor t2) {
  // This function tests that my__foreach_mul can take in std::initializer_lists
  // in addition to std::vectors.
  Tensor t1_1 = my_clone(t1);
  Tensor t1_2 = my_clone(t1);
  Tensor t2_1 = my_clone(t2);
  Tensor t2_2 = my_clone(t2);
  return my__foreach_mul({t1_1, t2_1}, {t1_2, t2_2});
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic, m) {
  m.def("my__foreach_mul(Tensor[] self, Tensor[] other) -> Tensor[]");
  m.def("my__foreach_mul_(Tensor(a!)[] self, Tensor[] other) -> ()");
  m.def("make_tensor_clones_and_call_foreach(Tensor t1, Tensor t2) -> Tensor[]");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CompositeExplicitAutograd, m) {
  m.impl("my__foreach_mul", TORCH_BOX(&my__foreach_mul));
  m.impl("my__foreach_mul_", TORCH_BOX(&my__foreach_mul_));
  m.impl("make_tensor_clones_and_call_foreach", TORCH_BOX(&make_tensor_clones_and_call_foreach));
}

// Test functions for torch::stable::accelerator APIs

#ifdef LAE_USE_CUDA
int64_t test_device_guard(int64_t device_index) {
  using torch::stable::accelerator::DeviceGuard;

  STD_TORCH_CHECK(
      device_index >= std::numeric_limits<int32_t>::min() &&
          device_index <= std::numeric_limits<int32_t>::max(),
      "Device index is out of range of DeviceIndex (int32_t).");

  DeviceGuard guard(device_index);
  int currentDevice;
  cudaError_t err = cudaGetDevice(&currentDevice);
  STD_TORCH_CHECK(err == cudaSuccess);
  return currentDevice;
}

int64_t test_device_guard_set_index() {
  using torch::stable::accelerator::DeviceGuard;

  DeviceGuard guard(1);
  guard.set_index(0);
  int currentDevice;
  cudaError_t err = cudaGetDevice(&currentDevice);
  STD_TORCH_CHECK(err == cudaSuccess);
  return currentDevice;
}

int64_t test_stream(int32_t device_index) {
  STD_TORCH_CHECK(
      device_index >= std::numeric_limits<int32_t>::min() &&
          device_index <= std::numeric_limits<int32_t>::max(),
      "Device index is out of range of DeviceIndex (int32_t).");

  return torch::stable::accelerator::getCurrentStream(device_index).id();
}

int64_t test_get_current_device_index() {
  return torch::stable::accelerator::getCurrentDeviceIndex();
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic, m) {
  m.def("test_device_guard(int device_index) -> int");
  m.def("test_device_guard_set_index() -> int");
  m.def("test_stream(int device_index) -> int");
  m.def("test_get_current_device_index() -> int");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CompositeExplicitAutograd, m) {
  m.impl("test_device_guard", TORCH_BOX(&test_device_guard));
  m.impl("test_device_guard_set_index", TORCH_BOX(&test_device_guard_set_index));
  m.impl("test_stream", TORCH_BOX(&test_stream));
  m.impl("test_get_current_device_index", TORCH_BOX(&test_get_current_device_index));
}

#endif // LAE_USE_CUDA
