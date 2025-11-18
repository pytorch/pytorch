#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/core/ScalarType.h>

#ifdef LAE_USE_CUDA
#include <cuda_runtime.h>
#endif

#include <optional>

using torch::stable::Tensor;

std::vector<Tensor> my__foreach_mul(torch::headeronly::HeaderOnlyArrayRef<Tensor> self, torch::headeronly::HeaderOnlyArrayRef<Tensor> other) {
  std::array<StableIValue, 2> stack = {torch::stable::detail::from(self), torch::stable::detail::from(other)};
  aoti_torch_call_dispatcher("aten::_foreach_mul", "List", stack.data());
  return torch::stable::detail::to<std::vector<Tensor>>(stack[0]);
}

void my__foreach_mul_(torch::headeronly::HeaderOnlyArrayRef<Tensor> self, torch::headeronly::HeaderOnlyArrayRef<Tensor> other) {
  std::array<StableIValue, 2> stack = {torch::stable::detail::from(self), torch::stable::detail::from(other)};
  aoti_torch_call_dispatcher("aten::_foreach_mul_", "List", stack.data());
}

Tensor my_clone(Tensor t) {
  return clone(t);
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

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic_2_10, m) {
  m.def("my__foreach_mul(Tensor[] self, Tensor[] other) -> Tensor[]");
  m.def("my__foreach_mul_(Tensor(a!)[] self, Tensor[] other) -> ()");
  m.def("make_tensor_clones_and_call_foreach(Tensor t1, Tensor t2) -> Tensor[]");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic_2_10, CompositeExplicitAutograd, m) {
  m.impl("my__foreach_mul", TORCH_BOX(&my__foreach_mul));
  m.impl("my__foreach_mul_", TORCH_BOX(&my__foreach_mul_));
  m.impl("make_tensor_clones_and_call_foreach", TORCH_BOX(&make_tensor_clones_and_call_foreach));
}

// Test functions for torch::stable::Tensor device method

torch::stable::Device test_tensor_device(torch::stable::Tensor tensor) {
  return tensor.device();
}

// Test functions for torch::stable::Device

torch::stable::Device test_device_constructor(
    bool is_cuda,
    torch::stable::DeviceIndex index,
    bool use_str) {
  using torch::stable::Device;
  using torch::stable::DeviceType;

  if (use_str) {
    std::string device_str;
    if (is_cuda) {
      device_str = "cuda:" + std::to_string(index);
    } else {
      device_str = "cpu";
    }
    return Device(device_str);
  } else {
    if (is_cuda) {
      return Device(DeviceType::CUDA, index);
    } else {
      return Device(DeviceType::CPU);
    }
  }
}

bool test_device_equality(torch::stable::Device d1, torch::stable::Device d2) {
  return d1 == d2;
}

torch::stable::Device test_device_set_index(
    torch::stable::Device device,
    torch::stable::DeviceIndex index) {
  device.set_index(index);
  return device;
}

torch::stable::DeviceIndex test_device_index(torch::stable::Device device) {
  return device.index();
}

bool test_device_is_cuda(torch::stable::Device device) {
  return device.is_cuda();
}

bool test_device_is_cpu(torch::stable::Device device) {
  return device.is_cpu();
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic_2_10, m) {
  m.def("test_tensor_device(Tensor t) -> Device");
  m.def(
      "test_device_constructor(bool is_cuda, DeviceIndex index, bool use_str) -> Device");
  m.def("test_device_equality(Device d1, Device d2) -> bool");
  m.def("test_device_set_index(Device device, DeviceIndex index) -> Device");
  m.def("test_device_index(Device device) -> DeviceIndex");
  m.def("test_device_is_cuda(Device device) -> bool");
  m.def("test_device_is_cpu(Device device) -> bool");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic_2_10, CompositeExplicitAutograd, m) {
  m.impl("test_tensor_device", TORCH_BOX(&test_tensor_device));
  m.impl("test_device_constructor", TORCH_BOX(&test_device_constructor));
  m.impl("test_device_equality", TORCH_BOX(&test_device_equality));
  m.impl("test_device_set_index", TORCH_BOX(&test_device_set_index));
  m.impl("test_device_index", TORCH_BOX(&test_device_index));
  m.impl("test_device_is_cuda", TORCH_BOX(&test_device_is_cuda));
  m.impl("test_device_is_cpu", TORCH_BOX(&test_device_is_cpu));
}

Tensor test_parallel_for(int64_t size, int64_t grain_size) {
  AtenTensorHandle tensor_handle;
  int64_t stride = 1;

  aoti_torch_empty_strided(
      1,
      &size,
      &stride,
      aoti_torch_dtype_int64(),
      aoti_torch_device_type_cpu(),
      0,
      &tensor_handle);

  Tensor tensor(tensor_handle);
  int64_t* data_ptr = reinterpret_cast<int64_t*>(tensor.data_ptr());

  torch::stable::zero_(tensor);

  // Use parallel_for to fill each element with its index
  // If using a parallel path, the thread id is encoded in the upper 32 bits
  torch::stable::parallel_for(
      0, size, grain_size, [data_ptr](int64_t begin, int64_t end) {
        for (auto i = begin; i < end; i++) {
          STD_TORCH_CHECK(i <= UINT32_MAX);
          uint32_t thread_id;
          torch_get_thread_idx(&thread_id);
          data_ptr[i] = i | (static_cast<int64_t>(thread_id) << 32);
        }
      });

  return tensor;
}

uint32_t test_get_num_threads() {
  return torch::stable::get_num_threads();
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic_2_10, m) {
  m.def("test_parallel_for(int size, int grain_size) -> Tensor");
  m.def("test_get_num_threads() -> int");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic_2_10, CompositeExplicitAutograd, m) {
  m.impl("test_parallel_for", TORCH_BOX(&test_parallel_for));
  m.impl("test_get_num_threads", TORCH_BOX(&test_get_num_threads));
}

Tensor my_empty(
    torch::headeronly::HeaderOnlyArrayRef<int64_t> size,
    std::optional<torch::headeronly::ScalarType> dtype,
    std::optional<torch::stable::Device> device,
    std::optional<bool> pin_memory) {
  return empty(size, dtype, device, pin_memory);
}

Tensor my_reshape(Tensor t, torch::headeronly::HeaderOnlyArrayRef<int64_t> shape) {
  return reshape(t, shape);
}

Tensor my_view(Tensor t, torch::headeronly::HeaderOnlyArrayRef<int64_t> size) {
  return view(t, size);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic_2_10, m) {
  m.def(
      "my_empty(int[] size, ScalarType? dtype=None, Device? device=None, bool? pin_memory=None) -> Tensor");
  m.def("my_reshape(Tensor t, int[] shape) -> Tensor");
  m.def("my_view(Tensor t, int[] size) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic_2_10, CompositeExplicitAutograd, m) {
  m.impl("my_empty", TORCH_BOX(&my_empty));
  m.impl("my_reshape", TORCH_BOX(&my_reshape));
  m.impl("my_view", TORCH_BOX(&my_view));
}

uint64_t get_any_data_ptr(Tensor t, bool mutable_) {
  if (mutable_) {
    return reinterpret_cast<uint64_t>(t.mutable_data_ptr());
  } else {
    return reinterpret_cast<uint64_t>(t.const_data_ptr());
  }
}

uint64_t get_template_any_data_ptr(Tensor t, c10::ScalarType dtype, bool mutable_) {
#define DEFINE_CASE(T, name)                                            \
  case torch::headeronly::ScalarType::name: {                           \
    if (mutable_) {                                                     \
      return reinterpret_cast<uint64_t>(t.mutable_data_ptr<T>());       \
    } else {                                                            \
      return reinterpret_cast<uint64_t>(t.const_data_ptr<T>());         \
    }                                                                   \
  }
  switch (dtype) {
    // per aten/src/ATen/templates/TensorMethods.cpp:
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CASE)
    DEFINE_CASE(uint16_t, UInt16)
    DEFINE_CASE(uint32_t, UInt32)
    DEFINE_CASE(uint64_t, UInt64)
  default:
      return 0;
  }
#undef DEFINE_CASE
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic_2_10, m) {
  m.def("get_any_data_ptr(Tensor t, bool mutable_) -> int");
  m.def("get_template_any_data_ptr(Tensor t, ScalarType dtype, bool mutable_) -> int");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic_2_10, CompositeExplicitAutograd, m) {
  m.impl("get_any_data_ptr", TORCH_BOX(&get_any_data_ptr));
  m.impl("get_template_any_data_ptr", TORCH_BOX(&get_template_any_data_ptr));
}
