// Checks that the PyObject<->Tensor/dtype/device conversion shims fall back to
// a clean error (rather than crashing) when libtorch_python is not loaded. This
// binary links only libtorch, so the conversion vtable is the default no-op.

#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/stable/c/shim.h>

#include <cstring>

TEST(TorchPyObjectConversion, NoopErrorsWithoutLibtorchPython) {
  // from_pyobject: the no-op errors before touching py_obj, so a dummy non-null
  // pointer is enough to reach it.
  int dummy = 0;
  AtenTensorHandle ath_out = nullptr;
  EXPECT_EQ(torch_tensor_from_pyobject(&dummy, &ath_out), AOTI_TORCH_FAILURE);
  EXPECT_EQ(ath_out, nullptr);

  // to_pyobject: needs a real tensor handle (the shim dereferences it before
  // reaching the no-op).
  AtenTensorHandle ath = torch::aot_inductor::new_tensor_handle(at::zeros({1}));
  void* py_out = nullptr;
  EXPECT_EQ(
      torch_tensor_to_pyobject(ath, /*py_type=*/nullptr, &py_out),
      AOTI_TORCH_FAILURE);
  EXPECT_EQ(py_out, nullptr);
  aoti_torch_delete_tensor_object(ath);
}

TEST(TorchPyObjectConversion, IsTensorNoopErrorsWithoutLibtorchPython) {
  // The no-op errors (rather than answering false): a silent false would mask
  // libtorch_python not being loaded.
  int dummy = 0;
  bool is_tensor_out = false;
  EXPECT_EQ(
      torch_is_tensor_pyobject(&dummy, &is_tensor_out), AOTI_TORCH_FAILURE);
  EXPECT_FALSE(is_tensor_out);
}

TEST(TorchPyObjectConversion, DtypeNoopErrorsWithoutLibtorchPython) {
  int dummy = 0;
  int32_t dtype_out = -1;
  EXPECT_EQ(torch_dtype_from_pyobject(&dummy, &dtype_out), AOTI_TORCH_FAILURE);
  EXPECT_EQ(dtype_out, -1);

  void* py_out = nullptr;
  EXPECT_EQ(
      torch_dtype_to_pyobject(aoti_torch_dtype_float32(), &py_out),
      AOTI_TORCH_FAILURE);
  EXPECT_EQ(py_out, nullptr);
}

TEST(TorchPyObjectConversion, DeviceNoopErrorsWithoutLibtorchPython) {
  int dummy = 0;
  int32_t device_type_out = -1;
  int32_t device_index_out = -1;
  EXPECT_EQ(
      torch_device_from_pyobject(&dummy, &device_type_out, &device_index_out),
      AOTI_TORCH_FAILURE);
  EXPECT_EQ(device_type_out, -1);
  EXPECT_EQ(device_index_out, -1);

  void* py_out = nullptr;
  EXPECT_EQ(
      torch_device_to_pyobject(
          aoti_torch_device_type_cpu(), /*device_index=*/-1, &py_out),
      AOTI_TORCH_FAILURE);
  EXPECT_EQ(py_out, nullptr);
}

TEST(TorchPyObjectConversion, DtypeCodeValidatedBeforeNarrowing) {
  void* py_out = nullptr;
  EXPECT_EQ(torch_dtype_to_pyobject(256, &py_out), AOTI_TORCH_FAILURE);
  EXPECT_EQ(py_out, nullptr);
  EXPECT_NE(
      std::strstr(
          torch_exception_get_what_without_backtrace(),
          "invalid dtype code 256"),
      nullptr);
}

TEST(TorchPyObjectConversion, DeviceValuesValidatedBeforeNarrowing) {
  void* py_out = nullptr;
  EXPECT_EQ(
      torch_device_to_pyobject(256, /*device_index=*/-1, &py_out),
      AOTI_TORCH_FAILURE);
  EXPECT_EQ(py_out, nullptr);
  EXPECT_NE(
      std::strstr(
          torch_exception_get_what_without_backtrace(),
          "invalid device type 256"),
      nullptr);

  EXPECT_EQ(
      torch_device_to_pyobject(
          aoti_torch_device_type_cuda(), /*device_index=*/256, &py_out),
      AOTI_TORCH_FAILURE);
  EXPECT_EQ(py_out, nullptr);
  EXPECT_NE(
      std::strstr(
          torch_exception_get_what_without_backtrace(),
          "device index 256 is out of range"),
      nullptr);
}
