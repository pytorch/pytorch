// Checks that the Python interop conversion shims fall back to a clean error
// (rather than crashing) when libtorch_python is not loaded. This binary links
// only libtorch, so the conversion vtable is the default no-op.

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
  // The no-op errors rather than answering false.
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

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0
  const int32_t mtia_device_type = aoti_torch_device_type_mtia();
  EXPECT_EQ(mtia_device_type, static_cast<int32_t>(c10::DeviceType::MTIA));
  EXPECT_EQ(
      torch_device_to_pyobject(mtia_device_type, /*device_index=*/0, &py_out),
      AOTI_TORCH_FAILURE);
  EXPECT_EQ(py_out, nullptr);
#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0
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

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

TEST(TorchPyObjectConversion, LayoutNoopErrorsWithoutLibtorchPython) {
  int dummy = 0;
  int32_t layout_out = -1;
  EXPECT_EQ(
      torch_layout_from_pyobject(&dummy, &layout_out), AOTI_TORCH_FAILURE);
  EXPECT_EQ(layout_out, -1);

  void* py_out = nullptr;
  EXPECT_EQ(
      torch_layout_to_pyobject(aoti_torch_layout_sparse_csr(), &py_out),
      AOTI_TORCH_FAILURE);
  EXPECT_EQ(py_out, nullptr);
}

TEST(TorchPyObjectConversion, LayoutCodeValidatedBeforeNarrowing) {
  void* py_out = nullptr;
  EXPECT_EQ(torch_layout_to_pyobject(256, &py_out), AOTI_TORCH_FAILURE);
  EXPECT_EQ(py_out, nullptr);
  EXPECT_NE(
      std::strstr(
          torch_exception_get_what_without_backtrace(),
          "invalid layout code 256"),
      nullptr);
}

namespace {

void expect_noop_error(AOTITorchError status) {
  EXPECT_EQ(status, AOTI_TORCH_FAILURE);
  const char* message = torch_exception_get_what_without_backtrace();
  ASSERT_NE(message, nullptr);
  EXPECT_NE(
      std::strstr(message, "requires libtorch_python to be loaded"), nullptr);
}

} // namespace

TEST(TorchPyObjectConversion, MemoryFormatNoopErrorsWithoutLibtorchPython) {
  int dummy = 0;
  int32_t memory_format_out = -1;
  expect_noop_error(
      torch_memory_format_from_pyobject(&dummy, &memory_format_out));
  EXPECT_EQ(memory_format_out, -1);

  void* py_out = nullptr;
  expect_noop_error(torch_memory_format_to_pyobject(
      aoti_torch_memory_format_channels_last(), &py_out));
  EXPECT_EQ(py_out, nullptr);
}

TEST(TorchPyObjectConversion, MemoryFormatCodeValidatedBeforeNarrowing) {
  void* py_out = nullptr;
  EXPECT_EQ(torch_memory_format_to_pyobject(256, &py_out), AOTI_TORCH_FAILURE);
  EXPECT_EQ(py_out, nullptr);
  const char* message = torch_exception_get_what_without_backtrace();
  ASSERT_NE(message, nullptr);
  EXPECT_NE(std::strstr(message, "invalid memory format code 256"), nullptr);
}

TEST(TorchPyObjectConversion, ScalarNoopErrorsWithoutLibtorchPython) {
  int dummy = 0;
  bool bool_out = true;
  expect_noop_error(torch_bool_from_pyobject(&dummy, &bool_out));
  EXPECT_TRUE(bool_out);
  int64_t int64_out = -1;
  expect_noop_error(torch_int64_from_pyobject(&dummy, &int64_out));
  EXPECT_EQ(int64_out, -1);
  double double_out = 1.25;
  expect_noop_error(torch_double_from_pyobject(&dummy, &double_out));
  EXPECT_EQ(double_out, 1.25);

  void* py_out = nullptr;
  expect_noop_error(torch_bool_to_pyobject(true, &py_out));
  EXPECT_EQ(py_out, nullptr);
  expect_noop_error(torch_int64_to_pyobject(17, &py_out));
  EXPECT_EQ(py_out, nullptr);
  expect_noop_error(torch_double_to_pyobject(1.5, &py_out));
  EXPECT_EQ(py_out, nullptr);
}

TEST(TorchPyObjectConversion, StringNoopErrorsWithoutLibtorchPython) {
  int dummy = 0;
  StringHandle string_out = nullptr;
  expect_noop_error(torch_string_from_pyobject(&dummy, &string_out));
  EXPECT_EQ(string_out, nullptr);

  const char data[] = "data";
  void* py_out = nullptr;
  expect_noop_error(torch_string_to_pyobject(data, sizeof(data) - 1, &py_out));
  EXPECT_EQ(py_out, nullptr);
}

TEST(TorchPyObjectConversion, StringDataValidatedBeforePythonConversion) {
  void* py_out = nullptr;
  EXPECT_EQ(torch_string_to_pyobject(nullptr, 1, &py_out), AOTI_TORCH_FAILURE);
  EXPECT_EQ(py_out, nullptr);
  const char* message = torch_exception_get_what_without_backtrace();
  ASSERT_NE(message, nullptr);
  EXPECT_NE(
      std::strstr(message, "data must not be null unless length is 0"),
      nullptr);
}

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0
