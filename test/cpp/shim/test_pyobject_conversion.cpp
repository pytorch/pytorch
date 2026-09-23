// Checks that the PyObject<->Tensor conversion shims fall back to a clean error
// (rather than crashing) when libtorch_python is not loaded. This binary links
// only libtorch, so the conversion vtable is the default no-op.

#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/stable/c/shim.h>

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
