#include <gtest/gtest.h>

#include <torch/csrc/inductor/aoti_runtime/utils.h>

#include <string>

namespace {

std::string error_of_empty_strided(int64_t size) {
  AtenTensorHandle out = nullptr;
  int64_t stride = 1;
  try {
    AOTI_TORCH_ERROR_CODE_CHECK_WITH_MESSAGE(aoti_torch_empty_strided(
        1,
        &size,
        &stride,
        aoti_torch_dtype_float32(),
        aoti_torch_device_type_cpu(),
        0,
        &out));
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  aoti_torch_delete_tensor_object(out);
  return "no error";
}

TEST(AOTITorchErrorCodeCheckWithMessage, KeepsTheShimMessage) {
  std::string what = error_of_empty_strided(-1);
  EXPECT_EQ(what.find("Trying to create tensor with negative dimension -1"), 0u)
      << what;
  EXPECT_NE(what.find(" (originally from aoti_torch_empty_strided("), what.npos)
      << what;
  EXPECT_NE(what.find("test_error_check.cpp, line "), what.npos) << what;
}

TEST(AOTITorchErrorCodeCheckWithMessage, PassesOnSuccess) {
  EXPECT_EQ(error_of_empty_strided(2), "no error");
}

} // namespace
