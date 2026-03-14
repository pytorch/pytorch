#include <torch/csrc/inductor/aoti_runtime/kernel_capture.h>
#include <torch/csrc/inductor/aoti_runtime/utils.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include <atomic>
#include <filesystem>
#include <thread>

using torch::aot_inductor::ConstantHandle;
using torch::aot_inductor::RAIIAtenTensorHandle;

// ===== Type Traits =====

TEST(KernelCaptureTypeTraits, TensorLike) {
  static_assert(
      aoti_kernel_capture::is_tensor_like<AtenTensorHandle>::value,
      "AtenTensorHandle should be tensor-like");
  static_assert(
      aoti_kernel_capture::is_tensor_like<RAIIAtenTensorHandle>::value,
      "RAIIAtenTensorHandle should be tensor-like");
  static_assert(
      aoti_kernel_capture::is_tensor_like<ConstantHandle>::value,
      "ConstantHandle should be tensor-like");

  static_assert(
      !aoti_kernel_capture::is_tensor_like<int64_t>::value,
      "int64_t should not be tensor-like");
  static_assert(
      !aoti_kernel_capture::is_tensor_like<double>::value,
      "double should not be tensor-like");
  static_assert(
      !aoti_kernel_capture::is_tensor_like<const char*>::value,
      "const char* should not be tensor-like");
}

TEST(KernelCaptureTypeTraits, IntScalar) {
  static_assert(aoti_kernel_capture::is_int_scalar<int64_t>::value);
  static_assert(aoti_kernel_capture::is_int_scalar<int32_t>::value);
  static_assert(aoti_kernel_capture::is_int_scalar<bool>::value);

  static_assert(!aoti_kernel_capture::is_int_scalar<double>::value);
  static_assert(!aoti_kernel_capture::is_int_scalar<float>::value);
  static_assert(!aoti_kernel_capture::is_int_scalar<uint32_t>::value);
}

TEST(KernelCaptureTypeTraits, FloatScalar) {
  static_assert(aoti_kernel_capture::is_float_scalar<double>::value);
  static_assert(aoti_kernel_capture::is_float_scalar<float>::value);

  static_assert(!aoti_kernel_capture::is_float_scalar<int64_t>::value);
  static_assert(!aoti_kernel_capture::is_float_scalar<int32_t>::value);
}

// ===== CaptureArgs accumulation =====

TEST(KernelCaptureCaptureArgs, TensorAccumulation) {
  aoti_kernel_capture::CaptureArgs args;
  at::Tensor t = at::zeros({2, 3});
  AtenTensorHandle handle =
      torch::aot_inductor::tensor_pointer_to_tensor_handle(&t);
  aoti_kernel_capture::maybe_append(args, handle);

  EXPECT_EQ(args.tensor_handles.size(), 1u);
  EXPECT_EQ(args.tensor_handles[0], handle);
  EXPECT_EQ(args.tensor_positions.size(), 1u);
  EXPECT_EQ(args.tensor_positions[0], 0);
  EXPECT_EQ(args.next_pos, 1);
}

TEST(KernelCaptureCaptureArgs, IntScalarAccumulation) {
  aoti_kernel_capture::CaptureArgs args;
  int64_t i64 = 42;
  int32_t i32 = 7;
  bool b = true;
  aoti_kernel_capture::maybe_append(args, i64);
  aoti_kernel_capture::maybe_append(args, i32);
  aoti_kernel_capture::maybe_append(args, b);

  ASSERT_EQ(args.int_values.size(), 3u);
  EXPECT_EQ(args.int_values[0], 42);
  EXPECT_EQ(args.int_values[1], 7);
  EXPECT_EQ(args.int_values[2], 1); // true -> 1
  EXPECT_EQ(args.int_positions[0], 0);
  EXPECT_EQ(args.int_positions[1], 1);
  EXPECT_EQ(args.int_positions[2], 2);
}

TEST(KernelCaptureCaptureArgs, FloatScalarAccumulation) {
  aoti_kernel_capture::CaptureArgs args;
  double d = 3.14;
  float f = 2.5f;
  aoti_kernel_capture::maybe_append(args, d);
  aoti_kernel_capture::maybe_append(args, f);

  ASSERT_EQ(args.float_values.size(), 2u);
  EXPECT_DOUBLE_EQ(args.float_values[0], 3.14);
  EXPECT_DOUBLE_EQ(args.float_values[1], static_cast<double>(2.5f));
  EXPECT_EQ(args.float_positions[0], 0);
  EXPECT_EQ(args.float_positions[1], 1);
}

TEST(KernelCaptureCaptureArgs, MixedPositions) {
  aoti_kernel_capture::CaptureArgs args;
  at::Tensor t = at::ones({4});
  AtenTensorHandle handle =
      torch::aot_inductor::tensor_pointer_to_tensor_handle(&t);

  // Interleave: tensor, int, float, tensor
  aoti_kernel_capture::maybe_append(args, handle);     // pos 0
  aoti_kernel_capture::maybe_append(args, int64_t{5}); // pos 1
  aoti_kernel_capture::maybe_append(args, double{1.0});// pos 2
  aoti_kernel_capture::maybe_append(args, handle);     // pos 3

  EXPECT_EQ(args.tensor_positions[0], 0);
  EXPECT_EQ(args.int_positions[0], 1);
  EXPECT_EQ(args.float_positions[0], 2);
  EXPECT_EQ(args.tensor_positions[1], 3);
  EXPECT_EQ(args.next_pos, 4);
}

TEST(KernelCaptureCaptureArgs, InfrastructureArgsSkipped) {
  aoti_kernel_capture::CaptureArgs args;
  const char* str = "cubin_dir";
  uint64_t u64 = 99;

  // These should all be silently skipped (no-op).
  aoti_kernel_capture::maybe_append(args, str);
  aoti_kernel_capture::maybe_append(args, u64);

  EXPECT_EQ(args.tensor_handles.size(), 0u);
  EXPECT_EQ(args.int_values.size(), 0u);
  EXPECT_EQ(args.float_values.size(), 0u);
  EXPECT_EQ(args.next_pos, 0);
}

// ===== Request ID (thread safety fixes) =====

TEST(KernelCaptureRequestId, NextRequestIdIncrementing) {
  int id1 = aoti_kernel_capture::next_request_id();
  int id2 = aoti_kernel_capture::next_request_id();
  int id3 = aoti_kernel_capture::next_request_id();
  EXPECT_LT(id1, id2);
  EXPECT_LT(id2, id3);
}

TEST(KernelCaptureRequestId, BeginCaptureRequestSetsCurrentId) {
  aoti_kernel_capture::begin_capture_request();
  int id1 = aoti_kernel_capture::current_request_id();

  aoti_kernel_capture::begin_capture_request();
  int id2 = aoti_kernel_capture::current_request_id();

  // Each call to begin_capture_request advances the ID.
  EXPECT_NE(id1, id2);
  // current_request_id() should persist until next begin_capture_request().
  EXPECT_EQ(aoti_kernel_capture::current_request_id(), id2);
}

TEST(KernelCaptureRequestId, ThreadLocalRequestIdIsolation) {
  // Verify that current_request_id() is thread-local: two threads can hold
  // independent request IDs simultaneously.
  std::atomic<bool> thread1_ready{false};
  std::atomic<bool> proceed{false};

  int thread1_id = -1;
  int thread2_id = -1;
  int thread1_id_after = -1;
  int thread2_id_after = -1;

  std::thread t1([&] {
    aoti_kernel_capture::begin_capture_request();
    thread1_id = aoti_kernel_capture::current_request_id();
    thread1_ready.store(true);
    // Spin until thread 2 has also set its ID.
    while (!proceed.load()) {}
    // After thread 2 set its ID, our thread-local should be unchanged.
    thread1_id_after = aoti_kernel_capture::current_request_id();
  });

  std::thread t2([&] {
    // Wait until thread 1 has its ID.
    while (!thread1_ready.load()) {}
    aoti_kernel_capture::begin_capture_request();
    thread2_id = aoti_kernel_capture::current_request_id();
    proceed.store(true);
    thread2_id_after = aoti_kernel_capture::current_request_id();
  });

  t1.join();
  t2.join();

  // The two threads should have different request IDs.
  EXPECT_NE(thread1_id, thread2_id);
  // Thread 1's ID should not have been overwritten by thread 2.
  EXPECT_EQ(thread1_id, thread1_id_after);
  // Thread 2's ID should remain stable too.
  EXPECT_EQ(thread2_id, thread2_id_after);
}

// ===== should_capture default =====

TEST(KernelCaptureShouldCapture, DefaultFalse) {
  if (std::getenv("AOTI_KERNEL_CAPTURE_DIR") != nullptr) {
    GTEST_SKIP() << "AOTI_KERNEL_CAPTURE_DIR is set; skipping";
  }
  EXPECT_FALSE(aoti_kernel_capture::should_capture("any_kernel"));
}

// ===== Null tensor handle (fix #2) =====

TEST(KernelCaptureSaveKernel, NullTensorHandleNoCrash) {
  auto tmpdir = std::filesystem::temp_directory_path() /
      "aoti_capture_test_null_tensor";
  std::filesystem::create_directories(tmpdir);
  std::string filepath = (tmpdir / "test.pt").string();

  AtenTensorHandle handles[] = {nullptr};
  int32_t positions[] = {0};

  EXPECT_NO_THROW(aoti_torch_save_kernel_capture(
      filepath.c_str(),
      "test_kernel",
      "0",
      /*num_tensors=*/1,
      handles,
      positions,
      /*num_int_scalars=*/0,
      nullptr,
      nullptr,
      /*num_float_scalars=*/0,
      nullptr,
      nullptr));

  std::filesystem::remove_all(tmpdir);
}

// ===== End-to-end save with valid tensors (fix #3) =====

TEST(KernelCaptureSaveKernel, SaveValidTensors) {
  auto tmpdir = std::filesystem::temp_directory_path() /
      "aoti_capture_test_valid";
  std::filesystem::create_directories(tmpdir);
  std::string filepath = (tmpdir / "test.pt").string();

  at::Tensor t1 = at::ones({2, 3});
  at::Tensor t2 = at::zeros({4});
  AtenTensorHandle handles[] = {
      torch::aot_inductor::tensor_pointer_to_tensor_handle(&t1),
      torch::aot_inductor::tensor_pointer_to_tensor_handle(&t2),
  };
  int32_t tensor_positions[] = {0, 2};

  int64_t int_vals[] = {42};
  int32_t int_positions[] = {1};

  double float_vals[] = {3.14};
  int32_t float_positions[] = {3};

  aoti_torch_save_kernel_capture(
      filepath.c_str(),
      "my_kernel",
      "tag1",
      /*num_tensors=*/2,
      handles,
      tensor_positions,
      /*num_int_scalars=*/1,
      int_vals,
      int_positions,
      /*num_float_scalars=*/1,
      float_vals,
      float_positions);

  EXPECT_TRUE(std::filesystem::exists(filepath));
  EXPECT_GT(std::filesystem::file_size(filepath), 0u);

  std::filesystem::remove_all(tmpdir);
}

TEST(KernelCaptureSaveKernel, SaveZeroArgs) {
  auto tmpdir = std::filesystem::temp_directory_path() /
      "aoti_capture_test_zero";
  std::filesystem::create_directories(tmpdir);
  std::string filepath = (tmpdir / "test.pt").string();

  EXPECT_NO_THROW(aoti_torch_save_kernel_capture(
      filepath.c_str(),
      "empty_kernel",
      "0",
      /*num_tensors=*/0,
      nullptr,
      nullptr,
      /*num_int_scalars=*/0,
      nullptr,
      nullptr,
      /*num_float_scalars=*/0,
      nullptr,
      nullptr));

  EXPECT_TRUE(std::filesystem::exists(filepath));

  std::filesystem::remove_all(tmpdir);
}
