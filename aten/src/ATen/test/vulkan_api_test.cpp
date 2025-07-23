#ifdef USE_VULKAN_API

// @lint-ignore-every CLANGTIDY

#include <gtest/gtest.h>
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/vulkan/api/api.h>
#include <c10/util/irange.h>
#include <c10/util/ArrayRef.h>

// TODO: These functions should move to a common place.

namespace {

#ifdef USE_VULKAN_FP16_INFERENCE
  constexpr float kTolerance = 1e-2;
#else
  constexpr float kTolerance = 1e-5;
#endif

bool checkRtol(const at::Tensor& diff, float maxTolerance) {
  if (diff.numel() == 0) {
    return true;
  }
  return diff.abs().max().item<float>() <= maxTolerance;
}

bool checkRtol(const at::Tensor& diff, const std::vector<at::Tensor>& inputs) {
  if (diff.numel() == 0) {
    return true;
  }
  float maxValue = 0.0f;

  for (const auto& tensor : inputs) {
    maxValue = fmax(tensor.abs().max().item<float>(), maxValue);
  }

  return checkRtol(diff, kTolerance * maxValue);
}

bool almostEqual(const at::Tensor& a, const at::Tensor& b) {
  return checkRtol(a - b, {a, b});
}

bool checkHardShrink(
    const at::Tensor& ref, const at::Tensor& out, const float clamp_thresh) {
  float* ref_ptr = ref.data_ptr<float>();
  float* out_ptr = out.data_ptr<float>();
  float ref_max = ref.abs().max().item<float>();
  float out_max = out.abs().max().item<float>();
  float max_val = std::fmax(ref_max, out_max);

  float abs_clamp_thresh = std::abs(clamp_thresh);

  for (int i = 0; i < ref.numel(); ++i) {
    float ref_val = ref_ptr[i];
    float out_val = out_ptr[i];

    float abs_diff = std::abs(ref_val - out_val);

    // For values near the clamp threshold, results may be ambiguous.
    float distance_from_thresh = std::abs(std::abs(ref_val) - abs_clamp_thresh);
    if (distance_from_thresh < kTolerance * abs_clamp_thresh) {
      if (out_val != 0.0f) {
        if (abs_diff >= kTolerance * max_val) {
          return false;
        }
      }
    }
    else if (std::abs(ref_val) < std::abs(abs_clamp_thresh)) {
      if (out_val != 0.0f) {
        return false;
      }
    }
    else if (abs_diff >= kTolerance * max_val) {
      return false;
    }
  }
  return true;
}

bool checkThreshold(
    const at::Tensor& ref,
    const at::Tensor& out,
    const float clamp_thresh,
    const float value) {
  float* ref_ptr = ref.data_ptr<float>();
  float* out_ptr = out.data_ptr<float>();
  float ref_max = ref.abs().max().item<float>();
  float out_max = out.abs().max().item<float>();
  float max_val = std::fmax(ref_max, out_max);

  for (int i = 0; i < ref.numel(); ++i) {
    float ref_val = ref_ptr[i];
    float out_val = out_ptr[i];

    float abs_diff = std::abs(ref_val - out_val);
    float val_diff = std::abs(out_val - value);

    // For values near the clamp threshold, results may be ambiguous.
    float distance_from_thresh = std::abs(std::abs(ref_val) - clamp_thresh);
    if (distance_from_thresh < kTolerance * clamp_thresh) {
      if (val_diff >= kTolerance * value) {
        if (abs_diff >= kTolerance * max_val) {
          return false;
        }
      }
    }
    else if (std::abs(ref_val) < std::abs(clamp_thresh)) {
      if (val_diff >= kTolerance * value) {
        return false;
      }
    }
    else if (abs_diff >= kTolerance * max_val) {
      return false;
    }
  }
  return true;
}

void showRtol(const at::Tensor& a, const at::Tensor& b) {
  const auto diff = (a - b).abs();

  float maxValue = a.abs().max().item<float>();
  maxValue = fmax(b.abs().max().item<float>(), maxValue);

  const float maxDiff = maxValue * kTolerance;
  std::cout << "Max Diff allowed: " << maxDiff << std::endl;
  if (diff.sizes().size() == 2) {
    for (const auto y : c10::irange(diff.sizes()[0])) {
      std::cout << y << ":";
      for (const auto x : c10::irange(diff.sizes()[1])) {
        float diff_xy = diff[y][x].item<float>();
        if (diff_xy > maxDiff) {
          std::cout << std::setw(5) << x;
        }
        else {
          std::cout << std::setw(5) << " ";
        }
      }
      std::cout << std::endl;
    }
  }
}


static void gen_allpermutations(std::vector<std::vector<int64_t>>& out, std::vector<int64_t> in, unsigned i) {
  // generate all permutations of a given dims
  if (i == in.size()) {
    out.push_back(in);
  }
  else {
    for (const auto j : c10::irange(i, in.size())) {
      std::swap(in[i], in[j]);
      gen_allpermutations(out, in, i + 1);
    }
  }
}

static void gen_all_subsets(
    std::vector<std::vector<int64_t>>& out,
    int64_t n,
    unsigned i,
    std::vector<int64_t> curr) {
  // generate all subsets of set {0,...,n - 1} through backtracking
  if (i == n) {
    out.push_back(curr);
  } else {
    curr.push_back(i);
    gen_all_subsets(out, n, i + 1, curr);
    curr.pop_back();
    gen_all_subsets(out, n, i + 1, curr);
  }
}

static void slice_test(
    const std::vector<int64_t>& size,
    int64_t dim,
    std::optional<int64_t> start,
    std::optional<int64_t> end,
    int64_t step) {
  // Arrange
  const auto in_cpu = at::rand(size, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan = in_cpu.vulkan();

  // Act
  const auto out_cpu = at::slice(in_cpu, dim, start, end, step);
  const auto out_vulkan = at::slice(in_vulkan, dim, start, end, step);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

static void slice_tests(const std::unordered_map<int64_t, std::vector<int64_t>>& dim2sizes) {
  for (const auto& dim2size : dim2sizes) {
    slice_test(dim2size.second, dim2size.first, 10, 30, 1);         // i.e., 4D tensor's equivalent indexing = [:,:,:,10:30:1]
    slice_test(dim2size.second, dim2size.first, 10, 30, 7);         // i.e., 4D tensor's equivalent indexing = [:,:,:,10:30:7]
    slice_test(dim2size.second, dim2size.first, 10, 50, 2);         // i.e., 4D tensor's equivalent indexing = [:,:,:,10:50:2] with end=out of range
    slice_test(dim2size.second, dim2size.first, -60, 60, 2);        // i.e., 4D tensor's equivalent indexing = [:,:,:,-60:60:2] with start/end=out of range
    slice_test(dim2size.second, dim2size.first, -30, -10, 1);       // i.e., 4D tensor's equivalent indexing = [:,:,:,-30:-10:1] with negative start/end
    slice_test(dim2size.second, dim2size.first, 0, INT64_MAX, 1);   // i.e., 4D 's equivalent indexing = [:,:,:,0:9223372036854775807:1] with end=INT64_MAX
    slice_test(dim2size.second, dim2size.first, -10, INT64_MAX, 1); // i.e., 4D 's equivalent indexing = [:,:,:,-10:9223372036854775807:1] with negative start and end=INT64_MAX
    // This triggers a SymInt assert since [-2^63, -2^62-1] range is reserved for packed symints
    //slice_test(dim2size.second, dim2size.first, INT64_MIN, INT64_MAX, 1); // i.e., 4D 's equivalent indexing = [:,:,:,-9223372036854775808:9223372036854775807:1] with start=INT64_MIN and end=INT64_MAX
    slice_test(dim2size.second, dim2size.first, {}, {}, 1);         // i.e., 4D 's equivalent indexing = [:,:,:,::1] with empty start/end
  }
}

static void clone_test(const std::vector<int64_t>& size, std::optional<at::MemoryFormat> optional_memory_format) {
  // Arrange
  const auto in_cpu = at::rand(size, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan = in_cpu.vulkan();

  // Act
  const auto out_cpu = at::clone(in_cpu, optional_memory_format);
  const auto out_vulkan = at::clone(in_vulkan, optional_memory_format);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

template <class... Inputs>
inline std::vector<c10::IValue> makeStack(Inputs&&... inputs) {
  return {std::forward<Inputs>(inputs)...};
}

template <class... Args>
inline std::vector<c10::IValue> callOpByHandle(
    const c10::OperatorHandle& op,
    Args... args) {
  auto stack = makeStack(std::forward<Args>(args)...);
  c10::Dispatcher::singleton().callBoxed(op, &stack);
  return stack;
}

template <class... Args>
inline std::vector<c10::IValue> callOpByName(
    const char* func_name,
    const char* overload_name,
    Args... args) {
  const std::optional<c10::OperatorHandle> op_handle =
      c10::Dispatcher::singleton().findSchema({func_name, overload_name});
  assert(op_handle.has_value());
  return callOpByHandle(op_handle.value(), std::forward<Args>(args)...);
}

} // namespace

namespace {

class VulkanAPITest : public ::testing::Test {
 public:
  void SetUp() {
    if (!at::is_vulkan_available()) {
      GTEST_SKIP() << "Vulkan is not available";
    }
#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
    if (at::native::vulkan::api::context()->op_profiling_enabled()) {
      at::native::vulkan::api::context()->reset_querypool();
    }
#endif
  }

  void TearDown() {
#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
    if (at::native::vulkan::api::context()->op_profiling_enabled()) {
      try {
        at::native::vulkan::api::context()->querypool().extract_results();
        at::native::vulkan::api::context()->querypool().print_results();
      } catch (const std::exception& e) {
        std::cout << "Could not get querypool results!"
                  << " Reason: " << e.what() << std::endl;
      }
    }
#endif
  }
};

TEST_F(VulkanAPITest, zero_size_tensor) {
  auto cpu = at::rand({1, 0, 0}, at::device(at::kCPU).dtype(at::kFloat));
  auto vk = cpu.vulkan();
  auto out_vk = vk.cpu();
  ASSERT_TRUE(at::equal(out_vk, cpu));
}

TEST_F(VulkanAPITest, zero_size_tensor_numel) {
  auto vk = at::rand({18, 0, 5}, at::device(at::kVulkan).dtype(at::kFloat));
  ASSERT_TRUE(vk.numel() == 0);
}

TEST_F(VulkanAPITest, zero_dim_tensor_1) {
  auto cpu = at::rand({}, at::device(at::kCPU).dtype(at::kFloat));
  auto vv = cpu.item<float>();

  auto vk = cpu.vulkan();
  auto out_vk = vk.cpu();
  ASSERT_TRUE(almostEqual(cpu, out_vk));

  auto vk_vv = out_vk.item<float>();
  EXPECT_NEAR(vv, vk_vv, kTolerance);
}

TEST_F(VulkanAPITest, zero_dim_tensor_2) {
  float v = 3.14f;
  auto cpu = at::zeros({}, at::device(at::kCPU).dtype(at::kFloat)) + v;
  auto vk = at::zeros({}, at::device(at::kVulkan).dtype(at::kFloat)) + v;

  ASSERT_TRUE(almostEqual(cpu, vk.cpu()));
}

TEST_F(VulkanAPITest, zero_dim_tensor_3) {
  auto vk = at::zeros({}, at::device(at::kVulkan).dtype(at::kFloat));

  ASSERT_TRUE(vk.cpu().item<float>() == 0.0f);
}

TEST_F(VulkanAPITest, local_scalar_dense) {
  float v = 8.31f;
  // Force the zero-dim tensor to a non-zero constant v.
  auto vk = at::zeros({}, at::device(at::kVulkan).dtype(at::kFloat)) + v;
  c10::Scalar scalar = at::_local_scalar_dense(vk);
  EXPECT_NEAR(v, scalar.toFloat(), kTolerance);
}

TEST_F(VulkanAPITest, copy_to_texture) {
  using namespace at::native::vulkan;
  at::Tensor test_tensors[] = {
    // 4D
    at::rand({7, 17, 134, 213}, at::TensorOptions(at::kCPU).dtype(at::kFloat)),
    // 3D
    at::rand({67, 134, 213}, at::TensorOptions(at::kCPU).dtype(at::kFloat)),
    // 2D
    at::rand({229, 213}, at::TensorOptions(at::kCPU).dtype(at::kFloat)),
    // 1D
    at::rand({1902}, at::TensorOptions(at::kCPU).dtype(at::kFloat)),
  };

  for (auto in_cpu : test_tensors) {
    at::Tensor in_vk_copied = in_cpu.vulkan();
    at::Tensor out_copied = in_vk_copied.cpu();

    const auto check_copy = almostEqual(out_copied, in_cpu);

    if(!check_copy) {
      std::cout << "Copy failed on size " << in_cpu.sizes()
                << "with dtype" << in_cpu.dtype() << std::endl;
    }

    ASSERT_TRUE(check_copy);
  }
}

void test_copy_to_texture_bool(const at::IntArrayRef input_shape) {
  using namespace at::native::vulkan;
  auto cpu = at::randint(0, 2, input_shape, at::TensorOptions(at::kCPU).dtype(at::kBool));
  auto in_vulkan = cpu.vulkan();

  auto out_vulkan = in_vulkan.cpu();
  auto check = at::equal(cpu, out_vulkan.cpu());

  if (!check) {
    std::cout << "Copy texture to bool failed on input_shape " << input_shape << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, copy_to_texture_bool_mul4_hw) {
  // Uses the shader: image_to_nchw_quantized_mul4 ((H * W) % 4 == 0)
  // ch % 4 != 0,  ch < 4
  test_copy_to_texture_bool({5, 1, 2, 2});
  test_copy_to_texture_bool({17, 2, 4, 2});
  test_copy_to_texture_bool({9, 3, 3, 8});

  // ch % 4 != 0, ch > 5
  test_copy_to_texture_bool({7, 17, 4, 8});
  test_copy_to_texture_bool({8, 6, 2, 4});
  test_copy_to_texture_bool({13, 31, 4, 57});

  // 3d, 2d, 1d
  test_copy_to_texture_bool({17, 31, 4});
  test_copy_to_texture_bool({64, 16});
  test_copy_to_texture_bool({8});
}

TEST_F(VulkanAPITest, copy_to_texture_bool_mul4_chw) {
  // Uses the shader: image_to_nchw_quantized_mul4 ((H * W) % 4 == 0)
  // ch % 4 == 0
  test_copy_to_texture_bool({5, 16, 2, 16});
  test_copy_to_texture_bool({8, 8, 2, 2});
  test_copy_to_texture_bool({16, 31, 4});
}

TEST_F(VulkanAPITest, copy_to_texture_bool) {
  // Uses the shader: image_to_nchw_uint ((H * W) % 4 != 0)
  test_copy_to_texture_bool({13, 1, 3, 5});
  test_copy_to_texture_bool({13, 7, 1, 5});
  test_copy_to_texture_bool({13, 8, 2, 5});
  test_copy_to_texture_bool({13, 31, 2, 57});

  test_copy_to_texture_bool({67, 19, 7});
  test_copy_to_texture_bool({229, 213});
  test_copy_to_texture_bool({1902});
}

TEST_F(VulkanAPITest, adaptive_avg_pool2d) {
  c10::InferenceMode mode;

  const auto in_cpu = at::rand({5, 7, 47, 31}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::adaptive_avg_pool2d(in_cpu, {3, 3});
  const auto out_vulkan = at::adaptive_avg_pool2d(in_cpu.vulkan(), {3, 3});

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

void test_add(const at::IntArrayRef input_shape, const at::IntArrayRef other_shape, float alpha) {
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto other_cpu = at::rand(other_shape, at::device(at::kCPU).dtype(at::kFloat));

  const auto in_vulkan = in_cpu.vulkan();
  const auto other_vulkan = other_cpu.vulkan();

  const auto out_cpu = at::add(in_cpu, other_cpu, alpha);
  const auto out_vulkan = at::add(in_vulkan, other_vulkan, alpha);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, add_invalid_inputs) {
  // Incompatible dimensions for broadcasting for binary elementwise op
  auto in_cpu = at::rand({2, 3, 4, 5}, at::device(at::kCPU).dtype(at::kFloat));
  auto other_cpu = at::rand({2, 4, 4, 5}, at::device(at::kCPU).dtype(at::kFloat));

  EXPECT_THROW(at::add(in_cpu.vulkan(), other_cpu.vulkan(), 1.0f), ::std::exception);
}

TEST_F(VulkanAPITest, add) {
  test_add({2, 3}, {2, 3}, 1.0f);
  test_add({11, 7, 139, 109}, {11, 7, 139, 109}, 2.1f);
}

TEST_F(VulkanAPITest, add_broadcast0) {
  test_add({3, 5, 179, 221}, {3, 5, 1, 1}, 1.8f);
}

TEST_F(VulkanAPITest, add_broadcast1) {
  test_add({3, 5, 179, 221}, {3, 5, 1, 221}, 1.8f);
}

TEST_F(VulkanAPITest, add_broadcast2) {
  test_add({3, 4, 179, 221}, {4, 1, 1}, 2.5f);
}

TEST_F(VulkanAPITest, add_broadcast3) {
  test_add({3, 4, 41, 53}, {1, 1, 41, 53}, 2.5f);
}

TEST_F(VulkanAPITest, add_broadcast4) {
  test_add({3, 4, 41, 1}, {1, 41, 53}, 2.5f);
}

TEST_F(VulkanAPITest, add_broadcast5) {
  test_add({2, 1, 7, 1}, {1, 5, 1, 4}, 1.2f);
}

TEST_F(VulkanAPITest, add_broadcast6) {
  test_add({1, 15, 5, 4}, {21, 1, 5, 4}, 1.8f);
}

TEST_F(VulkanAPITest, add_zero_dim) {
 test_add({2, 6, 5, 6}, {}, 1.5f);
}

void test_add_other_cpu_int(
    const at::IntArrayRef input_shape,
    const at::IntArrayRef other_shape,
    float alpha) {
  const auto in_cpu =
      at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto other_cpu =
      (at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat)) * 100)
          .to(at::kInt);

  const auto in_vulkan = in_cpu.vulkan();

  const auto out_cpu = at::add(in_cpu, other_cpu, alpha);
  const auto out_vulkan = at::add(in_vulkan, other_cpu, alpha);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, add_other_cpu_int) {
  test_add_other_cpu_int({2, 3}, {2, 3}, 1.0f);
  test_add_other_cpu_int({11, 7, 139, 109}, {11, 7, 139, 109}, 2.1f);
}

TEST_F(VulkanAPITest, add_broadcast0_other_cpu_int) {
  test_add_other_cpu_int({3, 5, 179, 221}, {3, 5, 1, 1}, 1.8f);
}

TEST_F(VulkanAPITest, add_other_cpu_unsupported_type_should_fail) {
  const auto in_cpu = at::rand({2,2,2}, at::device(at::kCPU).dtype(at::kFloat));

  const auto other_cpu =
    at::zeros({2, 2, 2}, at::device(at::kCPU).dtype(at::kComplexFloat));

  EXPECT_THROW(at::add(in_cpu.vulkan(), other_cpu.vulkan(), 1.0f), ::std::exception);
}

TEST_F(VulkanAPITest, add_) {
  auto a_cpu = at::rand({61, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({61, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  a_cpu.add_(b_cpu, 2.1f);
  a_vulkan.add_(b_vulkan, 2.1f);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, add_broadcast0_) {
  auto a_cpu = at::rand({16, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({16, 17, 29, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  a_cpu.add_(b_cpu, 2.1f);
  a_vulkan.add_(b_vulkan, 2.1f);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, add_other_cpu_int_) {
  std::vector<int64_t> input_shape{12, 17, 29, 33};
  const auto in_cpu =
      at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto other_cpu =
      (at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat)) * 100)
          .to(at::kInt);

  const auto in_vulkan = in_cpu.vulkan();

  float alpha = -8.31f;
  in_cpu.add(other_cpu, alpha);
  in_vulkan.add(other_cpu, alpha);

  const auto check = almostEqual(in_cpu, in_vulkan.cpu());
  if (!check) {
    showRtol(in_cpu, in_vulkan.cpu());
  }
}

TEST_F(VulkanAPITest, add_broadcast1_) {
  auto a_cpu = at::rand({3, 8, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({3, 8, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  a_cpu.add_(b_cpu, 2.1f);
  a_vulkan.add_(b_vulkan, 2.1f);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(b_cpu, b_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, add_scalar) {
  const auto a_cpu = at::rand({13, 23, 59, 73}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const float b_scalar = 3.1415f;

  const auto c_cpu = at::add(a_cpu, b_scalar, 2.1f);
  const auto c_vulkan = at::add(a_vulkan, b_scalar, 2.1f);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, add_scalar_) {
  auto a_cpu = at::rand({47, 2, 23, 97}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const float b_scalar = 3.1415f;

  a_cpu.add_(b_scalar, 2.1f);
  a_vulkan.add_(b_scalar, 2.1f);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, add_scalar_wrapped) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto a_cpu = at::rand({13, 23, 59, 73}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_scalar = at::rand({1}, at::device(at::kCPU).dtype(at::kFloat));

  const auto c_cpu = at::add(a_cpu, b_scalar, 2.1f);
  const auto c_vulkan = at::add(a_vulkan, b_scalar, 2.1f);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, add_scalar_wrapped_) {
  if (!at::is_vulkan_available()) {
    return;
  }

  auto a_cpu = at::rand({47, 2, 23, 97}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_scalar = at::rand({1}, at::device(at::kCPU).dtype(at::kFloat));

  a_cpu.add_(b_scalar, 2.1f);
  a_vulkan.add_(b_scalar, 2.1f);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, add_to_scalar_wrapped) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto a = at::rand({1}, at::device(at::kCPU).dtype(at::kFloat));

  const auto b_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::add(a, b_cpu, 2.1f);
  const auto c_vulkan = at::add(a, b_vulkan, 2.1f);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, addmm) {
  constexpr float alpha = 2.1f;
  constexpr float beta = 103.24;

  const auto bias_cpu = at::rand({179, 163}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu = at::rand({179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu = at::rand({67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::addmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);

  const auto m1_vulkan = m1_cpu.vulkan();
  const auto out_vulkan = at::addmm(bias_cpu, m1_vulkan, m2_cpu, beta, alpha);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, addmm_expand) {
  constexpr float alpha = 2.1f;
  constexpr float beta = 103.24;

  const auto bias_cpu = at::rand({1000}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu = at::rand({1, 1280}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu = at::rand({1280, 1000}, at::device(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::addmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);

  const auto m1_vulkan = m1_cpu.vulkan();
  const auto out_vulkan = at::addmm(bias_cpu, m1_vulkan, m2_cpu, beta, alpha);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, addmm_expand2) {
  constexpr float alpha = 2.1f;
  constexpr float beta = 103.24;

  const auto bias_cpu = at::rand({9}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu = at::rand({17, 6}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu = at::rand({6, 9}, at::device(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::addmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);

  const auto m1_vulkan = m1_cpu.vulkan();
  const auto out_vulkan = at::addmm(bias_cpu, m1_vulkan, m2_cpu, beta, alpha);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, addmm_error_bias) {
  constexpr float alpha = 2.1f;
  constexpr float beta = 103.24;

  // mismatched bias size (should be 1-dim or {17, 9})
  const auto bias_cpu = at::rand({5, 5}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu = at::rand({17, 6}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu = at::rand({6, 9}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_vulkan = m1_cpu.vulkan();
  EXPECT_THROW(at::addmm(bias_cpu, m1_vulkan, m2_cpu, beta, alpha), ::std::exception);
}

TEST_F(VulkanAPITest, avg_pool2d) {
  const auto in_cpu = at::rand({3, 19, 43, 79}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::avg_pool2d(in_cpu, {5, 3}, {1, 2}, {2, 0}, true);
  const auto out_vulkan = at::avg_pool2d(in_cpu.vulkan(), {5, 3}, {1, 2}, {2, 0}, true);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, DISABLED_batch_norm_invalid_inputs) {
  c10::InferenceMode mode;

  // Act: Vulkan batchnorm only supports evaluation mode
  EXPECT_THROW({
    at::batch_norm(
      at::rand({3, 8, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      true,
      0.1,
      1e-05,
      false);
  }, ::std::exception);

  // Act: Vulkan batchnorm expects 4-dim input
  EXPECT_THROW({
    at::batch_norm(
      at::rand({3, 8, 5}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      false,
      0.1,
      1e-05,
      false);
  }, ::std::exception);

  // Act: Vulkan batchnorm expects 4-dim input
  EXPECT_THROW({
    at::batch_norm(
      at::rand({2, 8, 3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      false,
      0.1,
      1e-05,
      false);
  }, ::std::exception);

  // Act: Vulkan batchnorm expects channel dim to be multiple of 4
  EXPECT_THROW({
    at::batch_norm(
      at::rand({4, 7, 4, 4}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      false,
      0.1,
      1e-05,
      false);
  }, ::std::exception);

  // Act: weight tensor contains incorrect number of elements
  EXPECT_THROW({
    at::batch_norm(
      at::rand({4, 8, 4, 4}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({12}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      false,
      0.1,
      1e-05,
      false);
  }, ::std::exception);

  // Act: bias tensor contains incorrect number of elements
  EXPECT_THROW({
    at::batch_norm(
      at::rand({4, 8, 4, 4}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({12}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      false,
      0.1,
      1e-05,
      false);
  }, ::std::exception);

  // Act: running mean tensor contains incorrect number of elements
  EXPECT_THROW({
    at::batch_norm(
      at::rand({4, 8, 4, 4}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({12}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      false,
      0.1,
      1e-05,
      false);
  }, ::std::exception);

  // Act: running var tensor contains incorrect number of elements
  EXPECT_THROW({
    at::batch_norm(
      at::rand({4, 8, 4, 4}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({12}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      false,
      0.1,
      1e-05,
      false);
  }, ::std::exception);
}

TEST_F(VulkanAPITest, batch_norm_small) {
  c10::InferenceMode mode;

  const auto input_cpu = at::rand({1, 4, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto input_vulkan = input_cpu.vulkan();

  const auto weight_cpu = at::rand({4}, at::device(at::kCPU).dtype(at::kFloat));
  const auto weight_vulkan = weight_cpu.vulkan();

  const auto bias_cpu = at::rand({4}, at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_vulkan = bias_cpu.vulkan();

  const auto running_mean_cpu = at::rand({4}, at::device(at::kCPU).dtype(at::kFloat));
  const auto running_mean_vulkan = running_mean_cpu.vulkan();

  const auto running_var_cpu = at::rand({4}, at::device(at::kCPU).dtype(at::kFloat));
  const auto running_var_vulkan = running_var_cpu.vulkan();

  const auto output_cpu = at::batch_norm(input_cpu, weight_cpu, bias_cpu, running_mean_cpu, running_var_cpu, false, 0.1, 1e-05, false);
  const auto output_vulkan = at::batch_norm(input_vulkan, weight_vulkan, bias_vulkan, running_mean_vulkan, running_var_vulkan, false, 0.1, 1e-05, false);

  const auto check = almostEqual(output_cpu, output_vulkan.cpu());
  if (!check) {
    showRtol(output_cpu, output_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, batch_norm_medium) {
  c10::InferenceMode mode;

  const auto input_cpu = at::rand({3, 8, 5, 7}, at::device(at::kCPU).dtype(at::kFloat));
  const auto input_vulkan = input_cpu.vulkan();

  const auto weight_cpu = at::rand({8}, at::device(at::kCPU).dtype(at::kFloat));
  const auto weight_vulkan = weight_cpu.vulkan();

  const auto bias_cpu = at::rand({8}, at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_vulkan = bias_cpu.vulkan();

  const auto running_mean_cpu = at::rand({8}, at::device(at::kCPU).dtype(at::kFloat));
  const auto running_mean_vulkan = running_mean_cpu.vulkan();

  const auto running_var_cpu = at::rand({8}, at::device(at::kCPU).dtype(at::kFloat));
  const auto running_var_vulkan = running_var_cpu.vulkan();

  const auto output_cpu = at::batch_norm(input_cpu, weight_cpu, bias_cpu, running_mean_cpu, running_var_cpu, false, 0.1, 1e-05, false);
  const auto output_vulkan = at::batch_norm(input_vulkan, weight_vulkan, bias_vulkan, running_mean_vulkan, running_var_vulkan, false, 0.1, 1e-05, false);

  const auto check = almostEqual(output_cpu, output_vulkan.cpu());
  if (!check) {
    showRtol(output_cpu, output_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, batch_norm_large) {
  c10::InferenceMode mode;


  const auto input_cpu = at::rand({11, 52, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto input_vulkan = input_cpu.vulkan();

  const auto weight_cpu = at::rand({52}, at::device(at::kCPU).dtype(at::kFloat));
  const auto weight_vulkan = weight_cpu.vulkan();

  const auto bias_cpu = at::rand({52}, at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_vulkan = bias_cpu.vulkan();

  const auto running_mean_cpu = at::rand({52}, at::device(at::kCPU).dtype(at::kFloat));
  const auto running_mean_vulkan = running_mean_cpu.vulkan();

  const auto running_var_cpu = at::rand({52}, at::device(at::kCPU).dtype(at::kFloat));
  const auto running_var_vulkan = running_var_cpu.vulkan();

  const auto output_cpu = at::batch_norm(input_cpu, weight_cpu, bias_cpu, running_mean_cpu, running_var_cpu, false, 0.1, 1e-05, false);
  const auto output_vulkan = at::batch_norm(input_vulkan, weight_vulkan, bias_vulkan, running_mean_vulkan, running_var_vulkan, false, 0.1, 1e-05, false);

  const auto check = almostEqual(output_cpu, output_vulkan.cpu());
  if (!check) {
    showRtol(output_cpu, output_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

void test_baddbmm(
    at::Tensor bias_cpu,
    at::Tensor m1_cpu,
    at::Tensor m2_cpu,
    float beta,
    float alpha) {
  const auto out_cpu = at::baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);

  const auto m1_vulkan = m1_cpu.vulkan();
  const auto out_vulkan =
      at::baddbmm(bias_cpu, m1_vulkan, m2_cpu.vulkan(), beta, alpha);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, baddbmm) {
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  int batch = 9;
  int n = 10;
  int p = 41;
  int m = 13;

  const auto bias_cpu =
      at::rand({batch, n, m}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu =
      at::rand({batch, n, p}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({batch, p, m}, at::device(at::kCPU).dtype(at::kFloat));

  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_small) {
  constexpr float alpha = -1.0f;
  constexpr float beta = 2.0f;
  int batch = 3;
  int n = 3;
  int p = 5;
  int m = 4;

  const auto bias_cpu_0 =
      at::rand({1, n, m}, at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_cpu_1 =
      at::ones({1, n, m}, at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_cpu_2 =
      at::rand({1, n, m}, at::device(at::kCPU).dtype(at::kFloat)) * -1;
  const auto bias_cpu = at::cat({bias_cpu_0, bias_cpu_1, bias_cpu_2}, 0);

  const auto m1_cpu =
      at::rand({batch, n, p}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({batch, p, m}, at::device(at::kCPU).dtype(at::kFloat));

  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_one) {
  constexpr float alpha = 2.1f;
  constexpr float beta = 103.24;

  const auto bias_cpu =
      at::rand({1, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu =
      at::rand({1, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({1, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));

  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bais_error) {
  constexpr float alpha = 2.1f;
  constexpr float beta = 103.24;

  // mismatched dimensions of batch sizes.
  const auto bias_cpu =
      at::rand({200, 179, 163}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_vulkan = m1_cpu.vulkan();
  EXPECT_THROW(
      at::baddbmm(bias_cpu, m1_vulkan, m2_cpu, beta, alpha), ::std::exception);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_batch) {
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  const auto bias_cpu =
      at::rand({1, 179, 163}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_height) {
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  const auto bias_cpu =
      at::rand({150, 1, 163}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_width) {
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  const auto bias_cpu =
      at::rand({150, 179, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_batch_width) {
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  const auto bias_cpu =
      at::rand({1, 179, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_batch_height) {
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  const auto bias_cpu =
      at::rand({1, 1, 163}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_one) {
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  const auto bias_cpu =
      at::rand({1, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_reduce_batch) {
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  const auto bias_cpu =
      at::rand({179, 163}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_reduce_batch1) {
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  const auto bias_cpu =
      at::rand({179, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_reduce_batch2) {
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  const auto bias_cpu =
      at::rand({1, 163}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_reduce_batch_height) {
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  const auto bias_cpu = at::rand({163}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_reduce_all) {
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  const auto bias_cpu = at::rand({1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

void test_matmul(
    at::Tensor m1_cpu,
    at::Tensor m2_cpu,
    bool m2_use_vulkan = false) {
  c10::InferenceMode mode;
  const auto out_cpu = at::matmul(m1_cpu, m2_cpu);
  auto out_vk =
      at::matmul(m1_cpu.vulkan(), m2_use_vulkan ? m2_cpu.vulkan() : m2_cpu);

  const auto check = almostEqual(out_cpu, out_vk.cpu());
  if (!check) {
    showRtol(out_cpu, out_vk.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, DISABLED_matmul_3d_weight_vulkan) {
  // This will call at::bmm. Will crash for unknown reason.
  const auto m1_cpu =
      at::rand({13, 23, 45}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({13, 45, 26}, at::device(at::kCPU).dtype(at::kFloat));
  test_matmul(m1_cpu, m2_cpu, true);
}

TEST_F(VulkanAPITest, DISABLED_matmul_3d_weight_cpu) {
  // This will call at::bmm. Will crash for unknown reason.
  const auto m1_cpu =
      at::rand({13, 23, 45}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({13, 45, 26}, at::device(at::kCPU).dtype(at::kFloat));
  test_matmul(m1_cpu, m2_cpu);
}

TEST_F(VulkanAPITest, matmul_2d_weight_vulkan) {
  // This will call at::mm
  const auto m1_cpu = at::rand({7, 42}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu = at::rand({42, 9}, at::device(at::kCPU).dtype(at::kFloat));
  test_matmul(m1_cpu, m2_cpu, true);
}

TEST_F(VulkanAPITest, matmul_2d_weight_cpu) {
  // This will call at::mm
  const auto m1_cpu =
      at::rand({23, 45}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({45, 26}, at::device(at::kCPU).dtype(at::kFloat));
  test_matmul(m1_cpu, m2_cpu);
}

void test_bmm(
    at::Tensor m1_cpu,
    at::Tensor m2_cpu,
    bool m2_use_vulkan = false) {
  const auto out_cpu = m1_cpu.bmm(m2_cpu);

  const auto m1_vulkan = m1_cpu.vulkan();
  const auto out_vulkan =
      m1_vulkan.bmm(m2_use_vulkan ? m2_cpu.vulkan() : m2_cpu);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, bmm_vulkan_small) {
  const auto m1_cpu =
      at::rand({5, 2, 3}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({5, 3, 4}, at::device(at::kCPU).dtype(at::kFloat));
  test_bmm(m1_cpu, m2_cpu, true);
}

TEST_F(VulkanAPITest, bmm_vulkan_small_width) {
  const auto m1_cpu =
      at::rand({9, 32, 5}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({9, 5, 13}, at::device(at::kCPU).dtype(at::kFloat));
  test_bmm(m1_cpu, m2_cpu, true);
}

TEST_F(VulkanAPITest, bmm_vulkan_large_width) {
  const auto m1_cpu =
      at::rand({9, 7, 45}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({9, 45, 6}, at::device(at::kCPU).dtype(at::kFloat));
  test_bmm(m1_cpu, m2_cpu, true);
}

TEST_F(VulkanAPITest, bmm_cpu) {
  const auto m1_cpu =
      at::rand({13, 23, 45}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({13, 45, 26}, at::device(at::kCPU).dtype(at::kFloat));
  test_bmm(m1_cpu, m2_cpu);
}

TEST_F(VulkanAPITest, bmm_small) {
  const auto m1_cpu =
      at::rand({2, 6, 5}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({2, 5, 3}, at::device(at::kCPU).dtype(at::kFloat));
  test_bmm(m1_cpu, m2_cpu);
}

TEST_F(VulkanAPITest, bmm_one) {
  const auto m1_cpu =
      at::rand({1, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({1, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  test_bmm(m1_cpu, m2_cpu);
}

TEST_F(VulkanAPITest, bmm_error) {
  // mismatched dimensions of batch sizes.
  const auto m1_cpu =
      at::rand({100, 235, 546}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({200, 546, 267}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_vulkan = m1_cpu.vulkan();
  EXPECT_THROW(m1_vulkan.bmm(m2_cpu), ::std::exception);
}

TEST_F(VulkanAPITest, clamp) {
  const auto in_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan = in_cpu.vulkan();

  const float min_value = 0.2f;
  const float max_value = 0.8f;

  const auto out_cpu = at::clamp(in_cpu, min_value, max_value);
  const auto out_vulkan = at::clamp(in_vulkan, min_value, max_value);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, clamp_) {
  const auto cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat));
  const auto vulkan = cpu.vulkan();

  const float min_value = 0.2f;
  const float max_value = 0.8f;

  cpu.clamp_(min_value, max_value);
  vulkan.clamp_(min_value, max_value);

  const auto check = almostEqual(cpu, vulkan.cpu());
  if (!check) {
    showRtol(cpu, vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, conv1d_simple) {
  // This is a simple case using arange for input, ones for weights, and arange
  // for bias. This makes debugging easiser.
  int64_t kernel_size = 3;
  int64_t channels = 5;
  int64_t lengths = 9;

  c10::InferenceMode mode;

  const auto input_cpu = at::arange(lengths * channels, at::kFloat).reshape({1, channels, lengths});
  const auto weights_cpu = at::ones({channels, 1, kernel_size}, at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_cpu = at::arange(channels, at::kFloat);

  const auto input_vk = input_cpu.vulkan();
  const auto weights_vk = weights_cpu.vulkan();
  const auto bias_vk = bias_cpu.vulkan();

  int64_t stride = 1;
  int64_t padding = 0;
  int64_t dilation = 1;

  const auto output_cpu = at::conv1d(
      input_cpu, weights_cpu, bias_cpu, stride, padding, dilation, channels);

  const auto output_vk = at::conv1d(
      input_vk, weights_vk, bias_vk, stride, padding, dilation, channels);
  const auto output_vk_cpu = output_vk.cpu();

  const bool check = almostEqual(output_cpu, output_vk_cpu);
  if (!check) {
    showRtol(output_cpu, output_vk_cpu);
  }

  ASSERT_TRUE(check);
}

void test_conv1d(
    int64_t kernel_size,
    int64_t groups,
    int64_t lengths,
    int64_t stride = 1,
    int64_t padding = 0,
    int64_t dilation = 1,
    int64_t in_group_size = 1,
    int64_t out_group_size = 1,
    int64_t batch_size = 1) {
  c10::InferenceMode mode;

  int64_t in_channels = in_group_size * groups;
  int64_t out_channels = out_group_size * groups;

  const auto input_cpu = at::rand({batch_size, in_channels, lengths}, at::kFloat);
  const auto weights_cpu = at::rand({out_channels, in_group_size, kernel_size}, at::kFloat);
  const auto bias_cpu = at::rand({out_channels,}, at::kFloat);

  const auto input_vk = input_cpu.vulkan();
  const auto weights_vk = weights_cpu.vulkan();
  const auto bias_vk = bias_cpu.vulkan();

  const auto output_cpu = at::conv1d(
      input_cpu, weights_cpu, bias_cpu, stride, padding, dilation, groups);

  const auto output_vk = at::conv1d(
      input_vk, weights_vk, bias_vk, stride, padding, dilation, groups);
  const auto output_vk_cpu = output_vk.cpu();

  const bool check = almostEqual(output_cpu, output_vk_cpu);
  if (!check) {
    showRtol(output_cpu, output_vk_cpu);
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, conv1d) {
  test_conv1d(3, 5, 8);
  test_conv1d(9, 5, 9);
  test_conv1d(1, 12, 3);
  test_conv1d(1, 12, 1);
  test_conv1d(10, 12, 20);
  test_conv1d(3, 5, 9, 2, 0, 1);
  test_conv1d(3, 5, 9, 2, 1, 1);
  test_conv1d(3, 5, 9, 2, 1, 2);
  test_conv1d(3, 5, 9, 1, 4, 2);
  test_conv1d(6, 22, 30, 5, 5, 3);
  test_conv1d(6, 5, 30, 5, 5, 3, 3, 5);
  test_conv1d(6, 5, 30, 5, 5, 3, 4, 2);
  test_conv1d(6, 5, 30, 5, 5, 3, 4, 2, 2);
  test_conv1d(6, 5, 30, 5, 5, 3, 4, 2, 5);
  test_conv1d(6, 5, 30, 5, 5, 3, 4, 2, 9);
}



void test_conv2d_context(
    const at::IntArrayRef input_shape,
    const at::IntArrayRef weight_shape,
    const at::IntArrayRef bias_shape,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups) {
  c10::InferenceMode mode;

  at::Tensor input = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor weight = at::rand(weight_shape, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor bias = at::rand(bias_shape, at::device(at::kCPU).dtype(at::kFloat));

  // cpu
  const auto out_cpu = at::conv2d(
    input, weight, bias, stride, padding, dilation, groups);

  // vulkan
  const auto prepack_vulkan = callOpByName(
      "vulkan_prepack::create_conv2d_context",
      "",
      weight, bias, stride, padding, dilation, groups, std::nullopt, std::nullopt);

  const auto vulkan_output = callOpByName(
      "vulkan_prepack::run_conv2d_context",
      "",
      input.vulkan(), prepack_vulkan[0]);

  const auto out_vulkan = vulkan_output[0].toTensor();
  const auto out_vk_cpu = out_vulkan.cpu();

  // check
  const bool check = almostEqual(out_cpu, out_vk_cpu);
  if (!check) {
    showRtol(out_cpu, out_vk_cpu);
  }

  ASSERT_TRUE(check);
}

void test_backwards_compatible_conv2d_context(
    const at::IntArrayRef input_shape,
    const at::IntArrayRef weight_shape,
    const at::IntArrayRef bias_shape,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups) {
  c10::InferenceMode mode;

  at::Tensor input = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor weight = at::rand(weight_shape, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor bias = at::rand(bias_shape, at::device(at::kCPU).dtype(at::kFloat));

  // cpu
  const auto out_cpu = at::conv2d(
    input, weight, bias, stride, padding, dilation, groups);

  // vulkan
  const auto prepack_vulkan = callOpByName(
      "vulkan_prepack::conv2d_clamp_prepack",
      "",
      weight, bias, stride, padding, dilation, groups, std::nullopt, std::nullopt);

  const auto vulkan_output = callOpByName(
      "vulkan_prepack::conv2d_clamp_run",
      "",
      input.vulkan(), prepack_vulkan[0]);

  const auto out_vulkan = vulkan_output[0].toTensor();
  const auto out_vk_cpu = out_vulkan.cpu();

  // check
  const bool check = almostEqual(out_cpu, out_vk_cpu);
  if (!check) {
    showRtol(out_cpu, out_vk_cpu);
  }

  ASSERT_TRUE(check);
}

void test_transposed_conv2d_context(
    const at::IntArrayRef input_shape,
    const at::IntArrayRef weight_shape,
    const at::IntArrayRef bias_shape,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups) {
  c10::InferenceMode mode;

  at::Tensor input = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor weight = at::rand(weight_shape, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor bias = at::rand(bias_shape, at::device(at::kCPU).dtype(at::kFloat));

  // cpu
  const auto out_cpu = at::conv_transpose2d(
    input, weight, bias, stride, padding, output_padding, groups, dilation);

  // vulkan
  const auto prepack_vulkan = callOpByName(
      "vulkan_prepack::create_tconv2d_context",
      "",
      weight, bias, stride, padding, output_padding, dilation, groups, std::nullopt, std::nullopt);

  const auto vulkan_output = callOpByName(
      "vulkan_prepack::run_tconv2d_context",
      "",
      input.vulkan(), prepack_vulkan[0]);

  const auto out_vulkan = vulkan_output[0].toTensor();
  const auto out_vk_cpu = out_vulkan.cpu();

  // check
  const bool check = almostEqual(out_cpu, out_vk_cpu);
  if (!check) {
    showRtol(out_cpu, out_vk_cpu);
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, conv2d) {
  constexpr int64_t groups = 1;
  constexpr std::array<int64_t, 2u> stride{2, 2};
  constexpr std::array<int64_t, 2u> padding{1, 1};
  //TODO: Support conv2d with dilation != 1
  constexpr std::array<int64_t, 2u> dilation{1, 1};

  constexpr struct {
    uint32_t batches;
    uint32_t channels;
    uint32_t width;
    uint32_t height;

    std::array<int64_t, 4u> size() const {
      return {
        batches,
        channels,
        width,
        height,
      };
    }
  } input {1, 3, 8, 8};

  constexpr struct {
    uint32_t output_channels;
    uint32_t input_channels;
    uint32_t width;
    uint32_t height;

    std::array<int64_t, 4u> size() const {
      return {
        output_channels,
        input_channels,
        width,
        height,
      };
    }
  } weights {1, input.channels, 3, 3};

  const auto input_cpu = at::randn(input.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto weights_cpu = at::randn(weights.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_cpu = at::randn({weights.output_channels}, at::device(at::kCPU).dtype(at::kFloat));

  const auto output_cpu = at::conv2d(
      input_cpu,
      weights_cpu,
      bias_cpu,
      stride,
      padding,
      dilation,
      groups);

  const auto output_vulkan = at::conv2d(
      input_cpu.vulkan(),
      weights_cpu,
      bias_cpu,
      stride,
      padding,
      dilation,
      groups).cpu();

  const bool check = almostEqual(output_cpu, output_vulkan);
  if (!check) {
    showRtol(output_cpu, output_vulkan);
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, conv2d_prepack) {
  test_conv2d_context(
    {1, 3, 8, 8}, // input_shape
    {1, 3, 3, 3}, // weight_shape
    {1},          // bias_shape
    {2, 2},       // stride
    {1, 1},       // padding
    {1, 1},       // dilation
    1);           // groups
}

TEST_F(VulkanAPITest, conv2d_prepack_bc) {
  test_backwards_compatible_conv2d_context(
    {1, 3, 8, 8}, // input_shape
    {1, 3, 3, 3}, // weight_shape
    {1},          // bias_shape
    {2, 2},       // stride
    {1, 1},       // padding
    {1, 1},       // dilation
    1);           // groups
}

TEST_F(VulkanAPITest, conv2d_dw_3x3) {
  constexpr int64_t groups = 7;
  constexpr std::array<int64_t, 2u> stride{2, 3};
  constexpr std::array<int64_t, 2u> padding{0, 4};
  constexpr std::array<int64_t, 2u> dilation{3, 1};

  constexpr struct {
    uint32_t batches;
    uint32_t channels;
    uint32_t width;
    uint32_t height;

    std::array<int64_t, 4u> size() const {
      return {
          batches,
          channels,
          width,
          height,
      };
    }
  } input{1, groups, 137, 199};

  constexpr struct {
    uint32_t output_channels;
    uint32_t input_channels;
    uint32_t width;
    uint32_t height;

    std::array<int64_t, 4u> size() const {
      return {
          output_channels,
          input_channels,
          width,
          height,
      };
    }
  } weights{groups, 1, 3, 3};

  const auto input_cpu =
      at::rand(input.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto weights_cpu =
      at::rand(weights.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_cpu = at::rand(
      {weights.output_channels}, at::device(at::kCPU).dtype(at::kFloat));

  const auto output_cpu = at::conv2d(
      input_cpu, weights_cpu, bias_cpu, stride, padding, dilation, groups);

  const auto output_vulkan = at::conv2d(
      input_cpu.vulkan(),
      weights_cpu,
      bias_cpu,
      stride,
      padding,
      dilation,
      groups);

  const bool check = almostEqual(output_cpu, output_vulkan.cpu());
  if (!check) {
    showRtol(output_cpu, output_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, conv2d_dw_5x5) {
  constexpr int64_t groups = 7;
  constexpr std::array<int64_t, 2u> stride{2, 3};
  constexpr std::array<int64_t, 2u> padding{0, 4};
  constexpr std::array<int64_t, 2u> dilation{3, 1};

  constexpr struct {
    uint32_t batches;
    uint32_t channels;
    uint32_t width;
    uint32_t height;

    std::array<int64_t, 4u> size() const {
      return {
          batches,
          channels,
          width,
          height,
      };
    }
  } input{1, groups, 137, 199};

  constexpr struct {
    uint32_t output_channels;
    uint32_t input_channels;
    uint32_t width;
    uint32_t height;

    std::array<int64_t, 4u> size() const {
      return {
          output_channels,
          input_channels,
          width,
          height,
      };
    }
  } weights{groups, 1, 5, 5};

  const auto input_cpu =
      at::rand(input.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto weights_cpu =
      at::rand(weights.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_cpu = at::rand(
      {weights.output_channels}, at::device(at::kCPU).dtype(at::kFloat));

  const auto output_cpu = at::conv2d(
      input_cpu, weights_cpu, bias_cpu, stride, padding, dilation, groups);

  const auto output_vulkan = at::conv2d(
      input_cpu.vulkan(),
      weights_cpu,
      bias_cpu,
      stride,
      padding,
      dilation,
      groups);

  const bool check = almostEqual(output_cpu, output_vulkan.cpu());
  if (!check) {
    showRtol(output_cpu, output_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, conv2d_dw) {
  constexpr int64_t groups = 7;
  constexpr std::array<int64_t, 2u> stride{2, 3};
  constexpr std::array<int64_t, 2u> padding{0, 4};
  constexpr std::array<int64_t, 2u> dilation{3, 1};

  constexpr struct {
    uint32_t batches;
    uint32_t channels;
    uint32_t width;
    uint32_t height;

    std::array<int64_t, 4u> size() const {
      return {
        batches,
        channels,
        width,
        height,
      };
    }
  } input {1, groups, 137, 199};

  constexpr struct {
    uint32_t output_channels;
    uint32_t input_channels;
    uint32_t width;
    uint32_t height;

    std::array<int64_t, 4u> size() const {
      return {
        output_channels,
        input_channels,
        width,
        height,
      };
    }
  } weights {groups, 1, 17, 7};

  const auto input_cpu = at::rand(input.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto weights_cpu = at::rand(weights.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_cpu = at::rand({weights.output_channels}, at::device(at::kCPU).dtype(at::kFloat));

  const auto output_cpu = at::conv2d(
      input_cpu,
      weights_cpu,
      bias_cpu,
      stride,
      padding,
      dilation,
      groups);

  const auto output_vulkan = at::conv2d(
      input_cpu.vulkan(),
      weights_cpu,
      bias_cpu,
      stride,
      padding,
      dilation,
      groups);

  const bool check = almostEqual(output_cpu, output_vulkan.cpu());
  if (!check) {
    showRtol(output_cpu, output_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, conv2d_dw_prepack) {
  test_conv2d_context(
    {1, 7, 137, 199}, // input_shape
    {7, 1, 17, 7},    // weight_shape
    {7},              // bias_shape
    {2, 3},           // stride
    {0, 4},           // padding
    {3, 1},           // dilation
    7);               // groups
}

TEST_F(VulkanAPITest, conv2d_dw_prepack_bc) {
  test_backwards_compatible_conv2d_context(
    {1, 7, 137, 199}, // input_shape
    {7, 1, 17, 7},    // weight_shape
    {7},              // bias_shape
    {2, 3},           // stride
    {0, 4},           // padding
    {3, 1},           // dilation
    7);               // groups
}

TEST_F(VulkanAPITest, conv2d_pw) {
  constexpr int64_t groups = 1;
  constexpr std::array<int64_t, 2u> stride{1, 1};
  constexpr std::array<int64_t, 2u> padding{0, 0};
  constexpr std::array<int64_t, 2u> dilation{1, 1};

  constexpr struct {
    uint32_t batches;
    uint32_t channels;
    uint32_t width;
    uint32_t height;

    std::array<int64_t, 4u> size() const {
      return {
        batches,
        channels,
        width,
        height,
      };
    }
  } input {1, 17, 127, 397};

  constexpr struct {
    uint32_t output_channels;
    uint32_t input_channels;
    uint32_t width;
    uint32_t height;

    std::array<int64_t, 4u> size() const {
      return {
        output_channels,
        input_channels,
        width,
        height,
      };
    }
  } weights {29, input.channels, 1, 1};

  const auto input_cpu = at::randn(input.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto weights_cpu = at::randn(weights.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_cpu = at::randn({weights.output_channels}, at::device(at::kCPU).dtype(at::kFloat));

  const auto output_cpu = at::conv2d(
      input_cpu,
      weights_cpu,
      bias_cpu,
      stride,
      padding,
      dilation,
      groups);

  const auto output_vulkan = at::conv2d(
      input_cpu.vulkan(),
      weights_cpu,
      bias_cpu,
      stride,
      padding,
      dilation,
      groups);

  const bool check = almostEqual(output_cpu, output_vulkan.cpu());
  if (!check) {
    showRtol(output_cpu, output_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, conv2d_pw_prepack_medium) {
  int in_channels = 17;
  int out_channels = 29;
  int height = 27;
  int width = 39;
  test_conv2d_context(
    {1, in_channels, height, width},  // input_shape
    {out_channels, in_channels, 1, 1},     // weight_shape
    {out_channels},               // bias_shape
    {1, 1},             // stride
    {0, 0},             // padding
    {1, 1},             // dilation
    1);                 // groups
}

TEST_F(VulkanAPITest, conv2d_pw_prepack_bc_medium) {
  int in_channels = 17;
  int out_channels = 29;
  int height = 27;
  int width = 39;
  test_backwards_compatible_conv2d_context(
    {1, in_channels, height, width},  // input_shape
    {out_channels, in_channels, 1, 1},     // weight_shape
    {out_channels},               // bias_shape
    {1, 1},             // stride
    {0, 0},             // padding
    {1, 1},             // dilation
    1);                 // groups
}

// The following 2 tests failed on Meta's CI when all tests are executed.  Output
// has lots of nan. Cause unknown.
// When this test is run alone (with gtest_filter), it passes.
// The test also passes with smaller planes, see "conv2d_pw_prepack_medium".
TEST_F(VulkanAPITest, DISABLED_conv2d_pw_prepack) {
  test_conv2d_context(
    {1, 17, 127, 397},  // input_shape
    {29, 17, 1, 1},     // weight_shape
    {29},               // bias_shape
    {1, 1},             // stride
    {0, 0},             // padding
    {1, 1},             // dilation
    1);                 // groups
}

TEST_F(VulkanAPITest, DISABLED_conv2d_pw_prepack_bc) {
  test_backwards_compatible_conv2d_context(
    {1, 17, 127, 397},  // input_shape
    {29, 17, 1, 1},     // weight_shape
    {29},               // bias_shape
    {1, 1},             // stride
    {0, 0},             // padding
    {1, 1},             // dilation
    1);                 // groups
}

TEST_F(VulkanAPITest, conv2d_transposed) {
  // Arrange
  constexpr int64_t groups = 1;
  constexpr std::array<int64_t, 2u> stride{1, 2};
  constexpr std::array<int64_t, 2u> padding{1, 0};
  constexpr std::array<int64_t, 2u> output_padding{0, 1};
  //TODO: Support conv_transpose2d with dilation != 1
  constexpr std::array<int64_t, 2u> dilation{1, 1};

  constexpr struct {
    uint32_t batches;
    uint32_t channels;
    uint32_t height;
    uint32_t width;

    std::array<int64_t, 4u> size() const {
      return {
        batches,
        channels,
        height,
        width,
      };
    }
  } input {1, 55, 7, 19};

  constexpr struct {
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t height;
    uint32_t width;

    std::array<int64_t, 4u> size() const {
      return {
        input_channels,
        output_channels,
        height,
        width,
      };
    }
  } weights {input.channels, 47, 2, 3};

  const auto input_cpu = at::randn(input.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto weights_cpu = at::randn(weights.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_cpu = at::zeros({weights.output_channels}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto output_cpu = at::conv_transpose2d(
      input_cpu,
      weights_cpu,
      bias_cpu,
      stride,
      padding,
      output_padding,
      groups,
      dilation);

  const auto output_vk = at::conv_transpose2d(
      input_cpu.vulkan(),
      weights_cpu,
      bias_cpu,
      stride,
      padding,
      output_padding,
      groups,
      dilation).cpu();

  // Assert
  const bool check = almostEqual(output_cpu, output_vk);
  if (!check) {
    showRtol(output_cpu, output_vk);
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, conv2d_transposed_prepack) {
  test_transposed_conv2d_context(
    {1, 55, 7, 19}, // input_shape
    {55, 47, 2, 3}, // weight_shape
    {47},           // bias_shape
    {1, 2},         // stride
    {1, 0},         // padding
    {0, 1},         // output_padding
    {1, 1},         // dilation
    1);             // groups
}

TEST_F(VulkanAPITest, conv2d_clamp_after_div) {
  c10::InferenceMode mode;

  constexpr std::array<int64_t, 2u> stride{2, 2};
  constexpr std::array<int64_t, 2u> padding{1, 1};
  constexpr std::array<int64_t, 2u> dilation{1, 1};
  constexpr int64_t groups = 1;

  const auto input_numerator = at::rand({1, 3, 64, 64}, at::device(at::kCPU).dtype(at::kFloat));
  const auto input_denominator = at::rand({3, 1, 1}, at::device(at::kCPU).dtype(at::kFloat)) + 0.01;
  const auto input_cpu = at::div(input_numerator, input_denominator);
  const auto input_vk = at::div(input_numerator.vulkan(), input_denominator.vulkan());
  at::Tensor weight = at::rand({24, 3, 3, 3}, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor bias = at::rand({24}, at::device(at::kCPU).dtype(at::kFloat));

  // cpu
  const auto prepack_cpu = callOpByName(
      "prepacked::conv2d_clamp_prepack",
      "",
      weight, bias, stride, padding, dilation, groups, 0.0f, std::nullopt)[0];

  const auto out_cpu = callOpByName(
      "prepacked::conv2d_clamp_run",
      "",
      input_cpu, prepack_cpu)[0].toTensor();

  // vulkan
  const auto prepack_vk = callOpByName(
      "vulkan_prepack::create_conv2d_context",
      "",
      weight, bias, stride, padding, dilation, groups, 0.0f, std::nullopt)[0];

  const auto out_vk = callOpByName(
      "vulkan_prepack::run_conv2d_context",
      "",
      input_vk, prepack_vk)[0].toTensor();

  const auto out_vk_cpu = out_vk.cpu();

  // check
  const bool check = almostEqual(out_cpu, out_vk_cpu);
  if (!check) {
    showRtol(out_cpu, out_vk_cpu);
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, copy) {
  const auto cpu = at::rand({13, 17, 37, 19}, at::device(at::kCPU).dtype(at::kFloat));
  const auto vulkan = cpu.vulkan();

  const auto check = almostEqual(cpu, vulkan.cpu());
  if (!check) {
    showRtol(cpu, vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

void test_cumsum(const at::IntArrayRef input_shape, const int64_t dim) {
  const auto in_cpu = at::rand(input_shape, at::TensorOptions(at::kCPU).dtype(at::kFloat));

  const auto out_cpu = at::cumsum(in_cpu, dim);
  const auto out_vulkan = at::cumsum(in_cpu.vulkan(), dim);
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cumsum_1d) {
  test_cumsum({37}, 0);
  test_cumsum({37}, -1);
}

TEST_F(VulkanAPITest, cumsum_2d) {
  for (int64_t i = -1; i <= 1; i++) {
    test_cumsum({17, 37}, i);
  }
}

TEST_F(VulkanAPITest, cumsum_3d) {
  for (int64_t i = -2; i <= 2; i++) {
    test_cumsum({17, 37, 49}, i);
  }
}

TEST_F(VulkanAPITest, cumsum_4d) {
  for (int64_t i = -3; i <= 3; i++) {
    test_cumsum({12, 17, 37, 49}, i);
  }
}

void test_div(const at::IntArrayRef input_shape, const at::IntArrayRef other_shape) {
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto other_cpu = at::rand(other_shape, at::device(at::kCPU).dtype(at::kFloat)) + 0.01;

  const auto in_vulkan = in_cpu.vulkan();
  const auto other_vulkan = other_cpu.vulkan();

  const auto out_cpu = at::div(in_cpu, other_cpu);
  const auto out_vulkan = at::div(in_vulkan, other_vulkan);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, div) {
  test_div({11, 7, 139, 109}, {11, 7, 139, 109});
}

TEST_F(VulkanAPITest, div_broadcast0) {
  test_div({3, 5, 1, 1}, {3, 5, 179, 221});
}

TEST_F(VulkanAPITest, div_broadcast1) {
  test_div({3, 5, 179, 221}, {3, 5, 1, 221});
}

TEST_F(VulkanAPITest, div_broadcast2) {
  test_div({3, 4, 179, 221}, {4, 1, 1});
}

TEST_F(VulkanAPITest, div_broadcast3) {
  test_div({3, 4, 179, 221}, {1, 1, 179, 221});
}

TEST_F(VulkanAPITest, div_broadcast4) {
  test_div({3, 4, 41, 1}, {1, 41, 53});
}

TEST_F(VulkanAPITest, div_broadcast5) {
  test_div({2, 1, 7, 1}, {1, 5, 1, 4});
}

TEST_F(VulkanAPITest, div_broadcast6) {
  test_div({1, 15, 5, 4}, {21, 1, 5, 4});
}

TEST_F(VulkanAPITest, div_zero_dim) {
  test_div({1, 15, 5, 4}, {});
}

TEST_F(VulkanAPITest, div_) {
  auto a_cpu = at::rand({61, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({61, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat)) + 0.01;
  const auto b_vulkan = b_cpu.vulkan();

  a_cpu.div_(b_cpu);
  a_vulkan.div_(b_vulkan);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(b_cpu, b_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, div_broadcast0_) {
  auto a_cpu = at::rand({12, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({12, 17, 29, 1}, at::device(at::kCPU).dtype(at::kFloat)) + 0.01;
  const auto b_vulkan = b_cpu.vulkan();

  a_cpu.div_(b_cpu);
  a_vulkan.div_(b_vulkan);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(b_cpu, b_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, div_broadcast1_) {
  auto a_cpu = at::rand({3, 8, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({8, 1, 1}, at::device(at::kCPU).dtype(at::kFloat)) + 0.01;
  const auto b_vulkan = b_cpu.vulkan();

  a_cpu.div_(b_cpu);
  a_vulkan.div_(b_vulkan);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(b_cpu, b_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, div_scalar) {

  const auto a_cpu = at::rand({17, 213, 213, 7}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const float b_scalar = 3.1415f;

  const auto c_cpu = at::div(a_cpu, b_scalar);
  const auto c_vulkan = at::div(a_vulkan, b_scalar);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, div_scalar_) {
  auto a_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const float b_scalar = 3.1415f;

  a_cpu.div_(b_scalar);
  a_vulkan.div_(b_scalar);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, div_scalar_wrapped) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto a_cpu = at::rand({17, 213, 213, 7}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_scalar = at::rand({1}, at::device(at::kCPU).dtype(at::kFloat)) + 0.01;

  const auto c_cpu = at::div(a_cpu, b_scalar);
  const auto c_vulkan = at::div(a_vulkan, b_scalar);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, div_scalar_wrapped_) {
  if (!at::is_vulkan_available()) {
    return;
  }

  auto a_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_scalar = at::rand({1}, at::device(at::kCPU).dtype(at::kFloat)) + 0.01;

  a_cpu.div_(b_scalar);
  a_vulkan.div_(b_scalar);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, div_to_scalar_wrapped) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto a = at::rand({1}, at::device(at::kCPU).dtype(at::kFloat));

  const auto b_cpu = at::rand({2, 3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)) + 0.01;
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::div(a, b_cpu);
  const auto c_vulkan = at::div(a, b_vulkan);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, empty) {

  ASSERT_NO_THROW(at::empty({1, 17, 41, 53}, at::device(at::kVulkan).dtype(at::kFloat)));
}

void test_expand(const at::IntArrayRef input_shape, const at::IntArrayRef output_shape) {
  c10::InferenceMode mode;
  const auto cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto vulkan = cpu.vulkan();

  cpu.expand(output_shape);
  vulkan.expand(output_shape);

  const auto check = almostEqual(cpu, vulkan.cpu());
  if (!check) {
    showRtol(cpu, vulkan.cpu());
  }
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, expand_exceptions) {
  // Vulkan expand supports input dims <= 4
  auto in_cpu = at::rand({1, 2, 3, 4, 5}, at::device(at::kCPU).dtype(at::kFloat));
  EXPECT_THROW(const auto out_vulkan = in_cpu.vulkan().expand({1, 2, 3, 4}), ::std::exception);

  // Vulkan expand supports output_size <= 4
  in_cpu = at::rand({1, 2, 3, 4}, at::device(at::kCPU).dtype(at::kFloat));
  EXPECT_THROW(const auto out_vulkan = in_cpu.vulkan().expand({1, 1, 2, 3, 4}), ::std::exception);

  // Vulkan expand expects output size >= input
  in_cpu = at::rand({1, 2, 3}, at::device(at::kCPU).dtype(at::kFloat));
  EXPECT_THROW(const auto out_vulkan = in_cpu.vulkan().expand({2, 3}), ::std::exception);

  // Non-singleton dimensions must match
  in_cpu = at::rand({3, 1}, at::device(at::kCPU).dtype(at::kFloat));
  EXPECT_THROW(const auto out_vulkan = in_cpu.vulkan().expand({1, 1}), ::std::exception);

  // -1 not allowed in leading, non-existing dimension
  in_cpu = at::rand({3, 1}, at::device(at::kCPU).dtype(at::kFloat));
  EXPECT_THROW(const auto out_vulkan = in_cpu.vulkan().expand({-1, 3, 1}), ::std::exception);
}

TEST_F(VulkanAPITest, expand_1d) {
  test_expand({1}, {3});

  test_expand({1}, {9, 3});       // 1d->2d
  test_expand({1}, {8, 9, 3});    // 1d->3d
  test_expand({1}, {7, 8, 9, 3}); // 1d->4d
}

TEST_F(VulkanAPITest, expand_2d) {
  test_expand({5, 1}, {-1, 5}); // W
  test_expand({1, 5}, {5, 5});  // H

  test_expand({5, 1}, {2, -1, 5});    // 2d->3d
  test_expand({1, 5}, {2, 5, 3, -1}); // 2d->4d
}

TEST_F(VulkanAPITest, expand_3d) {
  test_expand({3, 4, 1}, {3, 4, -1}); // W
  test_expand({3, 1, 5}, {-1, 4, 5}); // H
  test_expand({1, 4, 5}, {3, -1, 5}); // C

  test_expand({5, 4, 3}, {2, -1, -1, -1}); // 3d->4d
}

TEST_F(VulkanAPITest, expand_4d) {
  test_expand({5, 4, 3, 1}, {5, 4, 3, 9}); // W
  test_expand({5, 4, 1, 2}, {5, 4, 9, 2}); // H
  test_expand({5, 1, 3, 2}, {5, 9, 3, 2}); // C
  test_expand({1, 4, 3, 2}, {9, 4, 3, 2}); // N
}

TEST_F(VulkanAPITest, expand_as) {
  // expand_as calls into expand, without negative sizes, those tests should be sufficient.
  c10::InferenceMode mode;
  const auto cpu = at::rand({1, 1, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto vulkan = cpu.vulkan();
  const auto other = at::rand({9, 11, 33, 22}, at::device(at::kCPU).dtype(at::kFloat));

  cpu.expand_as(other);
  vulkan.expand_as(other);

  const auto check = almostEqual(cpu, vulkan.cpu());
  if (!check) {
    showRtol(cpu, vulkan.cpu());
  }
  ASSERT_TRUE(check);
}

void test_flip(const at::IntArrayRef input_shape, const at::IntArrayRef dim_list) {
  c10::InferenceMode mode;
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan = in_cpu.vulkan();

  const auto out_cpu = at::flip(in_cpu, dim_list);
  const auto out_vulkan = at::flip(in_vulkan, dim_list);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
    std::cout << "test flip failed with input_shape: " << input_shape
              << " and dim_list: " << dim_list << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, flip_1d) {
  test_flip({5}, {0});
  test_flip({5}, {-1});
}

TEST_F(VulkanAPITest, flip_2d) {
  test_flip({5, 5}, {-1});
  test_flip({2, 7}, {-2});

  test_flip({5, 5}, {0, 1});
}

TEST_F(VulkanAPITest, flip_3d) {
  test_flip({5, 7, 5}, {-1});
  test_flip({2, 9, 7}, {-2});
  test_flip({9, 7, 5}, {-3});

  test_flip({10, 7, 5}, {0, 1});
  test_flip({10, 7, 5}, {0, 2});
  test_flip({10, 7, 5}, {1, 2});

  test_flip({10, 7, 5}, {2, 1, 0});
}

TEST_F(VulkanAPITest, flip_4d) {
  test_flip({2, 9, 1, 1}, {-1});
  test_flip({7, 5, 9, 3}, {-2});
  test_flip({3, 8, 5, 2}, {-3});
  test_flip({7, 9, 5, 3}, {-4});

  test_flip({10, 7, 5, 6}, {0, 1});
  test_flip({10, 7, 5, 6}, {0, 2});
  test_flip({10, 7, 5, 6}, {0, 3});
  test_flip({10, 7, 5, 6}, {1, 2});
  test_flip({10, 7, 5, 6}, {1, 3});
  test_flip({10, 7, 5, 6}, {2, 3});

  test_flip({10, 7, 5, 6}, {0, 1, 2});
  test_flip({10, 7, 5, 6}, {0, 1, 3});
  test_flip({10, 7, 5, 6}, {0, 2, 3});
  test_flip({10, 7, 5, 6}, {3, 2, 1});

  test_flip({10, 7, 5, 6}, {3, 2, 1, 0});
}

TEST_F(VulkanAPITest, gelu) {
  const auto in_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan = in_cpu.vulkan();

  auto out_cpu = at::gelu(in_cpu, "tanh");
  auto out_vulkan = at::gelu(in_vulkan, "tanh");

  auto check = almostEqual(out_cpu, out_vulkan.cpu());

  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, gelu_) {
  auto cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat));
  auto vulkan = cpu.vulkan();

  at::gelu_(cpu, "tanh");
  at::gelu_(vulkan, "tanh");

  auto check = almostEqual(cpu, vulkan.cpu());
  if (!check) {
    showRtol(cpu, vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

void test_glu(const at::IntArrayRef input_shape) {
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan = in_cpu.vulkan();

  const auto out_cpu = at::glu(in_cpu, 1);
  const auto out_vulkan = at::glu(in_vulkan, 1);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, glu_ch_200) {
  test_glu({17, 200, 302, 5});
}

TEST_F(VulkanAPITest, glu_ch_64) {
  test_glu({1, 64, 100, 8});
}

TEST_F(VulkanAPITest, glu_ch_32) {
  test_glu({1, 32, 100, 19});
}

// Re-enable once glu_channel shader is fixed
TEST_F(VulkanAPITest, DISABLED_glu_ch_10) {
  test_glu({17, 10, 57, 41});
}

// Re-enable once glu_channel shader is fixed
TEST_F(VulkanAPITest, DISABLED_glu_ch_2) {
  test_glu({1, 2, 100, 40});
}

TEST_F(VulkanAPITest, hardsigmoid) {
  const auto in_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat))*12 - 6;
  const auto in_vulkan = in_cpu.vulkan();

  const auto out_cpu = at::hardsigmoid(in_cpu);
  const auto out_vulkan = at::hardsigmoid(in_vulkan);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, hardsigmoid_) {
  auto cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat))*12 - 6;
  auto vulkan = cpu.vulkan();

  at::hardsigmoid_(cpu);
  at::hardsigmoid_(vulkan);

  const auto check = almostEqual(cpu, vulkan.cpu());
  if (!check) {
    showRtol(cpu, vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, hardshrink) {
  for (const auto lambd_value : {-4.2, -1.0, 0.42, 1.0, 4.2, 13.7}) {
    // Generate values between -10 and +10
    const auto in_cpu = (at::rand({3, 63, 79, 17}, at::device(at::kCPU).dtype(at::kFloat)) - 0.5) * 20;
    const auto in_vulkan = in_cpu.vulkan();

    const auto out_vulkan = at::hardshrink(in_vulkan, lambd_value);

    const auto check = checkHardShrink(in_cpu, out_vulkan.cpu(), lambd_value);
    ASSERT_TRUE(check);
  }
}

TEST_F(VulkanAPITest, hardshrink_) {
  for (const auto lambd_value : {0.42, 1.0, 4.2, 13.7}) {
    // Generate values between -10 and +10
    const auto in_cpu = (at::rand({3, 63, 79, 17}, at::device(at::kCPU).dtype(at::kFloat)) - 0.5) * 20;
    const auto in_vulkan = in_cpu.vulkan();

    const auto out_cpu = in_cpu.hardshrink(lambd_value);
    const auto out_vulkan = in_vulkan.hardshrink(lambd_value).cpu();

    const auto check = checkHardShrink(out_cpu, out_vulkan, lambd_value);
    ASSERT_TRUE(check);
  }
}

TEST_F(VulkanAPITest, hardtanh) {
  const auto in_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat)) * 10;
  const auto in_vulkan = in_cpu.vulkan();

  const auto out_cpu = at::hardtanh(in_cpu, 3, 7);
  const auto out_vulkan = at::hardtanh(in_vulkan, 3, 7);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, hardtanh_) {
  auto a_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat)) * 10;
  auto a_vulkan = a_cpu.vulkan();

  at::hardtanh_(a_cpu, 3, 7);
  at::hardtanh_(a_vulkan, 3, 7);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

void test_packed_layer_norm(
    const at::IntArrayRef input_shape,
    const at::IntArrayRef normalized_shape,
    const at::IntArrayRef weight_shape,
    const at::IntArrayRef bias_shape,
    const float eps) {
  c10::InferenceMode mode;

  const auto input_cpu =
      at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto input_vulkan = input_cpu.vulkan();

  const auto weight_cpu =
      at::rand(weight_shape, at::device(at::kCPU).dtype(at::kFloat));

  const auto bias_cpu =
      at::rand(bias_shape, at::device(at::kCPU).dtype(at::kFloat));

  const auto output_cpu = at::layer_norm(
      input_cpu, normalized_shape, weight_cpu, bias_cpu, eps, false);

  auto prepack = callOpByName(
      "vulkan_prepack::create_layernorm_context",
      "",
      weight_cpu, bias_cpu, eps);

  auto vulkan_output = callOpByName(
      "vulkan_prepack::run_layernorm_context",
      "",
      input_cpu.vulkan(), normalized_shape, prepack[0]);

  auto output_vulkan = vulkan_output[0].toTensor();

  const auto check = almostEqual(output_cpu, output_vulkan.cpu());
  if (!check) {
    showRtol(output_cpu, output_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, packed_layer_norm_2d) {
  test_packed_layer_norm({5, 7}, {7}, {7}, {7}, 1e-05);
  test_packed_layer_norm({5, 7}, {5, 7}, {5, 7}, {5, 7}, 1e-05);
}

TEST_F(VulkanAPITest, packed_layer_norm_3d) {
  test_packed_layer_norm({11, 5, 7}, {7}, {7}, {7}, 1e-05);
  test_packed_layer_norm({11, 5, 7}, {5, 7}, {5, 7}, {5, 7}, 1e-05);
  test_packed_layer_norm({11, 5, 7}, {11, 5, 7}, {11, 5, 7}, {11, 5, 7}, 1e-05);
}

TEST_F(VulkanAPITest, packed_layer_norm_4d) {
  test_packed_layer_norm({3, 11, 5, 7}, {7}, {7}, {7}, 1e-05);
  test_packed_layer_norm({3, 11, 5, 7}, {5, 7}, {5, 7}, {5, 7}, 1e-05);
  test_packed_layer_norm({3, 11, 5, 7}, {11, 5, 7}, {11, 5, 7}, {11, 5, 7}, 1e-05);
  test_packed_layer_norm(
      {3, 11, 5, 7}, {3, 11, 5, 7}, {3, 11, 5, 7}, {3, 11, 5, 7}, 1e-05);
}

TEST_F(VulkanAPITest, layer_norm_invalid_inputs) {
  c10::InferenceMode mode;

  // Act: incorrect normalized shape
  EXPECT_THROW({
    at::layer_norm(
      at::rand({3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      {8, 5},
      at::rand({8, 5}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8, 5}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      1e-05,
      false);
  }, ::std::exception);

  // Act: incorrect weight dimensions
  EXPECT_THROW({
    at::layer_norm(
      at::rand({3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      {3, 5, 7},
      at::rand({3, 5}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      1e-05,
      false);
  }, ::std::exception);

  // Act: incorrect bias dimensions
  EXPECT_THROW({
    at::layer_norm(
      at::rand({3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      {3, 5, 7},
      at::rand({3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      1e-05,
      false);
  }, ::std::exception);

  // Act: input has too many dimensions
  EXPECT_THROW({
    at::layer_norm(
      at::rand({1, 2, 3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      {3, 5, 7},
      at::rand({3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      1e-05,
      false);
  }, ::std::exception);
}

void test_layer_norm(
    const at::IntArrayRef input_shape,
    const at::IntArrayRef normalized_shape,
    const at::IntArrayRef weight_shape,
    const at::IntArrayRef bias_shape,
    const float eps) {
  c10::InferenceMode mode;

  const auto input_cpu =
      at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto input_vulkan = input_cpu.vulkan();

  const auto weight_cpu =
      at::rand(weight_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto weight_vulkan = weight_cpu.vulkan();

  const auto bias_cpu =
      at::rand(bias_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_vulkan = bias_cpu.vulkan();

  const auto output_cpu = at::layer_norm(
      input_cpu, normalized_shape, weight_cpu, bias_cpu, eps, false);
  const auto output_vulkan = at::layer_norm(
      input_vulkan, normalized_shape, weight_vulkan, bias_vulkan, eps, false);

  const auto check = almostEqual(output_cpu, output_vulkan.cpu());
  if (!check) {
    showRtol(output_cpu, output_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, layer_norm_2d) {
  test_layer_norm({5, 7}, {7}, {7}, {7}, 1e-05);
  test_layer_norm({5, 7}, {5, 7}, {5, 7}, {5, 7}, 1e-05);
}

TEST_F(VulkanAPITest, layer_norm_3d) {
  test_layer_norm({11, 5, 7}, {7}, {7}, {7}, 1e-05);
  test_layer_norm({11, 5, 7}, {5, 7}, {5, 7}, {5, 7}, 1e-05);
  test_layer_norm({11, 5, 7}, {11, 5, 7}, {11, 5, 7}, {11, 5, 7}, 1e-05);
}

TEST_F(VulkanAPITest, layer_norm_4d) {
  test_layer_norm({3, 11, 5, 7}, {7}, {7}, {7}, 1e-05);
  test_layer_norm({3, 11, 5, 7}, {5, 7}, {5, 7}, {5, 7}, 1e-05);
  test_layer_norm({3, 11, 5, 7}, {11, 5, 7}, {11, 5, 7}, {11, 5, 7}, 1e-05);
  test_layer_norm(
      {3, 11, 5, 7}, {3, 11, 5, 7}, {3, 11, 5, 7}, {3, 11, 5, 7}, 1e-05);
}

void test_native_layer_norm(
    const at::IntArrayRef input_shape,
    const at::IntArrayRef normalized_shape,
    const at::IntArrayRef weight_shape,
    const at::IntArrayRef bias_shape,
    const float eps) {
  c10::InferenceMode mode;

  const auto input_cpu =
      at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto input_vulkan = input_cpu.vulkan();

  const auto weight_cpu =
      at::rand(weight_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto weight_vulkan = weight_cpu.vulkan();

  const auto bias_cpu =
      at::rand(bias_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_vulkan = bias_cpu.vulkan();

  const auto output_cpu = at::native_layer_norm(
      input_cpu, normalized_shape, weight_cpu, bias_cpu, eps);
  const auto output_vulkan = at::native_layer_norm(
      input_vulkan, normalized_shape, weight_vulkan, bias_vulkan, eps);

  const auto check0 =
      almostEqual(std::get<0>(output_cpu), std::get<0>(output_vulkan).cpu());
  const auto check1 =
      almostEqual(std::get<1>(output_cpu), std::get<1>(output_vulkan).cpu());
  const auto check2 =
      almostEqual(std::get<2>(output_cpu), std::get<2>(output_vulkan).cpu());

  if (!check0) {
    std::cout
        << "the first output of native_layer_norm: layer_norm is incorrect"
        << std::endl;
    showRtol(std::get<0>(output_cpu), std::get<0>(output_vulkan).cpu());
  }
  if (!check1) {
    std::cout << "the second output of native_layer_norm: mean is incorrect"
              << std::endl;
    showRtol(std::get<1>(output_cpu), std::get<1>(output_vulkan).cpu());
  }
  if (!check2) {
    std::cout
        << "the third output of native_layer_norm: 1/sqrt(var+eps) is incorrect"
        << std::endl;
    showRtol(std::get<2>(output_cpu), std::get<2>(output_vulkan).cpu());
  }

  ASSERT_TRUE(check0 && check2 && check2);
}

TEST_F(VulkanAPITest, native_layer_norm_2d) {
  test_native_layer_norm({5, 7}, {7}, {7}, {7}, 1e-05);
  test_native_layer_norm({5, 7}, {5, 7}, {5, 7}, {5, 7}, 1e-05);
}

TEST_F(VulkanAPITest, native_layer_norm_3d) {
  test_native_layer_norm({11, 5, 7}, {7}, {7}, {7}, 1e-05);
  test_native_layer_norm({11, 5, 7}, {5, 7}, {5, 7}, {5, 7}, 1e-05);
  test_native_layer_norm({11, 5, 7}, {11, 5, 7}, {11, 5, 7}, {11, 5, 7}, 1e-05);
}

TEST_F(VulkanAPITest, native_layer_norm_4d) {
  test_native_layer_norm({3, 11, 5, 7}, {7}, {7}, {7}, 1e-05);
  test_native_layer_norm({3, 11, 5, 7}, {5, 7}, {5, 7}, {5, 7}, 1e-05);
  test_native_layer_norm(
      {3, 11, 5, 7}, {11, 5, 7}, {11, 5, 7}, {11, 5, 7}, 1e-05);
  test_native_layer_norm(
      {3, 11, 5, 7}, {3, 11, 5, 7}, {3, 11, 5, 7}, {3, 11, 5, 7}, 1e-05);
}

TEST_F(VulkanAPITest, leaky_relu) {
  for (const auto negative_slope : {0.01, 0.001, 1.0, -0.001}) {
    const auto in_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_vulkan = in_cpu.vulkan();

    const auto out_cpu = at::leaky_relu(in_cpu, negative_slope);
    const auto out_vulkan = at::leaky_relu(in_vulkan, negative_slope);

    const auto check = almostEqual(out_cpu, out_vulkan.cpu());

    if (!check) {
      showRtol(out_cpu, out_vulkan.cpu());
    }

    ASSERT_TRUE(check);
  }
}

TEST_F(VulkanAPITest, leaky_relu_) {
  for (const auto negative_slope : {0.01, 0.001, 1.0, -0.001}) {
    auto cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat));
    auto vulkan = cpu.vulkan();

    at::leaky_relu_(cpu, negative_slope);
    at::leaky_relu_(vulkan, negative_slope);

    const auto check = almostEqual(cpu, vulkan.cpu());
    if (!check) {
      showRtol(cpu, vulkan.cpu());
    }

    ASSERT_TRUE(check);
  }
}

TEST_F(VulkanAPITest, lerp) {
  const auto a_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto w_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto w_vulkan = w_cpu.vulkan();

  const auto c_cpu = at::lerp(a_cpu, b_cpu, w_cpu);
  const auto c_vulkan = at::lerp(a_vulkan, b_vulkan, w_vulkan);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, lerp_broadcast0) {
  const auto a_cpu = at::rand({3, 5, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({3, 5, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto w_cpu = at::rand({3, 5, 1, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto w_vulkan = w_cpu.vulkan();

  const auto c_cpu = at::lerp(a_cpu, b_cpu, w_cpu);
  const auto c_vulkan = at::lerp(a_vulkan, b_vulkan, w_vulkan);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, lerp_broadcast1) {
  const auto a_cpu = at::rand({3, 4, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({4, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto w_cpu = at::rand({4, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto w_vulkan = w_cpu.vulkan();

  const auto c_cpu = at::lerp(a_cpu, b_cpu, w_cpu);
  const auto c_vulkan = at::lerp(a_vulkan, b_vulkan, w_vulkan);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, lerp_) {
  auto a_cpu = at::rand({61, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({61, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto w_cpu = at::rand({61, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  const auto w_vulkan = w_cpu.vulkan();

  a_cpu.lerp_(b_cpu, w_cpu);
  a_vulkan.lerp_(b_vulkan, w_vulkan);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, lerp_broadcast0_) {
  auto a_cpu = at::rand({3, 5, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({3, 5, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto w_cpu = at::rand({3, 5, 1, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto w_vulkan = w_cpu.vulkan();

  a_cpu.lerp_(b_cpu, w_cpu);
  a_vulkan.lerp_(b_vulkan, w_vulkan);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, lerp_broadcast1_) {
  auto a_cpu = at::rand({3, 4, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({4, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto w_cpu = at::rand({4, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto w_vulkan = w_cpu.vulkan();

  a_cpu.lerp_(b_cpu, w_cpu);
  a_vulkan.lerp_(b_vulkan, w_vulkan);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, lerp_scalar) {
  const auto a_cpu = at::rand({13, 23, 59, 73}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({13, 23, 59, 73}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const float w_scalar = 3.1415f;

  const auto c_cpu = at::lerp(a_cpu, b_cpu, w_scalar);
  const auto c_vulkan = at::lerp(a_vulkan, b_vulkan, w_scalar);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, lerp_scalar_) {
  auto a_cpu = at::rand({47, 2, 23, 97}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({47, 2, 23, 97}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const float w_scalar = 3.1415f;

  a_cpu.lerp_(b_cpu, w_scalar);
  a_vulkan.lerp_(b_vulkan, w_scalar);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, hardswish) {
  const auto in_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat))*12 - 6;
  const auto in_vulkan = in_cpu.vulkan();

  const auto out_cpu = at::hardswish(in_cpu);
  const auto out_vulkan = at::hardswish(in_vulkan);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, threshold) {
  const auto in_cpu = at::rand({2, 11, 57, 23}, at::device(at::kCPU).dtype(at::kFloat))*12 - 6;
  const auto in_vulkan = in_cpu.vulkan();

  const float threshold = 2.0f;
  const float value = 5.0f;

  const auto out_cpu = at::threshold(in_cpu, threshold, value);
  const auto out_vulkan = at::threshold(in_vulkan, threshold, value);

  const auto check = checkThreshold(out_cpu, out_vulkan.cpu(), threshold, value);
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, hardswish_) {
  auto cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat))*12 - 6;
  auto vulkan = cpu.vulkan();

  at::hardswish_(cpu);
  at::hardswish_(vulkan);

  const auto check = almostEqual(cpu, vulkan.cpu());
  if (!check) {
    showRtol(cpu, vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, masked_fill_invalidinputs_exceptions) {
  // Arrange: Vulkan masked_fill expects inputs of dim <= 4
  {
    const auto in_cpu =
        at::rand({3, 5, 2, 3, 2}, at::device(at::kCPU).dtype(at::kFloat));
    const auto mask_cpu =
        at::randint(0, 2, {2, 3, 2}, at::device(at::kCPU).dtype(at::kBool));

    // Act
    EXPECT_THROW(
        {
          const auto out_vulkan =
              in_cpu.vulkan().masked_fill(mask_cpu.vulkan(), -7.0f);
          ;
        },
        ::std::exception);
  }

  // Arrange: Vulkan masked_fill expects mask of dim <= 4
  {
    const auto in_cpu =
        at::rand({2, 3, 2}, at::device(at::kCPU).dtype(at::kFloat));
    const auto mask_cpu = at::randint(
        0, 2, {3, 5, 2, 3, 2}, at::device(at::kCPU).dtype(at::kBool));

    // Act
    EXPECT_THROW(
        {
          const auto out_vulkan =
              in_cpu.vulkan().masked_fill(mask_cpu.vulkan(), -7.0f);
          ;
        },
        ::std::exception);
  }

  // Arrange: shapes of input tensor and mask tensor should be broadcastable
  {
    const auto in_cpu =
        at::rand({2, 3, 2}, at::device(at::kCPU).dtype(at::kFloat));
    const auto mask_cpu =
        at::randint(0, 2, {3, 3, 2}, at::device(at::kCPU).dtype(at::kBool));

    // Act
    EXPECT_THROW(
        {
          const auto out_vulkan =
              in_cpu.vulkan().masked_fill(mask_cpu.vulkan(), -7.0f);
          ;
        },
        ::std::exception);
  }

  // Arrange: value should be a 0-dimensional value tensor or a scalar
  {
    const auto in_cpu =
        at::rand({2, 3, 2}, at::device(at::kCPU).dtype(at::kFloat));
    const auto mask_cpu =
        at::randint(0, 2, {2, 3, 2}, at::device(at::kCPU).dtype(at::kBool));

    // Act
    EXPECT_THROW(
        {
          const auto out_vulkan =
              in_cpu.vulkan().masked_fill(mask_cpu.vulkan(), at::rand({1, 2}));
          ;
        },
        ::std::exception);
  }
}

void print_shape(const std::vector<int64_t>& shape) {
  for (const auto& num : shape) {
    std::cout << num << " ";
  }
}

void test_masked_fill_scalar(
    const at::IntArrayRef input_shape,
    const at::IntArrayRef mask_shape) {
  c10::InferenceMode mode;

  /**
   * We test masked_fill by considering all possible broadcasting cases of
   * input_shape and mask_shape. The given input_shape and mask_shape are
   * identical, e.g. both are equal to [3, 5, 2, 3]. First we truncate all
   * possible proceeding dimensions of input_shape and mask_shape respectively.
   * Denote the results as curr_input_shape and curr_mask_shape, e.g.
   * curr_input_shape = [5, 2, 3] and curr_mask_shape = [2, 3]. Then for both
   * curr_input_shape and curr_mask_shape we generate all possible subsets of
   * the indices and set the corresponding elements to 1 for each subset. For
   * example, for curr_input_shape = [5, 2, 3], a possible input_idx_subset =
   * [0, 2]. We set the 0th and 2nd elements of curr_input_shape to be 1, then
   * curr_input_shape = [1, 2, 1]. Similarly for curr_mask_shape = [2, 3], a
   * possible mask_idx_subset = [0], then the updated curr_mask_shape = [1, 3].
   * In the end, we test masked_fill with the combinations of curr_input_shape
   * and curr_mask_shape. In the example above, an output tensor of shape [1, 2,
   * 3] will be generated.
   */
  const size_t input_dim = input_shape.size();
  const size_t mask_dim = mask_shape.size();
  for (int input_shape_id = input_dim - 1; input_shape_id >= 0;
       --input_shape_id) {
    // truncate input_shape by the proceeding dimensitions
    auto curr_input_shape =
        input_shape.slice(input_shape_id, input_dim - input_shape_id);

    // generate all possible subsets of numbers between 0 and input_dim -
    // input_shape_id - 1 (inclusive)
    std::vector<std::vector<int64_t>> input_indices_subsets;
    std::vector<int64_t> curr_input_indices;
    gen_all_subsets(
        input_indices_subsets,
        input_dim - input_shape_id,
        0,
        curr_input_indices);

    for (auto input_idx_subset : input_indices_subsets) {
      // set the elements at indices of the subset of curr_input_shape to 1
      auto tmp_curr_input_shape = curr_input_shape.vec();
      for (auto input_idx : input_idx_subset) {
        tmp_curr_input_shape[input_idx] = 1;
      }

      for (int mask_shape_id = mask_dim - 1; mask_shape_id >= 0;
           --mask_shape_id) {
        // truncate amsk_shape by the proceeding dimensitions
        auto curr_mask_shape =
            mask_shape.slice(mask_shape_id, mask_dim - mask_shape_id);

        // generate all possible subsets of numbers between 0 and mask_dim -
        // mask_shape_id - 1 (inclusive)
        std::vector<std::vector<int64_t>> mask_indices_subsets;
        std::vector<int64_t> curr_mask_indices;
        gen_all_subsets(
            mask_indices_subsets,
            mask_dim - mask_shape_id,
            0,
            curr_mask_indices);

        for (auto mask_idx_subset : mask_indices_subsets) {
          // set the elements at indices of the subset of curr_mask_shape to 1
          auto tmp_curr_mask_shape = curr_mask_shape.vec();
          for (auto mask_idx : mask_idx_subset) {
            tmp_curr_mask_shape[mask_idx] = 1;
          }

          at::Tensor in_cpu = at::rand(
              tmp_curr_input_shape, at::device(at::kCPU).dtype(at::kFloat));
          at::Tensor mask_cpu = at::randint(
              0, 2, tmp_curr_mask_shape, at::device(at::kCPU).dtype(at::kBool));
          at::Tensor out_cpu = in_cpu.masked_fill(mask_cpu, -7.0f);

          at::Tensor in_vulkan = in_cpu.vulkan();
          at::Tensor mask_vulkan = mask_cpu.vulkan();
          at::Tensor out_vulkan = in_vulkan.masked_fill(mask_vulkan, -7.0f);
          const bool check = almostEqual(out_cpu, out_vulkan.cpu());

          if (!check) {
            showRtol(out_cpu, out_vulkan.cpu());
            std::cout << "Masked_fill test failed when input is of shape [";
            print_shape(tmp_curr_input_shape);
            std::cout << "], and mask of shape [";
            print_shape(tmp_curr_mask_shape);
            std::cout << "]" << std::endl;
          }

          ASSERT_TRUE(check);
        }
      }
    }
  }
}

TEST_F(VulkanAPITest, masked_fill_scalar_mult4ch) {
  test_masked_fill_scalar({3, 4, 5, 7}, {3, 4, 5, 7});
}

TEST_F(VulkanAPITest, masked_fill_scalar_nonmult4ch) {
  test_masked_fill_scalar({3, 5, 2, 3}, {3, 5, 2, 3});
}

void test_masked_fill_tensor(
    const at::IntArrayRef input_shape,
    const at::IntArrayRef mask_shape) {
  c10::InferenceMode mode;

  at::Tensor in_cpu =
      at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor mask_cpu =
      at::randint(0, 2, mask_shape, at::device(at::kCPU).dtype(at::kBool));
  at::Tensor out_cpu = in_cpu.masked_fill(mask_cpu, at::scalar_tensor(-7.0f));
  at::Tensor in_vulkan = in_cpu.vulkan();
  at::Tensor mask_vulkan = mask_cpu.vulkan();
  at::Tensor out_vulkan =
      in_vulkan.masked_fill(mask_vulkan, at::scalar_tensor(-7.0f));
  const bool check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, masked_fill_tensor_mult4ch) {
  test_masked_fill_tensor({3, 4, 2, 3}, {1, 4, 1, 1});
}

TEST_F(VulkanAPITest, masked_fill_tensor_nonmult4ch) {
  test_masked_fill_tensor({3, 5, 2, 3}, {1, 5, 1, 1});
}

TEST_F(VulkanAPITest, max_pool2d) {
  c10::InferenceMode mode;

  const auto in_cpu = at::rand({5, 13, 55, 68}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::max_pool2d(in_cpu, {3, 4}, {2, 1}, {1, 1}, {1, 1}, false);
  const auto out_vulkan = at::max_pool2d(in_cpu.vulkan(), {3, 4}, {2, 1}, {1, 1}, {1,1}, false);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}


TEST_F(VulkanAPITest, mean_invalid_inputs) {
  c10::InferenceMode mode;

  // Act: input dimension too large
  EXPECT_THROW({
    at::mean(at::rand({3, 5, 7, 8, 9}, at::device(at::kCPU).dtype(at::kFloat))
      .vulkan(), {3});
  }, ::std::exception);

  // Act: dimension out of range
  EXPECT_THROW({
    at::mean(at::rand({7, 8, 9}, at::device(at::kCPU).dtype(at::kFloat))
      .vulkan(), {3});
  }, ::std::exception);

  // Act: dimension out of range
  EXPECT_THROW({
    at::mean(at::rand({7, 8, 9}, at::device(at::kCPU).dtype(at::kFloat))
      .vulkan(), {-4});
  }, ::std::exception);

  // Act: repeated dimensions
  EXPECT_THROW({
    at::mean(at::rand({7, 8, 9}, at::device(at::kCPU).dtype(at::kFloat))
      .vulkan(), {1, 1});
  }, ::std::exception);

  // Act: repeated dimensions
  EXPECT_THROW({
    at::mean(at::rand({7, 8, 9}, at::device(at::kCPU).dtype(at::kFloat))
      .vulkan(), {1, -2});
  }, ::std::exception);
}

void test_mean_dim(const at::IntArrayRef input_shape, const at::IntArrayRef dim_list, bool keepdim=false) {
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan = in_cpu.vulkan();

  const auto out_cpu = at::mean(in_cpu, dim_list, keepdim);
  const auto out_vulkan = at::mean(in_vulkan, dim_list, keepdim);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    std::cout << "mean_dim test failed with input shape: "
              << input_shape << " and dim_list: " << dim_list << std::endl;
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, mean_dim_2d) {
  test_mean_dim({2, 3}, {-1});
  test_mean_dim({2, 7}, {-2});
}

TEST_F(VulkanAPITest, mean_dim_3d) {
  test_mean_dim({9, 7, 5}, {-1});
  test_mean_dim({5, 7, 9}, {-2});
  test_mean_dim({5, 7, 9}, {-3});

  test_mean_dim({10, 7, 5}, {0, 1});
  test_mean_dim({10, 7, 5}, {0, 2});
  test_mean_dim({10, 7, 5}, {1, 2});
  test_mean_dim({10, 7, 5}, {-1, -2});
  test_mean_dim({10, 7, 5}, {0, -2});
}

TEST_F(VulkanAPITest, mean_dim_4d) {
  test_mean_dim({7, 9, 6, 5}, {-1});
  test_mean_dim({6, 5, 7, 9}, {-2});
  test_mean_dim({6, 5, 7, 9}, {-3});
  test_mean_dim({6, 5, 7, 9}, {-4});

  test_mean_dim({10, 7, 5, 6}, {0, 1});
  test_mean_dim({10, 7, 5, 6}, {0, 2});
  test_mean_dim({10, 7, 5, 6}, {0, 3});
  test_mean_dim({10, 7, 5, 6}, {1, 2});
  test_mean_dim({10, 7, 5, 6}, {1, 3});
  test_mean_dim({10, 7, 5, 6}, {2, 3});
  test_mean_dim({10, 7, 5, 6}, {-2, -4});

  test_mean_dim({10, 7, 5, 6}, {0, 1, 2});
  test_mean_dim({10, 7, 5, 6}, {0, 1, 3});
  test_mean_dim({10, 7, 5, 6}, {0, 2, 3});
  test_mean_dim({10, 7, 5, 6}, {3, 2, 1});
  test_mean_dim({10, 7, 5, 6}, {3, -2, 1});
  test_mean_dim({10, 7, 5, 6}, {-3, -2, -1});
}

TEST_F(VulkanAPITest, mean_dim_keepdim_2d) {
  test_mean_dim({5, 7}, {-1}, true);
  test_mean_dim({5, 7}, {-2}, true);
}

TEST_F(VulkanAPITest, mean_dim_keepdim_3d) {
  test_mean_dim({9, 5, 7}, {-1}, true);
  test_mean_dim({5, 9, 7}, {-2}, true);
  test_mean_dim({7, 9, 5}, {-3}, true);

  test_mean_dim({9, 5, 7}, {0, 1}, true);
  test_mean_dim({5, 9, 7}, {0, 2}, true);
  test_mean_dim({7, 9, 5}, {1, 2}, true);
}

TEST_F(VulkanAPITest, mean_dim_keepdim_4d) {
  test_mean_dim({9, 5, 7, 11}, {-1}, true);
  test_mean_dim({5, 9, 11, 7}, {-2}, true);
  test_mean_dim({7, 11, 9, 5}, {-3}, true);
  test_mean_dim({11, 7, 9, 5}, {-4}, true);

  test_mean_dim({9, 5, 7, 11}, {0, 1}, true);
  test_mean_dim({5, 9, 11, 7}, {0, 2}, true);
  test_mean_dim({7, 11, 9, 5}, {0, 3}, true);
  test_mean_dim({11, 7, 9, 5}, {1, 2}, true);
  test_mean_dim({9, 5, 7, 11}, {1, 3}, true);
  test_mean_dim({5, 9, 11, 7}, {2, 3}, true);

  test_mean_dim({7, 11, 9, 5}, {-1, -2, -3}, true);
  test_mean_dim({11, 7, 9, 5}, {-1, -2, -4}, true);
  test_mean_dim({9, 5, 7, 11}, {-2, -3, -4}, true);
}

TEST_F(VulkanAPITest, mm) {
  const auto m1_cpu = at::rand({179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu = at::rand({67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = m1_cpu.mm(m2_cpu);

  const auto m1_vulkan = m1_cpu.vulkan();
  const auto out_vulkan = m1_vulkan.mm(m2_cpu);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, mm_m2_is_variable) {
  int n = 19;
  int p = 25;
  int m = 21;
  const auto m1_cpu = at::rand({n, p}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu = at::rand({p, m}, at::device(at::kCPU).dtype(at::kFloat));

  const auto out_cpu = m1_cpu.mm(m2_cpu);

  const auto m1_vulkan = m1_cpu.vulkan();
  const auto m2_vulkan = m2_cpu.vulkan();

  const auto out_vulkan = m1_vulkan.mm(m2_vulkan);
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, mm_m1_m2_variable) {
  int n = 19;
  int p = 25;
  int m = 21;
  const auto m1_cpu = at::rand({n, p}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu = at::rand({p, m}, at::device(at::kCPU).dtype(at::kFloat));

  const auto out_cpu = at::mm(m1_cpu, m2_cpu);

  const auto m1_vulkan = m1_cpu.vulkan();
  const auto m2_vulkan = m2_cpu.vulkan();

  const auto out_vulkan = at::mm(m1_vulkan, m2_vulkan);
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, mm_error) {
  // mismatched dimensions of m1 and m2.
  const auto m1_cpu = at::rand({179, 99}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu = at::rand({67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_vulkan = m1_cpu.vulkan();

  EXPECT_THROW(m1_vulkan.mm(m2_cpu), ::std::exception);
}

void test_mul(const at::IntArrayRef input_shape, const at::IntArrayRef other_shape) {
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto other_cpu = at::rand(other_shape, at::device(at::kCPU).dtype(at::kFloat));

  const auto in_vulkan = in_cpu.vulkan();
  const auto other_vulkan = other_cpu.vulkan();

  const auto out_cpu = at::mul(in_cpu, other_cpu);
  const auto out_vulkan = at::mul(in_vulkan, other_vulkan);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, mul) {
  test_mul({11, 7, 139, 109}, {11, 7, 139, 109});
}

TEST_F(VulkanAPITest, mul_broadcast0) {
  test_mul({3, 5, 1, 1}, {3, 5, 179, 221});
}

TEST_F(VulkanAPITest, mul_broadcast1) {
  test_mul({3, 5, 179, 221}, {3, 5, 1, 221});
}

TEST_F(VulkanAPITest, mul_broadcast2) {
  test_mul({3, 4, 179, 221}, {4, 1, 1});
}

TEST_F(VulkanAPITest, mul_broadcast3) {
  test_mul({3, 4, 179, 221}, {1, 1, 179, 221});
}

TEST_F(VulkanAPITest, mul_broadcast4) {
  test_mul({3, 4, 179, 1}, {1, 179, 221});
}

TEST_F(VulkanAPITest, mul_broadcast5) {
  test_mul({2, 1, 7, 1}, {1, 5, 1, 4});
}

TEST_F(VulkanAPITest, mul_broadcast6) {
  test_mul({1, 15, 5, 4}, {21, 1, 5, 4});
}

TEST_F(VulkanAPITest, mul_zero_dim) {
  test_mul({1, 15, 5, 4}, {});
}

TEST_F(VulkanAPITest, mul_) {
  auto a_cpu = at::rand({61, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({61, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  a_cpu.mul_(b_cpu);
  a_vulkan.mul_(b_vulkan);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(b_cpu, b_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, mul_broadcast0_) {
  auto a_cpu = at::rand({12, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({12, 17, 29, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  a_cpu.mul_(b_cpu);
  a_vulkan.mul_(b_vulkan);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(b_cpu, b_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, mul_broadcast1_) {
  auto a_cpu = at::rand({3, 8, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({8, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  a_cpu.mul_(b_cpu);
  a_vulkan.mul_(b_vulkan);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(b_cpu, b_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, mul_scalar) {
  const auto a_cpu = at::rand({17, 213, 213, 7}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const float b_scalar = 3.1415f;

  const auto c_cpu = at::mul(a_cpu, b_scalar);
  const auto c_vulkan = at::mul(a_vulkan, b_scalar);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, mul_scalar_) {
  auto a_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const float b_scalar = 3.1415f;

  a_cpu.mul_(b_scalar);
  a_vulkan.mul_(b_scalar);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, mul_scalar_wrapped) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto a_cpu = at::rand({17, 213, 213, 7}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_scalar = at::rand({1}, at::device(at::kCPU).dtype(at::kFloat));

  const auto c_cpu = at::mul(a_cpu, b_scalar);
  const auto c_vulkan = at::mul(a_vulkan, b_scalar);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, mul_scalar_wrapped_) {
  if (!at::is_vulkan_available()) {
    return;
  }

  auto a_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_scalar = at::rand({1}, at::device(at::kCPU).dtype(at::kFloat));

  a_cpu.mul_(b_scalar);
  a_vulkan.mul_(b_scalar);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, mul_to_scalar_wrapped) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto a = at::rand({1}, at::device(at::kCPU).dtype(at::kFloat));

  const auto b_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::mul(a, b_cpu);
  const auto c_vulkan = at::mul(a, b_vulkan);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

void test_pow(const at::IntArrayRef input_shape, const at::IntArrayRef other_shape) {
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto other_cpu = at::rand(other_shape, at::device(at::kCPU).dtype(at::kFloat));

  const auto in_vulkan = in_cpu.vulkan();
  const auto other_vulkan = other_cpu.vulkan();

  const auto out_cpu = at::pow(in_cpu, other_cpu);
  const auto out_vulkan = at::pow(in_vulkan, other_vulkan);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
    std::cout << "pow test failed with input shape: "
              << input_shape << " and other shape: " << other_shape << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, pow) {
  test_pow({4}, {4});
  test_pow({4, 2}, {4, 2});
  test_pow({11, 7, 9}, {11, 7, 9});
  test_pow({3, 11, 9, 7}, {3, 11, 9, 7});
}

TEST_F(VulkanAPITest, pow_broadcast) {
  // broadcast input
  test_pow({1}, {3});
  test_pow({1, 1}, {3, 2});
  test_pow({2, 1, 3}, {2, 2, 5, 3});
  test_pow({1, 1, 4}, {4, 8, 5, 4}); // mul4ch
  test_pow({3, 7, 1, 4}, {3, 7, 9, 4});

  // broadcast other
  test_pow({3}, {1});
  test_pow({3, 2}, {1, 2});
  test_pow({2, 2, 5, 3}, {2, 1, 3});
  test_pow({3, 7, 9, 4}, {3, 7, 1, 4});
  test_pow({3, 8, 2, 5}, {1, 1, 2, 5}); // mul4ch

  // broadcast both
  test_pow({2, 1, 2}, {1, 5, 1});
  test_pow({5, 1, 4}, {7, 1, 2, 1});
  test_pow({2, 1, 7, 1}, {1, 5, 1, 4});
  test_pow({1, 15, 5, 4}, {21, 1, 5, 4});
  test_pow({1, 1, 5, 5}, {8, 8, 1, 1}); // mul4ch
}

TEST_F(VulkanAPITest, pow_zero_dim) {
  test_mul({1, 15, 5, 4}, {});
}

void test_pow_(const at::IntArrayRef input_shape, const at::IntArrayRef other_shape) {
  const auto cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto other_cpu = at::rand(other_shape, at::device(at::kCPU).dtype(at::kFloat));

  const auto vulkan = cpu.vulkan();
  const auto other_vulkan = other_cpu.vulkan();

  cpu.pow_(other_cpu);
  vulkan.pow_(other_vulkan);

  const auto check = almostEqual(cpu, vulkan.cpu());
  if (!check) {
    showRtol(cpu, vulkan.cpu());
    std::cout << "pow_ test failed with input shape: "
              << input_shape << " and other shape: " << other_shape << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, pow_) {
  test_pow_({4}, {4});
  test_pow_({4, 2}, {4, 2});
  test_pow_({11, 7, 9}, {11, 7, 9});
  test_pow_({3, 11, 9, 7}, {3, 11, 9, 7});
}

TEST_F(VulkanAPITest, pow_broadcast_other_) {
  test_pow_({3}, {1});
  test_pow_({3, 2}, {1, 2});
  test_pow_({2, 2, 5, 3}, {2, 1, 3});
  test_pow_({3, 7, 9, 4}, {3, 7, 1, 4});
}

void test_pow_tensor_scalar(const at::IntArrayRef input_shape, const float exp) {
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan = in_cpu.vulkan();

  const auto out_cpu = at::pow(in_cpu, exp);
  const auto out_vulkan = at::pow(in_vulkan, exp);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
    std::cout << "pow_tensor_scalar test failed with input shape: "
              << input_shape << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, pow_tensor_scalar) {
  test_pow_tensor_scalar({4}, 2.5);             // 1d
  test_pow_tensor_scalar({4, 2}, -1);           // 2d
  test_pow_tensor_scalar({11, 7, 9}, 7.7);      // 3d
  test_pow_tensor_scalar({3, 11, 9, 7}, -0.03); // 4d
}

void test_pow_tensor_scalar_(const at::IntArrayRef input_shape, const float exp) {
  // Make sure inputs are not 0, cannot compare
  const auto cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto vulkan = cpu.vulkan();

  cpu.pow_(exp);
  vulkan.pow_(exp);

  const auto check = almostEqual(cpu, vulkan.cpu());
  if (!check) {
    showRtol(cpu, vulkan.cpu());
    std::cout << "pow_scalar_ test failed with input shape: "
              << input_shape << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, pow_tensor_scalar_) {
  test_pow_tensor_scalar_({4}, 2.5);             // 1d
  test_pow_tensor_scalar_({4, 2}, -1);           // 2d
  test_pow_tensor_scalar_({11, 7, 9}, 7.7);      // 3d
  test_pow_tensor_scalar_({3, 11, 9, 7}, -0.03); // 4d
}

void test_pow_scalar_tensor(const float base, const at::IntArrayRef other) {
  const auto other_cpu = at::rand(other, at::device(at::kCPU).dtype(at::kFloat));
  const auto other_vulkan = other_cpu.vulkan();

  const auto out_cpu = at::pow(base, other_cpu);
  const auto out_vulkan = at::pow(base, other_vulkan);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
    std::cout << "pow_scalar_tensor test failed with other shape: "
              << other << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, pow_scalar_tensor) {
  test_pow_scalar_tensor(2.5, {4});             // 1d
  test_pow_scalar_tensor(2, {4, 2});            // 2d
  test_pow_scalar_tensor(7.7, {11, 7, 9});      // 3d
  test_pow_scalar_tensor(3, {3, 11, 9, 7});     // 4d
}

void test_floor_divide_scalar(const at::IntArrayRef input_shape, float input_scale, float other) {
  c10::InferenceMode mode;

  auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  in_cpu = at::mul(in_cpu, input_scale);

  auto in_vulkan = in_cpu.vulkan();
  auto out_vk = at::floor_divide(in_vulkan, other);
  auto out_cpu = at::floor_divide(in_cpu, other);

  // max tolerance is 1.0 due to floor.
  // may consider adding extra check on number of violation. it should be rare.
  const auto check = checkRtol(out_cpu - out_vk.cpu(), 1.0f);
  if (!check) {
    std::cout << "floor_divide test failed with "
              << "scale: " << input_scale
              << " other: " << other
              << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, floor_divide_scalar) {
  test_floor_divide_scalar({3, 3, 12, 12}, 100.0, 10.0);
  test_floor_divide_scalar({12, 12}, 10.0, 3.4);
  test_floor_divide_scalar({4, 5, 12, 12}, 100.0, 10.0);
  test_floor_divide_scalar({3, 3, 12, 12}, 0.3, 0.08);
}

TEST_F(VulkanAPITest, floor_divide_scalar_error) {
  c10::InferenceMode mode;

  auto in_cpu = at::rand({2, 3, 4}, at::device(at::kCPU).dtype(at::kFloat));
  auto in_vulkan = in_cpu.vulkan();
  EXPECT_THROW(at::floor_divide(in_vulkan, 0.0f), ::std::exception);
}

void test_floor_divide_scalar_inplace(const at::IntArrayRef input_shape, float input_scale, float other) {
  c10::InferenceMode mode;

  auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  in_cpu = at::mul(in_cpu, input_scale);
  auto in_vk = in_cpu.vulkan();

  in_cpu.floor_divide_(other);
  in_vk.floor_divide_(other);

  // max tolerance is 1.0 due to floor.
  // may consider adding extra check on number of violation. it should be rare.
  const auto check = checkRtol(in_cpu - in_vk.cpu(), 1.0f);
  if (!check) {
    std::cout << "floor_divide test failed with "
              << "scale: " << input_scale
              << " other: " << other
              << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, floor_divide_scalar_inplace_error) {
  c10::InferenceMode mode;

  auto in_cpu = at::rand({2, 3, 4}, at::device(at::kCPU).dtype(at::kFloat));
  auto in_vulkan = in_cpu.vulkan();
  EXPECT_THROW(in_vulkan.floor_divide(0.0f), ::std::exception);
}

TEST_F(VulkanAPITest, floor_divide_scalar_inplace) {
  test_floor_divide_scalar_inplace({3, 3, 12, 12}, 100.0, 10.0);
  test_floor_divide_scalar_inplace({12, 12}, 10.0, 3.4);
  test_floor_divide_scalar_inplace({4, 5, 12, 12}, 100.0, 10.0);
  test_floor_divide_scalar_inplace({3, 3, 12, 12}, 0.3, 0.08);
}

TEST_F(VulkanAPITest, floor_divide_zero_dim_tensor) {
  c10::InferenceMode mode;

  std::vector<int64_t> input_shape{5, 3, 4, 5};
  float input_scale = 100.0;

  auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  in_cpu = at::mul(in_cpu, input_scale);
  auto in_vk = in_cpu.vulkan();

  auto other_cpu = at::zeros({}, at::device(at::kCPU).dtype(at::kFloat)) + 10.0f;
  auto other_vk = other_cpu.vulkan();

  auto out_cpu = at::floor_divide(in_cpu, other_cpu);
  auto out_vk = at::floor_divide(in_vk, other_vk);

  // max tolerance is 1.0 due to floor.
  // may consider adding extra check on number of violation. it should be rare.
  const auto check = checkRtol(out_cpu - out_vk.cpu(), 1.0f);
  if (!check) {
    std::cout << "floor_divide test failed with "
              << "scale: " << input_scale
              << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, floor_divide_tensor) {
  c10::InferenceMode mode;

  std::vector<int64_t> input_shape{6, 3, 5, 5};
  float input_scale = 10.0;

  auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  in_cpu = at::mul(in_cpu, input_scale);
  // "other" is at least 0.5 to avoid rounding error causes by very small
  // values.
  auto other_cpu =
      at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat)) + 0.5;

  auto in_vk = in_cpu.vulkan();
  auto other_vk = other_cpu.vulkan();

  auto out_cpu = at::floor_divide(in_cpu, other_cpu);
  auto out_vk = at::floor_divide(in_vk, other_vk);

  // max tolerance is 1.0 due to floor.
  // may consider adding extra check on number of violation. it should be rare.
  const auto check = checkRtol(out_cpu - out_vk.cpu(), 1.0f);
  if (!check) {
    std::cout << "floor_divide test failed with "
              << "scale: " << input_scale << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, floor_divide_tensor_inplace) {
  c10::InferenceMode mode;

  std::vector<int64_t> input_shape{5, 3, 5, 5};
  float input_scale = 10.0;

  auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  in_cpu = at::mul(in_cpu, input_scale);
  // "other" is at least 0.5 to avoid rounding error causes by very small
  // values.
  auto other_cpu =
      at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat)) + 0.5;

  auto in_vk = in_cpu.vulkan();
  auto other_vk = other_cpu.vulkan();

  in_cpu.floor_divide_(other_cpu);
  in_vk.floor_divide_(other_vk);

  // max tolerance is 1.0 due to floor.
  // may consider adding extra check on number of violation. it should be rare.
  const auto check = checkRtol(in_cpu - in_vk.cpu(), 1.0f);
  if (!check) {
    std::cout << "floor_divide test failed with "
              << "scale: " << input_scale << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, relu) {
  const auto in_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan = in_cpu.vulkan();

  const auto out_cpu = at::relu(in_cpu);
  const auto out_vulkan = at::relu(in_vulkan);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());

  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, relu_) {
  auto a_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  at::relu_(a_cpu);
  at::relu_(a_vulkan);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());

  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, reflection_pad2d) {
  const auto a_cpu = at::rand({2, 3, 47, 63}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto out_cpu = at::reflection_pad2d(a_cpu, {9,8,5,12});
  const auto out_vulkan = at::reflection_pad2d(a_vulkan, {9,8,5,12}).cpu();

  const auto check = almostEqual(out_cpu, out_vulkan);
  if (!check) {
    showRtol(out_cpu, out_vulkan);
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, repeat_invalid_inputs_outputs_exceptions) {
  // Arrange: Vulkan repeat only supports input of dims <= 4
  {
    const auto in_cpu =
        at::rand({3, 9, 11, 7, 3}, at::device(at::kCPU).dtype(at::kFloat));
    const at::IntArrayRef repeats = {5, 7, 3, 9, 2};

    // Act
    EXPECT_THROW(
        { const auto out_vulkan = in_cpu.vulkan().repeat(repeats); },
        ::std::exception);
  }

  // Arrange: Number of dimensions of repeat dims can not be smaller than
  // number of dimensions of tensor
  {
    const auto in_cpu =
        at::rand({3, 5, 11, 13}, at::device(at::kCPU).dtype(at::kFloat));
    const at::IntArrayRef repeats = {5, 7};

    // Act
    EXPECT_THROW(
        { const auto out_vulkan = in_cpu.vulkan().repeat(repeats); },
        ::std::exception);
  }

  // Arrange: Vulkan repeat only supports output of dims <= 4
  {
    const auto in_cpu =
        at::rand({3, 9, 11, 7}, at::device(at::kCPU).dtype(at::kFloat));
    const at::IntArrayRef repeats = {5, 7, 3, 9, 2};

    // Act
    EXPECT_THROW(
        { const auto out_vulkan = in_cpu.vulkan().repeat(repeats); },
        ::std::exception);
  }
}

void test_repeat(
    const at::IntArrayRef input_shape,
    const at::IntArrayRef repeats) {
  c10::InferenceMode mode;

  at::Tensor in_cpu;
  at::Tensor out_cpu;
  at::Tensor in_vulkan;
  at::Tensor out_vulkan;
  at::IntArrayRef repeat;
  bool check = true;
  for (int idx_input = 1; (unsigned)idx_input < input_shape.size() + 1; ++idx_input) {
    for (int idx_repeat = idx_input; (unsigned)idx_repeat < repeats.size() + 1;
          ++idx_repeat) {
      in_cpu = at::rand(
          input_shape.slice(0, idx_input),
          at::device(at::kCPU).dtype(at::kFloat));
      repeat = repeats.slice(0, idx_repeat);
      out_cpu = in_cpu.repeat(repeats);
      in_vulkan = in_cpu.vulkan();
      out_vulkan = in_vulkan.repeat(repeats);
      bool local_check = almostEqual(out_cpu, out_vulkan.cpu());
      if (!local_check) {
        check = false;
        std::cout << "Repeat test failed when input is of shape "
                  << input_shape.slice(0, idx_input) << " and repeat of "
                  << repeat << std::endl;
        showRtol(out_cpu, out_vulkan.cpu());
      }
    }
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, repeat) {
  test_repeat({13, 5, 13, 7}, {7, 2, 3, 5});
}

TEST_F(VulkanAPITest, replication_pad2d) {
  const auto a_cpu = at::rand({2, 3, 47, 63}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  constexpr std::array<int64_t, 4u> padding_params{9, 8, 5, 12};

  const auto out_cpu = at::replication_pad2d(a_cpu, padding_params);
  const auto out_vulkan = at::replication_pad2d(a_vulkan, padding_params).cpu();

  const auto check = almostEqual(out_cpu, out_vulkan);
  if (!check) {
    showRtol(out_cpu, out_vulkan);
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, reshape) {
  c10::InferenceMode mode;

  const auto in_cpu = at::rand({7, 11, 8, 9}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan = in_cpu.vulkan();

  const std::array<int64_t, 2> shape{7 * 8, 11 * 9};

  const auto out_cpu = at::reshape(in_cpu, shape);
  const auto out_vulkan = at::reshape(in_vulkan, shape);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, reshape_) {
  c10::InferenceMode mode;

  const auto cpu = at::rand({9, 4, 12, 6}, at::device(at::kCPU).dtype(at::kFloat));
  const auto vulkan = cpu.vulkan();

  const std::array<int64_t, 3> shape{9, 4 * 6, 12};

  cpu.reshape(shape);
  vulkan.reshape(shape);

  const auto check = almostEqual(cpu, vulkan.cpu());
  if (!check) {
    showRtol(cpu, vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

void test_select(const at::IntArrayRef input_shape, int64_t dim, int64_t index) {
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::select(in_cpu, dim, index);

  const auto in_vulkan = in_cpu.vulkan();
  const auto out_vulkan = at::select(in_vulkan, dim, index);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, select_3d_depth_small) {
  test_select({1, 1, 1}, 0, 0);
}

TEST_F(VulkanAPITest, select_3d_depth_medium) {
  test_select({3, 2, 5}, 0, 2);
}

TEST_F(VulkanAPITest, select_3d_depth_large) {
  test_select({100, 1, 144}, 0, 50);
}

TEST_F(VulkanAPITest, select_3d_height_small) {
  test_select({1, 1, 1}, 1, 0);
}

TEST_F(VulkanAPITest, select_3d_height_medium) {
  test_select({3, 5, 2}, 1, 2);
}

TEST_F(VulkanAPITest, select_3d_height_medium1) {
  test_select({16, 16, 5}, 1, 6);
}

TEST_F(VulkanAPITest, select_3d_height_medium2) {
  test_select({17, 17, 5}, 1, 6);
}

TEST_F(VulkanAPITest, select_3d_height_large) {
  test_select({100, 144, 5}, 1, 50);
}

TEST_F(VulkanAPITest, select_3d_width_small) {
  test_select({1, 1, 1}, 2, 0);
}

TEST_F(VulkanAPITest, select_3d_width_medium) {
  test_select({3, 5, 3}, 2, 2);
}

TEST_F(VulkanAPITest, select_3d_width_medium2) {
  test_select({17, 17, 8}, 2, 6);
}

TEST_F(VulkanAPITest, select_3d_width_large) {
  test_select({100, 3, 144}, 2, 50);
}

TEST_F(VulkanAPITest, select_4d_batch_small) {
  test_select({1, 1, 1, 1}, 0, 0);
}

TEST_F(VulkanAPITest, select_4d_batch_medium) {
  test_select({3, 2, 5, 4}, 0, 1);
}

TEST_F(VulkanAPITest, select_4d_batch_large) {
  test_select({30, 8, 12, 17}, 0, 27);
}

TEST_F(VulkanAPITest, select_4d_depth_small) {
  test_select({1, 1, 1, 1}, 1, 0);
}

TEST_F(VulkanAPITest, select_4d_depth_medium) {
  test_select({7, 5, 2, 4}, 1, 4);
}

TEST_F(VulkanAPITest, select_4d_depth_large) {
  test_select({5, 30, 12, 30}, 1, 23);
}

TEST_F(VulkanAPITest, select_4d_height_small) {
  test_select({1, 1, 1, 1}, 2, 0);
}

TEST_F(VulkanAPITest, select_4d_height_medium) {
  test_select({3, 5, 4, 2}, 2, 3);
}

TEST_F(VulkanAPITest, select_4d_height_large) {
  test_select({5, 8, 50, 50}, 2, 41);
}

TEST_F(VulkanAPITest, select_4d_width_small) {
  test_select({1, 1, 1, 1}, 3, 0);
}

TEST_F(VulkanAPITest, select_4d_width_medium) {
  test_select({3, 5, 4, 2}, 3, 1);
}

TEST_F(VulkanAPITest, select_4d_width_large) {
  test_select({5, 8, 50, 50}, 3, 33);
}

TEST_F(VulkanAPITest, sigmoid) {
  const auto in_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan = in_cpu.vulkan();

  const auto out_cpu = at::sigmoid(in_cpu);
  const auto out_vulkan = at::sigmoid(in_vulkan);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, sigmoid_) {
  auto cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat));
  auto vulkan = cpu.vulkan();

  at::sigmoid_(cpu);
  at::sigmoid_(vulkan);

  const auto check = almostEqual(cpu, vulkan.cpu());
  if (!check) {
    showRtol(cpu, vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, DISABLED_log_softmax_underflow_exception) {
  // We apply softmax and log in a sequence to the tesnor [20, 0].
  // The output of softmax on CPU is [1.0000e+00, 2.0612e-09]; while
  // the output on Vulkan is [1, 0] since 2.0612e-09 is smaller than
  // the smallest represetable positive 5.96e−8. We expect to see nan
  // or -inf when applying log.
  float data[] = {20, 0};
  const auto in_cpu = at::from_blob(data, {2}, at::kFloat);
  const auto in_vulkan = in_cpu.vulkan();

  const auto softmax_out_cpu = at::softmax(in_cpu, 0);
  const auto softmax_out_vulkan = at::softmax(in_vulkan, 0);

  const auto log_out_cpu = at::log(softmax_out_cpu);
  const auto log_out_vulkan = at::log(softmax_out_vulkan);

  auto has_nan = log_out_vulkan.cpu().isnan().any().item().to<bool>();
  auto has_inf = log_out_vulkan.cpu().isinf().any().item().to<bool>();

  // We expect the output of log containing nan or inf.
  const auto check = has_nan || has_inf;
  if (!check) {
    std::cout << "expect log_out_vulkan contains nan or inf, but got" << std::endl;
    std::cout << log_out_vulkan.cpu() << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, log_softmax_underflow) {
  // The minimum strictly positive (subnormal) value of float16 on Vulkan is 2−24 ≈ 5.96 × 10^−8.
  // https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Exponent_encoding
  // then smallest_representable_log = log(5.96 × 10^−8) = -16.64.
  // The implementation of `log_softmax` adds 6e-8 to the output of softmax before applying `log`
  // to deal with underflow, so there won't be nan or -inf as shown in the
  // `log_softmax_underflow_exception` test above
  float smallest_representable_log = -16.64f;
  float data[] = {20, 0};
  const auto in_cpu = at::from_blob(data, {2}, at::kFloat);
  const auto in_vulkan = in_cpu.vulkan();

  const auto log_softmax_cpu = at::log_softmax(in_cpu, 0);
  const auto log_softmax_vulkan = at::log_softmax(in_vulkan, 0);

  const auto check = checkRtol(log_softmax_cpu - log_softmax_vulkan.cpu(), -smallest_representable_log);
  if (!check) {
    showRtol(log_softmax_cpu, log_softmax_vulkan.cpu());
  }
  ASSERT_TRUE(check);
}

void test_softmax(const at::IntArrayRef shape, bool log_softmax = false) {
  at::Tensor in_cpu =
      at::rand(shape, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const at::Tensor in_vulkan = in_cpu.vulkan();

  // Cast to signed to test negative index for dim
  int64_t size = static_cast<int64_t>(shape.size());

  // Test on all dim
  for (auto dim = -size; dim < size; dim++) {
    const at::Tensor out_cpu =
        log_softmax ? at::log_softmax(in_cpu, dim) : at::softmax(in_cpu, dim);

    const at::Tensor out_vulkan = log_softmax ? at::log_softmax(in_vulkan, dim)
                                              : at::softmax(in_vulkan, dim);
    const bool check = almostEqual(out_cpu, out_vulkan.cpu());

    if (!check) {
      std::cout << "Softmax test failed on axis " << dim << "for tensor dims {";
      for (uint32_t place = 0; place < shape.size() - 1; place++) {
        std::cout << shape[place] << " ";
      }
      std::cout << shape.back() << "}" << std::endl;
      showRtol(out_cpu, out_vulkan.cpu());
    }
    ASSERT_TRUE(check);
  }
}

TEST_F(VulkanAPITest, softmax) {
  c10::InferenceMode mode;
  std::vector<std::vector<int64_t>> test_in_dims = {
      {1, 3, 4, 2},
      {4, 8, 5, 7},
      {9, 11, 12, 12},
  };
  bool log_softmax = false;
  for (const std::vector<int64_t>& dim_vec : test_in_dims) {
    for (uint32_t trunc = 0; trunc < dim_vec.size(); trunc++) {
      const std::vector<int64_t> trunc_dim_vec =
          std::vector<int64_t>(dim_vec.begin(), dim_vec.end() - trunc);
      test_softmax(trunc_dim_vec, log_softmax);
    }
  }
}

TEST_F(VulkanAPITest, DISABLED_log_softmax) {
  c10::InferenceMode mode;
  std::vector<std::vector<int64_t>> test_in_dims = {
      {1, 3, 4, 2},
      {4, 8, 5, 7},
      {9, 11, 12, 12},
  };
  bool log_softmax = true;
  for (const std::vector<int64_t>& dim_vec : test_in_dims) {
    for (uint32_t trunc = 0; trunc < dim_vec.size(); trunc++) {
      const std::vector<int64_t> trunc_dim_vec =
          std::vector<int64_t>(dim_vec.begin(), dim_vec.end() - trunc);
      test_softmax(trunc_dim_vec, log_softmax);
    }
  }
}

TEST_F(VulkanAPITest, abs) {
  const auto in_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat)) * 30;
  const auto in_vulkan = in_cpu.vulkan();

  const auto out_cpu = at::abs(in_cpu);
  const auto out_vulkan = at::abs(in_vulkan);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, abs_) {
  auto cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat)) * 30;
  auto vulkan = cpu.vulkan();

  at::abs_(cpu);
  at::abs_(vulkan);

  const auto check = almostEqual(cpu, vulkan.cpu());
  if (!check) {
    showRtol(cpu, vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, tanh) {
  const auto in_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat)) * 30;
  const auto in_vulkan = in_cpu.vulkan();

  const auto out_cpu = at::tanh(in_cpu);
  const auto out_vulkan = at::tanh(in_vulkan);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, tanh_) {
  auto cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat)) * 30;
  auto vulkan = cpu.vulkan();

  at::tanh_(cpu);
  at::tanh_(vulkan);

  const auto check = almostEqual(cpu, vulkan.cpu());
  if (!check) {
    showRtol(cpu, vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

void test_sub(const at::IntArrayRef input_shape, const at::IntArrayRef other_shape, float alpha) {
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto other_cpu = at::rand(other_shape, at::device(at::kCPU).dtype(at::kFloat));

  const auto in_vulkan = in_cpu.vulkan();
  const auto other_vulkan = other_cpu.vulkan();

  const auto out_cpu = at::sub(in_cpu, other_cpu, alpha);
  const auto out_vulkan = at::sub(in_vulkan, other_vulkan, alpha);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, sub) {
  test_sub({11, 7, 139, 109}, {11, 7, 139, 109}, 2.1f);
}

TEST_F(VulkanAPITest, sub_broadcast0) {
  test_sub({3, 5, 179, 221}, {3, 5, 1, 1}, 1.8f);
}

TEST_F(VulkanAPITest, sub_broadcast1) {
  test_sub({3, 5, 179, 221}, {3, 5, 1, 221}, 1.8f);
}

TEST_F(VulkanAPITest, sub_broadcast2) {
  test_sub({3, 4, 179, 221}, {4, 1, 1}, 2.5f);
}

TEST_F(VulkanAPITest, sub_broadcast3) {
  test_sub({3, 4, 179, 221}, {1, 1, 179, 221}, 2.5f);
}

TEST_F(VulkanAPITest, sub_broadcast4) {
  test_sub({3, 4, 179, 1}, {1, 179, 221}, 2.5f);
}

TEST_F(VulkanAPITest, sub_broadcast5) {
  test_sub({2, 1, 7, 1}, {1, 5, 1, 4}, 1.2f);
}

TEST_F(VulkanAPITest, sub_broadcast6) {
  test_sub({1, 15, 5, 4}, {21, 1, 5, 4}, 1.8f);
}

TEST_F(VulkanAPITest, sub_zero_dim) {
  test_sub({1, 15, 5, 4}, {}, 1.8f);
}

TEST_F(VulkanAPITest, sub_) {
  auto a_cpu = at::rand({61, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({61, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  a_cpu.sub_(b_cpu, 2.1f);
  a_vulkan.sub_(b_vulkan, 2.1f);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(b_cpu, b_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, sub_broadcast0_) {
  auto a_cpu = at::rand({16, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({16, 17, 29, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  a_cpu.sub_(b_cpu, 2.1f);
  a_vulkan.sub_(b_vulkan, 2.1f);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(b_cpu, b_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, sub_broadcast1_) {
  auto a_cpu = at::rand({3, 8, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({3, 8, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  a_cpu.sub_(b_cpu, 2.1f);
  a_vulkan.sub_(b_vulkan, 2.1f);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(b_cpu, b_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, sub_scalar) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto a_cpu = at::rand({13, 23, 59, 73}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const float b_scalar = 3.1415f;

  const auto c_cpu = at::sub(a_cpu, b_scalar, 2.1f);
  const auto c_vulkan = at::sub(a_vulkan, b_scalar, 2.1f);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, sub_scalar_) {
  if (!at::is_vulkan_available()) {
    return;
  }

  auto a_cpu = at::rand({47, 2, 23, 97}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const float b_scalar = 3.1415f;

  a_cpu.sub_(b_scalar, 2.1f);
  a_vulkan.sub_(b_scalar, 2.1f);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, sub_scalar_wrapped) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto a_cpu = at::rand({13, 23, 59, 73}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_scalar = at::rand({1}, at::device(at::kCPU).dtype(at::kFloat));

  const auto c_cpu = at::sub(a_cpu, b_scalar, 2.1f);
  const auto c_vulkan = at::sub(a_vulkan, b_scalar, 2.1f);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, sub_scalar_wrapped_) {
  if (!at::is_vulkan_available()) {
    return;
  }

  auto a_cpu = at::rand({47, 2, 23, 97}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_scalar = at::rand({1}, at::device(at::kCPU).dtype(at::kFloat));

  a_cpu.sub_(b_scalar, 2.1f);
  a_vulkan.sub_(b_scalar, 2.1f);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, sub_to_scalar_wrapped) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto a = at::rand({1}, at::device(at::kCPU).dtype(at::kFloat));

  const auto b_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::sub(a, b_cpu, 2.1f);
  const auto c_vulkan = at::sub(a, b_vulkan, 2.1f);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, sum_invalid_inputs) {
  c10::InferenceMode mode;

  // Act: input dimension too large
  EXPECT_THROW({
    at::sum(at::rand({3, 5, 7, 8, 9}, at::device(at::kCPU).dtype(at::kFloat))
      .vulkan(), {3});
  }, ::std::exception);

  // Act: dimension out of range
  EXPECT_THROW({
    at::sum(at::rand({7, 8, 9}, at::device(at::kCPU).dtype(at::kFloat))
      .vulkan(), {3});
  }, ::std::exception);

  // Act: dimension out of range
  EXPECT_THROW({
    at::sum(at::rand({7, 8, 9}, at::device(at::kCPU).dtype(at::kFloat))
      .vulkan(), {-4});
  }, ::std::exception);

  // Act: repeated dimensions
  EXPECT_THROW({
    at::sum(at::rand({7, 8, 9}, at::device(at::kCPU).dtype(at::kFloat))
      .vulkan(), {1, 1});
  }, ::std::exception);

  // Act: repeated dimensions
  EXPECT_THROW({
    at::sum(at::rand({7, 8, 9}, at::device(at::kCPU).dtype(at::kFloat))
      .vulkan(), {1, -2});
  }, ::std::exception);
}

void test_sum_dim(const at::IntArrayRef input_shape, const at::IntArrayRef dim_list, bool keepdim=false) {
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan = in_cpu.vulkan();

  const auto out_cpu = at::sum(in_cpu, dim_list, keepdim);
  const auto out_vulkan = at::sum(in_vulkan, dim_list, keepdim);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    std::cout << "sum_dim test failed with input shape: "
              << input_shape << " and dim_list: " << dim_list << std::endl;
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, sum_dim_1d) {
  test_sum_dim({7}, {-1});
  test_sum_dim({3}, {0});
}

TEST_F(VulkanAPITest, sum_dim_2d) {
  test_sum_dim({2, 3}, {-1});
  test_sum_dim({2, 7}, {-2});
  test_sum_dim({2, 7}, {-1, -2});
}

TEST_F(VulkanAPITest, sum_dim_3d) {
  test_sum_dim({9, 7, 5}, {-1});
  test_sum_dim({5, 7, 9}, {-2});
  test_sum_dim({5, 7, 9}, {-3});

  test_sum_dim({10, 7, 5}, {0, 1});
  test_sum_dim({10, 7, 5}, {0, 2});
  test_sum_dim({10, 7, 5}, {1, 2});

  test_sum_dim({10, 7, 5}, {-1, -2});
  test_sum_dim({10, 7, 5}, {-1, -3});
  test_sum_dim({10, 7, 5}, {-2, -3});

  test_sum_dim({10, 7, 5}, {0, 1, 2});
  test_sum_dim({10, 7, 5}, {-1, -2, -3});
}

TEST_F(VulkanAPITest, sum_dim_4d) {
  test_sum_dim({7, 9, 6, 5}, {-1});
  test_sum_dim({6, 5, 7, 9}, {-2});
  test_sum_dim({6, 5, 7, 9}, {-3});
  test_sum_dim({6, 5, 7, 9}, {-4});

  test_sum_dim({10, 7, 5, 6}, {0, 1});
  test_sum_dim({10, 7, 5, 6}, {0, 2});
  test_sum_dim({10, 7, 5, 6}, {0, 3});
  test_sum_dim({10, 7, 5, 6}, {1, 2});
  test_sum_dim({10, 7, 5, 6}, {1, 3});
  test_sum_dim({10, 7, 5, 6}, {2, 3});
  test_sum_dim({10, 7, 5, 6}, {-2, -4});

  test_sum_dim({10, 7, 5, 6}, {0, 1, 2});
  test_sum_dim({10, 7, 5, 6}, {0, 1, 3});
  test_sum_dim({10, 7, 5, 6}, {0, 2, 3});
  test_sum_dim({10, 7, 5, 6}, {3, 2, 1});
  test_sum_dim({10, 7, 5, 6}, {3, -2, 1});
  test_sum_dim({10, 7, 5, 6}, {-3, -2, -1});

  test_sum_dim({10, 7, 5, 6}, {-1, -2, -3});
  test_sum_dim({10, 7, 5, 6}, {-1, -2, -4});
  test_sum_dim({10, 7, 5, 6}, {-1, -3, -4});
  test_sum_dim({10, 7, 5, 6}, {-2, -3, -4});

  test_sum_dim({10, 7, 5, 6}, {-1, -2, -3, -4});
}

TEST_F(VulkanAPITest, sum_dim_keepdim_1d) {
  test_sum_dim({5}, {-1}, true);
  test_sum_dim({3}, {-1}, true);
}

TEST_F(VulkanAPITest, sum_dim_keepdim_2d) {
  test_sum_dim({5, 7}, {-1}, true);
  test_sum_dim({5, 7}, {-2}, true);
}

TEST_F(VulkanAPITest, sum_dim_keepdim_3d) {
  test_sum_dim({9, 5, 7}, {-1}, true);
  test_sum_dim({5, 9, 7}, {-2}, true);
  test_sum_dim({7, 9, 5}, {-3}, true);

  test_sum_dim({9, 5, 7}, {0, 1}, true);
  test_sum_dim({5, 9, 7}, {0, 2}, true);
  test_sum_dim({7, 9, 5}, {1, 2}, true);

  test_sum_dim({7, 9, 5}, {0, 1, 2}, true);
}

TEST_F(VulkanAPITest, sum_dim_keepdim_4d) {
  test_sum_dim({9, 5, 7, 11}, {-1}, true);
  test_sum_dim({5, 9, 11, 7}, {-2}, true);
  test_sum_dim({7, 11, 9, 5}, {-3}, true);
  test_sum_dim({11, 7, 9, 5}, {-4}, true);

  test_sum_dim({9, 5, 7, 11}, {0, 1}, true);
  test_sum_dim({5, 9, 11, 7}, {0, 2}, true);
  test_sum_dim({7, 11, 9, 5}, {0, 3}, true);
  test_sum_dim({11, 7, 9, 5}, {1, 2}, true);
  test_sum_dim({9, 5, 7, 11}, {1, 3}, true);
  test_sum_dim({5, 9, 11, 7}, {2, 3}, true);

  test_sum_dim({7, 11, 9, 5}, {-1, -2, -3}, true);
  test_sum_dim({11, 7, 9, 5}, {-1, -2, -4}, true);
  test_sum_dim({9, 5, 7, 11}, {-2, -3, -4}, true);

  test_sum_dim({9, 5, 7, 11}, {-1, -2, -3, -4}, true);
}

void test_sum(const at::IntArrayRef input_shape) {
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan = in_cpu.vulkan();

  const auto out_cpu = at::sum(in_cpu);
  const auto out_vulkan = at::sum(in_vulkan);

  ASSERT_TRUE(out_vulkan.dim() == 0);
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    std::cout << "sum test failed with input shape: "
              << input_shape << std::endl;
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, sum_test) {
  test_sum({6});
  test_sum({5, 6});
  test_sum({0, 3, 1});
  test_sum({5, 0, 1});
  test_sum({5, 3, 0});
  test_sum({3, 3, 1});
  test_sum({7, 6, 6});
  test_sum({7, 8, 5, 6});
}


void test_uniform(at::Tensor a_vulkan, const float a_min, const float a_max) {
  auto a_cpu = a_vulkan.cpu();
  ASSERT_TRUE(a_cpu.max().item<float>() <= a_max);
  ASSERT_TRUE(a_cpu.min().item<float>() >= a_min);

  // Verify range, also perform a loose check with on histogram distribution.
  float b_min = 0.0f;
  float b_max = 10.0f;

  auto b_vulkan =
      at::rand({80, 7, 12, 10}, at::device(at::kCPU).dtype(at::kFloat))
          .vulkan();
  b_vulkan.uniform_(b_min, b_max);
  auto b_cpu = b_vulkan.cpu();

  int bins = 10;
  auto b_hist_tuple = at::histogram(b_cpu, bins);

  int64_t expected_per_bin = b_vulkan.numel() / bins;
  auto b_hist = std::get<0>(b_hist_tuple);

  // Very relaxed definition of uniform. Pass if all bins are within 5% of
  // expected.
  ASSERT_TRUE(
      (b_hist - expected_per_bin).abs().max().item<float>() <=
      (expected_per_bin * 0.05));
}

TEST_F(VulkanAPITest, uniform) {
  float a_min = -8.2f;
  float a_max = -1.4f;
  auto a_vulkan =
      at::rand({8, 7, 12, 10}, at::device(at::kCPU).dtype(at::kFloat)).vulkan();
  a_vulkan.uniform_(a_min, a_max);
  test_uniform(a_vulkan, a_min, a_max);
}

TEST_F(VulkanAPITest, rand_like) {
  float a_min = 0.0f;
  float a_max = 1.0f;
  auto a_vulkan =
      at::zeros({8, 7, 12, 10}, at::device(at::kCPU).dtype(at::kFloat)).vulkan();
  const auto out_vulkan = at::rand_like(a_vulkan);
  // verify that the input are still all zeros (not in-place)
  ASSERT_TRUE(at::mean(a_vulkan.cpu()).item<float>() == 0.0);
  test_uniform(out_vulkan, a_min, a_max);
}

void test_normal(at::Tensor out_vulkan, const float mean, const float std) {
  // Verify the distribution is normal. The difference between given mean vs generated mean should be within 5% of standard deviation, and the same for standard deviation itself.
  ASSERT_TRUE(std::abs(at::mean(out_vulkan.cpu()).item<float>() - mean) < std::abs(std) * 0.05);
  ASSERT_TRUE(std::abs(at::std(out_vulkan.cpu()).item<float>() - std) < std::abs(std) * 0.05);
}

TEST_F(VulkanAPITest, normal_) {
  float a_mean = -10.0;
  float a_std = 2.0;

  auto a_vulkan =
      at::zeros({3, 4, 5, 6}, at::device(at::kCPU).dtype(at::kFloat)).vulkan();
  a_vulkan.normal_(a_mean, a_std);

  test_normal(a_vulkan, a_mean, a_std);
}

TEST_F(VulkanAPITest, normal_large) {
  float a_mean = 1.0;
  float a_std = 0.01;

  auto a_vulkan =
      at::zeros({30, 40, 50, 60}, at::device(at::kCPU).dtype(at::kFloat)).vulkan();
  a_vulkan.normal_(a_mean, a_std);

  test_normal(a_vulkan, a_mean, a_std);
}

TEST_F(VulkanAPITest, normal_error) {
  float a_mean = 1.0;
  float a_std = -1;

  auto a_vulkan =
      at::zeros({30, 40, 50, 60}, at::device(at::kCPU).dtype(at::kFloat)).vulkan();
  EXPECT_THROW(a_vulkan.normal_(a_mean, a_std), ::std::exception);
}

TEST_F(VulkanAPITest, randn_like) {
  float a_mean = 0.0;
  float a_std = 1.0;

  auto a_vulkan =
      at::zeros({8, 7, 6, 5}, at::device(at::kCPU).dtype(at::kFloat)).vulkan();
  const auto out_vulkan = at::randn_like(a_vulkan);
  // verify that the input are still all zeros (not in-place)
  ASSERT_TRUE(at::mean(a_vulkan.cpu()).item<float>() == 0.0);
  test_normal(out_vulkan, a_mean, a_std);
}

TEST_F(VulkanAPITest, randn_like_large) {
  float a_mean = 0.0;
  float a_std = 1.0;

  auto a_vulkan =
      at::zeros({80, 70, 60, 50}, at::device(at::kCPU).dtype(at::kFloat)).vulkan();
  const auto out_vulkan = at::randn_like(a_vulkan);

  test_normal(out_vulkan, a_mean, a_std);
}

void test_t(const at::IntArrayRef input_shape) {
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::t(in_cpu);

  const auto in_vulkan = in_cpu.vulkan();
  const auto out_vulkan = at::t(in_vulkan);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, transpose_t_1d) {
  test_t({7});
}

TEST_F(VulkanAPITest, transpose_t_2d_small) {
  test_t({1, 1});
}

TEST_F(VulkanAPITest, transpose_t_2d_medium) {
  test_t({7, 5});
}

TEST_F(VulkanAPITest, transpose_t_2d_large) {
  test_t({53, 117});
}

void test_transpose(const at::IntArrayRef input_shape, int64_t index0, int64_t index1) {
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::transpose(in_cpu, index0, index1);

  const auto in_vulkan = in_cpu.vulkan();
  const auto out_vulkan = at::transpose(in_vulkan, index0, index1);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, transpose_2d_height_and_width_small) {
  test_transpose({1, 1}, 0, 1);
}

TEST_F(VulkanAPITest, transpose_2d_height_and_width_medium) {
  test_transpose({7, 5}, 0, 1);
}

TEST_F(VulkanAPITest, transpose_2d_height_and_width_large) {
  test_transpose({53, 117}, 0, 1);
}

TEST_F(VulkanAPITest, transpose_2d_height_and_height_large) {
  test_transpose({53, 117}, 0, 0);
}

TEST_F(VulkanAPITest, transpose_2d_width_and_width_large) {
  test_transpose({53, 117}, 1, 1);
}

TEST_F(VulkanAPITest, transpose_3d_height_and_width_small) {
  test_transpose({1, 1, 1}, 1, 2);
}

TEST_F(VulkanAPITest, transpose_3d_height_and_width_medium) {
  test_transpose({3, 2, 5}, 1, 2);
}

TEST_F(VulkanAPITest, transpose_3d_height_and_width_large) {
  test_transpose({100, 1, 144}, 1, 2);
}

TEST_F(VulkanAPITest, transpose_3d_width_and_width_large) {
  test_transpose({100, 1, 144}, 2, 2);
}

TEST_F(VulkanAPITest, transpose_3d_depth_and_width_small) {
  test_transpose({1, 1, 1}, 0, 2);
}

TEST_F(VulkanAPITest, transpose_3d_depth_and_width_medium) {
  test_transpose({3, 2, 5}, 0, 2);
}

TEST_F(VulkanAPITest, transpose_3d_depth_and_width_large) {
  test_transpose({113, 1, 141}, 0, 2);
}

TEST_F(VulkanAPITest, transpose_3d_depth_and_depth_large) {
  test_transpose({113, 2, 131}, 0, 0);
}

TEST_F(VulkanAPITest, transpose_3d_depth_and_height_small) {
  test_transpose({1, 1, 1}, 0, 1);
}

TEST_F(VulkanAPITest, transpose_3d_depth_and_height_medium) {
  test_transpose({3, 7, 5}, 0, 1);
}

TEST_F(VulkanAPITest, transpose_3d_depth_and_height_large) {
  test_transpose({113, 141, 1}, 0, 1);
}

TEST_F(VulkanAPITest, transpose_3d_height_and_height_large) {
  test_transpose({101, 1, 141}, 1, 1);
}

TEST_F(VulkanAPITest, transpose_4d_batch_and_batch_large) {
  test_transpose({7, 51, 41, 3}, 0, 0);
}

TEST_F(VulkanAPITest, transpose_4d_depth_and_depth_large) {
  test_transpose({7, 51, 41, 3}, 1, 1);
}

TEST_F(VulkanAPITest, transpose_4d_height_and_height_large) {
  test_transpose({7, 51, 41, 3}, 2, 2);
}

TEST_F(VulkanAPITest, transpose_4d_width_and_width_large) {
  test_transpose({7, 51, 41, 3}, 3, 3);
}

TEST_F(VulkanAPITest, transpose_4d_batch_and_depth_large) {
  test_transpose({7, 51, 41, 3}, 0, 1);
}

TEST_F(VulkanAPITest, transpose_4d_batch_and_height_large) {
  test_transpose({7, 51, 41, 3}, 0, 2);
}

TEST_F(VulkanAPITest, transpose_4d_batch_and_width_large) {
  test_transpose({7, 51, 41, 3}, 0, 3);
}

TEST_F(VulkanAPITest, transpose_4d_depth_and_height_large) {
  test_transpose({7, 51, 41, 3}, 1, 2);
}

TEST_F(VulkanAPITest, transpose_4d_depth_and_width_large) {
  test_transpose({7, 51, 41, 3}, 1, 3);
}

TEST_F(VulkanAPITest, transpose_4d_height_and_width_large) {
  test_transpose({7, 51, 41, 3}, 2, 3);
}

// Test Unary Ops
void test_exp(const at::IntArrayRef input_shape) {
  c10::InferenceMode mode;
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::exp(in_cpu);

  const auto in_vulkan = in_cpu.vulkan();
  const auto out_vulkan = at::exp(in_vulkan);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
    std::cout << "exp test failed with input shape: "
              << input_shape << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, unary_op_exp) {
  test_exp({5});
  test_exp({5, 6});
  test_exp({7, 3, 5});
  test_exp({11, 1, 4, 2});
}

void test_exp_(const at::IntArrayRef input_shape) {
  c10::InferenceMode mode;
  const auto cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto vulkan = cpu.vulkan();

  cpu.exp_();
  vulkan.exp_();

  const auto check = almostEqual(cpu, vulkan.cpu());
  if (!check) {
    showRtol(cpu, vulkan.cpu());
    std::cout << "exp_ test failed with input shape: "
              << input_shape << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, unary_op_exp_) {
  test_exp_({5});
  test_exp_({5, 6});
  test_exp_({7, 3, 5});
  test_exp_({11, 1, 4, 2});
}

void test_sqrt(const at::IntArrayRef input_shape) {
  c10::InferenceMode mode;
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::sqrt(in_cpu);

  const auto in_vulkan = in_cpu.vulkan();
  const auto out_vulkan = at::sqrt(in_vulkan);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
    std::cout << "sqrt test failed with input shape: "
              << input_shape << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, unary_op_sqrt) {
  test_sqrt({5});
  test_sqrt({5, 6});
  test_sqrt({7, 3, 5});
  test_sqrt({11, 1, 4, 2});
}

void test_sqrt_(const at::IntArrayRef input_shape) {
  c10::InferenceMode mode;
  const auto cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto vulkan = cpu.vulkan();

  cpu.sqrt_();
  vulkan.sqrt_();

  const auto check = almostEqual(cpu, vulkan.cpu());
  if (!check) {
    showRtol(cpu, vulkan.cpu());
    std::cout << "sqrt_ test failed with input shape: "
              << input_shape << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, unary_op_sqrt_) {
  test_sqrt_({5});
  test_sqrt_({5, 6});
  test_sqrt_({7, 3, 5});
  test_sqrt_({11, 1, 4, 2});
}

void test_log(const at::IntArrayRef input_shape) {
  c10::InferenceMode mode;
  // Need to add a very small constant to avoid 0.
  const auto in_cpu =
      at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat)) + 0.0001;
  const auto out_cpu = at::log(in_cpu);

  const auto in_vulkan = in_cpu.vulkan();
  const auto out_vulkan = at::log(in_vulkan);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
    std::cout << "log test failed with input shape: " << input_shape
              << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, unary_op_log) {
  test_log({5});
  test_log({5, 6});
  test_log({7, 3, 5});
  test_log({11, 1, 4, 2});
}

void test_log_(const at::IntArrayRef input_shape) {
  c10::InferenceMode mode;
  // Need to add a very small constant to avoid 0.
  const auto cpu =
      at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat)) + 0.0001;
  const auto vulkan = cpu.vulkan();

  cpu.log_();
  vulkan.log_();

  const auto check = almostEqual(cpu, vulkan.cpu());
  if (!check) {
    showRtol(cpu, vulkan.cpu());
    std::cout << "log_ test failed with input shape: " << input_shape
              << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, unary_op_log_) {
  test_log_({5});
  test_log_({5, 6});
  test_log_({7, 3, 5});
  test_log_({11, 1, 4, 2});
}

void test_unsqueeze(const at::IntArrayRef input_shape, int64_t dim) {
  c10::InferenceMode mode;
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::unsqueeze(in_cpu, dim);

  const auto in_vulkan = in_cpu.vulkan();
  const auto out_vulkan = at::unsqueeze(in_vulkan, dim);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
    std::cout << "unsqueeze test failed with input shape: "
              << input_shape << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, unsqueeze_0dto1d_dim0) {
  test_unsqueeze({}, 0);
  test_unsqueeze({}, -1);
}

TEST_F(VulkanAPITest, unsqueeze_1dto2d_dim0) {
  test_unsqueeze({5}, 0);
  test_unsqueeze({6}, -2);
  test_unsqueeze({111}, 0);
  test_unsqueeze({112}, -2);
}

TEST_F(VulkanAPITest, unsqueeze_1dto2d_dim1) {
  test_unsqueeze({5}, 1);
  test_unsqueeze({6}, -1);
  test_unsqueeze({111}, 1);
  test_unsqueeze({112}, -1);
}

TEST_F(VulkanAPITest, unsqueeze_2dto3d_dim0) {
  test_unsqueeze({1, 5}, 2);
  test_unsqueeze({5, 7}, 0);
  test_unsqueeze({7, 5}, -3);
  test_unsqueeze({111, 222}, 0);
  test_unsqueeze({222, 111}, -3);
}

TEST_F(VulkanAPITest, unsqueeze_2dto3d_dim1) {
  test_unsqueeze({5, 7}, 1);
  test_unsqueeze({7, 5}, -2);
  test_unsqueeze({111, 222}, 1);
  test_unsqueeze({222, 111}, -2);
}

TEST_F(VulkanAPITest, unsqueeze_2dto3d_dim2) {
  test_unsqueeze({5, 7}, 2);
  test_unsqueeze({7, 5}, -1);
  test_unsqueeze({111, 222}, 2);
  test_unsqueeze({222, 111}, -1);
}

TEST_F(VulkanAPITest, unsqueeze_3dto4d_dim0) {
  test_unsqueeze({2, 3, 4}, 0);
  test_unsqueeze({4, 3, 2}, -4);
  test_unsqueeze({22, 33, 11}, 0);
  test_unsqueeze({33, 11, 22}, -4);
}

TEST_F(VulkanAPITest, unsqueeze_3dto4d_dim1) {
  test_unsqueeze({2, 3, 4}, 1);
  test_unsqueeze({4, 3, 2}, -3);
  test_unsqueeze({22, 33, 11}, 1);
  test_unsqueeze({33, 11, 22}, -3);
}

TEST_F(VulkanAPITest, unsqueeze_3dto4d_dim2) {
  test_unsqueeze({2, 3, 4}, 2);
  test_unsqueeze({4, 3, 2}, -2);
  test_unsqueeze({22, 33, 11}, 2);
  test_unsqueeze({33, 11, 22}, -2);
}

TEST_F(VulkanAPITest, unsqueeze_3dto4d_dim3) {
  test_unsqueeze({1, 5, 2}, 3);
  test_unsqueeze({2, 3, 4}, 3);
  test_unsqueeze({4, 3, 2}, -1);
  test_unsqueeze({22, 33, 11}, 3);
  test_unsqueeze({33, 11, 22}, -1);
}

TEST_F(VulkanAPITest, upsample_nearest2d) {
  const auto in_cpu = at::rand({1, 2, 2, 3}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::upsample_nearest2d(in_cpu, {4, 6});

  const auto in_vulkan = in_cpu.vulkan();
  const auto out_vulkan = at::upsample_nearest2d(in_vulkan, {4, 6});

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, upsample_bilinear2d_align_false_small) {
  const auto in_cpu = at::rand({1, 2, 2, 3}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::upsample_bilinear2d(in_cpu, {4, 6}, false);

  const auto in_vulkan = in_cpu.vulkan();
  const auto out_vulkan = at::upsample_bilinear2d(in_vulkan, {4, 6}, false);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, upsample_bilinear2d_align_false_large) {
  const auto in_cpu = at::rand({1, 7, 25, 25}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::upsample_bilinear2d(in_cpu, {45, 45}, false);

  const auto in_vulkan = in_cpu.vulkan();
  const auto out_vulkan = at::upsample_bilinear2d(in_vulkan, {45, 45}, false);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, upsample_bilinear2d_align_true_small) {
  const auto in_cpu = at::rand({1, 2, 2, 3}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::upsample_bilinear2d(in_cpu, {4, 6}, true);

  const auto in_vulkan = in_cpu.vulkan();
  const auto out_vulkan = at::upsample_bilinear2d(in_vulkan, {4, 6}, true);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, upsample_bilinear2d_align_true_large) {
  const auto in_cpu = at::rand({1, 7, 25, 25}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::upsample_bilinear2d(in_cpu, {45, 45}, true);

  const auto in_vulkan = in_cpu.vulkan();
  const auto out_vulkan = at::upsample_bilinear2d(in_vulkan, {45, 45}, true);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

void test_unbind(const at::IntArrayRef input_shape, int64_t dim) {
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::unbind(in_cpu, dim);

  const auto in_vulkan = in_cpu.vulkan();
  const auto out_vulkan = at::unbind(in_vulkan, dim);

  int64_t size = out_vulkan.size();

  for (const auto i : c10::irange(size)) {
    const auto check = almostEqual(out_cpu[i], out_vulkan[i].cpu());
    if (!check) {
      std::cout << "The " << i << "th vectors aren't equal." << std::endl;
      showRtol(out_cpu[i], out_vulkan[i].cpu());
    }

    ASSERT_TRUE(check);
  }
}

TEST_F(VulkanAPITest, unbind_3d_depth_small) {
  test_unbind({1, 1, 1}, 0);
}

TEST_F(VulkanAPITest, unbind_3d_depth_medium) {
  test_unbind({3, 2, 5}, 0);
}

TEST_F(VulkanAPITest, unbind_3d_depth_large) {
  test_unbind({100, 1, 144}, 0);
}

void test_var(const at::IntArrayRef input_shape, const at::IntArrayRef dim_list, bool unbiased=true, bool keepdim=false) {
  c10::InferenceMode mode;

  const auto in_cpu = at::rand(input_shape, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::var(in_cpu, dim_list, unbiased, keepdim);

  const auto in_vulkan = in_cpu.vulkan();
  const auto out_vulkan = at::var(in_vulkan, dim_list, unbiased, keepdim);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, var_2d_unbiased) {
  test_var({3, 5}, {1}, true, true);
  test_var({3, 5}, {1}, true, false);

  // input.dim() == dim_list.size(), only keepdim == true is supported
  test_var({3, 5}, {0, 1}, true, true);
}

TEST_F(VulkanAPITest, var_2d_biased) {
  test_var({3, 5}, {1}, false, true);
  test_var({3, 5}, {1}, false, false);

  // input.dim() == dim_list.size(), only keepdim == true is supported
  test_var({3, 5}, {0, 1}, false, true);
}

TEST_F(VulkanAPITest, var_3d_unbiased) {
  test_var({3, 5, 7}, {1}, true, true);
  test_var({3, 5, 7}, {1}, true, false);

  test_var({3, 5, 7}, {0, 1}, true, true);
  test_var({3, 5, 7}, {0, 1}, true, false);

  test_var({3, 5, 7}, {0, 2}, true, true);
  test_var({3, 5, 7}, {0, 2}, true, false);

  test_var({3, 5, 7}, {-1, -2}, true, true);
  test_var({3, 5, 7}, {-1, -2}, true, false);

  test_var({3, 5, 7}, {0, 1, 2}, true, true);
}

TEST_F(VulkanAPITest, var_3d_biased) {
  test_var({3, 5, 7}, {1}, false, true);
  test_var({3, 5, 7}, {1}, false, false);

  test_var({3, 5, 7}, {0, 1}, false, true);
  test_var({3, 5, 7}, {0, 1}, false, false);

  test_var({3, 5, 7}, {0, 2}, false, true);
  test_var({3, 5, 7}, {0, 2}, false, false);

  test_var({3, 5, 7}, {-1, -2}, false, true);
  test_var({3, 5, 7}, {-1, -2}, false, false);

  test_var({3, 5, 7}, {0, 1, 2}, false, true);
}

TEST_F(VulkanAPITest, var_4d_unbiased) {
  test_var({3, 5, 7, 11}, {0}, true, true);
  test_var({3, 5, 7, 11}, {1}, true, false);

  test_var({3, 5, 7, 11}, {0, 1}, true, true);
  test_var({3, 5, 7, 11}, {0, 1}, true, false);

  test_var({3, 5, 7, 11}, {0, 2}, true, true);
  test_var({3, 5, 7, 11}, {0, 2}, true, false);

  test_var({3, 5, 7, 11}, {-1, -2}, true, true);
  test_var({3, 5, 7, 11}, {-1, -2}, true, false);

  test_var({3, 5, 7, 11}, {0, 1, 2}, true, true);
  test_var({3, 5, 7, 11}, {0, -1, 2}, true, false);

  test_var({3, 5, 7, 11}, {0, 1, 2, 3}, true, true);
}

TEST_F(VulkanAPITest, var_4d_biased) {
  test_var({3, 5, 7, 11}, {0}, false, true);
  test_var({3, 5, 7, 11}, {1}, false, false);

  test_var({3, 5, 7, 11}, {0, 1}, false, true);
  test_var({3, 5, 7, 11}, {0, 1}, false, false);

  test_var({3, 5, 7, 11}, {0, 2}, false, true);
  test_var({3, 5, 7, 11}, {0, 2}, false, false);

  test_var({3, 5, 7, 11}, {-1, -2}, false, true);
  test_var({3, 5, 7, 11}, {-1, -2}, false, false);

  test_var({3, 5, 7, 11}, {0, 1, 2}, false, true);
  test_var({3, 5, 7, 11}, {0, -1, 2}, false, false);

  test_var({3, 5, 7, 11}, {0, 1, 2, 3}, false, true);
}

TEST_F(VulkanAPITest, view_explicit) {
  c10::InferenceMode mode;

  const auto in_cpu = at::rand({7, 8, 9}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan = in_cpu.vulkan();

  const std::array<int64_t, 4> shape{7, 8, 9, 1};

  const auto out_cpu = in_cpu.view(shape);
  const auto out_vulkan = in_vulkan.view(shape);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, view_inferred) {
  c10::InferenceMode mode;

  const auto in_cpu = at::rand({7, 11, 8, 9}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan = in_cpu.vulkan();

  const std::array<int64_t, 3> shape{7, 11, -1};

  const auto out_cpu = in_cpu.view(shape);
  const auto out_vulkan = in_vulkan.view(shape);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, view_invalid_inputs) {
  c10::InferenceMode mode;

  // Act: only one dimension can be inferred
  EXPECT_THROW({
    at::rand({7, 8, 9}, at::device(at::kCPU).dtype(at::kFloat))
      .vulkan().view({7, -1, -1});
  }, ::std::runtime_error);

  // Act: invalid shape dimension
  EXPECT_THROW({
    at::rand({7, 8, 9}, at::device(at::kCPU).dtype(at::kFloat))
      .vulkan().view({7, 8, -2});
  }, ::std::exception);

  // Act: incompatible shape
  EXPECT_THROW({
    at::rand({7, 8, 9}, at::device(at::kCPU).dtype(at::kFloat))
      .vulkan().view({7, 70});
  }, ::std::runtime_error);
}

TEST_F(VulkanAPITest, cat_4d_dim0_invalidinputs_exceptions) {
  // Arrange: Vulkan cat inputs must have matching sizes except concatenated dimension
  {
    const auto in_cpu1 = at::rand({3, 5, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_cpu2 = at::rand({3, 9, 112, 193}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_cpu3 = at::rand({3, 9, 331, 193}, at::device(at::kCPU).dtype(at::kFloat));

    // Act
    EXPECT_THROW({
      const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 0);
    }, ::std::exception);
  }

  // Arrange: Vulkan cat expects 4 dimensional inputs
  {
    const auto in_cpu1 = at::rand({3, 9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_cpu2 = at::rand({9, 112, 193}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_cpu3 = at::rand({3, 9, 331, 193}, at::device(at::kCPU).dtype(at::kFloat));

    // Act
    EXPECT_THROW({
      const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 0);
    }, ::std::exception);
  }
}

TEST_F(VulkanAPITest, cat_4d_dim0_samebatch_success) {
  // Arrange
  const auto in_cpu1 = at::rand({221, 3, 9, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({221, 3, 9, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({221, 3, 9, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 0);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 0); // dim=batch

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_4d_dim0_diffbatch_success) {
  // Arrange
  const auto in_cpu1 = at::rand({221, 3, 9, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({117, 3, 9, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({139, 3, 9, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 0);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 0); // dim=batch

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_4d_dim0_singledepth_success) {
  // Arrange: batch x channel (1x1) = single depth texture
  const auto in_cpu1 = at::rand({1, 1, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({1, 1, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({1, 1, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 0);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 0); // dim=batch

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_4d_dim0_singletensor_success) {
  // Arrange: single input tensor
  const auto in_cpu1 = at::rand({3, 7, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1}, 0);
  const auto out_vulkan = at::cat({in_cpu1}, 0); // dim=batch

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_4d_dim0_twotensors_success) {
  // Arrange: two input tensors
  const auto in_cpu1 = at::rand({3, 7, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({3, 7, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2}, 0);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan()}, 0); // dim=batch

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_4d_dim0_negdim_success) {
  // Arrange
  const auto in_cpu1 = at::rand({221, 9, 193, 3}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({113, 9, 193, 3}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({331, 9, 193, 3}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, -4);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, -4);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_4d_dim1_negdim_success) {
  // Arrange
  const auto in_cpu1 = at::rand({9, 221, 193, 3}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({9, 113, 193, 3}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({9, 331, 193, 3}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, -3);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, -3);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_4d_dim2_negdim_success) {
  // Arrange
  const auto in_cpu1 = at::rand({9, 193, 221, 3}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({9, 193, 113, 3}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({9, 193, 331, 3}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, -2);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, -2);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_4d_dim3_negdim_success) {
  // Arrange
  const auto in_cpu1 = at::rand({9, 193, 3, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({9, 193, 3, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({9, 193, 3, 331}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, -1);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, -1);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

#if !defined(__APPLE__)
TEST_F(VulkanAPITest, DISABLED_cat_4d_dim1_samefeature_success) {
  // Arrange
  const auto in_cpu1 = at::rand({3, 9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({3, 9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({3, 9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 1);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 1); // dim=feature(channel)

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, DISABLED_cat_4d_dim1_difffeature_success) {
  // Arrange
  const auto in_cpu1 = at::rand({3, 3, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({3, 8, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({3, 11, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 1);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 1); // dim=feature(channel)

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_4d_dim1_texture2d_success) {
  // Arrange: 2D Texture (VK_IMAGE_VIEW_TYPE_2D)
  const auto in_cpu1 = at::rand({2, 3, 2, 2}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({2, 3, 2, 2}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({2, 3, 2, 2}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 1);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 1); // dim=feature(channel)

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}
#endif /* !defined(__APPLE__) */

TEST_F(VulkanAPITest, cat_4d_dim1_singledepth_success) {
  // Arrange: batch x channel (1x1) = single depth texture
  const auto in_cpu1 = at::rand({1, 1, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({1, 1, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({1, 1, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 1);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 1); // dim=feature(channel)

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_4d_dim1_singletensor_success) {
  // Arrange: single input tensor
  const auto in_cpu1 = at::rand({3, 7, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1}, 1);
  const auto out_vulkan = at::cat({in_cpu1}, 1); // dim=feature(channel)

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, DISABLED_cat_4d_dim1_twotensors_success) {
  // Arrange: two input tensors
  const auto in_cpu1 = at::rand({3, 7, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({3, 7, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2}, 1);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan()}, 1); // dim=feature(channel)

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_4d_dim1_bat1_mult4ch_success) {
  // Arrange: batch=1 and channel (a multiple of 4 <-> channel %4 == 0)
  const auto in_cpu1 = at::rand({1, 4, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({1, 4, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({1, 4, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 1);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 1); // dim=feature(channel)

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_4d_dim1_bat2_mult4ch_success) {
  // Arrange: batch=2 and channel (a multiple of 4 <-> channel %4 == 0)
  const auto in_cpu1 = at::rand({2, 4, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({2, 4, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({2, 4, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 1);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 1); // dim=feature(channel)

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_4d_dim1_mult4ch_mixed_success) {
  // Arrange: batch=1 and channel (different multiples of 4 <-> channel %4 == 0)
  const auto in_cpu1 = at::rand({3, 4, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({3, 8, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({3, 12, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 1);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 1); // dim=feature(channel)

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, DISABLED_cat_4d_dim1_mult4ch_nonmult4ch_success) {
  // Arrange: batch=1 and channel (a mixed set of multiples and non-multiples of 4)
  const auto in_cpu1 = at::rand({3, 3, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({3, 4, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({3, 7, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu4 = at::rand({3, 8, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3, in_cpu4}, 1);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan(), in_cpu4.vulkan()}, 1); // dim=feature(channel)

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_4d_dim2_sameheight_success) {
  // Arrange
  const auto in_cpu1 = at::rand({3, 9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({3, 9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({3, 9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 2);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 2);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_4d_dim2_diffheight_success) {
  // Arrange
  const auto in_cpu1 = at::rand({3, 9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({3, 9, 112, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({3, 9, 331, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 2);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 2);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_4d_dim2_singledepth_success) {
  // Arrange: batch x channel (1x1) = single depth texture
  const auto in_cpu1 = at::rand({1, 1, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({1, 1, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({1, 1, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 2);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 2);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_4d_dim2_invalidinputs_exceptions) {
  // Arrange: Vulkan cat inputs must have matching sizes except concatenated dimension
  {
    const auto in_cpu1 = at::rand({3, 5, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_cpu2 = at::rand({3, 9, 112, 193}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_cpu3 = at::rand({3, 9, 331, 193}, at::device(at::kCPU).dtype(at::kFloat));

    // Act
    EXPECT_THROW({
      const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 2);
    }, ::std::exception);
  }

  // Arrange: Vulkan cat expects inputs of same dimensions
  {
    const auto in_cpu1 = at::rand({3, 9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_cpu2 = at::rand({9, 112, 193}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_cpu3 = at::rand({3, 9, 331, 193}, at::device(at::kCPU).dtype(at::kFloat));

    // Act
    EXPECT_THROW({
      const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 2);
    }, ::std::exception);
  }
}

TEST_F(VulkanAPITest, cat_4d_dim3_invalidinputs_exceptions) {
  // Arrange: Vulkan cat inputs must have matching sizes except concatenated dimension
  {
    const auto in_cpu1 = at::rand({3, 5, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_cpu2 = at::rand({3, 9, 112, 193}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_cpu3 = at::rand({3, 9, 331, 193}, at::device(at::kCPU).dtype(at::kFloat));

    // Act
    EXPECT_THROW({
      const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 3);
    }, ::std::exception);
  }

  // Arrange: Vulkan cat expects 4 dimensional inputs
  {
    const auto in_cpu1 = at::rand({3, 9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_cpu2 = at::rand({9, 112, 193}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_cpu3 = at::rand({3, 9, 331, 193}, at::device(at::kCPU).dtype(at::kFloat));

    // Act
    EXPECT_THROW({
      const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 3);
    }, ::std::exception);
  }
}

TEST_F(VulkanAPITest, cat_4d_dim3_samewidth_success) {
  // Arrange
  const auto in_cpu1 = at::rand({3, 9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({3, 9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({3, 9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 3);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 3);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_4d_dim3_diffwidth_success) {
  // Arrange
  const auto in_cpu1 = at::rand({3, 9, 193, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({3, 9, 193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({3, 9, 193, 331}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 3);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 3);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_3d_dim0_mult4ch_success) {
  // Arrange
  const auto in_cpu1 = at::rand({4, 193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({4, 193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({4, 193, 113}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 0);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 0);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_3d_dim0_diff_channel_success) {
  // Arrange
  const auto in_cpu1 = at::rand({221, 9, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({113, 9, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({331, 9, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 0);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 0);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_3d_dim0_same_channel_success) {
  // Arrange
  const auto in_cpu1 = at::rand({9, 193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({9, 193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({9, 193, 113}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 0);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 0);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_3d_dim1_diffheight_success) {
  // Arrange
  const auto in_cpu1 = at::rand({9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({9, 113, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({9, 331, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 1);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 1);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_3d_dim1_same_height_success) {
  // Arrange
  const auto in_cpu1 = at::rand({9, 193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({9, 193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({9, 193, 113}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 1);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 1);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_3d_dim2_diffwidth_success) {
  // Arrange
  const auto in_cpu1 = at::rand({9, 193, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({9, 193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({9, 193, 331}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 2);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 2);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_3d_dim2_samewidth_success) {
  // Arrange
  const auto in_cpu1 = at::rand({9, 193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({9, 193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({9, 193, 113}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 2);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 2);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_3d_dim0_negdim_success) {
  // Arrange
  const auto in_cpu1 = at::rand({221, 9, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({113, 9, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({331, 9, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, -3);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, -3);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_3d_dim1_negdim_success) {
  // Arrange
  const auto in_cpu1 = at::rand({9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({9, 113, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({9, 331, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, -2);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, -2);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_3d_dim2_negdim_success) {
  // Arrange
  const auto in_cpu1 = at::rand({193, 13, 89}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({193, 13, 59}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({193, 13, 67}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, -1);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, -1);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_2d_dim0_same_height_success) {
  // Arrange
  const auto in_cpu1 = at::rand({193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({193, 113}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 0);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 0);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_2d_dim0_diff_height_success) {
  // Arrange
  const auto in_cpu1 = at::rand({193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({191, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({137, 113}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 0);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 0);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_2d_dim1_same_width_success) {
  // Arrange
  const auto in_cpu1 = at::rand({193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({193, 113}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 1);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 1);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_2d_dim1_diff_width_success) {
  // Arrange
  const auto in_cpu1 = at::rand({193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({193, 131}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({193, 127}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 1);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 1);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_2d_dim0_negdim_success) {
  // Arrange
  const auto in_cpu1 = at::rand({113, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({131, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({127, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, -2);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, -2);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_2d_dim1_negdim_success) {
  // Arrange
  const auto in_cpu1 = at::rand({193, 113}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({193, 131}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({193, 127}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, -1);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, -1);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_1d_dim0_same_width_success) {
  // Arrange
  const auto in_cpu1 = at::rand({193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 0);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 0);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_1d_dim0_diff_width_success) {
  // Arrange
  const auto in_cpu1 = at::rand({193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({137}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({131}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, 0);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 0);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, cat_1d_dim0_negdim_success) {
  // Arrange
  const auto in_cpu1 = at::rand({193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({137}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({131}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::cat({in_cpu1, in_cpu2, in_cpu3}, -1);
  const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, -1);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, permute_2d_success) {
  // Arrange
  const auto in_cpu = at::rand({2, 3}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::permute(in_cpu, {1, 0});
  const auto out_vulkan = at::permute(in_cpu.vulkan(), {1, 0});

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, permute_3d_success) {
  // Arrange
  const auto in_cpu = at::rand({2, 3, 2}, at::device(at::kCPU).dtype(at::kFloat));
  std::vector<std::vector<int64_t>> all_dims;
  std::vector<int64_t> in{0, 1, 2};
  gen_allpermutations(all_dims, in, 0);

  for (const auto i : c10::irange(1, all_dims.size())) {
    const auto dims = all_dims[i];

    // Act
    const auto out_cpu = at::permute(in_cpu, dims);
    const auto out_vulkan = at::permute(in_cpu.vulkan(), dims);

    // Assert
    const auto check = almostEqual(out_cpu, out_vulkan.cpu());
    if (!check) {
      showRtol(out_cpu, out_vulkan.cpu());
    }

    ASSERT_TRUE(check);
  }
}

TEST_F(VulkanAPITest, permute_4d_success) {
  // Arrange
  const auto in_cpu = at::rand({2, 3, 4, 5}, at::device(at::kCPU).dtype(at::kFloat));
  std::vector<std::vector<int64_t>> all_dims;
  std::vector<int64_t> in{0, 1, 2, 3};
  gen_allpermutations(all_dims, in, 0);

  for (const auto i : c10::irange(1, all_dims.size())) {
    const auto dims = all_dims[i];

    // Act
    const auto out_cpu = at::permute(in_cpu, dims);
    const auto out_vulkan = at::permute(in_cpu.vulkan(), dims);

    // Assert
    const auto check = almostEqual(out_cpu, out_vulkan.cpu());
    if (!check) {
      showRtol(out_cpu, out_vulkan.cpu());
    }

    ASSERT_TRUE(check);
  }
}

TEST_F(VulkanAPITest, permute_4dmclaren_success) {
  // Arrange: McLaren Model usage
  const auto in_cpu = at::rand({1, 2, 1, 161}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::permute(in_cpu, {0, 2, 1, 3});
  const auto out_vulkan = at::permute(in_cpu.vulkan(), {0, 2, 1, 3});

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, permute_4dbig_success) {
  // Arrange
  const auto in_cpu = at::rand({3, 9, 51, 41}, at::device(at::kCPU).dtype(at::kFloat));
  std::vector<std::vector<int64_t>> all_dims;
  std::vector<int64_t> in{0, 1, 2, 3};
  gen_allpermutations(all_dims, in, 0);

  for (const auto i : c10::irange(1, all_dims.size())) {
    const auto dims = all_dims[i];
    // Act
    const auto out_cpu = at::permute(in_cpu, dims);
    const auto out_vulkan = at::permute(in_cpu.vulkan(), dims);

    // Assert
    const auto check = almostEqual(out_cpu, out_vulkan.cpu());
    if (!check) {
      showRtol(out_cpu, out_vulkan.cpu());
    }

    ASSERT_TRUE(check);
  }
}

TEST_F(VulkanAPITest, permute_negativedims_success) {
  // Arrange
  const auto in_cpu = at::rand({5, 4, 3, 2}, at::device(at::kCPU).dtype(at::kFloat));

  // Act: {-1,-2,-3,0} is equivalent to {3,2,1,0}
  const auto out_cpu = at::permute(in_cpu, {-1, -2, -3, 0});
  const auto out_vulkan = at::permute(in_cpu.vulkan(), {-1, -2, -3, 0});

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, permute_invalidinputs_exceptions) {
  // Arrange
  const auto in_cpu = at::rand({1, 2, 1, 161}, at::device(at::kCPU).dtype(at::kFloat));

  // Act: Repeated dim
  EXPECT_THROW({
    const auto out_vulkan = at::permute(in_cpu.vulkan(), {2, 2, 1, 0});
  }, ::std::exception);

  EXPECT_THROW({
    const auto out_vulkan = in_cpu.vulkan();
    out_vulkan.permute({2, 2, 1, 0});
  }, ::std::exception);

  // Act: Number of dims don't match
  EXPECT_THROW({
    const auto out_vulkan = at::permute(in_cpu.vulkan(), {4, 3, 2, 1, 0});
  }, ::std::exception);

  EXPECT_THROW({
    const auto out_vulkan = at::permute(in_cpu.vulkan(), {2, 1, 0});
  }, ::std::exception);

  EXPECT_THROW({
    const auto out_vulkan = in_cpu.vulkan();
    out_vulkan.permute({4, 3, 2, 1, 0});
  }, ::std::exception);

  EXPECT_THROW({
    const auto out_vulkan = in_cpu.vulkan();
    out_vulkan.permute({2, 1, 0});
  }, ::std::exception);

  // Act: Dim out of range
  EXPECT_THROW({
    const auto out_vulkan = at::permute(in_cpu.vulkan(), {5, 2, 1, 0});
  }, ::std::exception);

  EXPECT_THROW({
    const auto out_vulkan = in_cpu.vulkan();
    out_vulkan.permute({5, 2, 1, 0});
  }, ::std::exception);

  // Act: Input tensor size > 4D
  const auto in_cpu_5d = at::rand({1, 2, 1, 2, 161}, at::device(at::kCPU).dtype(at::kFloat));
  EXPECT_THROW({
    const auto out_vulkan_5d = at::permute(in_cpu_5d.vulkan(), {4, 3, 2, 1, 0});
  }, ::std::exception);

  EXPECT_THROW({
    const auto out_vulkan_5d = in_cpu_5d.vulkan();
    out_vulkan_5d.permute({4, 3, 2, 1, 0});
  }, ::std::exception);
}

TEST_F(VulkanAPITest, slice_width_success) {
  // Arrange
  std::unordered_map<int64_t, std::vector<int64_t>> dim2sizes {
    {3, {2, 3, 40, 50}},  // 4D tensors with dim=width
    {2, {3, 40, 50}},     // 3D tensors with dim=width
    {1, {40, 50}},        // 2D tensors with dim=width
    {0, {50}},            // 1D tensors with dim=width
  };

  // Act/Assert
  slice_tests(dim2sizes);
}

TEST_F(VulkanAPITest, slice_height_success) {
  // Arrange
  std::unordered_map<int64_t, std::vector<int64_t>> dim2sizes {
    {2, {2, 3, 40, 50}},  // 4D tensors with dim=height
    {1, {3, 40, 50}},     // 3D tensors with dim=height
    {0, {40, 50}},        // 2D tensors with dim=height
                          // 1D tesnors don't have height dim for test
  };

  // Act/Assert
  slice_tests(dim2sizes);
}

TEST_F(VulkanAPITest, slice_feature_success) {
  // Arrange
  std::unordered_map<int64_t, std::vector<int64_t>> dim2sizes {
    {1, {2, 40, 13, 14}}, // 4D tensors with dim=feature(channel)
    {0, {40, 13, 14}},    // 3D tensors with dim=feature(channel)
                          // 1D and 2D tesnors don't have feature(channel) dim for test
  };

  // Act/Assert
  slice_tests(dim2sizes);
}

TEST_F(VulkanAPITest, slice_batch_success) {
  // Arrange
  std::unordered_map<int64_t, std::vector<int64_t>> dim2sizes {
    {0, {40, 3, 13, 14}}, // 4D tensors with dim=batch
                          // 1D, 2D and 3D tesnors don't have batch dim for test
  };

  // Act/Assert
  slice_tests(dim2sizes);
}

TEST_F(VulkanAPITest, slice_zero_sized) {
  // When start == end
  slice_test({2, 3, 4, 5}, 3, 0, 0, 1);
  // When start > end
  slice_test({2, 3, 4, 5}, 3, 3, 2, 1);
}

TEST_F(VulkanAPITest, slice_invalidinputs_exceptions) {
  // Act: slice step must be positive
  EXPECT_THROW({
    slice_test({2, 3, 4, 5}, 3, 0, 3, 0);
  }, ::std::exception);
}

TEST_F(VulkanAPITest, stack_invalid_inputs) {
  // Act: Vulkan stack expects at least one tensor
  EXPECT_THROW({
    at::stack({}, 0);
  }, ::std::exception);

  // Act: Vulkan stack inputs must have matching sizes
  EXPECT_THROW({
    at::stack({
        at::rand({5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
        at::rand({5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
        at::rand({6, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan()}, 0);
  }, ::std::exception);
}

void test_stack(const at::IntArrayRef input_shape, int64_t dim, int numTensors) {
  std::vector<at::Tensor> tensors_cpu = {};
  std::vector<at::Tensor> tensors_vulkan = {};

  for (int i = 0; i < numTensors; i++) {
    at::Tensor in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
    tensors_cpu.emplace_back(in_cpu);
    tensors_vulkan.emplace_back(in_cpu.vulkan());
  }

  at::Tensor out_cpu = at::stack(tensors_cpu, 0);
  at::Tensor out_vulkan = at::stack(tensors_vulkan, 0);
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    std::cout << "Error when stacking " << numTensors << " tensors" << std::endl;
    showRtol(out_cpu, out_vulkan.cpu());
  }
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, stack_0d) {
  test_stack({}, 0, 1);
  test_stack({}, 0, 2);
  test_stack({}, 0, 3);
}

TEST_F(VulkanAPITest, stack_1d) {
  test_stack({221}, 0, 2);
  test_stack({193}, 1, 3);

  test_stack({221}, -1, 2);
  test_stack({193}, -2, 3);
}

TEST_F(VulkanAPITest, stack_2d) {
  test_stack({221, 193}, 0, 2);
  test_stack({221, 193}, 1, 3);
  test_stack({221, 193}, 2, 4);

  test_stack({221, 193}, -1, 2);
  test_stack({221, 193}, -2, 3);
  test_stack({221, 193}, -3, 4);
}

TEST_F(VulkanAPITest, stack_3d) {
  test_stack({221, 193, 11}, 0, 2);
  test_stack({221, 193, 11}, 1, 3);
  test_stack({221, 193, 11}, 2, 4);
  test_stack({221, 193, 11}, 3, 5);

  test_stack({221, 193, 11}, -1, 2);
  test_stack({221, 193, 11}, -2, 3);
  test_stack({221, 193, 11}, -3, 4);
  test_stack({221, 193, 11}, -4, 5);
}

TEST_F(VulkanAPITest, tile_invalid_inputs_exceptions) {
  // Arrange: Vulkan tile only supports input of dims <= 4
  {
    const auto in_cpu =
        at::rand({3, 9, 5, 7, 3}, at::device(at::kCPU).dtype(at::kFloat));
    const at::IntArrayRef repeats = {7, 3, 9, 2};

    // Act
    EXPECT_THROW(
        { const auto out_vulkan = at::tile(in_cpu.vulkan(), repeats); },
        ::std::exception);
  }
}

TEST_F(VulkanAPITest, tile_invalid_outpus_exceptions) {
  // Arrange: Vulkan tile only supports output of dims <= 4
  {
    const auto in_cpu =
        at::rand({3, 9, 5, 13}, at::device(at::kCPU).dtype(at::kFloat));
    const at::IntArrayRef repeats = {5, 7, 3, 9, 2};

    // Act
    EXPECT_THROW(
        { const auto out_vulkan = at::tile(in_cpu.vulkan(), repeats); },
        ::std::exception);
  }
}

void test_tile(
    const at::IntArrayRef input_shape,
    const at::IntArrayRef repeats) {
  c10::InferenceMode mode;

  at::Tensor in_cpu;
  at::Tensor out_cpu;
  at::Tensor in_vulkan;
  at::Tensor out_vulkan;
  at::IntArrayRef repeat;
  bool check = true;
  for (int idx_input = 1; (unsigned)idx_input < input_shape.size() + 1; ++idx_input) {
    for (int idx_repeat = 1; (unsigned)idx_repeat < repeats.size() + 1; ++idx_repeat) {
      in_cpu = at::rand(
          input_shape.slice(0, idx_input),
          at::device(at::kCPU).dtype(at::kFloat));
      repeat = repeats.slice(0, idx_repeat);
      out_cpu = at::tile(in_cpu, repeat);
      in_vulkan = in_cpu.vulkan();
      out_vulkan = at::tile(in_vulkan, repeat);
      check = almostEqual(out_cpu, out_vulkan.cpu());
      if (!check) {
        check = false;
        std::cout << "Tile test failed when input is of shape "
                  << input_shape.slice(0, idx_input) << " and repeat of "
                  << repeat << std::endl;
        showRtol(out_cpu, out_vulkan.cpu());
      }
    }
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, tile) {
  test_tile({13, 5, 13, 7}, {7, 2, 3, 5});
}

void test_zero_(const at::IntArrayRef input_shape) {
  auto cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  auto vulkan = cpu.vulkan();

  cpu.zero_();
  vulkan.zero_();

  const auto check = almostEqual(cpu, vulkan.cpu());
  if (!check) {
    showRtol(cpu, vulkan.cpu());
    std::cout << "zero_ test failed with input shape: "
              << input_shape << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, zero_) {
  test_zero_({5});
  test_zero_({5, 7});
  test_zero_({9, 7, 5});
  test_zero_({22, 11, 19, 17});
}

void test_zeros(const at::IntArrayRef input_shape) {
  auto cpu = at::zeros(input_shape);
  auto vulkan = at::zeros(input_shape, at::device(at::kVulkan));

  const auto check = almostEqual(cpu, vulkan.cpu());
  if (!check) {
    showRtol(cpu, vulkan.cpu());
    std::cout << "zeros test failed with input shape: "
              << input_shape << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, zeros) {
  test_zeros({5});
  test_zeros({5, 7});
  test_zeros({9, 7, 5});
  test_zeros({22, 11, 19, 17});
}

TEST_F(VulkanAPITest, clone_success) {
  // Arrange
  std::multimap<std::optional<c10::MemoryFormat>, std::vector<int64_t>> mem2sizes {
    {c10::MemoryFormat::Preserve, {2, 3, 5, 161}},    // 4D tensors with MemoryFormat::Preserve
    {c10::MemoryFormat::Contiguous, {2, 3, 5, 161}},  // 4D tensors with MemoryFormat::Contiguous
    {{}, {2, 3, 5, 161}},                             // 4D tensors with null
    {c10::MemoryFormat::Preserve, {3, 5, 161}},       // 3D tensors with MemoryFormat::Preserve
    {c10::MemoryFormat::Contiguous, {3, 5, 161}},     // 3D tensors with MemoryFormat::Contiguous
    {{}, {3, 5, 161}},                                // 3D tensors with null
    {c10::MemoryFormat::Preserve, {5, 161}},          // 2D tensors with MemoryFormat::Preserve
    {c10::MemoryFormat::Contiguous, {5, 161}},        // 2D tensors with MemoryFormat::Contiguous
    {{}, {5, 161}},                                   // 2D tensors with null
    {c10::MemoryFormat::Preserve, {161}},             // 1D tensors with MemoryFormat::Preserve
    {c10::MemoryFormat::Contiguous, {161}},           // 1D tensors with MemoryFormat::Contiguous
    {{}, {161}},                                      // 1D tensors with null
  };

  // Act/Assert
  for (const auto& mem2size : mem2sizes) {
    clone_test(mem2size.second, mem2size.first);
  }
}

TEST_F(VulkanAPITest, clone_invalidinputs_exceptions) {
  // Act: Vulkan supports Preserve and Contiguous memory formats
  EXPECT_THROW({
    clone_test({2, 3, 5, 161}, c10::MemoryFormat::ChannelsLast);
  }, ::std::exception);

  // Act: Vulkan supports Preserve and Contiguous memory formats
  EXPECT_THROW({
    clone_test({2, 3, 5, 161}, c10::MemoryFormat::ChannelsLast3d);
  }, ::std::exception);
}

enum class OpType {
  addmm,
  conv2d,
  hardtanh_,
  mean,
 };

class BaseOp {
 public:
  explicit BaseOp(const OpType) {}
  virtual ~BaseOp() = default;

  virtual at::Tensor run(at::Tensor&) const = 0;
  virtual std::string toString() const = 0;

};

class Addmm final : public BaseOp {
 public:
  Addmm(
      const int64_t m1H,
      const int64_t m1W,
      const int64_t m2W,
      const float beta,
      const float alpha)
    : BaseOp(OpType::addmm),
      m2_(at::rand(c10::IntArrayRef({m1W, m2W}), at::device(at::kCPU).dtype(at::kFloat))),
      b_(at::rand(c10::IntArrayRef({m1H, m2W}), at::device(at::kCPU).dtype(at::kFloat))),
      beta_(beta),
      alpha_(alpha) {
  }

  at::Tensor run(at::Tensor& t) const override {
    if (t.is_vulkan()) {
      return at::addmm(b_, t, m2_, beta_, alpha_);
    }

    return at::addmm(b_, t, m2_, beta_, alpha_);
  }

  std::string toString() const override {
    return "addmm";
  }

 private:
  at::Tensor m2_;
  at::Tensor b_;
  float beta_;
  float alpha_;
};

class Conv2d final : public BaseOp {
 public:
  Conv2d(
      const c10::IntArrayRef wsizes,
      const int64_t groups,
      const int64_t stride,
      const int64_t padding)
      : BaseOp(OpType::conv2d),
        groups_(groups),
        stride_(stride),
        padding_(padding),
        w_(at::rand(wsizes, at::device(at::kCPU).dtype(at::kFloat))),
        b_(at::rand(wsizes[0], at::device(at::kCPU).dtype(at::kFloat))){
  }

  at::Tensor run(at::Tensor& t) const override {
    return at::conv2d(t, w_, b_, {stride_}, {padding_}, {1}, groups_);
  }

  std::string toString() const override {
    return "conv2d";
  }

 private:
  int64_t groups_;
  int64_t stride_;
  int64_t padding_;
  at::Tensor w_;
  at::Tensor b_;
};

class Hardtanh_ final : public BaseOp {
 public:
  Hardtanh_() : BaseOp(OpType::hardtanh_) {}

  at::Tensor run(at::Tensor& input) const override {
    return at::hardtanh_(input, 0, 6);
  }

  std::string toString() const override {
    return "hardtanh_";
  }
};

class Mean final : public BaseOp {
 public:
  Mean() : BaseOp(OpType::mean) {}

  at::Tensor run(at::Tensor& input) const override {
    return at::mean(input, {2, 3}, false);
  }

  std::string toString() const override {
    return "mean";
  }
};

class OpsList {
 public:
  OpsList() {}
  explicit OpsList(std::vector<std::unique_ptr<BaseOp>> ops)
    : ops_(std::move(ops)) {
  }

  auto run(const at::Tensor& input) {
    at::Tensor output = input;

    for (const auto& op : ops_) {
      output = op->run(output);
    }

    return output;
  }

  auto run(const at::Tensor& input, const at::Tensor& v_input) {
    at::Tensor output = input;
    at::Tensor v_output = v_input;

    for (const auto& op : ops_) {
      output = op->run(output);
      v_output = op->run(v_output);
    }

    return std::make_pair(output, v_output);
  }

 protected:
  std::vector<std::unique_ptr<BaseOp>> ops_;
};

class MobileNetV2 final : public OpsList {
 public:
  MobileNetV2() {
    ops_.emplace_back(new Conv2d({32, 3, 3, 3}, 1, 2, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({32, 1, 3, 3}, 32, 1, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({16, 32, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({96, 16, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({96, 1, 3, 3}, 96, 2, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({24, 96, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({144, 24, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({144, 1, 3, 3}, 144, 1, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({24, 144, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({144, 24, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({144, 1, 3, 3}, 144, 2, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({32, 144, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({192, 32, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({192, 1, 3, 3}, 192, 1, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({32, 192, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({192, 32, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({192, 1, 3, 3}, 192, 1, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({32, 192, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({192, 32, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({192, 1, 3, 3}, 192, 2, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({64, 192, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({384, 64, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({384, 1, 3, 3}, 384, 1, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({64, 384, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({384, 64, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({384, 1, 3, 3}, 384, 1, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({64, 384, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({384, 64, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({384, 1, 3, 3}, 384, 1, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({64, 384, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({384, 64, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({384, 1, 3, 3}, 384, 1, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({96, 384, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({576, 96, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({576, 1, 3, 3}, 576, 1, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({96, 576, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({576, 96, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({576, 1, 3, 3}, 576, 1, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({96, 576, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({576, 96, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({576, 1, 3, 3}, 576, 2, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({160, 576, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({960, 160, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({960, 1, 3, 3}, 960, 1, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({160, 960, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({960, 160, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({960, 1, 3, 3}, 960, 1, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({160, 960, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({960, 160, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({960, 1, 3, 3}, 960, 1, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({320, 960, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({1280, 320, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Mean());
    ops_.emplace_back(new Addmm(1, 1280, 1000, 0, 1));
  }
};

TEST_F(VulkanAPITest, mobilenetv2) {
  c10::InferenceMode mode;

  MobileNetV2 mn2;

  const auto input = at::rand({1, 3, 224, 224}, at::device(at::kCPU).dtype(at::kFloat));
  const auto output = mn2.run(input, input.vulkan());

  const auto check = almostEqual(output.first, output.second.cpu());
  if (!check) {
    showRtol(output.first, output.second.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, gru_success) {
  // Arrange
  const int H_in = 5;  // input_size
  const int H_out = 7; // hidden_size
  const int num_layers = 3;
  const int L = 1;
  const int N = 1;
  const double gru_dropout = .0;
  const bool has_biases = true;
  const bool train = false;
  const bool bidirectional = false;
  const bool batch_first = true;
  const auto in_cpu = at::rand({N, L, H_in}, at::device(at::kCPU).dtype(at::kFloat));
  const auto h0_cpu = at::rand({num_layers, N, H_out}, at::device(at::kCPU).dtype(at::kFloat));

  c10::List<at::Tensor> weight_ih_l; // shape (3 * hidden_size, input_size)
  c10::List<at::Tensor> weight_hh_l; // shape (3 * hidden_size, hidden_size)
  c10::List<at::Tensor> bias_ih_l;   // shape (3 * hidden_size)
  c10::List<at::Tensor> bias_hh_l;   // shape (3 * hidden_size)
  for (int i = 0; i < num_layers; ++i) {
    if (i == 0) {
      weight_ih_l.emplace_back(at::rand({3 * H_out, H_in}, at::device(at::kCPU).dtype(at::kFloat)));
    } else {
      weight_ih_l.emplace_back(at::rand({3 * H_out, H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    }
    weight_hh_l.emplace_back(at::rand({3 * H_out, H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    bias_ih_l.emplace_back(at::rand({3 * H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    bias_hh_l.emplace_back(at::rand({3 * H_out}, at::device(at::kCPU).dtype(at::kFloat)));
  }

  // put this guard here to run inference inststead of training
  // to avoid the following error:
  //     C++ exception with description "0INTERNAL ASSERT FAILED at "xplat/caffe2/aten/src/ATen/core/boxing/KernelFunction.cpp":31, please report a bug to PyTorch. aten::gru.input has kernels registered to both CompositeImplicitAutograd and a backend mapped to AutogradOther. This makes the backend kernel unreachable; the dispatcher will always prefer the CompositeImplicitAutograd lowering (see Note [Ambiguity in AutogradOther kernel]). If you want to override CompositeImplicitAutograd, please open an issue to request a dedicated Autograd dispatch key for the backend.
  //     If you only want to run inference instead of training, add `c10::InferenceMode mode;` before model.forward(). Note this guard is only available in C++ but not Python at present.
  c10::InferenceMode mode;

  // Act
  const auto out_cpu = at::gru(in_cpu, h0_cpu,
      { weight_ih_l[0], weight_hh_l[0], bias_ih_l[0], bias_hh_l[0],
        weight_ih_l[1], weight_hh_l[1], bias_ih_l[1], bias_hh_l[1],
        weight_ih_l[2], weight_hh_l[2], bias_ih_l[2], bias_hh_l[2] },
      has_biases, num_layers, gru_dropout, train, bidirectional, batch_first);

  // weights/biases should be always on CPU.
  const auto out_vulkan = at::gru(in_cpu.vulkan(), h0_cpu.vulkan(),
      { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
        weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1),
        weight_ih_l.get(2), weight_hh_l.get(2), bias_ih_l.get(2), bias_hh_l.get(2) },
      has_biases, num_layers, gru_dropout, train, bidirectional, batch_first);

  auto cpu_output = std::get<0>(out_cpu);
  auto cpu_hidden = std::get<1>(out_cpu);
  auto vulkan_output = std::get<0>(out_vulkan);
  auto vulkan_hidden = std::get<1>(out_vulkan);

  // Assert
  const auto check_output = almostEqual(cpu_output, vulkan_output.cpu());
  if (!check_output) {
    showRtol(cpu_output, vulkan_output.cpu());
  }
  ASSERT_TRUE(check_output);

  const auto check_hidden = almostEqual(cpu_hidden, vulkan_hidden.cpu());
  if (!check_hidden) {
    showRtol(cpu_hidden, vulkan_hidden.cpu());
  }
  ASSERT_TRUE(check_hidden);
}

TEST_F(VulkanAPITest, gru_mclareninputs_success) {
  // Arrange
  const int H_in = 384;  // input_size
  const int H_out = 384; // hidden_size
  const int num_layers = 2;
  const int L = 1;
  const int N = 1;
  const double gru_dropout = .0;
  const bool has_biases = true;
  const bool train = false;
  const bool bidirectional = false;
  const bool batch_first = true;
  const auto in_cpu = at::rand({N, L, H_in}, at::device(at::kCPU).dtype(at::kFloat));
  const auto h0_cpu = at::rand({num_layers, N, H_out}, at::device(at::kCPU).dtype(at::kFloat));

  c10::List<at::Tensor> weight_ih_l; // shape (3 * hidden_size, input_size)
  c10::List<at::Tensor> weight_hh_l; // shape (3 * hidden_size, hidden_size)
  c10::List<at::Tensor> bias_ih_l;   // shape (3 * hidden_size)
  c10::List<at::Tensor> bias_hh_l;   // shape (3 * hidden_size)
  for (int i = 0; i < num_layers; ++i) {
    if (i == 0) {
      weight_ih_l.emplace_back(at::rand({3 * H_out, H_in}, at::device(at::kCPU).dtype(at::kFloat)));
    } else {
      weight_ih_l.emplace_back(at::rand({3 * H_out, H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    }
    weight_hh_l.emplace_back(at::rand({3 * H_out, H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    bias_ih_l.emplace_back(at::rand({3 * H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    bias_hh_l.emplace_back(at::rand({3 * H_out}, at::device(at::kCPU).dtype(at::kFloat)));
  }

  // put this guard here to run inference inststead of training
  // to avoid the following error:
  //     C++ exception with description "0INTERNAL ASSERT FAILED at "xplat/caffe2/aten/src/ATen/core/boxing/KernelFunction.cpp":31, please report a bug to PyTorch. aten::gru.input has kernels registered to both CompositeImplicitAutograd and a backend mapped to AutogradOther. This makes the backend kernel unreachable; the dispatcher will always prefer the CompositeImplicitAutograd lowering (see Note [Ambiguity in AutogradOther kernel]). If you want to override CompositeImplicitAutograd, please open an issue to request a dedicated Autograd dispatch key for the backend.
  //     If you only want to run inference instead of training, add `c10::InferenceMode mode;` before model.forward(). Note this guard is only available in C++ but not Python at present.
  c10::InferenceMode mode;

  // Act
  const auto out_cpu = at::gru(in_cpu, h0_cpu,
      { weight_ih_l[0], weight_hh_l[0], bias_ih_l[0], bias_hh_l[0], weight_ih_l[1], weight_hh_l[1], bias_ih_l[1], bias_hh_l[1] },
      has_biases, num_layers, gru_dropout, train, bidirectional, batch_first);

  // weights/biases should be always on CPU.
  const auto out_vulkan = at::gru(in_cpu.vulkan(), h0_cpu.vulkan(), { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
      weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) },
      has_biases, num_layers, gru_dropout, train, bidirectional, batch_first);

  auto cpu_output = std::get<0>(out_cpu);
  auto cpu_hidden = std::get<1>(out_cpu);
  auto vulkan_output = std::get<0>(out_vulkan);
  auto vulkan_hidden = std::get<1>(out_vulkan);

  // Assert
  const auto check_output = almostEqual(cpu_output, vulkan_output.cpu());
  if (!check_output) {
    showRtol(cpu_output, vulkan_output.cpu());
  }
  ASSERT_TRUE(check_output);

  const auto check_hidden = almostEqual(cpu_hidden, vulkan_hidden.cpu());
  if (!check_hidden) {
    showRtol(cpu_hidden, vulkan_hidden.cpu());
  }
  ASSERT_TRUE(check_hidden);
}

TEST_F(VulkanAPITest, gru_invalidinputs_exceptions) {
  // Arrange
  const int H_in = 17;  // input_size
  const int H_out = 50; // hidden_size
  const int num_layers = 2;
  const int L = 5;
  const int N = 4;
  const double gru_dropout = .0;
  const bool has_biases = true;
  const bool train = false;
  const bool bidirectional = false;
  const bool batch_first = true;
  const auto in_cpu = at::rand({N, L, H_in}, at::device(at::kCPU).dtype(at::kFloat));
  const auto h0_cpu = at::rand({num_layers, N, H_out}, at::device(at::kCPU).dtype(at::kFloat));

  c10::List<at::Tensor> weight_ih_l; // shape (3 * hidden_size, input_size)
  c10::List<at::Tensor> weight_hh_l; // shape (3 * hidden_size, hidden_size)
  c10::List<at::Tensor> bias_ih_l;   // shape (3 * hidden_size)
  c10::List<at::Tensor> bias_hh_l;   // shape (3 * hidden_size)
  for (int i = 0; i < num_layers; ++i) {
    if (i == 0) {
      weight_ih_l.emplace_back(at::rand({3 * H_out, H_in}, at::device(at::kCPU).dtype(at::kFloat)));
    } else {
      weight_ih_l.emplace_back(at::rand({3 * H_out, H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    }
    weight_hh_l.emplace_back(at::rand({3 * H_out, H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    bias_ih_l.emplace_back(at::rand({3 * H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    bias_hh_l.emplace_back(at::rand({3 * H_out}, at::device(at::kCPU).dtype(at::kFloat)));
  }

  // put this guard here to run inference inststead of training
  // to avoid the following error:
  //     C++ exception with description "0INTERNAL ASSERT FAILED at "xplat/caffe2/aten/src/ATen/core/boxing/KernelFunction.cpp":31, please report a bug to PyTorch. aten::gru.input has kernels registered to both CompositeImplicitAutograd and a backend mapped to AutogradOther. This makes the backend kernel unreachable; the dispatcher will always prefer the CompositeImplicitAutograd lowering (see Note [Ambiguity in AutogradOther kernel]). If you want to override CompositeImplicitAutograd, please open an issue to request a dedicated Autograd dispatch key for the backend.
  //     If you only want to run inference instead of training, add `c10::InferenceMode mode;` before model.forward(). Note this guard is only available in C++ but not Python at present.
  c10::InferenceMode mode;

  // Act: incorrect # of weights/biases
  EXPECT_THROW({
    at::gru(in_cpu.vulkan(), h0_cpu.vulkan(), { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
      weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1) },
      has_biases, num_layers, gru_dropout, train, bidirectional, batch_first);
  }, ::std::exception);

  // Act: non-3D input tensor
  EXPECT_THROW({
    const auto in_cpu_2d = at::rand({1, H_in}, at::device(at::kCPU).dtype(at::kFloat));
    at::gru(in_cpu_2d.vulkan(), h0_cpu.vulkan(), { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
      weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) },
      has_biases, num_layers, gru_dropout, train, bidirectional, batch_first);
  }, ::std::exception);

  // Act: non-3D hidden tensor
  EXPECT_THROW({
    const auto h0_cpu_2d = at::rand({num_layers, H_out}, at::device(at::kCPU).dtype(at::kFloat));
    at::gru(in_cpu.vulkan(), h0_cpu_2d.vulkan(), { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
      weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) },
      has_biases, num_layers, gru_dropout, train, bidirectional, batch_first);
  }, ::std::exception);

  // Act: has_biases should be true
  EXPECT_THROW({
    at::gru(in_cpu.vulkan(), h0_cpu.vulkan(), { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
      weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) },
      false, num_layers, gru_dropout, train, bidirectional, batch_first);
  }, ::std::exception);

  // Act: train should be false
  EXPECT_THROW({
    at::gru(in_cpu.vulkan(), h0_cpu.vulkan(), { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
      weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) },
      has_biases, num_layers, gru_dropout, true, bidirectional, batch_first);
  }, ::std::exception);

  // Act: bidirectional should be false
  EXPECT_THROW({
    at::gru(in_cpu.vulkan(), h0_cpu.vulkan(), { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
      weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) },
      has_biases, num_layers, gru_dropout, train, true, batch_first);
  }, ::std::exception);

  // Act: batch_first should be true
  EXPECT_THROW({
    at::gru(in_cpu.vulkan(), h0_cpu.vulkan(), { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
      weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) },
      has_biases, num_layers, gru_dropout, train, bidirectional, false);
  }, ::std::exception);

  // Act: dropout should be 0.0
  EXPECT_THROW({
    at::gru(in_cpu.vulkan(), h0_cpu.vulkan(), { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
      weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) },
      has_biases, num_layers, 1.0, train, bidirectional, batch_first);
  }, ::std::exception);
}

TEST_F(VulkanAPITest, gru_prepack_success) {
  // Arrange
  const int H_in = 81;  // input_size
  const int H_out = 10; // hidden_size
  const int num_layers = 2;
  const int L = 1;
  const int N = 1;
  const double gru_dropout = .0;
  const bool has_biases = true;
  const bool train = false;
  const bool bidirectional = false;
  const bool batch_first = true;
  const auto in_cpu = at::rand({N, L, H_in}, at::device(at::kCPU).dtype(at::kFloat));
  const auto h0_cpu = at::rand({num_layers, N, H_out}, at::device(at::kCPU).dtype(at::kFloat));

  c10::List<at::Tensor> weight_ih_l; // shape (3 * hidden_size, input_size)
  c10::List<at::Tensor> weight_hh_l; // shape (3 * hidden_size, hidden_size)
  c10::List<at::Tensor> bias_ih_l;   // shape (3 * hidden_size)
  c10::List<at::Tensor> bias_hh_l;   // shape (3 * hidden_size)
  for (int i = 0; i < num_layers; ++i) {
    if (i == 0) {
      weight_ih_l.emplace_back(at::rand({3 * H_out, H_in}, at::device(at::kCPU).dtype(at::kFloat)));
    } else {
      weight_ih_l.emplace_back(at::rand({3 * H_out, H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    }
    weight_hh_l.emplace_back(at::rand({3 * H_out, H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    bias_ih_l.emplace_back(at::rand({3 * H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    bias_hh_l.emplace_back(at::rand({3 * H_out}, at::device(at::kCPU).dtype(at::kFloat)));
  }

  // put this guard here to run inference inststead of training
  // to avoid the following error:
  //     C++ exception with description "0INTERNAL ASSERT FAILED at "xplat/caffe2/aten/src/ATen/core/boxing/KernelFunction.cpp":31, please report a bug to PyTorch. aten::gru.input has kernels registered to both CompositeImplicitAutograd and a backend mapped to AutogradOther. This makes the backend kernel unreachable; the dispatcher will always prefer the CompositeImplicitAutograd lowering (see Note [Ambiguity in AutogradOther kernel]). If you want to override CompositeImplicitAutograd, please open an issue to request a dedicated Autograd dispatch key for the backend.
  //     If you only want to run inference instead of training, add `c10::InferenceMode mode;` before model.forward(). Note this guard is only available in C++ but not Python at present.
  c10::InferenceMode mode;

  // Act
  const auto out_cpu = at::gru(in_cpu, h0_cpu,
      { weight_ih_l[0], weight_hh_l[0], bias_ih_l[0], bias_hh_l[0], weight_ih_l[1], weight_hh_l[1], bias_ih_l[1], bias_hh_l[1] },
      has_biases, num_layers, gru_dropout, train, bidirectional, batch_first);

  auto prepack = callOpByName(
      "vulkan_prepack::create_gru_context",
      "",
      std::vector<at::Tensor>({ weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
        weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) }),
      has_biases, num_layers, gru_dropout, train, bidirectional, batch_first);
  auto out_vulkan = callOpByName(
      "vulkan_prepack::run_gru_context",
      "",
      in_cpu.vulkan(), h0_cpu.vulkan(), prepack[0]);

  auto cpu_output = std::get<0>(out_cpu);
  auto cpu_hidden = std::get<1>(out_cpu);
  auto vulkan_output = out_vulkan[0].toTensor();
  auto vulkan_hidden = out_vulkan[1].toTensor();

  // Assert
  const auto check_output = almostEqual(cpu_output, vulkan_output.cpu());
  if (!check_output) {
    showRtol(cpu_output, vulkan_output.cpu());
  }
  ASSERT_TRUE(check_output);

  const auto check_hidden = almostEqual(cpu_hidden, vulkan_hidden.cpu());
  if (!check_hidden) {
    showRtol(cpu_hidden, vulkan_hidden.cpu());
  }
  ASSERT_TRUE(check_hidden);
}

TEST_F(VulkanAPITest, gru_prepack_invalidinputs_exceptions) {
  // Arrange
  const int H_in = 70;  // input_size
  const int H_out = 2; // hidden_size
  const int num_layers = 2;
  const int L = 3;
  const int N = 5;
  const double gru_dropout = .0;
  const bool has_biases = true;
  const bool train = false;
  const bool bidirectional = false;
  const bool batch_first = true;
  const auto in_cpu = at::rand({N, L, H_in}, at::device(at::kCPU).dtype(at::kFloat));
  const auto h0_cpu = at::rand({num_layers, N, H_out}, at::device(at::kCPU).dtype(at::kFloat));

  c10::List<at::Tensor> weight_ih_l; // shape (3 * hidden_size, input_size)
  c10::List<at::Tensor> weight_hh_l; // shape (3 * hidden_size, hidden_size)
  c10::List<at::Tensor> bias_ih_l;   // shape (3 * hidden_size)
  c10::List<at::Tensor> bias_hh_l;   // shape (3 * hidden_size)
  for (int i = 0; i < num_layers; ++i) {
    if (i == 0) {
      weight_ih_l.emplace_back(at::rand({3 * H_out, H_in}, at::device(at::kCPU).dtype(at::kFloat)));
    } else {
      weight_ih_l.emplace_back(at::rand({3 * H_out, H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    }
    weight_hh_l.emplace_back(at::rand({3 * H_out, H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    bias_ih_l.emplace_back(at::rand({3 * H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    bias_hh_l.emplace_back(at::rand({3 * H_out}, at::device(at::kCPU).dtype(at::kFloat)));
  }

  // put this guard here to run inference inststead of training
  // to avoid the following error:
  //     C++ exception with description "0INTERNAL ASSERT FAILED at "xplat/caffe2/aten/src/ATen/core/boxing/KernelFunction.cpp":31, please report a bug to PyTorch. aten::gru.input has kernels registered to both CompositeImplicitAutograd and a backend mapped to AutogradOther. This makes the backend kernel unreachable; the dispatcher will always prefer the CompositeImplicitAutograd lowering (see Note [Ambiguity in AutogradOther kernel]). If you want to override CompositeImplicitAutograd, please open an issue to request a dedicated Autograd dispatch key for the backend.
  //     If you only want to run inference instead of training, add `c10::InferenceMode mode;` before model.forward(). Note this guard is only available in C++ but not Python at present.
  c10::InferenceMode mode;

  // Act: incorrect # of weights/biases
  EXPECT_THROW({
    auto prepack = callOpByName(
        "vulkan_prepack::create_gru_context",
        "",
        std::vector<at::Tensor>({ weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
            weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1) }),
        has_biases, num_layers, gru_dropout, train, bidirectional, batch_first);
  }, ::std::exception);

  // Act: non-3D input tensor
  EXPECT_THROW({
    const auto in_cpu_2d = at::rand({1, H_in}, at::device(at::kCPU).dtype(at::kFloat));
    auto prepack = callOpByName(
        "vulkan_prepack::create_gru_context",
        "",
        std::vector<at::Tensor>({ weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
            weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) }),
        has_biases, num_layers, gru_dropout, train, bidirectional, batch_first);
    auto out_vulkan = callOpByName(
        "vulkan_prepack::run_gru_context",
        "",
        in_cpu_2d.vulkan(), h0_cpu.vulkan(), prepack[0]);
  }, ::std::exception);

  // Act: non-3D hidden tensor
  EXPECT_THROW({
    const auto h0_cpu_2d = at::rand({num_layers, H_out}, at::device(at::kCPU).dtype(at::kFloat));
    auto prepack = callOpByName(
        "vulkan_prepack::create_gru_context",
        "",
        std::vector<at::Tensor>({ weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
            weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) }),
        has_biases, num_layers, gru_dropout, train, bidirectional, batch_first);
    auto out_vulkan = callOpByName(
        "vulkan_prepack::run_gru_context",
        "",
        in_cpu.vulkan(), h0_cpu_2d.vulkan(), prepack[0]);
  }, ::std::exception);

  // Act: has_biases should be true
  EXPECT_THROW({
    auto prepack = callOpByName(
        "vulkan_prepack::create_gru_context",
        "",
        std::vector<at::Tensor>({ weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
           weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) }),
        false, num_layers, gru_dropout, train, bidirectional, batch_first);
  }, ::std::exception);

  // Act: train should be false
  EXPECT_THROW({
    auto prepack = callOpByName(
        "vulkan_prepack::create_gru_context",
        "",
        std::vector<at::Tensor>({ weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
           weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) }),
        has_biases, num_layers, gru_dropout, true, bidirectional, batch_first);
  }, ::std::exception);

  // Act: bidirectional should be false
  EXPECT_THROW({
     auto prepack = callOpByName(
        "vulkan_prepack::create_gru_context",
        "",
        std::vector<at::Tensor>({ weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
           weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) }),
        has_biases, num_layers, gru_dropout, train, true, batch_first);
 }, ::std::exception);

  // Act: batch_first should be true
  EXPECT_THROW({
    auto prepack = callOpByName(
        "vulkan_prepack::create_gru_context",
        "",
        std::vector<at::Tensor>({ weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
           weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) }),
        has_biases, num_layers, gru_dropout, train, bidirectional, false);
    auto out_vulkan = callOpByName(
        "vulkan_prepack::run_gru_context",
        "",
        in_cpu.vulkan(), h0_cpu.vulkan(), prepack[0]);
  }, ::std::exception);

  // Act: dropout should be 0.0
  EXPECT_THROW({
    auto prepack = callOpByName(
        "vulkan_prepack::create_gru_context",
        "",
        std::vector<at::Tensor>({ weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
           weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) }),
        has_biases, num_layers, 1.0, train, bidirectional, batch_first);
  }, ::std::exception);
}

void test_linear(
    const at::IntArrayRef input_shape,
    const at::IntArrayRef weight_shape,
    const at::IntArrayRef bias_shape) {
  c10::InferenceMode mode;

  const auto input_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto weight = at::rand(weight_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto bias = at::rand(bias_shape, at::device(at::kCPU).dtype(at::kFloat));

  const auto out_cpu = at::linear(input_cpu, weight, bias);

  auto prepack = callOpByName(
      "vulkan_prepack::create_linear_context",
      "",
      weight.t(), bias);

  auto vulkan_output = callOpByName(
      "vulkan_prepack::run_linear_context",
      "",
      input_cpu.vulkan(), prepack[0]);

  auto out_vulkan = vulkan_output[0].toTensor();

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, linear_1d_small) {
  test_linear({3}, {4, 3}, {4});
}

TEST_F(VulkanAPITest, linear_1d_large) {
  test_linear({37}, {23, 37}, {23});
}

TEST_F(VulkanAPITest, linear_2d_flat) {
  test_linear({1, 37}, {41, 37}, {41});
}

TEST_F(VulkanAPITest, linear_2d_small) {
  test_linear({2, 3}, {4, 3}, {4});
}

TEST_F(VulkanAPITest, linear_2d_large) {
  test_linear({49, 37}, {23, 37}, {23});
}

TEST_F(VulkanAPITest, linear_3d_flat) {
  test_linear({1, 1, 37}, {41, 37}, {41});
}

TEST_F(VulkanAPITest, linear_3d_small) {
  test_linear({2, 3, 4}, {5, 4}, {5});
}

TEST_F(VulkanAPITest, linear_3d_large) {
  test_linear({23, 17, 41}, {15, 41}, {15});
}

TEST_F(VulkanAPITest, linear_4d_flat) {
  test_linear({1, 1, 1, 37}, {41, 37}, {41});
}

TEST_F(VulkanAPITest, linear_4d_small) {
  test_linear({2, 3, 4, 5}, {6, 5}, {6});
}

TEST_F(VulkanAPITest, linear_4d_large) {
  test_linear({9, 13, 11, 17}, {23, 17}, {23});
}

TEST_F(VulkanAPITest, lstm_success) {
  // Arrange
  const int input_size = 5;
  const int hidden_size = 7;
  const int num_layers = 4;
  const int L = 1;
  const int N = 1;
  const double lstm_dropout = .0;
  const bool has_biases = true;
  const bool train = false;
  const bool bidirectional = false;
  const bool batch_first = true;
  const auto in_cpu = at::rand({N, L, input_size}, at::device(at::kCPU).dtype(at::kFloat));
  const auto h0_cpu = at::rand({num_layers, N, hidden_size}, at::device(at::kCPU).dtype(at::kFloat));
  const auto c0_cpu = at::rand({num_layers, N, hidden_size}, at::device(at::kCPU).dtype(at::kFloat));

  c10::List<at::Tensor> weight_ih_l; // shape (4 * hidden_size, input_size)
  c10::List<at::Tensor> weight_hh_l; // shape (4 * hidden_size, hidden_size)
  c10::List<at::Tensor> bias_ih_l;   // shape (4 * hidden_size)
  c10::List<at::Tensor> bias_hh_l;   // shape (4 * hidden_size)
  for (int l = 0; l < num_layers; ++l) {
    if (l == 0) {
      weight_ih_l.emplace_back(at::rand({4 * hidden_size, input_size}, at::device(at::kCPU).dtype(at::kFloat)));
    } else {
      weight_ih_l.emplace_back(at::rand({4 * hidden_size, hidden_size}, at::device(at::kCPU).dtype(at::kFloat)));
    }
    weight_hh_l.emplace_back(at::rand({4 * hidden_size, hidden_size}, at::device(at::kCPU).dtype(at::kFloat)));
    bias_ih_l.emplace_back(at::rand({4 * hidden_size}, at::device(at::kCPU).dtype(at::kFloat)));
    bias_hh_l.emplace_back(at::rand({4 * hidden_size}, at::device(at::kCPU).dtype(at::kFloat)));
  }

  // put this guard here to run inference inststead of training
  // to avoid the following error:
  //     C++ exception with description "0INTERNAL ASSERT FAILED at "xplat/caffe2/aten/src/ATen/core/boxing/KernelFunction.cpp":31, please report a bug to PyTorch. aten::gru.input has kernels registered to both CompositeImplicitAutograd and a backend mapped to AutogradOther. This makes the backend kernel unreachable; the dispatcher will always prefer the CompositeImplicitAutograd lowering (see Note [Ambiguity in AutogradOther kernel]). If you want to override CompositeImplicitAutograd, please open an issue to request a dedicated Autograd dispatch key for the backend.
  //     If you only want to run inference instead of training, add `c10::InferenceMode mode;` before model.forward(). Note this guard is only available in C++ but not Python at present.
  c10::InferenceMode mode;

  // Act
  const auto out_cpu = at::lstm(in_cpu, {h0_cpu, c0_cpu},
      { weight_ih_l[0], weight_hh_l[0], bias_ih_l[0], bias_hh_l[0],
        weight_ih_l[1], weight_hh_l[1], bias_ih_l[1], bias_hh_l[1],
        weight_ih_l[2], weight_hh_l[2], bias_ih_l[2], bias_hh_l[2],
        weight_ih_l[3], weight_hh_l[3], bias_ih_l[3], bias_hh_l[3] },
      has_biases, num_layers, lstm_dropout, train, bidirectional, batch_first);

  // weights/biases should be always on CPU.
  const auto out_vulkan = at::lstm(in_cpu.vulkan(), {h0_cpu.vulkan(), c0_cpu.vulkan()},
      { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
        weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1),
        weight_ih_l.get(2), weight_hh_l.get(2), bias_ih_l.get(2), bias_hh_l.get(2),
        weight_ih_l.get(3), weight_hh_l.get(3), bias_ih_l.get(3), bias_hh_l.get(3) },
      has_biases, num_layers, lstm_dropout, train, bidirectional, batch_first);

  auto cpu_output = std::get<0>(out_cpu);
  auto cpu_hidden = std::get<1>(out_cpu);
  auto cpu_cell = std::get<2>(out_cpu);
  auto vulkan_output = std::get<0>(out_vulkan);
  auto vulkan_hidden = std::get<1>(out_vulkan);
  auto vulkan_cell = std::get<2>(out_vulkan);

  // Assert
  const auto check_output = almostEqual(cpu_output, vulkan_output.cpu());
  if (!check_output) {
    showRtol(cpu_output, vulkan_output.cpu());
  }
  ASSERT_TRUE(check_output);

  const auto check_hidden = almostEqual(cpu_hidden, vulkan_hidden.cpu());
  if (!check_hidden) {
    showRtol(cpu_hidden, vulkan_hidden.cpu());
  }
  ASSERT_TRUE(check_hidden);

  const auto check_cell = almostEqual(cpu_cell, vulkan_cell.cpu());
  if (!check_cell) {
    showRtol(cpu_cell, vulkan_cell.cpu());
  }
  ASSERT_TRUE(check_cell);
}

TEST_F(VulkanAPITest, lstm_mclareninputs_success) {
  // Arrange
  const int input_size = 384;
  const int hidden_size = 384;
  const int num_layers = 2;
  const int L = 1;
  const int N = 1;
  const double lstm_dropout = .0;
  const bool has_biases = true;
  const bool train = false;
  const bool bidirectional = false;
  const bool batch_first = true;
  const auto in_cpu = at::rand({N, L, input_size}, at::device(at::kCPU).dtype(at::kFloat));
  const auto h0_cpu = at::rand({num_layers, N, hidden_size}, at::device(at::kCPU).dtype(at::kFloat));
  const auto c0_cpu = at::rand({num_layers, N, hidden_size}, at::device(at::kCPU).dtype(at::kFloat));

  c10::List<at::Tensor> weight_ih_l; // shape (4 * hidden_size, input_size)
  c10::List<at::Tensor> weight_hh_l; // shape (4 * hidden_size, hidden_size)
  c10::List<at::Tensor> bias_ih_l;   // shape (4 * hidden_size)
  c10::List<at::Tensor> bias_hh_l;   // shape (4 * hidden_size)
  for (int l = 0; l < num_layers; ++l) {
    if (l == 0) {
      weight_ih_l.emplace_back(at::rand({4 * hidden_size, input_size}, at::device(at::kCPU).dtype(at::kFloat)));
    } else {
      weight_ih_l.emplace_back(at::rand({4 * hidden_size, hidden_size}, at::device(at::kCPU).dtype(at::kFloat)));
    }
    weight_hh_l.emplace_back(at::rand({4 * hidden_size, hidden_size}, at::device(at::kCPU).dtype(at::kFloat)));
    bias_ih_l.emplace_back(at::rand({4 * hidden_size}, at::device(at::kCPU).dtype(at::kFloat)));
    bias_hh_l.emplace_back(at::rand({4 * hidden_size}, at::device(at::kCPU).dtype(at::kFloat)));
  }

  // put this guard here to run inference inststead of training
  // to avoid the following error:
  //     C++ exception with description "0INTERNAL ASSERT FAILED at "xplat/caffe2/aten/src/ATen/core/boxing/KernelFunction.cpp":31, please report a bug to PyTorch. aten::gru.input has kernels registered to both CompositeImplicitAutograd and a backend mapped to AutogradOther. This makes the backend kernel unreachable; the dispatcher will always prefer the CompositeImplicitAutograd lowering (see Note [Ambiguity in AutogradOther kernel]). If you want to override CompositeImplicitAutograd, please open an issue to request a dedicated Autograd dispatch key for the backend.
  //     If you only want to run inference instead of training, add `c10::InferenceMode mode;` before model.forward(). Note this guard is only available in C++ but not Python at present.
  c10::InferenceMode mode;

  // Act
  const auto out_cpu = at::lstm(in_cpu, {h0_cpu, c0_cpu},
      { weight_ih_l[0], weight_hh_l[0], bias_ih_l[0], bias_hh_l[0],
        weight_ih_l[1], weight_hh_l[1], bias_ih_l[1], bias_hh_l[1] },
      has_biases, num_layers, lstm_dropout, train, bidirectional, batch_first);

  // weights/biases should be always on CPU.
  const auto out_vulkan = at::lstm(in_cpu.vulkan(), {h0_cpu.vulkan(), c0_cpu.vulkan()},
      { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
        weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) },
      has_biases, num_layers, lstm_dropout, train, bidirectional, batch_first);

  auto cpu_output = std::get<0>(out_cpu);
  auto cpu_hidden = std::get<1>(out_cpu);
  auto cpu_cell = std::get<2>(out_cpu);
  auto vulkan_output = std::get<0>(out_vulkan);
  auto vulkan_hidden = std::get<1>(out_vulkan);
  auto vulkan_cell = std::get<2>(out_vulkan);

  // Assert
  const auto check_output = almostEqual(cpu_output, vulkan_output.cpu());
  if (!check_output) {
    showRtol(cpu_output, vulkan_output.cpu());
  }
  ASSERT_TRUE(check_output);

  const auto check_hidden = almostEqual(cpu_hidden, vulkan_hidden.cpu());
  if (!check_hidden) {
    showRtol(cpu_hidden, vulkan_hidden.cpu());
  }
  ASSERT_TRUE(check_hidden);

  const auto check_cell = almostEqual(cpu_cell, vulkan_cell.cpu());
  if (!check_cell) {
    showRtol(cpu_cell, vulkan_cell.cpu());
  }
  ASSERT_TRUE(check_cell);
}

TEST_F(VulkanAPITest, lstm_prepack_success) {
  // Arrange
  const int input_size = 81;
  const int hidden_size = 10;
  const int num_layers = 2;
  const int L = 1;
  const int N = 1;
  const double lstm_dropout = .0;
  const bool has_biases = true;
  const bool train = false;
  const bool bidirectional = false;
  const bool batch_first = true;
  const auto in_cpu = at::rand({N, L, input_size}, at::device(at::kCPU).dtype(at::kFloat));
  const auto h0_cpu = at::rand({num_layers, N, hidden_size}, at::device(at::kCPU).dtype(at::kFloat));
  const auto c0_cpu = at::rand({num_layers, N, hidden_size}, at::device(at::kCPU).dtype(at::kFloat));

  c10::List<at::Tensor> weight_ih_l; // shape (4 * hidden_size, l == 0 ? input_size : hidden_size)
  c10::List<at::Tensor> weight_hh_l; // shape (4 * hidden_size, hidden_size)
  c10::List<at::Tensor> bias_ih_l;   // shape (4 * hidden_size)
  c10::List<at::Tensor> bias_hh_l;   // shape (4 * hidden_size)
  for (int l = 0; l < num_layers; ++l) {
    if (l == 0) {
      weight_ih_l.emplace_back(at::rand({4 * hidden_size, input_size}, at::device(at::kCPU).dtype(at::kFloat)));
    } else {
      weight_ih_l.emplace_back(at::rand({4 * hidden_size, hidden_size}, at::device(at::kCPU).dtype(at::kFloat)));
    }
    weight_hh_l.emplace_back(at::rand({4 * hidden_size, hidden_size}, at::device(at::kCPU).dtype(at::kFloat)));
    bias_ih_l.emplace_back(at::rand({4 * hidden_size}, at::device(at::kCPU).dtype(at::kFloat)));
    bias_hh_l.emplace_back(at::rand({4 * hidden_size}, at::device(at::kCPU).dtype(at::kFloat)));
  }

  // put this guard here to run inference inststead of training
  // to avoid the following error:
  //     C++ exception with description "0INTERNAL ASSERT FAILED at "xplat/caffe2/aten/src/ATen/core/boxing/KernelFunction.cpp":31, please report a bug to PyTorch. aten::gru.input has kernels registered to both CompositeImplicitAutograd and a backend mapped to AutogradOther. This makes the backend kernel unreachable; the dispatcher will always prefer the CompositeImplicitAutograd lowering (see Note [Ambiguity in AutogradOther kernel]). If you want to override CompositeImplicitAutograd, please open an issue to request a dedicated Autograd dispatch key for the backend.
  //     If you only want to run inference instead of training, add `c10::InferenceMode mode;` before model.forward(). Note this guard is only available in C++ but not Python at present.
  c10::InferenceMode mode;

  // Act
  const auto out_cpu = at::lstm(in_cpu, {h0_cpu, c0_cpu},
      { weight_ih_l[0], weight_hh_l[0], bias_ih_l[0], bias_hh_l[0],
        weight_ih_l[1], weight_hh_l[1], bias_ih_l[1], bias_hh_l[1] },
      has_biases, num_layers, lstm_dropout, train, bidirectional, batch_first);

  auto prepack = callOpByName(
      "vulkan_prepack::create_lstm_context",
      "",
      std::vector<at::Tensor>({ weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
                                weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) }),
      has_biases, num_layers, lstm_dropout, train, bidirectional, batch_first);

  auto out_vulkan = callOpByName(
      "vulkan_prepack::run_lstm_context",
      "",
      in_cpu.vulkan(), h0_cpu.vulkan(), c0_cpu.vulkan(), prepack[0]);

  auto cpu_output = std::get<0>(out_cpu);
  auto cpu_hidden = std::get<1>(out_cpu);
  auto cpu_cell = std::get<2>(out_cpu);
  auto vulkan_output = out_vulkan[0].toTensor();
  auto vulkan_hidden = out_vulkan[1].toTensor();
  auto vulkan_cell = out_vulkan[2].toTensor();

  // Assert
  const auto check_output = almostEqual(cpu_output, vulkan_output.cpu());
  if (!check_output) {
    showRtol(cpu_output, vulkan_output.cpu());
  }
  ASSERT_TRUE(check_output);

  const auto check_hidden = almostEqual(cpu_hidden, vulkan_hidden.cpu());
  if (!check_hidden) {
    showRtol(cpu_hidden, vulkan_hidden.cpu());
  }
  ASSERT_TRUE(check_hidden);

  const auto check_cell = almostEqual(cpu_cell, vulkan_cell.cpu());
  if (!check_cell) {
    showRtol(cpu_cell, vulkan_cell.cpu());
  }
  ASSERT_TRUE(check_cell);
}

TEST_F(VulkanAPITest, querypool_flushed_shader_log) {
#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  const bool op_profiling_enabled_initially =
      at::native::vulkan::api::context()->op_profiling_enabled();

  at::native::vulkan::api::context()->enable_op_profiling();

  const at::Tensor a_add_cpu =
      at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const at::Tensor a_add_vulkan = a_add_cpu.vulkan();

  const at::Tensor b_add_cpu =
      at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const at::Tensor b_add_vulkan = b_add_cpu.vulkan();

  at::add(a_add_vulkan, b_add_vulkan, 2.1f).cpu();

  at::native::vulkan::api::context()->querypool().extract_results();
  at::native::vulkan::api::context()->reset_querypool();

  const at::Tensor a_sub_cpu =
      at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const at::Tensor a_sub_vulkan = a_sub_cpu.vulkan();

  const at::Tensor b_sub_cpu =
      at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const at::Tensor b_sub_vulkan = b_sub_cpu.vulkan();

  at::sub(a_sub_vulkan, b_sub_vulkan, 2.1f).cpu();

  at::native::vulkan::api::context()->querypool().extract_results();
  at::native::vulkan::api::context()->reset_querypool();

  const at::Tensor a_mul_cpu =
      at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const at::Tensor a_mul_vulkan = a_mul_cpu.vulkan();

  const at::Tensor b_mul_cpu =
      at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const at::Tensor b_mul_vulkan = b_mul_cpu.vulkan();

  at::mul(a_mul_vulkan, b_mul_vulkan).cpu();

  /*
    The most recent shaders should be
    (-12) vulkan.nchw_to_image
    (-11) vulkan.nchw_to_image
    (-10) vulkan.add
    (-9)  vulkan.image_to_nchw

    (-8)  vulkan.nchw_to_image
    (-7)  vulkan.nchw_to_image
    (-6)  vulkan.sub
    (-5)  vulkan.image_to_nchw

    (-4)  vulkan.nchw_to_image
    (-3)  vulkan.nchw_to_image
    (-2)  vulkan.mul
    (-1)  vulkan.image_to_nchw
  */

  const size_t entry_count =
      at::native::vulkan::api::context()->querypool().shader_logs_entry_count();

  std::tuple<std::string, uint64_t> add_shader_details =
      at::native::vulkan::api::context()
          ->querypool()
          .get_shader_name_and_execution_duration_ns(entry_count - 10);
  std::tuple<std::string, uint64_t> sub_shader_details =
      at::native::vulkan::api::context()
          ->querypool()
          .get_shader_name_and_execution_duration_ns(entry_count - 6);
  std::tuple<std::string, uint64_t> mul_shader_details =
      at::native::vulkan::api::context()
          ->querypool()
          .get_shader_name_and_execution_duration_ns(entry_count - 2);

  EXPECT_EQ(std::get<0>(add_shader_details), "vulkan.add");
  EXPECT_EQ(std::get<0>(sub_shader_details), "vulkan.sub");
  EXPECT_EQ(std::get<0>(mul_shader_details), "vulkan.mul");

  if (!op_profiling_enabled_initially) {
    at::native::vulkan::api::context()->reset_querypool();
    at::native::vulkan::api::context()->disable_op_profiling();
  }
#else
  GTEST_SKIP() << "QueryPool is not available";
#endif
}

} // namespace

#endif /* USE_VULKAN_API */
