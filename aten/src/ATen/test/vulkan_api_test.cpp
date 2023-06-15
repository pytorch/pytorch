#ifdef USE_VULKAN_API

#include <gtest/gtest.h>
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/vulkan/api/api.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/Convolution.h>
#include <c10/util/irange.h>

// TODO: These functions should move to a common place.

namespace {

#ifdef USE_VULKAN_FP16_INFERENCE
  constexpr float kTolerance = 1e-2;
#else
  constexpr float kTolerance = 1e-5;
#endif

bool checkRtol(const at::Tensor& diff, const std::vector<at::Tensor>& inputs) {
  float maxValue = 0.0f;

  for (const auto& tensor : inputs) {
    maxValue = fmax(tensor.abs().max().item<float>(), maxValue);
  }

  return diff.abs().max().item<float>() <= (kTolerance * maxValue);
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

static void slice_test(const std::vector<int64_t>& size, int64_t dim, c10::optional<int64_t> start, c10::optional<int64_t> end, int64_t step) {
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

static void clone_test(const std::vector<int64_t>& size, c10::optional<at::MemoryFormat> optional_memory_format) {
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
  const c10::optional<c10::OperatorHandle> op_handle =
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

TEST_F(VulkanAPITest, add) {
  const auto a_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::add(a_cpu, b_cpu, 2.1f);
  const auto c_vulkan = at::add(a_vulkan, b_vulkan, 2.1f);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, add_broadcast0) {
  const auto a_cpu = at::rand({3, 5, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({3, 5, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::add(a_cpu, b_cpu, 1.8f);
  const auto c_vulkan = at::add(a_vulkan, b_vulkan, 1.8f);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, add_broadcast1) {
  const auto a_cpu = at::rand({3, 5, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({3, 5, 1, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::add(a_cpu, b_cpu, 1.8f);
  const auto c_vulkan = at::add(a_vulkan, b_vulkan, 1.8f);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, add_broadcast2) {

  const auto a_cpu = at::rand({3, 4, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({4, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::add(a_cpu, b_cpu, 2.5f);
  const auto c_vulkan = at::add(a_vulkan, b_vulkan, 2.5f);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, add_broadcast3) {

  const auto a_cpu = at::rand({3, 4, 41, 53}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({1, 1, 41, 53}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::add(a_cpu, b_cpu, 2.5f);
  const auto c_vulkan = at::add(a_vulkan, b_vulkan, 2.5f);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, add_broadcast4) {
  const auto a_cpu = at::rand({3, 4, 41, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({1, 41, 53}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::add(a_cpu, b_cpu, 2.5f);
  const auto c_vulkan = at::add(a_vulkan, b_vulkan, 2.5f);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
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
    showRtol(b_cpu, b_vulkan.cpu());
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
    showRtol(b_cpu, b_vulkan.cpu());
  }

  ASSERT_TRUE(check);
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

TEST_F(VulkanAPITest, batch_norm_invalid_inputs) {
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
  }, ::c10::Error);

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
  }, ::c10::Error);

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
  }, ::c10::Error);

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
  }, ::c10::Error);

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
  }, ::c10::Error);

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
  }, ::c10::Error);

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
  }, ::c10::Error);

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
  }, ::c10::Error);
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
      weight, bias, stride, padding, dilation, groups, c10::nullopt, c10::nullopt);

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
      weight, bias, stride, padding, dilation, groups, c10::nullopt, c10::nullopt);

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
      weight, bias, stride, padding, output_padding, dilation, groups, c10::nullopt, c10::nullopt);

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

TEST_F(VulkanAPITest, conv2d_pw_prepack) {
  test_conv2d_context(
    {1, 17, 127, 397},  // input_shape
    {29, 17, 1, 1},     // weight_shape
    {29},               // bias_shape
    {1, 1},             // stride
    {0, 0},             // padding
    {1, 1},             // dilation
    1);                 // groups
}

TEST_F(VulkanAPITest, conv2d_pw_prepack_bc) {
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
      weight, bias, stride, padding, dilation, groups, 0.0f, c10::nullopt)[0];

  const auto out_cpu = callOpByName(
      "prepacked::conv2d_clamp_run",
      "",
      input_cpu, prepack_cpu)[0].toTensor();

  // vulkan
  const auto prepack_vk = callOpByName(
      "vulkan_prepack::create_conv2d_context",
      "",
      weight, bias, stride, padding, dilation, groups, 0.0f, c10::nullopt)[0];

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

TEST_F(VulkanAPITest, cumsum) {
  c10::InferenceMode mode;

  const auto in_cpu = at::rand({1, 17, 37, 49}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  // 0 do nothing
  // 1 frame
  // not implemented

  // 2 height
  const auto out_cpu2 = at::cumsum(in_cpu, 2);
  const auto out_vulkan2 = at::cumsum(in_cpu.vulkan(), 2);
  const auto check2 = almostEqual(out_cpu2, out_vulkan2.cpu());
  if (!check2) {
    showRtol(out_cpu2, out_vulkan2.cpu());
  }
  ASSERT_TRUE(check2);

  // 3 width
  const auto out_cpu3 = at::cumsum(in_cpu, 3);
  const auto out_vulkan3 = at::cumsum(in_cpu.vulkan(), 3);
  const auto check3 = almostEqual(out_cpu3, out_vulkan3.cpu());
  if (!check3) {
    showRtol(out_cpu3, out_vulkan3.cpu());
  }
  ASSERT_TRUE(check3);
}

TEST_F(VulkanAPITest, div) {
  const auto a_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat)) + 0.01;
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::div(a_cpu, b_cpu);
  const auto c_vulkan = at::div(a_vulkan, b_vulkan);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, div_broadcast0) {
  const auto a_cpu = at::rand({3, 5, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({3, 5, 179, 221}, at::device(at::kCPU).dtype(at::kFloat)) + 0.01;
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::div(a_cpu, b_cpu);
  const auto c_vulkan = at::div(a_vulkan, b_vulkan);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, div_broadcast1) {
  const auto a_cpu = at::rand({3, 5, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({3, 5, 1, 221}, at::device(at::kCPU).dtype(at::kFloat)) + 0.01;
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::div(a_cpu, b_cpu);
  const auto c_vulkan = at::div(a_vulkan, b_vulkan);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, div_broadcast2) {
  const auto a_cpu = at::rand({3, 4, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({4, 1, 1}, at::device(at::kCPU).dtype(at::kFloat)) + 0.01;
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::div(a_cpu, b_cpu);
  const auto c_vulkan = at::div(a_vulkan, b_vulkan);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, div_broadcast3) {
  const auto a_cpu = at::rand({3, 4, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({1, 1, 179, 221}, at::device(at::kCPU).dtype(at::kFloat)) + 0.01;
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::div(a_cpu, b_cpu);
  const auto c_vulkan = at::div(a_vulkan, b_vulkan);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, div_broadcast4) {
  const auto a_cpu = at::rand({3, 4, 41, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({1, 41, 53}, at::device(at::kCPU).dtype(at::kFloat)) + 0.01;
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::div(a_cpu, b_cpu);
  const auto c_vulkan = at::div(a_vulkan, b_vulkan);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
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
  }, ::c10::Error);

  // Act: normalized shape must be [C, H, W]
  EXPECT_THROW({
    at::layer_norm(
      at::rand({3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      {5, 7},
      at::rand({5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      1e-05,
      false);
  }, ::c10::Error);

  // Act: incorrect weight dimensions
  EXPECT_THROW({
    at::layer_norm(
      at::rand({3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      {3, 5, 7},
      at::rand({3, 5}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      1e-05,
      false);
  }, ::c10::Error);

  // Act: incorrect bias dimensions
  EXPECT_THROW({
    at::layer_norm(
      at::rand({3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      {3, 5, 7},
      at::rand({3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      1e-05,
      false);
  }, ::c10::Error);

  // Act: batch dim must be 1
  EXPECT_THROW({
    at::layer_norm(
      at::rand({2, 3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      {3, 5, 7},
      at::rand({3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      1e-05,
      false);
  }, ::c10::Error);

  // Act: input has too many dimensions
  EXPECT_THROW({
    at::layer_norm(
      at::rand({1, 2, 3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      {3, 5, 7},
      at::rand({3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      1e-05,
      false);
  }, ::c10::Error);

  // Act: input has too few dimensions
  EXPECT_THROW({
    at::layer_norm(
      at::rand({3, 5}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      {3, 5},
      at::rand({3, 5}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({3, 5}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      1e-05,
      false);
  }, ::c10::Error);
}

TEST_F(VulkanAPITest, layer_norm_3d_small) {
  c10::InferenceMode mode;

  const auto input_cpu = at::rand({1, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto input_vulkan = input_cpu.vulkan();

  const auto weight_cpu = at::rand({1, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto weight_vulkan = weight_cpu.vulkan();

  const auto bias_cpu = at::rand({1, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_vulkan = bias_cpu.vulkan();

  const auto output_cpu = at::layer_norm(input_cpu, {1, 1, 1}, weight_cpu, bias_cpu, 1e-05, false);
  const auto output_vulkan = at::layer_norm(input_vulkan, {1, 1, 1}, weight_vulkan, bias_vulkan, 1e-05, false);

  const auto check = almostEqual(output_cpu, output_vulkan.cpu());
  if (!check) {
    showRtol(output_cpu, output_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, layer_norm_3d_medium) {
  c10::InferenceMode mode;

  const auto input_cpu = at::rand({3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat));
  const auto input_vulkan = input_cpu.vulkan();

  const auto weight_cpu = at::rand({3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat));
  const auto weight_vulkan = weight_cpu.vulkan();

  const auto bias_cpu = at::rand({3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_vulkan = bias_cpu.vulkan();

  const auto output_cpu = at::layer_norm(input_cpu, {3, 5, 7}, weight_cpu, bias_cpu, 1e-05, false);
  const auto output_vulkan = at::layer_norm(input_vulkan, {3, 5, 7}, weight_vulkan, bias_vulkan, 1e-05, false);

  const auto check = almostEqual(output_cpu, output_vulkan.cpu());
  if (!check) {
    showRtol(output_cpu, output_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, layer_norm_3d_large) {
  c10::InferenceMode mode;

  const auto input_cpu = at::rand({53, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto input_vulkan = input_cpu.vulkan();

  const auto weight_cpu = at::rand({53, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto weight_vulkan = weight_cpu.vulkan();

  const auto bias_cpu = at::rand({53, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_vulkan = bias_cpu.vulkan();

  const auto output_cpu = at::layer_norm(input_cpu, {53, 139, 109}, weight_cpu, bias_cpu, 1e-05, false);
  const auto output_vulkan = at::layer_norm(input_vulkan, {53, 139, 109}, weight_vulkan, bias_vulkan, 1e-05, false);

  const auto check = almostEqual(output_cpu, output_vulkan.cpu());
  if (!check) {
    showRtol(output_cpu, output_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, layer_norm_4d_small) {
  c10::InferenceMode mode;

  const auto input_cpu = at::rand({1, 1, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto input_vulkan = input_cpu.vulkan();

  const auto weight_cpu = at::rand({1, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto weight_vulkan = weight_cpu.vulkan();

  const auto bias_cpu = at::rand({1, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_vulkan = bias_cpu.vulkan();

  const auto output_cpu = at::layer_norm(input_cpu, {1, 1, 1}, weight_cpu, bias_cpu, 1e-05, false);
  const auto output_vulkan = at::layer_norm(input_vulkan, {1, 1, 1}, weight_vulkan, bias_vulkan, 1e-05, false);

  const auto check = almostEqual(output_cpu, output_vulkan.cpu());
  if (!check) {
    showRtol(output_cpu, output_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, layer_norm_4d_medium) {
  c10::InferenceMode mode;

  const auto input_cpu = at::rand({1, 3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat));
  const auto input_vulkan = input_cpu.vulkan();

  const auto weight_cpu = at::rand({3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat));
  const auto weight_vulkan = weight_cpu.vulkan();

  const auto bias_cpu = at::rand({3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_vulkan = bias_cpu.vulkan();

  const auto output_cpu = at::layer_norm(input_cpu, {3, 5, 7}, weight_cpu, bias_cpu, 1e-05, false);
  const auto output_vulkan = at::layer_norm(input_vulkan, {3, 5, 7}, weight_vulkan, bias_vulkan, 1e-05, false);

  const auto check = almostEqual(output_cpu, output_vulkan.cpu());
  if (!check) {
    showRtol(output_cpu, output_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, layer_norm_4d_large) {
  c10::InferenceMode mode;

  const auto input_cpu = at::rand({1, 53, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto input_vulkan = input_cpu.vulkan();

  const auto weight_cpu = at::rand({53, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto weight_vulkan = weight_cpu.vulkan();

  const auto bias_cpu = at::rand({53, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_vulkan = bias_cpu.vulkan();

  const auto output_cpu = at::layer_norm(input_cpu, {53, 139, 109}, weight_cpu, bias_cpu, 1e-05, false);
  const auto output_vulkan = at::layer_norm(input_vulkan, {53, 139, 109}, weight_vulkan, bias_vulkan, 1e-05, false);

  const auto check = almostEqual(output_cpu, output_vulkan.cpu());
  if (!check) {
    showRtol(output_cpu, output_vulkan.cpu());
  }

  ASSERT_TRUE(check);
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

TEST_F(VulkanAPITest, mean) {
  const auto in_cpu = at::rand({17, 3, 79, 53}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::mean(in_cpu, {-1, -2}, true);

  const auto in_vulkan = in_cpu.vulkan();
  const auto out_vulkan = at::mean(in_vulkan, {-1, -2}, true);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, mean2d) {
  const auto in_cpu = at::rand({11, 7, 173, 37}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::mean(in_cpu, {-1, -2}, false);

  const auto in_vulkan = in_cpu.vulkan();
  const auto out_vulkan = at::mean(in_vulkan, {-1, -2}, false);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
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

TEST_F(VulkanAPITest, mul) {
  const auto a_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::mul(a_cpu, b_cpu);
  const auto c_vulkan = at::mul(a_vulkan, b_vulkan);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, mul_broadcast0) {
  const auto a_cpu = at::rand({3, 5, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({3, 5, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::mul(a_cpu, b_cpu);
  const auto c_vulkan = at::mul(a_vulkan, b_vulkan);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, mul_broadcast1) {
  const auto a_cpu = at::rand({3, 5, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({3, 5, 1, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::mul(a_cpu, b_cpu);
  const auto c_vulkan = at::mul(a_vulkan, b_vulkan);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, mul_broadcast2) {
  const auto a_cpu = at::rand({3, 4, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({4, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::mul(a_cpu, b_cpu);
  const auto c_vulkan = at::mul(a_vulkan, b_vulkan);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, mul_broadcast3) {
  const auto a_cpu = at::rand({3, 4, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({1, 1, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::mul(a_cpu, b_cpu);
  const auto c_vulkan = at::mul(a_vulkan, b_vulkan);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, mul_broadcast4) {
  const auto a_cpu = at::rand({3, 4, 179, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({1, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::mul(a_cpu, b_cpu);
  const auto c_vulkan = at::mul(a_vulkan, b_vulkan);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
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

TEST_F(VulkanAPITest, softmax) {
  c10::InferenceMode mode;

  at::Tensor test_in[] = {
    at::rand({1, 196, 302, 5}, at::TensorOptions(at::kCPU).dtype(at::kFloat)),
    at::rand({1, 197, 302, 5}, at::TensorOptions(at::kCPU).dtype(at::kFloat)),
    at::rand({1, 198, 302, 5}, at::TensorOptions(at::kCPU).dtype(at::kFloat)),
    at::rand({1, 199, 302, 5}, at::TensorOptions(at::kCPU).dtype(at::kFloat)),
  };

  for (auto in_cpu : test_in) {
    const auto out_cpu = at::softmax(in_cpu, 1);

    const auto in_vulkan = in_cpu.vulkan();
    const auto out_vulkan = at::softmax(in_vulkan, 1);

    const auto check = almostEqual(out_cpu, out_vulkan.cpu());
    if (!check) {
      showRtol(out_cpu, out_vulkan.cpu());
    }

    ASSERT_TRUE(check);
  }
}

// TODO: Currently the op is not working correctly. Add it back when it is fixed.
TEST_F(VulkanAPITest, DISABLED_log_softmax) {
  at::Tensor test_in[] = {
    at::rand({1, 196, 302, 5}, at::TensorOptions(at::kCPU).dtype(at::kFloat)),
    at::rand({1, 197, 302, 5}, at::TensorOptions(at::kCPU).dtype(at::kFloat)),
    at::rand({1, 198, 302, 5}, at::TensorOptions(at::kCPU).dtype(at::kFloat)),
    at::rand({1, 199, 302, 5}, at::TensorOptions(at::kCPU).dtype(at::kFloat)),
  };

  for (auto in_cpu : test_in) {
    const auto out_cpu = at::softmax(in_cpu, 1);

    const auto in_vulkan = in_cpu.vulkan();
    const auto out_vulkan = at::log_softmax(in_vulkan, 1);

    const auto check = almostEqual(out_cpu, out_vulkan.cpu());
    if (!check) {
      showRtol(out_cpu, out_vulkan.cpu());
    }

    ASSERT_TRUE(check);
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

TEST_F(VulkanAPITest, sub) {
  const auto a_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::sub(a_cpu, b_cpu, 2.1f);
  const auto c_vulkan = at::sub(a_vulkan, b_vulkan, 2.1f);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, sub_broadcast0) {
  const auto a_cpu = at::rand({3, 5, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({3, 5, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::sub(a_cpu, b_cpu, 1.8f);
  const auto c_vulkan = at::sub(a_vulkan, b_vulkan, 1.8f);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, sub_broadcast1) {
  const auto a_cpu = at::rand({3, 5, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({3, 5, 1, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::sub(a_cpu, b_cpu, 1.8f);
  const auto c_vulkan = at::sub(a_vulkan, b_vulkan, 1.8f);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, sub_broadcast2) {
  const auto a_cpu = at::rand({3, 4, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({4, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::sub(a_cpu, b_cpu, 2.5f);
  const auto c_vulkan = at::sub(a_vulkan, b_vulkan, 2.5f);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, sub_broadcast3) {
  const auto a_cpu = at::rand({3, 4, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({1, 1, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::sub(a_cpu, b_cpu, 2.5f);
  const auto c_vulkan = at::sub(a_vulkan, b_vulkan, 2.5f);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, sub_broadcast4) {
  const auto a_cpu = at::rand({3, 4, 179, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({1, 179, 221}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::sub(a_cpu, b_cpu, 2.5f);
  const auto c_vulkan = at::sub(a_vulkan, b_vulkan, 2.5f);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
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

TEST_F(VulkanAPITest, uniform) {
  float a_min = -8.2f;
  float a_max = -1.4f;

  auto a_vulkan =
      at::rand({8, 7, 12, 10}, at::device(at::kCPU).dtype(at::kFloat)).vulkan();
  a_vulkan.uniform_(a_min, a_max);
  auto a_cpu = a_vulkan.cpu();

  ASSERT_TRUE(a_cpu.max().item<float>() < a_max);
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

void test_unsqueeze(const at::IntArrayRef input_shape, int64_t dim) {
  c10::InferenceMode mode;
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::unsqueeze(in_cpu, dim);

  const auto in_vulkan = in_cpu.vulkan();
  const auto out_vulkan = at::unsqueeze(in_vulkan, dim);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }
  ASSERT_TRUE(check);

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
  }, ::c10::Error);

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
    }, ::c10::Error);
  }

  // Arrange: Vulkan cat expects 4 dimensional inputs
  {
    const auto in_cpu1 = at::rand({3, 9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_cpu2 = at::rand({9, 112, 193}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_cpu3 = at::rand({3, 9, 331, 193}, at::device(at::kCPU).dtype(at::kFloat));

    // Act
    EXPECT_THROW({
      const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 0);
    }, ::c10::Error);
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
    }, ::c10::Error);
  }

  // Arrange: Vulkan cat expects inputs of same dimensions
  {
    const auto in_cpu1 = at::rand({3, 9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_cpu2 = at::rand({9, 112, 193}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_cpu3 = at::rand({3, 9, 331, 193}, at::device(at::kCPU).dtype(at::kFloat));

    // Act
    EXPECT_THROW({
      const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 2);
    }, ::c10::Error);
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
    }, ::c10::Error);
  }

  // Arrange: Vulkan cat expects 4 dimensional inputs
  {
    const auto in_cpu1 = at::rand({3, 9, 221, 193}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_cpu2 = at::rand({9, 112, 193}, at::device(at::kCPU).dtype(at::kFloat));
    const auto in_cpu3 = at::rand({3, 9, 331, 193}, at::device(at::kCPU).dtype(at::kFloat));

    // Act
    EXPECT_THROW({
      const auto out_vulkan = at::cat({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 3);
    }, ::c10::Error);
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
  }, ::c10::Error);

  EXPECT_THROW({
    const auto out_vulkan = in_cpu.vulkan();
    out_vulkan.permute({2, 2, 1, 0});
  }, ::c10::Error);

  // Act: Number of dims don't match
  EXPECT_THROW({
    const auto out_vulkan = at::permute(in_cpu.vulkan(), {4, 3, 2, 1, 0});
  }, ::c10::Error);

  EXPECT_THROW({
    const auto out_vulkan = at::permute(in_cpu.vulkan(), {2, 1, 0});
  }, ::c10::Error);

  EXPECT_THROW({
    const auto out_vulkan = in_cpu.vulkan();
    out_vulkan.permute({4, 3, 2, 1, 0});
  }, ::c10::Error);

  EXPECT_THROW({
    const auto out_vulkan = in_cpu.vulkan();
    out_vulkan.permute({2, 1, 0});
  }, ::c10::Error);

  // Act: Dim out of range
  EXPECT_THROW({
    const auto out_vulkan = at::permute(in_cpu.vulkan(), {5, 2, 1, 0});
  }, ::c10::Error);

  EXPECT_THROW({
    const auto out_vulkan = in_cpu.vulkan();
    out_vulkan.permute({5, 2, 1, 0});
  }, ::c10::Error);

  // Act: Input tensor size > 4D
  const auto in_cpu_5d = at::rand({1, 2, 1, 2, 161}, at::device(at::kCPU).dtype(at::kFloat));
  EXPECT_THROW({
    const auto out_vulkan_5d = at::permute(in_cpu_5d.vulkan(), {4, 3, 2, 1, 0});
  }, ::c10::Error);

  EXPECT_THROW({
    const auto out_vulkan_5d = in_cpu_5d.vulkan();
    out_vulkan_5d.permute({4, 3, 2, 1, 0});
  }, ::c10::Error);
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

TEST_F(VulkanAPITest, slice_invalidinputs_exceptions) {
  // Act: slice step must be positive
  EXPECT_THROW({
    slice_test({2, 3, 4, 5}, 3, 0, 3, 0);
  }, ::c10::Error);

  // Act: Vulkan doesn't support zero-sized slice (when start=end)
  EXPECT_THROW({
    slice_test({2, 3, 4, 5}, 3, 0, 0, 1);
  }, ::c10::Error);

  // Act: Vulkan doesn't support zero-sized slice (when start > end)
  EXPECT_THROW({
    slice_test({2, 3, 4, 5}, 3, 3, 2, 1);
  }, ::c10::Error);
}

TEST_F(VulkanAPITest, stack_invalid_inputs) {
  // Act: Vulkan stack expects at least one tensor
  EXPECT_THROW({
    at::stack({}, 0);
  }, ::c10::Error);

  // Act: Vulkan stack expects dim = 0
  EXPECT_THROW({
    at::stack({
        at::rand({5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
        at::rand({5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
        at::rand({5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan()}, 1);
  }, ::c10::Error);

  // Act: Vulkan stack expects 2 dimensional inputs
  EXPECT_THROW({
    at::stack({
        at::rand({5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
        at::rand({5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
        at::rand({3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan()}, 0);
  }, ::c10::Error);

  // Act: Vulkan stack inputs must have matching sizes
  EXPECT_THROW({
    at::stack({
        at::rand({5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
        at::rand({5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
        at::rand({6, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan()}, 0);
  }, ::c10::Error);
}

TEST_F(VulkanAPITest, stack_1_tensor) {
  // Arrange
  const auto in_cpu1 = at::rand({221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::stack({in_cpu1}, 0);
  const auto out_vulkan = at::stack({in_cpu1.vulkan()}, 0);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, stack_2_tensors) {
  // Arrange
  const auto in_cpu1 = at::rand({221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::stack({in_cpu1, in_cpu2}, 0);
  const auto out_vulkan = at::stack({in_cpu1.vulkan(), in_cpu2.vulkan()}, 0);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, stack_3_tensors) {
  // Arrange
  const auto in_cpu1 = at::rand({221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::stack({in_cpu1, in_cpu2, in_cpu3}, 0);
  const auto out_vulkan = at::stack({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan()}, 0);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, stack_4_tensors) {
  // Arrange
  const auto in_cpu1 = at::rand({221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu3 = at::rand({221, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu4 = at::rand({221, 193}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  const auto out_cpu = at::stack({in_cpu1, in_cpu2, in_cpu3, in_cpu4}, 0);
  const auto out_vulkan = at::stack({in_cpu1.vulkan(), in_cpu2.vulkan(), in_cpu3.vulkan(), in_cpu4.vulkan()}, 0);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, stack_from_1_to_20_tensors) {
  std::vector<at::Tensor> tensors_cpu = {};
  std::vector<at::Tensor> tensors_vulkan = {};

  for (const auto i : c10::irange(20)) {
    at::Tensor in_cpu = at::rand({221, 193}, at::device(at::kCPU).dtype(at::kFloat));
    tensors_cpu.emplace_back(in_cpu);
    tensors_vulkan.emplace_back(in_cpu.vulkan());
    at::Tensor out_cpu = at::stack(tensors_cpu, 0);
    at::Tensor out_vulkan = at::stack(tensors_vulkan, 0);
    const auto check = almostEqual(out_cpu, out_vulkan.cpu());
    if (!check) {
      std::cout << "Error when stacking " << i << " tensors" << std::endl;
      showRtol(out_cpu, out_vulkan.cpu());
    }
    ASSERT_TRUE(check);
  }
}

TEST_F(VulkanAPITest, clone_success) {
  // Arrange
  std::multimap<c10::optional<c10::MemoryFormat>, std::vector<int64_t>> mem2sizes {
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
  // Act: Vulkan supports Preserve and Contiguous memory foramts
  EXPECT_THROW({
    clone_test({2, 3, 5, 161}, c10::MemoryFormat::ChannelsLast);
  }, ::c10::Error);

  // Act: Vulkan supports Preserve and Contiguous memory foramts
  EXPECT_THROW({
    clone_test({2, 3, 5, 161}, c10::MemoryFormat::ChannelsLast3d);
  }, ::c10::Error);
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
  }, ::c10::Error);

  // Act: non-3D input tensor
  EXPECT_THROW({
    const auto in_cpu_2d = at::rand({1, H_in}, at::device(at::kCPU).dtype(at::kFloat));
    at::gru(in_cpu_2d.vulkan(), h0_cpu.vulkan(), { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
      weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) },
      has_biases, num_layers, gru_dropout, train, bidirectional, batch_first);
  }, ::c10::Error);

  // Act: non-3D hidden tensor
  EXPECT_THROW({
    const auto h0_cpu_2d = at::rand({num_layers, H_out}, at::device(at::kCPU).dtype(at::kFloat));
    at::gru(in_cpu.vulkan(), h0_cpu_2d.vulkan(), { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
      weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) },
      has_biases, num_layers, gru_dropout, train, bidirectional, batch_first);
  }, ::c10::Error);

  // Act: has_biases should be true
  EXPECT_THROW({
    at::gru(in_cpu.vulkan(), h0_cpu.vulkan(), { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
      weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) },
      false, num_layers, gru_dropout, train, bidirectional, batch_first);
  }, ::c10::Error);

  // Act: train should be false
  EXPECT_THROW({
    at::gru(in_cpu.vulkan(), h0_cpu.vulkan(), { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
      weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) },
      has_biases, num_layers, gru_dropout, true, bidirectional, batch_first);
  }, ::c10::Error);

  // Act: bidirectional should be false
  EXPECT_THROW({
    at::gru(in_cpu.vulkan(), h0_cpu.vulkan(), { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
      weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) },
      has_biases, num_layers, gru_dropout, train, true, batch_first);
  }, ::c10::Error);

  // Act: batch_first should be true
  EXPECT_THROW({
    at::gru(in_cpu.vulkan(), h0_cpu.vulkan(), { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
      weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) },
      has_biases, num_layers, gru_dropout, train, bidirectional, false);
  }, ::c10::Error);

  // Act: dropout should be 0.0
  EXPECT_THROW({
    at::gru(in_cpu.vulkan(), h0_cpu.vulkan(), { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
      weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) },
      has_biases, num_layers, 1.0, train, bidirectional, batch_first);
  }, ::c10::Error);
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
  }, ::c10::Error);

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
  }, ::c10::Error);

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
  }, ::c10::Error);

  // Act: has_biases should be true
  EXPECT_THROW({
    auto prepack = callOpByName(
        "vulkan_prepack::create_gru_context",
        "",
        std::vector<at::Tensor>({ weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
           weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) }),
        false, num_layers, gru_dropout, train, bidirectional, batch_first);
  }, ::c10::Error);

  // Act: train should be false
  EXPECT_THROW({
    auto prepack = callOpByName(
        "vulkan_prepack::create_gru_context",
        "",
        std::vector<at::Tensor>({ weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
           weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) }),
        has_biases, num_layers, gru_dropout, true, bidirectional, batch_first);
  }, ::c10::Error);

  // Act: bidirectional should be false
  EXPECT_THROW({
     auto prepack = callOpByName(
        "vulkan_prepack::create_gru_context",
        "",
        std::vector<at::Tensor>({ weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
           weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) }),
        has_biases, num_layers, gru_dropout, train, true, batch_first);
 }, ::c10::Error);

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
  }, ::c10::Error);

  // Act: dropout should be 0.0
  EXPECT_THROW({
    auto prepack = callOpByName(
        "vulkan_prepack::create_gru_context",
        "",
        std::vector<at::Tensor>({ weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
           weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) }),
        has_biases, num_layers, 1.0, train, bidirectional, batch_first);
  }, ::c10::Error);
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

TEST_F(VulkanAPITest, linear_2d) {
  test_linear({1, 37}, {41, 37}, {41});
}

TEST_F(VulkanAPITest, linear_3d) {
  test_linear({1, 1, 37}, {41, 37}, {41});
}

TEST_F(VulkanAPITest, linear_4d) {
  test_linear({1, 1, 1, 37}, {41, 37}, {41});
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
