#include <gtest/gtest.h>

#include <torch/types.h>
#include <torch/utils.h>

#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/xnnpack/Engine.h>
#include <ATen/native/xnnpack/OpContext.h>
#include <ATen/native/xnnpack/Pooling.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/MemoryFormat.h>

#include <atomic>
#include <condition_variable>
#include <thread>

#if defined(C10_MOBILE) && defined(USE_XNNPACK)

bool checkRtol(const at::Tensor& diff, const std::vector<at::Tensor> inputs) {
  double maxValue = 0.0;
  for (auto& tensor : inputs) {
    maxValue = fmax(tensor.abs().max().item<float>(), maxValue);
  }
  return diff.abs().max().item<float>() < (0.01 + 2e-2 * maxValue);
}
bool almostEqual(const at::Tensor& a, const at::Tensor& b) {
  return checkRtol(a - b, {a, b});
}

bool exactlyEqual(const at::Tensor& a, const at::Tensor& b) {
  return (a - b).abs().max().item<float>() == 0.f;
}

void test_hardswish(const at::Tensor& input, const at::Tensor& expected) {
  ASSERT_TRUE(at::native::xnnpack::use_hardswish(input));
  auto result = at::native::xnnpack::hardswish(input);
  auto check = almostEqual(expected, result);
  ASSERT_TRUE(check);
  ASSERT_TRUE(
      expected.suggest_memory_format() == input.suggest_memory_format());
}

void test_hardswish_(at::Tensor input, const at::Tensor& expected) {
  ASSERT_TRUE(at::native::xnnpack::use_hardswish(input));
  at::native::xnnpack::hardswish_(input);
  auto check = almostEqual(expected, input);
  ASSERT_TRUE(check);
  ASSERT_TRUE(
      expected.suggest_memory_format() == input.suggest_memory_format());
}

void test_global_average_pool(at::Tensor input, const at::Tensor& expected) {
  ASSERT_TRUE(at::native::xnnpack::use_global_average_pool(input));
  auto result = at::native::xnnpack::global_average_pool(input);
  auto check = almostEqual(expected, result);
  ASSERT_TRUE(check);
}

// Since XNNPACK path is only taken #if defined(C10_MOBILE) &&
// defined(USE_XNNPACK) We can't compare regular CPU path with XNNPACK path in
// the same test binary Instead we precompute regular results and compare with
// XNNPACK path here
TEST(TestXNNPackOps, TestLinear) {
  constexpr std::array<int64_t, 2u> input_shape{1, 37};
  constexpr std::array<int64_t, 2u> weight_shape{41, 37};
  constexpr std::array<int64_t, 2u> bias_shape{1, 41};
  const auto input_cpu =
      at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto weight =
      at::rand(weight_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto bias =
      at::rand(bias_shape, at::device(at::kCPU).dtype(at::kFloat));

  const auto out_cpu = at::linear(input_cpu, weight, bias);

  const auto xnnpack_bias = bias.view({41});
  ASSERT_TRUE(at::native::xnnpack::use_linear(input_cpu, weight, xnnpack_bias));
  const auto result =
      at::native::xnnpack::linear(input_cpu, weight, xnnpack_bias);

  auto check = almostEqual(out_cpu, result);
  ASSERT_TRUE(check);
}

TEST(TestXNNPackOps, TestMaxPool2d) {
  const auto in_cpu =
      at::rand({5, 13, 55, 68}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const auto out_cpu =
      at::max_pool2d(in_cpu, {3, 4}, {2, 1}, {1, 1}, {1, 1}, false);
  ASSERT_TRUE(at::native::xnnpack::use_max_pool2d(
      in_cpu, {3, 4}, {1, 1}, {2, 1}, {1, 1}, false));
  const auto result = at::native::xnnpack::max_pool2d(
      in_cpu, {3, 4}, {1, 1}, {2, 1}, {1, 1}, false);

  auto check = almostEqual(out_cpu, result);
  ASSERT_TRUE(check);
}

TEST(TestXNNPackOps, TestConvolution2d) {
  constexpr int64_t groups = 1;
  constexpr std::array<int64_t, 2u> stride{2, 2};
  constexpr std::array<int64_t, 2u> padding{1, 1};
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
  } input{1, 3, 8, 8};

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
  } weights{1, input.channels, 3, 3};

  const auto input_cpu =
      at::randn(input.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto weights_cpu =
      at::randn(weights.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_cpu = at::randn(
      {weights.output_channels}, at::device(at::kCPU).dtype(at::kFloat));

  const auto output_cpu = at::conv2d(
      input_cpu, weights_cpu, bias_cpu, stride, padding, dilation, groups);

  ASSERT_TRUE(at::native::xnnpack::use_convolution2d(
      input_cpu,
      weights_cpu,
      weights.output_channels,
      padding,
      stride,
      dilation,
      groups,
      false));
  const auto result = at::native::xnnpack::convolution2d(
      input_cpu, weights_cpu, bias_cpu, padding, stride, dilation, groups);
  auto check = almostEqual(output_cpu, result);
  ASSERT_TRUE(check);
}

TEST(TestXNNPackOps, TestHardSwish) {
  // input, expected_result pair
  auto in = torch::tensor({{1, 1}, {1, 1}}, {torch::kFloat32});
  auto in_slice = in.index({"...", 0});

  std::vector<std::pair<at::Tensor, at::Tensor>> input_result_pairs = {
      {torch::tensor({1, 2, 3, 4, 5}, {torch::kFloat32}),
       torch::tensor(
           {0.6667, 1.6667, 3.0000, 4.0000, 5.0000}, {torch::kFloat32})},
      {torch::tensor({0.3330}, {torch::kFloat32}),
       torch::tensor({0.1850}, {torch::kFloat32})},
      {torch::tensor({{0.4523, 0.8131, 0.9829}, {0.0782, 0.7395, 0.0787}}),
       torch::tensor({{0.2602, 0.5167, 0.6525}, {0.0401, 0.4609, 0.0404}})},
      {in_slice, torch::tensor({0.6667, 0.6667}, {torch::kFloat32})},
      {torch::tensor({{{{0.4993, 0.3835}, {0.3163, 0.2348}},
                       {{0.4705, 0.4129}, {0.9314, 0.0631}}},
                      {{{0.0030, 0.5656}, {0.1413, 0.1943}},
                       {{0.1380, 0.1985}, {0.2746, 0.8109}}}})
           .contiguous(at::MemoryFormat::ChannelsLast),
       torch::tensor({{{{0.2912, 0.2163}, {0.1748, 0.1266}},
                       {{0.2722, 0.2349}, {0.6103, 0.0322}}},
                      {{{0.0015, 0.3361}, {0.0740, 0.1034}},
                       {{0.0722, 0.1058}, {0.1499, 0.5150}}}})
           .contiguous(at::MemoryFormat::ChannelsLast)}};

  for (const auto& input_result : input_result_pairs) {
    test_hardswish(input_result.first, input_result.second);
    test_hardswish_(input_result.first, input_result.second);
  }
}

TEST(TestXNNPackOps, TestConvolution2dMultiThreaded) {
  constexpr int64_t groups = 1;

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
  } input{1, 3, 8, 8};

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
  } weights{1, input.channels, 3, 3};

  const auto input_cpu =
      at::randn(input.size(), at::device(at::kCPU).dtype(at::kFloat));
  auto weights_cpu =
      at::randn(weights.size(), at::device(at::kCPU).dtype(at::kFloat));
  auto bias_cpu = at::randn(
      {weights.output_channels}, at::device(at::kCPU).dtype(at::kFloat));

  auto context = at::native::xnnpack::XNNPackConv2dOpContext::create_context(
      std::move(weights_cpu), std::move(bias_cpu), {1, 1}, {2, 2}, {1, 1}, groups, c10::nullopt, c10::nullopt);
  std::atomic<int64_t> count{0};
  int64_t num_workers = 5;
  std::mutex lock;
  std::condition_variable cond;
  auto sync_and_run_conv = [&](int64_t h, int64_t w) -> at::Tensor
  {
    auto input_tensor = at::randn({1, 3, h, w}, at::device(at::kCPU).dtype(at::kFloat));
    int64_t count_val = ++count;
    if (count_val < num_workers) {
      std::unique_lock<std::mutex> g(lock);
      while ((count_val = count.load()) < num_workers) {
        cond.wait(g, [&]() {
            auto new_val = count.load();
            return new_val >= num_workers;});
      }
    } else {
      std::unique_lock<std::mutex> g(lock);
      cond.notify_all();
    }
    for (int64_t i = 0; i < 30; i++) {
      context->run(input_tensor);
    }
    return context->run(input_tensor);
  };

  auto conv = [sync_and_run_conv](int64_t h, int64_t w) -> at::Tensor
  {
    return sync_and_run_conv(h, w);
  };

  std::thread t1(conv, 16, 16);
  std::thread t2(conv, 12, 12);
  std::thread t3(conv, 20, 20);
  std::thread t4(conv, 22, 22);
  std::thread t5(conv, 8, 8);
  t1.join();
  t2.join();
  t3.join();
  t4.join();
  t5.join();
}

TEST(TestXNNPackOps, TestGlobal) {
  // input, expected_result pair
  std::vector<std::pair<at::Tensor, at::Tensor>> input_result_pairs = {
      {torch::tensor(
           {{{{0.0852, 0.7312, 0.9943, 0.7105},
              {0.0956, 0.9072, 0.3124, 0.9362},
              {0.5878, 0.8883, 0.5086, 0.9494}},
             {{0.1056, 0.4968, 0.7740, 0.7593},
              {0.8519, 0.3543, 0.8078, 0.5517},
              {0.1413, 0.4608, 0.1706, 0.0314}}}},
           {torch::kFloat32}),
       torch::tensor({{{{0.6422}}, {{0.4588}}}}, {torch::kFloat32})},
      {torch::tensor(
           {{{{0.0280, 0.9073}, {0.2103, 0.5298}},
             {{0.5335, 0.9901}, {0.2902, 0.2955}}},
            {{{0.2363, 0.7024}, {0.7903, 0.8260}},
             {{0.3802, 0.5959}, {0.5749, 0.8855}}}},
           {torch::kFloat32}),
       torch::tensor(
           {{{{0.4188}}, {{0.5273}}}, {{{0.6388}}, {{0.6091}}}},
           {torch::kFloat32})}};

  for (const auto& input_result : input_result_pairs) {
    test_global_average_pool(input_result.first, input_result.second);
  }
}

int main(int argc, char* argv[]) {
  // Setting default allocator as mobile to test copy / no copy cases
  c10::SetCPUAllocator(c10::GetDefaultMobileCPUAllocator(), /*priority*/ 100);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
