#import <ATen/native/metal/MetalConvolution.h>
#import <ATen/native/metal/MetalUtils.h>
#import <ATen/native/metal/mpscnn/MPSCNNOps.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>
#import <ATen/native/metal/mpscnn/tests/MPSCNNTests.h>
#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <ATen/ATen.h>
#import <ATen/native/metal/mpscnn/tests/MPSCNNTests.h>

#include <stdlib.h>
#include <torch/script.h>
#include <sstream>

#define ITER_COUNT 10

namespace at {
namespace native {
namespace metal {

int64_t rand(int64_t min, int64_t max) {
  return min + (std::rand() % static_cast<int64_t>(max - min + 1));
}

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

bool almostEqualTensor(const at::Tensor& a, const at::Tensor& b, float t) {
  if (a.sizes() != b.sizes()) {
    return false;
  }
  if (a.numel() != b.numel()) {
    return false;
  }
  for (int i = 0; i < a.numel(); ++i) {
    float x1 = a.data_ptr<float>()[i];
    float x2 = b.data_ptr<float>()[i];
    if (std::abs(x1 - x2) > t) {
      return false;
    }
  }
  return true;
}

bool almostEqualVec(
    const std::vector<float> vec1,
    const std::vector<float> vec2,
    float t) {
  if (vec1.size() != vec2.size()) {
    return false;
  }
  for (int i = 0; i < vec1.size(); ++i) {
    if (std::abs(vec1[i] - vec2[i]) > t) {
      return false;
    }
  }
  return true;
}

void print(bool cond, NSString* name, const std::vector<int64_t>& sizes) {
  NSMutableString* strSizes = [NSMutableString new];
  std::for_each(sizes.begin(), sizes.end(), ^(int64_t n) {
    [strSizes appendString:[NSString stringWithFormat:@"%lld,", n]];
  });
  void (^print)(NSString*) = ^(NSString* str) {
    NSLog(@"[TEST_%@], [%@], [%@]", name, strSizes, str);
  };
  cond ? print(@"SUCCEED") : print(@"FAILED");
}

bool test_aten() {
  auto x1 =
      at::rand({1, 2, 2, 2}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto mx1 = x1.metal();
  TORCH_CHECK(mx1.device().type() == at::kMetal);
  auto x2 = mx1.cpu();
  TORCH_CHECK(x2.device().type() == at::kCPU);
  bool b = almostEqual(x1, x2);
  print(b, @"ATEN", {1, 2, 2, 2});
  return b;
}

bool test_NC4() {
#define TEST_NC4(n, c, h, w)                                                   \
  {                                                                            \
    auto t =                                                                   \
        at::rand({n, c, h, w}, at::TensorOptions(at::kCPU).dtype(at::kFloat)); \
    auto b = std::vector<float>{t.data_ptr<float>(),                           \
                                t.data_ptr<float>() + n * c * h * w};          \
    auto c4 = NCHW_to_NC4((float*)t.data_ptr<float>(), t.sizes().vec());       \
    auto n4 = NC4_to_NCHW((float*)c4.data(), t.sizes().vec());                 \
    if (n4 == b) {                                                             \
      print(true, @"NC4", {n, c, h, w});                                       \
    } else {                                                                   \
      return false;                                                            \
    }                                                                          \
  }
  for (int i = 0; i < ITER_COUNT; ++i) {
    int64_t N = rand(1, 24);
    int64_t C = rand(1, 48);
    int64_t H = rand(1, 320);
    int64_t W = rand(1, 320);
    std::vector<int64_t> x{N, C, H, W};
    auto t = at::rand(x, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto b = std::vector<float>{t.data_ptr<float>(),
                                t.data_ptr<float>() + N * C * H * W};
    auto c4 = NCHW_to_NC4((float*)t.data_ptr<float>(), t.sizes().vec());
    auto n4 = NC4_to_NCHW((float*)c4.data(), t.sizes().vec());
    if (n4 == b) {
      print(true, @"NC4", x);
    } else {
      return false;
    }
  }
  return true;
}

bool test_MPSImage() API_AVAILABLE(ios(10.0), macos(10.13)) {
#define TEST_MPS_IMAGE(n, c, h, w)                                             \
  {                                                                            \
    auto t1 =                                                                  \
        at::rand({n, c, h, w}, at::TensorOptions(at::kCPU).dtype(at::kFloat)); \
    auto b = std::vector<float>{t1.data_ptr<float>(),                          \
                                t1.data_ptr<float>() + n * c * h * w};         \
    MPSImage* img = [MPSImage imageFromCPUTensor:t1];                          \
    auto t2 = [img toCPUTensor];                                               \
    bool result = almostEqual(t1, t2);                                         \
    if (result) {                                                              \
      print(result, @"MPS_IMAGE", {n, c, h, w});                               \
    } else {                                                                   \
      return false;                                                            \
    }                                                                          \
  }
  for (int i = 0; i < ITER_COUNT; ++i) {
    int64_t N = rand(1, 24);
    int64_t C = rand(1, 48);
    int64_t H = rand(1, 320);
    int64_t W = rand(1, 320);
    std::vector<int64_t> x{N, C, H, W};
    auto t1 = at::rand(x, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto b = std::vector<float>{t1.data_ptr<float>(),
                                t1.data_ptr<float>() + N * C * H * W};
    MPSImage* img = [MPSImage imageFromCPUTensor:t1];
    auto t2 = [img toCPUTensor];
    bool result = almostEqual(t1, t2);
    if (result) {
      print(result, @"MPS_IMAGE", {N, C, H, W});
    } else {
      return false;
    }
  }
  return true;
}

bool test_MPSImageCopy() {
  std::vector<int64_t> sz{2, 3, 1, 1};
  auto t1 = at::rand(sz, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  float* src = t1.data_ptr<float>();
  MPSImage* im = [MPSImage imageFromHost:src Sizes:t1.sizes().vec()];
  MPSImage* cim = [MPSImage imageFromImage:im];
  auto t2 = [cim toCPUTensor];
  bool b = almostEqual(t1, t2);
  print(b, @"MPSImageCopy", sz);
  return b;
}

bool test_MPSTemporaryImageCopy() {
  std::vector<int64_t> sz{2, 3, 1, 1};
  auto t1 = at::rand(sz, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  MetalCommandBuffer* cb = [MetalCommandBuffer newBuffer];
  float* src = t1.data_ptr<float>();
  MPSTemporaryImage* tim = [MPSImage temporaryImageFromHost:src
                                                      Sizes:t1.sizes().vec()
                                              CommandBuffer:cb];
  MPSImage* im = [MPSImage imageFromTemporaryImage:tim
                                     CommandBuffer:cb
                                waitUntilCompleted:YES];
  auto t2 = [im toCPUTensor];
  bool b = almostEqual(t1, t2);
  print(b, @"MPSTemporaryImageCopy", sz);
  return b;
}

bool test_conv2d() {
#define ARRAY(...) __VA_ARGS__
#define TEST_CONV2D(x, w, b, pad)                                           \
  {                                                                         \
    auto X = torch::rand(x, at::TensorOptions(at::kCPU).dtype(at::kFloat)); \
    auto W = torch::rand(w, at::TensorOptions(at::kCPU).dtype(at::kFloat)); \
    auto B = torch::rand(b, at::TensorOptions(at::kCPU).dtype(at::kFloat)); \
    auto S = c10::IntArrayRef{1, 1};                                        \
    auto P = c10::IntArrayRef(pad);                                         \
    auto D = c10::IntArrayRef{1, 1};                                        \
    int64_t groups = 1;                                                     \
    auto Y1 = at::native::conv2d(X, W, B, S, P, D, groups);                 \
    auto X2 = X.metal();                                                    \
    at::native::metal::Conv2DParams params{                                 \
        X.sizes(), W.sizes(), P, S, D, groups};                             \
    auto Y2 = at::native::metal::mpscnn::conv2d(X2, W, B, params).cpu();    \
    bool check = almostEqual(Y1, Y2);                                       \
    if (check) {                                                            \
      print(check, @"CONV2D", x);                                           \
    } else {                                                                \
      return false;                                                         \
    }                                                                       \
  }
  for (int i = 0; i < ITER_COUNT; ++i) {
    int64_t N = rand(1, 10);
    int64_t C = rand(1, 48);
    int64_t IH = rand(1, 300);
    int64_t IW = rand(1, 300);
    int64_t OC = rand(1, 48);
    int64_t IC = C;
    int64_t KH = rand(1, MIN(10, IH));
    int64_t KW = rand(1, MIN(10, IW));
    int64_t PH = rand(1, 10);
    int64_t PW = rand(1, 10);
    int64_t SH = rand(1, 10);
    int64_t SW = rand(1, 10);
    auto X = torch::rand(
        {N, C, IH, IW}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto W = torch::rand(
        {OC, IC, KH, KW}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto B = torch::rand({OC}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto S = c10::IntArrayRef({SH, SW});
    auto P = c10::IntArrayRef({PH, PW});
    auto D =
        c10::IntArrayRef({1, 1}); // Dilated convolution is not supported yet
    int64_t groups = 1;
    auto Y1 = at::native::conv2d(X, W, B, S, P, D, groups);
    auto X2 = X.metal();
    at::native::metal::Conv2DParams params{
        X.sizes(), W.sizes(), P, S, D, groups};
    auto Y2 = at::native::metal::mpscnn::conv2d(X2, W, B, params).cpu();
    bool check = almostEqual(Y1, Y2);
    if (check) {
      print(check, @"CONV2D", {N, C, IH, IW});
    } else {
      return false;
    }
  }
  return true;
}

bool test_depthwiseConv() {
#define ARRAY(...) __VA_ARGS__
#define TEST_DEPTHWISECONV(x, w, b, p, g)                                     \
  {                                                                           \
    auto S = c10::IntArrayRef{1, 1};                                          \
    auto D = c10::IntArrayRef{1, 1};                                          \
    auto OP = c10::IntArrayRef({0, 0});                                       \
    auto X = torch::rand(x, at::TensorOptions(at::kCPU).dtype(at::kFloat));   \
    auto W = torch::rand(w, at::TensorOptions(at::kCPU).dtype(at::kFloat));   \
    auto B = torch::rand(b, at::TensorOptions(at::kCPU).dtype(at::kFloat));   \
    auto Y1 = at::native::_convolution(                                       \
        X, W, B, S, p, D, false, OP, g, false, false, true, true);            \
    auto X2 = X.metal();                                                      \
    at::native::metal::Conv2DParams params{X.sizes(), W.sizes(), p, S, D, g}; \
    auto Y2 = at::native::metal::mpscnn::conv2d(X2, W, B, params).cpu();      \
    bool check = almostEqual(Y1, Y2);                                         \
    if (check) {                                                              \
      print(check, @"DEPTHWISECONV", x);                                      \
    } else {                                                                  \
      return false;                                                           \
    }                                                                         \
  }

  TEST_DEPTHWISECONV(
      ARRAY({1, 32, 112, 112}),
      ARRAY({32, 1, 3, 3}),
      ARRAY({32}),
      ARRAY({1, 1}),
      32);

  return true;
}

bool test_max_pool2d() {
  auto X =
      torch::rand({1, 3, 4, 4}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto Y1 = at::native::max_pool2d(X, {2, 2}, {2, 2}, {0, 0}, {1, 1}, false);
  auto X2 = X.metal();
  auto Y2 = at::native::metal::mpscnn::max_pool2d(
                X2, {2, 2}, {2, 2}, {0, 0}, {1, 1}, false)
                .cpu();
  bool check = almostEqual(Y1, Y2);
  if (check) {
    print(check, @"MAX_POOL2D", {1, 3, 4, 4});
  } else {
    return false;
  }
  return true;
}

bool test_relu() {
  auto X =
      torch::rand({1, 3, 4, 4}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto Y1 = torch::native::relu(X);
  auto X2 = X.metal();
  auto Y2 = torch::native::metal::mpscnn::relu(X2).cpu();
  bool check = almostEqual(Y1, Y2);
  if (check) {
    print(check, @"RELU", {1, 3, 4, 4});
  } else {
    return false;
  }
  return true;
}

bool test_sigmoid() {
  auto X =
      torch::rand({1, 3, 4, 4}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto Y1 = torch::native::sigmoid(X);
  auto X2 = X.metal();
  auto Y2 = torch::native::metal::mpscnn::sigmoid(X2).cpu();
  bool check = almostEqual(Y1, Y2);
  print(check, @"SIGMOID", {1, 3, 4, 4});
  return true;
}

bool test_addmm() {
#define ARRAY(...) __VA_ARGS__
#define TEST_ADDMM(x, w, b)                                                   \
  {                                                                           \
    auto X1 = torch::rand(x, at::TensorOptions(at::kCPU).dtype(at::kFloat));  \
    auto W1 = torch::rand(w, at::TensorOptions(at::kCPU).dtype(at::kFloat));  \
    auto B = torch::rand(b, at::TensorOptions(at::kCPU).dtype(at::kFloat));   \
    auto Y1 = at::native::addmm_cpu(B, X1, W1);                               \
    auto X2 = X1.metal();                                                     \
    auto W2 = W1.t().view({W1.sizes()[1], W1.sizes()[0], 1, 1}).contiguous(); \
    auto Y2 = at::native::metal::mpscnn::addmm(B, X2, W2).cpu();              \
    bool check = almostEqual(Y1, Y2);                                         \
    if (check) {                                                              \
      print(check, @"ADDMM", x);                                              \
    } else {                                                                  \
      return false;                                                           \
    }                                                                         \
  }
  for (int i = 0; i < ITER_COUNT; ++i) {
    int64_t N = rand(1, 10);
    int64_t IC = rand(1, 128);
    int64_t OC = rand(1, 128);
    auto X1 =
        torch::rand({N, IC}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto W1 =
        torch::rand({IC, OC}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto B1 =
        torch::rand({1, OC}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::native::addmm_cpu(B1, X1, W1);
    // MPSCNNFullyConnected
    auto X2 = X1.view({N, IC, 1, 1}).contiguous().metal();
    auto W2 = W1.t()
                  .view({W1.sizes()[1], W1.sizes()[0], 1, 1})
                  .contiguous(); // W2 lives in CPU
    auto Y2 = mpscnn::addmm(B1, X2, W2).cpu();
    bool check = almostEqual(Y1, Y2);
    if (check) {
      print(check, @"ADDMM", {N, IC});
    } else {
      return false;
    }
  }
  return true;
}

bool test_add() {
#define ARRAY(...) __VA_ARGS__
#define TEST_ADD(a1, a2)                                                      \
  {                                                                           \
    auto X1 = torch::rand(a1, at::TensorOptions(at::kCPU).dtype(at::kFloat)); \
    auto X2 = torch::rand(a2, at::TensorOptions(at::kCPU).dtype(at::kFloat)); \
    auto Y1 = at::add(X1, X2);                                        \
    auto MX1 = X1.metal();                                                    \
    auto MX2 = X2.metal();                                                    \
    auto Y2 = at::native::metal::mpscnn::add(MX1, MX2).cpu();                 \
    bool check = almostEqual(Y1, Y2);                                         \
    if (check) {                                                              \
      print(check, @"ADD", a1);                                               \
    } else {                                                                  \
      return false;                                                           \
    }                                                                         \
  }
  TEST_ADD(ARRAY({1, 180, 12, 12}), ARRAY({1, 180, 12, 12}));
  return true;
}

bool test_sub() {
#define ARRAY(...) __VA_ARGS__
#define TEST_SUB(a1, a2)                                                      \
  {                                                                           \
    auto X1 = torch::rand(a1, at::TensorOptions(at::kCPU).dtype(at::kFloat)); \
    auto X2 = torch::rand(a2, at::TensorOptions(at::kCPU).dtype(at::kFloat)); \
    auto Y1 = at::native::sub(X1, X2);                                        \
    auto MX1 = X1.metal();                                                    \
    auto MX2 = X2.metal();                                                    \
    auto Y2 = at::native::metal::mpscnn::sub(MX1, MX2).cpu();                 \
    bool check = almostEqual(Y1, Y2);                                         \
    if (check) {                                                              \
      print(check, @"SUB", a1);                                               \
    } else {                                                                  \
      return false;                                                           \
    }                                                                         \
  }
  TEST_SUB(ARRAY({1, 3, 192, 192}), ARRAY({1, 3, 1, 1}));
  return true;
}

bool test_mul() {
#define ARRAY(...) __VA_ARGS__
#define TEST_MUL(a1, a2)                                                      \
  {                                                                           \
    auto X1 = torch::rand(a1, at::TensorOptions(at::kCPU).dtype(at::kFloat)); \
    auto X2 = torch::rand(a2, at::TensorOptions(at::kCPU).dtype(at::kFloat)); \
    auto Y1 = at::native::mul(X1, X2);                                        \
    auto MX1 = X1.metal();                                                    \
    auto MX2 = X2.metal();                                                    \
    auto Y2 = at::native::metal::mpscnn::mul(MX1, MX2).cpu();                 \
    bool check = almostEqual(Y1, Y2);                                         \
    if (check) {                                                              \
      print(check, @"MUL", a1);                                               \
    } else {                                                                  \
      return false;                                                           \
    }                                                                         \
  }
  TEST_MUL(ARRAY({1, 3, 192, 192}), ARRAY({1, 3, 1, 1}));
  return true;
}

bool test_t() {
#define ARRAY(...) __VA_ARGS__
#define TEST_TRANSPOSE(a1)                                                    \
  {                                                                           \
    auto X1 = torch::rand(a1, at::TensorOptions(at::kCPU).dtype(at::kFloat)); \
    auto Y1 = at::native::t(X1).contiguous();                                 \
    auto X2 = X1.metal();                                                     \
    auto Y2 = at::native::metal::mpscnn::t(X2).cpu();                         \
    bool check = almostEqual(Y1, Y2);                                         \
    if (check) {                                                              \
      print(check, @"TRANSPOSE_2D", a1);                                      \
    } else {                                                                  \
      return false;                                                           \
    }                                                                         \
  }
  for (int i = 0; i < ITER_COUNT; ++i) {
    int64_t H = rand(1, 256);
    int64_t W = rand(1, 256);
    auto X1 =
        torch::rand({H, W}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::native::t(X1).contiguous();
    auto X2 = X1.metal();
    auto Y2 = at::native::metal::mpscnn::t(X2).cpu();
    bool check = almostEqual(Y1, Y2);
    if (check) {
      print(check, @"TRANSPOSE_2D", {H, W});
    } else {
      return false;
    }
  }
  return true;
}

bool test_view() {
  auto X1 =
      torch::rand({1, 3, 2, 2}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto Y1 = X1.view({3, 4}).contiguous();
  auto X2 = X1.metal();
  auto Y2 = at::native::metal::mpscnn::view(X2, {3, 4}).cpu();
  bool b1 = (Y1.sizes() == Y2.sizes());
  bool b2 = (Y1.strides() == Y2.strides());
  if (b1 && b2) {
    print(true, @"VIEW", {1, 3, 2, 2});
  } else {
    return false;
  }
  return true;
}

bool test_softmax() {
  auto X1 =
      torch::rand({2, 3, 1, 1}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto Y1 = torch::native::log_softmax(X1, 1);
  auto X2 = X1.metal();
  auto Y2 = torch::native::metal::mpscnn::log_softmax_int(X2).cpu();
  bool check = almostEqual(Y1, Y2);
  print(check, @"SOFTMAX", {2, 3, 1, 1});
  return check;
}

bool test_upsampling_nearest2d_vec() {
  auto X1 = torch::rand(
      {1, 48, 24, 24}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto Y1 = torch::native::upsample_nearest2d_cpu(
      X1,
      c10::optional<IntArrayRef>({}),
      c10::optional<ArrayRef<double>>({2, 2}));
  auto X2 = X1.metal();
  auto Y2 = torch::native::metal::mpscnn::upsample_nearest2d_vec(
                X2,
                c10::optional<IntArrayRef>({}),
                c10::optional<ArrayRef<double>>({2, 2}))
                .cpu();
  bool check = almostEqual(Y1, Y2);
  print(check, @"UPSAMPLING_NEAREST2D", {1, 48, 24, 24});
  return check;
}

bool test_adaptive_avg_pool2d() {
  auto X1 = torch::rand(
      {1, 48, 24, 24}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto Y1 = at::native::adaptive_avg_pool2d(X1, {1, 1});
  auto X2 = X1.metal();
  auto Y2 = torch::native::metal::mpscnn::global_avg_pool2d(X2, {1, 1}).cpu();
  bool check = almostEqual(Y1, Y2);
  print(check, @"ADAPTIVE_AVG_POOL2D", {1, 48, 24, 24});
  return check;
}

bool test_reshape() {
  auto X1 = torch::rand(
      {1, 1280, 1, 1}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto Y1 = at::native::reshape(X1, {1, -1});
  auto X2 = X1.metal();
  auto Y2 = torch::native::metal::mpscnn::reshape(X2, {1, -1}).cpu();
  bool check = almostEqual(Y1, Y2);
  print(check, @"RESHAPE", {1, 1280, 1, 1});
  return check;
}

bool test_hardtanh_() {
  auto X1 = torch::rand(
      {1, 32, 112, 112}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto Y1 = at::native::hardtanh_(X1, 0, 6.0);
  auto X2 = X1.metal();
  auto Y2 = at::native::metal::mpscnn::hardtanh_(X2, 0, 6.0).cpu();
  bool check = almostEqual(Y1, Y2);
  print(check, @"HARDTANH_", {1, 32, 112, 112});
  return check;
}

}
}
}
