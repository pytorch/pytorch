#import <ATen/native/metal/MetalConvolution.h>
#import <ATen/native/metal/MetalUtils.h>
#import <ATen/native/metal/mpscnn/MPSCNNOps.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>
#import <ATen/native/metal/mpscnn/tests/MPSCNNTests.h>
#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <ATen/ATen.h>
#include <ATen/Utils.h>
#include <c10/util/accumulate.h>
#import <ATen/native/metal/mpscnn/tests/MPSCNNTests.h>

#include <stdlib.h>
#include <torch/script.h>
#include <sstream>

#define ITER_COUNT 5

namespace {

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
  return checkRtol(a - b, {a, b}) && a.strides().vec() == b.strides().vec();
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

typedef bool (^Func)(void);
bool TEST(const std::vector<int64_t>& sizes, std::string name, Func block) {
  std::stringstream ss;
  std::copy(sizes.begin(), sizes.end(), std::ostream_iterator<int>(ss, " "));
  __block std::string str1 = ss.str();
  bool b = block();
  void (^print)(NSString*) = ^(NSString* result) {
    NSLog(@"[%s],[%s],[%@]", name.c_str(), str1.c_str(), result);
  };
  b ? print(@"SUCCEED") : print(@"FAILED");
  return b;
}

}

using namespace at::native::metal;

bool test_synchronization() {
  __block std::vector<int64_t> size{1, 3, 2, 2};
  return TEST(size, __PRETTY_FUNCTION__, ^bool(void) {
    auto x1 = at::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto mx1 = x1.metal();
    TORCH_CHECK(mx1.device().type() == at::kMetal);
    auto x2 = mx1.cpu();
    TORCH_CHECK(x2.device().type() == at::kCPU);
    return almostEqual(x1, x2);
  });
}

bool test_nchw_to_nc4_cpu() {
  bool result = true;
  for (int i = 0; i < ITER_COUNT; ++i) {
    int64_t N = rand(1, 24);
    int64_t C = rand(1, 48);
    int64_t H = rand(1, 320);
    int64_t W = rand(1, 320);
    __block std::vector<int64_t> size{N, C, H, W};
    bool b = TEST(size, __PRETTY_FUNCTION__, ^bool {
      auto t = at::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat));
      const auto len = c10::multiply_integers(std::begin(size), std::end(size));
      auto buf =
          std::vector<float>{t.data_ptr<float>(), t.data_ptr<float>() + len};
      auto c4 = NCHW_to_NC4((float*)t.data_ptr<float>(), t.sizes().vec());
      auto n4 = NC4_to_NCHW((float*)c4.data(), t.sizes().vec());
      return n4 == buf;
    });
    if (!b) {
      result = false;
    }
  }
  return result;
}

bool test_copy_nchw_to_metal() {
  __block std::vector<int64_t> size{1, 3, 224, 224};
  return TEST(size, __PRETTY_FUNCTION__, ^bool(void) {
    auto t1 = at::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    MetalCommandBuffer* cb = [MetalCommandBuffer newBuffer];
    MPSTemporaryImage* img1 =
        [MPSImage temporaryImageFromHost:t1.data_ptr<float>()
                                   Sizes:t1.sizes().vec()
                           CommandBuffer:cb];
    MPSImage* img2 = [MPSImage imageFromTemporaryImage:img1
                                         CommandBuffer:cb
                                    waitUntilCompleted:YES];
    auto t2 = at::zeros(size);
    [MPSImage copyToHost:t2.data_ptr<float>() FromImage:img2];
    return almostEqual(t1, t2);
  });
}

bool test_conv2d() {
  bool result = true;
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
    bool b = TEST({N, C, IH, IW}, __PRETTY_FUNCTION__, ^bool {
      auto X = at::rand(
          {N, C, IH, IW}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
      auto W = at::rand(
          {OC, IC, KH, KW}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
      auto B = at::rand({OC}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
      auto S = c10::IntArrayRef({SH, SW});
      auto P = c10::IntArrayRef({PH, PW});
      auto D =
          c10::IntArrayRef({1, 1}); // Dilated convolution is not supported yet
      int64_t groups = 1;
      auto Y1 = at::native::conv2d(X, W, B, S, P, D, groups);
      auto X2 = X.metal();
      Conv2DParams params{X.sizes(), W.sizes(), P, S, D, groups};
      auto Y2 = mpscnn::conv2d(X2, W, B, params).cpu();
      return almostEqual(Y1, Y2);
    });
    if (!b) {
      result = false;
    }
  }
  return result;
}

bool test_depthwiseConv() {
  __block std::vector<int64_t> x{1, 32, 112, 112};
  __block std::vector<int64_t> w{32, 1, 3, 3};
  __block std::vector<int64_t> b{32};
  __block std::vector<int64_t> p{1, 1};
  int g = 32;
  return TEST(x, __PRETTY_FUNCTION__, ^bool {
    auto S = std::vector<int64_t>{1, 1};
    auto D = std::vector<int64_t>{1, 1};
    auto OP = std::vector<int64_t>({0, 0});
    auto X = at::rand(x, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto W = at::rand(w, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto B = at::rand(b, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::native::_convolution(
        X, W, B, {1, 1}, p, {1, 1}, false, {0, 0}, g, false, false, true, true);
    auto X2 = X.metal();
    Conv2DParams params{X.sizes(), W.sizes(), p, S, D, g};
    if (!params.isDepthwise()) {
      return false;
    }
    auto Y2 = mpscnn::conv2d(X2, W, B, params).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_max_pool2d() {
  __block std::vector<int64_t> size{1, 3, 4, 4};
  return TEST(size, __PRETTY_FUNCTION__, ^bool {
    auto X = at::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::native::max_pool2d(X, {2, 2}, {2, 2}, {0, 0}, {1, 1}, false);
    auto X2 = X.metal();
    auto Y2 =
        mpscnn::max_pool2d(X2, {2, 2}, {2, 2}, {0, 0}, {1, 1}, false).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_max_pool2d_padding() {
    __block std::vector<int64_t> size{1, 3, 4, 4};
    return TEST(size, __PRETTY_FUNCTION__, ^bool {
      auto X = at::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat));
      auto Y1 = at::native::max_pool2d(X, {2, 2}, {2, 2}, {1, 1}, {1, 1}, false);
      auto X2 = X.metal();
      auto Y2 =
          mpscnn::max_pool2d(X2, {2, 2}, {2, 2}, {1, 1}, {1, 1}, false).cpu();
      return almostEqual(Y1, Y2);
    });
}

bool test_max_pool2d_ceil() {
  __block std::vector<int64_t> size{1, 96, 55, 55};
  return TEST(size, __PRETTY_FUNCTION__, ^bool {
    auto X = at::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::native::max_pool2d(X, {3, 3}, {2, 2}, {0, 0}, {1, 1}, true);
    auto X2 = X.metal();
    auto Y2 =
        mpscnn::max_pool2d(X2, {3, 3}, {2, 2}, {0, 0}, {1, 1}, true).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_relu() {
  __block std::vector<int64_t> size{1, 3, 4, 4};
  return TEST(size, __PRETTY_FUNCTION__, ^bool {
    auto X = at::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = torch::native::relu(X);
    auto X2 = X.metal();
    auto Y2 = mpscnn::relu(X2).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_sigmoid() {
  __block std::vector<int64_t> size{1, 3, 4, 4};
  return TEST(size, __PRETTY_FUNCTION__, ^bool {
    auto X = at::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::native::sigmoid(X);
    auto X2 = X.metal();
    auto Y2 = mpscnn::sigmoid(X2).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_hardsigmoid() {
  __block std::vector<int64_t> size{3, 3, 44, 44};
  return TEST(size, __PRETTY_FUNCTION__, ^bool {
    auto X = at::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat))*12 - 6;
    auto X2 = X.metal();
    auto Y1 = at::native::hardsigmoid_(X);
    auto Y2 = mpscnn::hardsigmoid_(X2).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_hardswish() {
  __block std::vector<int64_t> size{3, 3, 44, 44};
  return TEST(size, __PRETTY_FUNCTION__, ^bool {
    auto X = at::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat))*12 - 6;
    auto X2 = X.metal();
    auto Y1 = at::native::hardswish_(X);
    auto Y2 = mpscnn::hardswish_(X2).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_addmm() {
  bool result = true;
  for (int i = 0; i < ITER_COUNT; ++i) {
    int64_t N = rand(1, 10);
    int64_t IC = rand(1, 128);
    int64_t OC = rand(1, 128);
    bool b = TEST({N, IC, OC}, __PRETTY_FUNCTION__, ^bool {
      auto X1 =
          at::rand({N, IC}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
      auto W1 =
          at::rand({IC, OC}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
      auto B1 =
          at::rand({1, OC}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
      auto Y1 = at::native::addmm_cpu(B1, X1, W1);
      auto X2 = X1.view({N, IC, 1, 1}).contiguous().metal();
      auto W2 = W1.t().view({W1.sizes()[1], W1.sizes()[0], 1, 1}).contiguous();
      auto Y2 = mpscnn::addmm(B1, X2, W2).cpu();
      return almostEqual(Y1, Y2);
    });
    if (!b) {
      result = false;
    }
  }
  return result;
}

bool test_add() {
  __block std::vector<int64_t> x{1, 180, 12, 12};
  return TEST(x, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(x, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto X2 = at::rand(x, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::add(X1, X2);
    auto MX1 = X1.metal();
    auto MX2 = X2.metal();
    auto Y2 = mpscnn::add(MX1, MX2).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_add_broadcast() {
  __block std::vector<int64_t> x1{2, 17, 58, 67};
  __block std::vector<int64_t> x2{2, 17, 1, 1};
  return TEST(x1, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(x1, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto X2 = at::rand(x2, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::add(X1, X2);
    auto MX1 = X1.metal();
    auto MX2 = X2.metal();
    auto Y2 = mpscnn::add(MX1, MX2).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_sub() {
  __block std::vector<int64_t> x{5, 3, 167, 222};
  return TEST(x, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(x, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto X2 = at::rand(x, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::native::sub(X1, X2);
    auto MX1 = X1.metal();
    auto MX2 = X2.metal();
    auto Y2 = mpscnn::sub(MX1, MX2).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_sub_broadcast() {
  __block std::vector<int64_t> x1{3, 3, 1, 1};
  __block std::vector<int64_t> x2{3, 3, 192, 192};
  return TEST(x1, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(x1, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto X2 = at::rand(x2, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::native::sub(X1, X2);
    auto MX1 = X1.metal();
    auto MX2 = X2.metal();
    auto Y2 = mpscnn::sub(MX1, MX2).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_sub_broadcast2() {
  __block std::vector<int64_t> x1{3, 3, 192, 192};
  __block std::vector<int64_t> x2{3, 3, 1, 192};
  return TEST(x1, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(x1, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto X2 = at::rand(x2, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::native::sub(X1, X2);
    auto MX1 = X1.metal();
    auto MX2 = X2.metal();
    auto Y2 = mpscnn::sub(MX1, MX2).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_mul() {
  __block std::vector<int64_t> x{2, 7, 262, 119};
  return TEST(x, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(x, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto X2 = at::rand(x, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::native::mul(X1, X2);
    auto MX1 = X1.metal();
    auto MX2 = X2.metal();
    auto Y2 = mpscnn::mul(MX1, MX2).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_mul_broadcast() {
  __block std::vector<int64_t> x1{4, 3, 192, 192};
  __block std::vector<int64_t> x2{4, 3, 1, 1};
  return TEST(x1, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(x1, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto X2 = at::rand(x2, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::native::mul(X1, X2);
    auto MX1 = X1.metal();
    auto MX2 = X2.metal();
    auto Y2 = mpscnn::mul(MX1, MX2).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_mul_broadcast2() {
  __block std::vector<int64_t> x1{4, 3, 192, 1};
  __block std::vector<int64_t> x2{4, 3, 192, 192};
  return TEST(x1, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(x1, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto X2 = at::rand(x2, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::native::mul(X1, X2);
    auto MX1 = X1.metal();
    auto MX2 = X2.metal();
    auto Y2 = mpscnn::mul(MX1, MX2).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_t() {
  bool result = true;
  for (int i = 0; i < ITER_COUNT; ++i) {
    int64_t H = rand(1, 256);
    int64_t W = rand(1, 256);
    bool b = TEST({H, W}, __PRETTY_FUNCTION__, ^bool {
      auto X1 =
          torch::rand({H, W}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
      auto Y1 = at::native::t(X1).contiguous();
      auto X2 = X1.metal();
      auto Y2 = mpscnn::t(X2).cpu();
      return almostEqual(Y1, Y2);
    });
    if (!b) {
      result = false;
    }
  }
  return result;
}

bool test_view() {
  __block std::vector<int64_t> size{1, 3, 2, 2};
  return TEST(size, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = X1.view({3, 4}).contiguous();
    auto X2 = X1.metal();
    auto Y2 = mpscnn::view(X2, {3, 4}).cpu();
    bool b1 = (Y1.sizes() == Y2.sizes());
    bool b2 = (Y1.strides() == Y2.strides());
    return b1 && b2;
  });
}

bool test_cat_dim0() {
  __block std::vector<int64_t> x1{3, 9, 221, 193};
  __block std::vector<int64_t> x2{5, 9, 221, 193};
  __block std::vector<int64_t> x3{7, 9, 221, 193};
  return TEST(x1, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(x1, at::TensorOptions(at::kCPU).dtype(at::kFloat))*100;
    auto X2 = at::rand(x2, at::TensorOptions(at::kCPU).dtype(at::kFloat))*100;
    auto X3 = at::rand(x3, at::TensorOptions(at::kCPU).dtype(at::kFloat))*100;
    auto Y = at::cat({X1, X2, X3}, 0);

    auto MX1 = X1.metal();
    auto MX2 = X2.metal();
    auto MX3 = X3.metal();
    auto MY = mpscnn::cat({MX1, MX2, MX3}, 0).cpu();

    return almostEqual(Y, MY);
  });
}

bool test_cat_dim0_nonarray() {
  __block std::vector<int64_t> x1{1, 3, 90, 77};
  __block std::vector<int64_t> x2{1, 3, 90, 77};
  __block std::vector<int64_t> x3{1, 3, 90, 77};
  return TEST(x1, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(x1, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto X2 = at::rand(x2, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto X3 = at::rand(x3, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y = at::cat({X1, X2, X3}, 0);

    auto MX1 = X1.metal();
    auto MX2 = X2.metal();
    auto MX3 = X3.metal();
    auto MY = mpscnn::cat({MX1, MX2, MX3}, 0).cpu();

    return almostEqual(Y, MY);
  });
}

bool test_cat_dim1_0() {
#if TARGET_OS_IPHONE
  __block std::vector<int64_t> x1{4, 10, 271, 333};
  __block std::vector<int64_t> x2{4, 15, 271, 333};
  __block std::vector<int64_t> x3{4, 16, 271, 333};
  return TEST(x1, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(x1, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto X2 = at::rand(x2, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto X3 = at::rand(x3, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y = at::cat({X1, X2, X3}, 1);

    auto MX1 = X1.metal();
    auto MX2 = X2.metal();
    auto MX3 = X3.metal();
    auto MY = mpscnn::cat({MX1, MX2, MX3}, 1).cpu();

    return almostEqual(Y, MY);
  });
#else
  // Skip this test on MacOS, shader behaves unexpectedly on sandcastle machines
  // Will get back and fix it - T84963816
  return true;
#endif
}

bool test_cat_dim1_1() {
#if TARGET_OS_IPHONE
  __block std::vector<int64_t> x1{3, 11, 271, 333};
  __block std::vector<int64_t> x2{3, 17, 271, 333};
  __block std::vector<int64_t> x3{3, 21, 271, 333};
  return TEST(x1, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(x1, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto X2 = at::rand(x2, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto X3 = at::rand(x3, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y = at::cat({X1, X2, X3}, 1);

    auto MX1 = X1.metal();
    auto MX2 = X2.metal();
    auto MX3 = X3.metal();
    auto MY = mpscnn::cat({MX1, MX2, MX3}, 1).cpu();

    return almostEqual(Y, MY);
  });
#else
  // Skip this test on MacOS, shader behaves unexpectedly on sandcastle machines
  // Will get back and fix it - T84963816
  return true;
#endif
}

bool test_cat_dim1_nonarray_0() {
#if TARGET_OS_IPHONE
  __block std::vector<int64_t> x1{1, 3, 22, 33};
  __block std::vector<int64_t> x2{1, 2, 22, 33};
  __block std::vector<int64_t> x3{1, 1, 22, 33};
  return TEST(x1, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(x1, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto X2 = at::rand(x2, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto X3 = at::rand(x3, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y = at::cat({X1, X2, X3}, 1);

    auto MX1 = X1.metal();
    auto MX2 = X2.metal();
    auto MX3 = X3.metal();
    auto MY = mpscnn::cat({MX1, MX2, MX3}, 1).cpu();

    return almostEqual(Y, MY);
  });
#else
  // Skip this test on MacOS, shader behaves unexpectedly on sandcastle machines
  // Will get back and fix it - T84963816
  return true;
#endif
}

bool test_cat_dim1_nonarray_1() {
#if TARGET_OS_IPHONE
  __block std::vector<int64_t> x1{1, 9, 53, 67};
  __block std::vector<int64_t> x2{1, 2, 53, 67};
  __block std::vector<int64_t> x3{1, 3, 53, 67};
  return TEST(x1, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(x1, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto X2 = at::rand(x2, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto X3 = at::rand(x3, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y = at::cat({X1, X2, X3}, 1);

    auto MX1 = X1.metal();
    auto MX2 = X2.metal();
    auto MX3 = X3.metal();
    auto MY = mpscnn::cat({MX1, MX2, MX3}, 1).cpu();

    return almostEqual(Y, MY);
  });
#else
  // Skip this test on MacOS, shader behaves unexpectedly on sandcastle machines
  // Will get back and fix it - T84963816
  return true;
#endif
}

bool test_softmax() {
  __block std::vector<int64_t> size{2, 3, 1, 1};
  return TEST(size, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::native::log_softmax(X1, 1);
    auto X2 = X1.metal();
    auto Y2 = mpscnn::log_softmax_int(X2).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_upsampling_nearest2d_vec() {
  __block std::vector<int64_t> size{1, 48, 24, 24};
  return TEST(size, __PRETTY_FUNCTION__, ^bool {
    auto X1 = torch::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::native::upsample_nearest2d(
        X1,
        c10::optional<at::IntArrayRef>({}),
        c10::optional<at::ArrayRef<double>>({2, 2}));
    auto X2 = X1.metal();
    auto Y2 = mpscnn::upsample_nearest2d_vec(
                  X2,
                  c10::optional<at::IntArrayRef>({}),
                  c10::optional<at::ArrayRef<double>>({2, 2}))
                  .cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_adaptive_avg_pool2d() {
  __block std::vector<int64_t> size{1, 48, 24, 24};
  return TEST(size, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::native::adaptive_avg_pool2d(X1, {1, 1});
    auto X2 = X1.metal();
    auto Y2 = mpscnn::global_avg_pool2d(X2, {1, 1}).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_reshape() {
  __block std::vector<int64_t> size{1, 1280, 1, 1};
  return TEST(size, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::native::reshape(X1, {1, -1});
    auto X2 = X1.metal();
    auto Y2 = torch::native::metal::mpscnn::reshape(X2, {1, -1}).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_hardtanh_() {
#if TARGET_OS_IPHONE
  __block std::vector<int64_t> size{1, 32, 112, 112};
  return TEST(size, __PRETTY_FUNCTION__, ^bool {
    auto X1 = torch::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::native::hardtanh_(X1, 0, 6.0);
    auto X2 = X1.metal();
    auto Y2 = at::native::metal::mpscnn::hardtanh_(X2, 0, 6.0).cpu();
    return almostEqual(Y1, Y2);
  });
#else
    // Skip this test on MacOS as the shader function doesn't work well
    // Will get back and fix it - T82700462
    return true;
#endif
}
