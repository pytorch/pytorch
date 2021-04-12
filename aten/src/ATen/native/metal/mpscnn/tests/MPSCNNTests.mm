#import <ATen/ATen.h>
#import <ATen/native/metal/MetalUtils.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>
#import <ATen/native/metal/mpscnn/MPSImageUtils.h>
#import <ATen/native/metal/mpscnn/tests/MPSCNNTests.h>
#import <ATen/native/metal/ops/MetalConvolution.h>
#import <ATen/native/metal/ops/MetalTranspose.h>

#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

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
  c10::InferenceMode guard;
  bool b = block();
  void (^print)(NSString*) = ^(NSString* result) {
    NSLog(@"[%s],[%s],[%@]", name.c_str(), str1.c_str(), result);
  };
  b ? print(@"SUCCEED") : print(@"FAILED");
  return b;
}

void PRINT_TENSOR(std::string name, const at::Tensor& tensor) {
  std::string str = name + ": ";
  auto print = [&](const at::Tensor& t) {
    for (int i = 0; i < t.numel(); ++i) {
      NSString* sf =
          [NSString stringWithFormat:@"%.2f", t.data_ptr<float>()[i]];
      str += sf.UTF8String;
      str += ", ";
    }
    std::cout << str << std::endl;
  };
  if (tensor.is_metal()) {
    MPSImage* image = at::native::metal::imageFromTensor(tensor);
    auto t = at::native::metal::staticImageToTensor(image);
    print(t);
  } else {
    print(tensor);
  }
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
      auto c4 = NCHWToNC4((float*)t.data_ptr<float>(), t.sizes().vec());
      auto n4 = NC4ToNCHW((float*)c4.data(), t.sizes().vec());
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
        createTemporaryImage(cb, t1.sizes().vec(), t1.data_ptr<float>());
    MPSImage* img2 = createStaticImage(img1, cb, true);
    auto t2 = at::zeros(size);
    copyToHost(t2.data_ptr<float>(), img2);
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
      auto S = std::vector<int64_t>({SH, SW});
      auto P = std::vector<int64_t>({PH, PW});
      // Dilated convolution is not supported yet
      auto D = std::vector<int64_t>({1, 1});
      int64_t groups = 1;
      auto Y1 = at::conv2d(X, W, B, S, P, D, groups);
      auto X2 = X.metal();
      auto Y2 = at::conv2d(X2, W, B, S, P, D, groups).cpu();
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
    auto Y1 = at::_convolution(
        X, W, B, {1, 1}, p, {1, 1}, false, {0, 0}, g, false, false, true, true);
    auto X2 = X.metal();
    Conv2DParams params{X.sizes(), W.sizes(), p, S, D, g};
    if (!params.isDepthwise()) {
      return false;
    }
    auto Y2 = at::conv2d(X2, W, B, S, p, D, g).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_max_pool2d() {
  __block std::vector<int64_t> size{1, 3, 4, 4};
  return TEST(size, __PRETTY_FUNCTION__, ^bool {
    auto X = at::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::max_pool2d(X, {2, 2}, {2, 2}, {0, 0}, {1, 1}, false);
    auto X2 = X.metal();
    auto Y2 = at::max_pool2d(X2, {2, 2}, {2, 2}, {0, 0}, {1, 1}, false).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_max_pool2d_padding() {
  __block std::vector<int64_t> size{1, 3, 4, 4};
  return TEST(size, __PRETTY_FUNCTION__, ^bool {
    auto X = at::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::max_pool2d(X, {2, 2}, {2, 2}, {1, 1}, {1, 1}, false);
    auto X2 = X.metal();
    auto Y2 = at::max_pool2d(X2, {2, 2}, {2, 2}, {1, 1}, {1, 1}, false).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_max_pool2d_ceil() {
  __block std::vector<int64_t> size{1, 96, 55, 55};
  return TEST(size, __PRETTY_FUNCTION__, ^bool {
    auto X = at::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::max_pool2d(X, {3, 3}, {2, 2}, {0, 0}, {1, 1}, true);
    auto X2 = X.metal();
    auto Y2 = at::max_pool2d(X2, {3, 3}, {2, 2}, {0, 0}, {1, 1}, true).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_relu() {
  __block std::vector<int64_t> size{1, 3, 4, 4};
  return TEST(size, __PRETTY_FUNCTION__, ^bool {
    auto X = at::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::relu(X);
    auto X2 = X.metal();
    auto Y2 = at::relu(X2).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_sigmoid() {
  __block std::vector<int64_t> size{1, 3, 4, 4};
  return TEST(size, __PRETTY_FUNCTION__, ^bool {
    auto X = at::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::sigmoid(X);
    auto X2 = X.metal();
    auto Y2 = at::sigmoid(X2).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_hardsigmoid() {
  __block std::vector<int64_t> size{3, 3, 44, 44};
  return TEST(size, __PRETTY_FUNCTION__, ^bool {
    auto X =
        at::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat)) * 12 - 6;
    auto X2 = X.metal();
    auto Y1 = at::hardsigmoid_(X);
    auto Y2 = at::hardsigmoid_(X2).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_hardswish() {
  __block std::vector<int64_t> size{3, 3, 44, 44};
  return TEST(size, __PRETTY_FUNCTION__, ^bool {
    auto X =
        at::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat)) * 12 - 6;
    auto X2 = X.metal();
    auto Y1 = at::hardswish_(X);
    auto Y2 = at::hardswish_(X2).cpu();
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
      auto Y1 = at::addmm(B1, X1, W1);
      auto X2 = X1.metal();
      auto Y2 = at::addmm(B1, X2, W1).cpu();
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
    auto Y2 = at::add(MX1, MX2).cpu();
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
    auto Y2 = at::add(MX1, MX2).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_add_broadcast2() {
  __block std::vector<int64_t> x1{2, 17, 1, 67};
  __block std::vector<int64_t> x2{2, 17, 58, 67};
  return TEST(x1, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(x1, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto X2 = at::rand(x2, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::add(X1, X2);
    auto MX1 = X1.metal();
    auto MX2 = X2.metal();
    auto Y2 = at::add(MX1, MX2).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_sub() {
  __block std::vector<int64_t> x{5, 3, 167, 222};
  return TEST(x, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(x, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto X2 = at::rand(x, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::sub(X1, X2);
    auto MX1 = X1.metal();
    auto MX2 = X2.metal();
    auto Y2 = at::sub(MX1, MX2).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_sub_broadcast() {
  __block std::vector<int64_t> x1{3, 1, 1};
  __block std::vector<int64_t> x2{3, 192, 192};
  return TEST(x1, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(x1, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto X2 = at::rand(x2, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::sub(X1, X2);
    auto MX1 = X1.metal();
    auto MX2 = X2.metal();
    auto Y2 = at::sub(MX1, MX2).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_sub_broadcast2() {
  __block std::vector<int64_t> x1{2, 3, 3, 192, 192};
  __block std::vector<int64_t> x2{2, 3, 3, 1, 192};
  return TEST(x1, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(x1, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto X2 = at::rand(x2, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::sub(X1, X2);
    auto MX1 = X1.metal();
    auto MX2 = X2.metal();
    auto Y2 = at::sub(MX1, MX2).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_mul() {
  __block std::vector<int64_t> x{2, 7, 262, 119};
  return TEST(x, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(x, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto X2 = at::rand(x, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::mul(X1, X2);
    auto MX1 = X1.metal();
    auto MX2 = X2.metal();
    auto Y2 = at::mul(MX1, MX2).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_mul_broadcast() {
  __block std::vector<int64_t> x1{4, 3, 192, 192};
  __block std::vector<int64_t> x2{4, 3, 1, 1};
  return TEST(x1, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(x1, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto X2 = at::rand(x2, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::mul(X1, X2);
    auto MX1 = X1.metal();
    auto MX2 = X2.metal();
    auto Y2 = at::mul(MX1, MX2).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_mul_broadcast2() {
  __block std::vector<int64_t> x2{1, 3, 192, 1};
  __block std::vector<int64_t> x1{1, 3, 192, 192};
  return TEST(x1, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(x1, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto X2 = at::rand(x2, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::mul(X1, X2);
    auto MX1 = X1.metal();
    auto MX2 = X2.metal();
    auto Y2 = at::mul(MX1, MX2).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_div() {
  __block std::vector<int64_t> x{1, 3, 24, 24};
  return TEST(x, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(x, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto X2 = at::rand(x, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::div(X1, X2);
    auto MX1 = X1.metal();
    auto MX2 = X2.metal();
    auto Y2 = at::div(MX1, MX2).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_div_broadcast() {
  __block std::vector<int64_t> x1{4, 3, 24, 24};
  __block std::vector<int64_t> x2{4, 3, 1, 1};
  return TEST(x1, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(x1, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto X2 = at::rand(x2, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::div(X1, X2);
    auto MX1 = X1.metal();
    auto MX2 = X2.metal();
    auto Y2 = at::div(MX1, MX2).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_div_broadcast2() {
  __block std::vector<int64_t> x2{1, 3, 24, 1};
  __block std::vector<int64_t> x1{1, 3, 24, 24};
  return TEST(x1, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(x1, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto X2 = at::rand(x2, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::div(X1, X2);
    auto MX1 = X1.metal();
    auto MX2 = X2.metal();
    auto Y2 = at::div(MX1, MX2).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_t() {
  bool result = true;
  for (int i = 0; i < ITER_COUNT; ++i) {
    int64_t H = rand(1, 256);
    int64_t W = rand(1, 256);
    bool b = TEST({H, W}, __PRETTY_FUNCTION__, ^bool {
      auto X1 = at::rand({H, W}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
      auto Y1 = at::t(X1).contiguous();
      auto X2 = X1.metal();
      auto Y2 = at::native::metal::t(X2).cpu();
      return almostEqual(Y1, Y2);
    });
    if (!b) {
      result = false;
    }
  }
  return result;
}

bool test_view() {
  // array -> array
  __block std::vector<int64_t> size{1, 10, 2, 2};
  return TEST(size, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = X1.view({5, 4, 2}).contiguous();
    auto X2 = X1.metal();
    auto Y2 = X2.view({5, 4, 2}).cpu();
    bool b1 = (Y1.sizes() == Y2.sizes());
    bool b2 = (Y1.strides() == Y2.strides());
    bool b3 = almostEqual(Y1, Y2);
    return b1 && b2 && b3;
  });
}

bool test_view2() {
  // array -> nonarray
  __block std::vector<int64_t> size{1, 10, 2, 2};
  return TEST(size, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = X1.view({5, 8}).contiguous();
    auto X2 = X1.metal();
    auto Y2 = X2.view({5, 8}).cpu();
    bool b1 = (Y1.sizes() == Y2.sizes());
    bool b2 = (Y1.strides() == Y2.strides());
    bool b3 = almostEqual(Y1, Y2);
    return b1 && b2 && b3;
  });
}

bool test_view3() {
  // nonarry -> array
  __block std::vector<int64_t> size{5, 8};
  return TEST(size, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = X1.view({1, 10, 2, 2}).contiguous();
    auto X2 = X1.metal();
    auto Y2 = X2.view({1, 10, 2, 2}).cpu();
    bool b1 = (Y1.sizes() == Y2.sizes());
    bool b2 = (Y1.strides() == Y2.strides());
    bool b3 = almostEqual(Y1, Y2);
    return b1 && b2 && b3;
  });
}

bool test_view4() {
  // nonarray -> nonarray
  __block std::vector<int64_t> size{5, 8};
  return TEST(size, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = X1.view({4, 10}).contiguous();
    auto X2 = X1.metal();
    auto Y2 = X2.view({4, 10}).cpu();
    bool b1 = (Y1.sizes() == Y2.sizes());
    bool b2 = (Y1.strides() == Y2.strides());
    bool b3 = almostEqual(Y1, Y2);
    return b1 && b2 && b3;
  });
}

bool test_cat_dim0() {
  __block std::vector<int64_t> x1{3, 9, 221, 193};
  __block std::vector<int64_t> x2{5, 9, 221, 193};
  __block std::vector<int64_t> x3{7, 9, 221, 193};
  return TEST(x1, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(x1, at::TensorOptions(at::kCPU).dtype(at::kFloat)) * 100;
    auto X2 = at::rand(x2, at::TensorOptions(at::kCPU).dtype(at::kFloat)) * 100;
    auto X3 = at::rand(x3, at::TensorOptions(at::kCPU).dtype(at::kFloat)) * 100;
    auto Y = at::cat({X1, X2, X3}, 0);

    auto MX1 = X1.metal();
    auto MX2 = X2.metal();
    auto MX3 = X3.metal();
    auto MY = at::cat({MX1, MX2, MX3}, 0).cpu();

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
    auto MY = at::cat({MX1, MX2, MX3}, 0).cpu();

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
    auto MY = at::cat({MX1, MX2, MX3}, 1).cpu();

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
    auto MY = at::cat({MX1, MX2, MX3}, 1).cpu();

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
    auto MY = at::cat({MX1, MX2, MX3}, 1).cpu();

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
    auto MY = at::cat({MX1, MX2, MX3}, 1).cpu();

    return almostEqual(Y, MY);
  });
#else
  // Skip this test on MacOS, shader behaves unexpectedly on sandcastle machines
  // Will get back and fix it - T84963816
  return true;
#endif
}

bool test_softmax() {
    __block std::vector<int64_t> size{2,2};
    return TEST(size, __PRETTY_FUNCTION__, ^bool {
      auto X1 = at::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat));
      auto Y1 = at::softmax(X1, 0);
      auto X2 = X1.metal();
      auto Y2 = at::softmax(X2, 0).cpu();
      return almostEqual(Y1, Y2);
    });
}
bool test_log_softmax() {
    __block std::vector<int64_t> size{2,2};
    return TEST(size, __PRETTY_FUNCTION__, ^bool {
      auto X1 = at::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat));
      auto Y1 = at::log_softmax(X1, 1);
      auto X2 = X1.metal();
      auto Y2 = at::log_softmax(X2, 1).cpu();
      return almostEqual(Y1, Y2);
    });
}

bool test_upsampling_nearest2d_vec() {
  __block std::vector<int64_t> size{1, 48, 24, 24};
  return TEST(size, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::upsample_nearest2d(
        X1,
        c10::optional<at::IntArrayRef>({}),
        c10::optional<at::ArrayRef<double>>({2, 2}));
    auto X2 = X1.metal();
    auto Y2 = at::upsample_nearest2d(
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
    auto Y1 = at::adaptive_avg_pool2d(X1, {1, 1});
    auto X2 = X1.metal();
    auto Y2 = at::adaptive_avg_pool2d(X2, {1, 1}).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_reshape() {
  __block std::vector<int64_t> size{1, 1280, 1, 1};
  return TEST(size, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::reshape(X1, {1, -1});
    auto X2 = X1.metal();
    auto Y2 = at::reshape(X2, {1, -1}).cpu();
    return almostEqual(Y1, Y2);
  });
}

bool test_hardtanh_() {
#if TARGET_OS_IPHONE
  __block std::vector<int64_t> size{1, 32, 112, 112};
  return TEST(size, __PRETTY_FUNCTION__, ^bool {
    auto X1 = at::rand(size, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto Y1 = at::hardtanh_(X1, 0, 6.0);
    auto X2 = X1.metal();
    auto Y2 = at::hardtanh_(X2, 0, 6.0).cpu();
    return almostEqual(Y1, Y2);
  });
#else
  // Skip this test on MacOS as the shader function doesn't work well
  // Will get back and fix it - T82700462
  return true;
#endif
}
