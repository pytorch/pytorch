#include <gtest/gtest.h>

#include "ATen/ATen.h"
#include "ATen/vulkan/Context.h"

bool checkRtol(const at::Tensor& diff, const std::vector<at::Tensor> inputs) {
  double maxValue = 0.0;
  for (auto& tensor : inputs) {
    maxValue = fmax(tensor.abs().max().item<float>(), maxValue);
  }
  return diff.abs().max().item<float>() < (0.01 + 2e-3 * maxValue);
}
bool almostEqual(const at::Tensor& a, const at::Tensor& b) {
  return checkRtol(a - b, {a, b});
}

bool exactlyEqual(const at::Tensor& a, const at::Tensor& b) {
  return (a - b).abs().max().item<float>() == 0.f;
}

TEST(VulkanTest, ToVulkanToCpu) {
  if (!at::vulkan::is_available())
    return;
  auto t =
      at::rand({1, 2, 2, 3}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto tv = t.vulkan();
  ASSERT_TRUE(tv.options().device().type() == at::kVulkan);
  auto t2 = tv.cpu();
  ASSERT_TRUE(t2.options().device().type() == at::kCPU);
  ASSERT_TRUE(almostEqual(t2, t));
}

TEST(VulkanTest, FailOnStrides) {
  if (!at::vulkan::is_available())
    return;
  auto t = at::empty({1, 2, 3}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto tv = t.vulkan();
  ASSERT_ANY_THROW(tv.strides());
  ASSERT_ANY_THROW(tv.stride(0));
}

TEST(VulkanTest, upsampleNearest2D) {
  if (!at::vulkan::is_available())
    return;

  auto t_in =
      at::rand({1, 2, 2, 3}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto t_out_expected = at::upsample_nearest2d(t_in, {4, 6});
  auto tv_in =
      t_in.to(at::TensorOptions{at::Device{at::kVulkan}}.dtype(at::kFloat));

  auto tv_out = at::upsample_nearest2d(tv_in, {4, 6});
  auto t_out =
      tv_out.to(at::TensorOptions{at::Device{at::kCPU}}.dtype(at::kFloat));

  ASSERT_TRUE(almostEqual(t_out, t_out_expected));
}

TEST(VulkanTest, add) {
  if (!at::vulkan::is_available())
    return;
  auto t_in0 = at::rand({1, 2, 2, 3}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_in1 = at::rand({1, 2, 2, 3}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_out_expected = at::add(t_in0, t_in1, 2);
  auto tv_in0 = t_in0.vulkan();
  auto tv_in1 = t_in1.vulkan();
  auto tv_out = at::add(tv_in0, tv_in1, 2);
  auto t_out = tv_out.cpu();

  ASSERT_TRUE(almostEqual(t_out, t_out_expected));
}

TEST(VulkanTest, conv2dWeightsOnCPU) {
  if (!at::vulkan::is_available())
    return;
  auto OC = 2;
  auto C = 3;
  auto t_in = at::rand({1, C, 3, 3}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_w = at::rand({OC, C, 2, 2}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_b = at::zeros({OC}, at::device(at::kCPU).dtype(at::kFloat));
  auto stride = c10::IntArrayRef{1};
  auto padding = c10::IntArrayRef{0};
  auto dilation = c10::IntArrayRef{1};
  int64_t groups = 1;
  auto t_out_expected =
      at::conv2d(t_in, t_w, t_b, stride, padding, dilation, groups);
  auto tv_in = t_in.vulkan();
  auto tv_out = at::conv2d(tv_in, t_w, t_b, stride, padding, dilation, groups);
  auto t_out = tv_out.cpu();
  ASSERT_TRUE(almostEqual(t_out, t_out_expected));
}

TEST(VulkanTest, conv2dDWWeightsOnCPU) {
  if (!at::vulkan::is_available())
    return;
  auto C = 3;
  int64_t groups = C;
  auto t_in = at::rand({1, C, 3, 3}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_w =
      at::rand({groups, 1, 2, 2}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_b = at::zeros({groups}, at::device(at::kCPU).dtype(at::kFloat));
  auto stride = c10::IntArrayRef{1};
  auto padding = c10::IntArrayRef{0};
  auto dilation = c10::IntArrayRef{1};
  auto t_out_expected =
      at::conv2d(t_in, t_w, t_b, stride, padding, dilation, groups);
  auto tv_in = t_in.vulkan();
  auto tv_out = at::conv2d(tv_in, t_w, t_b, stride, padding, dilation, groups);
  auto t_out = tv_out.cpu();
  ASSERT_TRUE(almostEqual(t_out, t_out_expected));
}

TEST(VulkanTest, addmm) {
  if (!at::vulkan::is_available())
    return;
  auto t_m1 = at::rand({2, 2}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_m2 = at::rand({2, 3}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_b = at::rand({2, 3}, at::device(at::kCPU).dtype(at::kFloat));

  float beta = 100;
  float alpha = 2;
  auto t_out_expected = at::addmm(t_b, t_m1, t_m2, beta, alpha);

  auto tv_m1 = t_m1.vulkan();
  auto tv_m2 = t_m2.vulkan();
  auto tv_b = t_b.vulkan();
  auto tv_out = at::addmm(tv_b, tv_m1, tv_m2, beta, alpha);
  auto t_out = tv_out.cpu();
  ASSERT_TRUE(almostEqual(t_out, t_out_expected));
}

TEST(VulkanTest, clamp) {
  if (!at::vulkan::is_available())
    return;
  float min = -0.5;
  float max = 0.5;
  auto t_in = at::rand({1, 3, 16, 16}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_out_expected = at::clamp(t_in, min, max);

  auto tv_in = t_in.vulkan();
  auto tv_out = at::clamp(tv_in, min, max);
  auto t_out = tv_out.cpu();

  ASSERT_TRUE(almostEqual(t_out, t_out_expected));
}

TEST(VulkanTest, hardtanh_) {
  if (!at::vulkan::is_available())
    return;
  float min = -0.5;
  float max = 0.5;
  auto t_in = at::rand({1, 3, 16, 16}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_out_expected = at::hardtanh_(t_in, min, max);

  auto tv_in = t_in.vulkan();
  auto tv_out = at::hardtanh_(tv_in, min, max);
  auto t_out = tv_out.cpu();

  ASSERT_TRUE(almostEqual(t_out, t_out_expected));
}

TEST(VulkanTest, mean) {
  if (!at::vulkan::is_available())
    return;
  auto t_in = at::rand({2, 3, 3, 3}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_out_expected = at::mean(t_in, {2, 3}, false /* keepdim */);
  auto tv_in = t_in.vulkan();
  auto tv_out = at::mean(tv_in, {2, 3}, false /* keepdim */);
  auto t_out = tv_out.cpu();
  ASSERT_TRUE(almostEqual(t_out, t_out_expected));
}

enum class OpType { conv2d, hardtanh_, mean, addmm };

class BaseOp {
 public:
  BaseOp(OpType t) : type(t) {}
  virtual at::Tensor run(at::Tensor&) = 0;
  virtual std::string toString() = 0;
  OpType type;
};

class Hardtanh_ : public BaseOp {
 public:
  Hardtanh_() : BaseOp(OpType::hardtanh_) {}
  at::Tensor run(at::Tensor& t) override {
    return at::hardtanh_(t, 0, 6);
  }
  std::string toString() override {
    return "hardtanh_";
  }
};

class Mean : public BaseOp {
 public:
  Mean() : BaseOp(OpType::mean) {}
  at::Tensor run(at::Tensor& t) override {
    return at::mean(t, {2, 3}, false /* keepdim */);
  }
  std::string toString() override {
    return "mean";
  }
};

class Addmm : public BaseOp {
 public:
  Addmm(int64_t m1H, int64_t m1W, int64_t m2W, float _beta, float _alpha)
      : BaseOp(OpType::addmm), beta(_beta), alpha(_alpha) {
    m2 = at::rand(
        c10::IntArrayRef({m1W, m2W}), at::device(at::kCPU).dtype(at::kFloat));
    m2v = m2.vulkan();
    b = at::rand(
        c10::IntArrayRef({m1H, m2W}), at::device(at::kCPU).dtype(at::kFloat));
    bv = b.vulkan();
  }

  at::Tensor run(at::Tensor& t) override {
    if (t.is_vulkan()) {
      return at::addmm(bv, t, m2v, beta, alpha);
    }
    return at::addmm(b, t, m2, beta, alpha);
  }

  std::string toString() override {
    return "addmm";
  }

  at::Tensor m2;
  at::Tensor m2v;
  at::Tensor b;
  at::Tensor bv;
  float beta;
  float alpha;
};

class Conv2d : public BaseOp {
 public:
  Conv2d(c10::IntArrayRef wsizes, int64_t g, int64_t s, int64_t p)
      : BaseOp(OpType::conv2d), groups(g), stride(s), padding(p) {
    w = at::rand(wsizes, at::device(at::kCPU).dtype(at::kFloat));
    b = at::zeros(wsizes[0], at::device(at::kCPU).dtype(at::kFloat));
  };

  at::Tensor run(at::Tensor& t) override {
    return at::conv2d(t, w, b, {stride}, {padding}, {1}, groups);
  }
  std::string toString() override {
    return "conv2d";
  }

  int64_t stride;
  int64_t padding;
  int64_t groups;
  at::Tensor w;
  at::Tensor b;
};

class MobileNetV2 {
 public:
  MobileNetV2() {
    ops.emplace_back(new Conv2d({32, 3, 3, 3}, 1, 2, 1));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({32, 1, 3, 3}, 32, 1, 1));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({16, 32, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Conv2d({96, 16, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({96, 1, 3, 3}, 96, 2, 1));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({24, 96, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Conv2d({144, 24, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({144, 1, 3, 3}, 144, 1, 1));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({24, 144, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Conv2d({144, 24, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({144, 1, 3, 3}, 144, 2, 1));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({32, 144, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Conv2d({192, 32, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({192, 1, 3, 3}, 192, 1, 1));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({32, 192, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Conv2d({192, 32, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({192, 1, 3, 3}, 192, 1, 1));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({32, 192, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Conv2d({192, 32, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({192, 1, 3, 3}, 192, 2, 1));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({64, 192, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Conv2d({384, 64, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({384, 1, 3, 3}, 384, 1, 1));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({64, 384, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Conv2d({384, 64, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({384, 1, 3, 3}, 384, 1, 1));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({64, 384, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Conv2d({384, 64, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({384, 1, 3, 3}, 384, 1, 1));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({64, 384, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Conv2d({384, 64, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({384, 1, 3, 3}, 384, 1, 1));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({96, 384, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Conv2d({576, 96, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({576, 1, 3, 3}, 576, 1, 1));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({96, 576, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Conv2d({576, 96, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({576, 1, 3, 3}, 576, 1, 1));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({96, 576, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Conv2d({576, 96, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({576, 1, 3, 3}, 576, 2, 1));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({160, 576, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Conv2d({960, 160, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({960, 1, 3, 3}, 960, 1, 1));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({160, 960, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Conv2d({960, 160, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({960, 1, 3, 3}, 960, 1, 1));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({160, 960, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Conv2d({960, 160, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({960, 1, 3, 3}, 960, 1, 1));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Conv2d({320, 960, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Conv2d({1280, 320, 1, 1}, 1, 1, 0));
    ops.emplace_back(new Hardtanh_());
    ops.emplace_back(new Mean());
    ops.emplace_back(new Addmm(1, 1280, 1000, 0, 1));
  }

  auto run(at::Tensor& in, at::Tensor& vin) {
    at::Tensor t = in;
    at::Tensor tv = vin;
    int i = 0;
    auto size = ops.size();
    for (const auto& op : ops) {
      t = op->run(t);
      tv = op->run(tv);
      auto tv_cpu = t.cpu();
      TORCH_INTERNAL_ASSERT(
          almostEqual(t, tv_cpu),
          "Not almost equal cpu vs vulkan op i:",
          i,
          " ",
          op->toString());
      i++;
    }
    return std::make_pair(t, tv);
  }

  std::vector<std::unique_ptr<BaseOp>> ops;
};

TEST(VulkanTest, mobilenetv2) {
  if (!at::vulkan::is_available())
    return;

  MobileNetV2 mn2{};
  auto t_in =
      at::rand({1, 3, 224, 224}, at::device(at::kCPU).dtype(at::kFloat));
  auto tv_in = t_in.vulkan();
  mn2.run(t_in, tv_in);
}
