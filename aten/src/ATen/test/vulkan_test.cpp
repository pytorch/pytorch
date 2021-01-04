#ifndef USE_VULKAN_API

#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/vulkan/Context.h>

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

TEST(VulkanTest, ToVulkanToCpu) {
  if (!at::is_vulkan_available())
    return;
  auto t =
      at::rand({1, 2, 2, 3}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto tv = t.vulkan();
  ASSERT_TRUE(tv.options().device().type() == at::kVulkan);
  auto t2 = tv.cpu();
  ASSERT_TRUE(t2.options().device().type() == at::kCPU);
  ASSERT_TRUE(almostEqual(t2, t));
}

TEST(VulkanTest, upsampleNearest2D) {
  if (!at::is_vulkan_available())
    return;

  auto t_in =
      at::rand({1, 2, 2, 3}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto t_out_expected = at::upsample_nearest2d(t_in, {4, 6});
  auto tv_in =
      t_in.to(at::TensorOptions{at::Device{at::kVulkan}}.dtype(at::kFloat));

  auto tv_out = at::upsample_nearest2d(tv_in, {4, 6});
  auto t_out =
      tv_out.to(at::TensorOptions{at::Device{at::kCPU}}.dtype(at::kFloat));

  bool check = almostEqual(t_out_expected, t_out);
  if (!check) {
    std::cout << "expected:\n" << t_out_expected << std::endl;
    std::cout << "got:\n" << t_out << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST(VulkanTest, add) {
  if (!at::is_vulkan_available())
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

TEST(VulkanTest, add_not4dim) {
  if (!at::is_vulkan_available())
    return;
  auto t_in0 = at::rand({1, 1000}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_in1 = at::rand({1000}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_out_expected = at::add(t_in0, t_in1, 2);
  auto tv_in0 = t_in0.vulkan();
  auto tv_in1 = t_in1.vulkan();
  auto tv_out = at::add(tv_in0, tv_in1, 2);
  auto t_out = tv_out.cpu();

  ASSERT_TRUE(almostEqual(t_out, t_out_expected));
}

TEST(VulkanTest, add_cpu_vulkan) {
  if (!at::is_vulkan_available())
    return;
  auto t_in0 = at::rand({2, 96, 1000}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_in1 =
      at::rand({1, 2, 96, 1000}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_out_expected = at::add(t_in0, t_in1, 2);
  auto tv_in0 = t_in0.vulkan();
  auto tv_in1 = t_in1.vulkan();

  auto tv_out1 = at::add(tv_in0, t_in1, 2);
  auto t_out1 = tv_out1.cpu();
  ASSERT_TRUE(almostEqual(t_out1, t_out_expected));

  auto tv_out2 = at::add(t_in0, tv_in1, 2);
  auto t_out2 = tv_out2.cpu();
  ASSERT_TRUE(almostEqual(t_out2, t_out_expected));
}

TEST(VulkanTest, add_) {
  if (!at::is_vulkan_available())
    return;
  auto t_in0 = at::rand({1, 2, 2, 2}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_in1 = at::rand({1, 2, 2, 2}, at::device(at::kCPU).dtype(at::kFloat));
  auto tv_in0 = t_in0.vulkan();
  auto tv_in1 = t_in1.vulkan();

  t_in0.add_(t_in1, 2);
  tv_in0.add_(tv_in1, 2);
  auto t_out = tv_in0.cpu();
  bool check = almostEqual(t_out, t_in0);
  if (!check) {
    std::cout << "expected:\n" << t_in0 << std::endl;
    std::cout << "got:\n" << t_out << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST(VulkanTest, mulScalar) {
  if (!at::is_vulkan_available())
    return;
  auto t_in = at::rand({3, 2, 2, 3}, at::device(at::kCPU).dtype(at::kFloat));
  const float other = 3.14;
  auto t_out_expected = t_in.mul(other);
  auto tv_in = t_in.vulkan();
  auto tv_out = tv_in.mul(other);
  auto t_out = tv_out.cpu();

  bool check = almostEqual(t_out, t_out_expected);
  if (!check) {
    std::cout << "expected:\n" << t_out_expected << std::endl;
    std::cout << "got:\n" << t_out << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST(VulkanTest, addScalar) {
  if (!at::is_vulkan_available())
    return;
  auto t_in = at::rand({3, 2, 2, 3}, at::device(at::kCPU).dtype(at::kFloat));
  float* data = t_in.data_ptr<float>();
  auto numel = t_in.numel();
  for (int i = 0; i < numel; i++) {
    data[i] = i;
  }

  const float other = 3.14;
  const float alpha = 2;
  auto t_out_expected = t_in.add(other, alpha);
  auto tv_in = t_in.vulkan();
  auto tv_out = tv_in.add(other, alpha);
  auto t_out = tv_out.cpu();

  bool check = almostEqual(t_out, t_out_expected);
  if (!check) {
    std::cout << "expected:\n" << t_out_expected << std::endl;
    std::cout << "got:\n" << t_out << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST(VulkanTest, conv2d) {
  if (!at::is_vulkan_available())
    return;
  auto OC = 2;
  auto C = 3;
  int64_t H = 3;
  int64_t W = 3;
  int64_t KH = 2;
  int64_t KW = 2;
  auto t_in = at::rand({1, C, H, W}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_w = at::rand({OC, C, KH, KW}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_b = at::zeros({OC}, at::device(at::kCPU).dtype(at::kFloat));
  int64_t groups = 1;
  std::vector<int64_t> stride{1, 1};
  std::vector<int64_t> padding{0, 0};
  std::vector<int64_t> dilation{1, 1};

  auto t_out_expected =
      at::conv2d(t_in, t_w, t_b, stride, padding, dilation, groups);
  auto tv_in = t_in.vulkan();
  auto tv_out = at::conv2d(tv_in, t_w, t_b, stride, padding, dilation, groups);
  auto t_out = tv_out.cpu();
  bool check = almostEqual(t_out, t_out_expected);
  if (!check) {
    std::cout << "expected:\n" << t_out_expected << std::endl;
    std::cout << "got:\n" << t_out << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST(VulkanTest, conv2dDWWeightsOnCPU) {
  if (!at::is_vulkan_available())
    return;
  auto C = 3;
  int64_t groups = C;
  int64_t H = 3;
  int64_t W = 3;
  int64_t KH = 2;
  int64_t KW = 2;
  auto t_in = at::rand({1, C, H, W}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_w =
      at::rand({groups, 1, KH, KW}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_b = at::zeros({groups}, at::device(at::kCPU).dtype(at::kFloat));
  std::vector<int64_t> stride{1, 1};
  std::vector<int64_t> padding{0, 0};
  std::vector<int64_t> dilation{1, 1};
  auto t_out_expected =
      at::conv2d(t_in, t_w, t_b, stride, padding, dilation, groups);
  auto tv_in = t_in.vulkan();
  auto tv_out = at::conv2d(tv_in, t_w, t_b, stride, padding, dilation, groups);
  auto t_out = tv_out.cpu();
  bool check = almostEqual(t_out_expected, t_out);
  if (!check) {
    std::cout << "expected:\n" << t_out_expected << std::endl;
    std::cout << "got:\n" << t_out << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST(VulkanTest, addmm) {
  if (!at::is_vulkan_available())
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
  bool check = almostEqual(t_out_expected, t_out);
  if (!check) {
    std::cout << "expected:\n" << t_out_expected << std::endl;
    std::cout << "got:\n" << t_out << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST(VulkanTest, mm) {
  if (!at::is_vulkan_available())
    return;
  auto t_m1 = at::rand({10, 20}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_m2 = at::rand({20, 30}, at::device(at::kCPU).dtype(at::kFloat));

  auto t_out_expected = t_m1.mm(t_m2);

  auto tv_m1 = t_m1.vulkan();
  auto tv_m2 = t_m2.vulkan();
  auto tv_out = tv_m1.mm(tv_m2);
  auto t_out = tv_out.cpu();
  bool check = almostEqual(t_out_expected, t_out);
  if (!check) {
    std::cout << "expected:\n" << t_out_expected << std::endl;
    std::cout << "got:\n" << t_out << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST(VulkanTest, clamp) {
  if (!at::is_vulkan_available())
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
  if (!at::is_vulkan_available())
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

TEST(VulkanTest, relu_) {
  if (!at::is_vulkan_available())
    return;
  auto t = at::empty({1, 2, 2, 2}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_in = t.uniform_(-1, 1);
  auto tv_in = t_in.vulkan();

  t_in.relu_();
  tv_in.relu_();
  auto tv_out = tv_in.cpu();
  bool check = almostEqual(t_in, tv_out);
  if (!check) {
    std::cout << "expected:\n" << t_in << std::endl;
    std::cout << "got:\n" << tv_out << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST(VulkanTest, mean) {
  if (!at::is_vulkan_available())
    return;
  auto t_in = at::rand({2, 3, 3, 3}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_out_expected = at::mean(t_in, {2, 3}, false);
  auto tv_in = t_in.vulkan();
  auto tv_out = at::mean(tv_in, {2, 3}, false);
  auto t_out = tv_out.cpu();
  bool check = almostEqual(t_out_expected, t_out);
  if (!check) {
    std::cout << "expected:\n" << t_out_expected << std::endl;
    std::cout << "got:\n" << t_out << std::endl;
  }
  ASSERT_TRUE(check);
}

enum class OpType { conv2d, hardtanh_, mean, addmm };

class BaseOp {
 public:
  BaseOp(OpType t) : type(t) {}
  virtual ~BaseOp() = default;
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
    return at::mean(t, {2, 3}, false);
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
      : BaseOp(OpType::conv2d), stride(s), padding(p), groups(g) {
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

class OpsList {
 public:
  OpsList() {}
  OpsList(std::vector<std::unique_ptr<BaseOp>>& _ops) : ops(std::move(_ops)) {}

  auto runDual(at::Tensor& in, at::Tensor& vin) {
    at::Tensor t = in;
    at::Tensor tv = vin;
    int i = 0;
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

  auto run(at::Tensor& in) {
    at::Tensor t = in;
    int i = 0;
    for (const auto& op : ops) {
      t = op->run(t);
      i++;
    }
    return t;
  }

  std::vector<std::unique_ptr<BaseOp>> ops;
};

class MobileNetV2 : public OpsList {
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
};

TEST(VulkanTest, DISABLED_mobilenetv2) {
  if (!at::is_vulkan_available())
    return;

  MobileNetV2 mn2{};
  auto t_in =
      at::rand({1, 3, 224, 224}, at::device(at::kCPU).dtype(at::kFloat));
  auto tv_in = t_in.vulkan();
  mn2.runDual(t_in, tv_in);
}

TEST(VulkanTest, OpsList) {
  if (!at::is_vulkan_available())
    return;

  std::vector<std::unique_ptr<BaseOp>> ops;
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
  ops.emplace_back(new Conv2d({144, 24, 1, 1}, 1, 1, 0)); // 1, 144, 56, 56
  ops.emplace_back(new Hardtanh_());
  ops.emplace_back(new Mean());
  ops.emplace_back(new Addmm(1, 144, 1000, 0, 1));
  OpsList opsList(ops);
  auto t_in =
      at::rand({1, 3, 224, 224}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_out_expected = opsList.run(t_in);

  auto tv_in = t_in.vulkan();

  auto tv_out = opsList.run(t_in);
  auto t_out = tv_out.cpu();

  ASSERT_TRUE(almostEqual(t_out, t_out_expected));
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

TEST(VulkanTest, conv2dPrepack) {
  if (!at::is_vulkan_available())
    return;
  auto OC = 2;
  auto C = 3;
  int64_t groups = 1;
  auto t_in = at::rand({1, C, 3, 3}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_w = at::rand({OC, C, 2, 2}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_b = at::zeros({OC}, at::device(at::kCPU).dtype(at::kFloat));

  std::vector<int64_t> stride{1, 1};
  std::vector<int64_t> padding{0, 0};
  std::vector<int64_t> dilation{1, 1};
  float output_min = 0.25;
  float output_max = 1.0;

  auto t_out_conv2d =
      at::conv2d(t_in, t_w, t_b, stride, padding, dilation, groups);
  auto t_out_expected = at::clamp(t_out_conv2d, output_min, output_max);

  auto tv_in = t_in.vulkan();
  auto tv_out_conv2d =
      at::conv2d(tv_in, t_w, t_b, stride, padding, dilation, groups);
  auto tv_out = at::clamp(tv_out_conv2d, output_min, output_max);

  auto t_out = tv_out.cpu();
  bool no_prepack_check = almostEqual(t_out, t_out_expected);
  if (!no_prepack_check) {
    std::cout << "t_out_expected:\n" << t_out_expected << std::endl;
    std::cout << "t_out:\n" << t_out << std::endl;
  }
  ASSERT_TRUE(no_prepack_check);

  auto prepack = callOpByName(
      "vulkan_prepack::conv2d_clamp_prepack",
      "",
      t_w,
      t_b,
      stride,
      padding,
      dilation,
      groups,
      output_min,
      output_max);
  auto tv_out_prepack_ivalues =
      callOpByName("vulkan_prepack::conv2d_clamp_run", "", tv_in, prepack[0]);
  auto tv_out_prepack = tv_out_prepack_ivalues[0].toTensor();
  auto t_out_prepack = tv_out_prepack.cpu();
  const auto prepack_check = almostEqual(t_out_prepack, t_out_expected);
  if (!prepack_check) {
    std::cout << "expected:\n" << t_out_expected << std::endl;
    std::cout << "got:\n" << t_out_prepack << std::endl;
  }
  ASSERT_TRUE(prepack_check);
}

TEST(VulkanTest, adaptive_avg_pool2d) {
  if (!at::is_vulkan_available())
    return;

  auto t_in =
      at::rand({1, 2, 7, 7}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto t_out_expected = at::adaptive_avg_pool2d(t_in, {3, 3});
  auto tv_in = t_in.vulkan();

  auto tv_out = at::adaptive_avg_pool2d(tv_in, {3, 3});
  auto t_out = tv_out.cpu();

  const auto check = almostEqual(t_out, t_out_expected);
  if (!check) {
    std::cout << "expected:" << t_out_expected << std::endl;
    std::cout << "got:" << t_out << std::endl;
  }
  ASSERT_TRUE(check);
}

// TODO: Enable when view operator for Vulkan landed
TEST(VulkanTest, DISABLED_adaptive_avg_pool2d_2) {
  if (!at::is_vulkan_available())
    return;

  auto t_in =
      at::rand({1, 1280, 7, 7}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto t_out_expected = at::adaptive_avg_pool2d(t_in, {1, 1});
  auto tv_in = t_in.vulkan();

  auto tv_out = at::adaptive_avg_pool2d(tv_in, {1, 1});
  auto t_out = tv_out.cpu();

  const auto check = almostEqual(t_out, t_out_expected);
  if (!check) {
    std::cout << "expected:" << t_out_expected << std::endl;
    std::cout << "got:" << t_out << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST(VulkanTest, reshape) {
  if (!at::is_vulkan_available())
    return;

  auto t_in =
      at::rand({1, 8, 1, 1}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto t_out_expected = at::reshape(t_in, {1, 8});
  auto tv_in = t_in.vulkan();
  auto tv_out = at::reshape(tv_in, {1, 8});
  auto t_out = tv_out.cpu();

  const auto check = almostEqual(t_out, t_out_expected);
  if (!check) {
    std::cout << "expected:" << t_out_expected << std::endl;
    std::cout << "got:" << t_out << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST(VulkanTest, reshape2) {
  if (!at::is_vulkan_available())
    return;

  auto t_in =
      at::rand({1, 3, 2, 2}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto t_out_expected = at::reshape(t_in, {2, 3, 1, 2});

  auto tv_in = t_in.vulkan();
  auto tv_out = at::reshape(tv_in, {2, 3, 1, 2});
  auto t_out = tv_out.cpu();

  const auto check = almostEqual(t_out, t_out_expected);
  if (!check) {
    std::cout << "expected:" << t_out_expected << std::endl;
    std::cout << "got:" << t_out << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST(VulkanTest, tensor5d) {
  if (!at::is_vulkan_available())
    return;

  auto t_in =
      at::rand({2, 2, 2, 3, 3}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto tv_in = t_in.vulkan();
}

TEST(VulkanTest, tensor5d_transpose) {
  if (!at::is_vulkan_available())
    return;

  auto t_in =
      at::empty({1, 2, 3, 2, 1}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  float* data = t_in.data_ptr<float>();
  auto numel = t_in.numel();
  for (int i = 0; i < numel; i++) {
    data[i] = i;
  }

  auto tv_in = t_in.vulkan();

  auto t_out_expected = t_in.transpose(1, 2);
  auto t_out = tv_in.transpose(1, 2).cpu();
  const auto check = almostEqual(t_out, t_out_expected);
  if (!check) {
    std::cout << "expected:" << t_out_expected << std::endl;
    std::cout << "got:" << t_out << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST(VulkanTest, view) {
  if (!at::is_vulkan_available())
    return;

  auto t_in =
      at::rand({2, 4, 3, 3}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto t_out_expected = t_in.view({2, 2, 2, 3, 3});
  auto tv_in = t_in.vulkan();
  auto tv_out = tv_in.view({2, 2, 2, 3, 3});
  auto t_out = tv_out.cpu();

  const auto check = almostEqual(t_out, t_out_expected);
  if (!check) {
    std::cout << "expected:" << t_out_expected << std::endl;
    std::cout << "got:" << t_out << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST(VulkanTest, slice) {
  if (!at::is_vulkan_available())
    return;

  auto t_in =
      at::empty({1, 4, 2, 2}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  float* data = t_in.data_ptr<float>();
  auto numel = t_in.numel();
  for (int i = 0; i < numel; i++) {
    data[i] = i;
  }

  auto tv_in = t_in.vulkan();

  auto t_out_expected = t_in.slice(1, 2, 4, 1);
  auto t_out = tv_in.slice(1, 2, 4, 1).cpu();
  const auto check = almostEqual(t_out, t_out_expected);
  if (!check) {
    std::cout << "expected:" << t_out_expected << std::endl;
    std::cout << "got:" << t_out << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST(VulkanTest, select) {
  if (!at::is_vulkan_available())
    return;

  auto t_in =
      at::empty({1, 4, 2, 2}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  float* data = t_in.data_ptr<float>();
  auto numel = t_in.numel();
  for (int i = 0; i < numel; i++) {
    data[i] = i;
  }

  auto tv_in = t_in.vulkan();

  auto t_out_expected = t_in.slice(1, 1);
  auto t_out = tv_in.slice(1, 1).cpu();
  const auto check = almostEqual(t_out, t_out_expected);
  if (!check) {
    std::cout << "expected:" << t_out_expected << std::endl;
    std::cout << "got:" << t_out << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST(VulkanTest, unsqueeze) {
  if (!at::is_vulkan_available())
    return;

  auto t_in =
      at::empty({1, 2, 2}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  float* data = t_in.data_ptr<float>();
  auto numel = t_in.numel();
  for (int i = 0; i < numel; i++) {
    data[i] = i;
  }

  auto tv_in = t_in.vulkan();

  auto t_out_expected = t_in.unsqueeze(1);
  auto t_out = tv_in.unsqueeze(1).cpu();
  const auto check = almostEqual(t_out, t_out_expected);
  if (!check) {
    std::cout << "expected:" << t_out_expected << std::endl;
    std::cout << "got:" << t_out << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST(VulkanTest, cat) {
  if (!at::is_vulkan_available())
    return;

  auto t_in0 =
      at::rand({1, 1, 3, 3}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto t_in1 =
      at::rand({1, 2, 3, 3}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto t_in2 =
      at::rand({1, 5, 3, 3}, at::TensorOptions(at::kCPU).dtype(at::kFloat));

  auto t_out_expected = at::cat({t_in0, t_in1, t_in2}, 1);
  auto tv_out = at::cat({t_in0.vulkan(), t_in1.vulkan(), t_in2.vulkan()}, 1);
  auto t_out = tv_out.cpu();

  const auto check = almostEqual(t_out, t_out_expected);
  if (!check) {
    std::cout << "expected:" << t_out_expected << std::endl;
    std::cout << "got:" << t_out << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST(VulkanTest, DISABLED_max_pool2d) {
  if (!at::is_vulkan_available())
    return;

  auto t_in =
      at::rand({1, 3, 7, 7}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto t_out_expected = at::max_pool2d(t_in, {2, 2}, {1}, {0}, {1});
  auto tv_in = t_in.vulkan();

  auto tv_out = at::max_pool2d(tv_in, {2, 2}, {1}, {0}, {1});
  auto t_out = tv_out.cpu();

  const auto check = almostEqual(t_out, t_out_expected);
  if (!check) {
    std::cout << "expected:" << t_out_expected << std::endl;
    std::cout << "got:" << t_out << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST(VulkanTest, avg_pool2d) {
  if (!at::is_vulkan_available())
    return;

  auto t_in =
      at::rand({1, 3, 7, 7}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto t_out_expected = at::avg_pool2d(t_in, {2, 2}, {1}, {0}, true);
  auto tv_in = t_in.vulkan();

  auto tv_out = at::avg_pool2d(tv_in, {2, 2}, {1}, {0}, true);
  auto t_out = tv_out.cpu();

  const auto check = almostEqual(t_out, t_out_expected);
  if (!check) {
    std::cout << "expected:" << t_out_expected << std::endl;
    std::cout << "got:" << t_out << std::endl;
  }
  ASSERT_TRUE(check);
}

#endif /* USE_VULKAN_API */
