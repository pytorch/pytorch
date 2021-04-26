#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/core/Reduction.h>
#include <torch/cuda.h>
#include <ATen/test/test_assert.h>

// for TH compat test only...
struct THFloatTensor;

#include <iostream>
#include <chrono>
#include <string.h>
#include <sstream>

#define ASSERT_EQ_RESOLVED(X, Y) \
  {                              \
    bool isEQ = X == Y;          \
    ASSERT_TRUE(isEQ);           \
  }

using namespace at;

void TestResize(DeprecatedTypeProperties& type) {
  auto a = at::empty({0}, type.options());
  a.resize_({3, 4});
  ASSERT_EQ_RESOLVED(a.numel(), 12);
  a.resize_({5, 7});
  ASSERT_EQ_RESOLVED(a.numel(), 35);
}

void TestOnesAndDot(DeprecatedTypeProperties& type) {
  Tensor b0 = ones({1, 1}, type);
  ASSERT_EQ_RESOLVED((b0 + b0).sum().item<double>(), 2);

  Tensor b1 = ones({1, 2}, type);
  ASSERT_EQ_RESOLVED((b1 + b1).sum().item<double>(), 4);

  Tensor b = ones({3, 4}, type);
  ASSERT_EQ_RESOLVED((b + b).sum().item<double>(), 24);
  ASSERT_EQ_RESOLVED(b.numel(), 12);
  ASSERT_EQ_RESOLVED(b.view(-1).dot(b.view(-1)).item<double>(), 12);
}

void TestSort(DeprecatedTypeProperties& type) {
  Tensor b = rand({3, 4}, type);

  auto z = b.sort(1);
  auto z_sorted = std::get<0>(z);

  bool isLT = z_sorted[0][0].item<float>() < z_sorted[0][1].item<float>();
  ASSERT_TRUE(isLT);
}

void TestRandperm(DeprecatedTypeProperties& type) {
  if (type.backend() != Backend::CUDA) {
    Tensor b = randperm(15, type);
    Tensor rv, ri;
    std::tie(rv, ri) = sort(b, 0);
    bool isLE = (rv[0].item<float>() <= rv[1].item<float>());
    ASSERT_TRUE(isLE);
  }
}

void SendContext() {
  std::stringstream ss;
  ss << "context: " << std::hex << (int64_t)&globalContext() << std::endl;
}

void TestAdd(DeprecatedTypeProperties& type) {
  Tensor a = rand({3, 4}, type);
  Tensor b = rand({3, 4}, type);
  Tensor c = add(a, add(a, b));
  // TODO:0-dim Tensor d(3.f);
  Scalar d = 3.f;
  if (type.backend() == Backend::CPU && type.scalarType() == kHalf) {
      ASSERT_TRUE(add(c, d).allclose(a + a + b + d, 1e-2));
  } else {
      ASSERT_TRUE(add(c, d).allclose(a + a + b + d));
  }
}

void TestZeros(DeprecatedTypeProperties& type) {
  auto begin = std::chrono::high_resolution_clock::now();
  Tensor a = zeros({1024, 1024}, type);
  for (int i = 1; i < 1000; ++i) {
    a = zeros({128, 128}, type);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << std::dec << "   "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   end - begin)
                   .count()
            << " ms" << std::endl;

   std::srand(std::time(nullptr));
   ASSERT_EQ(norm(a).item<double>(), 0.0);
}

void TestLoadsOfAdds(DeprecatedTypeProperties& type) {
  auto begin = std::chrono::high_resolution_clock::now();
  Tensor d = ones({3, 4}, type);
  Tensor r = zeros({3, 4}, type);
  for (auto i = 0; i < 100000; i++) {
    add_out(r, r, d);
  }
  auto end = std::chrono::high_resolution_clock::now();
  // TODO TEST PERF?
  std::cout << std::dec << "   "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   end - begin)
                   .count()
            << " ms" << std::endl;
  ASSERT_EQ_RESOLVED(norm(100000 * d).item<double>(), norm(r).item<double>());
}

void TestLoadOfAddsWithCopy(DeprecatedTypeProperties& type) {
  auto begin = std::chrono::high_resolution_clock::now();
  Tensor d = ones({3, 4}, type);
  Tensor r = zeros({3, 4}, type);
  for (auto i = 0; i < 100000; i++) {
    r = add(r, d);
  }
  auto end = std::chrono::high_resolution_clock::now();
  // TODO TEST PERF?
  std::cout << std::dec << "   "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   end - begin)
                   .count()
            << " ms" << std::endl;
  ASSERT_EQ_RESOLVED(norm(100000 * d).item<double>(), norm(r).item<double>());
}

void TestIsContiguous(DeprecatedTypeProperties& type) {
  Tensor a = rand({3, 4}, type);
  ASSERT_TRUE(a.is_contiguous());
  a = a.transpose(0, 1);
  ASSERT_FALSE(a.is_contiguous());
}

void TestPermute(DeprecatedTypeProperties& type) {
  Tensor a = rand({3, 4, 5}, type);
  Tensor b = a.permute({1, 2, 0});
  ASSERT_TRUE(b.sizes().equals({4, 5, 3}));
  ASSERT_TRUE(b.strides().equals({5, 1, 20}));
}

void TestMm(DeprecatedTypeProperties& type) {
  if (type.backend() != Backend::CPU || type.scalarType() != kHalf) {
    Tensor a = rand({3, 4}, type);
    Tensor b = rand({4}, type);
    Tensor c = mv(a, b);
    ASSERT_TRUE(c.equal(addmv(zeros({3}, type), a, b, 0, 1)));
  }
}

void TestSqueeze(DeprecatedTypeProperties& type) {
  Tensor a = rand({2, 1}, type);
  Tensor b = squeeze(a);
  ASSERT_EQ_RESOLVED(b.dim(), 1);
  a = rand({1}, type);
  b = squeeze(a);
  // TODO 0-dim squeeze
  ASSERT_TRUE(a[0].equal(b));
}

void TestCopy(DeprecatedTypeProperties& type) {
  Tensor a = zeros({4, 3}, type);
  Tensor e = rand({4, 3}, type);
  a.copy_(e);
  ASSERT_TRUE(a.equal(e));
}

void TestCopyBroadcasting(DeprecatedTypeProperties& type) {
  Tensor a = zeros({4, 3}, type);
  Tensor e = rand({3}, type);
  a.copy_(e);
  for (int i = 0; i < 4; ++i) {
    ASSERT_TRUE(a[i].equal(e));
  }
}
void TestAbsValue(DeprecatedTypeProperties& type) {
  Tensor r = at::abs(at::scalar_tensor(-3, type.options()));
  ASSERT_EQ_RESOLVED(r.item<int32_t>(), 3);
}
/*
   TODO(zach): operator overloads
#if 0
{
std::cout << "eq (value):" << std::endl;
Tensor a = Tensor(10.f);
std::cout << (a == 11_i64) << " -- should be 0" << std::endl;
std::cout << (a == 10_i64) << " -- should be 1" << std::endl;
std::cout << (a == 10.) << " -- should be 1" << std::endl;
}
#endif
*/

void TestAddingAValueWithScalar(DeprecatedTypeProperties& type) {
  Tensor a = rand({4, 3}, type);
  ASSERT_TRUE((ones({4, 3}, type) + a).equal(add(a, 1)));
}

void TestSelect(DeprecatedTypeProperties& type) {
  Tensor a = rand({3, 7}, type);
  auto a_13 = select(a, 1, 3);
  auto a_13_02 = select(select(a, 1, 3), 0, 2);
  ASSERT_TRUE(a[0][3].equal(a_13[0]));
  ASSERT_TRUE(a[2][3].equal(a_13_02));
}

void TestZeroDim(DeprecatedTypeProperties& type) {
  Tensor a = at::scalar_tensor(4, type.options()); // rand(type, {1});

  Tensor b = rand({3, 4}, type);
  ASSERT_EQ_RESOLVED((a + a).dim(), 0);
  ASSERT_EQ_RESOLVED((1 + a).dim(), 0);
  ASSERT_EQ_RESOLVED((b + a).dim(), 2);
  ASSERT_EQ_RESOLVED((a + b).dim(), 2);
  auto c = rand({3, 4}, type);
  ASSERT_EQ_RESOLVED(c[1][2].dim(), 0);

  auto f = rand({3, 4}, type);
  f[2] = zeros({4}, type);
  f[1][0] = -1;
  ASSERT_EQ_RESOLVED(f[2][0].item<double>(), 0);
}

void TestToCFloat() {
  Tensor a = zeros({3, 4});
  Tensor b = ones({3, 7});
  Tensor c = cat({a, b}, 1);
  ASSERT_EQ_RESOLVED(c.size(1), 11);

  Tensor e = rand({});
  ASSERT_EQ_RESOLVED(*e.data_ptr<float>(), e.sum().item<float>());
}
void TestToString() {
  Tensor b = ones({3, 7}) * .0000001f;
  std::stringstream s;
  s << b << "\n";
  std::string expect = "1e-07 *";
  ASSERT_EQ_RESOLVED(s.str().substr(0, expect.size()), expect);
}

void TestIndexingByScalar() {
  Tensor tensor = arange(0, 10, kInt);
  Tensor one = ones({}, kInt);
  for (int64_t i = 0; i < tensor.numel(); ++i) {
    ASSERT_TRUE(tensor[i].equal(one * i));
  }
  for (size_t i = 0; i < static_cast<uint64_t>(tensor.numel()); ++i) {
    ASSERT_TRUE(tensor[i].equal(one * static_cast<int64_t>(i)));
  }
  for (int i = 0; i < tensor.numel(); ++i) {
    ASSERT_TRUE(tensor[i].equal(one * i));
  }
  for (int16_t i = 0; i < tensor.numel(); ++i) {
    ASSERT_TRUE(tensor[i].equal(one * i));
  }
  for (int8_t i = 0; i < tensor.numel(); ++i) {
    ASSERT_TRUE(tensor[i].equal(one * i));
  }
  // Throw StartsWith("Can only index tensors with integral scalars")
  ASSERT_ANY_THROW(tensor[Scalar(3.14)].equal(one));
}

void TestIndexingByZerodimTensor() {
  Tensor tensor = arange(0, 10, kInt);
  Tensor one = ones({}, kInt);
  for (int i = 0; i < tensor.numel(); ++i) {
    ASSERT_TRUE(tensor[one * i].equal(one * i));
  }
  // Throw StartsWith(
  //            "Can only index tensors with integral scalars")
  ASSERT_ANY_THROW(tensor[ones({}) * 3.14].equal(one));
  // Throw StartsWith("Can only index with tensors that are defined")
  ASSERT_ANY_THROW(tensor[Tensor()].equal(one));
  // Throw StartsWith("Can only index with tensors that are scalars (zero-dim)")
  ASSERT_ANY_THROW(tensor[ones({2, 3, 4}, kInt)].equal(one));
}
void TestIndexingMixedDevice(DeprecatedTypeProperties& type) {
  Tensor tensor = randn({20, 20}, type);
  Tensor index = arange(10, kLong).cpu();
  Tensor result = tensor.index({index});
  ASSERT_TRUE(result[0].equal(tensor[0]));
}
void TestDispatch() {
  Tensor tensor = randn({20, 20});
  Tensor other = randn({20, 20});
  auto result = tensor.m(relu).m(mse_loss, other, at::Reduction::Mean);
  ASSERT_TRUE(result.allclose(mse_loss(relu(tensor), other)));
}

void TestNegativeDim(DeprecatedTypeProperties& type) {
  ASSERT_ANY_THROW(empty({5, -5, 5}, type.options()));
  ASSERT_ANY_THROW(empty({5, -5, -5}, type.options()));
  Tensor tensor = empty({5, 5}, type.options());
  ASSERT_ANY_THROW(tensor.reshape({-5, -5}));
}

void TestView(DeprecatedTypeProperties& type) {
  // Testing the tensor view path, which is different from
  // the Variable view path, see https://github.com/pytorch/pytorch/pull/23452
  // for details
  Tensor tensor = randn({3, 4}, type);;
  Tensor viewed = tensor.view({3, 4});
  tensor.resize_({6, 2});
  ASSERT_TRUE(tensor.sizes().equals({6, 2}));
  ASSERT_TRUE(viewed.sizes().equals({3, 4}));
}

void TestIntArrayRefExpansion(DeprecatedTypeProperties& type) {
  if (type.backend() != Backend::CPU || type.scalarType() != kHalf) {
    max_pool2d(randn({3, 3, 3, 3}, type.options()), 2, 1, 1, 1);
    max_pool3d(randn({3, 3, 3, 3, 3}, type.options()), 2, 1, 1, 1);
    avg_pool2d(randn({3, 3, 3, 3}, type.options()), 2, 1, 1);
    avg_pool3d(randn({3, 3, 3, 3, 3}, type.options()), 2, 1, 1);
  }
}

void test(DeprecatedTypeProperties& type) {
  TestResize(type);
  TestOnesAndDot(type);

  TestSort(type);
  TestRandperm(type);
  TestAdd(type);
  TestZeros(type);
  TestLoadsOfAdds(type);
  TestLoadOfAddsWithCopy(type);
  TestIsContiguous(type);
  TestPermute(type);
  TestMm(type);
  TestSqueeze(type);
  TestCopy(type);
  TestCopyBroadcasting(type);
  TestAbsValue(type);
  TestAddingAValueWithScalar(type);
  TestSelect(type);
  TestZeroDim(type);
  TestToCFloat();
  TestToString();
  TestIndexingByScalar();
  TestIndexingByZerodimTensor();
  TestIndexingMixedDevice(type);
  TestDispatch();
  TestNegativeDim(type);
  TestView(type);
  TestIntArrayRefExpansion(type);
}

TEST(BasicTest, BasicTestCPU) {
  manual_seed(123);

  test(CPU(kFloat));
}

TEST(BasicTest, BasicTestHalfCPU) {
  manual_seed(234);

  test(CPU(kHalf));
}

TEST(BasicTest, BasicTestCUDA) {
  manual_seed(123);

  if (at::hasCUDA()) {
    test(CUDA(kFloat));
  }
}

TEST(BasicTest, FactoryMethodsTest) {
  // Test default values
  at::Tensor tensor0 = at::empty({4});
  ASSERT_EQ(tensor0.dtype(), at::kFloat);
  ASSERT_EQ(tensor0.layout(), at::kStrided);
  ASSERT_EQ(tensor0.device(), at::kCPU);
  ASSERT_FALSE(tensor0.requires_grad());
  ASSERT_FALSE(tensor0.is_pinned());

  // Test setting requires_grad to false.
  tensor0 = at::empty({4}, at::TensorOptions().requires_grad(false));
  ASSERT_EQ(tensor0.dtype(), at::kFloat);
  ASSERT_EQ(tensor0.layout(), at::kStrided);
  ASSERT_EQ(tensor0.device(), at::kCPU);
  ASSERT_FALSE(tensor0.requires_grad());
  ASSERT_FALSE(tensor0.is_pinned());

  // Test setting requires_grad to true.
  // This is a bug. Requires_grad was set to TRUE but this is not implemented.
  EXPECT_ANY_THROW(at::empty({4}, at::TensorOptions().requires_grad(true)));

  // Test setting dtype
  at::Tensor tensor1 = at::empty({4}, at::TensorOptions().dtype(at::kHalf));
  ASSERT_EQ(tensor1.dtype(), at::kHalf);
  ASSERT_EQ(tensor1.layout(), at::kStrided);
  ASSERT_EQ(tensor1.device(), at::kCPU);
  ASSERT_FALSE(tensor1.requires_grad());
  ASSERT_FALSE(tensor1.is_pinned());

  // Sparse tensor CPU test to avoid requiring CUDA to catch simple bugs.1
  tensor1 = at::empty({4}, at::TensorOptions().dtype(at::kHalf).layout(at::kSparse));
  ASSERT_EQ(tensor1.dtype(), at::kHalf);
  ASSERT_EQ(tensor1.layout(), at::kSparse);
  ASSERT_EQ(tensor1.device(), at::kCPU);
  ASSERT_FALSE(tensor1.requires_grad());
  ASSERT_ANY_THROW(tensor1.is_pinned());

  if (torch::cuda::is_available()) {
    // Test setting pin memory
    tensor1 = at::empty({4}, at::TensorOptions().pinned_memory(true));
    ASSERT_EQ(tensor1.dtype(), at::kFloat);
    ASSERT_EQ(tensor1.layout(), at::kStrided);
    ASSERT_EQ(tensor1.device(), at::kCPU);
    ASSERT_EQ(tensor1.requires_grad(), false);
    ASSERT_FALSE(tensor1.device().is_cuda());
    ASSERT_TRUE(tensor1.is_pinned());

    // Test setting device
    tensor1 = at::empty({4}, at::TensorOptions().device(at::kCUDA));
    ASSERT_EQ(tensor1.dtype(), at::kFloat);
    ASSERT_EQ(tensor1.layout(), at::kStrided);
    ASSERT_TRUE(tensor1.device().is_cuda());
    ASSERT_FALSE(tensor1.requires_grad());
    ASSERT_FALSE(tensor1.is_pinned());

    // Test set everything
    tensor1 = at::empty({4}, at::TensorOptions().dtype(at::kHalf).device(at::kCUDA).layout(at::kSparse).requires_grad(false));
    ASSERT_EQ(tensor1.dtype(), at::kHalf);
    ASSERT_EQ(tensor1.layout(), at::kSparse);
    ASSERT_TRUE(tensor1.device().is_cuda());
    ASSERT_THROWS(tensor1.nbytes());

    // This is a bug
    // Issue https://github.com/pytorch/pytorch/issues/30405
    ASSERT_FALSE(tensor1.requires_grad());

    // This will cause an exception
    // Issue https://github.com/pytorch/pytorch/issues/30405
    ASSERT_ANY_THROW(tensor1.is_pinned());
  }

  // Test _like variants
  if (torch::cuda::is_available()) {
    // Issue https://github.com/pytorch/pytorch/issues/28093
    at::Tensor proto = at::empty({1}, at::kDouble);
    tensor0 = at::empty_like(proto, at::kCUDA);
    ASSERT_EQ(tensor0.dtype(), at::kDouble);
    ASSERT_EQ(tensor0.layout(), at::kStrided);
    ASSERT_TRUE(tensor0.device().is_cuda());
    ASSERT_FALSE(tensor0.requires_grad());
    ASSERT_FALSE(tensor0.is_pinned());
  }
}
