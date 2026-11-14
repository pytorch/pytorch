#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/FakeTensor.h>
#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/ThreadLocalState.h>
#include <ATen/native/Pool.h>
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/avg_pool2d.h>
#include <ATen/ops/avg_pool3d.h>
#include <ATen/ops/quantize_per_tensor.h>
#endif
#include <c10/core/impl/FakeTensorModeTLS.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/InferenceMode.h>
#include <c10/util/irange.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <thread>

using namespace at;

#define ASSERT_EQUAL(t1, t2) ASSERT_TRUE(t1.equal(t2));

#define ASSERT_ALLCLOSE(t1, t2)     \
  ASSERT_TRUE(t1.is_same_size(t2)); \
  ASSERT_TRUE(t1.allclose(t2));

#define ASSERT_ALLCLOSE_TOLERANCES(t1, t2, atol, rtol) \
  ASSERT_TRUE(t1.is_same_size(t2));                    \
  ASSERT_TRUE(t1.allclose(t2, atol, rtol));

void requireEqualTensorList(TensorList t1, TensorList t2) {
  ASSERT_EQ(t1.size(), t2.size());
  for (const auto i : c10::irange(t1.size())) {
    ASSERT_EQUAL(t1[i], t2[i]);
  }
}

// split: test method, namespace give same result
void TestSplit(TensorOptions T, Tensor& t) {
  auto splitMethod = t.split(1, 0);
  auto splitNs = at::split(t, 1, 0);
  requireEqualTensorList(splitMethod, splitNs);

  // test rebuilding with cat
  ASSERT_EQUAL(at::cat(splitMethod, 0), t);
}

// chunk: test method, namespace give same result
void TestChunk(TensorOptions T, Tensor& t) {
  // test method, type, namespace give same result
  auto chunkMethod = t.chunk(3, 0);
  auto chunkNs = at::chunk(t, 3, 0);
  requireEqualTensorList(chunkMethod, chunkNs);

  // test rebuilding with cat
  ASSERT_EQUAL(at::cat(chunkMethod, 0), t);
}

typedef Tensor StackFunc (TensorList, int64_t);

// helper function for TestStack
void _test_stack(TensorList inputs, int64_t dim, StackFunc stack_func) {
  auto const &x = inputs[0];

  auto res = stack_func(inputs, dim);
  auto res_neg = stack_func(inputs, dim - x.dim() - 1);
  std::vector<int64_t> expected_size;
  expected_size.insert(
      expected_size.end(), x.sizes().begin(), x.sizes().begin() + dim);
  expected_size.insert(expected_size.end(), inputs.size());
  expected_size.insert(
      expected_size.end(), x.sizes().begin() + dim, x.sizes().end());

  ASSERT_EQUAL(res, res_neg);
  ASSERT_TRUE(res.sizes().equals(expected_size));

  int d = 0;
  for (auto& t : inputs) {
    ASSERT_EQUAL(res.select(dim, d), t);
    d++;
  }
}

void TestStack(TensorOptions T, Tensor& t) {
  { // at::stack
    auto x = rand({2, 3, 4});
    auto y = rand({2, 3, 4});
    auto z = rand({2, 3, 4});

    auto inputs = {x, y, z};
    for (const auto dim : c10::irange(4)) {
      _test_stack(inputs, dim, at::stack);
    }
  }

  { // at::native::_stack
    auto x = rand({2, 3, 4});
    auto y = rand({2, 3, 4});
    auto z = rand({2, 3, 4});

    auto inputs = {x, y, z};
    for (const auto dim : c10::irange(4)) {
      _test_stack(inputs, dim, at::native::_stack);
    }
  }

  { // at::native::_stack_cpu
    auto x = rand({2, 3, 4});
    auto y = rand({2, 3, 4});
    auto z = rand({2, 3, 4});

    auto inputs = {x, y, z};
    for (const auto dim : c10::irange(4)) {
      _test_stack(inputs, dim, at::native::_stack_cpu);
    }
  }
}

// size / stride
void TestSize(TensorOptions T, Tensor& t) {
  auto scalar = randn({}, T);
  // Throw StartsWith("dimension specified as 0 but tensor has no dimensions")
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(scalar.size(0));
  // Throw StartsWith("dimension specified as -1 but tensor has no dimensions")
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(scalar.size(-1));
  // Throw StartsWith("dimension specified as 0 but tensor has no dimensions")
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(scalar.stride(0));
  // Throw StartsWith("dimension specified as -1 but tensor has no dimensions")
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(scalar.stride(-1));

  auto empty = randn({0}, T);
  ASSERT_EQ(empty.size(0), 0);
  ASSERT_EQ(empty.size(-1), 0);
  ASSERT_EQ(empty.stride(0), 1);
  ASSERT_EQ(empty.stride(-1), 1);
}

void TestMatmul(TensorOptions T, Tensor& t, TensorOptions AccT) {
  auto scalar = randn({}, T);
  auto d1 = randn({3}, T);
  auto d2 = randn({2, 3}, T);

  // 0-d
  // Throw StartsWith("both arguments to matmul need to be at least 1D")
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(scalar.matmul(d2));
  // Throw StartsWith("both arguments to matmul need to be at least 1D")
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(d2.matmul(scalar));

  // 1-d
  ASSERT_ALLCLOSE(d1.matmul(d1), d1.dot(d1));
  ASSERT_ALLCLOSE(d2.matmul(d1), d2.mv(d1));
  auto d1o = randn({2}, T);
  ASSERT_ALLCLOSE(d1o.matmul(d2), d1o.unsqueeze(0).mm(d2).squeeze(0));

  // 2-d
  auto d2o = randn({3, 5}, T);
  ASSERT_ALLCLOSE(d2.matmul(d2o), d2.mm(d2o));

  // > 2-d, 1-d
  auto d3 = randn({5, 2, 3}, T);
  ASSERT_ALLCLOSE(
      d3.matmul(d1), d3.bmm(d1.view({1, 3, 1}).expand({5, 3, 1})).view({5, 2}));
  ASSERT_ALLCLOSE(d1o.matmul(d3), d1o.expand({5, 1, 2}).bmm(d3).view({5, 3}));

  auto d5 = randn({3, 2, 4, 2, 3}, T);
  ASSERT_ALLCLOSE(
      d5.matmul(d1),
      d5.view({24, 2, 3})
          .bmm(d1.view({1, 3, 1}).expand({24, 3, 1}))
          .view({3, 2, 4, 2}));
  ASSERT_ALLCLOSE(
      d1o.matmul(d5),
      d1o.expand({24, 1, 2}).bmm(d5.view({24, 2, 3})).view({3, 2, 4, 3}));

  // > 2-d, 2-d
  // we use a "folding" algorithm in this case of matmul, so the direct
  // comparison to bmm doesn't work; instead, compare to the higher precision
  // computation (technically, we should always do this). Tolerances are
  // selected empirically.
  double atol = 1e-04;
  double rtol = 1e-06;
  d2 = randn({3, 4}, T);
  d2o = randn({4, 2}, T);
  auto result = d5.matmul(d2).to(AccT);

  auto d5Acc = d5.to(AccT);
  auto d2Acc = d2.to(AccT);
  auto acc_result = d5Acc.view({24, 2, 3})
                        .bmm(d2Acc.expand({24, 3, 4}))
                        .view({3, 2, 4, 2, 4});
  ASSERT_ALLCLOSE_TOLERANCES(result, acc_result, atol, rtol);
  ASSERT_ALLCLOSE(
      d2o.matmul(d5),
      d2o.expand({24, 4, 2}).bmm(d5.view({24, 2, 3})).view({3, 2, 4, 4, 3}));

  // > 2-d, > 2-d
  auto d5o = randn({2, 1, 2, 4, 3, 2}, T);
  auto d5_bmm_view =
      d5.expand({2, 3, 2, 4, 2, 3}).contiguous().view({48, 2, 3});
  auto d5o_bmm_view =
      d5o.expand({2, 3, 2, 4, 3, 2}).contiguous().view({48, 3, 2});
  ASSERT_ALLCLOSE(
      d5.matmul(d5o), d5_bmm_view.bmm(d5o_bmm_view).view({2, 3, 2, 4, 2, 2}));

  // non-expandable case
  auto d5wrong = randn({2, 4, 2, 4, 3, 2}, T);
  // Throw Contains("must match the size")
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(d5.matmul(d5wrong));
}

void TestStandardGammaGrad(TensorOptions T, Tensor& t) {
  // check empty
  auto empty = ones({0}, T);
  ASSERT_EQUAL(empty, at::_standard_gamma_grad(empty, empty));

  // check scalar equals one element
  auto one_scalar = ones({}, T).mul(5);
  auto one_with_dim = ones({1}, T).mul(5);
  ASSERT_ALLCLOSE(
      at::_standard_gamma_grad(one_scalar, one_scalar),
      at::_standard_gamma_grad(one_with_dim, one_with_dim).sum());

  // check mixing types
  auto t1 = randn({3, 4}, T);
  auto t2 = randn({3, 4}, T).toType(kDouble);
  // Throw StartsWith("expected scalar type")
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(at::_standard_gamma_grad(t1, t2));
}

void TestWhere(TensorOptions T, Tensor& t) {
  // empty
  auto empty = ones({0}, T);
  auto bT = T.dtype(kByte);
  auto empty_byte = ones({0}, bT);
  ASSERT_EQUAL(empty, at::where(empty_byte, empty, empty));

  // check scalar equals one element
  auto x_scalar = ones({}, T).mul(5);
  auto y_scalar = ones({}, T).mul(7);
  auto cond_scalar = zeros({}, bT);
  auto x_1d = x_scalar.unsqueeze(0);
  auto y_1d = y_scalar.unsqueeze(0);
  auto cond_1d = cond_scalar.unsqueeze(0);
  ASSERT_ALLCLOSE(
      at::where(cond_scalar, x_scalar, y_scalar).unsqueeze(0),
      at::where(cond_1d, x_1d, y_1d));
}

void test(TensorOptions T, TensorOptions AccT) {
  auto t = randn({3, 3}, T);
  TestSplit(T, t);
  TestChunk(T, t);
  TestStack(T, t);
  TestSize(T, t);
  TestMatmul(T, t, AccT);
  TestStandardGammaGrad(T, t);
  TestWhere(T, t);
}

TEST(TestNative, NativeTestCPU) {
#if defined(__aarch64__)
  GTEST_SKIP() << "Known failure on AArch64 (label is too far / stack test mismatch)";
#endif
  manual_seed(123);

  test(at::device(kCPU).dtype(kFloat),
       at::device(kCPU).dtype(kDouble));
}

TEST(TestNative, PoolingConversionOverflow2d) {
  const auto input = at::ones({1, 1, 4, 4}, at::kFloat);
  try {
    (void)at::avg_pool2d(input, {std::numeric_limits<int64_t>::max(), 1});
    FAIL() << "Expected pooling conversion overflow";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(
        error.what(), "value cannot be converted to type int without overflow");
  }
}

TEST(TestNative, PoolingConversionOverflow3d) {
  const auto input = at::ones({1, 1, 4, 4, 4}, at::kFloat);
  try {
    (void)at::avg_pool3d(input, {std::numeric_limits<int64_t>::max(), 1, 1});
    FAIL() << "Expected pooling conversion overflow";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(
        error.what(), "value cannot be converted to type int without overflow");
  }
}

TEST(TestNative, PoolingConversionOverflowQuantized2d) {
  const auto input = at::quantize_per_tensor(
      at::ones({1, 1, 4, 4}, at::kFloat), 0.5, 0, at::kQUInt8);
  try {
    (void)at::avg_pool2d(input, {std::numeric_limits<int64_t>::max(), 1});
    FAIL() << "Expected pooling conversion overflow";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(
        error.what(), "value cannot be converted to type int without overflow");
  }
}

TEST(TestNative, PoolDowncastCompatibility) {
  EXPECT_EQ(at::native::safe_downcast<int32_t>(3.9), 3);
  EXPECT_EQ(
      at::native::safe_downcast<int32_t>(c10::complex<double>{3.9, 0}), 3);
  EXPECT_EQ(at::native::safe_downcast<uint8_t>(true), 1);
  EXPECT_TRUE(at::native::safe_downcast<bool>(42));
  try {
    (void)at::native::safe_downcast<uint8_t>(int64_t{-1});
    FAIL() << "Expected strict integer conversion overflow";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(
        error.what(),
        "value cannot be converted to type dest_t without overflow");
  }
}

// A mode made in C++ starts without a Python wrapper.
static std::shared_ptr<c10::FakeTensorMode> make_cpp_fake_mode(
    bool allow_meta = true,
    bool allow_unsafe_data_ptr_access = true) {
  return std::make_shared<c10::FakeTensorMode>(
      nullptr, nullptr, allow_meta, allow_unsafe_data_ptr_access);
}

TEST(TestNative, FakeDetach) {
  auto mode = make_cpp_fake_mode();
  for (auto layout : {kStrided, kSparse, kSparseCsr, kSparseCsc, kSparseBsr, kSparseBsc}) {
    auto real = at::eye(4);
    if (layout == kSparseBsr || layout == kSparseBsc) {
      real = real.to_sparse(layout, IntArrayRef{2, 2});
    } else if (layout != kStrided) {
      real = real.to_sparse(layout);
    }
    for (auto device : {
             c10::Device(c10::kCPU),
             c10::Device(c10::kCUDA, 1),
             c10::Device(c10::kXPU, 1),
             c10::Device(c10::kPrivateUse1, 1)}) {
      SCOPED_TRACE(c10::str(layout, " ", device));
      auto input = real.to(at::kMeta);
      input.set_requires_grad(true);
      at::init_cpp_fake(input.unsafeGetTensorImpl(), device, mode);
      auto keys = real.key_set();
      input.unsafeGetTensorImpl()->set_fake_dispatch_keys(keys);

      c10::impl::ExcludeDispatchKeyGuard no_fake(c10::DispatchKey::Fake);
      auto result = at::impl::fake_detach(input);
      EXPECT_NE(result.unsafeGetTensorImpl(), input.unsafeGetTensorImpl());
      if (layout == kSparse) {
        auto* src = static_cast<SparseTensorImpl*>(input.unsafeGetTensorImpl());
        auto* dst = static_cast<SparseTensorImpl*>(result.unsafeGetTensorImpl());
        EXPECT_TRUE(dst->indices().is_same(src->indices()));
        EXPECT_TRUE(dst->values().is_same(src->values()));
        EXPECT_EQ(dst->sparse_dim(), src->sparse_dim());
        EXPECT_EQ(dst->dense_dim(), src->dense_dim());
        EXPECT_EQ(dst->coalesced(), src->coalesced());
      } else if (layout != kStrided) {
        auto* src = static_cast<SparseCsrTensorImpl*>(input.unsafeGetTensorImpl());
        auto* dst = static_cast<SparseCsrTensorImpl*>(result.unsafeGetTensorImpl());
        EXPECT_TRUE(dst->compressed_indices().is_same(src->compressed_indices()));
        EXPECT_TRUE(dst->plain_indices().is_same(src->plain_indices()));
        EXPECT_TRUE(dst->values().is_same(src->values()));
      } else {
        EXPECT_TRUE(result.is_alias_of(input));
      }
      EXPECT_EQ(result.layout(), input.layout());
      EXPECT_EQ(result.sizes(), input.sizes());
      EXPECT_EQ(result.dtype(), input.dtype());
      EXPECT_EQ(result.device(), input.device());
      EXPECT_TRUE(result.is_fake());
      EXPECT_EQ(result.unsafeGetTensorImpl()->fake_tensor_mode(), mode);
      EXPECT_EQ(result.unsafeGetTensorImpl()->fake_dispatch_keys(), keys);
      EXPECT_FALSE(result.requires_grad());
      EXPECT_TRUE(input.requires_grad());
      EXPECT_FALSE(c10::impl::FakeTensorModeTLS::in_kernel_invocation());
    }
  }
}

TEST(TestNative, FakeTensorData) {
  for (bool inference : {false, true}) {
    c10::InferenceMode guard(inference);
    for (bool share_version : {false, true}) {
      auto mode = make_cpp_fake_mode();
      auto input = at::empty({1}, at::device(at::kMeta));
      input.set_requires_grad(true);
      at::init_cpp_fake(input.unsafeGetTensorImpl(), c10::Device(c10::kCPU), mode);
      auto constant = at::ones({1});
      mode->set_constant(input.unsafeGetTensorImpl(), constant.getIntrusivePtr());

      auto result = Tensor(share_version ? input.tensor_data() : input.variable_data());
      EXPECT_TRUE(result.is_fake());
      EXPECT_TRUE(result.is_alias_of(input));
      EXPECT_FALSE(result.requires_grad());
      EXPECT_EQ(result.is_inference(), inference);
      auto result_constant = mode->get_constant(result.unsafeGetTensorImpl());
      ASSERT_NE(result_constant, nullptr);
      EXPECT_NE(result_constant.get(), constant.unsafeGetTensorImpl());
      EXPECT_EQ(
          result_constant->storage().unsafeGetStorageImpl(),
          constant.storage().unsafeGetStorageImpl());
      if (!inference) {
        auto version = input._version();
        result.unsafeGetTensorImpl()->bump_version();
        EXPECT_EQ(input._version(), version + share_version);
        EXPECT_EQ(result._version(), 1);
      }

      mode->invalidate_constant_aliases(constant.storage().unsafeGetStorageImpl());
      EXPECT_EQ(mode->get_constant(input.unsafeGetTensorImpl()), nullptr);
      EXPECT_EQ(mode->get_constant(result.unsafeGetTensorImpl()), nullptr);
    }
  }
}

TEST(TestNative, FakeShallowCopyData) {
  auto mode = make_cpp_fake_mode();
  for (auto device : {c10::Device(c10::kCPU), c10::Device(c10::kCUDA, 0)}) {
    auto source_device = device.is_cpu() ? c10::Device(c10::kCUDA, 0)
                                         : c10::Device(c10::kCPU);
    auto input = at::empty({2}, at::device(at::kMeta));
    input.set_requires_grad(true);
    auto source = at::empty({3, 4}, at::device(at::kMeta));
    at::init_cpp_fake(input.unsafeGetTensorImpl(), device, mode);
    at::init_cpp_fake(source.unsafeGetTensorImpl(), source_device, mode);
    const auto version =
        input.unsafeGetTensorImpl()->version_counter().current_version();

    c10::impl::ExcludeDispatchKeyGuard no_fake(c10::DispatchKey::Fake);
    auto& result = at::impl::fake_shallow_copy_data_(input, source);
    EXPECT_TRUE(result.is_same(input));
    EXPECT_EQ(
        input.storage().unsafeGetStorageImpl(),
        source.storage().unsafeGetStorageImpl());
    EXPECT_EQ(input.sizes(), source.sizes());
    EXPECT_EQ(input.strides(), source.strides());
    EXPECT_EQ(input.device(), source_device);
    EXPECT_TRUE(input.is_fake());
    EXPECT_EQ(input.unsafeGetTensorImpl()->fake_tensor_mode(), mode);
    EXPECT_EQ(input.unsafeGetTensorImpl()->version_counter().current_version(), version);
    EXPECT_TRUE(input.requires_grad());
    EXPECT_FALSE(source.requires_grad());
  }
}

TEST(TestNative, AsStridedPreservesFakeTensorMetadata) {
  auto input = at::empty({4}, at::device(at::kMeta));
  auto mode = make_cpp_fake_mode();
  auto* input_impl = input.unsafeGetTensorImpl();
  at::set_and_normalize_fake_device(input_impl, c10::Device(c10::kCPU));
  input_impl->set_fake_tensor_mode(mode);

  auto result = at::native::as_strided_tensorimpl(input, {2, 2}, {2, 1}, 0);
  auto* result_impl = result.unsafeGetTensorImpl();
  auto fake_device = result_impl->fake_device();

  ASSERT_TRUE(fake_device.has_value());
  EXPECT_EQ(*fake_device, c10::Device(c10::kCPU));
  auto handle = result_impl->fake_tensor_mode();
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(handle, mode);
}

TEST(TestNative, InitCppFake) {
  auto mode = make_cpp_fake_mode();
  auto make_fake = [&]() {
    auto t = at::empty({2}, at::device(at::kMeta));
    at::init_cpp_fake(t.unsafeGetTensorImpl(), c10::Device(c10::kCPU), mode);
    auto handle = t.unsafeGetTensorImpl()->fake_tensor_mode();
    EXPECT_EQ(handle, mode);
    EXPECT_EQ(t.device(), c10::Device(c10::kCPU));
    return t;
  };
  auto expect_throws = [](const at::Tensor& t) {
    try {
      t.mutable_data_ptr();
      ADD_FAILURE() << "expected mutable_data_ptr() to throw";
    } catch (const c10::Error& e) {
      EXPECT_NE(
          std::string(e.what()).find(
              "Cannot access data pointer of Tensor (e.g. FakeTensor"),
          std::string::npos);
    }
  };
  struct CaptureWarnings : c10::WarningHandler {
    std::vector<std::string> messages;
    void process(const c10::Warning& warning) override {
      messages.push_back(warning.msg());
    }
  };
  CaptureWarnings handler;
  c10::WarningUtils::WarningHandlerGuard handler_guard(&handler);
  c10::WarningUtils::WarnAlways warn_always;

  auto expect_warns = [&](const at::Tensor& t) {
    auto count = handler.messages.size();
    t.mutable_data_ptr();
    ASSERT_EQ(handler.messages.size(), count + 1);
    const auto& msg = handler.messages.back();
    EXPECT_NE(
        msg.find("Accessing the data pointer of FakeTensor"),
        std::string::npos);
  };

  // Reading the data pointer of a fake of a default mode warns
  auto fake = make_fake();
  expect_warns(fake);

  // Disallowing unsafe access makes fakes made afterwards throw. A view made
  // within a kernel does not mark the shared storage again, but a new fake of
  // it, as the fake fallback makes inside a kernel for a view op's output,
  // does; its fake device is replaced.
  mode->allow_unsafe_data_ptr_access_ = false;
  expect_throws(make_fake());
  expect_warns(fake);
  auto view = at::native::as_strided_tensorimpl(fake, {1}, {1});
  expect_warns(view);
  auto cpu = c10::Device(c10::kCPU);
  auto meta = c10::Device(c10::kMeta);
  {
    c10::impl::InKernelInvocationGuard in_kernel;
    EXPECT_TRUE(view.is_meta());
    at::init_cpp_fake(view.unsafeGetTensorImpl(), meta, mode);
  }
  EXPECT_EQ(view.device(), meta);
  expect_throws(fake);

  // A rejected tensor is left unchanged
  auto expect_rejected = [&](const at::Tensor& t,
                             c10::Device device,
                             const std::shared_ptr<c10::FakeTensorMode>& m,
                             const std::string& msg) {
    auto* impl = t.unsafeGetTensorImpl();
    try {
      at::init_cpp_fake(impl, device, m);
      ADD_FAILURE() << "expected init_cpp_fake to throw";
    } catch (const c10::Error& e) {
      EXPECT_NE(std::string(e.what()).find(msg), std::string::npos) << e.what();
    }
    EXPECT_FALSE(impl->is_fake());
    EXPECT_EQ(impl->fake_tensor_mode(), nullptr);
    EXPECT_FALSE(impl->fake_device().has_value());
    EXPECT_FALSE(impl->storage().unsafeGetStorageImpl()->throw_on_mutable_data_ptr());
    auto count = handler.messages.size();
    t.mutable_data_ptr();
    EXPECT_EQ(handler.messages.size(), count);
  };
  auto strict_mode = make_cpp_fake_mode(
      /*allow_meta=*/false, /*allow_unsafe_data_ptr_access=*/false);
  expect_rejected(at::empty({2}), cpu, mode, "expects a meta tensor");
  expect_rejected(
      at::empty({2}, at::device(at::kMeta)), meta, strict_mode, "allow_meta");
}

TEST(TestNative, InKernelInvocationReportsMetaDevice) {
  auto t = at::empty({2}, at::device(at::kMeta));
  at::set_and_normalize_fake_device(
      t.unsafeGetTensorImpl(), c10::Device(c10::kCPU));
  EXPECT_EQ(t.device(), c10::Device(c10::kCPU));
  {
    // Suspending fake dispatch does not change the reported device.
    c10::impl::ExcludeDispatchKeyGuard no_fake(c10::DispatchKey::Fake);
    EXPECT_EQ(t.device(), c10::Device(c10::kCPU));
  }
  {
    c10::impl::InKernelInvocationGuard in_kernel;
    EXPECT_EQ(t.device(), c10::Device(c10::kMeta));
    at::ThreadLocalState state;
    std::thread worker([&] {
      at::ThreadLocalStateGuard guard(state);
      EXPECT_TRUE(c10::impl::FakeTensorModeTLS::in_kernel_invocation());
      EXPECT_EQ(t.device(), c10::Device(c10::kMeta));
    });
    worker.join();
  }
  EXPECT_FALSE(c10::impl::FakeTensorModeTLS::in_kernel_invocation());
  EXPECT_EQ(t.device(), c10::Device(c10::kCPU));

  EXPECT_THROW(
      {
        c10::impl::InKernelInvocationGuard in_kernel;
        TORCH_CHECK(false, "meta kernel failed");
      },
      c10::Error);
  EXPECT_FALSE(c10::impl::FakeTensorModeTLS::in_kernel_invocation());
}

TEST(TestNative, NativeTestGPU) {
  manual_seed(123);

  if (at::hasCUDA()) {
    test(at::device(kCUDA).dtype(kFloat),
         at::device(kCUDA).dtype(kDouble));
  }
}
