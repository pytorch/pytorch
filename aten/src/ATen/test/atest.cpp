#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <c10/core/StorageImpl.h>
#include <c10/util/ScopeExit.h>
#include <c10/util/irange.h>

#include <array>
#include <iostream>
using namespace std;
using namespace at;

class atest : public ::testing::Test {
 protected:
  void SetUp() override {
    x_tensor = tensor({10, -1, 0, 1, -10});
    y_tensor = tensor({-10, 1, 0, -1, 10});
    x_logical = tensor({1, 1, 0, 1, 0});
    y_logical = tensor({0, 1, 0, 1, 1});
    x_float = tensor({2.0, 2.4, 5.6, 7.0, 36.0});
    y_float = tensor({1.0, 1.1, 8.7, 10.0, 24.0});
  }

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  Tensor x_tensor, y_tensor;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  Tensor x_logical, y_logical;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  Tensor x_float, y_float;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  const int INT = 1;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  const int FLOAT = 2;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  const int INTFLOAT = 3;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  const int INTBOOL = 5;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  const int INTBOOLFLOAT = 7;
};

namespace BinaryOpsKernel {
const int IntMask = 1; // test dtype = kInt
const int FloatMask = 2; // test dtype = kFloat
const int BoolMask = 4; // test dtype = kBool
} // namespace BinaryOpsKernel

template <typename T, typename... Args>
void unit_binary_ops_test(
    T func,
    const Tensor& x_tensor,
    const Tensor& y_tensor,
    const Tensor& exp,
    ScalarType dtype,
    Args... args) {
  auto out_tensor = empty({5}, dtype);
  func(out_tensor, x_tensor.to(dtype), y_tensor.to(dtype), args...);
  ASSERT_EQ(out_tensor.dtype(), dtype);
  if (dtype == kFloat) {
    ASSERT_TRUE(exp.to(dtype).allclose(out_tensor));
  } else {
    ASSERT_TRUE(exp.to(dtype).equal(out_tensor));
  }
}

/*
  template function for running binary operator test
  - exp: expected output
  - func: function to be tested
  - option: 3 bits,
    - 1st bit: Test op over integer tensors
    - 2nd bit: Test op over float tensors
    - 3rd bit: Test op over boolean tensors
    For example, if function should be tested over integer/boolean but not for
    float, option will be 1 * 1 + 0 * 2 + 1 * 4 = 5. If tested over all the
    type, option should be 7.
*/
template <typename T, typename... Args>
void run_binary_ops_test(
    T func,
    const Tensor& x_tensor,
    const Tensor& y_tensor,
    const Tensor& exp,
    int option,
    Args... args) {
  // Test op over integer tensors
  if (option & BinaryOpsKernel::IntMask) {
    unit_binary_ops_test(func, x_tensor, y_tensor, exp, kInt, args...);
  }

  // Test op over float tensors
  if (option & BinaryOpsKernel::FloatMask) {
    unit_binary_ops_test(func, x_tensor, y_tensor, exp, kFloat, args...);
  }

  // Test op over boolean tensors
  if (option & BinaryOpsKernel::BoolMask) {
    unit_binary_ops_test(func, x_tensor, y_tensor, exp, kBool, args...);
  }
}

void trace() {
  Tensor foo = rand({12, 12});

  // ASSERT foo is 2-dimensional and holds floats.
  auto foo_a = foo.accessor<float, 2>();
  float trace = 0;

  for (const auto i : c10::irange(foo_a.size(0))) {
    trace += foo_a[i][i];
  }

  ASSERT_FLOAT_EQ(foo.trace().item<float>(), trace);
}

TEST_F(atest, operators) {
  int a = 0b10101011;
  int b = 0b01111011;

  auto a_tensor = tensor({a});
  auto b_tensor = tensor({b});

  ASSERT_TRUE(tensor({~a}).equal(~a_tensor));
  ASSERT_TRUE(tensor({a | b}).equal(a_tensor | b_tensor));
  ASSERT_TRUE(tensor({a & b}).equal(a_tensor & b_tensor));
  ASSERT_TRUE(tensor({a ^ b}).equal(a_tensor ^ b_tensor));
}

TEST_F(atest, logical_and_operators) {
  auto exp_tensor = tensor({0, 1, 0, 1, 0});
  run_binary_ops_test(
      logical_and_out, x_logical, y_logical, exp_tensor, INTBOOL);
}

TEST_F(atest, logical_or_operators) {
  auto exp_tensor = tensor({1, 1, 0, 1, 1});
  run_binary_ops_test(
      logical_or_out, x_logical, y_logical, exp_tensor, INTBOOL);
}

TEST_F(atest, logical_xor_operators) {
  auto exp_tensor = tensor({1, 0, 0, 0, 1});
  run_binary_ops_test(
      logical_xor_out, x_logical, y_logical, exp_tensor, INTBOOL);
}

TEST_F(atest, lt_operators) {
  auto exp_tensor = tensor({0, 0, 0, 0, 1});
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&)>(
      lt_out, x_logical, y_logical, exp_tensor, INTBOOL);
}

TEST_F(atest, le_operators) {
  auto exp_tensor = tensor({0, 1, 1, 1, 1});
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&)>(
      le_out, x_logical, y_logical, exp_tensor, INTBOOL);
}

TEST_F(atest, gt_operators) {
  auto exp_tensor = tensor({1, 0, 0, 0, 0});
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&)>(
      gt_out, x_logical, y_logical, exp_tensor, INTBOOL);
}

TEST_F(atest, ge_operators) {
  auto exp_tensor = tensor({1, 1, 1, 1, 0});
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&)>(
      ge_out, x_logical, y_logical, exp_tensor, INTBOOL);
}

TEST_F(atest, eq_operators) {
  auto exp_tensor = tensor({0, 1, 1, 1, 0});
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&)>(
      eq_out, x_logical, y_logical, exp_tensor, INTBOOL);
}

TEST_F(atest, ne_operators) {
  auto exp_tensor = tensor({1, 0, 0, 0, 1});
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&)>(
      ne_out, x_logical, y_logical, exp_tensor, INTBOOL);
}

TEST_F(atest, add_operators) {
  auto exp_tensor = tensor({-10, 1, 0, -1, 10});
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Scalar&)>(
      add_out, x_tensor, y_tensor, exp_tensor, INTBOOL, 2);
}

TEST_F(atest, max_operators) {
  auto exp_tensor = tensor({10, 1, 0, 1, 10});
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&)>(
      max_out, x_tensor, y_tensor, exp_tensor, INTBOOLFLOAT);
}

TEST_F(atest, min_operators) {
  auto exp_tensor = tensor({-10, -1, 0, -1, -10});
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&)>(
      min_out, x_tensor, y_tensor, exp_tensor, INTBOOLFLOAT);
}

TEST_F(atest, sigmoid_backward_operator) {
  auto exp_tensor = tensor({-1100, 0, 0, -2, 900});
  // only test with type Float
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&)>(
      sigmoid_backward_out, x_tensor, y_tensor, exp_tensor, FLOAT);
}

TEST_F(atest, fmod_tensor_operators) {
  auto exp_tensor = tensor({0.0, 0.2, 5.6, 7.0, 12.0});
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&)>(
      fmod_out, x_float, y_float, exp_tensor, INTFLOAT);
}

TEST_F(atest, from_blob_empty_without_deleter) {
  auto t = from_blob(nullptr, {0}, kFloat);

  ASSERT_TRUE(t.defined());
  EXPECT_TRUE(t.is_cpu());
  EXPECT_EQ(t.scalar_type(), kFloat);
  EXPECT_EQ(t.sizes(), IntArrayRef({0}));
  EXPECT_EQ(t.numel(), 0);
  EXPECT_EQ(t.data_ptr(), nullptr);
  EXPECT_EQ(t.storage().data_ptr().get_context(), nullptr);
  EXPECT_EQ(t.storage().nbytes(), 0);
  EXPECT_EQ(t.storage().allocator(), nullptr);
  EXPECT_FALSE(t.storage().resizable());

  auto lazy_clone = at::_lazy_clone(t);
  EXPECT_TRUE(lazy_clone.is_cpu());
  EXPECT_EQ(lazy_clone.sizes(), t.sizes());
  EXPECT_EQ(lazy_clone.const_data_ptr(), nullptr);
  EXPECT_EQ(lazy_clone.storage().nbytes(), 0);
}

namespace {

int from_blob_storage_factory_calls = 0;

enum class FromBlobStorageFactoryMode { Default, WrappedContext, GuardedDataPtr };
FromBlobStorageFactoryMode from_blob_storage_factory_mode =
    FromBlobStorageFactoryMode::Default;
int from_blob_wrapped_context_deletions = 0;

void delete_from_blob_wrapped_context(void* context) {
  delete static_cast<c10::DataPtr*>(context);
  ++from_blob_wrapped_context_deletions;
}

c10::intrusive_ptr<c10::StorageImpl> make_from_blob_storage_impl(
    c10::StorageImpl::use_byte_size_t use_byte_size,
    c10::SymInt size_bytes,
    c10::DataPtr data_ptr,
    c10::Allocator* allocator,
    bool resizable) {
  ++from_blob_storage_factory_calls;
  if (data_ptr == nullptr) {
    return c10::make_intrusive<c10::StorageImpl>(
        use_byte_size, std::move(size_bytes), allocator, resizable);
  }
  if (from_blob_storage_factory_mode ==
      FromBlobStorageFactoryMode::WrappedContext) {
    auto* context = new c10::DataPtr(std::move(data_ptr));
    data_ptr = c10::DataPtr{
        context->get(),
        context,
        &delete_from_blob_wrapped_context,
        context->device()};
  }
  auto storage = c10::make_intrusive<c10::StorageImpl>(
      use_byte_size,
      std::move(size_bytes),
      std::move(data_ptr),
      allocator,
      resizable);
  if (from_blob_storage_factory_mode ==
      FromBlobStorageFactoryMode::GuardedDataPtr) {
    storage->release_data_and_set_meta_custom_data_ptr_error_msg_(
        "PrivateUse1 test storage access is guarded");
  }
  return storage;
}

void register_from_blob_storage_factory() {
  // Storage factories cannot be unregistered, so also support repeated tests.
  if (c10::GetStorageImplCreate(kPrivateUse1) == nullptr) {
    c10::SetStorageImplCreate(kPrivateUse1, &make_from_blob_storage_impl);
  }
}

class FromBlobRecordingAllocator final : public c10::Allocator {
 public:
  explicit FromBlobRecordingAllocator(Device device) : device_(device) {}

  c10::DataPtr allocate(size_t /*nbytes*/) override {
    ++allocation_calls_;
    return {nullptr, device_};
  }

  void copy_data(void* dest, const void* src, size_t count) const override {
    default_copy_data(dest, src, count);
  }

  int allocation_calls() const {
    return allocation_calls_;
  }

 private:
  Device device_;
  int allocation_calls_ = 0;
};

} // namespace

TEST_F(atest, from_blob_privateuse1_storage_factory) {
  register_from_blob_storage_factory();
  ASSERT_EQ(
      c10::GetStorageImplCreate(kPrivateUse1), &make_from_blob_storage_impl);

  std::array<float, 3> data = {1, 2, 3};
  const Device device(kPrivateUse1, 0);
  const auto calls_before = from_blob_storage_factory_calls;
  int deleter_calls = 0;
  void* deleted_pointer = nullptr;
  {
    Storage shared_storage;
    {
      auto t = from_blob(
          data.data(),
          {3},
          [&](void* pointer) {
            ++deleter_calls;
            deleted_pointer = pointer;
          },
          TensorOptions().dtype(kFloat).device(device),
          device);
      EXPECT_EQ(from_blob_storage_factory_calls, calls_before + 1);
      EXPECT_EQ(t.device(), device);
      EXPECT_EQ(t.data_ptr(), data.data());
      EXPECT_EQ(t.storage().nbytes(), data.size() * sizeof(float));
      EXPECT_FALSE(t.storage().resizable());
      EXPECT_EQ(deleter_calls, 0);
      shared_storage = t.storage();
    }
    EXPECT_EQ(shared_storage.data(), data.data());
    EXPECT_EQ(deleter_calls, 0);
  }
  EXPECT_EQ(deleter_calls, 1);
  EXPECT_EQ(deleted_pointer, data.data());
}

TEST_F(atest, from_blob_empty_privateuse1_storage_factory) {
  register_from_blob_storage_factory();
  ASSERT_EQ(
      c10::GetStorageImplCreate(kPrivateUse1), &make_from_blob_storage_impl);

  const Device device(kPrivateUse1, 1);
  const auto calls_before = from_blob_storage_factory_calls;
  auto t = for_blob(nullptr, {0})
               .options(TensorOptions().dtype(kFloat).device(device))
               .target_device(device)
               .make_tensor();

  EXPECT_EQ(from_blob_storage_factory_calls, calls_before + 1);
  EXPECT_EQ(t.device(), device);
  EXPECT_EQ(t.scalar_type(), kFloat);
  EXPECT_EQ(t.sizes(), IntArrayRef({0}));
  EXPECT_EQ(t.numel(), 0);
  EXPECT_EQ(t.data_ptr(), nullptr);
  EXPECT_EQ(t.storage().data_ptr().get_context(), nullptr);
  EXPECT_EQ(t.storage().device(), device);
  EXPECT_EQ(t.storage().nbytes(), 0);
  EXPECT_EQ(t.storage().allocator(), nullptr);
  EXPECT_FALSE(t.storage().resizable());
}

TEST_F(atest, from_blob_empty_with_allocator) {
  register_from_blob_storage_factory();
  ASSERT_EQ(
      c10::GetStorageImplCreate(kPrivateUse1), &make_from_blob_storage_impl);

  for (const auto device_type : {kCUDA, kPrivateUse1}) {
    const Device device(device_type, 1);
    SCOPED_TRACE(device);
    // Simulate allocation on a different current device, without CUDA kernels.
    FromBlobRecordingAllocator allocator(Device(device_type, 0));
    const auto calls_before = from_blob_storage_factory_calls;
    auto t = for_blob(nullptr, {0})
                 .options(TensorOptions().dtype(kFloat).device(device))
                 .target_device(device)
                 .allocator(&allocator)
                 .resizeable_storage()
                 .make_tensor();

    EXPECT_EQ(allocator.allocation_calls(), 0);
    EXPECT_EQ(
        from_blob_storage_factory_calls,
        calls_before + (device_type == kPrivateUse1 ? 1 : 0));
    EXPECT_EQ(t.device(), device);
    EXPECT_EQ(t.scalar_type(), kFloat);
    EXPECT_EQ(t.sizes(), IntArrayRef({0}));
    EXPECT_EQ(t.numel(), 0);
    EXPECT_EQ(t.data_ptr(), nullptr);
    EXPECT_EQ(t.storage().data_ptr().get_context(), nullptr);
    EXPECT_EQ(t.storage().device(), device);
    EXPECT_EQ(t.storage().nbytes(), 0);
    EXPECT_EQ(t.storage().allocator(), &allocator);
    EXPECT_TRUE(t.storage().resizable());
  }
}

TEST_F(atest, from_blob_empty_preserves_factory_storage) {
  register_from_blob_storage_factory();
  ASSERT_EQ(
      c10::GetStorageImplCreate(kPrivateUse1), &make_from_blob_storage_impl);

  const Device device(kPrivateUse1, 1);
  for (const auto mode : {FromBlobStorageFactoryMode::WrappedContext,
                         FromBlobStorageFactoryMode::GuardedDataPtr}) {
    from_blob_storage_factory_mode = mode;
    auto reset_mode = c10::make_scope_exit([] {
      from_blob_storage_factory_mode = FromBlobStorageFactoryMode::Default;
    });
    const auto calls_before = from_blob_storage_factory_calls;
    const auto deletions_before = from_blob_wrapped_context_deletions;
    {
      auto t = for_blob(nullptr, {0})
                   .options(TensorOptions().dtype(kFloat).device(device))
                   .target_device(device)
                   .make_tensor();
      EXPECT_EQ(from_blob_storage_factory_calls, calls_before + 1);
      EXPECT_EQ(t.device(), device);
      EXPECT_EQ(from_blob_wrapped_context_deletions, deletions_before);
      if (mode == FromBlobStorageFactoryMode::WrappedContext) {
        EXPECT_NE(t.storage().data_ptr().get_context(), nullptr);
      } else {
        EXPECT_THROW(t.storage().data_ptr(), c10::Error);
        t.storage().unsafeGetStorageImpl()->clear_data_ptr_access_error_msg_();
        EXPECT_EQ(t.storage().data_ptr().get_context(), nullptr);
      }
      EXPECT_EQ(t.const_data_ptr(), nullptr);
      EXPECT_EQ(t.numel(), 0);
    }
    EXPECT_EQ(
        from_blob_wrapped_context_deletions,
        deletions_before +
            (mode == FromBlobStorageFactoryMode::WrappedContext ? 1 : 0));
  }
}

// TEST_CASE( "atest", "[]" ) {
TEST_F(atest, atest) {
  manual_seed(123);

  auto foo = rand({12, 6});

  ASSERT_EQ(foo.size(0), 12);
  ASSERT_EQ(foo.size(1), 6);

  foo = foo + foo * 3;
  foo -= 4;

  Scalar a = 4;
  float b = a.to<float>();
  ASSERT_EQ(b, 4);

  foo = ((foo * foo) == (foo.pow(3))).to(kByte);
  foo = 2 + (foo + 1);
  // foo = foo[3];
  auto foo_v = foo.accessor<uint8_t, 2>();

  for (const auto i : c10::irange(foo_v.size(0))) {
    for (const auto j : c10::irange(foo_v.size(1))) {
      foo_v[i][j]++;
    }
  }

  ASSERT_TRUE(foo.equal(4 * ones({12, 6}, kByte)));

  trace();

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  float data[] = {1, 2, 3, 4, 5, 6};

  auto f = from_blob(data, {1, 2, 3});
  auto f_a = f.accessor<float, 3>();

  ASSERT_EQ(f_a[0][0][0], 1.0);
  ASSERT_EQ(f_a[0][1][1], 5.0);

  ASSERT_EQ(f.strides()[0], 6);
  ASSERT_EQ(f.strides()[1], 3);
  ASSERT_EQ(f.strides()[2], 1);
  ASSERT_EQ(f.sizes()[0], 1);
  ASSERT_EQ(f.sizes()[1], 2);
  ASSERT_EQ(f.sizes()[2], 3);

  // TODO(ezyang): maybe do a more precise exception type.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(f.resize_({3, 4, 5}), std::exception);
  {
    int isgone = 0;
    {
      auto f2 = from_blob(data, {1, 2, 3}, [&](void*) { isgone++; });
    }
    ASSERT_EQ(isgone, 1);
  }
  {
    int isgone = 0;
    Tensor a_view;
    {
      auto f2 = from_blob(data, {1, 2, 3}, [&](void*) { isgone++; });
      a_view = f2.view({3, 2, 1});
    }
    ASSERT_EQ(isgone, 0);
    a_view.reset();
    ASSERT_EQ(isgone, 1);
  }

  if (at::hasCUDA()) {
    int isgone = 0;
    {
      auto base = at::empty({1, 2, 3}, TensorOptions(kCUDA));
      auto f2 = from_blob(base.mutable_data_ptr(), {1, 2, 3}, [&](void*) { isgone++; });
    }
    ASSERT_EQ(isgone, 1);

    // Attempt to specify the wrong device in from_blob
    auto t = at::empty({1, 2, 3}, TensorOptions(kCUDA, 0));
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    EXPECT_ANY_THROW(from_blob(t.data_ptr(), {1, 2, 3}, at::Device(kCUDA, 1)));

    // Infers the correct device
    auto t_ = from_blob(t.data_ptr(), {1, 2, 3}, kCUDA);
    ASSERT_EQ(t_.device(), at::Device(kCUDA, 0));
  }
}
