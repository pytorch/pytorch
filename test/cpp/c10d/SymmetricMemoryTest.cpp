#include <gtest/gtest.h>

#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include <atomic>
#include <cstdint>
#include <utility>

namespace c10d::symmetric_memory {
namespace {

uint64_t get_unique_test_alloc_id() {
  static std::atomic<uint64_t> next_alloc_id{1};
  return next_alloc_id.fetch_add(1, std::memory_order_relaxed);
}

class TestSymmetricMemoryAllocator final : public SymmetricMemoryAllocator {
 public:
  ~TestSymmetricMemoryAllocator() override {
    delete[] static_cast<uint8_t*>(allocation_);
  }

  at::Tensor make_tensor(
      void* ptr,
      c10::IntArrayRef sizes,
      c10::IntArrayRef strides,
      c10::ScalarType dtype,
      c10::Device device,
      const std::optional<std::string>& group_name,
      std::function<void(void*)> deleter) override {
    ++make_tensor_calls_;
    sizes_ = sizes.vec();
    strides_ = strides.vec();
    dtype_ = dtype;
    device_ = device;
    make_tensor_group_name_ = group_name;
    saw_deleter_ = static_cast<bool>(deleter);
    if (offset_storage_ptr_) {
      EXPECT_FALSE(deleter);
      ptr = static_cast<uint8_t*>(ptr) + 1;
    }
    return SymmetricMemoryAllocator::make_tensor(
        ptr, sizes, strides, dtype, device, group_name, std::move(deleter));
  }

  void* alloc(
      size_t size,
      int /*device_idx*/,
      const std::optional<std::string>& /*group_name*/) override {
    allocation_ = new uint8_t[size];
    allocation_size_ = size;
    return allocation_;
  }

  void free(void* ptr) override {
    EXPECT_EQ(ptr, allocation_);
    delete[] static_cast<uint8_t*>(ptr);
    allocation_ = nullptr;
    ++free_calls_;
  }

  size_t get_alloc_size(void* ptr) override {
    EXPECT_EQ(ptr, allocation_);
    return allocation_size_;
  }

  c10::intrusive_ptr<SymmetricMemory> rendezvous(
      void* ptr,
      const std::optional<std::string>& /*group_name*/) override {
    rendezvous_ptr_ = ptr;
    ++rendezvous_calls_;
    return nullptr;
  }

  bool has_multicast_support(int /*device_idx*/) override {
    return false;
  }

  c10::DeviceType supported_device_type() override {
    return c10::DeviceType::CPU;
  }

  std::string name() override {
    return "TEST";
  }

  bool has_allocation(void* ptr) override {
    has_allocation_ptr_ = ptr;
    ++has_allocation_calls_;
    return ptr == allocation_;
  }

  int make_tensor_calls_ = 0;
  int free_calls_ = 0;
  int rendezvous_calls_ = 0;
  int has_allocation_calls_ = 0;
  bool saw_deleter_ = false;
  bool offset_storage_ptr_ = false;
  void* allocation_ = nullptr;
  void* rendezvous_ptr_ = nullptr;
  void* has_allocation_ptr_ = nullptr;
  size_t allocation_size_ = 0;
  std::optional<std::string> make_tensor_group_name_;
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
  c10::ScalarType dtype_ = c10::ScalarType::Undefined;
  c10::Device device_ = c10::Device(c10::DeviceType::CPU);
};

TEST(SymmetricMemoryAllocatorTest, EmptyUsesBackendTensorWrapper) {
  auto allocator = c10::make_intrusive<TestSymmetricMemoryAllocator>();
  register_allocator(c10::DeviceType::CPU, allocator);
  const std::optional<std::string> group_name = "test_group";

  auto tensor = empty_strided_p2p(
      {2, 3},
      {3, 1},
      c10::ScalarType::Float,
      c10::Device(c10::DeviceType::CPU),
      group_name,
      std::nullopt);

  EXPECT_EQ(allocator->make_tensor_calls_, 1);
  EXPECT_TRUE(allocator->saw_deleter_);
  EXPECT_EQ(allocator->sizes_, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(allocator->strides_, (std::vector<int64_t>{3, 1}));
  EXPECT_EQ(allocator->dtype_, c10::ScalarType::Float);
  EXPECT_EQ(allocator->device_, c10::Device(c10::DeviceType::CPU));
  EXPECT_EQ(allocator->make_tensor_group_name_, group_name);
  EXPECT_EQ(tensor.sizes(), (c10::IntArrayRef{2, 3}));
  EXPECT_EQ(tensor.strides(), (c10::IntArrayRef{3, 1}));
  EXPECT_EQ(tensor.storage().data_ptr().get(), allocator->allocation_);

  EXPECT_EQ(rendezvous(tensor, group_name), nullptr);
  EXPECT_EQ(allocator->rendezvous_calls_, 1);
  EXPECT_EQ(allocator->rendezvous_ptr_, allocator->allocation_);

  EXPECT_TRUE(is_symm_mem_tensor(tensor));
  EXPECT_EQ(allocator->has_allocation_calls_, 1);
  EXPECT_EQ(allocator->has_allocation_ptr_, allocator->allocation_);

  tensor.reset();
  EXPECT_EQ(allocator->free_calls_, 1);
}

TEST(SymmetricMemoryAllocatorTest, PersistentEmptyUsesBackendTensorWrapper) {
  auto allocator = c10::make_intrusive<TestSymmetricMemoryAllocator>();
  register_allocator(c10::DeviceType::CPU, allocator);
  const std::optional<std::string> group_name = "persistent_group";

  const uint64_t alloc_id = get_unique_test_alloc_id();
  auto tensor = empty_strided_p2p(
      {2, 3},
      {3, 1},
      c10::ScalarType::Float,
      c10::Device(c10::DeviceType::CPU),
      group_name,
      alloc_id);
  auto* data_ptr = tensor.data_ptr();

  EXPECT_EQ(allocator->make_tensor_calls_, 1);
  EXPECT_FALSE(allocator->saw_deleter_);
  EXPECT_EQ(allocator->make_tensor_group_name_, group_name);

  tensor.reset();
  allocator->make_tensor_group_name_.reset();
  tensor = empty_strided_p2p(
      {2, 3},
      {3, 1},
      c10::ScalarType::Float,
      c10::Device(c10::DeviceType::CPU),
      group_name,
      alloc_id);

  EXPECT_EQ(allocator->make_tensor_calls_, 2);
  EXPECT_EQ(allocator->make_tensor_group_name_, group_name);
  EXPECT_EQ(tensor.data_ptr(), data_ptr);
  EXPECT_EQ(allocator->free_calls_, 0);
}

TEST(SymmetricMemoryAllocatorTest, RejectsChangedStoragePointer) {
  auto allocator = c10::make_intrusive<TestSymmetricMemoryAllocator>();
  allocator->offset_storage_ptr_ = true;
  register_allocator(c10::DeviceType::CPU, allocator);

  const uint64_t alloc_id = get_unique_test_alloc_id();
  EXPECT_THROW(
      empty_strided_p2p(
          {2, 3},
          {3, 1},
          c10::ScalarType::Float,
          c10::Device(c10::DeviceType::CPU),
          std::nullopt,
          alloc_id),
      c10::Error);
}

} // namespace
} // namespace c10d::symmetric_memory
