#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/TensorIndexing.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CachingHostAllocator.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>

constexpr int64_t N = 100;

// NOTE: please leave this as the first test to ensure that
// the allocator is not used and stats are zero.
TEST(CachingHostAllocatorTest, check_stats) {
  if (!at::cuda::is_available()) {
    return;
  }

  // Clear the stats and ensure they are zero.
  size_t round_size = c10::llvm::PowerOf2Ceil(N);
  auto stats = at::cuda::CachingHostAllocator_getStats();
  ASSERT_EQ(stats.allocation.current, 0);
  ASSERT_EQ(stats.allocation.peak, 0);
  ASSERT_EQ(stats.allocation.allocated, 0);
  ASSERT_EQ(stats.allocation.freed, 0);

  void* ptr{nullptr};
  void* ctx{nullptr};
  {
    auto pinned_tensor = at::empty(
        {N}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    ptr = pinned_tensor.data_ptr();
    ctx = pinned_tensor.storage().data_ptr().get_context();
    auto stats = at::cuda::CachingHostAllocator_getStats();
    ASSERT_EQ(stats.allocation.current, 1);
    ASSERT_EQ(stats.allocation.peak, 1);
    ASSERT_EQ(stats.allocation.allocated, 1);
    ASSERT_EQ(stats.allocation.freed, 0);
    ASSERT_EQ(stats.segment.allocated, 1);
    ASSERT_EQ(stats.segment.freed, 0);
    ASSERT_EQ(stats.reserved_bytes.current, round_size);
    ASSERT_EQ(stats.allocated_bytes.current, round_size);
    ASSERT_EQ(stats.host_alloc_time.max, stats.host_alloc_time.min);
    ASSERT_EQ(stats.host_free_time.total, 0);
  }
  // Ensure we reuse the allocation.
  {
    auto pinned_tensor = at::empty(
        {N}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    auto stats = at::cuda::CachingHostAllocator_getStats();
    ASSERT_EQ(ptr, pinned_tensor.data_ptr());
    ASSERT_EQ(ctx, pinned_tensor.storage().data_ptr().get_context());
    ASSERT_EQ(stats.allocation.current, 1);
    ASSERT_EQ(stats.allocation.peak, 1);
    ASSERT_EQ(stats.allocation.allocated, 2);
    ASSERT_EQ(stats.allocation.freed, 1);
    ASSERT_EQ(stats.segment.allocated, 1);
    ASSERT_EQ(stats.segment.freed, 0);
    ASSERT_EQ(stats.reserved_bytes.current, round_size);
    ASSERT_EQ(stats.allocated_bytes.current, round_size);
  }
  // Ensure we don't reuse the allocation, due to size mismatch.
  {
    int64_t new_size = N*2;
    size_t new_round_size = c10::llvm::PowerOf2Ceil(new_size);
    auto pinned_tensor = at::empty(
        {new_size}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    auto stats = at::cuda::CachingHostAllocator_getStats();
    ASSERT_NE(ptr, pinned_tensor.data_ptr());
    ASSERT_NE(ctx, pinned_tensor.storage().data_ptr().get_context());
    ASSERT_EQ(stats.allocation.current, 1);
    ASSERT_EQ(stats.allocation.peak, 2);
    ASSERT_EQ(stats.allocation.allocated, 3);
    ASSERT_EQ(stats.allocation.freed, 2);
    ASSERT_EQ(stats.segment.allocated, 2);
    ASSERT_EQ(stats.segment.freed, 0);
    ASSERT_EQ(stats.reserved_bytes.current, round_size + new_round_size);
    ASSERT_EQ(stats.allocated_bytes.current, new_round_size);
    ASSERT_NE(stats.host_alloc_time.total, stats.host_alloc_time.min);
  }

  // Test the empty cache.
  {
    at::cuda::CachingHostAllocator_emptyCache();
    auto stats = at::cuda::CachingHostAllocator_getStats();
    ASSERT_EQ(stats.allocation.current, 0);
    ASSERT_EQ(stats.allocated_bytes.current, 0);
    ASSERT_EQ(stats.allocation.peak, 2);
    ASSERT_EQ(stats.allocation.allocated, 3);
    ASSERT_EQ(stats.allocation.freed, 3);
    ASSERT_EQ(stats.segment.allocated, 2);
    ASSERT_EQ(stats.segment.freed, 2);
    ASSERT_EQ(stats.num_host_alloc, 2);
    ASSERT_EQ(stats.num_host_free, 2);
    ASSERT_NE(stats.host_free_time.total, stats.host_free_time.min);
  }

  // Test the reset stats.
  {
    at::cuda::CachingHostAllocator_resetAccumulatedStats();
    at::cuda::CachingHostAllocator_resetPeakStats();
    auto stats = at::cuda::CachingHostAllocator_getStats();
    ASSERT_EQ(stats.allocation.peak, 0);
    ASSERT_EQ(stats.allocation.allocated, 0);
    ASSERT_EQ(stats.allocation.freed, 0);
    ASSERT_EQ(stats.allocated_bytes.peak, 0);
    ASSERT_EQ(stats.num_host_alloc, 0);
    ASSERT_EQ(stats.num_host_free, 0);
  }

  // At this point, the allocator should be empty, and stats should be zero,
  // leaving the test harness in a clean state for the next test.
}

TEST(CachingHostAllocatorTest, pinned_alias_slice) {
  if (!at::cuda::is_available()) {
    return;
  }

  // Check a standard pinned tensor can be correctly recorded.
  auto pinned_tensor =
      at::empty({N}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
  ASSERT_TRUE(pinned_tensor.is_pinned());
  ASSERT_TRUE(at::cuda::CachingHostAllocator_recordEvent(
      pinned_tensor.data_ptr(),
      pinned_tensor.storage().data_ptr().get_context(),
      at::cuda::getCurrentCUDAStream()));

  // Check an tensor constructed with from_blob can be correctly recorded (via
  // the shared data_ptr)
  auto alias_tensor = at::from_blob(
      pinned_tensor.data_ptr(), pinned_tensor.sizes(), pinned_tensor.options());
  ASSERT_TRUE(alias_tensor.is_pinned());

  ASSERT_FALSE(
      alias_tensor.storage().data_ptr().get_context() ==
      pinned_tensor.storage().data_ptr().get_context());
  ASSERT_EQ(alias_tensor.data_ptr(), pinned_tensor.data_ptr());
  ASSERT_TRUE(at::cuda::CachingHostAllocator_recordEvent(
      alias_tensor.data_ptr(),
      alias_tensor.storage().data_ptr().get_context(),
      at::cuda::getCurrentCUDAStream()));

  // Check an tensor constructed with slicing can be correctly recorded (via
  // the shared context)
  auto slice_tensor =
      pinned_tensor.index({at::indexing::Slice(1, at::indexing::None, 2)});
  ASSERT_EQ(
      slice_tensor.storage().data_ptr().get_context(),
      pinned_tensor.storage().data_ptr().get_context());
  ASSERT_NE(slice_tensor.data_ptr(), pinned_tensor.data_ptr());
  ASSERT_TRUE(at::cuda::CachingHostAllocator_recordEvent(
      slice_tensor.data_ptr(),
      slice_tensor.storage().data_ptr().get_context(),
      at::cuda::getCurrentCUDAStream()));

  // Check a tensor that has neither a matching context nor data_ptr cannot be
  // recorded.
  auto alias_slice_tensor = at::from_blob(
      slice_tensor.data_ptr(), slice_tensor.sizes(), slice_tensor.options());
  ASSERT_TRUE(alias_slice_tensor.is_pinned());
  ASSERT_FALSE(at::cuda::CachingHostAllocator_recordEvent(
      alias_slice_tensor.data_ptr(),
      alias_slice_tensor.storage().data_ptr().get_context(),
      at::cuda::getCurrentCUDAStream()));
  ASSERT_NE(
      alias_slice_tensor.storage().data_ptr().get(),
      slice_tensor.storage().data_ptr().get());
}

TEST(CachingHostAllocatorTest, check_raw_allocation) {
  if (!at::cuda::is_available()) {
    return;
  }

  auto data_ptr = at::cuda::getCachingHostAllocator()->allocate(N);
  class UserDataDeleter {
   public:
    explicit UserDataDeleter(std::unique_ptr<void, c10::DeleterFnPtr> ptr)
        : ptr_(std::move(ptr)) {}

   private:
    std::unique_ptr<void, c10::DeleterFnPtr> ptr_;
  };
  auto* user_data_deleter = new UserDataDeleter(data_ptr.move_context());

  struct IOBuf {
    explicit IOBuf(void* buf, void* ctx, std::function<void(void*)> deleter)
        : buf_(buf), ctx_(ctx), deleter_(std::move(deleter)) {}
    void* buf_;
    void* ctx_;
    std::function<void(void*)> deleter_;
    ~IOBuf() {
      deleter_(ctx_);
    }
  };
  auto iobuf =
      std::make_unique<IOBuf>(data_ptr.get(), user_data_deleter, [](void* ctx) {
        delete static_cast<UserDataDeleter*>(ctx);
      });
  auto pinned_tensor =
      at::for_blob(iobuf->buf_, {N})
          .context(
              iobuf.release(),
              [](void* ctx) { delete static_cast<IOBuf*>(ctx); })
          .make_tensor();

  ASSERT_TRUE(pinned_tensor.is_pinned());
  ASSERT_TRUE(at::cuda::CachingHostAllocator_recordEvent(
      pinned_tensor.data_ptr(),
      pinned_tensor.storage().data_ptr().get_context(),
      at::cuda::getCurrentCUDAStream()));
}

TEST(CachingHostAllocatorTest, check_unknown_tensor) {
  if (!at::cuda::is_available()) {
    return;
  }

  auto unpinned_tensor =
      at::empty({N}, at::TensorOptions().dtype(at::kByte).pinned_memory(false));

  ASSERT_FALSE(at::cuda::CachingHostAllocator_recordEvent(
      unpinned_tensor.data_ptr(),
      unpinned_tensor.storage().data_ptr().get_context(),
      at::cuda::getCurrentCUDAStream()));
}

TEST(CachingHostAllocatorTest, check_empty_cache) {
  if (!at::cuda::is_available()) {
    return;
  }

  void* ptr{nullptr};
  void* ctx{nullptr};
  {
    auto pinned_tensor = at::empty(
        {N}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    ptr = pinned_tensor.data_ptr();
    ctx = pinned_tensor.storage().data_ptr().get_context();
    ASSERT_TRUE(at::cuda::CachingHostAllocator_recordEvent(
        ptr, ctx, at::cuda::getCurrentCUDAStream()));
  }

  at::cuda::CachingHostAllocator_emptyCache();
  ASSERT_FALSE(at::cuda::CachingHostAllocator_recordEvent(
      ptr, ctx, at::cuda::getCurrentCUDAStream()));
}

TEST(CachingHostAllocatorTest, check_reuse) {
  if (!at::cuda::is_available()) {
    return;
  }

  void* ptr{nullptr};
  void* ctx{nullptr};
  {
    auto pinned_tensor = at::empty(
        {N}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    ptr = pinned_tensor.data_ptr();
    ctx = pinned_tensor.storage().data_ptr().get_context();
  }
  // Ensure we reuse the allocation.
  {
    auto pinned_tensor = at::empty(
        {N}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    ASSERT_EQ(ptr, pinned_tensor.data_ptr());
    ASSERT_EQ(ctx, pinned_tensor.storage().data_ptr().get_context());
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  at::manual_seed(42);
  return RUN_ALL_TESTS();
}
