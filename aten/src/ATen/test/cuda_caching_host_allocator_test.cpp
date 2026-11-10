#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/TensorIndexing.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CachingHostAllocator.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <future>
#include <mutex>
#include <optional>
#include <thread>

constexpr int64_t N = 100;

// NOTE: please leave this as the first test to ensure that
// the allocator is not used and stats are zero.
TEST(CachingHostAllocatorTest, check_stats) {
  if (!at::cuda::is_available()) {
    return;
  }

  // Clear the stats and ensure they are zero.
  size_t round_size = std::bit_ceil(static_cast<size_t>(N));
  auto stats = at::getHostAllocator(at::kCUDA)->get_stats();
  ASSERT_EQ(stats.allocations.current, 0);
  ASSERT_EQ(stats.allocations.peak, 0);
  ASSERT_EQ(stats.allocations.allocated, 0);
  ASSERT_EQ(stats.allocations.freed, 0);

  void* ptr{nullptr};
  void* ctx{nullptr};
  {
    auto pinned_tensor = at::empty(
        {N}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    ptr = pinned_tensor.data_ptr();
    ctx = pinned_tensor.storage().data_ptr().get_context();
    auto stats = at::getHostAllocator(at::kCUDA)->get_stats();
    ASSERT_EQ(stats.allocations.current, 1);
    ASSERT_EQ(stats.allocations.peak, 1);
    ASSERT_EQ(stats.allocations.allocated, 1);
    // We dont track active bytes as free blocks are added in process_events
    ASSERT_EQ(stats.host_alloc_time.max, stats.host_alloc_time.min);
    ASSERT_EQ(stats.host_free_time.total, 0);
  }
  // Ensure we reuse the allocation.
  {
    auto pinned_tensor = at::empty(
        {N}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    auto stats = at::getHostAllocator(at::kCUDA)->get_stats();
    ASSERT_EQ(ptr, pinned_tensor.data_ptr());
    ASSERT_EQ(ctx, pinned_tensor.storage().data_ptr().get_context());
    ASSERT_EQ(stats.allocations.current, 1);
    ASSERT_EQ(stats.allocations.peak, 1);
    ASSERT_EQ(stats.allocations.allocated, 1);
    ASSERT_EQ(stats.allocated_bytes.current, round_size);
  }
  // Ensure we don't reuse the allocation, due to size mismatch.
  {
    int64_t new_size = N*2;
    size_t new_round_size = std::bit_ceil(static_cast<size_t>(new_size));
    auto pinned_tensor = at::empty(
        {new_size}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    auto stats = at::getHostAllocator(at::kCUDA)->get_stats();
    ASSERT_NE(ptr, pinned_tensor.data_ptr());
    ASSERT_NE(ctx, pinned_tensor.storage().data_ptr().get_context());
    ASSERT_EQ(stats.allocations.current, 2);
    ASSERT_EQ(stats.allocations.peak, 2);
    ASSERT_EQ(stats.allocations.allocated, 2);
    ASSERT_EQ(stats.allocated_bytes.current, new_round_size + round_size);
    ASSERT_NE(stats.host_alloc_time.total, stats.host_alloc_time.min);
  }

  // Test the empty cache.
  {
    at::getHostAllocator(at::kCUDA)->empty_cache();
    auto stats = at::getHostAllocator(at::kCUDA)->get_stats();
    ASSERT_EQ(stats.allocations.current, 0);
    ASSERT_EQ(stats.allocated_bytes.current, 0);
    ASSERT_EQ(stats.allocations.peak, 2);
    ASSERT_EQ(stats.allocations.allocated, 2);
    ASSERT_EQ(stats.num_host_alloc, 2);
    ASSERT_EQ(stats.num_host_free, 2);
    ASSERT_NE(stats.host_free_time.total, stats.host_free_time.min);
  }

  // Test the reset stats.
  {
    at::getHostAllocator(at::kCUDA)->reset_accumulated_stats();
    at::getHostAllocator(at::kCUDA)->reset_peak_stats();
    auto stats = at::getHostAllocator(at::kCUDA)->get_stats();
    ASSERT_EQ(stats.allocations.peak, 0);
    ASSERT_EQ(stats.allocations.allocated, 0);
    ASSERT_EQ(stats.allocations.freed, 0);
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
  ASSERT_TRUE(at::getHostAllocator(at::kCUDA)->record_event(
      pinned_tensor.data_ptr(),
      pinned_tensor.storage().data_ptr().get_context(),
      at::cuda::getCurrentCUDAStream().unwrap()));

  // Check an tensor constructed with from_blob can be correctly recorded (via
  // the shared data_ptr)
  auto alias_tensor = at::from_blob(
      pinned_tensor.data_ptr(), pinned_tensor.sizes(), pinned_tensor.options());
  ASSERT_TRUE(alias_tensor.is_pinned());

  ASSERT_FALSE(
      alias_tensor.storage().data_ptr().get_context() ==
      pinned_tensor.storage().data_ptr().get_context());
  ASSERT_EQ(alias_tensor.data_ptr(), pinned_tensor.data_ptr());
  ASSERT_TRUE(at::getHostAllocator(at::kCUDA)->record_event(
      alias_tensor.data_ptr(),
      alias_tensor.storage().data_ptr().get_context(),
      at::cuda::getCurrentCUDAStream().unwrap()));

  // Check an tensor constructed with slicing can be correctly recorded (via
  // the shared context)
  auto slice_tensor =
      pinned_tensor.index({at::indexing::Slice(1, at::indexing::None, 2)});
  ASSERT_EQ(
      slice_tensor.storage().data_ptr().get_context(),
      pinned_tensor.storage().data_ptr().get_context());
  ASSERT_NE(slice_tensor.data_ptr(), pinned_tensor.data_ptr());
  ASSERT_TRUE(at::getHostAllocator(at::kCUDA)->record_event(
      slice_tensor.data_ptr(),
      slice_tensor.storage().data_ptr().get_context(),
      at::cuda::getCurrentCUDAStream().unwrap()));

  // Check a tensor that has neither a matching context nor data_ptr cannot be
  // recorded.
  auto alias_slice_tensor = at::from_blob(
      slice_tensor.data_ptr(), slice_tensor.sizes(), slice_tensor.options());
  ASSERT_TRUE(alias_slice_tensor.is_pinned());
  ASSERT_FALSE(at::getHostAllocator(at::kCUDA)->record_event(
      alias_slice_tensor.data_ptr(),
      alias_slice_tensor.storage().data_ptr().get_context(),
      at::cuda::getCurrentCUDAStream().unwrap()));
  ASSERT_NE(
      alias_slice_tensor.storage().data_ptr().get(),
      slice_tensor.storage().data_ptr().get());
}

TEST(CachingHostAllocatorTest, check_raw_allocation) {
  if (!at::cuda::is_available()) {
    return;
  }

  auto data_ptr = at::getHostAllocator(at::kCUDA)->allocate(N);
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
  ASSERT_TRUE(at::getHostAllocator(at::kCUDA)->record_event(
      pinned_tensor.data_ptr(),
      pinned_tensor.storage().data_ptr().get_context(),
      at::cuda::getCurrentCUDAStream().unwrap()));
}

TEST(CachingHostAllocatorTest, check_unknown_tensor) {
  if (!at::cuda::is_available()) {
    return;
  }

  auto unpinned_tensor =
      at::empty({N}, at::TensorOptions().dtype(at::kByte).pinned_memory(false));

  ASSERT_FALSE(at::getHostAllocator(at::kCUDA)->record_event(
      unpinned_tensor.data_ptr(),
      unpinned_tensor.storage().data_ptr().get_context(),
      at::cuda::getCurrentCUDAStream().unwrap()));
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
    ASSERT_TRUE(at::getHostAllocator(at::kCUDA)->record_event(
        ptr, ctx, at::cuda::getCurrentCUDAStream().unwrap()));
  }

  at::getHostAllocator(at::kCUDA)->empty_cache();
  ASSERT_FALSE(at::getHostAllocator(at::kCUDA)->record_event(
      ptr, ctx, at::cuda::getCurrentCUDAStream().unwrap()));
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

namespace {

// Stand-ins so the allocator can be instantiated with no device present.
struct TestStream {
  TestStream() = default;
  /* implicit */ TestStream(c10::Stream) {}
  /* implicit */ operator c10::Stream() const {
    return c10::Stream(c10::Stream::DEFAULT, c10::Device(c10::kCPU));
  }
  bool operator==(const TestStream&) const {
    return true;
  }
};

struct TestEvent {};

} // namespace

template <>
struct std::hash<TestStream> {
  std::size_t operator()(const TestStream&) const noexcept {
    return 0;
  }
};

namespace {

struct ThrowingRecordAllocator
    : at::CachingHostAllocatorImpl<TestStream, TestEvent> {
  void allocate_host_memory(size_t size, void** ptr) override {
    *ptr = std::malloc(size);
  }

  void free_block(at::HostBlock<TestStream>* block) override {
    std::free(block->ptr_);
  }

  void record_stream(std::optional<std::vector<TestEvent>>&, TestStream)
      override {
    // @allow-raw-throw: tests catch(...) path in free() for non-std exceptions
    throw 1;
  }
};

} // namespace

// On a real device every event record fails once the context is poisoned, and
// free() is reached from ~StorageImpl, so an escaping throw would cross a
// noexcept destructor and terminate the process.
TEST(CachingHostAllocatorTest, free_does_not_propagate_record_failure) {
  ThrowingRecordAllocator allocator;

  auto [ptr, ctx] = allocator.allocate(N);
  ASSERT_NE(ptr, nullptr);
  ASSERT_TRUE(allocator.record_event(
      ptr, ctx, c10::Stream(c10::Stream::DEFAULT, c10::Device(c10::kCPU))));

  EXPECT_NO_THROW(allocator.free(ctx));
}

namespace {

// Host memory comes from malloc, and events complete at once.
struct MockAllocator : at::CachingHostAllocatorImpl<TestStream, TestEvent> {
  size_t max_cached_size{SIZE_MAX};
  // Whether the current stream is capturing. free() records no event for a
  // block allocated during capture.
  bool capturing{true};

  bool pinned_use_background_threads() override {
    return false;
  }

  size_t pinned_max_cached_size() const override {
    return max_cached_size;
  }

  void allocate_host_memory(size_t size, void** ptr) override {
    *ptr = std::malloc(size);
  }

  void free_block(at::HostBlock<TestStream>* block) override {
    std::free(block->ptr_);
  }

  void record_stream(std::optional<std::vector<TestEvent>>& events, TestStream)
      override {
    events->emplace_back();
  }

  bool query_event(TestEvent&) override {
    return true;
  }

  TestStream get_current_stream() const override {
    return {};
  }

  bool stream_is_capturing(TestStream) const override {
    return capturing;
  }
};

const auto kStream = c10::Stream(c10::Stream::DEFAULT, c10::Device(c10::kCPU));

// free_block() waits for unblock(), like cudaFreeHost waits for the work
// already queued on the current device.
struct BlockingFreeAllocator : MockAllocator {
  bool wait_until_freeing(std::chrono::seconds timeout) {
    std::unique_lock<std::mutex> g(free_mutex_);
    return free_cv_.wait_for(g, timeout, [&] { return freeing_; });
  }

  void unblock() {
    std::lock_guard<std::mutex> g(free_mutex_);
    unblocked_ = true;
    free_cv_.notify_all();
  }

  void free_block(at::HostBlock<TestStream>* block) override {
    {
      std::unique_lock<std::mutex> g(free_mutex_);
      freeing_ = true;
      free_cv_.notify_all();
      free_cv_.wait(g, [&] { return unblocked_; });
    }
    MockAllocator::free_block(block);
  }

 private:
  std::mutex free_mutex_;
  std::condition_variable free_cv_;
  bool freeing_{false};
  bool unblocked_{false};
};

// Runs stuck() on another thread until it blocks in free_block(), then checks
// that record_event() on the live block ctx, allocate() and free() finish
// meanwhile.
template <typename F>
void expect_progress_while_freeing(
    BlockingFreeAllocator& allocator,
    void* ptr,
    void* ctx,
    F stuck) {
  constexpr auto kTimeout = std::chrono::seconds(10);
  auto freeing_thread = std::async(std::launch::async, stuck);
  bool freeing = allocator.wait_until_freeing(kTimeout);
  auto other = std::async(std::launch::async, [&] {
    bool recorded = allocator.record_event(ptr, ctx, kStream);
    allocator.free(allocator.allocate(N).second);
    return recorded;
  });
  bool done = other.wait_for(kTimeout) == std::future_status::ready;
  // Unblock before checking anything, so that both threads finish.
  allocator.unblock();
  freeing_thread.get();
  EXPECT_TRUE(other.get());
  EXPECT_TRUE(freeing);
  EXPECT_TRUE(done) << "another thread waited for the blocked free_block()";
}

// free_block() and query_event() fail when asked to, as they do on a device
// after a sticky error. live counts the host allocations not yet freed.
struct FailingAllocator : MockAllocator {
  int live{0};
  bool fail_free{false};
  bool fail_query{false};

  void allocate_host_memory(size_t size, void** ptr) override {
    MockAllocator::allocate_host_memory(size, ptr);
    ++live;
  }

  // Frees the memory, then fails once.
  void free_block(at::HostBlock<TestStream>* block) override {
    MockAllocator::free_block(block);
    --live;
    if (fail_free) {
      fail_free = false;
      TORCH_CHECK(false, "free_block failed");
    }
  }

  bool query_event(TestEvent&) override {
    TORCH_CHECK(!fail_query, "query_event failed");
    return true;
  }
};

} // namespace

// empty_cache() used to free blocks while holding locks that record_event()
// takes, so a non_blocking copy on any device waited for cudaFreeHost.
TEST(CachingHostAllocatorTest, empty_cache_frees_default_pool_outside_locks) {
  BlockingFreeAllocator allocator;
  auto [ptr, ctx] = allocator.allocate(N);
  allocator.free(allocator.allocate(N).second);

  expect_progress_while_freeing(
      allocator, ptr, ctx, [&] { allocator.empty_cache(); });
  allocator.free(ctx);
  allocator.empty_cache();
}

TEST(CachingHostAllocatorTest, empty_cache_frees_released_pool_outside_locks) {
  BlockingFreeAllocator allocator;
  auto [ptr, ctx] = allocator.allocate(N);
  c10::MempoolId_t pool_id{1, 0};
  allocator.begin_allocate_to_pool(pool_id, [](c10::Stream) { return true; });
  void* pool_block = allocator.allocate(N).second;
  allocator.end_allocate_to_pool(pool_id);
  allocator.free(pool_block);
  allocator.release_pool(pool_id);

  expect_progress_while_freeing(
      allocator, ptr, ctx, [&] { allocator.empty_cache(); });
  allocator.free(ctx);
  allocator.empty_cache();
}

// A block larger than pinned_max_cached_size is freed once its event
// completes, here in empty_cache() while it holds instance_mutex_.
TEST(CachingHostAllocatorTest, empty_cache_frees_large_pool_block_outside_locks) {
  BlockingFreeAllocator allocator;
  allocator.max_cached_size = 1024;
  allocator.capturing = false;
  auto [ptr, ctx] = allocator.allocate(N);
  c10::MempoolId_t pool_id{1, 0};
  allocator.begin_allocate_to_pool(pool_id, [](c10::Stream) { return true; });
  auto [pool_ptr, pool_block] = allocator.allocate(4096);
  allocator.end_allocate_to_pool(pool_id);
  ASSERT_TRUE(allocator.record_event(pool_ptr, pool_block, kStream));
  allocator.free(pool_block);
  allocator.release_pool(pool_id);

  expect_progress_while_freeing(
      allocator, ptr, ctx, [&] { allocator.empty_cache(); });
  allocator.free(ctx);
  allocator.empty_cache();
}

TEST(CachingHostAllocatorTest, free_frees_large_block_outside_locks) {
  BlockingFreeAllocator allocator;
  allocator.max_cached_size = 1024;
  auto [ptr, ctx] = allocator.allocate(N);
  void* large = allocator.allocate(4096).second;

  expect_progress_while_freeing(
      allocator, ptr, ctx, [&] { allocator.free(large); });
  allocator.free(ctx);
  allocator.empty_cache();
}

// empty_cache() takes blocks out of their pools before freeing them, so after
// a failure it must still free every block it took out.
TEST(CachingHostAllocatorTest, empty_cache_frees_detached_blocks_on_error) {
  FailingAllocator allocator;
  // Cached blocks in two size buckets.
  allocator.free(allocator.allocate(N).second);
  allocator.free(allocator.allocate(4 * N).second);
  allocator.fail_free = true;
  EXPECT_ANY_THROW(allocator.empty_cache());
  EXPECT_EQ(allocator.live, 0);

  allocator.free(allocator.allocate(N).second);
  allocator.capturing = false;
  c10::MempoolId_t pool_id{1, 0};
  allocator.begin_allocate_to_pool(pool_id, [](c10::Stream) { return true; });
  auto [ptr, ctx] = allocator.allocate(N);
  allocator.end_allocate_to_pool(pool_id);
  ASSERT_TRUE(allocator.record_event(ptr, ctx, kStream));
  allocator.free(ctx);
  allocator.release_pool(pool_id);
  allocator.fail_query = true;
  EXPECT_ANY_THROW(allocator.empty_cache());
  // The default pool's cached block is freed despite the failure. The private
  // pool's block is not, since the failed query dropped its event.
  EXPECT_EQ(allocator.live, 1);
}

// empty_cache() holds instance_mutex_ exclusively while it processes the
// events of released private pools. Returning a block whose event completed
// used to take instance_mutex_ again to find its pool, which fails with
// "Resource deadlock avoided".
TEST(CachingHostAllocatorTest, empty_cache_released_pool_with_completed_event) {
  if (!at::cuda::is_available()) {
    return;
  }

  auto* allocator = at::getHostAllocator(at::kCUDA);
  auto pool_id = at::cuda::graph_pool_handle();
  auto stream = at::cuda::getStreamFromPool();

  void* ptr{nullptr};
  void* ctx{nullptr};
  {
    allocator->begin_allocate_to_pool(
        pool_id, [](c10::Stream) { return true; });
    auto pinned_tensor = at::empty(
        {N}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    allocator->end_allocate_to_pool(pool_id);
    ptr = pinned_tensor.data_ptr();
    ctx = pinned_tensor.storage().data_ptr().get_context();
    auto segments = allocator->get_segments();
    auto seg = std::find_if(segments.begin(), segments.end(), [&](const auto& s) {
      return s.address == reinterpret_cast<size_t>(ptr);
    });
    ASSERT_NE(seg, segments.end());
    ASSERT_EQ(seg->owner_private_pool_id, pool_id);
    ASSERT_TRUE(allocator->record_event(ptr, ctx, stream.unwrap()));
    allocator->release_pool(pool_id);
  }
  stream.synchronize();

  ASSERT_NO_THROW(allocator->empty_cache());
  ASSERT_FALSE(allocator->record_event(ptr, ctx, stream.unwrap()));
}

// Each pool keeps its own bucket stats. getStats() adds them up, and erasing a
// private pool leaves the totals unchanged.
TEST(CachingHostAllocatorTest, stats_include_private_pools) {
  if (!at::cuda::is_available()) {
    return;
  }

  auto* allocator = at::getHostAllocator(at::kCUDA);
  // Let earlier tests' pending events complete, so that their blocks do not
  // come back during this test.
  at::cuda::device_synchronize();
  allocator->empty_cache();
  auto before = allocator->get_stats();
  auto pool_id = at::cuda::graph_pool_handle();
  {
    allocator->begin_allocate_to_pool(
        pool_id, [](c10::Stream) { return true; });
    auto pinned_tensor = at::empty(
        {N}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    allocator->end_allocate_to_pool(pool_id);
    auto stats = allocator->get_stats();
    ASSERT_EQ(stats.active_requests.current, before.active_requests.current + 1);
    ASSERT_EQ(stats.allocations.current, before.allocations.current + 1);
    allocator->release_pool(pool_id);
  }
  auto released = allocator->get_stats();

  allocator->empty_cache();
  auto after = allocator->get_stats();
  ASSERT_EQ(after.allocations.current, before.allocations.current);
  ASSERT_EQ(after.active_requests.allocated, released.active_requests.allocated);
  ASSERT_EQ(after.active_requests.freed, released.active_requests.freed);
  ASSERT_EQ(after.active_requests.peak, released.active_requests.peak);
  ASSERT_EQ(after.active_bytes.allocated, released.active_bytes.allocated);
  ASSERT_EQ(after.active_bytes.freed, released.active_bytes.freed);
  ASSERT_EQ(after.active_bytes.peak, released.active_bytes.peak);
  ASSERT_EQ(after.bucket_allocation, released.bucket_allocation);
}

// The default pool and a private pool used to update shared bucket stats, each
// under its own free list mutex, so using both at once lost updates.
TEST(CachingHostAllocatorTest, stats_concurrent_pools) {
  if (!at::cuda::is_available()) {
    return;
  }

  auto* allocator = at::getHostAllocator(at::kCUDA);
  at::cuda::device_synchronize();
  allocator->empty_cache();
  auto before = allocator->get_stats();
  auto pool_id = at::cuda::graph_pool_handle();
  auto pool_stream = at::cuda::getStreamFromPool();
  allocator->begin_allocate_to_pool(
      pool_id, [s = pool_stream.unwrap()](c10::Stream stream) { return stream == s; });

  constexpr int kIters = 100000;
  auto churn = [](std::optional<c10::cuda::CUDAStream> stream) {
    std::optional<c10::cuda::CUDAStreamGuard> guard;
    if (stream) {
      guard.emplace(*stream);
    }
    for (int i = 0; i < kIters; ++i) {
      at::empty({N}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    }
  };
  std::thread in_pool(churn, pool_stream);
  std::thread in_default(churn, std::nullopt);
  in_pool.join();
  in_default.join();
  allocator->end_allocate_to_pool(pool_id);
  allocator->release_pool(pool_id);

  auto after = allocator->get_stats();
  ASSERT_EQ(after.active_requests.allocated - before.active_requests.allocated, 2 * kIters);
  ASSERT_EQ(after.active_requests.freed - before.active_requests.freed, 2 * kIters);
  ASSERT_EQ(after.active_requests.current, before.active_requests.current);
  int64_t bytes = 2 * kIters * std::bit_ceil(static_cast<size_t>(N));
  ASSERT_EQ(after.active_bytes.allocated - before.active_bytes.allocated, bytes);
  ASSERT_EQ(after.active_bytes.freed - before.active_bytes.freed, bytes);
  ASSERT_EQ(after.active_bytes.current, before.active_bytes.current);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  at::manual_seed(42);
  return RUN_ALL_TESTS();
}
