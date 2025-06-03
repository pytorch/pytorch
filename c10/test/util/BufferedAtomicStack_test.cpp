#include <c10/util/BufferedAtomicStack.h>
#include <c10/util/irange.h>

#include <gtest/gtest.h>

TEST(BufferedAtomicStack, TestSerialAcquireRelease) {
  constexpr size_t capacity = 5;
  c10::BufferedAtomicStack<int> queue(capacity);

  auto* a = queue.acquire();
  auto* b = queue.acquire();

  EXPECT_EQ(*a, 0);
  EXPECT_EQ(*b, 0);

  *b = 1;

  queue.release(b);

  *a = 2;

  queue.release(a);

  a = queue.acquire();
  b = queue.acquire();
  EXPECT_EQ(*a, 2);
  EXPECT_EQ(*b, 1);

  std::vector<int*> values;
  for ([[maybe_unused]] const auto _ : c10::irange(2, capacity)) {
    auto* v = values.emplace_back(queue.acquire());
    EXPECT_EQ(*v, 0);
  }

  EXPECT_EQ(queue.try_acquire(), nullptr);

  *values.back() = 3;
  queue.release(values.back());
  values.erase(values.end() - 1);

  auto* v = queue.try_acquire();
  EXPECT_EQ(*v, 3);
}

TEST(BufferedAtomicStack, TestPointerAcquireRelease) {
  constexpr size_t capacity = 5;
  c10::BufferedAtomicStack<int*> queue(capacity);

  int** a = queue.acquire();
  int** b = queue.acquire();

  EXPECT_EQ(*a, nullptr);
  EXPECT_EQ(*b, nullptr);

  *b = new int(1);

  queue.release(b);

  *a = new int(2);

  queue.release(a);

  a = queue.acquire();
  b = queue.acquire();
  EXPECT_EQ(**a, 2);
  EXPECT_EQ(**b, 1);

  delete *b;
  delete *a;

  queue.release(b);
  queue.release(a);
}

TEST(BufferedAtomicStack, TestPointerAcquireScopedRelease) {
  constexpr size_t capacity = 5;
  c10::BufferedAtomicStack<int*> queue(capacity);

  std::unique_ptr<int> a_ptr(std::make_unique<int>(2)),
      b_ptr(std::make_unique<int>(1));

  {
    auto a = queue.acquire_scoped();
    auto b = queue.acquire_scoped();

    EXPECT_EQ(*a, nullptr);
    EXPECT_EQ(*b, nullptr);

    *b = b_ptr.get();
    *a = a_ptr.get();
  }

  auto a = queue.acquire_scoped();
  auto b = queue.acquire_scoped();

  EXPECT_EQ(**a, 2);
  EXPECT_EQ(**b, 1);
}

TEST(BufferedAtomicStack, TestConcurrentAcquireRelease) {
  struct MyStruct {
    uint64_t a;
    uint64_t b;
  };

  const size_t capacity = std::thread::hardware_concurrency();
  c10::BufferedAtomicStack<MyStruct> queue(capacity, 1, 2);

  constexpr size_t iters_per_thread = 100000;

  std::vector<std::thread> threads;
  for ([[maybe_unused]] const auto _ : c10::irange(capacity)) {
    threads.emplace_back([&queue]() {
      for ([[maybe_unused]] const auto _ : c10::irange(iters_per_thread)) {
        auto v = queue.acquire_scoped();
        (*v).a++;
        (*v).b++;
      }
    });
  }

  std::for_each(threads.begin(), threads.end(), [](auto& t) { t.join(); });
}

TEST(BufferedAtomicStack, TestSerialAcquireReleaseWithSpin) {
  constexpr size_t capacity = 5;

  c10::BufferedAtomicStack<int, /* use_cv_on_empty= */ true> queue_false(
      capacity);
  c10::BufferedAtomicStack<int, /* use_cv_on_empty= */ true> queue_true(
      capacity);

  auto exec = [](auto& queue) {
    auto* a = queue.acquire();
    auto* b = queue.acquire();

    EXPECT_EQ(*a, 0);
    EXPECT_EQ(*b, 0);

    *b = 1;

    queue.release(b);

    *a = 2;

    queue.release(a);

    a = queue.acquire();
    b = queue.acquire();
    EXPECT_EQ(*a, 2);
    EXPECT_EQ(*b, 1);

    std::vector<int*> values;
    for ([[maybe_unused]] const auto _ : c10::irange(2, capacity)) {
      auto* v = values.emplace_back(queue.acquire());
      EXPECT_EQ(*v, 0);
    }

    std::atomic_bool started = false;

    std::thread t([&]() {
      started = true;
      EXPECT_EQ(*queue.acquire_scoped(), 3);
    });

    do {
      std::this_thread::yield();

    } while (!started);

    *values.back() = 3;
    queue.release(values.back());
    values.erase(values.end() - 1);

    t.join();
  };

  exec(queue_false);
  exec(queue_true);
}
