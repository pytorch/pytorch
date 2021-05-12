#include <c10/util/LeftRight.h>
#include <gtest/gtest.h>
#include <vector>

using c10::LeftRight;
using std::vector;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LeftRightTest, givenInt_whenWritingAndReading_thenChangesArePresent) {
  LeftRight<int> obj;

  obj.write([](int& obj) { obj = 5; });
  int read = obj.read([](const int& obj) { return obj; });
  EXPECT_EQ(5, read);

  // check changes are also present in background copy
  obj.write([](int&) {}); // this switches to the background copy
  read = obj.read([](const int& obj) { return obj; });
  EXPECT_EQ(5, read);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LeftRightTest, givenVector_whenWritingAndReading_thenChangesArePresent) {
  LeftRight<vector<int>> obj;

  obj.write([](vector<int>& obj) { obj.push_back(5); });
  vector<int> read = obj.read([](const vector<int>& obj) { return obj; });
  EXPECT_EQ((vector<int>{5}), read);

  obj.write([](vector<int>& obj) { obj.push_back(6); });
  read = obj.read([](const vector<int>& obj) { return obj; });
  EXPECT_EQ((vector<int>{5, 6}), read);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LeftRightTest, givenVector_whenWritingReturnsValue_thenValueIsReturned) {
  LeftRight<vector<int>> obj;

  auto a = obj.write([](vector<int>&) -> int { return 5; });
  static_assert(std::is_same<int, decltype(a)>::value, "");
  EXPECT_EQ(5, a);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LeftRightTest, readsCanBeConcurrent) {
  LeftRight<int> obj;
  std::atomic<int> num_running_readers{0};

  std::thread reader1([&]() {
    obj.read([&](const int&) {
      ++num_running_readers;
      while (num_running_readers.load() < 2) {
      }
    });
  });

  std::thread reader2([&]() {
    obj.read([&](const int&) {
      ++num_running_readers;
      while (num_running_readers.load() < 2) {
      }
    });
  });

  // the threads only finish after both entered the read function.
  // if LeftRight didn't allow concurrency, this would cause a deadlock.
  reader1.join();
  reader2.join();
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LeftRightTest, writesCanBeConcurrentWithReads_readThenWrite) {
  LeftRight<int> obj;
  std::atomic<bool> reader_running{false};
  std::atomic<bool> writer_running{false};

  std::thread reader([&]() {
    obj.read([&](const int&) {
      reader_running = true;
      while (!writer_running.load()) {
      }
    });
  });

  std::thread writer([&]() {
    // run read first, write second
    while (!reader_running.load()) {
    }

    obj.write([&](int&) { writer_running = true; });
  });

  // the threads only finish after both entered the read function.
  // if LeftRight didn't allow concurrency, this would cause a deadlock.
  reader.join();
  writer.join();
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LeftRightTest, writesCanBeConcurrentWithReads_writeThenRead) {
  LeftRight<int> obj;
  std::atomic<bool> writer_running{false};
  std::atomic<bool> reader_running{false};

  std::thread writer([&]() {
    obj.read([&](const int&) {
      writer_running = true;
      while (!reader_running.load()) {
      }
    });
  });

  std::thread reader([&]() {
    // run write first, read second
    while (!writer_running.load()) {
    }

    obj.read([&](const int&) { reader_running = true; });
  });

  // the threads only finish after both entered the read function.
  // if LeftRight didn't allow concurrency, this would cause a deadlock.
  writer.join();
  reader.join();
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LeftRightTest, writesCannotBeConcurrentWithWrites) {
  LeftRight<int> obj;
  std::atomic<bool> first_writer_started{false};
  std::atomic<bool> first_writer_finished{false};

  std::thread writer1([&]() {
    obj.write([&](int&) {
      first_writer_started = true;
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      first_writer_finished = true;
    });
  });

  std::thread writer2([&]() {
    // make sure the other writer runs first
    while (!first_writer_started.load()) {
    }

    obj.write([&](int&) {
      // expect the other writer finished before this one starts
      EXPECT_TRUE(first_writer_finished.load());
    });
  });

  writer1.join();
  writer2.join();
}

namespace {
class MyException : public std::exception {};
} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LeftRightTest, whenReadThrowsException_thenThrowsThrough) {
  LeftRight<int> obj;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_THROW(obj.read([](const int&) { throw MyException(); }), MyException);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LeftRightTest, whenWriteThrowsException_thenThrowsThrough) {
  LeftRight<int> obj;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_THROW(obj.write([](int&) { throw MyException(); }), MyException);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(
    LeftRightTest,
    givenInt_whenWriteThrowsExceptionOnFirstCall_thenResetsToOldState) {
  LeftRight<int> obj;

  obj.write([](int& obj) { obj = 5; });

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_THROW(
      obj.write([](int& obj) {
        obj = 6;
        throw MyException();
      }),
      MyException);

  // check reading it returns old value
  int read = obj.read([](const int& obj) { return obj; });
  EXPECT_EQ(5, read);

  // check changes are also present in background copy
  obj.write([](int&) {}); // this switches to the background copy
  read = obj.read([](const int& obj) { return obj; });
  EXPECT_EQ(5, read);
}

// note: each write is executed twice, on the foreground and background copy.
// We need to test a thrown exception in either call is handled correctly.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(
    LeftRightTest,
    givenInt_whenWriteThrowsExceptionOnSecondCall_thenKeepsNewState) {
  LeftRight<int> obj;

  obj.write([](int& obj) { obj = 5; });
  bool write_called = false;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_THROW(
      obj.write([&](int& obj) {
        obj = 6;
        if (write_called) {
          // this is the second time the write callback is executed
          throw MyException();
        } else {
          write_called = true;
        }
      }),
      MyException);

  // check reading it returns new value
  int read = obj.read([](const int& obj) { return obj; });
  EXPECT_EQ(6, read);

  // check changes are also present in background copy
  obj.write([](int&) {}); // this switches to the background copy
  read = obj.read([](const int& obj) { return obj; });
  EXPECT_EQ(6, read);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LeftRightTest, givenVector_whenWriteThrowsException_thenResetsToOldState) {
  LeftRight<vector<int>> obj;

  obj.write([](vector<int>& obj) { obj.push_back(5); });

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_THROW(
      obj.write([](vector<int>& obj) {
        obj.push_back(6);
        throw MyException();
      }),
      MyException);

  // check reading it returns old value
  vector<int> read = obj.read([](const vector<int>& obj) { return obj; });
  EXPECT_EQ((vector<int>{5}), read);

  // check changes are also present in background copy
  obj.write([](vector<int>&) {}); // this switches to the background copy
  read = obj.read([](const vector<int>& obj) { return obj; });
  EXPECT_EQ((vector<int>{5}), read);
}
