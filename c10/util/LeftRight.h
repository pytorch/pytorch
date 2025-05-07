#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/Synchronized.h>
#include <array>
#include <atomic>
#include <mutex>
#include <thread>

namespace c10 {

namespace detail {

struct IncrementRAII final {
 public:
  explicit IncrementRAII(std::atomic<int32_t>* counter) : _counter(counter) {
    _counter->fetch_add(1);
  }

  ~IncrementRAII() {
    _counter->fetch_sub(1);
  }
  IncrementRAII(IncrementRAII&&) = delete;
  IncrementRAII& operator=(IncrementRAII&&) = delete;

 private:
  std::atomic<int32_t>* _counter;

  C10_DISABLE_COPY_AND_ASSIGN(IncrementRAII);
};

} // namespace detail

// LeftRight wait-free readers synchronization primitive
// https://hal.archives-ouvertes.fr/hal-01207881/document
//
// LeftRight is quite easy to use (it can make an arbitrary
// data structure permit wait-free reads), but it has some
// particular performance characteristics you should be aware
// of if you're deciding to use it:
//
//  - Reads still incur an atomic write (this is how LeftRight
//    keeps track of how long it needs to keep around the old
//    data structure)
//
//  - Writes get executed twice, to keep both the left and right
//    versions up to date.  So if your write is expensive or
//    nondeterministic, this is also an inappropriate structure
//
// LeftRight is used fairly rarely in PyTorch's codebase.  If you
// are still not sure if you need it or not, consult your local
// C++ expert.
//
template <class T>
class LeftRight final {
 public:
  template <class... Args>
  explicit LeftRight(const Args&... args)
      : _counters{{{0}, {0}}},
        _foregroundCounterIndex(0),
        _foregroundDataIndex(0),
        _data{{T{args...}, T{args...}}} {}

  // Copying and moving would not be threadsafe.
  // Needs more thought and careful design to make that work.
  LeftRight(const LeftRight&) = delete;
  LeftRight(LeftRight&&) noexcept = delete;
  LeftRight& operator=(const LeftRight&) = delete;
  LeftRight& operator=(LeftRight&&) noexcept = delete;

  ~LeftRight() {
    // wait until any potentially running writers are finished
    { std::unique_lock<std::mutex> lock(_writeMutex); }

    // wait until any potentially running readers are finished
    while (_counters[0].load() != 0 || _counters[1].load() != 0) {
      std::this_thread::yield();
    }
  }

  template <typename F>
  auto read(F&& readFunc) const {
    detail::IncrementRAII _increment_counter(
        &_counters[_foregroundCounterIndex.load()]);

    return std::forward<F>(readFunc)(_data[_foregroundDataIndex.load()]);
  }

  // Throwing an exception in writeFunc is ok but causes the state to be either
  // the old or the new state, depending on if the first or the second call to
  // writeFunc threw.
  template <typename F>
  auto write(F&& writeFunc) {
    std::unique_lock<std::mutex> lock(_writeMutex);

    return _write(std::forward<F>(writeFunc));
  }

 private:
  template <class F>
  auto _write(const F& writeFunc) {
    /*
     * Assume, A is in background and B in foreground. In simplified terms, we
     * want to do the following:
     * 1. Write to A (old background)
     * 2. Switch A/B
     * 3. Write to B (new background)
     *
     * More detailed algorithm (explanations on why this is important are below
     * in code):
     * 1. Write to A
     * 2. Switch A/B data pointers
     * 3. Wait until A counter is zero
     * 4. Switch A/B counters
     * 5. Wait until B counter is zero
     * 6. Write to B
     */

    auto localDataIndex = _foregroundDataIndex.load();

    // 1. Write to A
    _callWriteFuncOnBackgroundInstance(writeFunc, localDataIndex);

    // 2. Switch A/B data pointers
    localDataIndex = localDataIndex ^ 1;
    _foregroundDataIndex = localDataIndex;

    /*
     * 3. Wait until A counter is zero
     *
     * In the previous write run, A was foreground and B was background.
     * There was a time after switching _foregroundDataIndex (B to foreground)
     * and before switching _foregroundCounterIndex, in which new readers could
     * have read B but incremented A's counter.
     *
     * In this current run, we just switched _foregroundDataIndex (A back to
     * foreground), but before writing to the new background B, we have to make
     * sure A's counter was zero briefly, so all these old readers are gone.
     */
    auto localCounterIndex = _foregroundCounterIndex.load();
    _waitForBackgroundCounterToBeZero(localCounterIndex);

    /*
     * 4. Switch A/B counters
     *
     * Now that we know all readers on B are really gone, we can switch the
     * counters and have new readers increment A's counter again, which is the
     * correct counter since they're reading A.
     */
    localCounterIndex = localCounterIndex ^ 1;
    _foregroundCounterIndex = localCounterIndex;

    /*
     * 5. Wait until B counter is zero
     *
     * This waits for all the readers on B that came in while both data and
     * counter for B was in foreground, i.e. normal readers that happened
     * outside of that brief gap between switching data and counter.
     */
    _waitForBackgroundCounterToBeZero(localCounterIndex);

    // 6. Write to B
    return _callWriteFuncOnBackgroundInstance(writeFunc, localDataIndex);
  }

  template <class F>
  auto _callWriteFuncOnBackgroundInstance(
      const F& writeFunc,
      uint8_t localDataIndex) {
    try {
      return writeFunc(_data[localDataIndex ^ 1]);
    } catch (...) {
      // recover invariant by copying from the foreground instance
      _data[localDataIndex ^ 1] = _data[localDataIndex];
      // rethrow
      throw;
    }
  }

  void _waitForBackgroundCounterToBeZero(uint8_t counterIndex) {
    while (_counters[counterIndex ^ 1].load() != 0) {
      std::this_thread::yield();
    }
  }

  mutable std::array<std::atomic<int32_t>, 2> _counters;
  std::atomic<uint8_t> _foregroundCounterIndex;
  std::atomic<uint8_t> _foregroundDataIndex;
  std::array<T, 2> _data;
  std::mutex _writeMutex;
};

// RWSafeLeftRightWrapper is API compatible with LeftRight and uses a
// read-write lock to protect T (data).
template <class T>
class RWSafeLeftRightWrapper final {
 public:
  template <class... Args>
  explicit RWSafeLeftRightWrapper(const Args&... args) : data_{args...} {}

  // RWSafeLeftRightWrapper is not copyable or moveable since LeftRight
  // is not copyable or moveable.
  RWSafeLeftRightWrapper(const RWSafeLeftRightWrapper&) = delete;
  RWSafeLeftRightWrapper(RWSafeLeftRightWrapper&&) noexcept = delete;
  RWSafeLeftRightWrapper& operator=(const RWSafeLeftRightWrapper&) = delete;
  RWSafeLeftRightWrapper& operator=(RWSafeLeftRightWrapper&&) noexcept = delete;
  ~RWSafeLeftRightWrapper() = default;

  template <typename F>
  // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
  auto read(F&& readFunc) const {
    return data_.withLock(
        [&readFunc](T const& data) { return std::forward<F>(readFunc)(data); });
  }

  template <typename F>
  // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
  auto write(F&& writeFunc) {
    return data_.withLock(
        [&writeFunc](T& data) { return std::forward<F>(writeFunc)(data); });
  }

 private:
  c10::Synchronized<T> data_;
};

} // namespace c10
