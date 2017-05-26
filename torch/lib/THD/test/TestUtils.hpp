#include <THPP/tensors/THTensor.hpp>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

constexpr char RANK_ENV[] = "RANK";
constexpr char WORLD_SIZE_ENV[] = "WORLD_SIZE";
constexpr char MASTER_PORT_ENV[] = "MASTER_PORT";
constexpr char MASTER_ADDR_ENV[] = "MASTER_ADDR";

struct Barrier {
  Barrier() : _count(0) {}
  Barrier(std::size_t count) : _count(count) {}

  void wait() {
    std::unique_lock<std::mutex> lock{_mutex};
    if (--_count == 0) {
      _cv.notify_all();
    } else {
      _cv.wait(lock);
    }
  }

private:
  std::mutex _mutex;
  std::condition_variable _cv;
  std::size_t _count;
};

template<typename T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
    check_equal(T x, T y, int ulp = 5) {
  auto eps = std::numeric_limits<T>::epsilon();
  auto min = std::numeric_limits<T>::min();
  return (std::abs(x-y) < eps * std::abs(x+y) * ulp) || (std::abs(x-y) < min);
}

template<typename T>
typename std::enable_if<std::numeric_limits<T>::is_integer, bool>::type
    check_equal(T x, T y) {
  return x == y;
}

template<typename T>
std::shared_ptr<thpp::THTensor<T>> buildTensor(std::vector<long> shape, T value) {
  auto tensor = std::make_shared<thpp::THTensor<T>>();
  tensor->resize(shape);
  tensor->fill(value);
  return tensor;
}

template<typename T>
inline bool contains(std::vector<T> v, T value) {
  return std::find(v.begin(), v.end(), value) != v.end();
}

inline long nowInMilliseconds() {
  return std::chrono::duration_cast<std::chrono::milliseconds>
      (std::chrono::system_clock::now().time_since_epoch()).count();
}

inline long long factorial(int n) {
  long long a = 1;
  for (long long i = 1; i <= n; ++i) { a *= i; }
  return a;
}

#define ASSERT_TENSOR_VALUE(T, tensor, value) {            \
  for (std::size_t idx = 0; idx < (tensor).numel(); idx++) \
    assert(check_equal(                                    \
      reinterpret_cast<T*>((tensor).data())[idx], static_cast<T>(value) \
    ));                                                    \
}

#define ASSERT_THROWS(exception, expr) {                       \
  try { (expr); assert(false); } catch (const exception& e) {} \
}
