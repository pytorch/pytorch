#include <queue>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <c10/util/Optional.h>

template<typename T>
class BlockingQueue {
  std::queue<T> queue;
  std::mutex mutex;
  std::condition_variable cv;
public:
  c10::optional<T> pop() {
    using namespace std::literals::chrono_literals;
    std::unique_lock<std::mutex> lock(mutex);
    bool result = cv.wait_for(lock, 1ms, [=]{ return !queue.empty(); });
    if (result) {
      T ret = queue.front();
      queue.pop();
      return ret;
    }
    return c10::nullopt;
  }
  void push(const T &value) {
    {
      std::unique_lock<std::mutex> lock(mutex);
      queue.push(value);
    }
    cv.notify_one();
  }
};
