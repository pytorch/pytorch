#include <queue>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <c10/util/Optional.h>

// BlockingQueue is a thread-safe implementation of queue.
//
// BlockingQueue::pop() returns a c10::optional of the object.
// If there is no object in the queue, then pop will block for
// 1ms or until there is an element pushed to the queue. If pop
// returns becaue of getting an object, then that object will be
// returned. If pop returns because of timeout, then c10::nullopt
// will be returned.
//
// BlockingQueue::push() pushes an object to the queue.

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
