#include <torch/torch.h>
#include <chrono>
#include <condition_variable>
#include <mutex>

class Baton {
 public:
  void post() {
    std::unique_lock<std::mutex> l(lock_);
    done_ = true;
    cv_.notify_all();
  }
  void wait() {
    std::unique_lock<std::mutex> l(lock_);
    while (!done_) {
      cv_.wait(l);
    }
  }

 private:
  std::mutex lock_;
  std::condition_variable cv_;
  bool done_{false};
};

void AtLaunch_Base(int32_t numIters) {
  struct Helper {
    explicit Helper(int32_t lim) : limit_(lim) {}
    void operator()() {
      if (++val_ == limit_) {
        done.post();
      } else {
        at::launch([this]() { (*this)(); });
      }
    }
    int val_{0};
    int limit_;
    Baton done;
  };
  Helper h(numIters);
  auto start = std::chrono::system_clock::now();
  h();
  h.done.wait();
  std::cout << "NoData "
            << static_cast<double>(
                   std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::system_clock::now() - start)
                       .count()) /
          static_cast<double>(numIters)
            << " usec/each\n";
}

void AtLaunch_WithData(int32_t numIters, int32_t vecSize) {
  struct Helper {
    explicit Helper(int32_t lim) : limit_(lim) {}
    void operator()(std::vector<int32_t> v) {
      if (++val_ == limit_) {
        done.post();
      } else {
        at::launch([this, v = std::move(v)]() { (*this)(v); });
      }
    }
    int val_{0};
    int limit_;
    Baton done;
  };
  Helper h(numIters);
  std::vector<int32_t> v(vecSize, 0);
  auto start = std::chrono::system_clock::now();
  h(v);
  h.done.wait();
  std::cout << "WithData(" << vecSize << "): "
            << static_cast<double>(
                   std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::system_clock::now() - start)
                       .count()) /
          static_cast<double>(numIters)
            << " usec/each\n";
}

int main(int argc, char** argv) {
  int32_t N = 1000000;
  AtLaunch_Base(N);
  AtLaunch_WithData(N, 0);
  AtLaunch_WithData(N, 4);
  AtLaunch_WithData(N, 256);
  return 0;
}
