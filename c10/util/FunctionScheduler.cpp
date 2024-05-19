#include <c10/util/FunctionScheduler.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <thread>
#include <vector>

namespace c10 {

Job::Job(std::function<void()> function, std::chrono::microseconds interval)
    : _function(function), _interval(interval) {}

std::chrono::microseconds Job::delta() const {
  return _interval;
}

void Job::run() {
  _function();
}

} // namespace c10
