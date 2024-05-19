#include <c10/util/FunctionScheduler.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <thread>
#include <vector>

namespace c10 {

Job::Job(std::function<void()> function, std::chrono::microseconds interval)
    : _function(function), _interval(interval) {}

std::chrono::microseconds Job::interval() const {
  return _interval;
}

void Job::run() {
  _function();
}

FunctionScheduler::FunctionScheduler() : _running(false) {}

FunctionScheduler::~FunctionScheduler() {
  stop();
}

void FunctionScheduler::runNextJob() {}

int FunctionScheduler::scheduleJob(
    std::function<void()> function,
    std::chrono::microseconds interval) {}

int FunctionScheduler::removeJob(int id) {}

void FunctionScheduler::start() {}

void FunctionScheduler::stop() {}

} // namespace c10
