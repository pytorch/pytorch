#ifndef COMPUTATION_CLIENT_ASYNC_TASK_H_
#define COMPUTATION_CLIENT_ASYNC_TASK_H_

#include <c10/util/Optional.h>

#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>

#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/thread_pool.h"

namespace lazy_tensors {
namespace util {

template <typename T>
class AsyncTask {
  struct Data {
    Data(std::function<T()> taskfn) : taskfn(std::move(taskfn)) {}

    std::function<T()> taskfn;
    std::mutex mutex;
    std::condition_variable cv;
    bool scheduled = false;
    bool completed = false;
    c10::optional<T> result;
    std::exception_ptr exptr;
  };

 public:
  explicit AsyncTask(std::function<T()> taskfn)
      : data_(std::make_shared<Data>(std::move(taskfn))) {}

  AsyncTask& Wait() {
    std::unique_lock<std::mutex> lock(data_->mutex);
    LTC_CHECK(data_->scheduled);
    data_->cv.wait(lock, [this] { return data_->completed; });
    if (data_->exptr != nullptr) {
      std::rethrow_exception(data_->exptr);
    }
    return *this;
  }

  AsyncTask& Schedule() {
    auto completer = [data = data_]() {
      c10::optional<T> result;
      std::exception_ptr exptr;
      try {
        result = data->taskfn();
      } catch (...) {
        exptr = std::current_exception();
      }

      std::lock_guard<std::mutex> lock(data->mutex);
      if (result) {
        data->result = std::move(*result);
      } else {
        data->exptr = std::move(exptr);
      }
      data->completed = true;
      data->cv.notify_all();
    };

    {
      std::lock_guard<std::mutex> lock(data_->mutex);
      LTC_CHECK(!data_->scheduled);
      data_->scheduled = true;
    }
    lazy_tensors::env::ScheduleIoClosure(std::move(completer));
    return *this;
  }

  const T& GetValue() const {
    std::lock_guard<std::mutex> lock(data_->mutex);
    return *data_->result;
  }

  T ConsumeValue() {
    std::lock_guard<std::mutex> lock(data_->mutex);
    return std::move(*data_->result);
  }

 private:
  std::shared_ptr<Data> data_;
};

}  // namespace util
}  // namespace lazy_tensors

#endif  // COMPUTATION_CLIENT_ASYNC_TASK_H_
