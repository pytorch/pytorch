#pragma once

#include "../DataChannel.hpp"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <stdexcept>
#include <thread>

namespace thd {

inline void assertTensorEqual(const thpp::Tensor& tensor1,
                              const thpp::Tensor& tensor2,
                              std::string prefix = std::string()) {
  bool equal = tensor1.elementSize() == tensor2.elementSize() &&
               tensor1.numel() == tensor2.numel() &&
               tensor1.type() == tensor2.type();

  if (!prefix.empty())
    prefix = prefix + ": ";

  if (!equal)
    throw std::logic_error(prefix + "tensors are not equal in size or data type");
}

struct QueueWorker {
private:
  struct Task {
    Task(std::function<void ()>&& handler): _handler(handler), _completed(false) {}
    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;

    void run() {
      std::unique_lock<std::mutex> ulock(_mutex);

      try {
        _handler();
      } catch (...) {
        // Do not propagate exception here. We should save it and throw it
        // in `complete` or `wait` function to user.
        _exception = std::current_exception();
      }

      _completed = true;
      ulock.unlock();
      _cond.notify_all();
    }

    bool isCompleted() {
      std::unique_lock<std::mutex> ulock(_mutex);
      _validate();
      return _completed;
    }

    void wait() {
      std::unique_lock<std::mutex> ulock(_mutex);
      if (!_completed)
        _cond.wait(ulock);

      _validate();
    }

  private:
    void _validate() {
      if (_exception)
        std::rethrow_exception(_exception);
    }

    std::function<void ()> _handler;
    std::atomic<bool> _completed;
    std::mutex _mutex;
    std::condition_variable _cond;
    std::exception_ptr _exception;
  };

public:
  struct Request {
    Request(std::shared_ptr<QueueWorker::Task> item) : _item(item) {}

    void wait() { _item->wait(); }
    bool isCompleted() { return _item->isCompleted(); }

  private:
    std::shared_ptr<QueueWorker::Task> _item;
  };

  QueueWorker() : _exiting(false) {
    _main_thread = std::thread(&QueueWorker::_runner, this);
  }

  ~QueueWorker() {
    _exiting = true;
    _cond.notify_one();
    _main_thread.join();
  }

  QueueWorker(const QueueWorker&) = delete;
  QueueWorker& operator=(const QueueWorker&) = delete;

  Request push(std::function<void ()>&& f) {
    auto item = _push(std::make_shared<Task>(std::move(f)));
    return Request(item);
  }

private:
  std::shared_ptr<Task> _pop() {
    std::unique_lock<std::mutex> ulock(_mutex);
    if (_queue.empty())
      _cond.wait(ulock);

    if (_exiting) // check if we were woken up by destructor
      return nullptr;

    auto val = _queue.front();
    _queue.pop();
    return val;
  }

  std::shared_ptr<Task> _push(std::shared_ptr<Task> item) {
    std::unique_lock<std::mutex> ulock(_mutex);
    _queue.push(item);
    ulock.unlock();
    _cond.notify_one();
    return item;
  }


  void _runner() {
    while (true) {
      auto item = _pop();
      if (!item) // empty item -> we need to end (descructor called)
        return;

      item->run();
    }
  }

  std::atomic<bool> _exiting;
  std::queue<std::shared_ptr<Task>> _queue;
  std::mutex _mutex;
  std::condition_variable _cond;

  std::thread _main_thread;
};

} // namespace thd
