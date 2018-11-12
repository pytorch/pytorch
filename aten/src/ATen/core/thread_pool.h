#pragma once

#include <functional>
#include <queue>

#include <ATen/core/intrusive_ptr.h>

namespace c10 {

namespace ivalue {
struct Future;
} // namespace ivalue

struct C10_API ThreadPool final {
  ThreadPool();

  ~ThreadPool();

  void schedule(std::function<void(void)> task);

  void workOnTasksUntilCompleted(c10::intrusive_ptr<ivalue::Future> future);

 private:
  std::queue<std::function<void(void)>> tasks;
};

C10_API extern ThreadPool global_work_queue;

} // namespace c10
