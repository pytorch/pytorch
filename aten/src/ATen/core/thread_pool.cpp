#include <ATen/core/thread_pool.h>
#include <ATen/core/ivalue.h>

namespace c10 {

ThreadPool::ThreadPool() { /* intended to leave empty */ }

ThreadPool::~ThreadPool() { /* intended to leave empty */ }

void ThreadPool::schedule(std::function<void(void)> task) {
  tasks.push(task);
}

void ThreadPool::workOnTasksUntilCompleted(
    c10::intrusive_ptr<ivalue::Future> future) {
  while (!future->completed()) {
    auto task = tasks.front();
    tasks.pop();
    task();
  }
}

ThreadPool global_work_queue;

} // namespace c10
