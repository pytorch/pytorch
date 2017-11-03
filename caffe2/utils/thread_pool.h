/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CAFFE2_UTILS_THREAD_POOL_H_
#define CAFFE2_UTILS_THREAD_POOL_H_

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>

namespace caffe2 {

class TaskThreadPool {
 private:
    struct task_element_t {
        bool run_with_id;
        const std::function< void() > no_id;
        const std::function< void(std::size_t) > with_id;

        explicit task_element_t(const std::function< void() >& f) :
            run_with_id(false), no_id(f), with_id(nullptr) { }
        explicit task_element_t(const std::function< void(std::size_t) >& f) :
            run_with_id(true), no_id(nullptr), with_id(f) { }
    };
    std::queue<task_element_t> tasks_;
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::condition_variable completed_;
    bool running_;
    bool complete_;
    std::size_t available_;
    std::size_t total_;

 public:
    /// @brief Constructor.
    explicit TaskThreadPool(std::size_t pool_size)
        :  threads_(pool_size), running_(true), complete_(true),
           available_(pool_size), total_(pool_size) {
        for ( std::size_t i = 0; i < pool_size; ++i ) {
            threads_[i] = std::thread(
                std::bind(&TaskThreadPool::main_loop, this, i));
        }
    }

    /// @brief Destructor.
    ~TaskThreadPool() {
        // Set running flag to false then notify all threads.
        {
            std::unique_lock< std::mutex > lock(mutex_);
            running_ = false;
            condition_.notify_all();
        }

        try {
            for (auto& t : threads_) {
              t.join();
            }
        }
        // Suppress all exceptions.
        catch (const std::exception&) {}
    }

    /// @brief Add task to the thread pool if a thread is currently available.
    template <typename Task>
    void runTask(Task task) {
        std::unique_lock<std::mutex> lock(mutex_);

        // Set task and signal condition variable so that a worker thread will
        // wake up and use the task.
        tasks_.push(task_element_t(static_cast<std::function< void() >>(task)));
        complete_ = false;
        condition_.notify_one();
    }

    void run(const std::function<void()>& func) {
      runTask(func);
    }

    template <typename Task>
    void runTaskWithID(Task task) {
      std::unique_lock<std::mutex> lock(mutex_);

      // Set task and signal condition variable so that a worker thread will
      // wake up and use the task.
      tasks_.push(task_element_t(static_cast<std::function< void(std::size_t) >>(
                                   task)));
      complete_ = false;
      condition_.notify_one();
    }

    /// @brief Wait for queue to be empty
    void waitWorkComplete() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!complete_)
          completed_.wait(lock);
    }

 private:
    /// @brief Entry point for pool threads.
    void main_loop(std::size_t index) {
        while (running_) {
            // Wait on condition variable while the task is empty and
            // the pool is still running.
            std::unique_lock<std::mutex> lock(mutex_);
            while (tasks_.empty() && running_) {
                condition_.wait(lock);
            }
            // If pool is no longer running, break out of loop.
            if (!running_) break;

            // Copy task locally and remove from the queue.  This is
            // done within its own scope so that the task object is
            // destructed immediately after running the task.  This is
            // useful in the event that the function contains
            // shared_ptr arguments bound via bind.
            {
                auto tasks = tasks_.front();
                tasks_.pop();
                // Decrement count, indicating thread is no longer available.
                --available_;

                lock.unlock();

                // Run the task.
                try {
                  if (tasks.run_with_id) {
                      tasks.with_id(index);
                  } else {
                      tasks.no_id();
                  }
                }
                // Suppress all exceptions.
                catch ( const std::exception& ) {}

                // Update status of empty, maybe
                // Need to recover the lock first
                lock.lock();

                // Increment count, indicating thread is available.
                ++available_;
                if (tasks_.empty() && available_ == total_) {
                    complete_ = true;
                    completed_.notify_one();
                }
            }
        }  // while running_
    }
};

} // namespace caffe2

#endif
