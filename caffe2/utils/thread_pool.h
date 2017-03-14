#ifndef CAFFE2_UTILS_THREAD_POOL_H_
#define CAFFE2_UTILS_THREAD_POOL_H_

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

class TaskThreadPool{
 private:
    std::queue< std::function< void() > > tasks_;
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
                std::bind(&TaskThreadPool::main_loop, this));
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
        tasks_.push(std::function<void()>(task));
        complete_ = false;
        condition_.notify_one();
    }

    /// @brief Wait for queue to be empty
    void waitWorkComplete() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!complete_)
            completed_.wait(lock);
    }

 private:
    /// @brief Entry point for pool threads.
    void main_loop() {
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
                std::function< void() > task = tasks_.front();
                tasks_.pop();
                // Decrement count, indicating thread is no longer available.
                --available_;

                lock.unlock();

                // Run the task.
                try {
                    task();
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

#endif
