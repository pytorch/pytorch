// dependencies:
#include "ThreadPool.h"
#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <stdlib.h>
#include <stdarg.h>

// constructor launches the specified number of workers:
ThreadPool::ThreadPool(uint64_t threads) : stop(false) {

   // loop over all threads:
   for(uint64_t i = 0; i < threads; ++i) {
      workers.emplace_back(
         [this] {
            for(;;) {
               std::function<void()> task;
               {
                  std::unique_lock<std::mutex> lock(this->queue_mutex);
                  this->condition.wait(lock,
                     [this]{ return this->stop || !this->tasks.empty(); });
                  if(this->stop && this->tasks.empty())
                     return;
                  task = std::move(this->tasks.front());
                  this->tasks.pop();
               }
               task();
            }
         }
      );
   }
}

// synchronize all the threads:
void ThreadPool::synchronize() {
   for(uint64_t i = 0; i < futures.size(); i++)
      futures[i].first.wait();
}

// poll the threads for results:
unsigned int ThreadPool::waitFor() {

   // wait until a task is finished:
   uint64_t i;
   std::future_status status;
   do {
      for(i = 0; i < futures.size(); i++) {
         status = futures[i].first.wait_for(std::chrono::microseconds(0));
         if(status == std::future_status::ready) break;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(1));
   } while (status != std::future_status::ready);

   // get the result and remove the future:
   futures[i].first.get();
   unsigned int handle = futures[i].second;
   iter_swap(futures.begin() + i, futures.end() - 1);
   futures.pop_back();
   return handle;
}

// the destructor joins all threads:
ThreadPool::~ThreadPool() {
   {
      std::unique_lock<std::mutex> lock(queue_mutex);
      stop = true;
   }
   condition.notify_all();
   for(std::thread &worker: workers)
      worker.join();
}
