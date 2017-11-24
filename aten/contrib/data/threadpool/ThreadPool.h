#ifndef AT_THREADPOOL_H
#define AT_THREADPOOL_H

// dependencies:
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>

// definition of the ThreadPool class:
class ThreadPool {
   public:
      explicit ThreadPool(uint64_t);
      void synchronize();
      unsigned int waitFor();
      template<class F, class... Args>
      unsigned int enqueue(F&& f, Args&&... args);
      ~ThreadPool();

   private:

      // all the threads that can perform work:
      std::vector< std::thread > workers;

      // the list of futures:
      unsigned int handle = 0;
      std::vector< std::pair< std::future<void>, unsigned int > > futures;

      // the task queue:
      std::queue< std::function<void()> > tasks;

      // synchronization:
      std::mutex queue_mutex;
      std::condition_variable condition;
      bool stop;
};

// enqueue new work item into the pool:
template<class F, class... Args>
unsigned int ThreadPool::enqueue(F&& f, Args&&... args)
{

   // create the task:
   auto task = std::make_shared< std::packaged_task<void()> >(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...)
   );

   // get future and enqueue the task:
   std::future<void> future = task->get_future();
   {
      std::unique_lock<std::mutex> lock(queue_mutex);
      if(stop) throw std::runtime_error("enqueue on stopped ThreadPool");
      tasks.emplace([task](){ (*task)(); });
   }
   condition.notify_one();

   // generate handle and store future:
   handle++;
   futures.push_back(std::make_pair(std::move(future), handle));
   return handle;
}

#endif
