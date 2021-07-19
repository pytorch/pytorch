#pragma once

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <torch/types.h>

#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <memory>

namespace torch {
namespace distributed {
namespace rpc {

using JitFuture = c10::ivalue::Future;


class ThreadSafeQueue {
 public:
  void push(at::Tensor&& tensor) {
    {
      std::lock_guard<std::mutex> guard(m);
      q.push(std::move(tensor));
    }

    cv.notify_one();
  }

  at::Tensor pop() {
    std::unique_lock<std::mutex> lock(m);
    cv.wait(lock, [=](){return !q.empty();});
    auto tensor = std::move(q.front());
    q.pop();
    return tensor;
  }

 private:
  std::mutex m;
  std::queue<at::Tensor> q;
  std::condition_variable cv;
};

class TORCH_API ParameterServer final {
 public:
  ParameterServer(int32_t num_trainers, int32_t num_buckets) : num_trainers_(num_trainers) {
    futures_.reserve(num_buckets);
    buckets_.reserve(num_buckets);
    threads_.reserve(num_buckets);
    queues_.reserve(num_buckets);
    pendingTrainers_ = std::vector<int16_t>(num_buckets, num_trainers);
    for (int i = 0; i < num_buckets; ++i) {
      futures_.emplace_back(c10::make_intrusive<JitFuture>(TensorType::get()));
      buckets_.emplace_back(torch::zeros({0}));
      queues_.emplace_back(std::make_shared<ThreadSafeQueue>());
      threads_.emplace_back(std::thread(&ParameterServer::accumulateGradBuckets, this, queues_.back(), i));
    }

    //thread_ = std::thread(&ParameterServer::accumulateGradBuckets, this);
  }

  c10::intrusive_ptr<JitFuture> addGradBucket(at::Tensor bucket, int16_t id) {
    queues_[id]->push(std::move(bucket));
    return futures_[id];
  }

  void accumulateGradBuckets(std::shared_ptr<ThreadSafeQueue> tsq, int16_t id) {
    auto stream = at::cuda::getStreamFromPool();
    at::cuda::CUDAStreamGuard guard(stream);
    while (true) {
      auto bucket = tsq->pop();
      c10::cuda::CUDACachingAllocator::recordStream(
          bucket.storage().data_ptr(), stream);

      if (pendingTrainers_[id] == num_trainers_){
        buckets_[id] = bucket;
      } else {
        buckets_[id] += bucket;
      }

      if (--pendingTrainers_[id] <= 0) {
        futures_[id]->markCompleted(buckets_[id]);
        futures_[id] = c10::make_intrusive<JitFuture>(TensorType::get());
        pendingTrainers_[id] = num_trainers_;
      }
    }

  }

 private:
  // one future per bucket
  std::vector<c10::intrusive_ptr<JitFuture>> futures_;
  int32_t num_trainers_;
  std::vector<std::shared_ptr<ThreadSafeQueue>> queues_;
  std::vector<at::Tensor> buckets_;
  std::vector<int16_t> pendingTrainers_;
  std::vector<std::thread> threads_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
