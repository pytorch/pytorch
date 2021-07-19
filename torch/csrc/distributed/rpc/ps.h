#pragma once

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <torch/types.h>

#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace torch {
namespace distributed {
namespace rpc {

using JitFuture = c10::ivalue::Future;

struct GradBucket {
    GradBucket(at::Tensor bucket, int16_t id)
        : bucket(std::move(bucket)), id(id) {}
    at::Tensor bucket;
    int16_t id;
};

class ThreadSafeQueue {
 public:
  void push(GradBucket&& bucket) {
    {
      std::lock_guard<std::mutex> guard(m);
      q.push(std::move(bucket));
    }

    cv.notify_one();
  }

  GradBucket pop() {
    std::unique_lock<std::mutex> lock(m);
    cv.wait(lock, [=](){return !q.empty();});
    auto gradBucket = std::move(q.front());
    q.pop();
    return gradBucket;
  }

 private:
  std::mutex m;
  std::queue<GradBucket> q;
  std::condition_variable cv;
};

class TORCH_API ParameterServer final {
 public:
  ParameterServer(int32_t num_trainers, int32_t num_buckets) : num_trainers_(num_trainers) {
    futures_.reserve(num_buckets);
    buckets_.reserve(num_buckets);
    pendingTrainers_ = std::vector<int16_t>(num_buckets, num_trainers);
    for (int i = 0; i < num_buckets; ++i) {
      futures_.emplace_back(c10::make_intrusive<JitFuture>(TensorType::get()));
      buckets_.emplace_back(torch::zeros({0}));
    }

    thread_ = std::thread(&ParameterServer::accumulateGradBuckets, this);
  }

  c10::intrusive_ptr<JitFuture> addGradBucket(at::Tensor bucket, int16_t id) {
    tsq_.push(GradBucket(std::move(bucket), id));
    return futures_[id];
  }

  void accumulateGradBuckets() {
    auto stream = at::cuda::getStreamFromPool();
    at::cuda::CUDAStreamGuard guard(stream);
    while (true) {
      auto gradBucket = tsq_.pop();
      auto id = gradBucket.id;
      auto& bucket = gradBucket.bucket;
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
  ThreadSafeQueue tsq_;
  std::vector<at::Tensor> buckets_;
  std::vector<int16_t> pendingTrainers_;
  std::thread thread_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
