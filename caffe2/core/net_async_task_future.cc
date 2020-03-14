#include "caffe2/core/net_async_task_future.h"

#include "c10/util/Logging.h"
#include "caffe2/core/common.h"

namespace caffe2 {

AsyncTaskFuture::AsyncTaskFuture() : completed_(false), failed_(false) {}

AsyncTaskFuture::AsyncTaskFuture(const std::vector<AsyncTaskFuture*>& futures)
    : completed_(false), failed_(false) {
  if (futures.size() > 1) {
    parent_counter_ = std::make_unique<ParentCounter>(futures.size());
    for (auto future : futures) {
      future->SetCallback([this](const AsyncTaskFuture* f) {
        if (f->IsFailed()) {
          std::unique_lock<std::mutex> lock(parent_counter_->err_mutex);
          if (parent_counter_->parent_failed) {
            parent_counter_->err_msg += ", " + f->ErrorMessage();
          } else {
            parent_counter_->parent_failed = true;
            parent_counter_->err_msg = f->ErrorMessage();
          }
        }
        int count = --parent_counter_->parent_count;
        if (count == 0) {
          // thread safe to use parent_counter here
          if (!parent_counter_->parent_failed) {
            SetCompleted();
          } else {
            SetCompleted(parent_counter_->err_msg.c_str());
          }
        }
      });
    }
  } else {
    CAFFE_ENFORCE_EQ(futures.size(), (size_t)1);
    auto future = futures.back();
    future->SetCallback([this](const AsyncTaskFuture* f) {
      if (!f->IsFailed()) {
        SetCompleted();
      } else {
        SetCompleted(f->ErrorMessage().c_str());
      }
    });
  }
}

bool AsyncTaskFuture::IsCompleted() const {
  return completed_;
}

bool AsyncTaskFuture::IsFailed() const {
  return failed_;
}

std::string AsyncTaskFuture::ErrorMessage() const {
  return err_msg_;
}

void AsyncTaskFuture::Wait() const {
  std::unique_lock<std::mutex> lock(mutex_);
  while (!completed_) {
    cv_completed_.wait(lock);
  }
}

void AsyncTaskFuture::SetCallback(
    std::function<void(const AsyncTaskFuture*)> callback) {
  std::unique_lock<std::mutex> lock(mutex_);

  callbacks_.push_back(callback);
  if (completed_) {
    callback(this);
  }
}

void AsyncTaskFuture::SetCompleted(const char* err_msg) {
  std::unique_lock<std::mutex> lock(mutex_);

  CAFFE_ENFORCE(!completed_, "Calling SetCompleted on a completed future");
  completed_ = true;

  if (err_msg) {
    failed_ = true;
    err_msg_ = err_msg;
  }

  for (auto& callback : callbacks_) {
    callback(this);
  }

  cv_completed_.notify_all();
}

// ResetState is called on a completed future,
// does not reset callbacks to keep task graph structure
void AsyncTaskFuture::ResetState() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (parent_counter_) {
    parent_counter_->Reset();
  }
  completed_ = false;
  failed_ = false;
  err_msg_ = "";
}

AsyncTaskFuture::~AsyncTaskFuture() {}

} // namespace caffe2
