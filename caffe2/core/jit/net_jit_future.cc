#include "caffe2/core/jit/net_jit_future.h"

#include "c10/util/Logging.h"

namespace caffe2 {

JITFuture::JITFuture() : completed_(false), failed_(false) {}

JITFuture::JITFuture(const std::vector<JITFuture*>& futures)
    : completed_(false), failed_(false) {
  auto parent_counter = std::make_shared<ParentCounter>(futures.size());
  for (auto future : futures) {
    future->SetCallback([this, parent_counter](const JITFuture* f) {
      if (f->IsFailed()) {
        std::unique_lock<std::mutex> lock(parent_counter->err_mutex);
        if (parent_counter->parent_failed) {
          parent_counter->err_msg += ", " + f->ErrorMessage();
        } else {
          parent_counter->parent_failed = true;
          parent_counter->err_msg = f->ErrorMessage();
        }
      }
      int count = --parent_counter->parent_count;
      if (count == 0) {
        // thread safe to use parent_counter here
        if (!parent_counter->parent_failed) {
          this->SetCompleted();
        } else {
          this->SetCompleted(parent_counter->err_msg.c_str());
        }
      }
    });
  }
}

bool JITFuture::IsCompleted() const {
  return completed_;
}

bool JITFuture::IsFailed() const {
  return failed_;
}

std::string JITFuture::ErrorMessage() const {
  return err_msg_;
}

void JITFuture::Wait() const {
  std::unique_lock<std::mutex> lock(mutex_);
  while (!completed_) {
    cv_completed_.wait(lock);
  }
}

void JITFuture::SetCallback(std::function<void(const JITFuture*)> callback) {
  std::unique_lock<std::mutex> lock(mutex_);

  callbacks_.push_back(callback);
  if (completed_) {
    callback(this);
  }
}

void JITFuture::SetCompleted(const char* err_msg) {
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

} // namespace caffe2
