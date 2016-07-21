#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>

#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/workspace.h"

namespace caffe2 {

// A thread-safe, bounded, blocking queue.
// Modelled as a circular buffer.

// Containing blobs are owned by the workspace.
// On read, we swap out the underlying data for the blob passed in for blobs

class BlobsQueue : public std::enable_shared_from_this<BlobsQueue> {
 public:
  BlobsQueue(
      Workspace* ws,
      const std::string& queueName,
      size_t capacity,
      size_t numBlobs,
      bool enforceUniqueName) {
    queue_.reserve(capacity);
    for (auto i = 0; i < capacity; ++i) {
      std::vector<Blob*> blobs;
      blobs.reserve(numBlobs);
      for (auto j = 0; j < numBlobs; ++j) {
        const auto blobName =
            queueName + "_" + std::to_string(i) + "_" + std::to_string(j);
        if (enforceUniqueName) {
          CHECK(!ws->GetBlob(blobName))
              << "Queue internal blob already exists: " << blobName;
        }
        blobs.push_back(ws->CreateBlob(blobName));
      }
      queue_.push_back(blobs);
    }
    DCHECK_EQ(queue_.size(), capacity);
  }

  ~BlobsQueue() {
    close();
  }

  bool blockingRead(const std::vector<Blob*>& inputs) {
    auto keeper = this->shared_from_this();
    std::unique_lock<std::mutex> g(mutex_);
    auto canRead = [this]() {
      CHECK_LE(reader_, writer_);
      return reader_ != writer_;
    };
    cv_.wait(g, [this, canRead]() { return closing_ || canRead(); });
    if (!canRead()) {
      return false;
    }
    DCHECK(canRead());
    auto& result = queue_[reader_ % queue_.size()];
    CHECK_EQ(inputs.size(), result.size());
    for (auto i = 0; i < inputs.size(); ++i) {
      using std::swap;
      swap(*(inputs[i]), *(result[i]));
    }
    ++reader_;
    cv_.notify_all();
    return true;
  }

  bool blockingWrite(const std::vector<Blob*>& inputs) {
    auto keeper = this->shared_from_this();
    std::unique_lock<std::mutex> g(mutex_);
    auto canWrite = [this]() {
      // writer is always within [reader, reader + size)
      // we can write if reader is within [reader, reader + size)
      CHECK_LE(reader_, writer_);
      CHECK_LE(writer_, reader_ + queue_.size());
      return writer_ != reader_ + queue_.size();
    };
    cv_.wait(g, [this, canWrite]() { return closing_ || canWrite(); });
    if (!canWrite()) {
      return false;
    }
    DCHECK(canWrite());
    auto& result = queue_[writer_ % queue_.size()];
    CHECK_EQ(inputs.size(), result.size());
    for (auto i = 0; i < inputs.size(); ++i) {
      using std::swap;
      swap(*(inputs[i]), *(result[i]));
    }
    ++writer_;
    cv_.notify_all();
    return true;
  }

  void close() {
    closing_ = true;

    std::lock_guard<std::mutex> g(mutex_);
    cv_.notify_all();
  }

 private:
  std::atomic<bool> closing_{false};

  std::mutex mutex_; // protects all variables in the class.
  std::condition_variable cv_;
  int64_t reader_{0};
  int64_t writer_{0};
  std::vector<std::vector<Blob*>> queue_;

};
}
