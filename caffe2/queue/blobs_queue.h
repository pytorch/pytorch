#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>

#include "caffe2/core/blob_stats.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/stats.h"
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
      bool enforceUniqueName,
      const std::vector<std::string>& fieldNames = {})
      : numBlobs_(numBlobs), stats_(queueName) {
    if (!fieldNames.empty()) {
      CAFFE_ENFORCE_EQ(
          fieldNames.size(), numBlobs, "Wrong number of fieldNames provided.");
      stats_.queue_dequeued_bytes.setDetails(fieldNames);
    }
    queue_.reserve(capacity);
    for (auto i = 0; i < capacity; ++i) {
      std::vector<Blob*> blobs;
      blobs.reserve(numBlobs);
      for (auto j = 0; j < numBlobs; ++j) {
        const auto blobName =
            queueName + "_" + to_string(i) + "_" + to_string(j);
        if (enforceUniqueName) {
          CAFFE_ENFORCE(
              !ws->GetBlob(blobName),
              "Queue internal blob already exists: ",
              blobName);
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

  bool blockingRead(
      const std::vector<Blob*>& inputs,
      float timeout_secs = 0.0f) {
    auto keeper = this->shared_from_this();
    std::unique_lock<std::mutex> g(mutex_);
    auto canRead = [this]() {
      CAFFE_ENFORCE_LE(reader_, writer_);
      return reader_ != writer_;
    };
    CAFFE_EVENT(stats_, queue_balance, -1);
    if (timeout_secs > 0) {
      std::chrono::milliseconds timeout_ms(int(timeout_secs * 1000));
      cv_.wait_for(
          g, timeout_ms, [this, canRead]() { return closing_ || canRead(); });
    } else {
      cv_.wait(g, [this, canRead]() { return closing_ || canRead(); });
    }
    if (!canRead()) {
      if (timeout_secs > 0 && !closing_) {
        LOG(ERROR) << "DequeueBlobs timed out in " << timeout_secs << " secs";
      }
      return false;
    }
    DCHECK(canRead());
    auto& result = queue_[reader_ % queue_.size()];
    CAFFE_ENFORCE(inputs.size() >= result.size());
    for (auto i = 0; i < result.size(); ++i) {
      auto bytes = BlobStat::sizeBytes(*result[i]);
      CAFFE_EVENT(stats_, queue_dequeued_bytes, bytes, i);
      using std::swap;
      swap(*(inputs[i]), *(result[i]));
    }
    CAFFE_EVENT(stats_, queue_dequeued_records);
    ++reader_;
    cv_.notify_all();
    return true;
  }

  bool tryWrite(const std::vector<Blob*>& inputs) {
    auto keeper = this->shared_from_this();
    std::unique_lock<std::mutex> g(mutex_);
    if (!canWrite()) {
      return false;
    }
    CAFFE_EVENT(stats_, queue_balance, 1);
    DCHECK(canWrite());
    doWrite(inputs);
    return true;
  }

  bool blockingWrite(const std::vector<Blob*>& inputs) {
    auto keeper = this->shared_from_this();
    std::unique_lock<std::mutex> g(mutex_);
    CAFFE_EVENT(stats_, queue_balance, 1);
    cv_.wait(g, [this]() { return closing_ || canWrite(); });
    if (!canWrite()) {
      return false;
    }
    DCHECK(canWrite());
    doWrite(inputs);
    return true;
  }

  void close() {
    closing_ = true;

    std::lock_guard<std::mutex> g(mutex_);
    cv_.notify_all();
  }

  size_t getNumBlobs() const {
    return numBlobs_;
  }

 private:
  bool canWrite() {
    // writer is always within [reader, reader + size)
    // we can write if reader is within [reader, reader + size)
    CAFFE_ENFORCE_LE(reader_, writer_);
    CAFFE_ENFORCE_LE(writer_, reader_ + queue_.size());
    return writer_ != reader_ + queue_.size();
  }

  void doWrite(const std::vector<Blob*>& inputs) {
    auto& result = queue_[writer_ % queue_.size()];
    CAFFE_ENFORCE(inputs.size() >= result.size());
    for (auto i = 0; i < result.size(); ++i) {
      using std::swap;
      swap(*(inputs[i]), *(result[i]));
    }
    ++writer_;
    cv_.notify_all();
  }

  std::atomic<bool> closing_{false};

  size_t numBlobs_;
  std::mutex mutex_; // protects all variables in the class.
  std::condition_variable cv_;
  int64_t reader_{0};
  int64_t writer_{0};
  std::vector<std::vector<Blob*>> queue_;

  struct QueueStats {
    CAFFE_STAT_CTOR(QueueStats);
    CAFFE_EXPORTED_STAT(queue_balance);
    CAFFE_EXPORTED_STAT(queue_dequeued_records);
    CAFFE_DETAILED_EXPORTED_STAT(queue_dequeued_bytes);
  } stats_;
};
}
