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

class CAFFE2_API BlobsQueue : public std::enable_shared_from_this<BlobsQueue> {
 public:
  BlobsQueue(
      Workspace* ws,
      const std::string& queueName,
      size_t capacity,
      size_t numBlobs,
      bool enforceUniqueName,
      const std::vector<std::string>& fieldNames = {});

  ~BlobsQueue() {
    close();
  }

  bool blockingRead(
      const std::vector<Blob*>& inputs,
      float timeout_secs = 0.0f);
  bool tryWrite(const std::vector<Blob*>& inputs);
  bool blockingWrite(const std::vector<Blob*>& inputs);
  void close();
  size_t getNumBlobs() const {
    return numBlobs_;
  }

 private:
  bool canWrite();
  void doWrite(const std::vector<Blob*>& inputs);

  std::atomic<bool> closing_{false};

  size_t numBlobs_;
  std::mutex mutex_; // protects all variables in the class.
  std::condition_variable cv_;
  int64_t reader_{0};
  int64_t writer_{0};
  std::vector<std::vector<Blob*>> queue_;
  const std::string name_;

  struct QueueStats {
    CAFFE_STAT_CTOR(QueueStats);
    CAFFE_EXPORTED_STAT(queue_balance);
    CAFFE_EXPORTED_STAT(queue_dequeued_records);
    CAFFE_DETAILED_EXPORTED_STAT(queue_dequeued_bytes);
    CAFFE_AVG_EXPORTED_STAT(read_time_ns);
    CAFFE_AVG_EXPORTED_STAT(write_time_ns);
  } stats_;
};
} // namespace caffe2
