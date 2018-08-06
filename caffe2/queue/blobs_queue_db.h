
#pragma once

#include <chrono>
#include <string>

#include "caffe2/core/db.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/stats.h"
#include "caffe2/queue/blobs_queue.h"

namespace caffe2 {
namespace db {

namespace {
const std::string& GetStringFromBlob(Blob* blob) {
  if (blob->template IsType<string>()) {
    return blob->template Get<string>();
  } else if (blob->template IsType<Tensor>()) {
    return *blob->template Get<Tensor>().template data<string>();
  } else {
    CAFFE_THROW("Unsupported Blob type");
  }
}
}

class BlobsQueueDBCursor : public Cursor {
 public:
  explicit BlobsQueueDBCursor(
      std::shared_ptr<BlobsQueue> queue,
      int key_blob_index,
      int value_blob_index,
      float timeout_secs)
      : queue_(queue),
        key_blob_index_(key_blob_index),
        value_blob_index_(value_blob_index),
        timeout_secs_(timeout_secs),
        inited_(false),
        valid_(false) {
    LOG(INFO) << "BlobsQueueDBCursor constructed";
    CAFFE_ENFORCE(queue_ != nullptr, "queue is null");
    CAFFE_ENFORCE(value_blob_index_ >= 0, "value_blob_index < 0");
  }

  virtual ~BlobsQueueDBCursor() {}

  void Seek(const string& /* unused */) override {
    CAFFE_THROW("Seek is not supported.");
  }

  bool SupportsSeek() override {
    return false;
  }

  void SeekToFirst() override {
    // not applicable
  }

  void Next() override {
    unique_ptr<Blob> blob = make_unique<Blob>();
    vector<Blob*> blob_vector{blob.get()};
    auto success = queue_->blockingRead(blob_vector, timeout_secs_);
    if (!success) {
      LOG(ERROR) << "Timed out reading from BlobsQueue or it is closed";
      valid_ = false;
      return;
    }

    if (key_blob_index_ >= 0) {
      key_ = GetStringFromBlob(blob_vector[key_blob_index_]);
    }
    value_ = GetStringFromBlob(blob_vector[value_blob_index_]);
    valid_ = true;
  }

  string key() override {
    if (!inited_) {
      Next();
      inited_ = true;
    }
    return key_;
  }

  string value() override {
    if (!inited_) {
      Next();
      inited_ = true;
    }
    return value_;
  }

  bool Valid() override {
    return valid_;
  }

 private:
  std::shared_ptr<BlobsQueue> queue_;
  int key_blob_index_;
  int value_blob_index_;
  float timeout_secs_;
  bool inited_;
  string key_;
  string value_;
  bool valid_;
};

class BlobsQueueDB : public DB {
 public:
  BlobsQueueDB(
      const string& source,
      Mode mode,
      std::shared_ptr<BlobsQueue> queue,
      int key_blob_index = -1,
      int value_blob_index = 0,
      float timeout_secs = 0.0)
      : DB(source, mode),
        queue_(queue),
        key_blob_index_(key_blob_index),
        value_blob_index_(value_blob_index),
        timeout_secs_(timeout_secs) {
    LOG(INFO) << "BlobsQueueDB constructed";
  }

  virtual ~BlobsQueueDB() {
    Close();
  }

  void Close() override {}
  unique_ptr<Cursor> NewCursor() override {
    return make_unique<BlobsQueueDBCursor>(
        queue_, key_blob_index_, value_blob_index_, timeout_secs_);
  }

  unique_ptr<Transaction> NewTransaction() override {
    CAFFE_THROW("Not implemented.");
  }

 private:
  std::shared_ptr<BlobsQueue> queue_;
  int key_blob_index_;
  int value_blob_index_;
  float timeout_secs_;
};
} // namespace db
} // namespace caffe2
