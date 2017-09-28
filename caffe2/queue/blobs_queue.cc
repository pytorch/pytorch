/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "caffe2/queue/blobs_queue.h"

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

// Constants for user tracepoints
static constexpr int SDT_NONBLOCKING_OP = 0;
static constexpr int SDT_BLOCKING_OP = 1;
static constexpr uint64_t SDT_TIMEOUT = (uint64_t)-1;
static constexpr uint64_t SDT_ABORT = (uint64_t)-2;
static constexpr uint64_t SDT_CANCEL = (uint64_t)-3;

BlobsQueue::BlobsQueue(
    Workspace* ws,
    const std::string& queueName,
    size_t capacity,
    size_t numBlobs,
    bool enforceUniqueName,
    const std::vector<std::string>& fieldNames)
    : numBlobs_(numBlobs), name_(queueName), stats_(queueName) {
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
      const auto blobName = queueName + "_" + to_string(i) + "_" + to_string(j);
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

bool BlobsQueue::blockingRead(
    const std::vector<Blob*>& inputs,
    float timeout_secs) {
  auto keeper = this->shared_from_this();
  const auto& name = name_.c_str();
  CAFFE_SDT(queue_read_start, name, (void*)this, SDT_BLOCKING_OP);
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
      CAFFE_SDT(queue_read_end, name, (void*)this, SDT_TIMEOUT);
    } else {
      CAFFE_SDT(queue_read_end, name, (void*)this, SDT_CANCEL);
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
  CAFFE_SDT(queue_read_end, name, (void*)this, writer_ - reader_);
  CAFFE_EVENT(stats_, queue_dequeued_records);
  ++reader_;
  cv_.notify_all();
  return true;
}

bool BlobsQueue::tryWrite(const std::vector<Blob*>& inputs) {
  auto keeper = this->shared_from_this();
  const auto& name = name_.c_str();
  CAFFE_SDT(queue_write_start, name, (void*)this, SDT_NONBLOCKING_OP);
  std::unique_lock<std::mutex> g(mutex_);
  if (!canWrite()) {
    CAFFE_SDT(queue_write_end, name, (void*)this, SDT_ABORT);
    return false;
  }
  CAFFE_EVENT(stats_, queue_balance, 1);
  DCHECK(canWrite());
  doWrite(inputs);
  return true;
}

bool BlobsQueue::blockingWrite(const std::vector<Blob*>& inputs) {
  auto keeper = this->shared_from_this();
  const auto& name = name_.c_str();
  CAFFE_SDT(queue_write_start, name, (void*)this, SDT_BLOCKING_OP);
  std::unique_lock<std::mutex> g(mutex_);
  CAFFE_EVENT(stats_, queue_balance, 1);
  cv_.wait(g, [this]() { return closing_ || canWrite(); });
  if (!canWrite()) {
    CAFFE_SDT(queue_write_end, name, (void*)this, SDT_ABORT);
    return false;
  }
  DCHECK(canWrite());
  doWrite(inputs);
  return true;
}

void BlobsQueue::close() {
  closing_ = true;

  std::lock_guard<std::mutex> g(mutex_);
  cv_.notify_all();
}

bool BlobsQueue::canWrite() {
  // writer is always within [reader, reader + size)
  // we can write if reader is within [reader, reader + size)
  CAFFE_ENFORCE_LE(reader_, writer_);
  CAFFE_ENFORCE_LE(writer_, reader_ + queue_.size());
  return writer_ != reader_ + queue_.size();
}

void BlobsQueue::doWrite(const std::vector<Blob*>& inputs) {
  auto& result = queue_[writer_ % queue_.size()];
  CAFFE_ENFORCE(inputs.size() >= result.size());
  const auto& name = name_.c_str();
  for (auto i = 0; i < result.size(); ++i) {
    using std::swap;
    swap(*(inputs[i]), *(result[i]));
  }
  CAFFE_SDT(
      queue_write_end, name, (void*)this, reader_ + queue_.size() - writer_);
  ++writer_;
  cv_.notify_all();
}

} // namespace caffe2
