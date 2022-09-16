#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread> // NOLINT

#include "caffe2/core/db.h"
#include "caffe2/core/logging.h"
#include "caffe2/utils/zmq_helper.h"

namespace caffe2 {
namespace db {

class ZmqDBCursor : public Cursor {
 public:
  explicit ZmqDBCursor(const string& source)
      : source_(source),
        socket_(ZMQ_PULL),
        prefetched_(false),
        finalize_(false) {
    socket_.Connect(source_);
    // Start prefetching thread.
    prefetch_thread_.reset(new std::thread([this] { this->Prefetch(); }));
    // obtain the first value.
    Next();
  }

  ~ZmqDBCursor() override {
    finalize_ = true;
    prefetched_ = false;
    producer_.notify_one();
    // Wait for the prefetch thread to finish elegantly.
    prefetch_thread_->join();
    socket_.Disconnect(source_);
  }

  void Seek(const string& /*key*/) override { /* do nothing */
  }

  void SeekToFirst() override { /* do nothing */
  }

  void Next() override {
    std::unique_lock<std::mutex> lock(prefetch_access_mutex_);
    while (!prefetched_)
      consumer_.wait(lock);
    key_ = prefetch_key_;
    value_ = prefetch_value_;
    prefetched_ = false;
    producer_.notify_one();
  }

  string key() override {
    return key_;
  }
  string value() override {
    return value_;
  }
  bool Valid() override {
    return true;
  }

 private:
  void Prefetch() {
    while (!finalize_) {
      std::unique_lock<std::mutex> lock(prefetch_access_mutex_);
      while (prefetched_)
        producer_.wait(lock);
      if (finalize_) {
        return;
      }
      ZmqMessage msg;
      socket_.RecvTillSuccess(&msg);
      prefetch_key_.assign(static_cast<char*>(msg.data()), msg.size());
      socket_.RecvTillSuccess(&msg);
      prefetch_value_.assign(static_cast<char*>(msg.data()), msg.size());
      prefetched_ = true;
      consumer_.notify_one();
    }
  }

  string source_;
  ZmqSocket socket_;
  string key_;
  string value_;
  string prefetch_key_;
  string prefetch_value_;

  unique_ptr<std::thread> prefetch_thread_;
  std::mutex prefetch_access_mutex_;
  std::condition_variable producer_, consumer_;
  std::atomic<bool> prefetched_;
  // finalize_ is used to tell the prefetcher to quit.
  std::atomic<bool> finalize_;
};

class ZmqDB : public DB {
 public:
  ZmqDB(const string& source, Mode mode) : DB(source, mode), source_(source) {
    CAFFE_ENFORCE(mode == READ, "ZeroMQ DB only supports read mode.");
  }

  ~ZmqDB() override {}

  void Close() override {}

  unique_ptr<Cursor> NewCursor() override {
    return make_unique<ZmqDBCursor>(source_);
  }

  unique_ptr<Transaction> NewTransaction() override {
    CAFFE_THROW("ZeroMQ DB does not support writing with a transaction.");
    return nullptr; // dummy placeholder to suppress old compiler warnings.
  }

 private:
  string source_;
};

REGISTER_CAFFE2_DB(ZmqDB, ZmqDB);
// For lazy-minded, one can also call with lower-case name.
REGISTER_CAFFE2_DB(zmqdb, ZmqDB);

} // namespace db
} // namespace caffe2
