#include <errno.h>

#include <cstdint>

#include "caffe2/core/db.h"
#include "caffe2/utils/zmq.hpp"
#include "caffe2/core/logging.h"

namespace caffe2 {
namespace db {

class ZmqDBCursor : public Cursor {
 public:
  explicit ZmqDBCursor(const string& source)
      : source_(source), context_(1), socket_(context_, ZMQ_PULL), key_("") {
    socket_.connect(source);
    // obtain the first value.
    Next();
  }

  ~ZmqDBCursor() {
    socket_.disconnect(source_);
  }
  void SeekToFirst() override { /* do nothing */ }

  inline void ReceiveWithRetry(zmq::message_t* content) {
    bool retry = true;
    while (retry) {
      try {
        socket_.recv(content);
        retry = false;
      } catch(const zmq::error_t& ze) {
        // CAFFE_LOG_ERROR << "Exception: " << ze.num() << " " << ze.what();
        if (ze.num() != EINTR && ze.num() != EAGAIN) {
          CAFFE_LOG_FATAL << "ZeroMQ received error that cannot continue. Quitting.";
        }
      }
    }
  }

  void Next() override {
    zmq::message_t content;
    ReceiveWithRetry(&content);
    key_.assign(static_cast<char*>(content.data()), content.size());
    ReceiveWithRetry(&content);
    value_.assign(static_cast<char*>(content.data()), content.size());
  }

  string key() override { return key_; }
  string value() override { return value_; }
  virtual bool Valid() { return true; }

 private:
  string source_;
  zmq::context_t context_;
  zmq::socket_t socket_;
  string key_;
  string value_;
};

class ZmqDB : public DB {
 public:
  ZmqDB(const string& source, Mode mode)
      : DB(source, mode), source_(source) {
    CAFFE_CHECK_EQ(mode, READ) << "ZeroMQ DB only supports read mode.";
  }

  ~ZmqDB() {}

  void Close() override {}

  Cursor* NewCursor() override {
    return new ZmqDBCursor(source_);
  }

  Transaction* NewTransaction() override {
    // TODO(Yangqing): Do I really need to do log fatal? Any elegant way to
    // warn the user?
    CAFFE_LOG_FATAL << "ZeroMQ DB does not support writing with a transaction.";
    return nullptr;  // dummy placeholder to suppress old compiler warnings.
  }

 private:
  string source_;
};

REGISTER_CAFFE2_DB(ZmqDB, ZmqDB);
// For lazy-minded, one can also call with lower-case name.
REGISTER_CAFFE2_DB(zmqdb, ZmqDB);

}  // namespace db
}  // namespace caffe2
