#include <errno.h>

#include <cstdint>

#include "caffe2/core/db.h"
#include "caffe2/utils/zmq_helper.h"
#include "caffe2/core/logging.h"

namespace caffe2 {
namespace db {

class ZmqDBCursor : public Cursor {
 public:
  explicit ZmqDBCursor(const string& source)
      : source_(source), socket_(ZMQ_PULL) {
    socket_.Connect(source_);
    // obtain the first value.
    Next();
  }

  ~ZmqDBCursor() {
    socket_.Disconnect(source_);
  }
  void SeekToFirst() override { /* do nothing */ }

  void Next() override {
    ZmqMessage msg;
    socket_.RecvTillSuccess(&msg);
    key_.assign(static_cast<char*>(msg.data()), msg.size());
    socket_.RecvTillSuccess(&msg);
    value_.assign(static_cast<char*>(msg.data()), msg.size());
  }

  string key() override { return key_; }
  string value() override { return value_; }
  bool Valid() override { return true; }

 private:
  string source_;
  ZmqSocket socket_;
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
