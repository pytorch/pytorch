#include <errno.h>

#include <cstdint>

#include "caffe2/core/db.h"
#include "glog/logging.h"
#include "zmq.h"

#if ZMQ_VERSION_MAJOR < 3
#error "ZmqDB requires ZMQ version 3 or above."
#endif

namespace caffe2 {
namespace db {

typedef char ZmqCommand;
typedef int ZmqMessageSize;
const ZmqCommand kQueryMessageSize = 's';
const ZmqCommand kGet = 'g';

class ZmqDBCursor : public Cursor {
 public:
  explicit ZmqDBCursor(void* requester)
      : requester_(requester), buffer_(nullptr), received_size_(0),
        buffer_size_(0) {
    // Figure out the buffer size.
    CHECK_EQ(
        zmq_send(requester_, &kQueryMessageSize, sizeof(ZmqCommand), 0),
        sizeof(ZmqCommand))
        << "Incorrect zmq communication when querying message size.";
    CHECK_EQ(
        zmq_recv(requester_, &buffer_size_, sizeof(ZmqMessageSize), 0),
        sizeof(ZmqMessageSize))
        << "Incorrect zmq communication when fetching message size.";
    CHECK_GT(buffer_size_, 0) << "Incorrect buffer size obtained.";
    buffer_.reset(new char[buffer_size_]);
    // obtain the first value.
    Next();
  }

  ~ZmqDBCursor() {}
  void SeekToFirst() override { /* do nothing */ }
  void Next() override {
    CHECK_EQ(
        zmq_send(requester_, &kGet, sizeof(ZmqCommand), 0), sizeof(ZmqCommand))
        << "Incorrect zmq communication when sending request.";
    received_size_ = zmq_recv(requester_, buffer_.get(), buffer_size_, 0);
    CHECK_GT(received_size_, 0) << "Received no message.";
  }
  string key() override { return ""; }
  string value() override {
    return string(buffer_.get(), received_size_);
  }
  virtual bool Valid() { return true; }

 private:
  void* requester_;
  unique_ptr<char[]> buffer_;
  int received_size_;
  ZmqMessageSize buffer_size_;
};


class ZmqDB : public DB {
 public:
  ZmqDB(const string& source, Mode mode)
      : DB(source, mode), context_(zmq_ctx_new()),
        requester_(zmq_socket(context_, ZMQ_REQ)) {
    CHECK_EQ(mode, READ) << "ZeroMQ DB only supports read mode.";
    VLOG(1) << "Connecting to ZeroMQ server: " << source;
    int ret = zmq_connect(requester_, source.c_str());
    CHECK_EQ(ret, 0) << "Error in connecting to zmq server. "
                     << "Error is: " << errno;
    VLOG(1) << "Opened ZeroMQ server: " << source;
  }

  ~ZmqDB() { Close(); }

  void Close() override {
    if (!requester_) {
      zmq_close(requester_);
      requester_ = nullptr;
      zmq_ctx_destroy(context_);
      context_ = nullptr;
    }
  }

  Cursor* NewCursor() override {
    return new ZmqDBCursor(requester_);
  }
  Transaction* NewTransaction() override {
    // TODO(Yangqing): Do I really need to just do log fatal?
    LOG(FATAL) << "ZeroMQ DB does not support writing with a transaction.";
    return nullptr;  // dummy placeholder to suppress old compiler warnings.
  }

 private:
  void* context_;
  void* requester_;
};

REGISTER_CAFFE2_DB(ZmqDB, ZmqDB);
// For lazy-minded, one can also call with lower-case name.
REGISTER_CAFFE2_DB(zmqdb, ZmqDB);

}  // namespace db
}  // namespace caffe2
