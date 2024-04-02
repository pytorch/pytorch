#ifndef CAFFE2_UTILS_ZMQ_HELPER_H_
#define CAFFE2_UTILS_ZMQ_HELPER_H_

#include <zmq.h>

#include "caffe2/core/logging.h"

namespace caffe2 {

class ZmqContext {
 public:
  explicit ZmqContext(int io_threads) : ptr_(zmq_ctx_new()) {
    CAFFE_ENFORCE(ptr_ != nullptr, "Failed to create zmq context.");
    int rc = zmq_ctx_set(ptr_, ZMQ_IO_THREADS, io_threads);
    CAFFE_ENFORCE_EQ(rc, 0);
    rc = zmq_ctx_set(ptr_, ZMQ_MAX_SOCKETS, ZMQ_MAX_SOCKETS_DFLT);
    CAFFE_ENFORCE_EQ(rc, 0);
  }
  ~ZmqContext() {
    int rc = zmq_ctx_destroy(ptr_);
    CAFFE_ENFORCE_EQ(rc, 0);
  }

  void* ptr() { return ptr_; }

 private:
  void* ptr_;

  C10_DISABLE_COPY_AND_ASSIGN(ZmqContext);
};

class ZmqMessage {
 public:
  ZmqMessage() {
    int rc = zmq_msg_init(&msg_);
    CAFFE_ENFORCE_EQ(rc, 0);
  }

  ~ZmqMessage() {
    int rc = zmq_msg_close(&msg_);
    CAFFE_ENFORCE_EQ(rc, 0);
  }

  zmq_msg_t* msg() { return &msg_; }

  void* data() { return zmq_msg_data(&msg_); }
  size_t size() { return zmq_msg_size(&msg_); }

 private:
  zmq_msg_t msg_;
  C10_DISABLE_COPY_AND_ASSIGN(ZmqMessage);
};

class ZmqSocket {
 public:
  explicit ZmqSocket(int type)
      : context_(1), ptr_(zmq_socket(context_.ptr(), type)) {
    CAFFE_ENFORCE(ptr_ != nullptr, "Failed to create zmq socket.");
  }

  ~ZmqSocket() {
    int rc = zmq_close(ptr_);
    CAFFE_ENFORCE_EQ(rc, 0);
  }

  void Bind(const string& addr) {
    int rc = zmq_bind(ptr_, addr.c_str());
    CAFFE_ENFORCE_EQ(rc, 0);
  }

  void Unbind(const string& addr) {
    int rc = zmq_unbind(ptr_, addr.c_str());
    CAFFE_ENFORCE_EQ(rc, 0);
  }

  void Connect(const string& addr) {
    int rc = zmq_connect(ptr_, addr.c_str());
    CAFFE_ENFORCE_EQ(rc, 0);
  }

  void Disconnect(const string& addr) {
    int rc = zmq_disconnect(ptr_, addr.c_str());
    CAFFE_ENFORCE_EQ(rc, 0);
  }

  int Send(const string& msg, int flags) {
    int nbytes = zmq_send(ptr_, msg.c_str(), msg.size(), flags);
    if (nbytes) {
      return nbytes;
    } else if (zmq_errno() == EAGAIN) {
      return 0;
    } else {
      LOG(FATAL) << "Cannot send zmq message. Error number: "
                      << zmq_errno();
      return 0;
    }
  }

  int SendTillSuccess(const string& msg, int flags) {
    CAFFE_ENFORCE(msg.size(), "You cannot send an empty message.");
    int nbytes = 0;
    do {
      nbytes = Send(msg, flags);
    } while (nbytes == 0);
    return nbytes;
  }

  int Recv(ZmqMessage* msg) {
    int nbytes = zmq_msg_recv(msg->msg(), ptr_, 0);
    if (nbytes >= 0) {
      return nbytes;
    } else if (zmq_errno() == EAGAIN || zmq_errno() == EINTR) {
      return 0;
    } else {
      LOG(FATAL) << "Cannot receive zmq message. Error number: "
                      << zmq_errno();
      return 0;
    }
  }

  int RecvTillSuccess(ZmqMessage* msg) {
    int nbytes = 0;
    do {
      nbytes = Recv(msg);
    } while (nbytes == 0);
    return nbytes;
  }

 private:
  ZmqContext context_;
  void* ptr_;
};

}  // namespace caffe2


#endif  // CAFFE2_UTILS_ZMQ_HELPER_H_
