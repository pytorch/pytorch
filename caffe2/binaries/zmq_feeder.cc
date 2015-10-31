// This binary provides an easy way to open a zeromq server and feeds data to
// clients connect to it. It uses the Caffe2 db as the backend, thus allowing
// one to convert any db-compliant storage to a zeromq service.

#include <atomic>

#include "caffe2/core/db.h"
#include "caffe2/core/init.h"
#include "caffe2/utils/zmq.hpp"
#include "caffe2/core/logging.h"

CAFFE2_DEFINE_string(server, "tcp://*:5555", "The server address.");
CAFFE2_DEFINE_string(input_db, "", "The input db.");
CAFFE2_DEFINE_string(input_db_type, "", "The input db type.");

using caffe2::db::DB;
using caffe2::db::Cursor;
using caffe2::string;

std::unique_ptr<DB> in_db;
std::unique_ptr<Cursor> cursor;

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, argv);

  CAFFE_LOG_INFO << "Opening DB...";
  in_db.reset(caffe2::db::CreateDB(
      caffe2::FLAGS_input_db_type, caffe2::FLAGS_input_db, caffe2::db::READ));
  CAFFE_CHECK(in_db.get() != nullptr) << "Cannot load input db.";
  cursor.reset(in_db->NewCursor());
  CAFFE_LOG_INFO << "DB opened.";

  CAFFE_LOG_INFO << "Starting ZeroMQ server...";

  //  Socket to talk to clients
  zmq::context_t context(1);
  zmq::socket_t sender(context, ZMQ_PUSH);
  try {
    sender.bind(caffe2::FLAGS_server);
    CAFFE_LOG_INFO << "Server created at " << caffe2::FLAGS_server;
  } catch (const zmq::error_t& ze) {
    CAFFE_LOG_FATAL << "ZeroMQ error: " << ze.num() << " " << ze.what();
  }

  while (1) {
    CAFFE_VLOG(1) << "Sending " << cursor->key();

    string key = cursor->key();
    zmq::message_t key_msg(key.size());
    memcpy(key_msg.data(), key.data(), key.size());
    string value = cursor->value();
    zmq::message_t value_msg(value.size());
    memcpy(value_msg.data(), value.data(), value.size());
    while (!sender.send(key_msg, ZMQ_SNDMORE)) {
      CAFFE_VLOG(1) << "Trying re-sending key...";
    }
    while (!sender.send(value_msg)) {
      CAFFE_VLOG(1) << "Trying re-sending...";
    }
    cursor->Next();
    if (!cursor->Valid()) {
      cursor->SeekToFirst();
    }
  }
  // We do not do an elegant quit since this binary is going to be terminated by
  // control+C.
  return 0;
}
