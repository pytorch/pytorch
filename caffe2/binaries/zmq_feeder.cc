// This binary provides an easy way to open a zeromq server and feeds data to
// clients connect to it. It uses the Caffe2 db as the backend, thus allowing
// one to convert any db-compliant storage to a zeromq service.

#include <atomic>

#include "caffe2/core/db.h"
#include "caffe2/core/init.h"
#include "caffe2/utils/zmq.hpp"
#include "glog/logging.h"

DEFINE_string(server, "tcp://*:5555", "The server address.");
DEFINE_string(input_db, "", "The input db.");
DEFINE_string(input_db_type, "", "The input db type.");

using caffe2::db::DB;
using caffe2::db::Cursor;
using caffe2::string;

std::unique_ptr<DB> in_db;
std::unique_ptr<Cursor> cursor;

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  gflags::SetUsageMessage("Runs a given plan.");

  LOG(INFO) << "Opening DB...";
  in_db.reset(caffe2::db::CreateDB(
      FLAGS_input_db_type, FLAGS_input_db, caffe2::db::READ));
  CHECK(in_db.get() != nullptr) << "Cannot load input db.";
  cursor.reset(in_db->NewCursor());
  LOG(INFO) << "DB opened.";

  LOG(INFO) << "Starting ZeroMQ server...";

  //  Socket to talk to clients
  zmq::context_t context(1);
  zmq::socket_t sender(context, ZMQ_PUSH);
  try {
    sender.bind(FLAGS_server);
    LOG(INFO) << "Server created at " << FLAGS_server;
  } catch (const zmq::error_t& ze) {
    LOG(FATAL) << "ZeroMQ error: " << ze.num() << " " << ze.what();
  }

  while (1) {
    VLOG(1) << "Sending " << cursor->key();

    string key = cursor->key();
    zmq::message_t key_msg(key.size());
    memcpy(key_msg.data(), key.data(), key.size());
    string value = cursor->value();
    zmq::message_t value_msg(value.size());
    memcpy(value_msg.data(), value.data(), value.size());
    while (!sender.send(key_msg, ZMQ_SNDMORE)) {
      VLOG(1) << "Trying re-sending key...";
    }
    while(!sender.send(value_msg)) {
      VLOG(1) << "Trying re-sending...";
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
