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

// This binary provides an easy way to open a zeromq server and feeds data to
// clients connect to it. It uses the Caffe2 db as the backend, thus allowing
// one to convert any db-compliant storage to a zeromq service.

#include "caffe2/core/db.h"
#include "caffe2/core/init.h"
#include "caffe2/core/logging.h"
#include "caffe2/utils/zmq_helper.h"

CAFFE2_DEFINE_string(server, "tcp://*:5555", "The server address.");
CAFFE2_DEFINE_string(input_db, "", "The input db.");
CAFFE2_DEFINE_string(input_db_type, "", "The input db type.");

using caffe2::db::DB;
using caffe2::db::Cursor;
using caffe2::string;

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);

  LOG(INFO) << "Opening DB...";
  auto in_db = caffe2::db::CreateDB(
      caffe2::FLAGS_input_db_type, caffe2::FLAGS_input_db, caffe2::db::READ);
  CAFFE_ENFORCE(
      in_db,
      "Cannot load input db " + caffe2::FLAGS_input_db + " of expected type " +
          caffe2::FLAGS_input_db_type);
  auto cursor = in_db->NewCursor();
  LOG(INFO) << "DB opened.";

  LOG(INFO) << "Starting ZeroMQ server...";

  //  Socket to talk to clients
  caffe2::ZmqSocket sender(ZMQ_PUSH);
  sender.Bind(caffe2::FLAGS_server);
  LOG(INFO) << "Server created at " << caffe2::FLAGS_server;

  while (1) {
    VLOG(1) << "Sending " << cursor->key();
    sender.SendTillSuccess(cursor->key(), ZMQ_SNDMORE);
    sender.SendTillSuccess(cursor->value(), 0);
    cursor->Next();
    if (!cursor->Valid()) {
      cursor->SeekToFirst();
    }
  }
  // We do not do an elegant quit since this binary is going to be terminated by
  // control+C.
  return 0;
}
