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

#include "caffe2/core/db.h"
#include "caffe2/core/init.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/core/logging.h"

C10_DEFINE_string(input_db, "", "The input db.");
C10_DEFINE_string(input_db_type, "", "The input db type.");
C10_DEFINE_string(output_db, "", "The output db.");
C10_DEFINE_string(output_db_type, "", "The output db type.");
C10_DEFINE_int(batch_size, 1000, "The write batch size.");

using caffe2::db::Cursor;
using caffe2::db::DB;
using caffe2::db::Transaction;

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);

  std::unique_ptr<DB> in_db(caffe2::db::CreateDB(
      FLAGS_input_db_type, FLAGS_input_db, caffe2::db::READ));
  std::unique_ptr<DB> out_db(caffe2::db::CreateDB(
      FLAGS_output_db_type, FLAGS_output_db, caffe2::db::NEW));
  std::unique_ptr<Cursor> cursor(in_db->NewCursor());
  std::unique_ptr<Transaction> transaction(out_db->NewTransaction());
  int count = 0;
  for (; cursor->Valid(); cursor->Next()) {
    transaction->Put(cursor->key(), cursor->value());
    if (++count % FLAGS_batch_size == 0) {
      transaction->Commit();
      LOG(INFO) << "Converted " << count << " items so far.";
    }
  }
  LOG(INFO) << "A total of " << count << " items processed.";
  return 0;
}
