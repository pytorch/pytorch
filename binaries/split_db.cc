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

#include <string>
#include <sstream>

#include "caffe2/core/db.h"
#include "caffe2/core/init.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/core/logging.h"

C10_DEFINE_string(input_db, "", "The input db.");
C10_DEFINE_int(splits, 0, "The number of splits.");
C10_DEFINE_string(db_type, "", "The db type.");
C10_DEFINE_int(batch_size, 1000, "The write batch size.");

namespace caffe2 {

static int Split(int argc, char** argv) {
  GlobalInit(&argc, &argv);

  CAFFE_ENFORCE(FLAGS_input_db.size(), "Must specify --input_db=/path/to/db.");
  CAFFE_ENFORCE(FLAGS_splits > 0, "Must specify a nonnegative split number.");
  CAFFE_ENFORCE(FLAGS_db_type.size(), "Must specify --db_type=[a db type].");

  unique_ptr<db::DB> in_db(
      db::CreateDB(FLAGS_db_type, FLAGS_input_db, db::READ));
  CAFFE_ENFORCE(in_db != nullptr, "Cannot open input db: ", FLAGS_input_db);
  unique_ptr<db::Cursor> cursor(in_db->NewCursor());
  // This usually won't happen, but FWIW.
  CAFFE_ENFORCE(
      cursor != nullptr, "Cannot obtain cursor for input db: ", FLAGS_input_db);

  vector<unique_ptr<db::DB>> out_dbs;
  vector<unique_ptr<db::Transaction>> transactions;
  for (int i = 0; i < FLAGS_splits; ++i) {
    out_dbs.push_back(unique_ptr<db::DB>(db::CreateDB(
        FLAGS_db_type, FLAGS_input_db + "_split_" + to_string(i), db::NEW)));
    CAFFE_ENFORCE(out_dbs.back().get(), "Cannot create output db #", i);
    transactions.push_back(
        unique_ptr<db::Transaction>(out_dbs[i]->NewTransaction()));
    CAFFE_ENFORCE(
        transactions.back().get(), "Cannot get transaction for output db #", i);
  }

  int count = 0;
  for (; cursor->Valid(); cursor->Next()) {
    transactions[count % FLAGS_splits]->Put(cursor->key(), cursor->value());
    if (++count % FLAGS_batch_size == 0) {
      for (int i = 0; i < FLAGS_splits; ++i) {
        transactions[i]->Commit();
      }
      LOG(INFO) << "Split " << count << " items so far.";
    }
  }
  LOG(INFO) << "A total of " << count << " items processed.";
  return 0;
}

} // namespace caffe2

int main(int argc, char** argv) {
  return caffe2::Split(argc, argv);
}
