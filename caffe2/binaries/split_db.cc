#include <string>
#include <sstream>

#include "caffe2/core/db.h"
#include "caffe2/core/init.h"
#include "caffe2/proto/caffe2.pb.h"
#include "glog/logging.h"

DEFINE_string(input_db, "", "The input db.");
DEFINE_int32(splits, 0, "The number of splits.");
DEFINE_string(db_type, "", "The db type.");
DEFINE_int32(batch_size, 1000, "The write batch size.");

using caffe2::db::Cursor;
using caffe2::db::DB;
using caffe2::db::Transaction;

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  gflags::SetUsageMessage(
      "This script converts databases between different formats.");

  std::unique_ptr<DB> in_db(caffe2::db::CreateDB(
      FLAGS_db_type, FLAGS_input_db, caffe2::db::READ));
  std::unique_ptr<Cursor> cursor(in_db->NewCursor());

  CHECK_GT(FLAGS_splits, 0) << "Must specify the number of splits.";
  std::vector<std::unique_ptr<DB> > out_dbs;
  std::vector<std::unique_ptr<Transaction> > transactions;
  for (int i = 0; i < FLAGS_splits; ++i) {
    out_dbs.push_back(
        std::unique_ptr<DB>(caffe2::db::CreateDB(
            FLAGS_db_type, FLAGS_input_db + "_split_" + std::to_string(i),
            caffe2::db::NEW)));
    transactions.push_back(
        std::unique_ptr<Transaction>(out_dbs[i]->NewTransaction()));
  }

  int count = 0;
  for (; cursor->Valid(); cursor->Next()) {
    transactions[count % FLAGS_splits]->Put(cursor->key(), cursor->value());
    if (++count % FLAGS_batch_size == 0) {
      for (int i = 0; i < FLAGS_splits; ++i) {
        transactions[i]->Commit();
      }
      LOG(INFO) << "Splitted " << count << " items so far.";
    }
  }
  LOG(INFO) << "A total of " << count << " items processed.";
  return 0;
}
