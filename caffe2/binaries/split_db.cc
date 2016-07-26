#include <string>
#include <sstream>

#include "caffe2/core/db.h"
#include "caffe2/core/init.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/core/logging.h"

CAFFE2_DEFINE_string(input_db, "", "The input db.");
CAFFE2_DEFINE_int(splits, 0, "The number of splits.");
CAFFE2_DEFINE_string(db_type, "", "The db type.");
CAFFE2_DEFINE_int(batch_size, 1000, "The write batch size.");

using caffe2::db::Cursor;
using caffe2::db::DB;
using caffe2::db::Transaction;

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);

  std::unique_ptr<DB> in_db(caffe2::db::CreateDB(
      caffe2::FLAGS_db_type, caffe2::FLAGS_input_db, caffe2::db::READ));
  std::unique_ptr<Cursor> cursor(in_db->NewCursor());

  CHECK_GT(caffe2::FLAGS_splits, 0) << "Must specify the number of splits.";
  std::vector<std::unique_ptr<DB> > out_dbs;
  std::vector<std::unique_ptr<Transaction> > transactions;
  for (int i = 0; i < caffe2::FLAGS_splits; ++i) {
    out_dbs.push_back(
        std::unique_ptr<DB>(caffe2::db::CreateDB(
            caffe2::FLAGS_db_type,
            caffe2::FLAGS_input_db + "_split_" + caffe2::to_string(i),
            caffe2::db::NEW)));
    transactions.push_back(
        std::unique_ptr<Transaction>(out_dbs[i]->NewTransaction()));
  }

  int count = 0;
  for (; cursor->Valid(); cursor->Next()) {
    transactions[count % caffe2::FLAGS_splits]->Put(cursor->key(), cursor->value());
    if (++count % caffe2::FLAGS_batch_size == 0) {
      for (int i = 0; i < caffe2::FLAGS_splits; ++i) {
        transactions[i]->Commit();
      }
      LOG(INFO) << "Split " << count << " items so far.";
    }
  }
  LOG(INFO) << "A total of " << count << " items processed.";
  return 0;
}
