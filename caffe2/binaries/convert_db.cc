#include "caffe2/core/db.h"
#include "caffe2/core/init.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/core/logging.h"

CAFFE2_DEFINE_string(input_db, "", "The input db.");
CAFFE2_DEFINE_string(input_db_type, "", "The input db type.");
CAFFE2_DEFINE_string(output_db, "", "The output db.");
CAFFE2_DEFINE_string(output_db_type, "", "The output db type.");
CAFFE2_DEFINE_int(batch_size, 1000, "The write batch size.");

using caffe2::db::Cursor;
using caffe2::db::DB;
using caffe2::db::Transaction;

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);

  std::unique_ptr<DB> in_db(caffe2::db::CreateDB(
      caffe2::FLAGS_input_db_type, caffe2::FLAGS_input_db, caffe2::db::READ));
  std::unique_ptr<DB> out_db(caffe2::db::CreateDB(
      caffe2::FLAGS_output_db_type, caffe2::FLAGS_output_db, caffe2::db::NEW));
  std::unique_ptr<Cursor> cursor(in_db->NewCursor());
  std::unique_ptr<Transaction> transaction(out_db->NewTransaction());
  int count = 0;
  for (; cursor->Valid(); cursor->Next()) {
    transaction->Put(cursor->key(), cursor->value());
    if (++count % caffe2::FLAGS_batch_size == 0) {
      transaction->Commit();
      LOG(INFO) << "Converted " << count << " items so far.";
    }
  }
  LOG(INFO) << "A total of " << count << " items processed.";
  return 0;
}
