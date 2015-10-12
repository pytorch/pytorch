#include <ctime>
#include <cstdio>

#include "caffe2/core/db.h"
#include "caffe2/core/init.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/core/logging.h"

CAFFE2_DEFINE_string(input_db, "", "The input db.");
CAFFE2_DEFINE_string(input_db_type, "", "The input db type.");
CAFFE2_DEFINE_int(report_interval, 1000, "The report interval.");
CAFFE2_DEFINE_int(repeat, 10, "The number to repeat the throughput test.");

using caffe2::db::Cursor;
using caffe2::db::DB;
using caffe2::string;

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, argv);

  std::unique_ptr<DB> in_db(caffe2::db::CreateDB(
      caffe2::FLAGS_input_db_type, caffe2::FLAGS_input_db, caffe2::db::READ));
  std::unique_ptr<Cursor> cursor(in_db->NewCursor());

  for (int iter_id = 0; iter_id < caffe2::FLAGS_repeat; ++iter_id) {
    clock_t start = clock();
    for (int i = 0; i < caffe2::FLAGS_report_interval; ++i) {
      string key = cursor->key();
      string value = cursor->value();
      CAFFE_VLOG(1) << "Key " << key;
      cursor->Next();
      if (!cursor->Valid()) {
        cursor->SeekToFirst();
      }
    }
    clock_t elapsed = clock() - start;
    double elapsed_seconds = static_cast<double>(elapsed) / CLOCKS_PER_SEC;
    printf("Iteration %03d, took %4.5f seconds, throughput %f items/sec.\n",
           iter_id, elapsed_seconds,
           caffe2::FLAGS_report_interval / elapsed_seconds);
  }
  return 0;
}
