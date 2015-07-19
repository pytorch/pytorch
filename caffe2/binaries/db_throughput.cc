#include <ctime>
#include <cstdio>

#include "caffe2/core/db.h"
#include "caffe2/proto/caffe2.pb.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "google/profiler.h"

DEFINE_string(input_db, "", "The input db.");
DEFINE_string(input_db_type, "", "The input db type.");
DEFINE_string(profile_file, "db_throughput_profile", "The profile output.");
DEFINE_int32(report_interval, 1000, "The report interval.");
DEFINE_int32(repeat, 10, "The number to repeat the throughput test.");

using caffe2::db::Cursor;
using caffe2::db::DB;
using caffe2::string;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::SetUsageMessage(
      "This script reports the throughput .");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::unique_ptr<DB> in_db(caffe2::db::CreateDB(
      FLAGS_input_db_type, FLAGS_input_db, caffe2::db::READ));
  std::unique_ptr<Cursor> cursor(in_db->NewCursor());

  ProfilerStart(FLAGS_profile_file.c_str());
  for (int iter_id = 0; iter_id < FLAGS_repeat; ++iter_id) {
    clock_t start = clock();
    for (int i = 0; i < FLAGS_report_interval; ++i) {
      string key = cursor->key();
      string value = cursor->value();
      VLOG(1) << "Key " << key;
      cursor->Next();
      if (!cursor->Valid()) {
        cursor->SeekToFirst();
      }
    }
    clock_t elapsed = clock() - start;
    double elapsed_seconds = static_cast<double>(elapsed) / CLOCKS_PER_SEC;
    printf("Iteration %03d, took %4.5f seconds, throughput %f items/sec.\n",
           iter_id, elapsed_seconds, FLAGS_report_interval / elapsed_seconds);
  }
  ProfilerStop();
  return 0;
}
