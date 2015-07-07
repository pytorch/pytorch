#include "caffe2/core/db.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe/proto/caffe.pb.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

DEFINE_string(input_db, "", "The input db.");
DEFINE_string(input_db_type, "", "The input db type.");
DEFINE_string(output_db, "", "The output db.");
DEFINE_string(output_db_type, "", "The output db type.");
DEFINE_int32(batch_size, 1000, "The write batch size.");

using caffe2::db::Cursor;
using caffe2::db::DB;
using caffe2::db::Transaction;
using caffe2::TensorProto;
using caffe2::TensorProtos;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::SetUsageMessage(
      "This script converts databases between different formats.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::unique_ptr<DB> in_db(caffe2::db::CreateDB(
      FLAGS_input_db_type, FLAGS_input_db, caffe2::db::READ));
  std::unique_ptr<DB> out_db(caffe2::db::CreateDB(
      FLAGS_output_db_type, FLAGS_output_db, caffe2::db::NEW));
  std::unique_ptr<Cursor> cursor(in_db->NewCursor());
  std::unique_ptr<Transaction> transaction(out_db->NewTransaction());
  int count = 0;
  for (; cursor->Valid(); cursor->Next()) {
    caffe::Datum datum;
    CHECK(datum.ParseFromString(cursor->value()));
    TensorProtos protos;
    TensorProto* data = protos.add_protos();
    TensorProto* label = protos.add_protos();
    label->set_data_type(TensorProto::INT32);
    label->add_dims(1);
    label->add_int32_data(datum.label());
    if (datum.encoded()) {
      // This is an encoded image. we will copy over the data directly.
      data->set_data_type(TensorProto::STRING);
      data->add_dims(1);
      data->add_string_data(datum.data());
    } else {
      CHECK_EQ(datum.float_data_size(), 0);  // float data not supported right now.
      char buffer[datum.data().size()];
      // swap order from CHW to HWC
      int channels = datum.channels();
      int size = datum.height() * datum.width();
      CHECK_EQ(datum.data().size(), channels * size);
      for (int c = 0; c < channels; ++c) {
        char* dst = buffer + c;
        const char* src = datum.data().c_str() + c * size;
        for (int n = 0; n < size; ++n) {
          dst[n*channels] = src[n];
        }
      }
      data->set_data_type(TensorProto::BYTE);
      data->add_dims(datum.height());
      data->add_dims(datum.width());
      data->add_dims(datum.channels());
      data->set_byte_data(buffer, datum.data().size());
    }
    transaction->Put(cursor->key(), protos.SerializeAsString());
    if (++count % FLAGS_batch_size == 0) {
      transaction->Commit();
      LOG(INFO) << "Converted " << count << " items so far.";
    }
  }
  LOG(INFO) << "A total of " << count << " items processed.";
  return 0;
}

