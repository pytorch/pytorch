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
#include "caffe2/proto/caffe2_legacy.pb.h"
#include "caffe2/core/logging.h"

C10_DEFINE_string(input_db, "", "The input db.");
C10_DEFINE_string(input_db_type, "", "The input db type.");
C10_DEFINE_string(output_db, "", "The output db.");
C10_DEFINE_string(output_db_type, "", "The output db type.");
C10_DEFINE_int(batch_size, 1000, "The write batch size.");

using caffe2::db::Cursor;
using caffe2::db::DB;
using caffe2::db::Transaction;
using caffe2::CaffeDatum;
using caffe2::TensorProto;
using caffe2::TensorProtos;

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
    CaffeDatum datum;
    CAFFE_ENFORCE(datum.ParseFromString(cursor->value()));
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
      // float data not supported right now.
      CAFFE_ENFORCE_EQ(datum.float_data_size(), 0);
      std::vector<char> buffer_vec(datum.data().size());
      char* buffer = buffer_vec.data();
      // swap order from CHW to HWC
      int channels = datum.channels();
      int size = datum.height() * datum.width();
      CAFFE_ENFORCE_EQ(datum.data().size(), channels * size);
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
    transaction->Put(cursor->key(), SerializeAsString_EnforceCheck(protos));
    if (++count % FLAGS_batch_size == 0) {
      transaction->Commit();
      LOG(INFO) << "Converted " << count << " items so far.";
    }
  }
  LOG(INFO) << "A total of " << count << " items processed.";
  return 0;
}
