#include "caffe2/core/blob.h"
#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/common.h"
#include "caffe2/mkl/mkl_utils.h"

#include <gtest/gtest.h>

#ifdef CAFFE2_HAS_MKL_DNN

namespace caffe2 {

using mkl::MKLMemory;

TEST(MKLTest, MKLMemorySerialization) {
  Blob blob;
  vector<int> shape{2, 3, 4};
  float data[2 * 3 * 4];
  for (int i = 0; i < 2 * 3 * 4; ++i) {
    data[i] = i;
  }
  blob.Reset<MKLMemory<float>>(new MKLMemory<float>(shape));
  MKLMemory<float>* mkl_memory = blob.GetMutable<MKLMemory<float>>();
  mkl_memory->CopyFrom(data);
  string serialized = blob.Serialize("test");
  BlobProto proto;
  CHECK(proto.ParseFromString(serialized));
  EXPECT_EQ(proto.name(), "test");
  EXPECT_EQ(proto.type(), "Tensor");
  EXPECT_TRUE(proto.has_tensor());
  const TensorProto& tensor_proto = proto.tensor();
  EXPECT_EQ(
      tensor_proto.data_type(), TypeMetaToDataType(TypeMeta::Make<float>()));
  EXPECT_EQ(tensor_proto.float_data_size(), 2 * 3 * 4);
  for (int i = 0; i < 2 * 3 * 4; ++i) {
    EXPECT_EQ(tensor_proto.float_data(i), static_cast<float>(i));
  }
  Blob new_blob;
  EXPECT_NO_THROW(new_blob.Deserialize(serialized));
  EXPECT_TRUE(new_blob.IsType<MKLMemory<float>>());
  const auto& new_mkl_memory = blob.Get<MKLMemory<float>>();
  EXPECT_EQ(new_mkl_memory.dims().size(), 3);
  EXPECT_EQ(new_mkl_memory.dims()[0], 2);
  EXPECT_EQ(new_mkl_memory.dims()[1], 3);
  EXPECT_EQ(new_mkl_memory.dims()[2], 4);
  float recovered_data[2 * 3 * 4];
  new_mkl_memory.CopyTo(recovered_data);
  for (int i = 0; i < 2 * 3 * 4; ++i) {
    EXPECT_EQ(recovered_data[i], i);
  }
}

} // namespace caffe2

#endif // CAFFE2_HAS_MKL_DNN
