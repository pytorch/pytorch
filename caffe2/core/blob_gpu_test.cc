#include <iostream>  // NOLINT

#include "caffe2/core/blob.h"
#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/proto/caffe2.pb.h"
#include <gtest/gtest.h>

namespace caffe2 {
namespace {

template <typename T> class TensorGPUTest : public ::testing::Test {};
template <typename T> class TensorGPUDeathTest : public ::testing::Test {};
typedef ::testing::Types<char, int, float> TensorTypes;
TYPED_TEST_CASE(TensorGPUTest, TensorTypes);
TYPED_TEST_CASE(TensorGPUDeathTest, TensorTypes);

TYPED_TEST(TensorGPUTest, TensorInitializedEmpty) {
  if (!caffe2::HasCudaGPU()) return;
  TensorCUDA tensor;
  EXPECT_EQ(tensor.ndim(), 0);
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  tensor.Resize(dims);
  EXPECT_EQ(tensor.ndim(), 3);
  EXPECT_EQ(tensor.dim32(0), 2);
  EXPECT_EQ(tensor.dim32(1), 3);
  EXPECT_EQ(tensor.dim32(2), 5);
  EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
  EXPECT_TRUE(tensor.data<TypeParam>() != nullptr);
}

TYPED_TEST(TensorGPUTest, TensorInitializedNonEmpty) {
  if (!HasCudaGPU()) return;
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  TensorCUDA tensor(dims);
  EXPECT_EQ(tensor.ndim(), 3);
  EXPECT_EQ(tensor.dim32(0), 2);
  EXPECT_EQ(tensor.dim32(1), 3);
  EXPECT_EQ(tensor.dim32(2), 5);
  EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
  EXPECT_TRUE(tensor.data<TypeParam>() != nullptr);
  dims[0] = 7;
  dims[1] = 11;
  dims[2] = 13;
  dims.push_back(17);
  tensor.Resize(dims);
  EXPECT_EQ(tensor.ndim(), 4);
  EXPECT_EQ(tensor.dim32(0), 7);
  EXPECT_EQ(tensor.dim32(1), 11);
  EXPECT_EQ(tensor.dim32(2), 13);
  EXPECT_EQ(tensor.dim32(3), 17);
  EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
  EXPECT_TRUE(tensor.data<TypeParam>() != nullptr);
}

TYPED_TEST(TensorGPUTest, TensorShareData) {
  if (!HasCudaGPU()) return;
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  TensorCUDA tensor(dims);
  TensorCUDA other_tensor(dims);
  EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
  other_tensor.ShareData(tensor);
  EXPECT_TRUE(tensor.data<TypeParam>() != nullptr);
  EXPECT_TRUE(other_tensor.data<TypeParam>() != nullptr);
  EXPECT_EQ(tensor.data<TypeParam>(), other_tensor.data<TypeParam>());
}

TYPED_TEST(TensorGPUTest, TensorShareDataCanUseDifferentShapes) {
  if (!HasCudaGPU()) return;
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  vector<int> alternate_dims(1);
  alternate_dims[0] = 2 * 3 * 5;
  TensorCUDA tensor(dims);
  TensorCUDA other_tensor(alternate_dims);
  EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
  other_tensor.ShareData(tensor);
  EXPECT_EQ(other_tensor.ndim(), 1);
  EXPECT_EQ(other_tensor.dim32(0), alternate_dims[0]);
  EXPECT_TRUE(tensor.data<TypeParam>() != nullptr);
  EXPECT_TRUE(other_tensor.data<TypeParam>() != nullptr);
  EXPECT_EQ(tensor.data<TypeParam>(), other_tensor.data<TypeParam>());
}

TYPED_TEST(TensorGPUTest, NoLongerSharesAfterResize) {
  if (!HasCudaGPU()) return;
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  TensorCUDA tensor(dims);
  TensorCUDA other_tensor(dims);
  EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
  other_tensor.ShareData(tensor);
  EXPECT_EQ(tensor.data<TypeParam>(), other_tensor.data<TypeParam>());
  auto* old_pointer = other_tensor.data<TypeParam>();

  dims[0] = 7;
  tensor.Resize(dims);
  EXPECT_EQ(old_pointer, other_tensor.data<TypeParam>());
  EXPECT_NE(old_pointer, tensor.mutable_data<TypeParam>());
}

TYPED_TEST(TensorGPUDeathTest, CannotAccessDataWhenEmpty) {
  if (!HasCudaGPU()) return;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  TensorCUDA tensor;
  EXPECT_EQ(tensor.ndim(), 0);
  EXPECT_THROW(tensor.data<TypeParam>(), EnforceNotMet);
}

#define TEST_SERIALIZATION_GPU_WITH_TYPE(TypeParam, field_name)            \
  TEST(TensorGPUTest, TensorSerialization_##TypeParam) {                   \
    if (!HasCudaGPU()) {                                                   \
      return;                                                              \
    }                                                                      \
    Blob blob;                                                             \
    TensorCPU cpu_tensor;                                                  \
    cpu_tensor.Resize(2, 3);                                               \
    for (int i = 0; i < 6; ++i) {                                          \
      cpu_tensor.mutable_data<TypeParam>()[i] = static_cast<TypeParam>(i); \
    }                                                                      \
    blob.GetMutable<TensorCUDA>()->CopyFrom(cpu_tensor);                   \
    string serialized = blob.Serialize("test");                            \
    BlobProto proto;                                                       \
    CAFFE_ENFORCE(proto.ParseFromString(serialized));                      \
    EXPECT_EQ(proto.name(), "test");                                       \
    EXPECT_EQ(proto.type(), "Tensor");                                     \
    EXPECT_TRUE(proto.has_tensor());                                       \
    const TensorProto& tensor_proto = proto.tensor();                      \
    EXPECT_EQ(                                                             \
        tensor_proto.data_type(),                                          \
        TypeMetaToDataType(TypeMeta::Make<TypeParam>()));                  \
    EXPECT_EQ(tensor_proto.field_name##_size(), 6);                        \
    for (int i = 0; i < 6; ++i) {                                          \
      EXPECT_EQ(tensor_proto.field_name(i), static_cast<TypeParam>(i));    \
    }                                                                      \
    Blob new_blob;                                                         \
    EXPECT_NO_THROW(new_blob.Deserialize(serialized));                     \
    EXPECT_TRUE(new_blob.IsType<TensorCUDA>());                            \
    TensorCPU new_cpu_tensor(blob.Get<TensorCUDA>());                      \
    EXPECT_EQ(new_cpu_tensor.ndim(), 2);                                   \
    EXPECT_EQ(new_cpu_tensor.dim(0), 2);                                   \
    EXPECT_EQ(new_cpu_tensor.dim(1), 3);                                   \
    for (int i = 0; i < 6; ++i) {                                          \
      EXPECT_EQ(                                                           \
          cpu_tensor.data<TypeParam>()[i],                                 \
          new_cpu_tensor.data<TypeParam>()[i]);                            \
    }                                                                      \
  }

TEST_SERIALIZATION_GPU_WITH_TYPE(bool, int32_data)
TEST_SERIALIZATION_GPU_WITH_TYPE(double, double_data)
TEST_SERIALIZATION_GPU_WITH_TYPE(float, float_data)
TEST_SERIALIZATION_GPU_WITH_TYPE(int, int32_data)
TEST_SERIALIZATION_GPU_WITH_TYPE(int8_t, int32_data)
TEST_SERIALIZATION_GPU_WITH_TYPE(int16_t, int32_data)
TEST_SERIALIZATION_GPU_WITH_TYPE(uint8_t, int32_data)
TEST_SERIALIZATION_GPU_WITH_TYPE(uint16_t, int32_data)
TEST_SERIALIZATION_GPU_WITH_TYPE(int64_t, int64_data)

TEST(TensorTest, TensorSerializationMultiDevices) {
  Blob blob;
  TensorCPU tensor;
  tensor.Resize(2, 3);
  for (int i = 0; i < 6; ++i) {
    tensor.mutable_data<float>()[i] = i;
  }
  for (int gpu_id = 0; gpu_id < NumCudaDevices(); ++gpu_id) {
    DeviceGuard guard(gpu_id);
    CUDAContext context(gpu_id);
    blob.Reset(new TensorCUDA(tensor, &context));
    string serialized = blob.Serialize("test");
    BlobProto proto;
    CAFFE_ENFORCE(proto.ParseFromString(serialized));
    EXPECT_EQ(proto.name(), "test");
    EXPECT_TRUE(proto.has_tensor());
    const TensorProto& tensor_proto = proto.tensor();
    EXPECT_EQ(tensor_proto.data_type(), TensorProto::FLOAT);
    EXPECT_EQ(tensor_proto.float_data_size(), 6);
    for (int i = 0; i < 6; ++i) {
      EXPECT_EQ(tensor_proto.float_data(i), i);
    }
    EXPECT_TRUE(tensor_proto.has_device_detail());
    EXPECT_EQ(tensor_proto.device_detail().device_type(), CUDA);
    EXPECT_EQ(tensor_proto.device_detail().cuda_gpu_id(), gpu_id);
    // Test if the restored blob is still of the same device.
    blob.Reset();
    EXPECT_NO_THROW(blob.Deserialize(serialized));
    EXPECT_TRUE(blob.IsType<TensorCUDA>());
    EXPECT_EQ(GetGPUIDForPointer(blob.Get<TensorCUDA>().data<float>()),
              gpu_id);
    // Test if we force the restored blob on a different device, we
    // can still get so.
    blob.Reset();
    proto.mutable_tensor()->mutable_device_detail()->set_cuda_gpu_id(0);
    EXPECT_NO_THROW(blob.Deserialize(proto.SerializeAsString()));
    EXPECT_TRUE(blob.IsType<TensorCUDA>());
    EXPECT_EQ(GetGPUIDForPointer(blob.Get<TensorCUDA>().data<float>()), 0);
  }
}

}  // namespace
}  // namespace caffe2
