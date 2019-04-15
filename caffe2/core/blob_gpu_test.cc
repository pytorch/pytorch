#include <iostream>  // NOLINT

#include <gtest/gtest.h>
#include "caffe2/core/blob.h"
#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/proto/caffe2_pb.h"

namespace caffe2 {
namespace {

template <typename T> class TensorGPUTest : public ::testing::Test {};
template <typename T> class TensorGPUDeathTest : public ::testing::Test {};
typedef ::testing::Types<char, int, float> TensorTypes;
TYPED_TEST_CASE(TensorGPUTest, TensorTypes);
TYPED_TEST_CASE(TensorGPUDeathTest, TensorTypes);

TYPED_TEST(TensorGPUTest, TensorInitializedEmpty) {
  if (!caffe2::HasCudaGPU()) return;
  Tensor tensor(CUDA);
  EXPECT_EQ(tensor.numel(), 0);
  EXPECT_EQ(tensor.dim(), 1);
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  tensor.Resize(dims);
  EXPECT_EQ(tensor.dim(), 3);
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
  Tensor tensor(dims, CUDA);
  EXPECT_EQ(tensor.dim(), 3);
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
  EXPECT_EQ(tensor.dim(), 4);
  EXPECT_EQ(tensor.dim32(0), 7);
  EXPECT_EQ(tensor.dim32(1), 11);
  EXPECT_EQ(tensor.dim32(2), 13);
  EXPECT_EQ(tensor.dim32(3), 17);
  EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
  EXPECT_TRUE(tensor.data<TypeParam>() != nullptr);
}

TYPED_TEST(TensorGPUTest, TensorAlias) {
  if (!HasCudaGPU()) return;
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  Tensor tensor(dims, CUDA);
  EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
  Tensor other_tensor = tensor.Alias();
  EXPECT_TRUE(tensor.data<TypeParam>() != nullptr);
  EXPECT_TRUE(other_tensor.data<TypeParam>() != nullptr);
  EXPECT_EQ(tensor.data<TypeParam>(), other_tensor.data<TypeParam>());
}

TYPED_TEST(TensorGPUTest, TensorAliasCanUseDifferentShapes) {
  if (!HasCudaGPU()) return;
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  vector<int> alternate_dims(1);
  alternate_dims[0] = 2 * 3 * 5;
  Tensor tensor(dims, CUDA);
  EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
  Tensor other_tensor = tensor.Alias();
  other_tensor.Resize(alternate_dims);
  EXPECT_EQ(other_tensor.dim(), 1);
  EXPECT_EQ(other_tensor.dim32(0), alternate_dims[0]);
  EXPECT_TRUE(tensor.data<TypeParam>() != nullptr);
  EXPECT_TRUE(other_tensor.data<TypeParam>() != nullptr);
  EXPECT_EQ(tensor.data<TypeParam>(), other_tensor.data<TypeParam>());
}

TYPED_TEST(TensorGPUTest, NoLongerAliasAfterNumelChanges) {
  if (!HasCudaGPU()) return;
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  Tensor tensor(dims, CUDA);
  EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
  Tensor other_tensor = tensor.Alias();
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
  Tensor tensor(CUDA);
  EXPECT_EQ(tensor.dim(), 1);
  EXPECT_EQ(tensor.numel(), 0);
  EXPECT_THROW(tensor.data<TypeParam>(), EnforceNotMet);
}

#define TEST_SERIALIZATION_GPU_WITH_TYPE(TypeParam, field_name)            \
  TEST(TensorGPUTest, TensorSerialization_##TypeParam) {                   \
    if (!HasCudaGPU()) {                                                   \
      return;                                                              \
    }                                                                      \
    Blob blob;                                                             \
    Tensor cpu_tensor(CPU);                                                \
    cpu_tensor.Resize(2, 3);                                               \
    for (int i = 0; i < 6; ++i) {                                          \
      cpu_tensor.mutable_data<TypeParam>()[i] = static_cast<TypeParam>(i); \
    }                                                                      \
    BlobGetMutableTensor(&blob, CUDA)->CopyFrom(cpu_tensor);               \
    string serialized = SerializeBlob(blob, "test");                       \
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
    EXPECT_NO_THROW(DeserializeBlob(serialized, &new_blob));               \
    EXPECT_TRUE(BlobIsTensorType(new_blob, CUDA));                         \
    Tensor new_cpu_tensor(blob.Get<Tensor>(), CPU);                        \
    EXPECT_EQ(new_cpu_tensor.dim(), 2);                                    \
    EXPECT_EQ(new_cpu_tensor.size(0), 2);                                  \
    EXPECT_EQ(new_cpu_tensor.size(1), 3);                                  \
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

TEST(TensorConstruction, ReinitializeTensorTest) {
  if (!HasCudaGPU()) return;
  Tensor x = caffe2::empty({1}, at::dtype<float>().device(CUDA, 0));
  auto* data_before = x.template mutable_data<float>();
  // We'll only compare device_type in ReinitializeTensor,
  // so no tensor reallocation will happen here
  ReinitializeTensor(&x, {1}, at::dtype<float>().device(CUDA));
  auto* data_after = x.template mutable_data<float>();
  EXPECT_EQ(data_before, data_after);
}

TEST(TensorTest, TensorSerializationMultiDevices) {
  Blob blob;
  Tensor tensor(CPU);
  tensor.Resize(2, 3);
  for (int i = 0; i < 6; ++i) {
    tensor.mutable_data<float>()[i] = i;
  }
  for (int gpu_id = 0; gpu_id < NumCudaDevices(); ++gpu_id) {
    CUDAGuard guard(gpu_id);
    CUDAContext context(gpu_id); // switch to the current gpu
    blob.Reset(new Tensor(tensor, CUDA));
    string serialized = SerializeBlob(blob, "test");
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
    EXPECT_EQ(tensor_proto.device_detail().device_type(), PROTO_CUDA);
    EXPECT_EQ(tensor_proto.device_detail().device_id(), gpu_id);
    // Test if the restored blob is still of the same device.
    blob.Reset();
    EXPECT_NO_THROW(DeserializeBlob(serialized, &blob));
    EXPECT_TRUE(BlobIsTensorType(blob, CUDA));
    EXPECT_EQ(GetGPUIDForPointer(blob.Get<TensorCUDA>().data<float>()),
              gpu_id);
    // Test if we force the restored blob on a different device, we
    // can still get so.
    blob.Reset();
    proto.mutable_tensor()->mutable_device_detail()->set_device_id(0);
    EXPECT_NO_THROW(DeserializeBlob(proto.SerializeAsString(), &blob));
    EXPECT_TRUE(BlobIsTensorType(blob, CUDA));
    EXPECT_EQ(GetGPUIDForPointer(blob.Get<TensorCUDA>().data<float>()), 0);
  }
}

}  // namespace
}  // namespace caffe2
