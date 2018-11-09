#include <iostream>
#include <memory>
#include <mutex>

#include <gtest/gtest.h>
#include "c10/util/Registry.h"
#include "caffe2/core/blob.h"
#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/common.h"
#include "caffe2/core/context.h"
#include "caffe2/core/db.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/qtensor.h"
#include "caffe2/core/qtensor_serialization.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/types.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/utils/proto_utils.h"

C10_DEFINE_int64(caffe2_test_big_tensor_size, 100000000, "");
C10_DECLARE_int(caffe2_tensor_chunk_size);
C10_DECLARE_bool(caffe2_serialize_fp16_as_bytes);

namespace caffe2 {
using namespace ::caffe2::db;
namespace {
class BlobTestFoo {
 public:
  int32_t val;
};
class BlobTestBar {};
class BlobTestNonDefaultConstructible {
 public:
  BlobTestNonDefaultConstructible() = delete;
  BlobTestNonDefaultConstructible(int x) : val(x) {}
  int32_t val;
};
}

CAFFE_KNOWN_TYPE(BlobTestFoo);
CAFFE_KNOWN_TYPE(BlobTestBar);
CAFFE_KNOWN_TYPE(BlobTestNonDefaultConstructible);

class BlobTestFooSerializer : public BlobSerializerBase {
 public:
  BlobTestFooSerializer() {}
  ~BlobTestFooSerializer() {}
  /**
   * Serializes a Blob. Note that this blob has to contain Tensor,
   * otherwise this function produces a fatal error.
   */
  void Serialize(
      const void* pointer,
      TypeMeta typeMeta,
      const string& name,
      SerializationAcceptor acceptor) override {
    CAFFE_ENFORCE(typeMeta.Match<BlobTestFoo>());

    BlobProto blob_proto;
    blob_proto.set_name(name);
    blob_proto.set_type("BlobTestFoo");
    // For simplicity we will just serialize the 4-byte content as a string.
    blob_proto.set_content(std::string(
        reinterpret_cast<const char*>(
            &static_cast<const BlobTestFoo*>(pointer)->val),
        sizeof(int32_t)));
    acceptor(name, SerializeBlobProtoAsString_EnforceCheck(blob_proto));
  }
};

class BlobTestFooDeserializer : public BlobDeserializerBase {
 public:
  void Deserialize(const BlobProto& proto, Blob* blob) override {
    blob->GetMutable<BlobTestFoo>()->val =
        reinterpret_cast<const int32_t*>(proto.content().c_str())[0];
  }
};

REGISTER_BLOB_SERIALIZER((TypeMeta::Id<BlobTestFoo>()), BlobTestFooSerializer);
REGISTER_BLOB_DESERIALIZER(BlobTestFoo, BlobTestFooDeserializer);

namespace {

TEST(BlobTest, Blob) {
  Blob blob;

  int* int_unused CAFFE2_UNUSED = blob.GetMutable<int>();
  EXPECT_TRUE(blob.IsType<int>());
  EXPECT_FALSE(blob.IsType<BlobTestFoo>());
  EXPECT_FALSE(BlobIsTensorType(blob, CPU));

  BlobTestFoo* foo_unused CAFFE2_UNUSED = blob.GetMutable<BlobTestFoo>();
  EXPECT_TRUE(blob.IsType<BlobTestFoo>());
  EXPECT_FALSE(blob.IsType<int>());
  EXPECT_FALSE(BlobIsTensorType(blob, CPU));

  Tensor* tensor_unused CAFFE2_UNUSED = BlobGetMutableTensor(&blob, CPU);
  EXPECT_TRUE(BlobIsTensorType(blob, CPU));
  EXPECT_FALSE(blob.IsType<BlobTestFoo>());
  EXPECT_FALSE(blob.IsType<int>());
}

TEST(BlobTest, BlobUninitialized) {
  Blob blob;
  ASSERT_THROW(blob.Get<int>(), EnforceNotMet);
}

TEST(BlobTest, BlobWrongType) {
  Blob blob;
  BlobTestFoo* foo_unused CAFFE2_UNUSED = blob.GetMutable<BlobTestFoo>();
  EXPECT_TRUE(blob.IsType<BlobTestFoo>());
  EXPECT_FALSE(blob.IsType<int>());
  // When not null, we should only call with the right type.
  EXPECT_NE(&blob.Get<BlobTestFoo>(), nullptr);
  ASSERT_THROW(blob.Get<int>(), EnforceNotMet);
}

TEST(BlobTest, BlobReset) {
  Blob blob;
  std::unique_ptr<BlobTestFoo> foo(new BlobTestFoo());
  EXPECT_TRUE(blob.Reset(foo.release()) != nullptr);
  // Also test that Reset works.
  blob.Reset();
}

TEST(BlobTest, BlobMove) {
  Blob blob1;
  std::unique_ptr<BlobTestFoo> foo(new BlobTestFoo());
  auto* fooPtr = foo.get();
  EXPECT_TRUE(blob1.Reset(foo.release()) != nullptr);
  Blob blob2;
  blob2 = std::move(blob1);
  ASSERT_THROW(blob1.Get<BlobTestFoo>(), EnforceNotMet);
  EXPECT_EQ(&blob2.Get<BlobTestFoo>(), fooPtr);
  Blob blob3{std::move(blob2)};
  EXPECT_EQ(&blob3.Get<BlobTestFoo>(), fooPtr);
}

TEST(BlobTest, BlobNonConstructible) {
  Blob blob;
  ASSERT_THROW(blob.Get<BlobTestNonDefaultConstructible>(), EnforceNotMet);
  // won't work because it's not default constructible
  // blob.GetMutable<BlobTestNonDefaultConstructible>();
  EXPECT_FALSE(
      blob.GetMutableOrNull<BlobTestNonDefaultConstructible>() != nullptr);
  EXPECT_TRUE(blob.Reset(new BlobTestNonDefaultConstructible(42)) != nullptr);
  ASSERT_NO_THROW(blob.Get<BlobTestNonDefaultConstructible>());
  ASSERT_TRUE(
      blob.GetMutableOrNull<BlobTestNonDefaultConstructible>() != nullptr);
  EXPECT_EQ(blob.Get<BlobTestNonDefaultConstructible>().val, 42);
  blob.GetMutableOrNull<BlobTestNonDefaultConstructible>()->val = 37;
  EXPECT_EQ(blob.Get<BlobTestNonDefaultConstructible>().val, 37);
}

TEST(BlobTest, BlobShareExternalPointer) {
  Blob blob;
  std::unique_ptr<BlobTestFoo> foo(new BlobTestFoo());
  EXPECT_EQ(blob.ShareExternal<BlobTestFoo>(foo.get()), foo.get());
  EXPECT_TRUE(blob.IsType<BlobTestFoo>());
  // Also test that Reset works.
  blob.Reset();
}

TEST(BlobTest, BlobShareExternalObject) {
  Blob blob;
  BlobTestFoo foo;
  EXPECT_EQ(blob.ShareExternal<BlobTestFoo>(&foo), &foo);
  EXPECT_TRUE(blob.IsType<BlobTestFoo>());
  // Also test that Reset works.
  blob.Reset();
}

TEST(BlobTest, StringSerialization) {
  const std::string kTestString = "Hello world?";
  Blob blob;
  *blob.GetMutable<std::string>() = kTestString;

  string serialized = SerializeBlob(blob, "test");
  BlobProto proto;
  CHECK(proto.ParseFromString(serialized));
  EXPECT_EQ(proto.name(), "test");
  EXPECT_EQ(proto.type(), "std::string");
  EXPECT_FALSE(proto.has_tensor());
  EXPECT_EQ(proto.content(), kTestString);
}

TEST(TensorNonTypedTest, TensorChangeType) {
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  Tensor tensor(dims, CPU);

  auto* ptr = tensor.mutable_data<int>();
  EXPECT_TRUE(ptr != nullptr);
  EXPECT_TRUE(tensor.data<int>() != nullptr);
  EXPECT_TRUE(tensor.dtype().Match<int>());

  // int and float are same size, so should retain the pointer
  // NB: this is only true when the use_count of the underlying Storage is 1, if
  // the underlying Storage is shared between multiple Tensors We'll create a
  // new Storage when the data type changes
  EXPECT_TRUE(tensor.mutable_data<float>() == (float*)ptr);
  EXPECT_TRUE(tensor.data<float>() == (const float*)ptr);
  EXPECT_TRUE(tensor.dtype().Match<float>());

  // at::Half is smaller, so still should share buffer
  EXPECT_TRUE(tensor.mutable_data<at::Half>() == (at::Half*)ptr);
  EXPECT_TRUE(tensor.data<at::Half>() == (const at::Half*)ptr);
  EXPECT_TRUE(tensor.dtype().Match<at::Half>());

  // share the data with other tensor so that the pointer won't be reused
  // when we reallocate
  Tensor other_tensor(dims, CPU);
  other_tensor.ShareData(tensor);
  // but double is bigger, so it should allocate a new one
  auto* doubleptr = tensor.mutable_data<double>();
  EXPECT_TRUE(doubleptr != (double*)ptr);
  EXPECT_TRUE(doubleptr != nullptr);
  EXPECT_TRUE(tensor.data<double>() != nullptr);
  EXPECT_TRUE(tensor.dtype().Match<double>());
}

TEST(TensorNonTypedTest, NonDefaultConstructible) {
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  Tensor tensor(dims, CPU);

  // this doesn't compile - good!
  // auto* ptr = tensor.mutable_data<BlobTestNonDefaultConstructible>();
  EXPECT_THROW(
      tensor.raw_mutable_data(
          TypeMeta::Make<BlobTestNonDefaultConstructible>()),
      EnforceNotMet);
}

template <typename T> class TensorCPUTest : public ::testing::Test {};
template <typename T> class TensorCPUDeathTest : public ::testing::Test {};
typedef ::testing::Types<char, int, float> TensorTypes;
TYPED_TEST_CASE(TensorCPUTest, TensorTypes);
TYPED_TEST_CASE(TensorCPUDeathTest, TensorTypes);

TYPED_TEST(TensorCPUTest, TensorInitializedEmpty) {
  Tensor tensor(CPU);
  EXPECT_EQ(tensor.dim(), 1);
  EXPECT_EQ(tensor.numel(), 0);
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  tensor.Resize(dims);
  EXPECT_EQ(tensor.dim(), 3);
  EXPECT_EQ(tensor.dim32(0), 2);
  EXPECT_EQ(tensor.dim32(1), 3);
  EXPECT_EQ(tensor.dim32(2), 5);
  EXPECT_EQ(tensor.numel(), 2 * 3 * 5);
  EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
  EXPECT_TRUE(tensor.data<TypeParam>() != nullptr);
}

TYPED_TEST(TensorCPUTest, TensorInitializedNonEmpty) {
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  Tensor tensor(dims, CPU);
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

TYPED_TEST(TensorCPUTest, TensorInitializedZeroDim) {
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 0;
  dims[2] = 5;
  Tensor tensor(dims, CPU);
  EXPECT_EQ(tensor.dim(), 3);
  EXPECT_EQ(tensor.dim32(0), 2);
  EXPECT_EQ(tensor.dim32(1), 0);
  EXPECT_EQ(tensor.dim32(2), 5);
  EXPECT_TRUE(tensor.mutable_data<TypeParam>() == nullptr);
  EXPECT_TRUE(tensor.data<TypeParam>() == nullptr);
}

TYPED_TEST(TensorCPUTest, TensorResizeZeroDim) {
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  Tensor tensor(dims, CPU);
  EXPECT_EQ(tensor.dim(), 3);
  EXPECT_EQ(tensor.dim32(0), 2);
  EXPECT_EQ(tensor.dim32(1), 3);
  EXPECT_EQ(tensor.dim32(2), 5);
  EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
  EXPECT_TRUE(tensor.data<TypeParam>() != nullptr);

  dims[0] = 7;
  dims[1] = 0;
  dims[2] = 13;
  tensor.Resize(dims);
  EXPECT_EQ(tensor.numel(), 0);
  EXPECT_EQ(tensor.dim(), 3);
  EXPECT_EQ(tensor.dim32(0), 7);
  EXPECT_EQ(tensor.dim32(1), 0);
  EXPECT_EQ(tensor.dim32(2), 13);
  // output value can be arbitrary, but the call to data() shouldn't crash
  tensor.mutable_data<TypeParam>();
  tensor.data<TypeParam>();
}

TYPED_TEST(TensorCPUTest, TensorInitializedScalar) {
  vector<int> dims;
  Tensor tensor(dims, CPU);
  EXPECT_EQ(tensor.dim(), 0);
  EXPECT_EQ(tensor.numel(), 1);
  EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
  EXPECT_TRUE(tensor.data<TypeParam>() != nullptr);
}

TYPED_TEST(TensorCPUTest, TensorShareData) {
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  Tensor tensor(dims, CPU);
  Tensor other_tensor(dims, CPU);
  EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
  other_tensor.ShareData(tensor);
  EXPECT_TRUE(tensor.data<TypeParam>() != nullptr);
  EXPECT_TRUE(other_tensor.data<TypeParam>() != nullptr);
  EXPECT_EQ(tensor.data<TypeParam>(), other_tensor.data<TypeParam>());
  // Set one value, check the other
  for (int i = 0; i < tensor.numel(); ++i) {
    tensor.mutable_data<TypeParam>()[i] = i;
    EXPECT_EQ(other_tensor.data<TypeParam>()[i], i);
  }
}

TYPED_TEST(TensorCPUTest, TensorShareDataRawPointer) {
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  std::unique_ptr<TypeParam[]> raw_buffer(new TypeParam[2*3*5]);
  Tensor tensor(dims, CPU);
  tensor.ShareExternalPointer(raw_buffer.get());
  EXPECT_EQ(tensor.mutable_data<TypeParam>(), raw_buffer.get());
  EXPECT_EQ(tensor.data<TypeParam>(), raw_buffer.get());
  // Set one value, check the other
  for (int i = 0; i < tensor.numel(); ++i) {
    raw_buffer.get()[i] = i;
    EXPECT_EQ(tensor.data<TypeParam>()[i], i);
  }
}

TYPED_TEST(TensorCPUTest, TensorShareDataRawPointerWithMeta) {
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  std::unique_ptr<TypeParam[]> raw_buffer(new TypeParam[2 * 3 * 5]);
  Tensor tensor(dims, CPU);
  TypeMeta meta = TypeMeta::Make<TypeParam>();
  tensor.ShareExternalPointer(raw_buffer.get(), meta);
  EXPECT_EQ(tensor.mutable_data<TypeParam>(), raw_buffer.get());
  EXPECT_EQ(tensor.data<TypeParam>(), raw_buffer.get());
  // Set one value, check the other
  for (int i = 0; i < tensor.numel(); ++i) {
    raw_buffer.get()[i] = i;
    EXPECT_EQ(tensor.data<TypeParam>()[i], i);
  }
}

TYPED_TEST(TensorCPUTest, TensorShareDataCanUseDifferentShapes) {
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  vector<int> alternate_dims(1);
  alternate_dims[0] = 2 * 3 * 5;
  Tensor tensor(dims, CPU);
  Tensor other_tensor(alternate_dims, CPU);
  EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
  other_tensor.ShareData(tensor);
  EXPECT_EQ(other_tensor.dim(), 1);
  EXPECT_EQ(other_tensor.dim32(0), alternate_dims[0]);
  EXPECT_TRUE(tensor.data<TypeParam>() != nullptr);
  EXPECT_TRUE(other_tensor.data<TypeParam>() != nullptr);
  EXPECT_EQ(tensor.data<TypeParam>(), other_tensor.data<TypeParam>());
  // Set one value, check the other
  for (int i = 0; i < tensor.numel(); ++i) {
    tensor.mutable_data<TypeParam>()[i] = i;
    EXPECT_EQ(other_tensor.data<TypeParam>()[i], i);
  }
}


TYPED_TEST(TensorCPUTest, NoLongerSharesAfterResize) {
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  Tensor tensor(dims, CPU);
  Tensor other_tensor(dims, CPU);
  EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
  other_tensor.ShareData(tensor);
  EXPECT_EQ(tensor.data<TypeParam>(), other_tensor.data<TypeParam>());
  auto* old_pointer = other_tensor.data<TypeParam>();

  dims[0] = 7;
  tensor.Resize(dims);
  EXPECT_EQ(old_pointer, other_tensor.data<TypeParam>());
  EXPECT_NE(old_pointer, tensor.mutable_data<TypeParam>());
}

TYPED_TEST(TensorCPUTest, NoLongerSharesAfterFreeMemory) {
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  Tensor tensor(dims, CPU);
  Tensor other_tensor(dims, CPU);
  EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
  other_tensor.ShareData(tensor);
  EXPECT_EQ(tensor.data<TypeParam>(), other_tensor.data<TypeParam>());
  auto* old_pointer = other_tensor.data<TypeParam>();

  tensor.FreeMemory();
  EXPECT_EQ(old_pointer, other_tensor.data<TypeParam>());
  EXPECT_NE(old_pointer, tensor.mutable_data<TypeParam>());
}

TYPED_TEST(TensorCPUTest, KeepOnShrink) {
  // Set flags (defaults)
  FLAGS_caffe2_keep_on_shrink = true;
  FLAGS_caffe2_max_keep_on_shrink_memory = LLONG_MAX;

  vector<int> dims{2, 3, 5};
  Tensor tensor(dims, CPU);
  TypeParam* ptr = tensor.mutable_data<TypeParam>();
  EXPECT_TRUE(ptr != nullptr);
  // Expanding - will reallocate
  tensor.Resize(3, 4, 6);
  TypeParam* larger_ptr = tensor.mutable_data<TypeParam>();
  EXPECT_TRUE(larger_ptr != nullptr);

  // This check can fail when malloc() returns the same recently freed address
  //EXPECT_NE(ptr, larger_ptr);

  // Shrinking - will not reallocate
  tensor.Resize(1, 2, 4);
  TypeParam* smaller_ptr = tensor.mutable_data<TypeParam>();
  EXPECT_TRUE(smaller_ptr != nullptr);
  EXPECT_EQ(larger_ptr, smaller_ptr);
  // resize to 0 in the meantime;
  tensor.Resize(3, 0, 6);
  // Expanding but still under capacity - will not reallocate
  tensor.Resize(2, 3, 5);
  TypeParam* new_ptr = tensor.mutable_data<TypeParam>();
  EXPECT_TRUE(new_ptr != nullptr);
  EXPECT_EQ(larger_ptr, new_ptr);
}

TYPED_TEST(TensorCPUTest, MaxKeepOnShrink) {
  // Set flags
  FLAGS_caffe2_keep_on_shrink = true;
  FLAGS_caffe2_max_keep_on_shrink_memory = 8 * 4 * sizeof(TypeParam);

  vector<int> dims{1, 8, 8};
  Tensor tensor(dims, CPU);
  TypeParam* ptr = tensor.mutable_data<TypeParam>();
  EXPECT_TRUE(ptr != nullptr);
  // Shrinking - will not reallocate
  tensor.Resize(1, 7, 8);
  TypeParam* smaller_ptr = tensor.mutable_data<TypeParam>();
  EXPECT_TRUE(smaller_ptr != nullptr);
  EXPECT_EQ(ptr, smaller_ptr);
  // Resize to more than maximum shrink, should reallocate
  tensor.Resize(1, 1, 8);
  TypeParam* new_ptr = tensor.mutable_data<TypeParam>();
  EXPECT_TRUE(new_ptr != nullptr);

  // This check can fail when malloc() returns the same recently freed address
  //EXPECT_NE(ptr, new_ptr);

  // Restore default flags
  FLAGS_caffe2_max_keep_on_shrink_memory = LLONG_MAX;
}

TYPED_TEST(TensorCPUDeathTest, CannotAccessRawDataWhenEmpty) {
  Tensor tensor(CPU);
  EXPECT_EQ(tensor.dim(), 1);
  EXPECT_EQ(tensor.numel(), 0);
  ASSERT_ANY_THROW(tensor.raw_data());
}

TYPED_TEST(TensorCPUDeathTest, CannotAccessDataWhenEmpty) {
  Tensor tensor(CPU);
  EXPECT_EQ(tensor.dim(), 1);
  EXPECT_EQ(tensor.numel(), 0);
  ASSERT_ANY_THROW(tensor.data<TypeParam>());
}

TEST(TensorTest, TensorNonFundamentalType) {
  Tensor tensor(vector<int>{2, 3, 4}, CPU);
  EXPECT_TRUE(tensor.mutable_data<std::string>() != nullptr);
  const std::string* ptr = tensor.data<std::string>();
  for (int i = 0; i < tensor.numel(); ++i) {
    EXPECT_TRUE(ptr[i] == "");
  }
}

TEST(TensorTest, TensorNonFundamentalTypeClone) {
  Tensor tensor(vector<int>{2, 3, 4}, CPU);
  std::string* ptr = tensor.mutable_data<std::string>();
  EXPECT_TRUE(ptr != nullptr);
  for (int i = 0; i < tensor.numel(); ++i) {
    EXPECT_TRUE(ptr[i] == "");
    ptr[i] = "filled";
  }
  Tensor dst_tensor = tensor.Clone();
  const std::string* dst_ptr = dst_tensor.data<std::string>();
  for (int i = 0; i < dst_tensor.numel(); ++i) {
    EXPECT_TRUE(dst_ptr[i] == "filled");
  }
  // Change the original tensor
  for (int i = 0; i < tensor.numel(); ++i) {
    EXPECT_TRUE(ptr[i] == "filled");
    ptr[i] = "changed";
  }
  // Confirm that the cloned tensor is not affect
  for (int i = 0; i < dst_tensor.numel(); ++i) {
    EXPECT_TRUE(dst_ptr[i] == "filled");
  }
}

TEST(TensorTest, Tensor64BitDimension) {
  // Initialize a large tensor.
  int64_t large_number =
      static_cast<int64_t>(std::numeric_limits<int>::max()) + 1;
  Tensor tensor(vector<int64_t>{large_number}, CPU);
  EXPECT_EQ(tensor.dim(), 1);
  EXPECT_EQ(tensor.size(0), large_number);
  EXPECT_EQ(tensor.numel(), large_number);
  try {
    EXPECT_TRUE(tensor.mutable_data<char>() != nullptr);
  } catch (const EnforceNotMet& e) {
    string msg = e.what();
    size_t found = msg.find("posix_memalign");
    if (found != string::npos) {
      msg = msg.substr(0, msg.find('\n'));
      LOG(WARNING) << msg;
      LOG(WARNING) << "Out of memory issue with posix_memalign;\n";
      return;
    } else {
      throw e;
    }
  }
  EXPECT_EQ(tensor.nbytes(), large_number * sizeof(char));
  EXPECT_EQ(tensor.itemsize(), sizeof(char));
  // Try to go even larger, but this time we will not do mutable_data because we
  // do not have a large enough memory.
  tensor.Resize(large_number, 100);
  EXPECT_EQ(tensor.dim(), 2);
  EXPECT_EQ(tensor.size(0), large_number);
  EXPECT_EQ(tensor.size(1), 100);
  EXPECT_EQ(tensor.numel(), large_number * 100);
}

TEST(TensorDeathTest, CannotCastDownLargeDims) {
  int64_t large_number =
      static_cast<int64_t>(std::numeric_limits<int>::max()) + 1;
  Tensor tensor(vector<int64_t>{large_number}, CPU);
  EXPECT_EQ(tensor.dim(), 1);
  EXPECT_EQ(tensor.size(0), large_number);
  ASSERT_THROW(tensor.dim32(0), EnforceNotMet);
}

#define TEST_SERIALIZATION_WITH_TYPE(TypeParam, field_name)               \
  TEST(TensorTest, TensorSerialization_##TypeParam) {                     \
    Blob blob;                                                            \
    Tensor* tensor = BlobGetMutableTensor(&blob, CPU);                    \
    tensor->Resize(2, 3);                                                 \
    for (int i = 0; i < 6; ++i) {                                         \
      tensor->mutable_data<TypeParam>()[i] = static_cast<TypeParam>(i);   \
    }                                                                     \
    string serialized = SerializeBlob(blob, "test");                      \
    BlobProto proto;                                                      \
    CHECK(proto.ParseFromString(serialized));                             \
    EXPECT_EQ(proto.name(), "test");                                      \
    EXPECT_EQ(proto.type(), "Tensor");                                    \
    EXPECT_TRUE(proto.has_tensor());                                      \
    const TensorProto& tensor_proto = proto.tensor();                     \
    EXPECT_EQ(                                                            \
        tensor_proto.data_type(),                                         \
        TypeMetaToDataType(TypeMeta::Make<TypeParam>()));                 \
    EXPECT_EQ(tensor_proto.field_name##_size(), 6);                       \
    for (int i = 0; i < 6; ++i) {                                         \
      EXPECT_EQ(tensor_proto.field_name(i), static_cast<TypeParam>(i));   \
    }                                                                     \
    Blob new_blob;                                                        \
    EXPECT_NO_THROW(DeserializeBlob(serialized, &new_blob));              \
    EXPECT_TRUE(BlobIsTensorType(new_blob, CPU));                         \
    const TensorCPU& new_tensor = blob.Get<TensorCPU>();                  \
    EXPECT_EQ(new_tensor.dim(), 2);                                       \
    EXPECT_EQ(new_tensor.size(0), 2);                                     \
    EXPECT_EQ(new_tensor.size(1), 3);                                     \
    for (int i = 0; i < 6; ++i) {                                         \
      EXPECT_EQ(                                                          \
          tensor->data<TypeParam>()[i], new_tensor.data<TypeParam>()[i]); \
    }                                                                     \
  }                                                                       \
                                                                          \
  TEST(EmptyTensorTest, TensorSerialization_##TypeParam) {                \
    Blob blob;                                                            \
    TensorCPU* tensor = BlobGetMutableTensor(&blob, CPU);                 \
    tensor->Resize(0, 3);                                                 \
    tensor->mutable_data<TypeParam>();                                    \
    string serialized = SerializeBlob(blob, "test");                      \
    BlobProto proto;                                                      \
    CHECK(proto.ParseFromString(serialized));                             \
    EXPECT_EQ(proto.name(), "test");                                      \
    EXPECT_EQ(proto.type(), "Tensor");                                    \
    EXPECT_TRUE(proto.has_tensor());                                      \
    const TensorProto& tensor_proto = proto.tensor();                     \
    EXPECT_EQ(                                                            \
        tensor_proto.data_type(),                                         \
        TypeMetaToDataType(TypeMeta::Make<TypeParam>()));                 \
    EXPECT_EQ(tensor_proto.field_name##_size(), 0);                       \
    Blob new_blob;                                                        \
    EXPECT_NO_THROW(DeserializeBlob(serialized, &new_blob));              \
    EXPECT_TRUE(BlobIsTensorType(new_blob, CPU));                         \
    const TensorCPU& new_tensor = blob.Get<TensorCPU>();                  \
    EXPECT_EQ(new_tensor.dim(), 2);                                       \
    EXPECT_EQ(new_tensor.size(0), 0);                                     \
    EXPECT_EQ(new_tensor.size(1), 3);                                     \
  }

TEST_SERIALIZATION_WITH_TYPE(bool, int32_data)
TEST_SERIALIZATION_WITH_TYPE(double, double_data)
TEST_SERIALIZATION_WITH_TYPE(float, float_data)
TEST_SERIALIZATION_WITH_TYPE(int, int32_data)
TEST_SERIALIZATION_WITH_TYPE(int8_t, int32_data)
TEST_SERIALIZATION_WITH_TYPE(int16_t, int32_data)
TEST_SERIALIZATION_WITH_TYPE(uint8_t, int32_data)
TEST_SERIALIZATION_WITH_TYPE(uint16_t, int32_data)
TEST_SERIALIZATION_WITH_TYPE(int64_t, int64_data)

TEST(TensorTest, TensorSerialization_CustomType) {
  Blob blob;
  TensorCPU* tensor = BlobGetMutableTensor(&blob, CPU);
  tensor->Resize(2, 3);
  for (int i = 0; i < 6; ++i) {
    tensor->mutable_data<BlobTestFoo>()[i].val = i;
  }
  string serialized = SerializeBlob(blob, "test");
  BlobProto proto;
  CHECK(proto.ParseFromString(serialized));
  EXPECT_EQ(proto.name(), "test");
  EXPECT_EQ(proto.type(), "Tensor");
  Blob new_blob;
  EXPECT_NO_THROW(DeserializeBlob(serialized, &new_blob));
  EXPECT_TRUE(BlobIsTensorType(new_blob, CPU));
  const TensorCPU& new_tensor = blob.Get<TensorCPU>();
  EXPECT_EQ(new_tensor.dim(), 2);
  EXPECT_EQ(new_tensor.size(0), 2);
  EXPECT_EQ(new_tensor.size(1), 3);
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(
        new_tensor.data<BlobTestFoo>()[i].val,
        tensor->data<BlobTestFoo>()[i].val);
  }
}

TEST(TensorTest, Half) {
  const int64_t kSize = 3000000;
  Blob blob;
  TensorCPU* tensor = BlobGetMutableTensor(&blob, CPU);
  tensor->Resize(kSize);
  for (int i = 0; i < tensor->numel(); ++i) {
    tensor->mutable_data<at::Half>()[i].x = i % 10000;
  }
  string serialized = SerializeBlob(blob, "test");
  BlobProto proto;
  CHECK(proto.ParseFromString(serialized));
  EXPECT_EQ(proto.name(), "test");
  EXPECT_EQ(proto.type(), "Tensor");
  EXPECT_TRUE(proto.has_tensor());
  const TensorProto& tensor_proto = proto.tensor();
  EXPECT_EQ(
      tensor_proto.data_type(), TypeMetaToDataType(TypeMeta::Make<at::Half>()));
  if (FLAGS_caffe2_serialize_fp16_as_bytes) {
    EXPECT_EQ(tensor_proto.byte_data().size(), 2 * kSize);
    for (int i = 0; i < kSize; ++i) {
      auto value = tensor->mutable_data<at::Half>()[i].x;
      auto low_bits = static_cast<char>(value & 0xff);
      auto high_bits = static_cast<char>(value >> 8);
      EXPECT_EQ(tensor_proto.byte_data()[2 * i], low_bits);
      EXPECT_EQ(tensor_proto.byte_data()[2 * i + 1], high_bits);
    }
  } else {
    EXPECT_EQ(tensor_proto.int32_data().size(), kSize);
  }
  Blob new_blob;
  EXPECT_NO_THROW(DeserializeBlob(serialized, &new_blob));
  EXPECT_TRUE(BlobIsTensorType(new_blob, CPU));
  const TensorCPU& new_tensor = blob.Get<TensorCPU>();
  EXPECT_EQ(new_tensor.dim(), 1);
  EXPECT_EQ(new_tensor.size(0), kSize);
  for (int i = 0; i < kSize; ++i) {
    EXPECT_EQ(new_tensor.data<at::Half>()[i].x, i % 10000);
  }
}

TEST(TensorTest, TensorFactory) {
  Tensor a = empty({1, 2, 3}, at::device(CPU).dtype<float>());
  EXPECT_NE(a.data<float>(), nullptr);
  a.mutable_data<float>()[0] = 3.0;
  Tensor b = empty({1, 2, 3}, at::device(CPU).dtype<int>());
  EXPECT_NE(b.data<int>(), nullptr);
  b.mutable_data<int>()[0] = 3;
}

TEST(QTensorTest, QTensorSerialization) {
  Blob blob;
  QTensor<CPUContext>* qtensor = blob.GetMutable<QTensor<CPUContext>>();
  qtensor->SetPrecision(5);
  qtensor->SetSigned(false);
  qtensor->SetScale(1.337);
  qtensor->SetBias(-1.337);
  qtensor->Resize(std::vector<int>{2, 3});
  // "Randomly" set bits.
  srand(0);
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 5; ++j) {
      qtensor->SetBitAtIndex(j, i, rand() % 2);
    }
  }

  string serialized = SerializeBlob(blob, "test");
  BlobProto proto;
  CHECK(proto.ParseFromString(serialized));
  EXPECT_EQ(proto.name(), "test");
  EXPECT_EQ(proto.type(), "QTensor");
  EXPECT_TRUE(proto.has_qtensor());
  const QTensorProto& qtensor_proto = proto.qtensor();

  EXPECT_EQ(qtensor_proto.precision(), qtensor->precision());
  EXPECT_EQ(qtensor_proto.scale(), qtensor->scale());
  EXPECT_EQ(qtensor_proto.bias(), qtensor->bias());
  EXPECT_EQ(qtensor_proto.is_signed(), qtensor->is_signed());

  Blob new_blob;
  DeserializeBlob(serialized, &new_blob);
  EXPECT_TRUE(new_blob.IsType<QTensor<CPUContext>>());
  const QTensor<CPUContext>& new_qtensor = blob.Get<QTensor<CPUContext>>();
  EXPECT_EQ(new_qtensor.ndim(), 2);
  EXPECT_EQ(new_qtensor.dim32(0), 2);
  EXPECT_EQ(new_qtensor.dim32(1), 3);
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 5; ++j) {
      EXPECT_EQ(qtensor->GetBitAtIndex(j, i), new_qtensor.GetBitAtIndex(j, i));
    }
  }
}

using StringMap = std::vector<std::pair<string, string>>;

class VectorCursor : public db::Cursor {
 public:
  explicit VectorCursor(StringMap* data) : data_(data) {
    pos_ = 0;
  }
  ~VectorCursor() {}
  void Seek(const string& /* unused */) override {}
  void SeekToFirst() override {}
  void Next() override {
    ++pos_;
  }
  string key() override {
    return (*data_)[pos_].first;
  }
  string value() override {
    return (*data_)[pos_].second;
  }
  bool Valid() override {
    return pos_ < data_->size();
  }

 private:
  StringMap* data_ = nullptr;
  size_t pos_ = 0;
};

class VectorDB : public db::DB {
 public:
  VectorDB(const string& source, db::Mode mode)
      : DB(source, mode), name_(source) {}
  ~VectorDB() {
    data_.erase(name_);
  }
  void Close() override {}
  std::unique_ptr<db::Cursor> NewCursor() override {
    return make_unique<VectorCursor>(getData());
  }
  std::unique_ptr<db::Transaction> NewTransaction() override {
    CAFFE_THROW("Not implemented");
  }
  static void registerData(const string& name, StringMap&& data) {
    std::lock_guard<std::mutex> guard(dataRegistryMutex_);
    data_[name] = std::move(data);
  }

 private:
  StringMap* getData() {
    auto it = data_.find(name_);
    CAFFE_ENFORCE(it != data_.end(), "Can't find ", name_);
    return &(it->second);
  }

 private:
  string name_;
  static std::mutex dataRegistryMutex_;
  static std::map<string, StringMap> data_;
};

std::mutex VectorDB::dataRegistryMutex_;
std::map<string, StringMap> VectorDB::data_;

REGISTER_CAFFE2_DB(vector_db, VectorDB);

template <typename TypeParam>
class TypedTensorTest : public ::testing::Test {};
typedef ::testing::
    Types<float, bool, double, int, int8_t, int16_t, uint8_t, uint16_t, int64_t>
        TensorDataTypes;
TYPED_TEST_CASE(TypedTensorTest, TensorDataTypes);

TYPED_TEST(TypedTensorTest, BigTensorSerialization) {
  int64_t d1 = 2;
  int64_t d2 = FLAGS_caffe2_test_big_tensor_size
      ? FLAGS_caffe2_test_big_tensor_size / d1
      : static_cast<int64_t>(std::numeric_limits<int>::max()) + 1;
  int64_t size = d1 * d2;
  string db_source = (string)std::tmpnam(nullptr);
  VLOG(1) << "db_source: " << db_source;

  {
    VLOG(1) << "Test begin";
    Blob blob;
    Tensor* tensor = BlobGetMutableTensor(&blob, CPU);
    VLOG(1) << "Allocating blob";
    tensor->Resize(d1, d2);
    auto mutableData = tensor->mutable_data<TypeParam>();
    VLOG(1) << "Filling out the blob";
    for (int64_t i = 0; i < size; ++i) {
      mutableData[i] = static_cast<TypeParam>(i);
    }
    StringMap data;
    std::mutex mutex;
    /*auto db = CreateDB("minidb", db_source, WRITE);*/
    auto acceptor = [&](const std::string& key, const std::string& value) {
      std::lock_guard<std::mutex> guard(mutex);
      /*db->NewTransaction()->Put(key, value);*/
      data.emplace_back(key, value);
    };
    SerializeBlob(blob, "test", acceptor);
    VectorDB::registerData(db_source, std::move(data));
    VLOG(1) << "finished writing to DB";
  }

  {
    DeviceOption option;
    option.set_device_type(PROTO_CPU);
    Argument db_type_arg = MakeArgument<string>("db_type", "vector_db");
    Argument absolute_path_arg = MakeArgument<bool>("absolute_path", true);
    Argument db_source_arg = MakeArgument<string>("db", db_source);
    auto op_def = CreateOperatorDef(
        "Load",
        "",
        std::vector<string>{},
        std::vector<string>({"test"}),
        std::vector<Argument>{db_type_arg, db_source_arg, absolute_path_arg},
        option,
        "DUMMY_ENGINE");
    Workspace ws;
    auto load_op = CreateOperator(op_def, &ws);
    EXPECT_TRUE(load_op != nullptr);
    VLOG(1) << "Running operator";

    load_op->Run();
    VLOG(1) << "Reading blob from workspace";
    auto new_blob = ws.GetBlob("test");
    EXPECT_TRUE(BlobIsTensorType(*new_blob, CPU));
    const auto& new_tensor = new_blob->Get<TensorCPU>();

    EXPECT_EQ(new_tensor.dim(), d1);
    EXPECT_EQ(new_tensor.size(0), d1);
    EXPECT_EQ(new_tensor.size(1), d2);
    for (int64_t i = 0; i < size; ++i) {
      EXPECT_EQ(static_cast<TypeParam>(i), new_tensor.data<TypeParam>()[i]);
    }
  }
}

struct DummyType {
  /* This struct is used to test serialization and deserialization of huge
   * blobs, that are not tensors.
   */

  /* implicit */ DummyType(int n_chunks_init = 0) : n_chunks(n_chunks_init) {}
  std::string serialize(const std::string& name, const int32_t chunk_id) const {
    BlobProto blobProto;
    blobProto.set_name(name);
    blobProto.set_type("DummyType");
    std::string content("");
    blobProto.set_content(content);
    blobProto.set_content_num_chunks(n_chunks);
    blobProto.set_content_chunk_id(chunk_id);
    return blobProto.SerializeAsString();
  }
  void deserialize(const BlobProto& /* unused */) {
    ++n_chunks;
  }
  int n_chunks;
};

class DummyTypeSerializer : public BlobSerializerBase {
 public:
  DummyTypeSerializer() {}
  ~DummyTypeSerializer() {}
  void Serialize(
      const void* pointer,
      TypeMeta typeMeta,
      const string& name,
      SerializationAcceptor acceptor) override {
    CAFFE_ENFORCE(typeMeta.Match<DummyType>());
    const auto& container = *static_cast<const DummyType*>(pointer);
    for (int k = 0; k < container.n_chunks; ++k) {
      std::string serialized_chunk = container.serialize(name, k);
      acceptor(c10::str(name, kChunkIdSeparator, k), serialized_chunk);
    }
  }
};

class DummyTypeDeserializer : public BlobDeserializerBase {
 public:
  void Deserialize(const BlobProto& proto, Blob* blob) override {
    auto* container = blob->GetMutable<DummyType>();
    container->deserialize(proto);
  }
};
}

CAFFE_KNOWN_TYPE(DummyType);

namespace {
REGISTER_BLOB_SERIALIZER((TypeMeta::Id<DummyType>()), DummyTypeSerializer);
C10_REGISTER_TYPED_CLASS(
    BlobDeserializerRegistry,
    "DummyType",
    DummyTypeDeserializer);

TEST(ContentChunks, Serialization) {
  string db_source = (string)std::tmpnam(nullptr);
  VLOG(1) << "db_source: " << db_source;

  {
    VLOG(1) << "Test begin";
    Blob blob;
    DummyType* container = blob.GetMutable<DummyType>();
    VLOG(1) << "Allocating blob";
    container->n_chunks = 10;
    VLOG(1) << "Filling out the blob";
    StringMap data;
    std::mutex mutex;
    auto acceptor = [&](const std::string& key, const std::string& value) {
      std::lock_guard<std::mutex> guard(mutex);
      data.emplace_back(key, value);
    };
    SerializeBlob(blob, "test", acceptor);
    VectorDB::registerData(db_source, std::move(data));
    VLOG(1) << "finished writing to DB";
  }

  {
    DeviceOption option;
    option.set_device_type(PROTO_CPU);
    Argument db_type_arg = MakeArgument<string>("db_type", "vector_db");
    Argument absolute_path_arg = MakeArgument<bool>("absolute_path", true);
    Argument db_source_arg = MakeArgument<string>("db", db_source);
    auto op_def = CreateOperatorDef(
        "Load",
        "",
        std::vector<string>{},
        std::vector<string>({"test"}),
        std::vector<Argument>{db_type_arg, db_source_arg, absolute_path_arg},
        option,
        "DUMMY_ENGINE");
    Workspace ws;
    auto load_op = CreateOperator(op_def, &ws);
    EXPECT_TRUE(load_op != nullptr);
    VLOG(1) << "Running operator";

    load_op->Run();
    VLOG(1) << "Reading blob from workspace";
    auto new_blob = ws.GetBlob("test");
    EXPECT_TRUE(new_blob->IsType<DummyType>());
    const auto& container = new_blob->Get<DummyType>();
    EXPECT_EQ(container.n_chunks, 10);
  }
}

TEST(CustomChunkSize, BigTensorSerialization) {
  int64_t d1 = 2;
  int64_t d2 = FLAGS_caffe2_test_big_tensor_size
      ? FLAGS_caffe2_test_big_tensor_size / d1
      : static_cast<int64_t>(std::numeric_limits<int>::max()) + 1;
  int64_t size = d1 * d2;

  Blob blob;
  TensorCPU* tensor = BlobGetMutableTensor(&blob, CPU);
  tensor->Resize(d1, d2);
  tensor->mutable_data<float>();
  std::mutex mutex;
  int counter = 0;
  auto acceptor = [&](const std::string& /*key*/,
                      const std::string& /*value*/) {
    std::lock_guard<std::mutex> guard(mutex);
    counter++;
  };
  SerializeBlob(blob, "test", acceptor, size);
  EXPECT_EQ(counter, 1);

  counter = 0;
  SerializeBlob(blob, "test", acceptor, (size / 2) + 1);
  EXPECT_EQ(counter, 2);

  counter = 0;
  SerializeBlob(blob, "test", acceptor, kNoChunking);
  EXPECT_EQ(counter, 1);
}

TEST(QTensor, QTensorSizingTest) {
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  QTensor<CPUContext> qtensor(dims, 3);
  EXPECT_TRUE(qtensor.mutable_data() != nullptr);
  EXPECT_EQ(qtensor.nbytes(), 12);
  EXPECT_EQ(qtensor.size(), 30);
}

TEST(BlobTest, CastingMessage) {
  Blob b;
  b.GetMutable<BlobTestFoo>();
  b.Get<BlobTestFoo>();
  try {
    b.Get<BlobTestBar>();
    FAIL() << "Should have thrown";
  } catch (const EnforceNotMet& e) {
    string msg = e.what_without_backtrace();
    LOG(INFO) << msg;
    EXPECT_NE(msg.find("BlobTestFoo"), std::string::npos) << msg;
    EXPECT_NE(msg.find("BlobTestBar"), std::string::npos) << msg;
  }
}

TEST(TensorConstruction, UninitializedCopyTest) {
  Tensor x(CPU);
  Tensor y(x, CPU);
  Tensor z = x.Clone();
  EXPECT_FALSE(x.dtype_initialized());
  EXPECT_FALSE(y.dtype_initialized());
  LOG(INFO) << "z.size()" << z.numel();
  EXPECT_FALSE(z.dtype_initialized());
}

TEST(TensorConstruction, CopyConstructorTest) {
  Tensor x(CPU);
  x.Resize(5);
  x.mutable_data<float>()[0] = 1;
  Tensor y = x.Clone();
  Tensor z(x, CPU);

  EXPECT_EQ(*x.data<float>(), 1);
  EXPECT_EQ(*y.data<float>(), 1);
  EXPECT_EQ(*z.data<float>(), 1);
  x.mutable_data<float>()[0] = 5;
  EXPECT_EQ(*x.data<float>(), 5);
  EXPECT_EQ(*y.data<float>(), 1);
  EXPECT_EQ(*z.data<float>(), 1);
}

TEST(TensorConstruction, MoveAssignmentOpTest) {
  Tensor x(CPU);
  x.Resize(5);
  x.mutable_data<float>()[0] = 1;
  Tensor y(CPU);
  y = std::move(x);

  EXPECT_EQ(*y.data<float>(), 1);
}

TEST(TensorSerialization, MistakenlySerializingDtypeUninitializedTensor) {
  // This test preserves a legacy behavior that dtype-unitialized tensors can
  // go through serialization. We want to kill this behavior - when it's done,
  // remove this test
  Blob blob;
  Tensor* x = BlobGetMutableTensor(&blob, CPU);
  x->Resize(0);
  string output;
  SerializeBlob(
      blob,
      "foo",
      [&output](const string& /*blobName*/, const std::string& data) {
        output = data;
      });
  BlobProto b;
  CHECK(b.ParseFromString(output));
  LOG(INFO) << "serialized proto: " << b.DebugString();

  Blob new_blob;
  DeserializeBlob(output, &new_blob);
  const Tensor& new_tensor = new_blob.Get<Tensor>();
  LOG(INFO) << "tensor " << new_tensor.DebugString();
  EXPECT_FALSE(new_tensor.dtype_initialized());
  EXPECT_EQ(0, new_tensor.numel());
  EXPECT_EQ(1, new_tensor.dim());
}

} // namespace
} // namespace caffe2
