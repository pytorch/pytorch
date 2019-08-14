#pragma once

#include <string>

#include <fbjni/ByteBuffer.h>
#include <fbjni/fbjni.h>

#include <torch/script.h>

namespace pytorch_jni {

template <typename K = jobject, typename V = jobject>
struct JHashMap : facebook::jni::JavaClass<JHashMap<K, V>, facebook::jni::JMap<K, V>> {
  constexpr static auto kJavaDescriptor = "Ljava/util/HashMap;";

  using Super = facebook::jni::JavaClass<JHashMap<K, V>, facebook::jni::JMap<K, V>>;

  static facebook::jni::local_ref<JHashMap<K, V>> create() {
    return Super::newInstance();
  }

  void put(
      facebook::jni::alias_ref<facebook::jni::JObject::javaobject> key,
      facebook::jni::alias_ref<facebook::jni::JObject::javaobject> value) {
    static auto putMethod =
        Super::javaClassStatic()->template getMethod<
            facebook::jni::alias_ref<facebook::jni::JObject::javaobject>(
                facebook::jni::alias_ref<facebook::jni::JObject::javaobject>,
                facebook::jni::alias_ref<facebook::jni::JObject::javaobject>
            )
        >("put");
    putMethod(Super::self(), key, value);
  }
};

class JLong : public facebook::jni::JavaClass<JLong> {
 public:
    static constexpr auto kJavaDescriptor = "Ljava/lang/Long;";
    int64_t toNative() const;
    static facebook::jni::local_ref<JLong> fromNative(const int64_t value);
};

class JDouble : public facebook::jni::JavaClass<JDouble> {
 public:
    static constexpr auto kJavaDescriptor = "Ljava/lang/Double;";
    double toNative() const;
    static facebook::jni::local_ref<JDouble> fromNative(const double value);
};

class JTensor : public facebook::jni::JavaClass<JTensor> {
public:
  constexpr static const char *kJavaDescriptor =
      "Lcom/facebook/pytorch/Tensor;";
  constexpr static int kTensorTypeCodeFloat32 = 0xF10A732;
  constexpr static int kTensorTypeCodeInt32 = 0x14732;
  constexpr static int kTensorTypeCodeByte = 0x5781460;

  static facebook::jni::local_ref<JTensor>
      newJTensor(facebook::jni::alias_ref<facebook::jni::JByteBuffer>,
                 facebook::jni::alias_ref<jintArray>, jint);

  static facebook::jni::local_ref<JTensor>
  newJTensorFromAtTensor(const at::Tensor &tensor);

  static at::Tensor
  newAtTensorFromJTensor(facebook::jni::alias_ref<JTensor> jtensor);
};

class JIValue : public facebook::jni::JavaClass<JIValue> {
public:
  constexpr static const char *kJavaDescriptor =
      "Lcom/facebook/pytorch/IValue;";

  constexpr static int kTypeCodeTensor = 1;
  constexpr static int kTypeCodeBool = 2;
  constexpr static int kTypeCodeLong64 = 3;
  constexpr static int kTypeCodeDouble64 = 4;
  constexpr static int kTypeCodeTuple = 5;
  constexpr static int kTypeCodeList = 6;
  constexpr static int kTypeCodeOptional = 7;
  constexpr static int kTypeCodeDictStringKey = 8;
  constexpr static int kTypeCodeDictDoubleKey = 9;
  constexpr static int kTypeCodeLongIntKey = 10;

  static facebook::jni::local_ref<JIValue>
  newJIValueFromAtIValue(const at::IValue &iValue);

  static at::IValue
  JIValueToAtIValue(facebook::jni::alias_ref<JIValue> jivalue);
};

class PytorchJni : public facebook::jni::HybridClass<PytorchJni> {
private:
  friend HybridBase;
  torch::jit::script::Module module_;

public:
  constexpr static auto kJavaDescriptor =
      "Lcom/facebook/pytorch/PytorchScriptModule$NativePeer;";

  constexpr static auto kJavaIllegalArgumentException =
      "java/lang/IllegalArgumentException";

  static facebook::jni::local_ref<jhybriddata>
  initHybrid(facebook::jni::alias_ref<jclass>,
             facebook::jni::alias_ref<jstring> modelPath);

  PytorchJni(facebook::jni::alias_ref<jstring> modelPath)
      : module_(torch::jit::load(std::move(modelPath->toStdString()))) {}

  static void registerNatives();

  static at::Tensor
  newAtTensor(facebook::jni::alias_ref<facebook::jni::JBuffer> inputData,
              facebook::jni::alias_ref<jintArray> inputDims, jint typeCode);

  facebook::jni::local_ref<JTensor>
  run(facebook::jni::alias_ref<facebook::jni::JBuffer> inputData,
      facebook::jni::alias_ref<jintArray> inputDims, jint typeCode);

  facebook::jni::local_ref<JIValue>
  forward(facebook::jni::alias_ref<facebook::jni::JArrayClass<JIValue::javaobject>::javaobject> jinputs);
};

} // namespace pytorch_jni