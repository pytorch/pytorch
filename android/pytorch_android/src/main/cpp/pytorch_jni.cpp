#include <pytorch_jni.h>

#include <cassert>
#include <iostream>
#include <memory>
#include <string.h>

#include <android/log.h>

#include <torch/script.h>

#define TAG "Pytorch_JNI"
#define __FILENAME__                                                           \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define LOG_PREFIX(s, ...)                                                     \
  "%s:%d %s " s, __FILENAME__, __LINE__, __FUNCTION__, __VA_ARGS__

#define ALOGV(...)                                                             \
  __android_log_print(ANDROID_LOG_VERBOSE, TAG, LOG_PREFIX(__VA_ARGS__))
#define ALOGD(...)                                                             \
  __android_log_print(ANDROID_LOG_DEBUG, TAG, LOG_PREFIX(__VA_ARGS__))
#define ALOGE(...)                                                             \
  __android_log_print(ANDROID_LOG_ERROR, TAG, LOG_PREFIX(__VA_ARGS__))
#define ALOGW(...)                                                             \
  __android_log_print(ANDROID_LOG_WARN, TAG, LOG_PREFIX(__VA_ARGS__))
#define ALOGI(...)                                                             \
  __android_log_print(ANDROID_LOG_INFO, TAG, LOG_PREFIX(__VA_ARGS__))

namespace pytorch_jni {

int64_t JLong::toNative() const {
  static const auto method = javaClassStatic()->getMethod<jlong()>("longValue");
  return method(self());
}

facebook::jni::local_ref<JLong> JLong::fromNative(const int64_t value) {
  return newInstance(value);
}

double JDouble::toNative() const {
  static const auto method = javaClassStatic()->getMethod<jdouble()>("doubleValue");
  return method(self());
}

facebook::jni::local_ref<JDouble> JDouble::fromNative(const double value) {
  return newInstance(value);
}

facebook::jni::local_ref<JTensor> JTensor::newJTensor(
    facebook::jni::alias_ref<facebook::jni::JByteBuffer> jBuffer,
    facebook::jni::alias_ref<jintArray> jDims, jint typeCode) {
  static auto jMethodNewTensor =
      JTensor::javaClassStatic()
          ->getStaticMethod<facebook::jni::local_ref<JTensor>(
              facebook::jni::alias_ref<facebook::jni::JByteBuffer>,
              facebook::jni::alias_ref<jintArray>, jint)>("nativeNewTensor");
  return jMethodNewTensor(JTensor::javaClassStatic(), jBuffer, jDims, typeCode);
}

facebook::jni::local_ref<JTensor>
JTensor::newJTensorFromAtTensor(const at::Tensor &tensor) {
  const auto scalarType = tensor.scalar_type();
  int typeCode = 0;
  if (at::kFloat == scalarType) {
    typeCode = JTensor::kTensorTypeCodeFloat32;
  } else if (at::kInt == scalarType) {
    typeCode = JTensor::kTensorTypeCodeInt32;
  } else if (at::kByte == scalarType) {
    typeCode = JTensor::kTensorTypeCodeByte;
  } else {
    facebook::jni::throwNewJavaException(
        PytorchJni::kJavaIllegalArgumentException,
        "at::Tensor scalar type is not supported on java side");
  }

  const auto &tensorDims = tensor.sizes();
  std::vector<int> tensorDimsVec;
  for (const auto &dim : tensorDims) {
    tensorDimsVec.push_back(dim);
  }

  facebook::jni::local_ref<jintArray> jTensorDims =
      facebook::jni::make_int_array(tensorDimsVec.size());

  jTensorDims->setRegion(0, tensorDimsVec.size(), tensorDimsVec.data());

  facebook::jni::local_ref<facebook::jni::JByteBuffer> jTensorBuffer =
      facebook::jni::JByteBuffer::allocateDirect(tensor.nbytes());
  jTensorBuffer->order(facebook::jni::JByteOrder::nativeOrder());
  std::memcpy(jTensorBuffer->getDirectBytes(), tensor.storage().data(),
              tensor.nbytes());
  return JTensor::newJTensor(jTensorBuffer, jTensorDims, typeCode);
}

at::Tensor JTensor::newAtTensorFromJTensor(
    facebook::jni::alias_ref<pytorch_jni::JTensor> jtensor) {
  static const auto typeCodeMethod =
      JTensor::javaClassStatic()->getMethod<jint()>("getTypeCode");
  jint typeCode = typeCodeMethod(jtensor);

  static const auto dimsField =
      JTensor::javaClassStatic()->getField<jintArray>("dims");
  auto jdims = jtensor->getFieldValue(dimsField);

  static auto dataBufferMethod =
      JTensor::javaClassStatic()
          ->getMethod<facebook::jni::local_ref<
              facebook::jni::JBuffer::javaobject>()>("getRawDataBuffer");
  facebook::jni::local_ref<facebook::jni::JBuffer> jbuffer =
      dataBufferMethod(jtensor);
  return PytorchJni::newAtTensor(jbuffer, jdims, typeCode);
}

facebook::jni::local_ref<JIValue>
JIValue::newJIValueFromAtIValue(const at::IValue &ivalue) {
  if (ivalue.isNone()) {
    static auto jMethodOptionalNull =
        JIValue::javaClassStatic()
            ->getStaticMethod<facebook::jni::local_ref<JIValue>()>("optionalNull");

    return jMethodOptionalNull(JIValue::javaClassStatic());
  } else if (ivalue.isTensor()) {
    static auto jMethodTensor =
        JIValue::javaClassStatic()
            ->getStaticMethod<facebook::jni::local_ref<JIValue>(facebook::jni::local_ref<JTensor>)>("tensor");
    return jMethodTensor(
        JIValue::javaClassStatic(),
        JTensor::newJTensorFromAtTensor(ivalue.toTensor()));
  } else if (ivalue.isBool()) {
    static auto jMethodBool =
        JIValue::javaClassStatic()
            ->getStaticMethod<facebook::jni::local_ref<JIValue>(jboolean)>("bool");
    return jMethodBool(
        JIValue::javaClassStatic(),
        ivalue.toBool());
  } else if (ivalue.isInt()) {
    static auto jMethodInt =
        JIValue::javaClassStatic()
            ->getStaticMethod<facebook::jni::local_ref<JIValue>(jlong)>("long64");
    return jMethodInt(
        JIValue::javaClassStatic(),
        ivalue.toInt());
  } else if (ivalue.isDouble()) {
    static auto jMethodDouble =
        JIValue::javaClassStatic()
            ->getStaticMethod<facebook::jni::local_ref<JIValue>(jdouble)>("double64");
    return jMethodDouble(
        JIValue::javaClassStatic(),
        ivalue.toDouble());
  } else if (ivalue.isTuple()) {
    auto elementsVec = ivalue.toTuple()->elements();
    static auto jMethodTupleArr =
        JIValue::javaClassStatic()
            ->getStaticMethod<
                facebook::jni::local_ref<JIValue>(
                    facebook::jni::alias_ref<facebook::jni::JArrayClass<JIValue::javaobject>::javaobject>
                        )>("tuple");
    auto jElementsArray = facebook::jni::JArrayClass<JIValue::javaobject>::newArray(elementsVec.size());
    auto index = 0;
    for (const auto& e : elementsVec) {
      (*jElementsArray)[index++] = JIValue::newJIValueFromAtIValue(e);
    }
    return jMethodTupleArr(
        JIValue::javaClassStatic(),
        jElementsArray);
  } else if (ivalue.isGenericList()) {
    //TODO: ??? ivalue.isTensorList(), ivalue.isIntList(), ivalue.isBoolList()
    auto list = ivalue.toGenericList();
    static auto jMethodListArr =
        JIValue::javaClassStatic()
            ->getStaticMethod<
                facebook::jni::local_ref<JIValue>(
                    facebook::jni::alias_ref<facebook::jni::JArrayClass<JIValue::javaobject>::javaobject>
                )>("list");
    auto jArray = facebook::jni::JArrayClass<JIValue::javaobject>::newArray(list.size());
    auto index = 0;
    for (const auto& e : list) {
      (*jArray)[index++] = JIValue::newJIValueFromAtIValue(e);
    }
    return jMethodListArr(JIValue::javaClassStatic(), jArray);
  } else if (ivalue.isGenericDict()) {
    auto dict = ivalue.toGenericDict();
    const auto keyType = dict._keyType();

    if (!keyType) {
      assert(false);
    }

    const auto keyTypeKind = keyType.value()->kind();
    if (c10::TypeKind::StringType == keyTypeKind) {
      static auto jMethodDictStringKey =
          JIValue::javaClassStatic()
              ->getStaticMethod<
                  facebook::jni::local_ref<JIValue>(
                      facebook::jni::alias_ref<
                          facebook::jni::JMap<
                              facebook::jni::alias_ref<facebook::jni::JString::javaobject>,
                              facebook::jni::alias_ref<JIValue::javaobject>
                          >
                      >
                  )>("dictStringKey");

      auto jmap =
          JHashMap<
              facebook::jni::alias_ref<facebook::jni::JString::javaobject>,
              facebook::jni::alias_ref<JIValue::javaobject>
          >::create();
      for (auto& pair : dict) {
        jmap->put(
            facebook::jni::make_jstring(pair.key().toString()->string()),
            JIValue::newJIValueFromAtIValue(pair.value())
        );
      }
      return jMethodDictStringKey(JIValue::javaClassStatic(), jmap);
    } else if (c10::TypeKind::IntType == keyTypeKind) {
      static auto jMethodDictLongKey =
          JIValue::javaClassStatic()
              ->getStaticMethod<
                  facebook::jni::local_ref<JIValue>(
                      facebook::jni::alias_ref<
                          facebook::jni::JMap<
                              facebook::jni::alias_ref<JLong::javaobject>,
                              facebook::jni::alias_ref<JIValue::javaobject>
                          >
                      >
                  )>("dictLongKey");
      auto jmap =
          JHashMap<
              facebook::jni::alias_ref<JLong::javaobject>,
              facebook::jni::alias_ref<JIValue::javaobject>
          >::create();
      for (auto& pair : dict) {
        jmap->put(
            JLong::fromNative(pair.key().toInt()),
            JIValue::newJIValueFromAtIValue(pair.value()));
      }
      return jMethodDictLongKey(JIValue::javaClassStatic(), jmap);
    } else if (c10::TypeKind::FloatType == keyTypeKind) {
      static auto jMethodDictDoubleKey =
          JIValue::javaClassStatic()
              ->getStaticMethod<
                  facebook::jni::local_ref<JIValue>(
                      facebook::jni::alias_ref<
                          facebook::jni::JMap<
                              facebook::jni::alias_ref<JDouble::javaobject>,
                              facebook::jni::alias_ref<JIValue::javaobject>
                          >
                      >
                  )>("dictDoubleKey");
      auto jmap =
          JHashMap<
              facebook::jni::alias_ref<JDouble::javaobject>,
              facebook::jni::alias_ref<JIValue::javaobject>
          >::create();
      for (auto& pair : dict) {
        jmap->put(
            JDouble::fromNative(pair.key().toDouble()),
            JIValue::newJIValueFromAtIValue(pair.value()));
      }
      return jMethodDictDoubleKey(JIValue::javaClassStatic(), jmap);
    }
    assert(false);
  }

  facebook::jni::local_ref<JIValue> tmp = JIValue::newInstance();
  return tmp;
}

at::IValue
JIValue::JIValueToAtIValue(facebook::jni::alias_ref<JIValue> jivalue) {
  static const auto typeCodeField =
      JIValue::javaClassStatic()->getField<jint>("typeCode");
  const auto typeCode = jivalue->getFieldValue(typeCodeField);

  if (JIValue::kTypeCodeTensor == typeCode) {
    static const auto tensorField =
        JIValue::javaClassStatic()->getField<JTensor::javaobject>("mTensor");
    facebook::jni::local_ref<JTensor> jtensor =
        jivalue->getFieldValue(tensorField);
    return JTensor::newAtTensorFromJTensor(jtensor);
  } else if (JIValue::kTypeCodeBool == typeCode) {
    static const auto booleanField =
        JIValue::javaClassStatic()->getField<jboolean>("mBool");
    bool b = jivalue->getFieldValue(booleanField);
    return at::IValue{b};
  } else if (JIValue::kTypeCodeLong64 == typeCode) {
    static const auto longField =
        JIValue::javaClassStatic()->getField<jlong>("mLong");
    auto i = jivalue->getFieldValue(longField);
    return at::IValue{i};
  } else if (JIValue::kTypeCodeDouble64 == typeCode) {
    static const auto doubleField =
        JIValue::javaClassStatic()->getField<jdouble>("mDouble");
    auto f = jivalue->getFieldValue(doubleField);
    return at::IValue{f};
  } else if (JIValue::kTypeCodeTuple == typeCode) {
    static const auto tupleField =
        JIValue::javaClassStatic()
            ->getField<
                facebook::jni::JArrayClass<JIValue::javaobject>::javaobject>(
                "mTuple");
    auto jarray = jivalue->getFieldValue(tupleField);
    size_t n = jarray->size();

    std::vector<at::IValue> elements;
    elements.reserve(n);
    std::vector<c10::TypePtr> types;
    types.reserve(n);
    for (auto i = 0; i < n; ++i) {
      auto jivalue_element = jarray->getElement(i);
      auto element = JIValue::JIValueToAtIValue(jivalue_element);
      c10::TypePtr typePtr = c10::attemptToRecoverType(element);
      elements.push_back(std::move(element));
      types.push_back(std::move(typePtr));
    }
    return c10::ivalue::Tuple::create(std::move(elements),
                                      c10::TupleType::create(std::move(types)));
  } else if (JIValue::kTypeCodeList == typeCode) {
    static const auto listField =
        JIValue::javaClassStatic()
            ->getField<
                facebook::jni::JArrayClass<JIValue::javaobject>::javaobject>(
                "mList");
    auto jarray = jivalue->getFieldValue(listField);
    size_t n = jarray->size();
    if (n == 0) {
      return at::IValue{
          c10::impl::GenericList(c10::impl::deprecatedUntypedList())};
    }

    auto jivalue_first_element = jarray->getElement(0);
    auto first_element = JIValue::JIValueToAtIValue(jivalue_first_element);
    c10::TypePtr typePtr = c10::attemptToRecoverType(first_element);
    c10::impl::GenericList list{typePtr};
    list.reserve(n);
    list.push_back(first_element);
    for (auto i = 1; i < n; ++i) {
      auto jivalue_element = jarray->getElement(i);
      auto element = JIValue::JIValueToAtIValue(jivalue_element);
      list.push_back(element);
    }
    return at::IValue{list};
  } else if (JIValue::kTypeCodeOptional == typeCode) {
    static const auto optionalValueField =
        JIValue::javaClassStatic()->getField<JIValue::javaobject>("mOptionalValue");
    auto jOptionalValue = jivalue->getFieldValue(optionalValueField);
    if (jOptionalValue) {
      return JIValue::JIValueToAtIValue(jOptionalValue);
    }
    return at::IValue{};
  } else if (JIValue::kTypeCodeDictStringKey == typeCode) {
    static const auto mapStringKeyField =
        JIValue::javaClassStatic()
            ->getField<
                facebook::jni::JMap<jstring, JIValue::javaobject>::javaobject>(
                "mMapStringKey");
    auto jmap = jivalue->getFieldValue(mapStringKeyField);
    auto it = jmap->begin();
    if (it == jmap->end()) {
      return at::IValue{
          c10::impl::GenericDict(c10::impl::deprecatedUntypedDict())};
    }

    auto firstEntryValue = JIValue::JIValueToAtIValue(it->second);
    c10::TypePtr typePtr = c10::attemptToRecoverType(firstEntryValue);
    c10::impl::GenericDict dict{c10::StringType::get(), typePtr};
    dict.insert(it->first->toStdString(), firstEntryValue);
    it++;
    for (; it != jmap->end(); it++) {
      dict.insert(it->first->toStdString(),
                  JIValue::JIValueToAtIValue(it->second));
    }

    return at::IValue{dict};
  } else if (JIValue::kTypeCodeLongIntKey == typeCode) {
    static const auto mapLongKeyField =
        JIValue::javaClassStatic()
            ->getField<facebook::jni::JMap<facebook::jni::JLong::javaobject,
                JIValue::javaobject>::javaobject>(
                "mMapLongKey");
    auto jmap = jivalue->getFieldValue(mapLongKeyField);
    auto it = jmap->begin();
    if (it == jmap->end()) {
      return at::IValue{
          c10::impl::GenericDict(c10::impl::deprecatedUntypedDict())};
    }

    auto firstEntryValue = JIValue::JIValueToAtIValue(it->second);
    c10::TypePtr typePtr = c10::attemptToRecoverType(firstEntryValue);
    c10::impl::GenericDict dict{c10::IntType::get(), typePtr};
    dict.insert(it->first->longValue(), firstEntryValue);
    it++;
    for (; it != jmap->end(); it++) {
      dict.insert(it->first->longValue(),
                  JIValue::JIValueToAtIValue(it->second));
    }

    return at::IValue{dict};
  } else if (JIValue::kTypeCodeDictDoubleKey == typeCode) {
    static const auto mapDoubleKeyField =
        JIValue::javaClassStatic()
            ->getField<facebook::jni::JMap<facebook::jni::JDouble::javaobject,
                JIValue::javaobject>::javaobject>(
                "mMapDoubleKey");
    auto jmap = jivalue->getFieldValue(mapDoubleKeyField);
    auto it = jmap->begin();
    if (it == jmap->end()) {
      return at::IValue{
          c10::impl::GenericDict(c10::impl::deprecatedUntypedDict())};
    }

    auto firstEntryValue = JIValue::JIValueToAtIValue(it->second);
    c10::TypePtr typePtr = c10::attemptToRecoverType(firstEntryValue);
    c10::impl::GenericDict dict{c10::FloatType::get(), typePtr};
    dict.insert(it->first->doubleValue(), firstEntryValue);
    it++;
    for (; it != jmap->end(); it++) {
      dict.insert(it->first->doubleValue(),
                  JIValue::JIValueToAtIValue(it->second));
    }
    return at::IValue{dict};
  }

  facebook::jni::throwNewJavaException(
      PytorchJni::kJavaIllegalArgumentException, "Unknown typeCode");
}

void PytorchJni::registerNatives() {
  registerHybrid({
      makeNativeMethod("initHybrid", PytorchJni::initHybrid),
      makeNativeMethod("run", PytorchJni::run),
      makeNativeMethod("forward", PytorchJni::forward),
  });
}

at::Tensor PytorchJni::newAtTensor(
    facebook::jni::alias_ref<facebook::jni::JBuffer> inputData,
    facebook::jni::alias_ref<jintArray> inputDims,
    jint typeCode) {
  const auto inputDimsRank = inputDims->size();
  const auto inputDimsArr = inputDims->getRegion(0, inputDimsRank);
  std::vector<int64_t> inputDimsVec;
  auto inputNumel = 1;
  for (auto i = 0; i < inputDimsRank; ++i) {
    inputDimsVec.push_back(inputDimsArr[i]);
    inputNumel *= inputDimsArr[i];
  }
  JNIEnv *jni = facebook::jni::Environment::current();
  caffe2::TypeMeta inputTypeMeta{};
  const auto directBufferCapacity =
      jni->GetDirectBufferCapacity(inputData.get());
  int inputDataElementSizeBytes = 0;
  if (JTensor::kTensorTypeCodeFloat32 == typeCode) {
    inputDataElementSizeBytes = 4;
    inputTypeMeta = caffe2::TypeMeta::Make<float>();
  } else if (JTensor::kTensorTypeCodeInt32 == typeCode) {
    inputDataElementSizeBytes = 4;
    inputTypeMeta = caffe2::TypeMeta::Make<int>();
  } else if (JTensor::kTensorTypeCodeByte == typeCode) {
    inputDataElementSizeBytes = 1;
    inputTypeMeta = caffe2::TypeMeta::Make<uint8_t>();
  } else {
    facebook::jni::throwNewJavaException(
        PytorchJni::kJavaIllegalArgumentException, "Unknown typeCode");
  }

  const auto inputDataCapacity =
      directBufferCapacity * inputDataElementSizeBytes;
  if (inputDataCapacity != (inputNumel * inputDataElementSizeBytes)) {
    facebook::jni::throwNewJavaException(
        PytorchJni::kJavaIllegalArgumentException,
        "Tensor dimensions(elements number:%d, element byte size:%d, total "
        "bytes:%d) inconsistent with buffer capacity(%d)",
        inputNumel, inputDataElementSizeBytes,
        inputNumel * inputDataElementSizeBytes, inputDataCapacity);
  }

  at::Tensor inputTensor = torch::empty(torch::IntArrayRef(inputDimsVec));
  inputTensor.unsafeGetTensorImpl()->ShareExternalPointer(
      {jni->GetDirectBufferAddress(inputData.get()), at::DeviceType::CPU},
      inputTypeMeta, inputDataCapacity);
  return inputTensor;
}

facebook::jni::local_ref<PytorchJni::jhybriddata>
PytorchJni::initHybrid(facebook::jni::alias_ref<jclass>,
                       facebook::jni::alias_ref<jstring> modelPath) {
  return makeCxxInstance(modelPath);
}

facebook::jni::local_ref<JTensor>
PytorchJni::run(facebook::jni::alias_ref<facebook::jni::JBuffer> inputData,
                facebook::jni::alias_ref<jintArray> inputDims, jint typeCode) {

  at::Tensor inputTensor =
      PytorchJni::newAtTensor(inputData, inputDims, typeCode);
  at::Tensor outputTensor =
      module_.forward({std::move(inputTensor)}).toTensor();

  return JTensor::newJTensorFromAtTensor(outputTensor);
}

facebook::jni::local_ref<JIValue>
PytorchJni::forward(
    facebook::jni::alias_ref<facebook::jni::JArrayClass<JIValue::javaobject>::javaobject> jinputs) {
  std::vector<at::IValue> inputs{};
  size_t n = jinputs->size();
  inputs.reserve(n);
  for(size_t i = 0; i < n; i++) {
    at::IValue atIValue = JIValue::JIValueToAtIValue(jinputs->getElement(i));
    inputs.push_back(std::move(atIValue));
  }
  auto output = module_.forward(std::move(inputs));
  return JIValue::newJIValueFromAtIValue(output);
}

} // namespace pytorch_jni