#include <cassert>
#include <iostream>
#include <memory>
#include <string>

#include <fbjni/ByteBuffer.h>
#include <fbjni/fbjni.h>

#include "pytorch_jni_common.h"
#if defined(__ANDROID__)
#include <caffe2/utils/threadpool/ThreadPool.h>
#include <caffe2/utils/threadpool/ThreadPoolMobile.h>
#endif

namespace pytorch_jni {

bool Trace::is_initialized_ = false;

#if defined(TRACE_ENABLED) && defined(__ANDROID__)
Trace::fp_ATrace_beginSection Trace::ATrace_beginSection;
Trace::fp_ATrace_endSection Trace::ATrace_endSection;
#endif

void Trace::init() {
#if defined(TRACE_ENABLED) && defined(__ANDROID__)
  void* lib = dlopen("libandroid.so", RTLD_NOW || RTLD_LOCAL);
  if (lib != NULL) {
    Trace::ATrace_beginSection = reinterpret_cast<fp_ATrace_beginSection>(
        dlsym(lib, "ATrace_beginSection"));
    Trace::ATrace_endSection =
        reinterpret_cast<fp_ATrace_endSection>(dlsym(lib, "ATrace_endSection"));
  }
#endif
}

// NOTE: Codes must be kept in sync with DType.java.
// NOTE: Never serialize these, because they can change between releases.
constexpr static int kTensorDTypeUInt8 = 1;
constexpr static int kTensorDTypeInt8 = 2;
constexpr static int kTensorDTypeInt32 = 3;
constexpr static int kTensorDTypeFloat32 = 4;
constexpr static int kTensorDTypeInt64 = 5;
constexpr static int kTensorDTypeFloat64 = 6;

template <typename K = jobject, typename V = jobject>
struct JHashMap
    : facebook::jni::JavaClass<JHashMap<K, V>, facebook::jni::JMap<K, V>> {
  constexpr static auto kJavaDescriptor = "Ljava/util/HashMap;";

  using Super =
      facebook::jni::JavaClass<JHashMap<K, V>, facebook::jni::JMap<K, V>>;

  static facebook::jni::local_ref<JHashMap<K, V>> create() {
    return Super::newInstance();
  }

  void put(
      facebook::jni::alias_ref<facebook::jni::JObject::javaobject> key,
      facebook::jni::alias_ref<facebook::jni::JObject::javaobject> value) {
    static auto putMethod =
        Super::javaClassStatic()
            ->template getMethod<facebook::jni::alias_ref<
                facebook::jni::JObject::javaobject>(
                facebook::jni::alias_ref<facebook::jni::JObject::javaobject>,
                facebook::jni::alias_ref<facebook::jni::JObject::javaobject>)>(
                "put");
    putMethod(Super::self(), key, value);
  }
};

static at::Tensor newAtTensor(
    facebook::jni::alias_ref<facebook::jni::JBuffer> jbuffer,
    facebook::jni::alias_ref<jlongArray> jshape,
    jint jdtype) {
  const auto rank = jshape->size();
  const auto shapeArr = jshape->getRegion(0, rank);
  std::vector<int64_t> shapeVec{};
  shapeVec.reserve(rank);
  auto numel = 1;
  for (auto i = 0; i < rank; ++i) {
    shapeVec.push_back(shapeArr[i]);
    numel *= shapeArr[i];
  }
  JNIEnv* jni = facebook::jni::Environment::current();
  caffe2::TypeMeta typeMeta{};
  int dataElementSizeBytes = 0;
  if (kTensorDTypeFloat32 == jdtype) {
    dataElementSizeBytes = 4;
    typeMeta = caffe2::TypeMeta::Make<float>();
  } else if (kTensorDTypeInt32 == jdtype) {
    dataElementSizeBytes = 4;
    typeMeta = caffe2::TypeMeta::Make<int32_t>();
  } else if (kTensorDTypeInt8 == jdtype) {
    dataElementSizeBytes = 1;
    typeMeta = caffe2::TypeMeta::Make<int8_t>();
  } else if (kTensorDTypeUInt8 == jdtype) {
    dataElementSizeBytes = 1;
    typeMeta = caffe2::TypeMeta::Make<uint8_t>();
  } else if (kTensorDTypeFloat64 == jdtype) {
    dataElementSizeBytes = 8;
    typeMeta = caffe2::TypeMeta::Make<double>();
  } else if (kTensorDTypeInt64 == jdtype) {
    dataElementSizeBytes = 8;
    typeMeta = caffe2::TypeMeta::Make<int64_t>();
  } else {
    facebook::jni::throwNewJavaException(
        facebook::jni::gJavaLangIllegalArgumentException,
        "Unknown Tensor jdtype %d",
        jdtype);
  }
  const auto dataCapacity = jni->GetDirectBufferCapacity(jbuffer.get());
  if (dataCapacity != numel) {
    facebook::jni::throwNewJavaException(
        facebook::jni::gJavaLangIllegalArgumentException,
        "Tensor dimensions(elements number:%d, element byte size:%d, total "
        "bytes:%d) inconsistent with buffer capacity(%d)",
        numel,
        dataElementSizeBytes,
        numel * dataElementSizeBytes,
        dataCapacity);
  }
  return torch::from_blob(
      jni->GetDirectBufferAddress(jbuffer.get()),
      torch::IntArrayRef(shapeVec),
      at::TensorOptions(typeMeta));
}

class TensorHybrid : public facebook::jni::HybridClass<TensorHybrid> {
 public:
  constexpr static const char* kJavaDescriptor = "Lorg/pytorch/Tensor;";

  explicit TensorHybrid(at::Tensor tensor) : tensor_(tensor) {}

  static facebook::jni::local_ref<TensorHybrid::jhybriddata> initHybrid(
      facebook::jni::alias_ref<TensorHybrid::javaobject> jTensorThis) {
    static auto cls = TensorHybrid::javaClassStatic();
    static const auto jMethodDTypeCode = cls->getMethod<jint()>("dtypeJniCode");
    static const auto jFieldShape = cls->getField<jlongArray>("shape");
    static const auto jMethodGetDataBuffer = cls->getMethod<
        facebook::jni::local_ref<facebook::jni::JBuffer::javaobject>()>(
        "getRawDataBuffer");

    at::Tensor tensor = newAtTensor(
        jMethodGetDataBuffer(jTensorThis),
        jTensorThis->getFieldValue(jFieldShape),
        jMethodDTypeCode(jTensorThis));
    return makeCxxInstance(std::move(tensor));
  }

  static facebook::jni::local_ref<TensorHybrid::javaobject>
  newJTensorFromAtTensor(const at::Tensor& input_tensor) {
    // Java wrapper currently only supports contiguous tensors.
    at::Tensor tensor = input_tensor.is_contiguous()
      ? input_tensor
      : input_tensor.contiguous();

    const auto scalarType = tensor.scalar_type();
    int jdtype = 0;
    if (at::kFloat == scalarType) {
      jdtype = kTensorDTypeFloat32;
    } else if (at::kInt == scalarType) {
      jdtype = kTensorDTypeInt32;
    } else if (at::kByte == scalarType) {
      jdtype = kTensorDTypeUInt8;
    } else if (at::kChar == scalarType) {
      jdtype = kTensorDTypeInt8;
    } else if (at::kLong == scalarType) {
      jdtype = kTensorDTypeInt64;
    } else if (at::kDouble == scalarType) {
      jdtype = kTensorDTypeFloat64;
    } else {
      facebook::jni::throwNewJavaException(
          facebook::jni::gJavaLangIllegalArgumentException,
          "at::Tensor scalar type is not supported on java side");
    }

    const auto& tensorShape = tensor.sizes();
    std::vector<jlong> tensorShapeVec;
    for (const auto& s : tensorShape) {
      tensorShapeVec.push_back(s);
    }
    facebook::jni::local_ref<jlongArray> jTensorShape =
        facebook::jni::make_long_array(tensorShapeVec.size());
    jTensorShape->setRegion(0, tensorShapeVec.size(), tensorShapeVec.data());

    static auto cls = TensorHybrid::javaClassStatic();
    facebook::jni::local_ref<facebook::jni::JByteBuffer> jTensorBuffer =
        facebook::jni::JByteBuffer::wrapBytes(
            (uint8_t*)tensor.data_ptr(), tensor.nbytes());
    jTensorBuffer->order(facebook::jni::JByteOrder::nativeOrder());

    static const auto jMethodNewTensor =
        cls->getStaticMethod<facebook::jni::local_ref<TensorHybrid::javaobject>(
            facebook::jni::alias_ref<facebook::jni::JByteBuffer>,
            facebook::jni::alias_ref<jlongArray>,
            jint,
            facebook::jni::alias_ref<jhybriddata>)>("nativeNewTensor");
    return jMethodNewTensor(
        cls, jTensorBuffer, jTensorShape, jdtype, makeCxxInstance(tensor));
  }

  static at::Tensor newAtTensorFromJTensor(
      facebook::jni::alias_ref<TensorHybrid::javaobject> jtensor) {
    static auto cls = TensorHybrid::javaClassStatic();
    static const auto dtypeMethod = cls->getMethod<jint()>("dtypeJniCode");
    jint jdtype = dtypeMethod(jtensor);

    static const auto shapeField = cls->getField<jlongArray>("shape");
    auto jshape = jtensor->getFieldValue(shapeField);

    static auto dataBufferMethod = cls->getMethod<
        facebook::jni::local_ref<facebook::jni::JBuffer::javaobject>()>(
        "getRawDataBuffer");
    facebook::jni::local_ref<facebook::jni::JBuffer> jbuffer =
        dataBufferMethod(jtensor);
    return newAtTensor(jbuffer, jshape, jdtype);
  }

  at::Tensor tensor() const {
    return tensor_;
  }

 private:
  friend HybridBase;
  at::Tensor tensor_;
};

facebook::jni::local_ref<JIValue> JIValue::newJIValueFromAtIValue(
    const at::IValue& ivalue) {
  Trace _s{"jni::JIValue::newJIValueFromAtIValue"};
  if (ivalue.isNone()) {
    static auto jMethodOptionalNull =
        JIValue::javaClassStatic()
            ->getStaticMethod<facebook::jni::local_ref<JIValue>()>(
                "optionalNull");
    return jMethodOptionalNull(JIValue::javaClassStatic());
  } else if (ivalue.isTensor()) {
    static auto jMethodTensor =
        JIValue::javaClassStatic()
            ->getStaticMethod<facebook::jni::local_ref<JIValue>(
                facebook::jni::local_ref<TensorHybrid::javaobject>)>("from");
    return jMethodTensor(
        JIValue::javaClassStatic(),
        TensorHybrid::newJTensorFromAtTensor(ivalue.toTensor()));
  } else if (ivalue.isBool()) {
    static auto jMethodBool =
        JIValue::javaClassStatic()
            ->getStaticMethod<facebook::jni::local_ref<JIValue>(jboolean)>(
                "from");
    return jMethodBool(JIValue::javaClassStatic(), ivalue.toBool());
  } else if (ivalue.isInt()) {
    static auto jMethodInt =
        JIValue::javaClassStatic()
            ->getStaticMethod<facebook::jni::local_ref<JIValue>(jlong)>("from");
    return jMethodInt(JIValue::javaClassStatic(), ivalue.toInt());
  } else if (ivalue.isDouble()) {
    static auto jMethodDouble =
        JIValue::javaClassStatic()
            ->getStaticMethod<facebook::jni::local_ref<JIValue>(jdouble)>(
                "from");
    return jMethodDouble(JIValue::javaClassStatic(), ivalue.toDouble());
  } else if (ivalue.isString()) {
    static auto jMethodString =
        JIValue::javaClassStatic()
            ->getStaticMethod<facebook::jni::local_ref<JIValue>(
                facebook::jni::alias_ref<facebook::jni::JString::javaobject>)>(
                "from");
    return jMethodString(
        JIValue::javaClassStatic(),
        facebook::jni::make_jstring(ivalue.toStringRef()));
  } else if (ivalue.isTuple()) {
    auto elementsVec = ivalue.toTuple()->elements();
    static auto jMethodTupleArr =
        JIValue::javaClassStatic()
            ->getStaticMethod<facebook::jni::local_ref<JIValue>(
                facebook::jni::alias_ref<facebook::jni::JArrayClass<
                    JIValue::javaobject>::javaobject>)>("tupleFrom");
    auto jElementsArray =
        facebook::jni::JArrayClass<JIValue::javaobject>::newArray(
            elementsVec.size());
    auto index = 0;
    for (const auto& e : elementsVec) {
      (*jElementsArray)[index++] = JIValue::newJIValueFromAtIValue(e);
    }
    return jMethodTupleArr(JIValue::javaClassStatic(), jElementsArray);
  } else if (ivalue.isBoolList()) {
    auto list = ivalue.toBoolList();
    static auto jMethodBoolListArr =
        JIValue::javaClassStatic()
            ->getStaticMethod<facebook::jni::local_ref<JIValue>(
                facebook::jni::alias_ref<jbooleanArray>)>("listFrom");
    size_t n = list.size();
    auto jArray = facebook::jni::make_boolean_array(n);
    auto jArrayPinned = jArray->pin();
    auto index = 0;
    for (const auto& e : list) {
      jArrayPinned[index++] = e;
    }
    return jMethodBoolListArr(JIValue::javaClassStatic(), jArray);
  } else if (ivalue.isIntList()) {
    auto list = ivalue.toIntList();
    static auto jMethodLongListArr =
        JIValue::javaClassStatic()
            ->getStaticMethod<facebook::jni::local_ref<JIValue>(
                facebook::jni::alias_ref<jlongArray>)>("listFrom");
    size_t n = list.size();
    auto jArray = facebook::jni::make_long_array(n);
    auto jArrayPinned = jArray->pin();
    auto index = 0;
    for (const auto& e : list) {
      jArrayPinned[index++] = e;
    }
    return jMethodLongListArr(JIValue::javaClassStatic(), jArray);
  } else if (ivalue.isDoubleList()) {
    auto list = ivalue.toDoubleList();
    static auto jMethoDoubleListArr =
        JIValue::javaClassStatic()
            ->getStaticMethod<facebook::jni::local_ref<JIValue>(
                facebook::jni::alias_ref<jdoubleArray>)>("listFrom");
    size_t n = list.size();
    auto jArray = facebook::jni::make_double_array(n);
    auto jArrayPinned = jArray->pin();
    auto index = 0;
    for (const auto& e : list) {
      jArrayPinned[index++] = e;
    }
    return jMethoDoubleListArr(JIValue::javaClassStatic(), jArray);
  } else if (ivalue.isTensorList()) {
    auto list = ivalue.toTensorList();
    static auto jMethodTensorListArr =
        JIValue::javaClassStatic()
            ->getStaticMethod<facebook::jni::local_ref<JIValue>(
                facebook::jni::alias_ref<facebook::jni::JArrayClass<
                    TensorHybrid::javaobject>::javaobject>)>("listFrom");
    auto jArray =
        facebook::jni::JArrayClass<TensorHybrid::javaobject>::newArray(
            list.size());
    auto index = 0;
    for (const auto& e : list) {
      (*jArray)[index++] = TensorHybrid::newJTensorFromAtTensor(e);
    }
    return jMethodTensorListArr(JIValue::javaClassStatic(), jArray);
  } else if (ivalue.isList()) {
    auto list = ivalue.toList();
    static auto jMethodListArr =
        JIValue::javaClassStatic()
            ->getStaticMethod<facebook::jni::local_ref<JIValue>(
                facebook::jni::alias_ref<facebook::jni::JArrayClass<
                    JIValue::javaobject>::javaobject>)>("listFrom");
    auto jArray =
        facebook::jni::JArrayClass<JIValue::javaobject>::newArray(list.size());
    auto index = 0;
    for (const auto& e : list) {
      (*jArray)[index++] = JIValue::newJIValueFromAtIValue(e);
    }
    return jMethodListArr(JIValue::javaClassStatic(), jArray);
  } else if (ivalue.isGenericDict()) {
    auto dict = ivalue.toGenericDict();
    const auto keyType = dict.keyType();

    if (!keyType) {
      facebook::jni::throwNewJavaException(
          facebook::jni::gJavaLangIllegalArgumentException,
          "Unknown IValue-Dict key type");
    }

    const auto keyTypeKind = keyType->kind();
    if (c10::TypeKind::StringType == keyTypeKind) {
      static auto jMethodDictStringKey =
          JIValue::javaClassStatic()
              ->getStaticMethod<facebook::jni::local_ref<JIValue>(
                  facebook::jni::alias_ref<facebook::jni::JMap<
                      facebook::jni::alias_ref<
                          facebook::jni::JString::javaobject>,
                      facebook::jni::alias_ref<JIValue::javaobject>>>)>(
                  "dictStringKeyFrom");

      auto jmap = JHashMap<
          facebook::jni::alias_ref<facebook::jni::JString::javaobject>,
          facebook::jni::alias_ref<JIValue::javaobject>>::create();
      for (auto& pair : dict) {
        jmap->put(
            facebook::jni::make_jstring(pair.key().toString()->string()),
            JIValue::newJIValueFromAtIValue(pair.value()));
      }
      return jMethodDictStringKey(JIValue::javaClassStatic(), jmap);
    } else if (c10::TypeKind::IntType == keyTypeKind) {
      static auto jMethodDictLongKey =
          JIValue::javaClassStatic()
              ->getStaticMethod<facebook::jni::local_ref<JIValue>(
                  facebook::jni::alias_ref<facebook::jni::JMap<
                      facebook::jni::alias_ref<
                          facebook::jni::JLong::javaobject>,
                      facebook::jni::alias_ref<JIValue::javaobject>>>)>(
                  "dictLongKeyFrom");
      auto jmap = JHashMap<
          facebook::jni::alias_ref<facebook::jni::JLong::javaobject>,
          facebook::jni::alias_ref<JIValue::javaobject>>::create();
      for (auto& pair : dict) {
        jmap->put(
            facebook::jni::JLong::valueOf(pair.key().toInt()),
            JIValue::newJIValueFromAtIValue(pair.value()));
      }
      return jMethodDictLongKey(JIValue::javaClassStatic(), jmap);
    }

    facebook::jni::throwNewJavaException(
        facebook::jni::gJavaLangIllegalArgumentException,
        "Unsupported IValue-Dict key type");
  }

  facebook::jni::throwNewJavaException(
      facebook::jni::gJavaLangIllegalArgumentException,
      "Unsupported IValue type %s",
      ivalue.tagKind().c_str());
}

at::IValue JIValue::JIValueToAtIValue(
    facebook::jni::alias_ref<JIValue> jivalue) {
  Trace _s{"jni::JIValue::JIValueToAtIValue"};
  static const auto typeCodeField =
      JIValue::javaClassStatic()->getField<jint>("mTypeCode");
  const auto typeCode = jivalue->getFieldValue(typeCodeField);
  if (JIValue::kTypeCodeNull == typeCode) {
    return at::IValue{};
  } else if (JIValue::kTypeCodeTensor == typeCode) {
    static const auto jMethodGetTensor =
        JIValue::javaClassStatic()
            ->getMethod<facebook::jni::alias_ref<TensorHybrid::javaobject>()>(
                "toTensor");
    return TensorHybrid::newAtTensorFromJTensor(jMethodGetTensor(jivalue));
  } else if (JIValue::kTypeCodeBool == typeCode) {
    static const auto jMethodGetBool =
        JIValue::javaClassStatic()->getMethod<jboolean()>("toBool");
    // explicit cast to bool as jboolean is defined as uint8_t, IValue ctor
    // for int will be called for jboolean
    bool b = jMethodGetBool(jivalue);
    return at::IValue{b};
  } else if (JIValue::kTypeCodeLong == typeCode) {
    static const auto jMethodGetLong =
        JIValue::javaClassStatic()->getMethod<jlong()>("toLong");
    return at::IValue{(int64_t)jMethodGetLong(jivalue)};
  } else if (JIValue::kTypeCodeDouble == typeCode) {
    static const auto jMethodGetDouble =
        JIValue::javaClassStatic()->getMethod<jdouble()>("toDouble");
    return at::IValue{jMethodGetDouble(jivalue)};
  } else if (JIValue::kTypeCodeString == typeCode) {
    static const auto jMethodGetString =
        JIValue::javaClassStatic()->getMethod<jstring()>("toStr");
    return at::IValue{jMethodGetString(jivalue)->toStdString()};
  } else if (JIValue::kTypeCodeTuple == typeCode) {
    static const auto jMethodGetTuple =
        JIValue::javaClassStatic()
            ->getMethod<
                facebook::jni::JArrayClass<JIValue::javaobject>::javaobject()>(
                "toTuple");
    auto jarray = jMethodGetTuple(jivalue);
    size_t n = jarray->size();

    std::vector<at::IValue> elements;
    elements.reserve(n);
    for (auto i = 0; i < n; ++i) {
      auto jivalue_element = jarray->getElement(i);
      auto element = JIValue::JIValueToAtIValue(jivalue_element);
      elements.push_back(std::move(element));
    }
    return c10::ivalue::Tuple::create(std::move(elements));
  } else if (JIValue::kTypeCodeBoolList == typeCode) {
    static const auto jMethodGetBoolList =
        JIValue::javaClassStatic()->getMethod<jbooleanArray()>("toBoolList");
    auto jArray = jMethodGetBoolList(jivalue);
    auto jArrayPinned = jArray->pin();
    size_t n = jArrayPinned.size();
    c10::List<bool> list{};
    list.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      list.push_back(jArrayPinned[i]);
    }
    return at::IValue{std::move(list)};
  } else if (JIValue::kTypeCodeLongList == typeCode) {
    static const auto jMethodGetLongList =
        JIValue::javaClassStatic()->getMethod<jlongArray()>("toLongList");
    auto jArray = jMethodGetLongList(jivalue);
    auto jArrayPinned = jArray->pin();
    size_t n = jArrayPinned.size();
    c10::List<int64_t> list{};
    list.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      list.push_back(jArrayPinned[i]);
    }
    return at::IValue{std::move(list)};
  } else if (JIValue::kTypeCodeDoubleList == typeCode) {
    static const auto jMethodGetDoubleList =
        JIValue::javaClassStatic()->getMethod<jdoubleArray()>("toDoubleList");
    auto jArray = jMethodGetDoubleList(jivalue);
    auto jArrayPinned = jArray->pin();
    size_t n = jArrayPinned.size();
    c10::List<double> list{};
    list.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      list.push_back(jArrayPinned[i]);
    }
    return at::IValue{std::move(list)};
  } else if (JIValue::kTypeCodeTensorList == typeCode) {
    static const auto jMethodGetTensorList =
        JIValue::javaClassStatic()
            ->getMethod<facebook::jni::JArrayClass<
                TensorHybrid::javaobject>::javaobject()>("toTensorList");
    auto jArray = jMethodGetTensorList(jivalue);
    size_t n = jArray->size();
    c10::List<at::Tensor> list{};
    list.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      list.push_back(
          TensorHybrid::newAtTensorFromJTensor(jArray->getElement(i)));
    }
    return at::IValue{std::move(list)};
  } else if (JIValue::kTypeCodeList == typeCode) {
    static const auto jMethodGetList =
        JIValue::javaClassStatic()
            ->getMethod<
                facebook::jni::JArrayClass<JIValue::javaobject>::javaobject()>(
                "toList");
    auto jarray = jMethodGetList(jivalue);
    size_t n = jarray->size();
    if (n == 0) {
      return at::IValue{c10::impl::GenericList(c10::TensorType::get())};
    }

    auto jivalue_first_element = jarray->getElement(0);
    auto first_element = JIValue::JIValueToAtIValue(jivalue_first_element);
    c10::impl::GenericList list{c10::unshapedType(first_element.type())};
    list.reserve(n);
    list.push_back(first_element);
    for (auto i = 1; i < n; ++i) {
      auto jivalue_element = jarray->getElement(i);
      auto element = JIValue::JIValueToAtIValue(jivalue_element);
      list.push_back(element);
    }
    return at::IValue{list};
  } else if (JIValue::kTypeCodeDictStringKey == typeCode) {
    static const auto jMethodGetDictStringKey =
        JIValue::javaClassStatic()
            ->getMethod<facebook::jni::JMap<jstring, JIValue::javaobject>::
                            javaobject()>("toDictStringKey");
    auto jmap = jMethodGetDictStringKey(jivalue);
    auto it = jmap->begin();
    if (it == jmap->end()) {
      return at::IValue{c10::impl::GenericDict(
          c10::StringType::get(), c10::TensorType::get())};
    }

    auto firstEntryValue = JIValue::JIValueToAtIValue(it->second);
    c10::impl::GenericDict dict{c10::StringType::get(),
                                c10::unshapedType(firstEntryValue.type())};
    dict.insert(it->first->toStdString(), firstEntryValue);
    it++;
    for (; it != jmap->end(); it++) {
      dict.insert(
          it->first->toStdString(), JIValue::JIValueToAtIValue(it->second));
    }
    return at::IValue{dict};
  } else if (JIValue::kTypeCodeDictLongKey == typeCode) {
    static const auto jMethodGetDictLongKey =
        JIValue::javaClassStatic()
            ->getMethod<facebook::jni::JMap<
                facebook::jni::JLong::javaobject,
                JIValue::javaobject>::javaobject()>("toDictLongKey");
    auto jmap = jMethodGetDictLongKey(jivalue);
    auto it = jmap->begin();
    if (it == jmap->end()) {
      return at::IValue{
          c10::impl::GenericDict(c10::IntType::get(), c10::TensorType::get())};
    }

    auto firstEntryValue = JIValue::JIValueToAtIValue(it->second);
    c10::impl::GenericDict dict{c10::IntType::get(),
                                c10::unshapedType(firstEntryValue.type())};
    dict.insert((int64_t)it->first->longValue(), firstEntryValue);
    it++;
    for (; it != jmap->end(); it++) {
      dict.insert(
          (int64_t)it->first->longValue(),
          JIValue::JIValueToAtIValue(it->second));
    }
    return at::IValue{dict};
  }

  facebook::jni::throwNewJavaException(
      facebook::jni::gJavaLangIllegalArgumentException,
      "Unknown IValue typeCode %d",
      typeCode);
}

#if defined(__ANDROID__)
class PyTorchAndroidJni : public facebook::jni::JavaClass<PyTorchAndroidJni> {
 public:
  constexpr static auto kJavaDescriptor = "Lorg/pytorch/PyTorchAndroid;";

  static void registerNatives() {
    javaClassStatic()->registerNatives({
        makeNativeMethod(
            "nativeSetNumThreads", PyTorchAndroidJni::setNumThreads),
    });
  }

  static void setNumThreads(facebook::jni::alias_ref<jclass>, jint numThreads) {
    caffe2::mobile_threadpool()->setNumThreads(numThreads);
  }
};
#endif

void common_registerNatives() {
  static const int once = []() {
#if defined(__ANDROID__)
    pytorch_jni::PyTorchAndroidJni::registerNatives();
#endif
    return 0;
  }();
  ((void)once);
}

} // namespace pytorch_jni
