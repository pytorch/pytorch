/*
 *  Copyright (c) 2018-present, Facebook, Inc.
 *
 *  This source code is licensed under the MIT license found in the LICENSE
 *  file in the root directory of this source tree.
 *
 */
#include <fbjni/fbjni.h>

#include <mutex>
#include <vector>

#include <fbjni/detail/utf8.h>

namespace facebook {
namespace jni {

jint initialize(JavaVM* vm, std::function<void()>&& init_fn) noexcept {
  // TODO (t7832883): DTRT when we have exception pointers
  static auto error_msg = std::string{"Failed to initialize fbjni"};
  static bool error_occured = [vm] {
      bool retVal = false;
      try {
        Environment::initialize(vm);
      } catch (std::exception& ex) {
        retVal = true;
        try {
          error_msg = std::string{"Failed to initialize fbjni: "} + ex.what();
        } catch (...) {
          // Ignore, we already have a fall back message
        }
      } catch (...) {
        retVal = true;
      }
      return retVal;
    }();

  try {
    if (error_occured) {
      throw std::runtime_error(error_msg);
    }

    init_fn();
  } catch (const std::exception& e) {
    FBJNI_LOGE("error %s", e.what());
    translatePendingCppExceptionToJavaException();
  } catch (...) {
    translatePendingCppExceptionToJavaException();
    // So Java will handle the translated exception, fall through and
    // return a good version number.
  }
  return JNI_VERSION_1_6;
}

alias_ref<JClass> findClassStatic(const char* name) {
  const auto env = detail::currentOrNull();
  if (!env) {
    throw std::runtime_error("Unable to retrieve JNIEnv*.");
  }
  local_ref<jclass> cls = adopt_local(env->FindClass(name));
  FACEBOOK_JNI_THROW_EXCEPTION_IF(!cls);
  auto leaking_ref = (jclass)env->NewGlobalRef(cls.get());
  FACEBOOK_JNI_THROW_EXCEPTION_IF(!leaking_ref);
  return wrap_alias(leaking_ref);
}

local_ref<JClass> findClassLocal(const char* name) {
  const auto env = detail::currentOrNull();
  if (!env) {
    throw std::runtime_error("Unable to retrieve JNIEnv*.");
  }
  auto cls = env->FindClass(name);
  FACEBOOK_JNI_THROW_EXCEPTION_IF(!cls);
  return adopt_local(cls);
}


// jstring /////////////////////////////////////////////////////////////////////////////////////////

std::string JString::toStdString() const {
  const auto env = Environment::current();
  auto utf16String = JStringUtf16Extractor(env, self());
  return detail::utf16toUTF8(utf16String.chars(), utf16String.length());
}

std::u16string JString::toU16String() const {
  const auto env = Environment::current();
  auto utf16String = JStringUtf16Extractor(env, self());
  if (!utf16String.chars() || utf16String.length() == 0) {
    return {};
  }
  return std::u16string(reinterpret_cast<const char16_t*>(utf16String.chars()), utf16String.length());
}

local_ref<JString> make_jstring(const char* utf8) {
  if (!utf8) {
    return {};
  }
  const auto env = Environment::current();
  size_t len;
  size_t modlen = detail::modifiedLength(reinterpret_cast<const uint8_t*>(utf8), &len);
  jstring result;
  if (modlen == len) {
    // The only difference between utf8 and modifiedUTF8 is in encoding 4-byte UTF8 chars
    // and '\0' that is encoded on 2 bytes.
    //
    // Since modifiedUTF8-encoded string can be no shorter than its UTF8 conterpart we
    // know that if those two strings are of the same length we don't need to do any
    // conversion -> no 4-byte chars nor '\0'.
    result = env->NewStringUTF(utf8);
  } else {
    auto modified = std::vector<char>(modlen + 1); // allocate extra byte for \0
    detail::utf8ToModifiedUTF8(
      reinterpret_cast<const uint8_t*>(utf8), len,
      reinterpret_cast<uint8_t*>(modified.data()), modified.size());
    result = env->NewStringUTF(modified.data());
  }
  FACEBOOK_JNI_THROW_PENDING_EXCEPTION();
  return adopt_local(result);
}

local_ref<JString> make_jstring(const std::u16string& utf16) {
  if (utf16.empty()) {
    return {};
  }
  const auto env = Environment::current();
  static_assert(
      sizeof(jchar) == sizeof(std::u16string::value_type),
      "Expecting jchar to be the same size as std::u16string::CharT");
  jstring result = env->NewString(reinterpret_cast<const jchar*>(utf16.c_str()), utf16.size());
  FACEBOOK_JNI_THROW_PENDING_EXCEPTION();
  return adopt_local(result);
}

// JniPrimitiveArrayFunctions //////////////////////////////////////////////////////////////////////

#pragma push_macro("DEFINE_PRIMITIVE_METHODS")
#undef DEFINE_PRIMITIVE_METHODS
#define DEFINE_PRIMITIVE_METHODS(TYPE, NAME, SMALLNAME)                        \
                                                                               \
template<>                                                                     \
TYPE* JPrimitiveArray<TYPE ## Array>::getElements(jboolean* isCopy) {          \
  auto env = Environment::current();                                           \
  TYPE* res =  env->Get ## NAME ## ArrayElements(self(), isCopy);              \
  FACEBOOK_JNI_THROW_PENDING_EXCEPTION();                                      \
  return res;                                                                  \
}                                                                              \
                                                                               \
template<>                                                                     \
void JPrimitiveArray<TYPE ## Array>::releaseElements(                          \
    TYPE* elements, jint mode) {                                               \
  auto env = Environment::current();                                           \
  env->Release ## NAME ## ArrayElements(self(), elements, mode);               \
  FACEBOOK_JNI_THROW_PENDING_EXCEPTION();                                      \
}                                                                              \
                                                                               \
template<>                                                                     \
void JPrimitiveArray<TYPE ## Array>::getRegion(                                \
    jsize start, jsize length, TYPE* buf) {                                    \
  auto env = Environment::current();                                           \
  env->Get ## NAME ## ArrayRegion(self(), start, length, buf);                 \
  FACEBOOK_JNI_THROW_PENDING_EXCEPTION();                                      \
}                                                                              \
                                                                               \
template<>                                                                     \
void JPrimitiveArray<TYPE ## Array>::setRegion(                                \
    jsize start, jsize length, const TYPE* elements) {                         \
  auto env = Environment::current();                                           \
  env->Set ## NAME ## ArrayRegion(self(), start, length, elements);            \
  FACEBOOK_JNI_THROW_PENDING_EXCEPTION();                                      \
}                                                                              \
                                                                               \
local_ref<TYPE ## Array> make_ ## SMALLNAME ## _array(jsize size) {            \
  auto array = Environment::current()->New ## NAME ## Array(size);             \
  FACEBOOK_JNI_THROW_EXCEPTION_IF(!array);                                     \
  return adopt_local(array);                                                   \
}                                                                              \
                                                                               \
template<>                                                                     \
local_ref<TYPE ## Array> JArray ## NAME::newArray(size_t count) {              \
  return make_ ## SMALLNAME ## _array(count);                                  \
}                                                                              \
                                                                               \

DEFINE_PRIMITIVE_METHODS(jboolean, Boolean, boolean)
DEFINE_PRIMITIVE_METHODS(jbyte, Byte, byte)
DEFINE_PRIMITIVE_METHODS(jchar, Char, char)
DEFINE_PRIMITIVE_METHODS(jshort, Short, short)
DEFINE_PRIMITIVE_METHODS(jint, Int, int)
DEFINE_PRIMITIVE_METHODS(jlong, Long, long)
DEFINE_PRIMITIVE_METHODS(jfloat, Float, float)
DEFINE_PRIMITIVE_METHODS(jdouble, Double, double)
#pragma pop_macro("DEFINE_PRIMITIVE_METHODS")

namespace detail {

detail::BaseHybridClass* HybridDestructor::getNativePointer() {
  static auto pointerField = javaClassStatic()->getField<jlong>("mNativePointer");
  auto* value = reinterpret_cast<detail::BaseHybridClass*>(getFieldValue(pointerField));
  if (!value) {
    throwNewJavaException("java/lang/NullPointerException", "java.lang.NullPointerException");
  }
  return value;
}

void HybridDestructor::setNativePointer(
    std::unique_ptr<detail::BaseHybridClass> new_value) {
  static auto pointerField = javaClassStatic()->getField<jlong>("mNativePointer");
  auto old_value = std::unique_ptr<detail::BaseHybridClass>(
    reinterpret_cast<detail::BaseHybridClass*>(getFieldValue(pointerField)));
  if (new_value && old_value) {
    FBJNI_LOGF("Attempt to set C++ native pointer twice");
  }
  setFieldValue(pointerField, reinterpret_cast<jlong>(new_value.release()));
}

}

// Internal debug /////////////////////////////////////////////////////////////////////////////////

namespace internal {

ReferenceStats g_reference_stats;

void facebook::jni::internal::ReferenceStats::reset() noexcept {
  locals_deleted = globals_deleted = weaks_deleted = 0;
}

}

}}
