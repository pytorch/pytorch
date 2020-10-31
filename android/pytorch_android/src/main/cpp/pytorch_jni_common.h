#pragma once

#include <fbjni/fbjni.h>
#include <torch/csrc/api/include/torch/types.h>

#include "cmake_macros.h"

#ifdef __ANDROID__
#include <android/log.h>
#define ALOGI(...) \
  __android_log_print(ANDROID_LOG_INFO, "pytorch-jni", __VA_ARGS__)
#define ALOGE(...) \
  __android_log_print(ANDROID_LOG_ERROR, "pytorch-jni", __VA_ARGS__)
#endif

#if defined(TRACE_ENABLED) && defined(__ANDROID__)
#include <android/trace.h>
#include <dlfcn.h>
#endif

namespace pytorch_jni {

constexpr static int kDeviceCPU = 1;
constexpr static int kDeviceVulkan = 2;

c10::DeviceType deviceJniCodeToDeviceType(jint deviceJniCode);

class Trace {
 public:
#if defined(TRACE_ENABLED) && defined(__ANDROID__)
  typedef void* (*fp_ATrace_beginSection)(const char* sectionName);
  typedef void* (*fp_ATrace_endSection)(void);

  static fp_ATrace_beginSection ATrace_beginSection;
  static fp_ATrace_endSection ATrace_endSection;
#endif

  static void ensureInit() {
    if (!Trace::is_initialized_) {
      init();
      Trace::is_initialized_ = true;
    }
  }

  static void beginSection(const char* name) {
    Trace::ensureInit();
#if defined(TRACE_ENABLED) && defined(__ANDROID__)
    ATrace_beginSection(name);
#endif
  }

  static void endSection() {
#if defined(TRACE_ENABLED) && defined(__ANDROID__)
    ATrace_endSection();
#endif
  }

  Trace(const char* name) {
    ensureInit();
    beginSection(name);
  }

  ~Trace() {
    endSection();
  }

 private:
  static void init();
  static bool is_initialized_;
};

class JIValue : public facebook::jni::JavaClass<JIValue> {
 public:
  constexpr static const char* kJavaDescriptor = "Lorg/pytorch/IValue;";

  constexpr static int kTypeCodeNull = 1;

  constexpr static int kTypeCodeTensor = 2;
  constexpr static int kTypeCodeBool = 3;
  constexpr static int kTypeCodeLong = 4;
  constexpr static int kTypeCodeDouble = 5;
  constexpr static int kTypeCodeString = 6;

  constexpr static int kTypeCodeTuple = 7;
  constexpr static int kTypeCodeBoolList = 8;
  constexpr static int kTypeCodeLongList = 9;
  constexpr static int kTypeCodeDoubleList = 10;
  constexpr static int kTypeCodeTensorList = 11;
  constexpr static int kTypeCodeList = 12;

  constexpr static int kTypeCodeDictStringKey = 13;
  constexpr static int kTypeCodeDictLongKey = 14;

  static facebook::jni::local_ref<JIValue> newJIValueFromAtIValue(
      const at::IValue& ivalue);

  static at::IValue JIValueToAtIValue(
      facebook::jni::alias_ref<JIValue> jivalue);
};

void common_registerNatives();
} // namespace pytorch_jni
