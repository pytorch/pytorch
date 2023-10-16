#include <cassert>
#include <iostream>
#include <memory>
#include <string>

#include <fbjni/ByteBuffer.h>
#include <fbjni/fbjni.h>

#include <c10/util/irange.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/script.h>
#include "caffe2/serialize/read_adapter_interface.h"

#include "pytorch_jni_common.h"

#ifdef __ANDROID__
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>

#ifndef USE_PTHREADPOOL
#define USE_PTHREADPOOL
#endif /* USE_PTHREADPOOL */
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#endif

namespace pytorch_jni {

namespace {

struct LiteJITCallGuard {
  // VariableType dispatch is not included in default mobile build. We need set
  // this guard globally to avoid dispatch error (only for dynamic dispatch).
  // Thanks to the unification of Variable class and Tensor class it's no longer
  // required to toggle the NonVariableTypeMode per op - so it doesn't hurt to
  // always set NonVariableTypeMode for inference only use case.
  // TODO: Ideally AutoNonVariableTypeMode in this file should be changed to
  // InferenceMode but it's blocked due to typeahead application on Oculus
  // (D27943428). To unblock, we need to find out which op is making inplace
  // update to an inference tensor outside InferenceMode and properly guard it.
  torch::AutoNonVariableTypeMode non_var_guard;
};

} // namespace

class PytorchJni : public facebook::jni::HybridClass<PytorchJni> {
 private:
  friend HybridBase;
  torch::jit::mobile::Module module_;
  c10::DeviceType deviceType_;

 public:
  constexpr static auto kJavaDescriptor = "Lorg/pytorch/LiteNativePeer;";

  static facebook::jni::local_ref<jhybriddata> initHybrid(
      facebook::jni::alias_ref<jclass>,
      facebook::jni::alias_ref<jstring> modelPath,
      facebook::jni::alias_ref<
          facebook::jni::JMap<facebook::jni::JString, facebook::jni::JString>>
          extraFiles,
      jint device) {
    return makeCxxInstance(modelPath, extraFiles, device);
  }

#ifdef __ANDROID__
  static facebook::jni::local_ref<jhybriddata> initHybridAndroidAsset(
      facebook::jni::alias_ref<jclass>,
      facebook::jni::alias_ref<jstring> assetName,
      facebook::jni::alias_ref<jobject> assetManager,
      jint device) {
    return makeCxxInstance(assetName, assetManager, device);
  }
#endif

  PytorchJni(
      facebook::jni::alias_ref<jstring> modelPath,
      facebook::jni::alias_ref<
          facebook::jni::JMap<facebook::jni::JString, facebook::jni::JString>>
          extraFiles,
      jint device) {
    LiteJITCallGuard guard;
    std::unordered_map<std::string, std::string> extra_files;
    const auto has_extra = extraFiles && extraFiles->size() > 0;
    if (has_extra) {
      for (const auto& e : *extraFiles) {
        extra_files[e.first->toStdString()] = "";
      }
    }
    deviceType_ = deviceJniCodeToDeviceType(device);
    module_ = torch::jit::_load_for_mobile(
        std::move(modelPath->toStdString()), c10::nullopt, extra_files);
    torch::jit::_load_extra_only_for_mobile(
        std::move(modelPath->toStdString()), c10::nullopt, extra_files);
    if (has_extra) {
      static auto putMethod =
          facebook::jni::JMap<facebook::jni::JString, facebook::jni::JString>::
              javaClassStatic()
                  ->template getMethod<facebook::jni::alias_ref<jobject>(
                      facebook::jni::alias_ref<jobject>,
                      facebook::jni::alias_ref<jobject>)>("put");
      for (const auto& ef : extra_files) {
        putMethod(
            extraFiles,
            facebook::jni::make_jstring(ef.first),
            facebook::jni::make_jstring(ef.second));
      }
    }
  }

#ifdef __ANDROID__
  PytorchJni(
      facebook::jni::alias_ref<jstring> assetName,
      facebook::jni::alias_ref<jobject> assetManager,
      jint device) {
    JNIEnv* env = facebook::jni::Environment::current();
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager.get());
    if (!mgr) {
      facebook::jni::throwNewJavaException(
          facebook::jni::gJavaLangIllegalArgumentException,
          "Unable to get asset manager");
    }
    AAsset* asset = AAssetManager_open(
        mgr, assetName->toStdString().c_str(), AASSET_MODE_BUFFER);
    if (!asset) {
      facebook::jni::throwNewJavaException(
          facebook::jni::gJavaLangIllegalArgumentException,
          "Failed to open asset '%s'",
          assetName->toStdString().c_str());
    }
    auto assetBuffer = AAsset_getBuffer(asset);
    if (!assetBuffer) {
      facebook::jni::throwNewJavaException(
          facebook::jni::gJavaLangIllegalArgumentException,
          "Could not get buffer for asset '%s'",
          assetName->toStdString().c_str());
    }
    LiteJITCallGuard guard;
    module_ =
        torch::jit::_load_for_mobile(std::make_unique<MemoryReadAdapter>(
            assetBuffer, AAsset_getLength(asset)));
    AAsset_close(asset);
    deviceType_ = deviceJniCodeToDeviceType(device);
  }
#endif

  static void registerNatives() {
    registerHybrid({
        makeNativeMethod("initHybrid", PytorchJni::initHybrid),
#ifdef __ANDROID__
        makeNativeMethod(
            "initHybridAndroidAsset", PytorchJni::initHybridAndroidAsset),
#endif
        makeNativeMethod("forward", PytorchJni::forward),
        makeNativeMethod("runMethod", PytorchJni::runMethod),
    });
  }

  facebook::jni::local_ref<JIValue> forward(
      facebook::jni::alias_ref<
          facebook::jni::JArrayClass<JIValue::javaobject>::javaobject>
          jinputs) {
    std::vector<at::IValue> inputs{};
    size_t n = jinputs->size();
    inputs.reserve(n);
    for (const auto i : c10::irange(n)) {
      at::IValue atIValue = JIValue::JIValueToAtIValue(jinputs->getElement(i));
      inputs.push_back(std::move(atIValue));
    }

    auto output = [&]() {
      LiteJITCallGuard guard;
      return module_.forward(inputs);
    }();
    return JIValue::newJIValueFromAtIValue(output);
  }

  facebook::jni::local_ref<JIValue> runMethod(
      facebook::jni::alias_ref<facebook::jni::JString::javaobject> jmethodName,
      facebook::jni::alias_ref<
          facebook::jni::JArrayClass<JIValue::javaobject>::javaobject>
          jinputs) {
    std::string methodName = jmethodName->toStdString();

    std::vector<at::IValue> inputs{};
    size_t n = jinputs->size();
    inputs.reserve(n);
    for (const auto i : c10::irange(n)) {
      at::IValue atIValue = JIValue::JIValueToAtIValue(jinputs->getElement(i));
      inputs.push_back(std::move(atIValue));
    }
    if (auto method = module_.find_method(methodName)) {
      auto output = [&]() {
        LiteJITCallGuard guard;
        return module_.get_method(methodName)(inputs);
      }();
      return JIValue::newJIValueFromAtIValue(output);
    }

    facebook::jni::throwNewJavaException(
        facebook::jni::gJavaLangIllegalArgumentException,
        "Undefined method %s",
        methodName.c_str());
  }
};

#if defined(__ANDROID__)
class PyTorchAndroidJni : public facebook::jni::JavaClass<PyTorchAndroidJni> {
 public:
  constexpr static auto kJavaDescriptor = "Lorg/pytorch/LitePyTorchAndroid;";

  static void registerNatives() {
    javaClassStatic()->registerNatives({
        makeNativeMethod(
            "nativeSetNumThreads", PyTorchAndroidJni::setNumThreads),
    });
  }

  static void setNumThreads(facebook::jni::alias_ref<jclass>, jint numThreads) {
    caffe2::pthreadpool()->set_thread_count(numThreads);
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

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  return facebook::jni::initialize(vm, [] {
    pytorch_jni::common_registerNatives();
    pytorch_jni::PytorchJni::registerNatives();
  });
}
