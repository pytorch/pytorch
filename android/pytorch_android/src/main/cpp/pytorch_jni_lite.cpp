#include <cassert>
#include <iostream>
#include <memory>
#include <string>

#include <fbjni/ByteBuffer.h>
#include <fbjni/fbjni.h>

#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/script.h>

#include "pytorch_jni_common.h"

namespace pytorch_jni {

namespace {

struct LiteJITCallGuard {
  // VariableType dispatch is not included in default mobile build. We need set
  // this guard globally to avoid dispatch error (only for dynamic dispatch).
  // Thanks to the unification of Variable class and Tensor class it's no longer
  // required to toggle the NonVariableTypeMode per op - so it doesn't hurt to
  // always set NonVariableTypeMode for inference only use case.
  // TODO: Ideally AutoNonVariableTypeMode in this file should be changed to InferenceMode
  // but it's blocked due to typeahead application on Oculus (D27943428). To unblock, we need
  // to find out which op is making inplace update to an inference tensor outside InferenceMode
  // and properly guard it.
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
        std::move(modelPath->toStdString()), deviceType_, extra_files);
    torch::jit::_load_extra_only_for_mobile(
        std::move(modelPath->toStdString()), deviceType_, extra_files);
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

  static void registerNatives() {
    registerHybrid({
        makeNativeMethod("initHybrid", PytorchJni::initHybrid),
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
    for (size_t i = 0; i < n; i++) {
      at::IValue atIValue = JIValue::JIValueToAtIValue(jinputs->getElement(i));
      if (at::kVulkan == deviceType_) {
        inputs.push_back(
            atIValue.isTensor() ? at::IValue{atIValue.toTensor().vulkan()}
                                : std::move(atIValue));
      } else {
        TORCH_CHECK(at::kCPU == deviceType_);
        inputs.push_back(std::move(atIValue));
      }
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
    for (size_t i = 0; i < n; i++) {
      at::IValue atIValue = JIValue::JIValueToAtIValue(jinputs->getElement(i));
      if (at::kVulkan == deviceType_) {
        inputs.push_back(
            atIValue.isTensor() ? at::IValue{atIValue.toTensor().vulkan()}
                                : std::move(atIValue));
      } else {
        TORCH_CHECK(at::kCPU == deviceType_);
        inputs.push_back(std::move(atIValue));
      }
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

} // namespace pytorch_jni

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  return facebook::jni::initialize(vm, [] {
    pytorch_jni::common_registerNatives();
    pytorch_jni::PytorchJni::registerNatives();
  });
}
