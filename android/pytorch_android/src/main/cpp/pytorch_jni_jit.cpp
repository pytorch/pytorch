#include <cassert>
#include <iostream>
#include <memory>
#include <string>

#include <fbjni/ByteBuffer.h>
#include <fbjni/fbjni.h>

#include <ATen/record_function.h>
#include <torch/csrc/jit/runtime/print_handler.h>
#include <torch/script.h>
#include "caffe2/serialize/read_adapter_interface.h"

#include "pytorch_jni_common.h"

#ifdef __ANDROID__
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#endif

using namespace torch::autograd::profiler;

namespace pytorch_jni {

namespace {

struct JITCallGuard {
  // AutoGrad is disabled for mobile by default.
  torch::autograd::AutoGradMode no_autograd_guard{false};
  // VariableType dispatch is not included in default mobile build. We need set
  // this guard globally to avoid dispatch error (only for dynamic dispatch).
  // Thanks to the unification of Variable class and Tensor class it's no longer
  // required to toggle the NonVariableTypeMode per op - so it doesn't hurt to
  // always set NonVariableTypeMode for inference only use case.
  torch::AutoNonVariableTypeMode non_var_guard{true};
  // Disable graph optimizer to ensure list of unused ops are not changed for
  // custom mobile build.
  torch::jit::GraphOptimizerEnabledGuard no_optimizer_guard{false};
};

} // namespace

class MemoryReadAdapter final : public caffe2::serialize::ReadAdapterInterface {
 public:
  explicit MemoryReadAdapter(const void* data, off_t size)
      : data_(data), size_(size){};

  size_t size() const override {
    return size_;
  }

  size_t read(uint64_t pos, void* buf, size_t n, const char* what = "")
      const override {
    memcpy(buf, (int8_t*)(data_) + pos, n);
    return n;
  }

  ~MemoryReadAdapter() {}

 private:
  const void* data_;
  off_t size_;
};

class PytorchJni : public facebook::jni::HybridClass<PytorchJni> {
 private:
  friend HybridBase;
  torch::jit::Module module_;
  c10::DeviceType deviceType_;

 public:
  constexpr static auto kJavaDescriptor = "Lorg/pytorch/NativePeer;";

  static facebook::jni::local_ref<jhybriddata> initHybrid(
      facebook::jni::alias_ref<jclass>,
      facebook::jni::alias_ref<jstring> modelPath,
      jint device) {
    return makeCxxInstance(modelPath, device);
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

#ifdef TRACE_ENABLED
  static std::unique_ptr<at::ObserverContext> onFunctionEnter(
      const at::RecordFunction& fn) {
    Trace::beginSection(fn.name().str());
    return nullptr;
  }

  static void onFunctionExit(const at::RecordFunction&, at::ObserverContext*) {
    Trace::endSection();
  }
#endif

  static void preModuleLoadSetupOnce() {
    auto qengines = at::globalContext().supportedQEngines();
    if (std::find(qengines.begin(), qengines.end(), at::QEngine::QNNPACK) !=
        qengines.end()) {
      at::globalContext().setQEngine(at::QEngine::QNNPACK);
    }

#ifdef __ANDROID__
    torch::jit::setPrintHandler([](const std::string& s) {
      __android_log_print(ANDROID_LOG_DEBUG, "pytorch-print", "%s", s.c_str());
    });
#endif

#ifdef TRACE_ENABLED
    at::addGlobalCallback(at::RecordFunctionCallback(
        &onFunctionEnter,
        &onFunctionExit)
      .scopes({RecordScope::FUNCTION, RecordScope::USER_SCOPE}));
#endif
  }

  void preModuleLoadSetup() {
    static const int once = []() {
      preModuleLoadSetupOnce();
      return 0;
    }();
    ((void)once);
  }

  PytorchJni(facebook::jni::alias_ref<jstring> modelPath, jint device) {
    preModuleLoadSetup();
    JITCallGuard guard;
    module_ = torch::jit::load(std::move(modelPath->toStdString()));
    module_.eval();
    deviceType_ = deviceJniCodeToDeviceType(device);
  }

#ifdef __ANDROID__
  PytorchJni(
      facebook::jni::alias_ref<jstring> assetName,
      facebook::jni::alias_ref<jobject> assetManager,
      jint device) {
    preModuleLoadSetup();
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
    JITCallGuard guard;
    module_ = torch::jit::load(torch::make_unique<MemoryReadAdapter>(
        assetBuffer, AAsset_getLength(asset)));
    AAsset_close(asset);
    module_.eval();
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
    Trace _s{"jni::Module::forward"};
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
      JITCallGuard guard;
      return module_.forward(std::move(inputs));
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
        JITCallGuard guard;
        return (*method)(std::move(inputs));
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
