#include <cassert>
#include <iostream>
#include <memory>
#include <string>

#include <fbjni/ByteBuffer.h>
#include <fbjni/fbjni.h>

#include <torch/csrc/autograd/record_function.h>
#include <torch/script.h>
#include "caffe2/serialize/read_adapter_interface.h"

#include "pytorch_jni_common.h"

namespace pytorch_jni {

class JReadAdapter : public facebook::jni::JavaClass<JReadAdapter> {
 public:
  static constexpr auto kJavaDescriptor = "Lorg/pytorch/ReadAdapter;";

  jlong size() {
    static const auto method =
        JReadAdapter::javaClassStatic()->getMethod<jlong()>("size");
    jlong result = method(self());
    return result;
  }
};

class ReadAdapter final : public caffe2::serialize::ReadAdapterInterface {
 public:
  explicit ReadAdapter(
      facebook::jni::alias_ref<JReadAdapter::javaobject> jReadAdapter)
      : jReadAdapter_(facebook::jni::make_global(jReadAdapter)),
        size_(jReadAdapter_->size()){};

  size_t size() const override {
    return size_;
  }

  size_t read(uint64_t pos, void* buf, size_t n, const char* what = "")
      const override {
    if (pos >= size_) {
      return 0;
    }
    facebook::jni::local_ref<facebook::jni::JByteBuffer> jBuf =
        facebook::jni::JByteBuffer::wrapBytes(static_cast<uint8_t*>(buf), n);
    static const auto method =
        JReadAdapter::javaClassStatic()
            ->getMethod<jint(
                jlong,
                facebook::jni::alias_ref<facebook::jni::JByteBuffer>,
                jint)>("read");
    jint result = method(jReadAdapter_, pos, jBuf, n);
    return result > 0 ? result : 0;
  }

  ~ReadAdapter() {}

 private:
  facebook::jni::global_ref<JReadAdapter::javaobject> jReadAdapter_;
  size_t size_ = {};
};

class PytorchJni : public facebook::jni::HybridClass<PytorchJni> {
 private:
  friend HybridBase;
  torch::jit::script::Module module_;

 public:
  constexpr static auto kJavaDescriptor = "Lorg/pytorch/NativePeer;";

  static facebook::jni::local_ref<jhybriddata> initHybridFilePath(
      facebook::jni::alias_ref<jclass>,
      facebook::jni::alias_ref<jstring> modelPath) {
    return makeCxxInstance(modelPath);
  }

  static facebook::jni::local_ref<jhybriddata> initHybridReadAdapter(
      facebook::jni::alias_ref<jclass>,
      facebook::jni::alias_ref<JReadAdapter::javaobject> jReadAdapter) {
    return makeCxxInstance(jReadAdapter);
  }

#ifdef TRACE_ENABLED
  static void onFunctionEnter(
      const torch::autograd::profiler::RecordFunction& fn) {
    Trace::beginSection(fn.name().str());
  }

  static void onFunctionExit(const torch::autograd::profiler::RecordFunction&) {
    Trace::endSection();
  }
#endif

  void preModuleLoadSetup() {
    auto qengines = at::globalContext().supportedQEngines();
    if (std::find(qengines.begin(), qengines.end(), at::QEngine::QNNPACK) !=
        qengines.end()) {
      at::globalContext().setQEngine(at::QEngine::QNNPACK);
    }
#ifdef TRACE_ENABLED
    torch::autograd::profiler::pushCallback(
        &onFunctionEnter,
        &onFunctionExit,
        /* need_inputs */ false,
        /* sampled */ false);
#endif
  }

  PytorchJni(facebook::jni::alias_ref<jstring> modelPath) {
    preModuleLoadSetup();
    module_ = torch::jit::load(std::move(modelPath->toStdString()));
    module_.eval();
  }

  PytorchJni(facebook::jni::alias_ref<JReadAdapter::javaobject> jReadAdapter) {
    preModuleLoadSetup();
    module_ = torch::jit::load(torch::make_unique<ReadAdapter>(jReadAdapter));
    module_.eval();
  }

  static void registerNatives() {
    registerHybrid({
        makeNativeMethod("initHybridFilePath", PytorchJni::initHybridFilePath),
        makeNativeMethod(
            "initHybridReadAdapter", PytorchJni::initHybridReadAdapter),
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
      inputs.push_back(std::move(atIValue));
    }
    auto output = [&]() {
      torch::autograd::AutoGradMode guard(false);
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
      inputs.push_back(std::move(atIValue));
    }
    if (auto method = module_.find_method(methodName)) {
      auto output = [&]() {
        torch::autograd::AutoGradMode guard(false);
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
  return facebook::jni::initialize(
      vm, [] { pytorch_jni::PytorchJni::registerNatives(); });
}
