#pragma once

#include <string>

#include <fbjni/ByteBuffer.h>
#include <fbjni/fbjni.h>

#include <torch/script.h>

namespace pytorch_jni {

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
};

} // namespace pytorch_jni