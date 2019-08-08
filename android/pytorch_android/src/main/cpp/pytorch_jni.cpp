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

void PytorchJni::registerNatives() {
  registerHybrid({
      makeNativeMethod("initHybrid", PytorchJni::initHybrid),
      makeNativeMethod("run", PytorchJni::run),
  });
}

at::Tensor PytorchJni::newAtTensor(
    facebook::jni::alias_ref<facebook::jni::JBuffer> inputData,
    facebook::jni::alias_ref<jintArray> inputDims, jint typeCode) {
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

  const auto &inputTensorDims = torch::IntArrayRef(inputDimsVec);
  at::Tensor inputTensor = torch::empty(inputTensorDims);
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
  /*// No-copy option, what to do with ownership of that tensor, tensorHolder to
  hybrid object? facebook::jni::local_ref<facebook::jni::JByteBuffer>
  jOutputTensorBuffer = facebook::jni::JByteBuffer::wrapBytes(
          reinterpret_cast<uint8_t*>(outputTensor.storage().data()),
          outputTensor.nbytes());
  //end plan A:no-copy*/
}

} // namespace pytorch_jni