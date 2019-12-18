#include <cassert>
#include <cmath>
#include <vector>

#include <fbjni/ByteBuffer.h>
#include <fbjni/fbjni.h>

#if defined(__ANDROID__)

#include <android/log.h>
#define ALOGI(...) \
  __android_log_print(ANDROID_LOG_INFO, "pytorch-vision-jni", __VA_ARGS__)

#endif
#define clamp0255(x) x > 255 ? 255 : x < 0 ? 0 : x

namespace pytorch_vision_jni {
class PytorchVisionJni : public facebook::jni::JavaClass<PytorchVisionJni> {
 public:
  constexpr static auto kJavaDescriptor =
      "Lorg/pytorch/torchvision/TensorImageUtils$NativePeer;";

  static void imageYUV420CenterCropToFloatBuffer(
      facebook::jni::alias_ref<jclass>,
      facebook::jni::alias_ref<facebook::jni::JByteBuffer> yBuffer,
      const int yRowStride,
      const int yPixelStride,
      facebook::jni::alias_ref<facebook::jni::JByteBuffer> uBuffer,
      facebook::jni::alias_ref<facebook::jni::JByteBuffer> vBuffer,
      const int uRowStride,
      const int uvPixelStride,
      const int imageWidth,
      const int imageHeight,
      const int rotateCWDegrees,
      const int tensorWidth,
      const int tensorHeight,
      facebook::jni::alias_ref<jfloatArray> jnormMeanRGB,
      facebook::jni::alias_ref<jfloatArray> jnormStdRGB,
      facebook::jni::alias_ref<facebook::jni::JBuffer> outBuffer,
      const int outOffset) {
    static JNIEnv* jni = facebook::jni::Environment::current();
    float* outData = (float*)jni->GetDirectBufferAddress(outBuffer.get());

    auto normMeanRGB = jnormMeanRGB->getRegion(0, 3);
    auto normStdRGB = jnormStdRGB->getRegion(0, 3);

    int widthAfterRtn = imageWidth;
    int heightAfterRtn = imageHeight;
    bool oddRotation = rotateCWDegrees == 90 || rotateCWDegrees == 270;
    if (oddRotation) {
      widthAfterRtn = imageHeight;
      heightAfterRtn = imageWidth;
    }

    int cropWidthAfterRtn = widthAfterRtn;
    int cropHeightAfterRtn = heightAfterRtn;

    if (tensorWidth * heightAfterRtn <= tensorHeight * widthAfterRtn) {
      cropWidthAfterRtn = tensorWidth * heightAfterRtn / tensorHeight;
    } else {
      cropHeightAfterRtn = tensorHeight * widthAfterRtn / tensorWidth;
    }

    int cropWidthBeforeRtn = cropWidthAfterRtn;
    int cropHeightBeforeRtn = cropHeightAfterRtn;
    if (oddRotation) {
      cropWidthBeforeRtn = cropHeightAfterRtn;
      cropHeightBeforeRtn = cropWidthAfterRtn;
    }

    const int offsetX = (imageWidth - cropWidthBeforeRtn) / 2.f;
    const int offsetY = (imageHeight - cropHeightBeforeRtn) / 2.f;

    const uint8_t* yData = yBuffer->getDirectBytes();
    const uint8_t* uData = uBuffer->getDirectBytes();
    const uint8_t* vData = vBuffer->getDirectBytes();

    float scale = cropWidthAfterRtn / tensorWidth;
    int uvRowStride = uRowStride >> 1;
    int cropXMult = 1;
    int cropYMult = 1;
    int cropXAdd = offsetX;
    int cropYAdd = offsetY;
    if (rotateCWDegrees == 90) {
      cropYMult = -1;
      cropYAdd = offsetY + (cropHeightBeforeRtn - 1);
    } else if (rotateCWDegrees == 180) {
      cropXMult = -1;
      cropXAdd = offsetX + (cropWidthBeforeRtn - 1);
      cropYMult = -1;
      cropYAdd = offsetY + (cropHeightBeforeRtn - 1);
    } else if (rotateCWDegrees == 270) {
      cropXMult = -1;
      cropXAdd = offsetX + (cropWidthBeforeRtn - 1);
    }

    float normMeanRm255 = 255 * normMeanRGB[0];
    float normMeanGm255 = 255 * normMeanRGB[1];
    float normMeanBm255 = 255 * normMeanRGB[2];
    float normStdRm255 = 255 * normStdRGB[0];
    float normStdGm255 = 255 * normStdRGB[1];
    float normStdBm255 = 255 * normStdRGB[2];

    int xBeforeRtn, yBeforeRtn;
    int yIdx, uvIdx, ui, vi, a0, ri, gi, bi;
    int channelSize = tensorWidth * tensorHeight;
    int wr = outOffset;
    int wg = wr + channelSize;
    int wb = wg + channelSize;
    for (int x = 0; x < tensorWidth; x++) {
      for (int y = 0; y < tensorHeight; y++) {
        xBeforeRtn = cropXAdd + cropXMult * (int)(x * scale);
        yBeforeRtn = cropYAdd + cropYMult * (int)(y * scale);
        yIdx = yBeforeRtn * yRowStride + xBeforeRtn * yPixelStride;
        uvIdx = (yBeforeRtn >> 1) * uvRowStride + xBeforeRtn * uvPixelStride;
        ui = uData[uvIdx];
        vi = vData[uvIdx];
        a0 = 1192 * (yData[yIdx] - 16);
        ri = (a0 + 1634 * (vi - 128)) >> 10;
        gi = (a0 - 832 * (vi - 128) - 400 * (ui - 128)) >> 10;
        bi = (a0 + 2066 * (ui - 128)) >> 10;
        outData[wr++] = (clamp0255(ri) - normMeanRm255) / normStdRm255;
        outData[wg++] = (clamp0255(gi) - normMeanGm255) / normStdGm255;
        outData[wb++] = (clamp0255(bi) - normMeanBm255) / normStdBm255;
      }
    }
  }

  static void registerNatives() {
    javaClassStatic()->registerNatives({
        makeNativeMethod(
            "imageYUV420CenterCropToFloatBuffer",
            PytorchVisionJni::imageYUV420CenterCropToFloatBuffer),
    });
  }
};
} // namespace pytorch_vision_jni

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  return facebook::jni::initialize(
      vm, [] { pytorch_vision_jni::PytorchVisionJni::registerNatives(); });
}