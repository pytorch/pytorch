#include <cassert>
#include <cmath>
#include <vector>

#include "jni.h"

#if defined(__ANDROID__)

#include <android/log.h>
#define ALOGI(...) \
  __android_log_print(ANDROID_LOG_INFO, "pytorch-vision-jni", __VA_ARGS__)

#endif
#define clamp0255(x) x > 255 ? 255 : x < 0 ? 0 : x

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL
Java_org_pytorch_torchvision_TensorImageUtils_00024NativePeer_imageYUV420CenterCropToFloatBuffer(
    JNIEnv* jniEnv,
    jclass,
    jobject yBuffer,
    jint yRowStride,
    jint yPixelStride,
    jobject uBuffer,
    jobject vBuffer,
    jint uRowStride,
    jint uvPixelStride,
    jint imageWidth,
    jint imageHeight,
    jint rotateCWDegrees,
    jint tensorWidth,
    jint tensorHeight,
    jfloatArray jnormMeanRGB,
    jfloatArray jnormStdRGB,
    jobject outBuffer,
    jint outOffset) {
  float* outData = (float*)jniEnv->GetDirectBufferAddress(outBuffer);

  auto normMeanRGB = jniEnv->GetFloatArrayElements(jnormMeanRGB, 0);
  auto normStdRGB = jniEnv->GetFloatArrayElements(jnormStdRGB, 0);

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

  const uint8_t* yData = (uint8_t*)jniEnv->GetDirectBufferAddress(yBuffer);
  const uint8_t* uData = (uint8_t*)jniEnv->GetDirectBufferAddress(uBuffer);
  const uint8_t* vData = (uint8_t*)jniEnv->GetDirectBufferAddress(vBuffer);

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

#ifdef __cplusplus
}
#endif
