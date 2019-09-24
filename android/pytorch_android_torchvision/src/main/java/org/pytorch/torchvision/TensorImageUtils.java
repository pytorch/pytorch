package org.pytorch.torchvision;

import android.graphics.Bitmap;
import android.graphics.ImageFormat;
import android.media.Image;

import org.pytorch.Tensor;

import java.nio.ByteBuffer;
import java.util.Locale;

public final class TensorImageUtils {
  private static float NORM_MEAN_R = 0.485f;
  private static float NORM_MEAN_G = 0.456f;
  private static float NORM_MEAN_B = 0.406f;

  private static float NORM_STD_R = 0.229f;
  private static float NORM_STD_G = 0.224f;
  private static float NORM_STD_B = 0.225f;

  public static Tensor bitmapToFloatTensorTorchVisionForm(final Bitmap bitmap) {
    return bitmapToFloatTensorTorchVisionForm(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight());
  }

  public static Tensor bitmapToFloatTensorTorchVisionForm(
      final Bitmap bitmap, int x, int y, int width, int height) {
    final int pixelsSize = height * width;
    final int[] pixels = new int[pixelsSize];
    bitmap.getPixels(pixels, 0, width, x, y, width, height);
    final float[] floatArray = new float[3 * pixelsSize];
    final int offset_g = pixelsSize;
    final int offset_b = 2 * pixelsSize;
    for (int i = 0; i < pixelsSize; i++) {
      final int c = pixels[i];
      float r = ((c >> 16) & 0xff) / 255.0f;
      float g = ((c >> 8) & 0xff) / 255.0f;
      float b = ((c) & 0xff) / 255.0f;
      floatArray[i] = (r - NORM_MEAN_R) / NORM_STD_R;
      floatArray[offset_g + i] = (g - NORM_MEAN_G) / NORM_STD_G;
      floatArray[offset_b + i] = (b - NORM_MEAN_B) / NORM_STD_B;
    }
    final long shape[] = new long[] {1, 3, height, width};
    return Tensor.newFloat32Tensor(shape, floatArray);
  }

  public static Tensor imageYUV420CenterCropToFloatTensorTorchVisionForm(
      final Image image, int rotateCWDegrees, final int tensorWidth, final int tensorHeight) {
    if (image.getFormat() != ImageFormat.YUV_420_888) {
      throw new IllegalArgumentException(
          String.format(
              Locale.US, "Image format %d != ImageFormat.YUV_420_888", image.getFormat()));
    }

    final int width = image.getWidth();
    final int height = image.getHeight();
    int offsetX = 0;
    int offsetY = 0;
    int centerCropSize;
    if (width > height) {
      offsetX = (int) Math.floor((width - height) / 2.f);
      centerCropSize = height;
    } else {
      offsetY = (int) Math.floor((height - width) / 2.f);
      centerCropSize = width;
    }

    final Image.Plane yPlane = image.getPlanes()[0];
    final Image.Plane uPlane = image.getPlanes()[1];
    final Image.Plane vPlane = image.getPlanes()[2];

    final ByteBuffer yBuffer = yPlane.getBuffer();
    final ByteBuffer uBuffer = uPlane.getBuffer();
    final ByteBuffer vBuffer = vPlane.getBuffer();

    int yRowStride = yPlane.getRowStride();
    int uRowStride = uPlane.getRowStride();

    int yPixelStride = yPlane.getPixelStride();
    int uPixelStride = uPlane.getPixelStride();

    float tx = (float) centerCropSize / tensorWidth;
    float ty = (float) centerCropSize / tensorHeight;
    int uvRowStride = uRowStride >> 1;

    int cSize = tensorHeight * tensorWidth;
    final int tensorInputOffsetG = cSize;
    final int tensorInputOffsetB = 2 * centerCropSize;
    final float[] floatArray = new float[3 * cSize];

    for (int x = 0; x < tensorWidth; x++) {
      for (int y = 0; y < tensorHeight; y++) {

        // scaling as nearest
        final int centerCropX = (int) Math.floor(x * tx);
        final int centerCropY = (int) Math.floor(y * ty);

        int srcX = centerCropY + offsetX;
        int srcY = (centerCropSize - 1) - centerCropX + offsetY;

        if (rotateCWDegrees == 90) {
          srcX = offsetX + centerCropY;
          srcY = offsetY + (centerCropSize - 1) - centerCropX;
        } else if (rotateCWDegrees == 180) {
          srcX = offsetX + (centerCropSize - 1) - centerCropX;
          srcY = offsetY + (centerCropSize - 1) - centerCropY;
        } else if (rotateCWDegrees == 270) {
          srcX = offsetX + (centerCropSize - 1) - centerCropY;
          srcY = offsetY + centerCropX;
        }

        final int yIdx = srcY * yRowStride + srcX * yPixelStride;
        final int uvIdx = (srcY >> 1) * uvRowStride + srcX * uPixelStride;

        int Yi = yBuffer.get(yIdx) & 0xff;
        int Ui = uBuffer.get(uvIdx) & 0xff;
        int Vi = vBuffer.get(uvIdx) & 0xff;

        int a0 = 1192 * (Yi - 16);
        int a1 = 1634 * (Vi - 128);
        int a2 = 832 * (Vi - 128);
        int a3 = 400 * (Ui - 128);
        int a4 = 2066 * (Ui - 128);

        int r = clamp((a0 + a1) >> 10, 0, 255);
        int g = clamp((a0 - a2 - a3) >> 10, 0, 255);
        int b = clamp((a0 + a4) >> 10, 0, 255);
        final int offset = y * tensorWidth + x;
        floatArray[offset] = ((r / 255.f) - NORM_MEAN_R) / NORM_STD_R;
        floatArray[tensorInputOffsetG + offset] = ((g / 255.f) - NORM_MEAN_G) / NORM_STD_G;
        floatArray[tensorInputOffsetB + offset] = ((b / 255.f) - NORM_MEAN_B) / NORM_STD_B;
      }
    }
    final long shape[] = new long[] {1, 3, tensorHeight, tensorHeight};
    return Tensor.newFloat32Tensor(shape, floatArray);
  }

  private static final int clamp(int c, int min, int max) {
    return c < min ? min : c > max ? max : c;
  }
}
