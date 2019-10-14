package org.pytorch.torchvision;

import android.graphics.Bitmap;
import android.graphics.ImageFormat;
import android.media.Image;

import org.pytorch.Tensor;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.Locale;

/**
 * Contains utility functions for {@link org.pytorch.Tensor} creation from
 * {@link android.graphics.Bitmap} or {@link android.media.Image} source.
 */
public final class TensorImageUtils {

  public static float[] TORCHVISION_NORM_MEAN_RGB = new float[]{0.485f, 0.456f, 0.406f};
  public static float[] TORCHVISION_NORM_STD_RGB = new float[]{0.229f, 0.224f, 0.225f};

  /**
   * Creates new {@link org.pytorch.Tensor} from full {@link android.graphics.Bitmap}, normalized
   * with specified in parameters mean and std.
   *
   * @param normMeanRGB means for RGB channels normalization, length must equal 3, RGB order
   * @param normStdRGB  standard deviation for RGB channels normalization, length must equal 3, RGB order
   */
  public static Tensor bitmapToFloat32Tensor(
      final Bitmap bitmap, final float[] normMeanRGB, final float normStdRGB[]) {
    checkNormMeanArg(normMeanRGB);
    checkNormStdArg(normStdRGB);

    return bitmapToFloat32Tensor(
        bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), normMeanRGB, normStdRGB);
  }

  /**
   * Writes tensor content from specified {@link android.graphics.Bitmap},
   * normalized with specified in parameters mean and std to specified {@link java.nio.FloatBuffer}
   * with specified offset.
   *
   * @param bitmap      {@link android.graphics.Bitmap} as a source for Tensor data
   * @param x           - x coordinate of top left corner of bitmap's area
   * @param y           - y coordinate of top left corner of bitmap's area
   * @param width       - width of bitmap's area
   * @param height      - height of bitmap's area
   * @param normMeanRGB means for RGB channels normalization, length must equal 3, RGB order
   * @param normStdRGB  standard deviation for RGB channels normalization, length must equal 3, RGB order
   */
  public static void bitmapToFloatBuffer(
      final Bitmap bitmap,
      final int x,
      final int y,
      final int width,
      final int height,
      final float[] normMeanRGB,
      final float[] normStdRGB,
      final FloatBuffer outBuffer,
      final int outBufferOffset) {
    checkOutBufferCapacity(outBuffer, outBufferOffset, width, height);
    checkNormMeanArg(normMeanRGB);
    checkNormStdArg(normStdRGB);

    final int pixelsCount = height * width;
    final int[] pixels = new int[pixelsCount];
    bitmap.getPixels(pixels, 0, width, x, y, width, height);
    final int offset_g = pixelsCount;
    final int offset_b = 2 * pixelsCount;
    for (int i = 0; i < pixelsCount; i++) {
      final int c = pixels[i];
      float r = ((c >> 16) & 0xff) / 255.0f;
      float g = ((c >> 8) & 0xff) / 255.0f;
      float b = ((c) & 0xff) / 255.0f;
      float rF = (r - normMeanRGB[0]) / normStdRGB[0];
      float gF = (g - normMeanRGB[1]) / normStdRGB[1];
      float bF = (b - normMeanRGB[2]) / normStdRGB[2];
      outBuffer.put(outBufferOffset + i, rF);
      outBuffer.put(outBufferOffset + offset_g + i, gF);
      outBuffer.put(outBufferOffset + offset_b + i, bF);
    }
  }

  /**
   * Creates new {@link org.pytorch.Tensor} from specified area of {@link android.graphics.Bitmap},
   * normalized with specified in parameters mean and std.
   *
   * @param bitmap      {@link android.graphics.Bitmap} as a source for Tensor data
   * @param x           - x coordinate of top left corner of bitmap's area
   * @param y           - y coordinate of top left corner of bitmap's area
   * @param width       - width of bitmap's area
   * @param height      - height of bitmap's area
   * @param normMeanRGB means for RGB channels normalization, length must equal 3, RGB order
   * @param normStdRGB  standard deviation for RGB channels normalization, length must equal 3, RGB order
   */
  public static Tensor bitmapToFloat32Tensor(
      final Bitmap bitmap,
      int x,
      int y,
      int width,
      int height,
      float[] normMeanRGB,
      float[] normStdRGB) {
    checkNormMeanArg(normMeanRGB);
    checkNormStdArg(normStdRGB);

    final FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(3 * width * height);
    bitmapToFloatBuffer(bitmap, x, y, width, height, normMeanRGB, normStdRGB, floatBuffer, 0);
    return Tensor.fromBlob(floatBuffer, new long[]{1, 3, height, width});
  }

  /**
   * Creates new {@link org.pytorch.Tensor} from specified area of {@link android.media.Image},
   * doing optional rotation, scaling (nearest) and center cropping.
   *
   * @param image           {@link android.media.Image} as a source for Tensor data
   * @param rotateCWDegrees Clockwise angle through which the input image needs to be rotated to be
   *                        upright. Range of valid values: 0, 90, 180, 270
   * @param tensorWidth     return tensor width, must be positive
   * @param tensorHeight    return tensor height, must be positive
   * @param normMeanRGB     means for RGB channels normalization, length must equal 3, RGB order
   * @param normStdRGB      standard deviation for RGB channels normalization, length must equal 3, RGB order
   */
  public static Tensor imageYUV420CenterCropToFloat32Tensor(
      final Image image,
      int rotateCWDegrees,
      final int tensorWidth,
      final int tensorHeight,
      float[] normMeanRGB,
      float[] normStdRGB) {
    if (image.getFormat() != ImageFormat.YUV_420_888) {
      throw new IllegalArgumentException(
          String.format(
              Locale.US, "Image format %d != ImageFormat.YUV_420_888", image.getFormat()));
    }

    checkNormMeanArg(normMeanRGB);
    checkNormStdArg(normStdRGB);
    checkRotateCWDegrees(rotateCWDegrees);
    checkTensorSize(tensorWidth, tensorHeight);

    final FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(3 * tensorWidth * tensorHeight);
    imageYUV420CenterCropToFloatBuffer(
        image,
        rotateCWDegrees,
        tensorWidth,
        tensorHeight,
        normMeanRGB, normStdRGB, floatBuffer, 0);
    return Tensor.fromBlob(floatBuffer, new long[]{1, 3, tensorHeight, tensorWidth});
  }

  /**
   * Writes tensor content from specified {@link android.media.Image}, doing optional rotation,
   * scaling (nearest) and center cropping to specified {@link java.nio.FloatBuffer} with specified offset.
   *
   * @param image           {@link android.media.Image} as a source for Tensor data
   * @param rotateCWDegrees Clockwise angle through which the input image needs to be rotated to be
   *                        upright. Range of valid values: 0, 90, 180, 270
   * @param tensorWidth     return tensor width, must be positive
   * @param tensorHeight    return tensor height, must be positive
   * @param normMeanRGB     means for RGB channels normalization, length must equal 3, RGB order
   * @param normStdRGB      standard deviation for RGB channels normalization, length must equal 3, RGB order
   * @param outBuffer       Output buffer, where tensor content will be written
   * @param outBufferOffset Output buffer offset with which tensor content will be written
   */
  public static void imageYUV420CenterCropToFloatBuffer(
      final Image image,
      int rotateCWDegrees,
      final int tensorWidth,
      final int tensorHeight,
      float[] normMeanRGB,
      float[] normStdRGB,
      final FloatBuffer outBuffer,
      final int outBufferOffset) {
    checkOutBufferCapacity(outBuffer, outBufferOffset, tensorWidth, tensorHeight);

    if (image.getFormat() != ImageFormat.YUV_420_888) {
      throw new IllegalArgumentException(
          String.format(
              Locale.US, "Image format %d != ImageFormat.YUV_420_888", image.getFormat()));
    }

    checkNormMeanArg(normMeanRGB);
    checkNormStdArg(normStdRGB);
    checkRotateCWDegrees(rotateCWDegrees);
    checkTensorSize(tensorWidth, tensorHeight);

    final int widthBeforeRotation = image.getWidth();
    final int heightBeforeRotation = image.getHeight();

    int widthAfterRotation = widthBeforeRotation;
    int heightAfterRotation = heightBeforeRotation;
    if (rotateCWDegrees == 90 || rotateCWDegrees == 270) {
      widthAfterRotation = heightBeforeRotation;
      heightAfterRotation = widthBeforeRotation;
    }

    int centerCropWidthAfterRotation = widthAfterRotation;
    int centerCropHeightAfterRotation = heightAfterRotation;

    if (tensorWidth * heightAfterRotation <= tensorHeight * widthAfterRotation) {
      centerCropWidthAfterRotation =
          (int) Math.floor((float) tensorWidth * heightAfterRotation / tensorHeight);
    } else {
      centerCropHeightAfterRotation =
          (int) Math.floor((float) tensorHeight * widthAfterRotation / tensorWidth);
    }

    int centerCropWidthBeforeRotation = centerCropWidthAfterRotation;
    int centerCropHeightBeforeRotation = centerCropHeightAfterRotation;
    if (rotateCWDegrees == 90 || rotateCWDegrees == 270) {
      centerCropHeightBeforeRotation = centerCropWidthAfterRotation;
      centerCropWidthBeforeRotation = centerCropHeightAfterRotation;
    }

    final int offsetX =
        (int) Math.floor((widthBeforeRotation - centerCropWidthBeforeRotation) / 2.f);
    final int offsetY =
        (int) Math.floor((heightBeforeRotation - centerCropHeightBeforeRotation) / 2.f);

    final Image.Plane yPlane = image.getPlanes()[0];
    final Image.Plane uPlane = image.getPlanes()[1];
    final Image.Plane vPlane = image.getPlanes()[2];

    final ByteBuffer yBuffer = yPlane.getBuffer();
    final ByteBuffer uBuffer = uPlane.getBuffer();
    final ByteBuffer vBuffer = vPlane.getBuffer();

    final int yRowStride = yPlane.getRowStride();
    final int uRowStride = uPlane.getRowStride();

    final int yPixelStride = yPlane.getPixelStride();
    final int uPixelStride = uPlane.getPixelStride();

    final float scale = (float) centerCropWidthAfterRotation / tensorWidth;
    final int uvRowStride = uRowStride >> 1;

    final int channelSize = tensorHeight * tensorWidth;
    final int tensorInputOffsetG = channelSize;
    final int tensorInputOffsetB = 2 * channelSize;
    for (int x = 0; x < tensorWidth; x++) {
      for (int y = 0; y < tensorHeight; y++) {

        final int centerCropXAfterRotation = (int) Math.floor(x * scale);
        final int centerCropYAfterRotation = (int) Math.floor(y * scale);

        int xBeforeRotation = offsetX + centerCropXAfterRotation;
        int yBeforeRotation = offsetY + centerCropYAfterRotation;
        if (rotateCWDegrees == 90) {
          xBeforeRotation = offsetX + centerCropYAfterRotation;
          yBeforeRotation =
              offsetY + (centerCropHeightBeforeRotation - 1) - centerCropXAfterRotation;
        } else if (rotateCWDegrees == 180) {
          xBeforeRotation =
              offsetX + (centerCropWidthBeforeRotation - 1) - centerCropXAfterRotation;
          yBeforeRotation =
              offsetY + (centerCropHeightBeforeRotation - 1) - centerCropYAfterRotation;
        } else if (rotateCWDegrees == 270) {
          xBeforeRotation =
              offsetX + (centerCropWidthBeforeRotation - 1) - centerCropYAfterRotation;
          yBeforeRotation = offsetY + centerCropXAfterRotation;
        }

        final int yIdx = yBeforeRotation * yRowStride + xBeforeRotation * yPixelStride;
        final int uvIdx = (yBeforeRotation >> 1) * uvRowStride + xBeforeRotation * uPixelStride;

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
        final int offset = outBufferOffset + y * tensorWidth + x;
        float rF = ((r / 255.f) - normMeanRGB[0]) / normStdRGB[0];
        float gF = ((g / 255.f) - normMeanRGB[1]) / normStdRGB[1];
        float bF = ((b / 255.f) - normMeanRGB[2]) / normStdRGB[2];

        outBuffer.put(offset, rF);
        outBuffer.put(offset + tensorInputOffsetG, gF);
        outBuffer.put(offset + tensorInputOffsetB, bF);
      }
    }
  }

  private static void checkOutBufferCapacity(FloatBuffer outBuffer, int outBufferOffset, int tensorWidth, int tensorHeight) {
    if (outBufferOffset + 3 * tensorWidth * tensorHeight > outBuffer.capacity()) {
      throw new IllegalStateException("Buffer underflow");
    }
  }

  private static void checkTensorSize(int tensorWidth, int tensorHeight) {
    if (tensorHeight <= 0 || tensorWidth <= 0) {
      throw new IllegalArgumentException("tensorHeight and tensorWidth must be positive");
    }
  }

  private static void checkRotateCWDegrees(int rotateCWDegrees) {
    if (rotateCWDegrees != 0
        && rotateCWDegrees != 90
        && rotateCWDegrees != 180
        && rotateCWDegrees != 270) {
      throw new IllegalArgumentException("rotateCWDegrees must be one of 0, 90, 180, 270");
    }
  }

  private static final int clamp(int c, int min, int max) {
    return c < min ? min : c > max ? max : c;
  }

  private static void checkNormStdArg(float[] normStdRGB) {
    if (normStdRGB.length != 3) {
      throw new IllegalArgumentException("normStdRGB length must be 3");
    }
  }

  private static void checkNormMeanArg(float[] normMeanRGB) {
    if (normMeanRGB.length != 3) {
      throw new IllegalArgumentException("normMeanRGB length must be 3");
    }
  }
}
