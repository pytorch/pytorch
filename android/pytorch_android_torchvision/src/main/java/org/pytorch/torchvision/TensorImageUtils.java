package org.pytorch.torchvision;

import android.graphics.Bitmap;
import android.graphics.ImageFormat;
import android.media.Image;
import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.Locale;
import org.pytorch.MemoryFormat;
import org.pytorch.Tensor;

/**
 * Contains utility functions for {@link org.pytorch.Tensor} creation from {@link
 * android.graphics.Bitmap} or {@link android.media.Image} source.
 */
public final class TensorImageUtils {

  public static float[] TORCHVISION_NORM_MEAN_RGB = new float[] {0.485f, 0.456f, 0.406f};
  public static float[] TORCHVISION_NORM_STD_RGB = new float[] {0.229f, 0.224f, 0.225f};

  /**
   * Creates new {@link org.pytorch.Tensor} from full {@link android.graphics.Bitmap}, normalized
   * with specified in parameters mean and std.
   *
   * @param normMeanRGB means for RGB channels normalization, length must equal 3, RGB order
   * @param normStdRGB standard deviation for RGB channels normalization, length must equal 3, RGB
   *     order
   */
  public static Tensor bitmapToFloat32Tensor(
      final Bitmap bitmap, final float[] normMeanRGB, final float normStdRGB[], final MemoryFormat memoryFormat) {
    checkNormMeanArg(normMeanRGB);
    checkNormStdArg(normStdRGB);

    return bitmapToFloat32Tensor(
        bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), normMeanRGB, normStdRGB, memoryFormat);
  }

  public static Tensor bitmapToFloat32Tensor(
      final Bitmap bitmap, final float[] normMeanRGB, final float normStdRGB[]) {
    return bitmapToFloat32Tensor(
        bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), normMeanRGB, normStdRGB, MemoryFormat.CONTIGUOUS);
  }

  /**
   * Writes tensor content from specified {@link android.graphics.Bitmap}, normalized with specified
   * in parameters mean and std to specified {@link java.nio.FloatBuffer} with specified offset.
   *
   * @param bitmap {@link android.graphics.Bitmap} as a source for Tensor data
   * @param x - x coordinate of top left corner of bitmap's area
   * @param y - y coordinate of top left corner of bitmap's area
   * @param width - width of bitmap's area
   * @param height - height of bitmap's area
   * @param normMeanRGB means for RGB channels normalization, length must equal 3, RGB order
   * @param normStdRGB standard deviation for RGB channels normalization, length must equal 3, RGB
   *     order
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
      final int outBufferOffset,
      final MemoryFormat memoryFormat) {
    checkOutBufferCapacity(outBuffer, outBufferOffset, width, height);
    checkNormMeanArg(normMeanRGB);
    checkNormStdArg(normStdRGB);
    if (memoryFormat != MemoryFormat.CONTIGUOUS && memoryFormat != MemoryFormat.CHANNELS_LAST) {
      throw new IllegalArgumentException("Unsupported memory format " + memoryFormat);
    }

    final int pixelsCount = height * width;
    final int[] pixels = new int[pixelsCount];
    bitmap.getPixels(pixels, 0, width, x, y, width, height);
    if (MemoryFormat.CONTIGUOUS == memoryFormat) {
      final int offset_g = pixelsCount;
      final int offset_b = 2 * pixelsCount;
      for (int i = 0; i < pixelsCount; i++) {
        final int c = pixels[i];
        float r = ((c >> 16) & 0xff) / 255.0f;
        float g = ((c >> 8) & 0xff) / 255.0f;
        float b = ((c) & 0xff) / 255.0f;
        outBuffer.put(outBufferOffset + i, (r - normMeanRGB[0]) / normStdRGB[0]);
        outBuffer.put(outBufferOffset + offset_g + i, (g - normMeanRGB[1]) / normStdRGB[1]);
        outBuffer.put(outBufferOffset + offset_b + i, (b - normMeanRGB[2]) / normStdRGB[2]);
      }
    } else {
      for (int i = 0; i < pixelsCount; i++) {
        final int c = pixels[i];
        float r = ((c >> 16) & 0xff) / 255.0f;
        float g = ((c >> 8) & 0xff) / 255.0f;
        float b = ((c) & 0xff) / 255.0f;
        outBuffer.put(outBufferOffset + 3 * i + 0, (r - normMeanRGB[0]) / normStdRGB[0]);
        outBuffer.put(outBufferOffset + 3 * i + 1, (g - normMeanRGB[1]) / normStdRGB[1]);
        outBuffer.put(outBufferOffset + 3 * i + 2, (b - normMeanRGB[2]) / normStdRGB[2]);
      }
    }
  }

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
    bitmapToFloatBuffer(bitmap, x, y, width, height, normMeanRGB, normStdRGB, outBuffer, outBufferOffset, MemoryFormat.CONTIGUOUS);
  }

  /**
   * Creates new {@link org.pytorch.Tensor} from specified area of {@link android.graphics.Bitmap},
   * normalized with specified in parameters mean and std.
   *
   * @param bitmap {@link android.graphics.Bitmap} as a source for Tensor data
   * @param x - x coordinate of top left corner of bitmap's area
   * @param y - y coordinate of top left corner of bitmap's area
   * @param width - width of bitmap's area
   * @param height - height of bitmap's area
   * @param normMeanRGB means for RGB channels normalization, length must equal 3, RGB order
   * @param normStdRGB standard deviation for RGB channels normalization, length must equal 3, RGB
   *     order
   */
  public static Tensor bitmapToFloat32Tensor(
      final Bitmap bitmap,
      int x,
      int y,
      int width,
      int height,
      float[] normMeanRGB,
      float[] normStdRGB,
      MemoryFormat memoryFormat) {
    checkNormMeanArg(normMeanRGB);
    checkNormStdArg(normStdRGB);

    final FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(3 * width * height);
    bitmapToFloatBuffer(bitmap, x, y, width, height, normMeanRGB, normStdRGB, floatBuffer, 0, memoryFormat);
    return Tensor.fromBlob(floatBuffer, new long[] {1, 3, height, width}, memoryFormat);
  }

  public static Tensor bitmapToFloat32Tensor(
      final Bitmap bitmap,
      int x,
      int y,
      int width,
      int height,
      float[] normMeanRGB,
      float[] normStdRGB) {
    return bitmapToFloat32Tensor(bitmap, x, y, width, height, normMeanRGB, normStdRGB, MemoryFormat.CONTIGUOUS);
  }

  /**
   * Creates new {@link org.pytorch.Tensor} from specified area of {@link android.media.Image},
   * doing optional rotation, scaling (nearest) and center cropping.
   *
   * @param image {@link android.media.Image} as a source for Tensor data
   * @param rotateCWDegrees Clockwise angle through which the input image needs to be rotated to be
   *     upright. Range of valid values: 0, 90, 180, 270
   * @param tensorWidth return tensor width, must be positive
   * @param tensorHeight return tensor height, must be positive
   * @param normMeanRGB means for RGB channels normalization, length must equal 3, RGB order
   * @param normStdRGB standard deviation for RGB channels normalization, length must equal 3, RGB
   *     order
   */
  public static Tensor imageYUV420CenterCropToFloat32Tensor(
      final Image image,
      int rotateCWDegrees,
      final int tensorWidth,
      final int tensorHeight,
      float[] normMeanRGB,
      float[] normStdRGB,
      MemoryFormat memoryFormat) {
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
        image, rotateCWDegrees, tensorWidth, tensorHeight, normMeanRGB, normStdRGB, floatBuffer, 0, memoryFormat);
    return Tensor.fromBlob(floatBuffer, new long[] {1, 3, tensorHeight, tensorWidth}, memoryFormat);
  }

  public static Tensor imageYUV420CenterCropToFloat32Tensor(
      final Image image,
      int rotateCWDegrees,
      final int tensorWidth,
      final int tensorHeight,
      float[] normMeanRGB,
      float[] normStdRGB) {
    return imageYUV420CenterCropToFloat32Tensor(image, rotateCWDegrees, tensorWidth, tensorHeight, normMeanRGB, normStdRGB, MemoryFormat.CONTIGUOUS);
  }

  /**
   * Writes tensor content from specified {@link android.media.Image}, doing optional rotation,
   * scaling (nearest) and center cropping to specified {@link java.nio.FloatBuffer} with specified
   * offset.
   *
   * @param image {@link android.media.Image} as a source for Tensor data
   * @param rotateCWDegrees Clockwise angle through which the input image needs to be rotated to be
   *     upright. Range of valid values: 0, 90, 180, 270
   * @param tensorWidth return tensor width, must be positive
   * @param tensorHeight return tensor height, must be positive
   * @param normMeanRGB means for RGB channels normalization, length must equal 3, RGB order
   * @param normStdRGB standard deviation for RGB channels normalization, length must equal 3, RGB
   *     order
   * @param outBuffer Output buffer, where tensor content will be written
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
      final int outBufferOffset,
      final MemoryFormat memoryFormat) {
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

    Image.Plane[] planes = image.getPlanes();
    Image.Plane Y = planes[0];
    Image.Plane U = planes[1];
    Image.Plane V = planes[2];

    int memoryFormatJniCode = 0;
    if (MemoryFormat.CONTIGUOUS == memoryFormat) {
      memoryFormatJniCode = 1;
    } else if (MemoryFormat.CHANNELS_LAST == memoryFormat) {
      memoryFormatJniCode = 2;
    }

    NativePeer.imageYUV420CenterCropToFloatBuffer(
        Y.getBuffer(),
        Y.getRowStride(),
        Y.getPixelStride(),
        U.getBuffer(),
        V.getBuffer(),
        U.getRowStride(),
        U.getPixelStride(),
        image.getWidth(),
        image.getHeight(),
        rotateCWDegrees,
        tensorWidth,
        tensorHeight,
        normMeanRGB,
        normStdRGB,
        outBuffer,
        outBufferOffset,
        memoryFormatJniCode);
  }

  public static void imageYUV420CenterCropToFloatBuffer(
      final Image image,
      int rotateCWDegrees,
      final int tensorWidth,
      final int tensorHeight,
      float[] normMeanRGB,
      float[] normStdRGB,
      final FloatBuffer outBuffer,
      final int outBufferOffset) {
    imageYUV420CenterCropToFloatBuffer(image, rotateCWDegrees, tensorWidth, tensorHeight, normMeanRGB, normStdRGB, outBuffer, outBufferOffset, MemoryFormat.CONTIGUOUS);
  }

  private static class NativePeer {
    static {
      if (!NativeLoader.isInitialized()) {
        NativeLoader.init(new SystemDelegate());
      }
      NativeLoader.loadLibrary("pytorch_vision_jni");
    }

    private static native void imageYUV420CenterCropToFloatBuffer(
        ByteBuffer yBuffer,
        int yRowStride,
        int yPixelStride,
        ByteBuffer uBuffer,
        ByteBuffer vBuffer,
        int uvRowStride,
        int uvPixelStride,
        int imageWidth,
        int imageHeight,
        int rotateCWDegrees,
        int tensorWidth,
        int tensorHeight,
        float[] normMeanRgb,
        float[] normStdRgb,
        Buffer outBuffer,
        int outBufferOffset,
        int memoryFormatJniCode);
  }

  private static void checkOutBufferCapacity(
      FloatBuffer outBuffer, int outBufferOffset, int tensorWidth, int tensorHeight) {
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
