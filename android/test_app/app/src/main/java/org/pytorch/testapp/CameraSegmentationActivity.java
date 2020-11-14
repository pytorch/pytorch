package org.pytorch.testapp;

import android.graphics.*;
import android.media.Image;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import androidx.annotation.UiThread;
import org.pytorch.torchvision.TensorImageUtils;
import androidx.camera.core.CameraX;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageAnalysisConfig;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.PreviewConfig;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

public class CameraSegmentationActivity extends CameraActivity {
  private static final String TAG = BuildConfig.LOGCAT_TAG;
  private static final int CIRCLE_RADIUS = 3;
  private static final double THRESHOLD = 0.7;

  private ImageView mCameraOverlayView;

  private static final float[] INPUT_TENSOR_YUV_NORM_MEAN = new float[] {0.46684004f, -0.01768881f, 0.02804728f};
  private static final float[] INPUT_TENSOR_YUV_NORM_STD = new float[] {0.24543354f, 0.03908971f, 0.06280618f};
  private Paint mResultPaint;

  private Bitmap mInputTensorBitmap;

  private Bitmap mOutputBitmap;
  private float mScaleX;
  private float mScaleY;
  private int mTensorBitmapLeft;
  private int mTensorBitmapTop;
  private int mTensorBitmapRight;
  private int mTensorBitmapBottom;

  protected boolean useFaceCamera() {
    return true;
  }

  private int clamp0255(int x) {
    if (x > 255) {
      return 255;
    }
    return x < 0 ? 0 : x;
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    mCameraOverlayView = findViewById(R.id.camera_overlay);

    mResultPaint = new Paint();
    mResultPaint.setAntiAlias(true);
    mResultPaint.setDither(true);
    mResultPaint.setColor(Color.GREEN);
    mResultPaint.setAlpha(224);
    mResultPaint.setStyle(Paint.Style.FILL_AND_STROKE);

    mInputTensorBitmap = Bitmap.createBitmap(getTensorSize(), getTensorSize(), Bitmap.Config.ARGB_8888);
  }

  protected void fillInputTensorBuffer(
      ImageProxy image,
      int rotationDegreesArg,
      FloatBuffer inputTensorBuffer) {
    int rotationDegrees = 270;
    ImageProxy.PlaneProxy[] planes = image.getPlanes();
    ImageProxy.PlaneProxy Y = planes[0];
    ImageProxy.PlaneProxy U = planes[1];
    ImageProxy.PlaneProxy V = planes[2];
    ByteBuffer yBuffer = Y.getBuffer();
    ByteBuffer uBuffer = U.getBuffer();
    ByteBuffer vBuffer = V.getBuffer();
    final int imageWidth = image.getWidth();
    final int imageHeight = image.getHeight();
    final int tensorSize = getTensorSize();

    int widthAfterRtn = imageWidth;
    int heightAfterRtn = imageHeight;
    boolean oddRotation = rotationDegrees == 90 || rotationDegrees == 270;
    if (oddRotation) {
      widthAfterRtn = imageHeight;
      heightAfterRtn = imageWidth;
    }

    int minSizeAfterRtn = Math.min(heightAfterRtn, widthAfterRtn);
    int cropWidthAfterRtn = minSizeAfterRtn;
    int cropHeightAfterRtn = minSizeAfterRtn;

    int cropWidthBeforeRtn = cropWidthAfterRtn;
    int cropHeightBeforeRtn = cropHeightAfterRtn;
    if (oddRotation) {
      cropWidthBeforeRtn = cropHeightAfterRtn;
      cropHeightBeforeRtn = cropWidthAfterRtn;
    }

    int offsetX = (int)((imageWidth - cropWidthBeforeRtn) / 2.f);
    int offsetY = (int)((imageHeight - cropHeightBeforeRtn) / 2.f);

    int yRowStride = Y.getRowStride();
    int yPixelStride = Y.getPixelStride();
    int uRowStride = U.getRowStride();
    int uvPixelStride = U.getPixelStride();

    float scale = cropWidthAfterRtn / tensorSize;
    int uvRowStride = uRowStride >> 1;
    int yIdx, uvIdx, yi, ui, vi;
    float yf, uf, vf;
    final int channelSize = tensorSize * tensorSize;

    for (int x = 0; x < tensorSize; x++) {
      for (int y = 0; y < tensorSize; y++) {
        final int centerCropX = (int) Math.floor(x * scale);
        final int centerCropY = (int) Math.floor(y * scale);
        int srcX = centerCropY + offsetX;
        int srcY = (minSizeAfterRtn - 1) - centerCropX + offsetY;

        if (rotationDegrees == 90) {
          srcX = offsetX + centerCropY;
          srcY = offsetY + (minSizeAfterRtn - 1) - centerCropX;
        } else if (rotationDegrees == 180) {
          srcX = offsetX + (minSizeAfterRtn - 1) - centerCropX;
          srcY = offsetY + (minSizeAfterRtn - 1) - centerCropY;
        } else if (rotationDegrees == 270) {
          srcX = offsetX + (minSizeAfterRtn - 1) - centerCropY;
          srcY = offsetY + centerCropX;
        }

        yIdx = srcY * yRowStride + srcX * yPixelStride;
        uvIdx = (srcY >> 1) * uvRowStride + srcX * uvPixelStride;
        yi = yBuffer.get(yIdx) & 0xff;
        ui = uBuffer.get(uvIdx) & 0xff;
        vi = vBuffer.get(uvIdx) & 0xff;

        int a0 = 1192 * (yi - 16);
        int ri = clamp0255((a0 + 1634 * (vi - 128)) >> 10);
        int gi = clamp0255((a0 - 832 * (vi - 128) - 400 * (ui - 128)) >> 10);
        int bi = clamp0255((a0 + 2066 * (ui - 128)) >> 10);

        mInputTensorBitmap.setPixel(x, y, Color.argb(255, ri, gi, bi));
        yf = yi / 255.f;
        uf = ui / 255.f;
        vf = vi / 255.f;

        final int idx0 = y * tensorSize + x;
        final int idx1 = idx0 + channelSize;
        final int idx2 = idx1 + channelSize;

        inputTensorBuffer.put(idx0, (yf - INPUT_TENSOR_YUV_NORM_MEAN[0]) / INPUT_TENSOR_YUV_NORM_STD[0]);
        inputTensorBuffer.put(idx1, (uf - INPUT_TENSOR_YUV_NORM_MEAN[1]) / INPUT_TENSOR_YUV_NORM_STD[1]);
        inputTensorBuffer.put(idx2, (vf - INPUT_TENSOR_YUV_NORM_MEAN[2]) / INPUT_TENSOR_YUV_NORM_STD[2]);
      }
    }
  }

  private static double sigmoid(double x) {
    return (1 / (1 + Math.pow(Math.E, (-1 * x))));
  }

  protected int getTensorSize() {
    return 96;
  }

  @UiThread
  protected void handleResult(Result result) {
    if (mOutputBitmap == null) {
      final int W = mCameraOverlayView.getMeasuredWidth();
      final int H = mCameraOverlayView.getMeasuredHeight();
      final int size = Math.min(W, H);
      mOutputBitmap = Bitmap.createBitmap(W, H, Bitmap.Config.ARGB_8888);
      final int offsetX = (W - size) / 2;
      final int offsetY = (H - size) / 2;

      mTensorBitmapLeft = offsetX;
      mTensorBitmapTop = offsetY;
      mTensorBitmapRight = offsetX + size;
      mTensorBitmapBottom = offsetY + size;

      mScaleX = (float) size / getTensorSize();
      mScaleY = (float) size / getTensorSize();
    }

    final Canvas canvas = new Canvas(mOutputBitmap);

    canvas.drawBitmap(
        mInputTensorBitmap,
        new Rect(0, 0, mInputTensorBitmap.getWidth(), mInputTensorBitmap.getHeight()),
        new Rect(mTensorBitmapLeft, mTensorBitmapTop, mTensorBitmapRight, mTensorBitmapBottom),
        null
    );

    int idx = 0;
    for (int yi = 0; yi < getTensorSize(); ++yi) {
      for (int xi = 0; xi < getTensorSize(); ++xi) {
        idx = yi * getTensorSize() + xi;
        final float logit = result.scores[idx];
        final float p = (float) sigmoid(logit);
        if (p < THRESHOLD) {
          float cx = mTensorBitmapLeft + mScaleX * xi;
          float cy = mTensorBitmapTop + mScaleY * yi;
          canvas.drawCircle(cx, cy, (int) ((1 - p) * CIRCLE_RADIUS), mResultPaint);
        }
      }
    }

    mCameraOverlayView.setImageBitmap(mOutputBitmap);
  }
}
