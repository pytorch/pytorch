package org.pytorch.testapp;

import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Log;
import android.widget.TextView;
import androidx.annotation.Nullable;
import androidx.annotation.UiThread;
import androidx.annotation.WorkerThread;
import androidx.appcompat.app.AppCompatActivity;
import java.nio.FloatBuffer;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.PyTorchAndroid;
import org.pytorch.Tensor;

public class MainActivity extends AppCompatActivity {

  private static final String TAG = BuildConfig.LOGCAT_TAG;
  private static final int TEXT_TRIM_SIZE = 4096;

  private TextView mTextView;

  protected HandlerThread mBackgroundThread;
  protected Handler mBackgroundHandler;
  private Module mModule;
  private FloatBuffer mInputTensorBuffer;
  private Tensor mInputTensor;
  private StringBuilder mTextViewStringBuilder = new StringBuilder();

  private final Runnable mModuleForwardRunnable =
      new Runnable() {
        @Override
        public void run() {
          final Result result = doModuleForward();
          runOnUiThread(
              new Runnable() {
                @Override
                public void run() {
                  handleResult(result);
                  if (mBackgroundHandler != null) {
                    mBackgroundHandler.post(mModuleForwardRunnable);
                  }
                }
              });
        }
      };

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    mTextView = findViewById(R.id.text);
    startBackgroundThread();
    mBackgroundHandler.post(mModuleForwardRunnable);
  }

  protected void startBackgroundThread() {
    mBackgroundThread = new HandlerThread(TAG + "_bg");
    mBackgroundThread.start();
    mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
  }

  @Override
  protected void onDestroy() {
    stopBackgroundThread();
    super.onDestroy();
  }

  protected void stopBackgroundThread() {
    mBackgroundThread.quitSafely();
    try {
      mBackgroundThread.join();
      mBackgroundThread = null;
      mBackgroundHandler = null;
    } catch (InterruptedException e) {
      Log.e(TAG, "Error stopping background thread", e);
    }
  }

  @WorkerThread
  @Nullable
  protected Result doModuleForward() {
    if (mModule == null) {
      final long[] shape = BuildConfig.INPUT_TENSOR_SHAPE;
      long numElements = 1;
      for (int i = 0; i < shape.length; i++) {
        numElements *= shape[i];
      }
      mInputTensorBuffer = Tensor.allocateFloatBuffer((int) numElements);
      mInputTensor = Tensor.fromBlob(mInputTensorBuffer, BuildConfig.INPUT_TENSOR_SHAPE);
      PyTorchAndroid.setNumThreads(1);
      mModule = PyTorchAndroid.loadModuleFromAsset(getAssets(), BuildConfig.MODULE_ASSET_NAME);
    }

    final long startTime = SystemClock.elapsedRealtime();
    final long moduleForwardStartTime = SystemClock.elapsedRealtime();
    final Tensor outputTensor = mModule.forward(IValue.from(mInputTensor)).toTensor();
    final long moduleForwardDuration = SystemClock.elapsedRealtime() - moduleForwardStartTime;
    final float[] scores = outputTensor.getDataAsFloatArray();
    final long analysisDuration = SystemClock.elapsedRealtime() - startTime;

    return new Result(scores, moduleForwardDuration, analysisDuration);
  }

  static class Result {

    private final float[] scores;
    private final long totalDuration;
    private final long moduleForwardDuration;

    public Result(float[] scores, long moduleForwardDuration, long totalDuration) {
      this.scores = scores;
      this.moduleForwardDuration = moduleForwardDuration;
      this.totalDuration = totalDuration;
    }
  }

  @UiThread
  protected void handleResult(Result result) {
    String message = String.format("forwardDuration:%d", result.moduleForwardDuration);
    Log.i(TAG, message);
    mTextViewStringBuilder.insert(0, '\n').insert(0, message);
    if (mTextViewStringBuilder.length() > TEXT_TRIM_SIZE) {
      mTextViewStringBuilder.delete(TEXT_TRIM_SIZE, mTextViewStringBuilder.length());
    }
    mTextView.setText(mTextViewStringBuilder.toString());
  }
}
