package org.pytorch;

import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;

public final class PyTorchAndroid {
  static {
    if (!NativeLoader.isInitialized()) {
      NativeLoader.init(new SystemDelegate());
    }
    NativeLoader.loadLibrary("pytorch_jni");
  }

  /**
   * Globally sets the number of threads used on native side. Attention: Has global effect, all
   * modules use one thread pool with specified number of threads.
   *
   * @param numThreads number of threads, must be positive number.
   */
  public static void setNumThreads(int numThreads) {
    if (numThreads < 1) {
      throw new IllegalArgumentException("Number of threads cannot be less than 1");
    }

    nativeSetNumThreads(numThreads);
  }

  private static native void nativeSetNumThreads(int numThreads);
}
