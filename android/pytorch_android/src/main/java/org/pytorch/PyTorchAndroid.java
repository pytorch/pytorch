package org.pytorch;

import android.content.res.AssetManager;
import com.facebook.jni.annotations.DoNotStrip;
import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;

public final class PyTorchAndroid {
  static {
    if (!NativeLoader.isInitialized()) {
      NativeLoader.init(new SystemDelegate());
    }
    NativeLoader.loadLibrary("pytorch_jni");
    PyTorchCodegenLoader.loadNativeLibs();
  }

  /**
   * Attention: This is not recommended way of loading production modules, as prepackaged assets
   * increase apk size etc. For production usage consider using loading from file on the disk {@link
   * org.pytorch.Module#load(String)}.
   *
   * <p>This method is meant to use in tests and demos.
   */
  public static Module loadModuleFromAsset(
      final AssetManager assetManager, final String assetName, final Device device) {
    return new Module(new NativePeer(assetName, assetManager, device));
  }

  public static Module loadModuleFromAsset(
      final AssetManager assetManager, final String assetName) {
    return new Module(new NativePeer(assetName, assetManager, Device.CPU));
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

  @DoNotStrip
  private static native void nativeSetNumThreads(int numThreads);
}
