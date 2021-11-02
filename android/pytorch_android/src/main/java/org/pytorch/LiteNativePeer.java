package org.pytorch;

import com.facebook.jni.HybridData;
import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;
import java.util.Map;

class LiteNativePeer implements INativePeer {
  static {
    if (!NativeLoader.isInitialized()) {
      NativeLoader.init(new SystemDelegate());
    }
    NativeLoader.loadLibrary("pytorch_jni_lite");
    PyTorchCodegenLoader.loadNativeLibs();
  }

  private final HybridData mHybridData;

  private static native HybridData initHybrid(
      String moduleAbsolutePath, Map<String, String> extraFiles, int deviceJniCode);

  private static native HybridData initHybridAndroidAsset(
      String assetName, /* android.content.res.AssetManager */
      Object androidAssetManager,
      int deviceJniCode);

  LiteNativePeer(String moduleAbsolutePath, Map<String, String> extraFiles, Device device) {
    mHybridData = initHybrid(moduleAbsolutePath, extraFiles, device.jniCode);
  }

  LiteNativePeer(
      String assetName, /* android.content.res.AssetManager */
      Object androidAssetManager,
      Device device) {
    mHybridData = initHybridAndroidAsset(assetName, androidAssetManager, device.jniCode);
  }

  /**
   * Explicitly destroys the native torch::jit::mobile::Module. Calling this method is not required,
   * as the native object will be destroyed when this object is garbage-collected. However, the
   * timing of garbage collection is not guaranteed, so proactively calling {@code resetNative} can
   * free memory more quickly. See {@link com.facebook.jni.HybridData#resetNative}.
   */
  public void resetNative() {
    mHybridData.resetNative();
  }

  /**
   * Runs the 'forward' method of this module with the specified arguments.
   *
   * @param inputs arguments for the TorchScript module's 'forward' method.
   * @return return value from the 'forward' method.
   */
  public native IValue forward(IValue... inputs);

  /**
   * Runs the specified method of this module with the specified arguments.
   *
   * @param methodName name of the TorchScript method to run.
   * @param inputs arguments that will be passed to TorchScript method.
   * @return return value from the method.
   */
  public native IValue runMethod(String methodName, IValue... inputs);
}
