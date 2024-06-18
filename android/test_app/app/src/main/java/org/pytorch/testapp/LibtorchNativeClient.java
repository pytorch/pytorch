package org.pytorch.testapp;

import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;

public final class LibtorchNativeClient {

  public static void loadAndForwardModel(final String modelPath) {
    NativePeer.loadAndForwardModel(modelPath);
  }

  private static class NativePeer {
    static {
      if (!NativeLoader.isInitialized()) {
        NativeLoader.init(new SystemDelegate());
      }
      NativeLoader.loadLibrary("pytorch_testapp_jni");
    }

    private static native void loadAndForwardModel(final String modelPath);
  }
}
