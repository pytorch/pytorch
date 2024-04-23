package org.pytorch;

import com.facebook.soloader.nativeloader.NativeLoader;

public class PyTorchCodegenLoader {

  public static void loadNativeLibs() {
    try {
      NativeLoader.loadLibrary("torch-code-gen");
    } catch (Throwable t) {
      // Loading the codegen lib is best-effort since it's only there for query based builds.
    }
  }

  private PyTorchCodegenLoader() {}
}
