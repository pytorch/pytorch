// Copyright 2004-present Facebook. All Rights Reserved.

package org.pytorch;

import com.facebook.jni.HybridData;

/**
 * Java wrapper for torch::jit::script::Module.
 */
public class Module {

  private NativePeer mNativePeer;

  /**
   * Loads a serialized TorchScript module from the specified path on the disk.
   *
   * @param modelPath path to file that contains the serialized TorchScript module.
   * @return new {@link org.pytorch.Module} object which owns torch::jit::script::Module.
   */
  public static Module load(final String modelPath) {
    return new Module(modelPath);
  }

  private Module(final String moduleAbsolutePath) {
    this.mNativePeer = new NativePeer(moduleAbsolutePath);
  }

  /**
   * Runs the 'forward' method of this module with the specified arguments.
   *
   * @param inputs arguments for the TorchScript module's 'forward' method.
   * @return return value from the 'forward' method.
   */
  public IValue forward(IValue... inputs) {
    return mNativePeer.forward(inputs);
  }

  /**
   * Runs the specified method of this module with the specified arguments.
   *
   * @param methodName name of the TorchScript method to run.
   * @param inputs     arguments that will be passed to TorchScript method.
   * @return return value from the method.
   */
  public IValue runMethod(String methodName, IValue... inputs) {
    return mNativePeer.runMethod(methodName, inputs);
  }

  /**
   * Explicitly destroys the native torch::jit::script::Module.
   * Calling this method is not required, as the native object will be destroyed
   * when this object is garbage-collected.  However, the timing of garbage collection
   * is not guaranteed, so proactively calling {@code destroy} can free memory more quickly.
   * See {@link com.facebook.jni.HybridData#resetNative}.
   */
  public void destroy() {
    mNativePeer.mHybridData.resetNative();
  }

  private static class NativePeer {
    static {
      System.loadLibrary("pytorch_jni");
    }

    private final HybridData mHybridData;

    private static native HybridData initHybrid(String moduleAbsolutePath);

    NativePeer(String moduleAbsolutePath) {
      mHybridData = initHybrid(moduleAbsolutePath);
    }

    private native IValue forward(IValue... inputs);

    private native IValue runMethod(String methodName, IValue... inputs);
  }
}
