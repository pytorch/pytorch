// Copyright 2004-present Facebook. All Rights Reserved.

package org.pytorch;

import com.facebook.jni.HybridData;

/**
 * Java holder for torch::jit::script::Module which owns it on jni side.
 */
public class Module {

  private NativePeer mNativePeer;

  /**
   * Loads serialized torchscript module from the specified absolute path on the disk.
   *
   * @param modelAbsolutePath absolute path to file that contains the serialized torchscript module.
   * @return new {@link org.pytorch.Module} object which owns torch::jit::script::Module on jni
   * side.
   */
  public static Module load(final String modelAbsolutePath) {
    return new Module(modelAbsolutePath);
  }

  private Module(final String moduleAbsolutePath) {
    this.mNativePeer = new NativePeer(moduleAbsolutePath);
  }

  /**
   * Runs 'forward' method of loaded torchscript module with specified arguments.
   *
   * @param inputs arguments for torchscript module 'forward' method.
   * @return result of torchscript module 'forward' method evaluation
   */
  public IValue forward(IValue... inputs) {
    return mNativePeer.forward(inputs);
  }

  /**
   * Runs specified method of loaded torchscript module with specified arguments.
   *
   * @param methodName torchscript module method to run
   * @param inputs     arguments that will be specified to torchscript module method call
   * @return result of torchscript module specified method evaluation
   */
  public IValue runMethod(String methodName, IValue... inputs) {
    return mNativePeer.runMethod(methodName, inputs);
  }

  /**
   * Explicitly destructs native part. Current instance can not be used after this call. This
   * method may be called multiple times safely. As fbjni library destructs native part
   * automatically when current instance will be
   * collected by Java GC, the instance will not leak if this method is not called,
   * but timing of deletion and the thread will be at the whim of the Java GC.
   * If you want to control the thread and timing of the destructor, you should call this method
   * explicitly.
   * {@link com.facebook.jni.HybridData#resetNative}
   */
  public void destroy() {
    mNativePeer.mHybridData.resetNative();
  }

  private static class NativePeer {
    static {
      System.loadLibrary("pytorch");
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
