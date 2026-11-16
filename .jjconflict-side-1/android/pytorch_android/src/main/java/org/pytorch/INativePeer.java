package org.pytorch;

interface INativePeer {
  void resetNative();

  IValue forward(IValue... inputs);

  IValue runMethod(String methodName, IValue... inputs);
}
