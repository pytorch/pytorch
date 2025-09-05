package org.pytorch;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import org.junit.Test;
import org.junit.Ignore;

public abstract class PytorchTestBase {
  private static final String TEST_MODULE_ASSET_NAME = "android_api_module.ptl";

  @Test
  public void testTensorMethods() {
    long[] shape = new long[] {1, 3, 224, 224};
    final int numel = (int) Tensor.numel(shape);
    int[] ints = new int[numel];
    float[] floats = new float[numel];

    byte[] bytes = new byte[numel];
    for (int i = 0; i < numel; i++) {
      bytes[i] = (byte) ((i % 255) - 128);
      ints[i] = i;
      floats[i] = i / 1000.f;
    }

    Tensor tensorBytes = Tensor.fromBlob(bytes, shape);
    assertTrue(tensorBytes.dtype() == DType.INT8);
    assertArrayEquals(bytes, tensorBytes.getDataAsByteArray());

    Tensor tensorInts = Tensor.fromBlob(ints, shape);
    assertTrue(tensorInts.dtype() == DType.INT32);
    assertArrayEquals(ints, tensorInts.getDataAsIntArray());

    Tensor tensorFloats = Tensor.fromBlob(floats, shape);
    assertTrue(tensorFloats.dtype() == DType.FLOAT32);
    float[] floatsOut = tensorFloats.getDataAsFloatArray();
    assertTrue(floatsOut.length == numel);
    for (int i = 0; i < numel; i++) {
      assertTrue(floats[i] == floatsOut[i]);
    }
  }

  @Test(expected = IllegalStateException.class)
  public void testTensorIllegalStateOnWrongType() {
    long[] shape = new long[] {1, 3, 224, 224};
    final int numel = (int) Tensor.numel(shape);
    float[] floats = new float[numel];
    Tensor tensorFloats = Tensor.fromBlob(floats, shape);
    assertTrue(tensorFloats.dtype() == DType.FLOAT32);
    tensorFloats.getDataAsByteArray();
  }

  @Test
  @Ignore
  public void testSpectralOps() throws IOException {
    // NB: This model fails without lite interpreter.  The error is as follows:
    // RuntimeError: stft requires the return_complex parameter be given for real inputs
    runModel("spectral_ops");
  }

  void runModel(final String name) throws IOException {
    final Module storage_module = loadModel(name + ".ptl");
    storage_module.forward();

    // TODO enable this once the on-the-fly script is ready
    // final Module on_the_fly_module = loadModel(name + "_temp.ptl");
    // on_the_fly_module.forward();
    assertTrue(true);
  }

  protected abstract Module loadModel(String assetName) throws IOException;
}
