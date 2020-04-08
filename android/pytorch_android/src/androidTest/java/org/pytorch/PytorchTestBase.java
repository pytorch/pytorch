package org.pytorch;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import org.junit.Test;

public abstract class PytorchTestBase {
  private static final String TEST_MODULE_ASSET_NAME = "test.pt";

  @Test
  public void testForwardNull() throws IOException {
    final Module module = Module.load(assetFilePath(TEST_MODULE_ASSET_NAME));
    final IValue input = IValue.from(Tensor.fromBlob(Tensor.allocateByteBuffer(1), new long[] {1}));
    assertTrue(input.isTensor());
    final IValue output = module.forward(input);
    assertTrue(output.isNull());
  }

  @Test
  public void testEqBool() throws IOException {
    final Module module = Module.load(assetFilePath(TEST_MODULE_ASSET_NAME));
    for (boolean value : new boolean[] {false, true}) {
      final IValue input = IValue.from(value);
      assertTrue(input.isBool());
      assertTrue(value == input.toBool());
      final IValue output = module.runMethod("eqBool", input);
      assertTrue(output.isBool());
      assertTrue(value == output.toBool());
    }
  }

  @Test
  public void testEqInt() throws IOException {
    final Module module = Module.load(assetFilePath(TEST_MODULE_ASSET_NAME));
    for (long value : new long[] {Long.MIN_VALUE, -1024, -1, 0, 1, 1024, Long.MAX_VALUE}) {
      final IValue input = IValue.from(value);
      assertTrue(input.isLong());
      assertTrue(value == input.toLong());
      final IValue output = module.runMethod("eqInt", input);
      assertTrue(output.isLong());
      assertTrue(value == output.toLong());
    }
  }

  @Test
  public void testEqFloat() throws IOException {
    final Module module = Module.load(assetFilePath(TEST_MODULE_ASSET_NAME));
    double[] values =
        new double[] {
          -Double.MAX_VALUE,
          Double.MAX_VALUE,
          -Double.MIN_VALUE,
          Double.MIN_VALUE,
          -Math.exp(1.d),
          -Math.sqrt(2.d),
          -3.1415f,
          3.1415f,
          -1,
          0,
          1,
        };
    for (double value : values) {
      final IValue input = IValue.from(value);
      assertTrue(input.isDouble());
      assertTrue(value == input.toDouble());
      final IValue output = module.runMethod("eqFloat", input);
      assertTrue(output.isDouble());
      assertTrue(value == output.toDouble());
    }
  }

  @Test
  public void testEqTensor() throws IOException {
    final long[] inputTensorShape = new long[] {1, 3, 224, 224};
    final long numElements = Tensor.numel(inputTensorShape);
    final float[] inputTensorData = new float[(int) numElements];
    for (int i = 0; i < numElements; ++i) {
      inputTensorData[i] = i;
    }
    final Tensor inputTensor = Tensor.fromBlob(inputTensorData, inputTensorShape);

    final Module module = Module.load(assetFilePath(TEST_MODULE_ASSET_NAME));
    final IValue input = IValue.from(inputTensor);
    assertTrue(input.isTensor());
    assertTrue(inputTensor == input.toTensor());
    final IValue output = module.runMethod("eqTensor", input);
    assertTrue(output.isTensor());
    final Tensor outputTensor = output.toTensor();
    assertNotNull(outputTensor);
    assertArrayEquals(inputTensorShape, outputTensor.shape());
    float[] outputData = outputTensor.getDataAsFloatArray();
    for (int i = 0; i < numElements; i++) {
      assertTrue(inputTensorData[i] == outputData[i]);
    }
  }

  @Test
  public void testEqDictIntKeyIntValue() throws IOException {
    final Module module = Module.load(assetFilePath(TEST_MODULE_ASSET_NAME));
    final Map<Long, IValue> inputMap = new HashMap<>();

    inputMap.put(Long.MIN_VALUE, IValue.from(-Long.MIN_VALUE));
    inputMap.put(Long.MAX_VALUE, IValue.from(-Long.MAX_VALUE));
    inputMap.put(0l, IValue.from(0l));
    inputMap.put(1l, IValue.from(-1l));
    inputMap.put(-1l, IValue.from(1l));

    final IValue input = IValue.dictLongKeyFrom(inputMap);
    assertTrue(input.isDictLongKey());

    final IValue output = module.runMethod("eqDictIntKeyIntValue", input);
    assertTrue(output.isDictLongKey());

    final Map<Long, IValue> outputMap = output.toDictLongKey();
    assertTrue(inputMap.size() == outputMap.size());
    for (Map.Entry<Long, IValue> entry : inputMap.entrySet()) {
      assertTrue(outputMap.get(entry.getKey()).toLong() == entry.getValue().toLong());
    }
  }

  @Test
  public void testEqDictStrKeyIntValue() throws IOException {
    final Module module = Module.load(assetFilePath(TEST_MODULE_ASSET_NAME));
    final Map<String, IValue> inputMap = new HashMap<>();

    inputMap.put("long_min_value", IValue.from(Long.MIN_VALUE));
    inputMap.put("long_max_value", IValue.from(Long.MAX_VALUE));
    inputMap.put("long_0", IValue.from(0l));
    inputMap.put("long_1", IValue.from(1l));
    inputMap.put("long_-1", IValue.from(-1l));

    final IValue input = IValue.dictStringKeyFrom(inputMap);
    assertTrue(input.isDictStringKey());

    final IValue output = module.runMethod("eqDictStrKeyIntValue", input);
    assertTrue(output.isDictStringKey());

    final Map<String, IValue> outputMap = output.toDictStringKey();
    assertTrue(inputMap.size() == outputMap.size());
    for (Map.Entry<String, IValue> entry : inputMap.entrySet()) {
      assertTrue(outputMap.get(entry.getKey()).toLong() == entry.getValue().toLong());
    }
  }

  @Test
  public void testListIntSumReturnTuple() throws IOException {
    final Module module = Module.load(assetFilePath(TEST_MODULE_ASSET_NAME));

    for (int n : new int[] {0, 1, 128}) {
      long[] a = new long[n];
      long sum = 0;
      for (int i = 0; i < n; i++) {
        a[i] = i;
        sum += a[i];
      }
      final IValue input = IValue.listFrom(a);
      assertTrue(input.isLongList());

      final IValue output = module.runMethod("listIntSumReturnTuple", input);

      assertTrue(output.isTuple());
      assertTrue(2 == output.toTuple().length);

      IValue output0 = output.toTuple()[0];
      IValue output1 = output.toTuple()[1];

      assertArrayEquals(a, output0.toLongList());
      assertTrue(sum == output1.toLong());
    }
  }

  @Test
  public void testOptionalIntIsNone() throws IOException {
    final Module module = Module.load(assetFilePath(TEST_MODULE_ASSET_NAME));

    assertFalse(module.runMethod("optionalIntIsNone", IValue.from(1l)).toBool());
    assertTrue(module.runMethod("optionalIntIsNone", IValue.optionalNull()).toBool());
  }

  @Test
  public void testIntEq0None() throws IOException {
    final Module module = Module.load(assetFilePath(TEST_MODULE_ASSET_NAME));

    assertTrue(module.runMethod("intEq0None", IValue.from(0l)).isNull());
    assertTrue(module.runMethod("intEq0None", IValue.from(1l)).toLong() == 1l);
  }

  @Test(expected = IllegalArgumentException.class)
  public void testRunUndefinedMethod() throws IOException {
    final Module module = Module.load(assetFilePath(TEST_MODULE_ASSET_NAME));
    module.runMethod("test_undefined_method_throws_exception");
  }

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
  public void testEqString() throws IOException {
    final Module module = Module.load(assetFilePath(TEST_MODULE_ASSET_NAME));
    String[] values =
        new String[] {
          "smoketest",
          "проверка не латинских символов", // not latin symbols check
          "#@$!@#)($*!@#$)(!@*#$"
        };
    for (String value : values) {
      final IValue input = IValue.from(value);
      assertTrue(input.isString());
      assertTrue(value.equals(input.toStr()));
      final IValue output = module.runMethod("eqStr", input);
      assertTrue(output.isString());
      assertTrue(value.equals(output.toStr()));
    }
  }

  @Test
  public void testStr3Concat() throws IOException {
    final Module module = Module.load(assetFilePath(TEST_MODULE_ASSET_NAME));
    String[] values =
        new String[] {
          "smoketest",
          "проверка не латинских символов", // not latin symbols check
          "#@$!@#)($*!@#$)(!@*#$"
        };
    for (String value : values) {
      final IValue input = IValue.from(value);
      assertTrue(input.isString());
      assertTrue(value.equals(input.toStr()));
      final IValue output = module.runMethod("str3Concat", input);
      assertTrue(output.isString());
      String expectedOutput =
          new StringBuilder().append(value).append(value).append(value).toString();
      assertTrue(expectedOutput.equals(output.toStr()));
    }
  }

  @Test
  public void testEmptyShape() throws IOException {
    final Module module = Module.load(assetFilePath(TEST_MODULE_ASSET_NAME));
    final long someNumber = 43;
    final IValue input = IValue.from(Tensor.fromBlob(new long[] {someNumber}, new long[] {}));
    final IValue output = module.runMethod("newEmptyShapeWithItem", input);
    assertTrue(output.isTensor());
    Tensor value = output.toTensor();
    assertArrayEquals(new long[] {}, value.shape());
    assertArrayEquals(new long[] {someNumber}, value.getDataAsLongArray());
  }

  @Test
  public void testAliasWithOffset() throws IOException {
    final Module module = Module.load(assetFilePath(TEST_MODULE_ASSET_NAME));
    final IValue output = module.runMethod("testAliasWithOffset");
    assertTrue(output.isTensorList());
    Tensor[] tensors = output.toTensorList();
    assertEquals(100, tensors[0].getDataAsLongArray()[0]);
    assertEquals(200, tensors[1].getDataAsLongArray()[0]);
  }

  @Test
  public void testNonContiguous() throws IOException {
    final Module module = Module.load(assetFilePath(TEST_MODULE_ASSET_NAME));
    final IValue output = module.runMethod("testNonContiguous");
    assertTrue(output.isTensor());
    Tensor value = output.toTensor();
    assertArrayEquals(new long[] {2}, value.shape());
    assertArrayEquals(new long[] {100, 300}, value.getDataAsLongArray());
  }

  protected abstract String assetFilePath(String assetName) throws IOException;
}
