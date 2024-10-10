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
import org.junit.Ignore;

public abstract class PytorchTestBase {
  private static final String TEST_MODULE_ASSET_NAME = "android_api_module.ptl";

  @Test
  public void testForwardNull() throws IOException {
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);
    final IValue input = IValue.from(Tensor.fromBlob(Tensor.allocateByteBuffer(1), new long[] {1}));
    assertTrue(input.isTensor());
    final IValue output = module.forward(input);
    assertTrue(output.isNull());
  }

  @Test
  public void testEqBool() throws IOException {
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);
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
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);
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
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);
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

    final Module module = loadModel(TEST_MODULE_ASSET_NAME);
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
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);
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
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);
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
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);

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
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);

    assertFalse(module.runMethod("optionalIntIsNone", IValue.from(1l)).toBool());
    assertTrue(module.runMethod("optionalIntIsNone", IValue.optionalNull()).toBool());
  }

  @Test
  public void testIntEq0None() throws IOException {
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);

    assertTrue(module.runMethod("intEq0None", IValue.from(0l)).isNull());
    assertTrue(module.runMethod("intEq0None", IValue.from(1l)).toLong() == 1l);
  }

  @Test(expected = IllegalArgumentException.class)
  public void testRunUndefinedMethod() throws IOException {
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);
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
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);
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
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);
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
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);
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
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);
    final IValue output = module.runMethod("testAliasWithOffset");
    assertTrue(output.isTensorList());
    Tensor[] tensors = output.toTensorList();
    assertEquals(100, tensors[0].getDataAsLongArray()[0]);
    assertEquals(200, tensors[1].getDataAsLongArray()[0]);
  }

  @Test
  public void testNonContiguous() throws IOException {
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);
    final IValue output = module.runMethod("testNonContiguous");
    assertTrue(output.isTensor());
    Tensor value = output.toTensor();
    assertArrayEquals(new long[] {2}, value.shape());
    assertArrayEquals(new long[] {100, 300}, value.getDataAsLongArray());
  }

  @Test
  public void testChannelsLast() throws IOException {
    long[] inputShape = new long[] {1, 3, 2, 2};
    long[] data = new long[] {1, 11, 101, 2, 12, 102, 3, 13, 103, 4, 14, 104};
    Tensor inputNHWC = Tensor.fromBlob(data, inputShape, MemoryFormat.CHANNELS_LAST);
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);
    final IValue outputNCHW = module.runMethod("contiguous", IValue.from(inputNHWC));
    assertIValueTensor(
        outputNCHW,
        MemoryFormat.CONTIGUOUS,
        new long[] {1, 3, 2, 2},
        new long[] {1, 2, 3, 4, 11, 12, 13, 14, 101, 102, 103, 104});
    final IValue outputNHWC = module.runMethod("contiguousChannelsLast", IValue.from(inputNHWC));
    assertIValueTensor(outputNHWC, MemoryFormat.CHANNELS_LAST, inputShape, data);
  }

  @Test
  public void testChannelsLast3d() throws IOException {
    long[] shape = new long[] {1, 2, 2, 2, 2};
    long[] dataNCHWD = new long[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    long[] dataNHWDC = new long[] {1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 8, 16};

    Tensor inputNHWDC = Tensor.fromBlob(dataNHWDC, shape, MemoryFormat.CHANNELS_LAST_3D);
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);
    final IValue outputNCHWD = module.runMethod("contiguous", IValue.from(inputNHWDC));
    assertIValueTensor(outputNCHWD, MemoryFormat.CONTIGUOUS, shape, dataNCHWD);

    Tensor inputNCHWD = Tensor.fromBlob(dataNCHWD, shape, MemoryFormat.CONTIGUOUS);
    final IValue outputNHWDC =
        module.runMethod("contiguousChannelsLast3d", IValue.from(inputNCHWD));
    assertIValueTensor(outputNHWDC, MemoryFormat.CHANNELS_LAST_3D, shape, dataNHWDC);
  }

  @Test
  public void testChannelsLastConv2d() throws IOException {
    long[] inputShape = new long[] {1, 3, 2, 2};
    long[] dataNCHW = new long[] {
      111, 112,
      121, 122,

      211, 212,
      221, 222,

      311, 312,
      321, 322};
    Tensor inputNCHW = Tensor.fromBlob(dataNCHW, inputShape, MemoryFormat.CONTIGUOUS);
    long[] dataNHWC = new long[] {
      111, 211, 311,       112, 212, 312,

      121, 221, 321,       122, 222, 322};
    Tensor inputNHWC = Tensor.fromBlob(dataNHWC, inputShape, MemoryFormat.CHANNELS_LAST);
    long[] weightShape = new long[] {3, 3, 1, 1};
    long[] dataWeightOIHW = new long[] {
      2, 0, 0,
      0, 1, 0,
      0, 0, -1};
    Tensor wNCHW = Tensor.fromBlob(dataWeightOIHW, weightShape, MemoryFormat.CONTIGUOUS);
    long[] dataWeightOHWI = new long[] {
      2, 0, 0,
      0, 1, 0,
      0, 0, -1};

    Tensor wNHWC = Tensor.fromBlob(dataWeightOHWI, weightShape, MemoryFormat.CHANNELS_LAST);

    final Module module = loadModel(TEST_MODULE_ASSET_NAME);

    final IValue outputNCHW =
        module.runMethod("conv2d", IValue.from(inputNCHW), IValue.from(wNCHW), IValue.from(false));
    assertIValueTensor(
        outputNCHW,
        MemoryFormat.CONTIGUOUS,
        new long[] {1, 3, 2, 2},
        new long[] {
          2*111, 2*112,
          2*121, 2*122,

          211, 212,
          221, 222,

          -311, -312,
          -321, -322});

    final IValue outputNHWC =
        module.runMethod("conv2d", IValue.from(inputNHWC), IValue.from(wNHWC), IValue.from(true));
    assertIValueTensor(
        outputNHWC,
        MemoryFormat.CHANNELS_LAST,
        new long[] {1, 3, 2, 2},
        new long[] {
          2*111, 211, -311,      2*112, 212, -312,
          2*121, 221, -321,      2*122, 222, -322});
  }

  @Test
  public void testChannelsLastConv3d() throws IOException {
    long[] inputShape = new long[] {1, 3, 2, 2, 2};
    long[] dataNCDHW = new long[] {
      1111, 1112,
      1121, 1122,
      1211, 1212,
      1221, 1222,

      2111, 2112,
      2121, 2122,
      2211, 2212,
      2221, 2222,

      3111, 3112,
      3121, 3122,
      3211, 3212,
      3221, 3222};
    Tensor inputNCDHW = Tensor.fromBlob(dataNCDHW, inputShape, MemoryFormat.CONTIGUOUS);
    long[] dataNDHWC = new long[] {
      1111, 2111, 3111,
      1112, 2112, 3112,

      1121, 2121, 3121,
      1122, 2122, 3122,

      1211, 2211, 3211,
      1212, 2212, 3212,

      1221, 2221, 3221,
      1222, 2222, 3222};

    Tensor inputNDHWC = Tensor.fromBlob(dataNDHWC, inputShape, MemoryFormat.CHANNELS_LAST_3D);

    long[] weightShape = new long[] {3, 3, 1, 1, 1};
    long[] dataWeightOIDHW = new long[] {
      2, 0, 0,
      0, 1, 0,
      0, 0, -1,
    };
    Tensor wNCDHW = Tensor.fromBlob(dataWeightOIDHW, weightShape, MemoryFormat.CONTIGUOUS);
    long[] dataWeightODHWI = new long[] {
      2, 0, 0,
      0, 1, 0,
      0, 0, -1,
    };
    Tensor wNDHWC = Tensor.fromBlob(dataWeightODHWI, weightShape, MemoryFormat.CHANNELS_LAST_3D);

    final Module module = loadModel(TEST_MODULE_ASSET_NAME);

    final IValue outputNCDHW =
        module.runMethod("conv3d", IValue.from(inputNCDHW), IValue.from(wNCDHW), IValue.from(false));
    assertIValueTensor(
        outputNCDHW,
        MemoryFormat.CONTIGUOUS,
        new long[] {1, 3, 2, 2, 2},
        new long[] {
          2*1111, 2*1112,     2*1121, 2*1122,
          2*1211, 2*1212,     2*1221, 2*1222,

          2111, 2112,     2121, 2122,
          2211, 2212,     2221, 2222,

          -3111, -3112,     -3121, -3122,
          -3211, -3212,     -3221, -3222});

    final IValue outputNDHWC =
        module.runMethod("conv3d", IValue.from(inputNDHWC), IValue.from(wNDHWC), IValue.from(true));
    assertIValueTensor(
        outputNDHWC,
        MemoryFormat.CHANNELS_LAST_3D,
        new long[] {1, 3, 2, 2, 2},
        new long[] {
          2*1111, 2111, -3111,      2*1112, 2112, -3112,
          2*1121, 2121, -3121,      2*1122, 2122, -3122,

          2*1211, 2211, -3211,      2*1212, 2212, -3212,
          2*1221, 2221, -3221,      2*1222, 2222, -3222});
  }

  @Test
  public void testMobileNetV2() throws IOException {
    try {
      final Module module = loadModel("mobilenet_v2.ptl");
      final IValue inputs = module.runMethod("get_all_bundled_inputs");
      assertTrue(inputs.isList());
      final IValue input = inputs.toList()[0];
      assertTrue(input.isTuple());
      module.forward(input.toTuple()[0]);
      assertTrue(true);
    } catch (Exception ex) {
      assertTrue("failed to run MobileNetV2 " + ex.getMessage(), false);
    }
  }

  @Test
  public void testPointwiseOps() throws IOException {
    runModel("pointwise_ops");
  }

  @Test
  public void testReductionOps() throws IOException {
    runModel("reduction_ops");
  }

  @Test
  public void testComparisonOps() throws IOException {
    runModel("comparison_ops");
  }

  @Test
  public void testOtherMathOps() throws IOException {
    runModel("other_math_ops");
  }

  @Test
  @Ignore
  public void testSpectralOps() throws IOException {
    // NB: This model fails without lite interpreter.  The error is as follows:
    // RuntimeError: stft requires the return_complex parameter be given for real inputs
    runModel("spectral_ops");
  }

  @Test
  public void testBlasLapackOps() throws IOException {
    runModel("blas_lapack_ops");
  }

  @Test
  public void testSamplingOps() throws IOException {
    runModel("sampling_ops");
  }

  @Test
  public void testTensorOps() throws IOException {
    runModel("tensor_general_ops");
  }

  @Test
  public void testTensorCreationOps() throws IOException {
    runModel("tensor_creation_ops");
  }

  @Test
  public void testTensorIndexingOps() throws IOException {
    runModel("tensor_indexing_ops");
  }

  @Test
  public void testTensorTypingOps() throws IOException {
    runModel("tensor_typing_ops");
  }

  @Test
  public void testTensorViewOps() throws IOException {
    runModel("tensor_view_ops");
  }

  @Test
  public void testConvolutionOps() throws IOException {
    runModel("convolution_ops");
  }

  @Test
  public void testPoolingOps() throws IOException {
    runModel("pooling_ops");
  }

  @Test
  public void testPaddingOps() throws IOException {
    runModel("padding_ops");
  }

  @Test
  public void testActivationOps() throws IOException {
    runModel("activation_ops");
  }

  @Test
  public void testNormalizationOps() throws IOException {
    runModel("normalization_ops");
  }

  @Test
  public void testRecurrentOps() throws IOException {
    runModel("recurrent_ops");
  }

  @Test
  public void testTransformerOps() throws IOException {
    runModel("transformer_ops");
  }

  @Test
  public void testLinearOps() throws IOException {
    runModel("linear_ops");
  }

  @Test
  public void testDropoutOps() throws IOException {
    runModel("dropout_ops");
  }

  @Test
  public void testSparseOps() throws IOException {
    runModel("sparse_ops");
  }

  @Test
  public void testDistanceFunctionOps() throws IOException {
    runModel("distance_function_ops");
  }

  @Test
  public void testLossFunctionOps() throws IOException {
    runModel("loss_function_ops");
  }

  @Test
  public void testVisionFunctionOps() throws IOException {
    runModel("vision_function_ops");
  }

  @Test
  public void testShuffleOps() throws IOException {
    runModel("shuffle_ops");
  }

  @Test
  public void testNNUtilsOps() throws IOException {
    runModel("nn_utils_ops");
  }

  @Test
  public void testQuantOps() throws IOException {
    runModel("general_quant_ops");
  }

  @Test
  public void testDynamicQuantOps() throws IOException {
    runModel("dynamic_quant_ops");
  }

  @Test
  public void testStaticQuantOps() throws IOException {
    runModel("static_quant_ops");
  }

  @Test
  public void testFusedQuantOps() throws IOException {
    runModel("fused_quant_ops");
  }

  @Test
  public void testTorchScriptBuiltinQuantOps() throws IOException {
    runModel("torchscript_builtin_ops");
  }

  @Test
  public void testTorchScriptCollectionQuantOps() throws IOException {
    runModel("torchscript_collection_ops");
  }

  static void assertIValueTensor(
      final IValue ivalue,
      final MemoryFormat memoryFormat,
      final long[] expectedShape,
      final long[] expectedData) {
    assertTrue(ivalue.isTensor());
    Tensor t = ivalue.toTensor();
    assertEquals(memoryFormat, t.memoryFormat());
    assertArrayEquals(expectedShape, t.shape());
    assertArrayEquals(expectedData, t.getDataAsLongArray());
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
