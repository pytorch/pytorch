package org.pytorch;

import android.content.Context;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.Map;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertFalse;

@RunWith(AndroidJUnit4.class)
public class PytorchInstrumentedTests {

  @Before
  public void setUp() {
    System.loadLibrary("pytorch");
  }

  @Test
  public void testRunSqueezeNet() throws IOException {
    final int[] inputDims = new int[]{1, 3, 224, 224};
    final int numElements = Tensor.numElements(inputDims);
    final float[] data = new float[numElements];
    final Tensor inputTensor = Tensor.newFloatTensor(inputDims, data);
    final Module module = Module.load(assetFilePath("squeezenet1_0.pt"));
    final Tensor outputTensor = module.forward(IValue.tensor(inputTensor)).getTensor();
    assertNotNull(outputTensor);
    assertArrayEquals(new int[]{1, 1000}, outputTensor.dims);
    float[] outputData = outputTensor.getDataAsFloatArray();
    assertEquals(1000, outputData.length);
  }

  @Test
  public void testEqBool() throws IOException {
    final Module module = Module.load(assetFilePath("EqBool.pt"));
    for (boolean value : new boolean[]{false, true}) {
      final IValue input = IValue.bool(value);
      assertTrue(input.isBool());
      assertTrue(value == input.getBoolean());
      final IValue output = module.forward(input);
      assertTrue(output.isBool());
      assertTrue(value == output.getBoolean());
    }
  }

  @Test
  public void testEqInt() throws IOException {
    final Module module = Module.load(assetFilePath("EqInt.pt"));
    for (long value : new long[]{Long.MIN_VALUE, -1024, -1, 0, 1, 1024, Long.MAX_VALUE}) {
      final IValue input = IValue.long64(value);
      assertTrue(input.isLong());
      assertTrue(value == input.getLong());
      final IValue output = module.forward(input);
      assertTrue(output.isLong());
      assertTrue(value == output.getLong());
    }
  }

  @Test
  public void testEqFloat() throws IOException {
    final Module module = Module.load(assetFilePath(
        "EqFloat.pt"));
    double[] values = new double[]{
        -Double.MAX_VALUE,
        Double.MAX_VALUE,
        -Double.MIN_VALUE,
        Double.MIN_VALUE,
        -1.f,
        0.f,
        1.f,
    };
    for (double value : values) {
      final IValue input = IValue.double64(value);
      assertTrue(input.isDouble());
      assertTrue(value == input.getDouble());
      final IValue output = module.forward(input);
      assertTrue(output.isDouble());
      assertTrue(value == output.getDouble());
    }
  }

  @Test
  public void testEqTensor() throws IOException {
    final int[] inputTensorDims = new int[]{1, 3, 224, 224};
    final int numElements = Tensor.numElements(inputTensorDims);
    final float[] inputTensorData = new float[numElements];
    for (int i = 0; i < numElements; ++i) {
      inputTensorData[i] = i;
    }
    final Tensor inputTensor = Tensor.newFloatTensor(inputTensorDims, inputTensorData);

    final Module module = Module.load(assetFilePath("EqTensor.pt"));
    final IValue input = IValue.tensor(inputTensor);
    assertTrue(input.isTensor());
    assertTrue(inputTensor == input.getTensor());
    final IValue output = module.forward(input);
    assertTrue(output.isTensor());
    final Tensor outputTensor = output.getTensor();
    assertNotNull(outputTensor);
    assertArrayEquals(inputTensorDims, outputTensor.dims);
    float[] outputData = outputTensor.getDataAsFloatArray();
    for (int i = 0; i < numElements; i++) {
      assertTrue(inputTensorData[i] == outputData[i]);
    }
  }

  @Test
  public void testEqDictIntKeyIntValue() throws IOException {
    final Module module = Module.load(assetFilePath(
        "EqDictIntKeyIntValue.pt"));
    final Map<Long, IValue> inputMap = new HashMap<>();

    inputMap.put(Long.MIN_VALUE, IValue.long64(-Long.MIN_VALUE));
    inputMap.put(Long.MAX_VALUE, IValue.long64(-Long.MAX_VALUE));
    inputMap.put(0l, IValue.long64(0l));
    inputMap.put(1l, IValue.long64(-1l));
    inputMap.put(-1l, IValue.long64(1l));

    final IValue input = IValue.dictLongKey(inputMap);
    assertTrue(input.isDictLongKey());

    final IValue output = module.forward(input);
    assertTrue(output.isDictLongKey());

    final Map<Long, IValue> outputMap = output.getDictLongKey();
    assertTrue(inputMap.size() == outputMap.size());
    for (Map.Entry<Long, IValue> entry : inputMap.entrySet()) {
      assertTrue(outputMap.get(entry.getKey()).getLong() == entry.getValue().getLong());
    }
  }

  @Test
  public void testEqDictStrKeyIntValue() throws IOException {
    final Module module = Module.load(assetFilePath(
        "EqDictStrKeyIntValue.pt"));
    final Map<String, IValue> inputMap = new HashMap<>();

    inputMap.put("long_min_value", IValue.long64(Long.MIN_VALUE));
    inputMap.put("long_max_value", IValue.long64(Long.MAX_VALUE));
    inputMap.put("long_0", IValue.long64(0l));
    inputMap.put("long_1", IValue.long64(1l));
    inputMap.put("long_-1", IValue.long64(-1l));

    final IValue input = IValue.dictStringKey(inputMap);
    assertTrue(input.isDictStringKey());

    final IValue output = module.forward(input);
    assertTrue(output.isDictStringKey());

    final Map<String, IValue> outputMap = output.getDictStringKey();
    assertTrue(inputMap.size() == outputMap.size());
    for (Map.Entry<String, IValue> entry : inputMap.entrySet()) {
      assertTrue(outputMap.get(entry.getKey()).getLong() == entry.getValue().getLong());
    }
  }

  @Test
  public void testEqDictFloatKeyIntValue() throws IOException {
    final Module module = Module.load(assetFilePath(
        "EqDictFloatKeyIntValue.pt"));
    final Map<Double, IValue> inputMap = new HashMap<>();

    inputMap.put(-Double.MAX_VALUE, IValue.long64(Long.MIN_VALUE));
    inputMap.put(Double.MAX_VALUE, IValue.long64(Long.MAX_VALUE));
    inputMap.put(0.d, IValue.long64(0l));
    inputMap.put(-1.d, IValue.long64(-1l));
    inputMap.put(1.d, IValue.long64(1l));

    final IValue input = IValue.dictDoubleKey(inputMap);
    assertTrue(input.isDictDoubleKey());

    final IValue output = module.forward(input);
    assertTrue(output.isDictDoubleKey());
    final Map<Double, IValue> outputMap = output.getDictDoubleKey();
    assertTrue(inputMap.size() == outputMap.size());
    for (Map.Entry<Double, IValue> entry : inputMap.entrySet()) {
      assertTrue(outputMap.get(entry.getKey()).getLong() == entry.getValue().getLong());
    }
  }

  @Test
  public void testListIntSumReturnTuple() throws IOException {
    final Module module = Module.load(assetFilePath(
        "ListIntSumReturnTuple.pt"));

    int n = 1;
    long[] a = new long[n];
    long sum = 0;
    for (int i = 0; i < n; i++) {
      a[i] = i;
      sum += a[i];
    }
    final IValue input = IValue.longList(a);
    assertTrue(input.isLongList());

    final IValue output = module.forward(input);

    assertTrue(output.isTuple());
    assertTrue(2 == output.getTuple().length);

    IValue output0 = output.getTuple()[0];
    IValue output1 = output.getTuple()[1];

    assertArrayEquals(a, output0.getLongList());
    assertTrue(sum == output1.getLong());
  }

  @Test
  public void testOptionalIntIsNone() throws IOException {
    final Module module = Module.load(assetFilePath(
        "OptionalIntIsNone.pt"));

    assertFalse(module.forward(IValue.long64(1l)).getBoolean());
    assertTrue(module.forward(IValue.optionalNull()).getBoolean());
  }

  @Test
  public void testIntEq0None() throws IOException {
    final Module module = Module.load(assetFilePath(
        "IntEq0None.pt"));

    assertTrue(module.forward(IValue.long64(0l)).isNull());
    assertTrue(module.forward(IValue.long64(1l)).getLong() == 1l);
  }

  private static String assetFilePath(String assetName) throws IOException {
    final Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
    File file = new File(appContext.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    try (InputStream is = appContext.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    } catch (IOException e) {
      throw e;
    }
  }
}
