package com.facebook.pytorch;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

@RunWith(AndroidJUnit4.class)
public class InstrumentedTests {

  private static final String TAG = "PytorchTest";

  @Before
  public void setUp() {
    System.loadLibrary("pytorch_android");
  }

  @Test
  public void testRunSqueezeNet() throws IOException {
    final int[] inputDims = new int[]{1, 3, 224, 224};
    final int numElements = Tensor.numElements(inputDims);
    final float[] data = new float[numElements];
    final Tensor inputTensor = Tensor.newFloatTensor(inputDims, data);
    final String modelFilePath = assetFilePath("squeezenet1_0.pt");
    Log.d(TAG, "smokeTest() modelFilePath:" + modelFilePath);
    final PytorchScriptModule module = new PytorchScriptModule(modelFilePath);
    final Tensor outputTensor = module.run(inputTensor);
    assertNotNull(outputTensor);
    assertArrayEquals(new int[]{1, 1000}, outputTensor.dims);
    float[] outputData = outputTensor.getDataAsFloatArray();
    assertEquals(1000, outputData.length);
  }

  @Test
  public void testBitmapToTensorOnSqueezeNet_husky() throws IOException {
    final Bitmap bitmap = assetAsBitmap("siberian_husky.jpg");
    final Tensor inputTensor = TensorUtils.bitmapToFloatTensorTorchVisionForm(bitmap);
    bitmap.recycle();
    final String modelFilePath = assetFilePath("squeezenet1_0.pt");
    Log.d(TAG, "smokeTest() modelFilePath:" + modelFilePath);
    final PytorchScriptModule module = new PytorchScriptModule(modelFilePath);
    final Tensor outputTensor = module.run(inputTensor);
    assertNotNull(outputTensor);
    assertArrayEquals(new int[]{1, 1000}, outputTensor.dims);
    float[] outputData = outputTensor.getDataAsFloatArray();
    assertEquals(1000, outputData.length);

    int maxI = -1;
    float maxValue = -Float.MAX_VALUE;
    for (int i = 0; i < outputData.length; i++) {
      if (outputData[i] > maxValue) {
        maxValue = outputData[i];
        maxI = i;
      }
    }
    final String s = TestConstants.IMAGENET_CLASSES[maxI];
    assertTrue(s.toLowerCase().contains("husky"));
  }

  @Test
  public void testBitmapToTensorOnSqueezeNet_husky_forward() throws IOException {
    final Bitmap bitmap = assetAsBitmap("siberian_husky.jpg");
    final Tensor inputTensor = TensorUtils.bitmapToFloatTensorTorchVisionForm(bitmap);
    bitmap.recycle();
    final String modelFilePath = assetFilePath("squeezenet1_0.pt");
    Log.d(TAG, "smokeTest() modelFilePath:" + modelFilePath);
    final PytorchScriptModule module = new PytorchScriptModule(modelFilePath);
    final IValue input = IValue.tensor(inputTensor);
    final IValue output = module.forward(input);
    assertNotNull(output);
    assertTrue(IValue.TYPE_CODE_TENSOR == output.typeCode);

    final Tensor outputTensor = output.getTensor();

    assertArrayEquals(new int[]{1, 1000}, outputTensor.dims);
    float[] outputData = outputTensor.getDataAsFloatArray();
    assertEquals(1000, outputData.length);

    int maxI = -1;
    float maxValue = -Float.MAX_VALUE;
    for (int i = 0; i < outputData.length; i++) {
      if (outputData[i] > maxValue) {
        maxValue = outputData[i];
        maxI = i;
      }
    }
    final String s = TestConstants.IMAGENET_CLASSES[maxI];
    assertTrue(s.toLowerCase().contains("husky"));
  }

  @Test
  public void testEqBool() throws IOException {
    final PytorchScriptModule module = new PytorchScriptModule(assetFilePath("EqBool.pt"));
    for (boolean value : new boolean[]{false, true}) {
      final IValue input = IValue.bool(value);
      assertTrue(IValue.TYPE_CODE_BOOL == input.typeCode);
      assertTrue(value == input.getBoolean());
      final IValue output = module.forward(input);
      assertTrue(IValue.TYPE_CODE_BOOL == output.typeCode);
      assertTrue(value == output.getBoolean());
    }
  }

  @Test
  public void testEqInt() throws IOException {
    final PytorchScriptModule module = new PytorchScriptModule(assetFilePath("EqInt.pt"));
    for (long value : new long[]{Long.MIN_VALUE, -1024, -1, 0, 1, 1024, Long.MAX_VALUE}) {
      final IValue input = IValue.long64(value);
      assertTrue(IValue.TYPE_CODE_LONG64 == input.typeCode);
      assertTrue(value == input.getLong());
      final IValue output = module.forward(input);
      assertTrue(IValue.TYPE_CODE_LONG64 == output.typeCode);
      assertTrue(value == output.getLong());
    }
  }

  @Test
  public void testEqFloat() throws IOException {
    final PytorchScriptModule module = new PytorchScriptModule(assetFilePath(
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
      assertTrue(IValue.TYPE_CODE_DOUBLE64 == input.typeCode);
      assertTrue(value == input.getDouble());
      final IValue output = module.forward(input);
      assertTrue(IValue.TYPE_CODE_DOUBLE64 == output.typeCode);
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

    final PytorchScriptModule module = new PytorchScriptModule(assetFilePath("EqTensor.pt"));
    final IValue input = IValue.tensor(inputTensor);
    assertTrue(IValue.TYPE_CODE_TENSOR == input.typeCode);
    assertTrue(inputTensor == input.getTensor());

    final IValue output = module.forward(input);
    assertTrue(IValue.TYPE_CODE_TENSOR == output.typeCode);
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
    final PytorchScriptModule module = new PytorchScriptModule(assetFilePath(
        "EqDictIntKeyIntValue.pt"));
    final Map<Long, IValue> inputMap = new HashMap<>();

    inputMap.put(Long.MIN_VALUE, IValue.long64(-Long.MIN_VALUE));
    inputMap.put(Long.MAX_VALUE, IValue.long64(-Long.MAX_VALUE));
    inputMap.put(0l, IValue.long64(0l));
    inputMap.put(1l, IValue.long64(-1l));
    inputMap.put(-1l, IValue.long64(1l));

    final IValue input = IValue.dictLongKey(inputMap);
    assertTrue(IValue.TYPE_CODE_DICT_LONG_KEY == input.typeCode);

    final IValue output = module.forward(input);
    assertTrue(IValue.TYPE_CODE_DICT_LONG_KEY == output.typeCode);

    final Map<Long, IValue> outputMap = output.getDictLongKey();
    assertTrue(inputMap.size() == outputMap.size());
    for (Map.Entry<Long, IValue> entry : inputMap.entrySet()) {
      assertTrue(outputMap.get(entry.getKey()).getLong() == entry.getValue().getLong());
    }
  }

  @Test
  public void testEqDictStrKeyIntValue() throws IOException {
    final PytorchScriptModule module = new PytorchScriptModule(assetFilePath(
        "EqDictStrKeyIntValue.pt"));
    final Map<String, IValue> inputMap = new HashMap<>();

    inputMap.put("long_min_value", IValue.long64(Long.MIN_VALUE));
    inputMap.put("long_max_value", IValue.long64(Long.MAX_VALUE));
    inputMap.put("long_0", IValue.long64(0l));
    inputMap.put("long_1", IValue.long64(1l));
    inputMap.put("long_-1", IValue.long64(-1l));

    final IValue input = IValue.dictStringKey(inputMap);
    assertTrue(IValue.TYPE_CODE_DICT_STRING_KEY == input.typeCode);

    final IValue output = module.forward(input);
    assertTrue(IValue.TYPE_CODE_DICT_STRING_KEY == output.typeCode);

    final Map<String, IValue> outputMap = output.getDictStringKey();
    assertTrue(inputMap.size() == outputMap.size());
    for (Map.Entry<String, IValue> entry : inputMap.entrySet()) {
      assertTrue(outputMap.get(entry.getKey()).getLong() == entry.getValue().getLong());
    }
  }

  @Test
  public void testEqDictFloatKeyIntValue() throws IOException {
    final PytorchScriptModule module = new PytorchScriptModule(assetFilePath(
        "EqDictFloatKeyIntValue.pt"));
    final Map<Double, IValue> inputMap = new HashMap<>();

    inputMap.put(-Double.MAX_VALUE, IValue.long64(Long.MIN_VALUE));
    inputMap.put(Double.MAX_VALUE, IValue.long64(Long.MAX_VALUE));
    inputMap.put(0.d, IValue.long64(0l));
    inputMap.put(-1.d, IValue.long64(-1l));
    inputMap.put(1.d, IValue.long64(1l));

    final IValue input = IValue.dictDoubleKey(inputMap);
    assertTrue(IValue.TYPE_CODE_DICT_DOUBLE_KEY == input.typeCode);

    final IValue output = module.forward(input);
    assertTrue(IValue.TYPE_CODE_DICT_DOUBLE_KEY == output.typeCode);
    final Map<Double, IValue> outputMap = output.getDictDoubleKey();
    assertTrue(inputMap.size() == outputMap.size());
    for (Map.Entry<Double, IValue> entry : inputMap.entrySet()) {
      assertTrue(outputMap.get(entry.getKey()).getLong() == entry.getValue().getLong());
    }
  }

  @Test
  public void testListIntSumReturnTuple() throws IOException {
    final PytorchScriptModule module = new PytorchScriptModule(assetFilePath(
        "ListIntSumReturnTuple.pt"));

    int n = 1;
    List<IValue> list = new ArrayList(n);
    long sum = 0;
    for (int i = 0; i < n; i++) {
      long l = i;
      list.add(IValue.long64(l));
      sum += l;
    }
    final IValue input = IValue.list(list);
    assertTrue(IValue.TYPE_CODE_LIST == input.typeCode);

    final IValue output = module.forward(input);

    assertTrue(IValue.TYPE_CODE_TUPLE == output.typeCode);
    assertTrue(2 == output.getTuple().length);

    IValue output0 = output.getTuple()[0];
    IValue output1 = output.getTuple()[1];

    assertTrue(IValue.TYPE_CODE_LIST == output0.typeCode);
    IValue[] output0List = output0.getList();
    for (int i = 0; i < output0List.length; i++) {
      long l = i;
      assertTrue(l == output0List[i].getLong());
    }

    assertTrue(sum == output1.getLong());
  }

  private static Bitmap assetAsBitmap(String assetName) throws IOException {
    final Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
    try (InputStream is = appContext.getAssets().open(assetName)) {
      return BitmapFactory.decodeStream(is);
    } catch (IOException e) {
      throw e;
    }
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
