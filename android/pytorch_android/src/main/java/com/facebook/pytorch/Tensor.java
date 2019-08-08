package com.facebook.pytorch;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.Arrays;
import java.util.Locale;

public abstract class Tensor {
  private static final int TYPE_CODE_FLOAT32 = 0xF10A732;
  private static final int TYPE_CODE_INT32 = 0x14732;
  private static final int TYPE_CODE_BYTE = 0xB17E5;

  private static final String ERROR_MSG_DATA_BUFFER_NOT_NULL = "Data buffer must be not null";
  private static final String ERROR_MSG_DATA_ARRAY_NOT_NULL = "Data array must be not null";
  private static final String ERROR_MSG_DIMS_NOT_NULL = "Dims must be not null";
  private static final String ERROR_MSG_DIMS_NOT_EMPTY = "Dims must be not empty";
  private static final String ERROR_MSG_INDEX_NOT_NULL = "Index must be not null";
  private static final String ERROR_MSG_DIMS_POSITIVE = "Dims must be positive";
  private static final String ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER =
      "Data buffer must have native byte order (java.nio.ByteOrder#nativeOrder)";
  private static final String ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT =
      "Data buffer must be direct (java.nio.ByteBuffer#allocateDirect)";

  public final int[] dims;
  private final int[] dimsNumElements;

  private static final int FLOAT_SIZE_BYTES = 4;
  private static final int INT_SIZE_BYTES = 4;

  public static FloatBuffer allocateFloatBuffer(int numElements) {
    return ByteBuffer.allocateDirect(numElements * FLOAT_SIZE_BYTES).order(ByteOrder.nativeOrder()).asFloatBuffer();
  }

  public static IntBuffer allocateIntBuffer(int numElements) {
    return ByteBuffer.allocateDirect(numElements * INT_SIZE_BYTES).order(ByteOrder.nativeOrder()).asIntBuffer();
  }

  public static ByteBuffer allocateByteBuffer(int numElements) {
    return ByteBuffer.allocateDirect(numElements).order(ByteOrder.nativeOrder());
  }

  public static Tensor newFloatTensor(int[] dims, float[] data) {
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    checkArgument(dims != null, ERROR_MSG_DIMS_NOT_NULL);
    checkDims(dims);
    checkDimsAndDataCapacityConsistency(data.length, dims);

    final FloatBuffer floatBuffer = allocateFloatBuffer(numElements(dims));
    floatBuffer.put(data);
    return new Tensor_float32(floatBuffer, dims);
  }

  public static Tensor newIntTensor(int[] dims, int[] data) {
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    checkArgument(dims != null, ERROR_MSG_DIMS_NOT_NULL);
    checkDims(dims);
    checkDimsAndDataCapacityConsistency(data.length, dims);

    final IntBuffer intBuffer = allocateIntBuffer(numElements(dims));
    intBuffer.put(data);
    return new Tensor_int32(intBuffer, dims);
  }

  public static Tensor newByteTensor(int[] dims, byte[] data) {
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    checkArgument(dims != null, ERROR_MSG_DIMS_NOT_NULL);
    checkDims(dims);
    checkDimsAndDataCapacityConsistency(data.length, dims);

    final ByteBuffer byteBuffer = allocateByteBuffer(numElements(dims));
    byteBuffer.put(data);
    return new Tensor_byte(byteBuffer, dims);
  }

  public static Tensor newFloatTensor(int[] dims, FloatBuffer data) {
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    checkArgument(dims != null, ERROR_MSG_DIMS_NOT_NULL);
    checkDims(dims);
    checkDimsAndDataCapacityConsistency(data.capacity(), dims);
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    checkArgument((data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    return new Tensor_float32(data, dims);
  }

  public static Tensor newIntTensor(int[] dims, IntBuffer data) {
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    checkArgument(dims != null, ERROR_MSG_DIMS_NOT_NULL);
    checkDims(dims);
    checkDimsAndDataCapacityConsistency(data.capacity(), dims);
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    checkArgument((data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    return new Tensor_int32(data, dims);
  }

  public static Tensor newByteTensor(int[] dims, ByteBuffer data) {
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    checkArgument(dims != null, ERROR_MSG_DIMS_NOT_NULL);
    checkDims(dims);
    checkDimsAndDataCapacityConsistency(data.capacity(), dims);
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    checkArgument((data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    return new Tensor_byte(data, dims);
  }

  private Tensor(int[] dims) {
    checkDims(dims);
    this.dims = dims;
    dimsNumElements = new int[dims.length];
    int numElements = 1;
    for (int i = dims.length  - 1; i >= 0; i--) {
      dimsNumElements[i] = numElements;
      numElements *= dims[i];
    }
  }

  public static int numElements(int[] dims) {
    checkDims(dims);
    int result = 1;
    for (int dim : dims) {
      result *= dim;
    }
    return result;
  }

  public int indexToBufferPos(int... index) {
    Tensor.checkIndex(index, this.dims);
    return unsafeIndexToBufferPos(dims, index);
  }

  public static int indexToBufferPos(int dims[], int... index) {
    checkDims(dims);
    checkIndex(index, dims);
    return unsafeIndexToBufferPos(dims, index);
  }

  private static int unsafeIndexToBufferPos(int dims[], int... index) {
    int pos = 0;
    int numel = 1;
    for (int i = dims.length  - 1; i >= 0; i--) {
      pos += numel * index[i];
      numel *= dims[i];
    }
    return pos;
  }

  public float[] getDataAsFloatArray() {
    throw new IllegalStateException("Tensor of type " + getClass().getSimpleName() + " cannot " +
        "return data as float.");
  }

  abstract int getTypeCode();

  Buffer getRawDataBuffer() {
    throw new IllegalStateException("Tensor of type " + getClass().getSimpleName() + " cannot " +
        "return raw data buffer.");
  }

  private static String invalidIndexErrorMessage(int[] index, int dims[]) {
    return String.format(Locale.US, "Invalid index %s for tensor dimensions %s",
        Arrays.toString(index), Arrays.toString(dims));
  }

  static class Tensor_float32 extends Tensor {
    private final FloatBuffer data;

    Tensor_float32(FloatBuffer data, int[] dims) {
      super(dims);
      this.data = data;
    }

    @Override
    public float[] getDataAsFloatArray() {
      data.rewind();
      float[] arr = new float[data.remaining()];
      data.get(arr);
      return arr;
    }

    @Override
    int getTypeCode() {
      return TYPE_CODE_FLOAT32;
    }

    @Override
    Buffer getRawDataBuffer() {
      return data;
    }

    @Override
    public String toString() {
      return String.format("Tensor_float32{dims:%s data:%s}", Arrays.toString(dims),
          Arrays.toString(getDataAsFloatArray()));
    }
  }

  static class Tensor_int32 extends Tensor {
    private final IntBuffer data;

    private Tensor_int32(IntBuffer data, int[] dims) {
      super(dims);
      this.data = data;
    }

    @Override
    int getTypeCode() {
      return TYPE_CODE_INT32;
    }

    @Override
    Buffer getRawDataBuffer() {
      return data;
    }

    public int[] getDataAsIntArray() {
      data.rewind();
      int[] arr = new int[data.remaining()];
      data.get(arr);
      return arr;
    }

    @Override
    public String toString() {
      return String.format("Tensor_int32{dims:%s data:%s}", Arrays.toString(dims),
          Arrays.toString(getDataAsIntArray()));
    }
  }

  static class Tensor_byte extends Tensor {
    private final ByteBuffer data;

    private Tensor_byte(ByteBuffer data, int[] dims) {
      super(dims);
      this.data = data;
    }

    @Override
    int getTypeCode() {
      return TYPE_CODE_BYTE;
    }

    @Override
    Buffer getRawDataBuffer() {
      return data;
    }

    public byte[] getDataAsByteArray() {
      data.rewind();
      byte[] arr = new byte[data.remaining()];
      data.get(arr);
      return arr;
    }

    @Override
    public String toString() {
      return String.format("Tensor_byte{dims:%s data:%s}", Arrays.toString(dims),
          Arrays.toString(getDataAsByteArray()));
    }
  }

  //region checks
  private static void checkArgument(boolean expression, String errorMessage, Object... args) {
    if (!expression) {
      throw new IllegalArgumentException(String.format(Locale.US, errorMessage, args));
    }
  }

  private static void checkDims(int[] dims) {
    checkArgument(dims != null, ERROR_MSG_DIMS_NOT_NULL);
    checkArgument(dims.length > 0, ERROR_MSG_DIMS_NOT_EMPTY);
    for (int i = 0; i < dims.length; i++) {
      checkArgument(dims[i] > 0, ERROR_MSG_DIMS_POSITIVE);
    }
  }

  private static void checkIndex(int[] index, int dims[]) {
    checkArgument(dims != null, ERROR_MSG_INDEX_NOT_NULL);

    if (index.length != dims.length) {
      throw new IllegalArgumentException(invalidIndexErrorMessage(index, dims));
    }

    for (int i = 0; i < index.length; i++) {
      if (index[i] >= dims[i]) {
        throw new IllegalArgumentException(invalidIndexErrorMessage(index, dims));
      }
    }
  }

  private static void checkDimsAndDataCapacityConsistency(int dataCapacity, int[] dims) {
    final int numElements = numElements(dims);
    checkArgument(numElements == dataCapacity,
        "Inconsistent data capacity:%d and dims number elements:%d dims:%s",
        dataCapacity, numElements, Arrays.toString(dims));
  }

  private void checkType(int expectedTypeCode, String errorMsg) {
    if (getTypeCode() != expectedTypeCode) {
      throw new IllegalStateException(errorMsg);
    }
  }
  //endregion checks

  // Called from native
  private static Tensor nativeNewTensor(ByteBuffer data, int[] dims, int typeCode) {
    if (TYPE_CODE_FLOAT32 == typeCode) {
      return new Tensor_float32(data.asFloatBuffer(), dims);
    } else if (TYPE_CODE_INT32 == typeCode) {
      return new Tensor_int32(data.asIntBuffer(), dims);
    } else if (TYPE_CODE_BYTE == typeCode) {
      return new Tensor_byte(data, dims);
    }
    throw new IllegalArgumentException("Unknown typeCode");
  }
}
