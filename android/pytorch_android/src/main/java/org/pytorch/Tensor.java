package org.pytorch;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;
import java.util.Locale;

/**
 * Representation of Tensor. Tensor shape is stored in {@link Tensor#shape}, elements are stored as
 * {@link java.nio.DirectByteBuffer} of one of the supported types.
 */
public abstract class Tensor {

  /** Code for dtype torch.uint8. {@link Tensor#dtype()} */
  public static final int DTYPE_UINT8 = 1;
  /** Code for dtype torch.int8. {@link Tensor#dtype()} */
  public static final int DTYPE_INT8 = 2;
  /** Code for dtype torch.int32. {@link Tensor#dtype()} */
  public static final int DTYPE_INT32 = 3;
  /** Code for dtype torch.float32. {@link Tensor#dtype()} */
  public static final int DTYPE_FLOAT32 = 4;
  /** Code for dtype torch.int64. {@link Tensor#dtype()} */
  public static final int DTYPE_INT64 = 5;
  /** Code for dtype torch.float64. {@link Tensor#dtype()} */
  public static final int DTYPE_FLOAT64 = 6;

  private static final String ERROR_MSG_DATA_BUFFER_NOT_NULL = "Data buffer must be not null";
  private static final String ERROR_MSG_DATA_ARRAY_NOT_NULL = "Data array must be not null";
  private static final String ERROR_MSG_SHAPE_NOT_NULL = "Shape must be not null";
  private static final String ERROR_MSG_SHAPE_NOT_EMPTY = "Shape must be not empty";
  private static final String ERROR_MSG_SHAPE_NON_NEGATIVE = "Shape elements must be non negative";
  private static final String ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER =
      "Data buffer must have native byte order (java.nio.ByteOrder#nativeOrder)";
  private static final String ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT =
      "Data buffer must be direct (java.nio.ByteBuffer#allocateDirect)";

  /** Shape of current tensor. */
  public final long[] shape;

  private static final int INT_SIZE_BYTES = 4;
  private static final int FLOAT_SIZE_BYTES = 4;
  private static final int LONG_SIZE_BYTES = 8;
  private static final int DOUBLE_SIZE_BYTES = 8;

  /**
   * Allocates a new direct {@link java.nio.ByteBuffer} with native byte order with specified
   * capacity that can be used in {@link Tensor#newInt8Tensor(long[], ByteBuffer)}, {@link
   * Tensor#newUInt8Tensor(long[], ByteBuffer)}.
   *
   * @param numElements capacity (number of elements) of result buffer.
   */
  public static ByteBuffer allocateByteBuffer(int numElements) {
    return ByteBuffer.allocateDirect(numElements).order(ByteOrder.nativeOrder());
  }

  public static IntBuffer allocateIntBuffer(int numElements) {
    return ByteBuffer.allocateDirect(numElements * INT_SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
        .asIntBuffer();
  }

  /**
   * Allocates a new direct {@link java.nio.FloatBuffer} with native byte order with specified
   * capacity that can be used in {@link Tensor#newFloat32Tensor(long[], FloatBuffer)}.
   *
   * @param numElements capacity (number of elements) of result buffer.
   */
  public static FloatBuffer allocateFloatBuffer(int numElements) {
    return ByteBuffer.allocateDirect(numElements * FLOAT_SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer();
  }

  /**
   * Allocates a new direct {@link java.nio.LongBuffer} with native byte order with specified
   * capacity that can be used in {@link Tensor#newInt64Tensor(long[], LongBuffer)}.
   *
   * @param numElements capacity (number of elements) of result buffer.
   */
  public static LongBuffer allocateLongBuffer(int numElements) {
    return ByteBuffer.allocateDirect(numElements * LONG_SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
        .asLongBuffer();
  }

  /**
   * Allocates a new direct {@link java.nio.DoubleBuffer} with native byte order with specified
   * capacity that can be used in {@link Tensor#newFloat64Tensor(long[], DoubleBuffer)}.
   *
   * @param numElements capacity (number of elements) of result buffer.
   */
  public static DoubleBuffer allocateDoubleBuffer(int numElements) {
    return ByteBuffer.allocateDirect(numElements * DOUBLE_SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
        .asDoubleBuffer();
  }

  /**
   * Creates a new Tensor instance with dtype torch.uint8 with specified shape and data as array of
   * bytes.
   *
   * @param shape Tensor shape
   * @param data Tensor elements
   */
  public static Tensor newUInt8Tensor(long[] shape, byte[] data) {
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.length, shape);
    final ByteBuffer byteBuffer = allocateByteBuffer((int) numel(shape));
    byteBuffer.put(data);
    return new Tensor_uint8(byteBuffer, shape);
  }

  /**
   * Creates a new Tensor instance with dtype torch.int8 with specified shape and data as array of
   * bytes.
   *
   * @param shape Tensor shape
   * @param data Tensor elements
   */
  public static Tensor newInt8Tensor(long[] shape, byte[] data) {
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.length, shape);
    final ByteBuffer byteBuffer = allocateByteBuffer((int) numel(shape));
    byteBuffer.put(data);
    return new Tensor_int8(byteBuffer, shape);
  }

  /**
   * Creates a new Tensor instance with dtype torch.int32 with specified shape and data as array of
   * ints.
   *
   * @param shape Tensor shape
   * @param data Tensor elements
   */
  public static Tensor newInt32Tensor(long[] shape, int[] data) {
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.length, shape);
    final IntBuffer intBuffer = allocateIntBuffer((int) numel(shape));
    intBuffer.put(data);
    return new Tensor_int32(intBuffer, shape);
  }

  /**
   * Creates a new Tensor instance with dtype torch.float32 with specified shape and data as array
   * of floats.
   *
   * @param shape Tensor shape
   * @param data Tensor elements
   */
  public static Tensor newFloat32Tensor(long[] shape, float[] data) {
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.length, shape);
    final FloatBuffer floatBuffer = allocateFloatBuffer((int) numel(shape));
    floatBuffer.put(data);
    return new Tensor_float32(floatBuffer, shape);
  }

  /**
   * Creates a new Tensor instance with dtype torch.int64 with specified shape and data as array of
   * longs.
   *
   * @param shape Tensor shape
   * @param data Tensor elements
   */
  public static Tensor newInt64Tensor(long[] shape, long[] data) {
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.length, shape);
    final LongBuffer longBuffer = allocateLongBuffer((int) numel(shape));
    longBuffer.put(data);
    return new Tensor_int64(longBuffer, shape);
  }

  /**
   * Creates a new Tensor instance with dtype torch.float64 with specified shape and data as array
   * of doubles.
   *
   * @param shape Tensor shape
   * @param data Tensor elements
   */
  public static Tensor newFloat64Tensor(long[] shape, double[] data) {
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.length, shape);
    final DoubleBuffer doubleBuffer = allocateDoubleBuffer((int) numel(shape));
    doubleBuffer.put(data);
    return new Tensor_float64(doubleBuffer, shape);
  }

  /**
   * Creates a new Tensor instance with dtype torch.uint8 with specified shape and data.
   *
   * @param shape Tensor shape
   * @param data Direct buffer with native byte order that contains {@code Tensor#numel(shape)}
   *     elements. The buffer is used directly without copying, and changes to its content will
   *     change the tensor.
   */
  public static Tensor newUInt8Tensor(long[] shape, ByteBuffer data) {
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.capacity(), shape);
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    checkArgument(
        (data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    return new Tensor_uint8(data, shape);
  }

  /**
   * Creates a new Tensor instance with dtype torch.int8 with specified shape and data.
   *
   * @param shape Tensor shape
   * @param data Direct buffer with native byte order that contains {@code Tensor#numel(shape)}
   *     elements. The buffer is used directly without copying, and changes to its content will
   *     change the tensor.
   */
  public static Tensor newInt8Tensor(long[] shape, ByteBuffer data) {
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.capacity(), shape);
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    checkArgument(
        (data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    return new Tensor_int8(data, shape);
  }

  /**
   * Creates a new Tensor instance with dtype torch.int32 with specified shape and data.
   *
   * @param shape Tensor shape
   * @param data Direct buffer with native byte order that contains {@code Tensor#numel(shape)}
   *     elements. The buffer is used directly without copying, and changes to its content will
   *     change the tensor.
   */
  public static Tensor newInt32Tensor(long[] shape, IntBuffer data) {
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.capacity(), shape);
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    checkArgument(
        (data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    return new Tensor_int32(data, shape);
  }

  /**
   * Creates a new Tensor instance with dtype torch.float32 with specified shape and data.
   *
   * @param shape Tensor shape
   * @param data Direct buffer with native byte order that contains {@code Tensor#numel(shape)}
   *     elements. The buffer is used directly without copying, and changes to its content will
   *     change the tensor.
   */
  public static Tensor newFloat32Tensor(long[] shape, FloatBuffer data) {
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.capacity(), shape);
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    checkArgument(
        (data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    return new Tensor_float32(data, shape);
  }

  /**
   * Creates a new Tensor instance with dtype torch.int64 with specified shape and data.
   *
   * @param shape Tensor shape
   * @param data Direct buffer with native byte order that contains {@code Tensor#numel(shape)}
   *     elements. The buffer is used directly without copying, and changes to its content will
   *     change the tensor.
   */
  public static Tensor newInt64Tensor(long[] shape, LongBuffer data) {
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.capacity(), shape);
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    checkArgument(
        (data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    return new Tensor_int64(data, shape);
  }

  /**
   * Creates a new Tensor instance with dtype torch.float64 with specified shape and data.
   *
   * @param shape Tensor shape
   * @param data Direct buffer with native byte order that contains {@code Tensor#numel(shape)}
   *     elements. The buffer is used directly without copying, and changes to its content will
   *     change the tensor.
   */
  public static Tensor newFloat64Tensor(long[] shape, DoubleBuffer data) {
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.capacity(), shape);
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    checkArgument(
        (data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    return new Tensor_float64(data, shape);
  }

  private Tensor(long[] shape) {
    checkShape(shape);
    this.shape = Arrays.copyOf(shape, shape.length);
  }

  /** Calculates number of elements in current tensor instance. */
  public long numel() {
    return numel(this.shape);
  }

  /** Calculates number of elements in tensor with specified shape. */
  public static long numel(long[] shape) {
    checkShape(shape);
    int result = 1;
    for (long s : shape) {
      result *= s;
    }
    return result;
  }

  /**
   * Returns dtype of current tensor. Can be one of {@link Tensor#DTYPE_UINT8}, {@link
   * Tensor#DTYPE_INT8}, {@link Tensor#DTYPE_INT32},{@link Tensor#DTYPE_FLOAT32}, {@link
   * Tensor#DTYPE_INT64}, {@link Tensor#DTYPE_FLOAT64}.
   */
  public abstract int dtype();

  /**
   * Returns newly allocated java byte array that contains a copy of tensor data.
   *
   * @throws IllegalStateException if it is called for a non-int8 tensor.
   */
  public byte[] getDataAsByteArray() {
    throw new IllegalStateException(
        "Tensor of type " + getClass().getSimpleName() + " cannot return data as byte array.");
  }

  /**
   * Returns newly allocated java byte array that contains a copy of tensor data.
   *
   * @throws IllegalStateException if it is called for a non-uint8 tensor.
   */
  public byte[] getDataAsUnsignedByteArray() {
    throw new IllegalStateException(
        "Tensor of type " + getClass().getSimpleName() + " cannot return data as byte array.");
  }

  /**
   * Returns newly allocated java byte array that contains a copy of tensor data.
   *
   * @throws IllegalStateException if it is called for a non-int32 tensor.
   */
  public int[] getDataAsIntArray() {
    throw new IllegalStateException(
        "Tensor of type " + getClass().getSimpleName() + " cannot return data as int array.");
  }

  /**
   * Returns newly allocated java byte array that contains a copy of tensor data.
   *
   * @throws IllegalStateException if it is called for a non-float32 tensor.
   */
  public float[] getDataAsFloatArray() {
    throw new IllegalStateException(
        "Tensor of type " + getClass().getSimpleName() + " cannot return data as float array.");
  }

  /**
   * Returns newly allocated java byte array that contains a copy of tensor data.
   *
   * @throws IllegalStateException if it is called for a non-int64 tensor.
   */
  public long[] getDataAsLongArray() {
    throw new IllegalStateException(
        "Tensor of type " + getClass().getSimpleName() + " cannot return data as float array.");
  }

  /**
   * Returns newly allocated java byte array that contains a copy of tensor data.
   *
   * @throws IllegalStateException if it is called for a non-float64 tensor.
   */
  public double[] getDataAsDoubleArray() {
    throw new IllegalStateException(
        "Tensor of type " + getClass().getSimpleName() + " cannot return data as double array.");
  }

  Buffer getRawDataBuffer() {
    throw new IllegalStateException(
        "Tensor of type " + getClass().getSimpleName() + " cannot " + "return raw data buffer.");
  }

  static class Tensor_uint8 extends Tensor {
    private final ByteBuffer data;

    private Tensor_uint8(ByteBuffer data, long[] shape) {
      super(shape);
      this.data = data;
    }

    @Override
    public int dtype() {
      return DTYPE_UINT8;
    }

    @Override
    Buffer getRawDataBuffer() {
      return data;
    }

    @Override
    public byte[] getDataAsUnsignedByteArray() {
      data.rewind();
      byte[] arr = new byte[data.remaining()];
      data.get(arr);
      return arr;
    }

    @Override
    public String toString() {
      return String.format("Tensor(%s, dtype=torch.uint8)", Arrays.toString(shape));
    }
  }

  static class Tensor_int8 extends Tensor {
    private final ByteBuffer data;

    private Tensor_int8(ByteBuffer data, long[] shape) {
      super(shape);
      this.data = data;
    }

    @Override
    public int dtype() {
      return DTYPE_INT8;
    }

    @Override
    Buffer getRawDataBuffer() {
      return data;
    }

    @Override
    public byte[] getDataAsByteArray() {
      data.rewind();
      byte[] arr = new byte[data.remaining()];
      data.get(arr);
      return arr;
    }

    @Override
    public String toString() {
      return String.format("Tensor(%s, dtype=torch.int8)", Arrays.toString(shape));
    }
  }

  static class Tensor_int32 extends Tensor {
    private final IntBuffer data;

    private Tensor_int32(IntBuffer data, long[] shape) {
      super(shape);
      this.data = data;
    }

    @Override
    public int dtype() {
      return DTYPE_INT32;
    }

    @Override
    Buffer getRawDataBuffer() {
      return data;
    }

    @Override
    public int[] getDataAsIntArray() {
      data.rewind();
      int[] arr = new int[data.remaining()];
      data.get(arr);
      return arr;
    }

    @Override
    public String toString() {
      return String.format("Tensor(%s, dtype=torch.int32)", Arrays.toString(shape));
    }
  }

  static class Tensor_float32 extends Tensor {
    private final FloatBuffer data;

    Tensor_float32(FloatBuffer data, long[] shape) {
      super(shape);
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
    public int dtype() {
      return DTYPE_FLOAT32;
    }

    @Override
    Buffer getRawDataBuffer() {
      return data;
    }

    @Override
    public String toString() {
      return String.format("Tensor(%s, dtype=torch.float32)", Arrays.toString(shape));
    }
  }

  static class Tensor_int64 extends Tensor {
    private final LongBuffer data;

    private Tensor_int64(LongBuffer data, long[] shape) {
      super(shape);
      this.data = data;
    }

    @Override
    public int dtype() {
      return DTYPE_INT64;
    }

    @Override
    Buffer getRawDataBuffer() {
      return data;
    }

    @Override
    public long[] getDataAsLongArray() {
      data.rewind();
      long[] arr = new long[data.remaining()];
      data.get(arr);
      return arr;
    }

    @Override
    public String toString() {
      return String.format("Tensor(%s, dtype=torch.int64)", Arrays.toString(shape));
    }
  }

  static class Tensor_float64 extends Tensor {
    private final DoubleBuffer data;

    private Tensor_float64(DoubleBuffer data, long[] shape) {
      super(shape);
      this.data = data;
    }

    @Override
    public int dtype() {
      return DTYPE_FLOAT64;
    }

    @Override
    Buffer getRawDataBuffer() {
      return data;
    }

    @Override
    public double[] getDataAsDoubleArray() {
      data.rewind();
      double[] arr = new double[data.remaining()];
      data.get(arr);
      return arr;
    }

    @Override
    public String toString() {
      return String.format("Tensor(%s, dtype=torch.float64)", Arrays.toString(shape));
    }
  }

  // region checks
  private static void checkArgument(boolean expression, String errorMessage, Object... args) {
    if (!expression) {
      throw new IllegalArgumentException(String.format(Locale.US, errorMessage, args));
    }
  }

  private static void checkShape(long[] shape) {
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkArgument(shape.length > 0, ERROR_MSG_SHAPE_NOT_EMPTY);
    for (int i = 0; i < shape.length; i++) {
      checkArgument(shape[i] >= 0, ERROR_MSG_SHAPE_NON_NEGATIVE);
    }
  }

  private static void checkShapeAndDataCapacityConsistency(int dataCapacity, long[] shape) {
    final long numel = numel(shape);
    checkArgument(
        numel == dataCapacity,
        "Inconsistent data capacity:%d and shape number elements:%d shape:%s",
        dataCapacity,
        numel,
        Arrays.toString(shape));
  }
  // endregion checks

  // Called from native
  private static Tensor nativeNewTensor(ByteBuffer data, long[] shape, int dtype) {
    if (DTYPE_FLOAT32 == dtype) {
      return new Tensor_float32(data.asFloatBuffer(), shape);
    } else if (DTYPE_INT32 == dtype) {
      return new Tensor_int32(data.asIntBuffer(), shape);
    } else if (DTYPE_INT64 == dtype) {
      return new Tensor_int64(data.asLongBuffer(), shape);
    } else if (DTYPE_FLOAT64 == dtype) {
      return new Tensor_float64(data.asDoubleBuffer(), shape);
    } else if (DTYPE_UINT8 == dtype) {
      return new Tensor_uint8(data, shape);
    } else if (DTYPE_INT8 == dtype) {
      return new Tensor_int8(data, shape);
    }
    throw new IllegalArgumentException("Unknown Tensor dtype");
  }
}
