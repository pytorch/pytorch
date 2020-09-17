package org.pytorch;

import com.facebook.jni.HybridData;
import com.facebook.jni.annotations.DoNotStrip;
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
 * Representation of a Tensor. Behavior is similar to PyTorch's tensor objects.
 *
 * <p>Most tensors will be constructed as {@code Tensor.fromBlob(data, shape)}, where {@code data}
 * can be an array or a direct {@link Buffer} (of the proper subclass). Helper methods are provided
 * to allocate buffers properly.
 *
 * <p>To access Tensor data, see {@link #dtype()}, {@link #shape()}, and various {@code getDataAs*}
 * methods.
 *
 * <p>When constructing {@code Tensor} objects with {@code data} as an array, it is not specified
 * whether this data is is copied or retained as a reference so it is recommended not to modify it
 * after constructing. {@code data} passed as a {@link Buffer} is not copied, so it can be modified
 * between {@link Module} calls to avoid reallocation. Data retrieved from {@code Tensor} objects
 * may be copied or may be a reference to the {@code Tensor}'s internal data buffer. {@code shape}
 * is always copied.
 */
public abstract class Tensor {
  private static final String ERROR_MSG_DATA_BUFFER_NOT_NULL = "Data buffer must be not null";
  private static final String ERROR_MSG_DATA_ARRAY_NOT_NULL = "Data array must be not null";
  private static final String ERROR_MSG_SHAPE_NOT_NULL = "Shape must be not null";
  private static final String ERROR_MSG_SHAPE_NON_NEGATIVE = "Shape elements must be non negative";
  private static final String ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER =
      "Data buffer must have native byte order (java.nio.ByteOrder#nativeOrder)";
  private static final String ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT =
      "Data buffer must be direct (java.nio.ByteBuffer#allocateDirect)";

  @DoNotStrip final long[] shape;
  final MemoryFormat memoryFormat;

  private static final int INT_SIZE_BYTES = 4;
  private static final int FLOAT_SIZE_BYTES = 4;
  private static final int LONG_SIZE_BYTES = 8;
  private static final int DOUBLE_SIZE_BYTES = 8;

  /**
   * Allocates a new direct {@link java.nio.ByteBuffer} with native byte order with specified
   * capacity that can be used in {@link Tensor#fromBlob(ByteBuffer, long[])}, {@link
   * Tensor#fromBlobUnsigned(ByteBuffer, long[])}.
   *
   * @param numElements capacity (number of elements) of result buffer.
   */
  public static ByteBuffer allocateByteBuffer(int numElements) {
    return ByteBuffer.allocateDirect(numElements).order(ByteOrder.nativeOrder());
  }

  /**
   * Allocates a new direct {@link java.nio.IntBuffer} with native byte order with specified
   * capacity that can be used in {@link Tensor#fromBlob(IntBuffer, long[])}.
   *
   * @param numElements capacity (number of elements) of result buffer.
   */
  public static IntBuffer allocateIntBuffer(int numElements) {
    return ByteBuffer.allocateDirect(numElements * INT_SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
        .asIntBuffer();
  }

  /**
   * Allocates a new direct {@link java.nio.FloatBuffer} with native byte order with specified
   * capacity that can be used in {@link Tensor#fromBlob(FloatBuffer, long[])}.
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
   * capacity that can be used in {@link Tensor#fromBlob(LongBuffer, long[])}.
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
   * capacity that can be used in {@link Tensor#fromBlob(DoubleBuffer, long[])}.
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
   * @param data Tensor elements
   * @param shape Tensor shape
   */
  public static Tensor fromBlobUnsigned(byte[] data, long[] shape, MemoryFormat memoryFormat) {
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.length, shape);
    final ByteBuffer byteBuffer = allocateByteBuffer((int) numel(shape));
    byteBuffer.put(data);
    return new Tensor_uint8(byteBuffer, shape, memoryFormat);
  }

  public static Tensor fromBlobUnsigned(byte[] data, long[] shape) {
    return fromBlobUnsigned(data, shape, MemoryFormat.CONTIGUOUS);
  }

  /**
   * Creates a new Tensor instance with dtype torch.int8 with specified shape and data as array of
   * bytes.
   *
   * @param data Tensor elements
   * @param shape Tensor shape
   */
  public static Tensor fromBlob(byte[] data, long[] shape, MemoryFormat memoryFormat) {
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.length, shape);
    final ByteBuffer byteBuffer = allocateByteBuffer((int) numel(shape));
    byteBuffer.put(data);
    return new Tensor_int8(byteBuffer, shape, memoryFormat);
  }

  public static Tensor fromBlob(byte[] data, long[] shape) {
    return fromBlob(data, shape, MemoryFormat.CONTIGUOUS);
  }

  /**
   * Creates a new Tensor instance with dtype torch.int32 with specified shape and data as array of
   * ints.
   *
   * @param data Tensor elements
   * @param shape Tensor shape
   */
  public static Tensor fromBlob(int[] data, long[] shape, MemoryFormat memoryFormat) {
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.length, shape);
    final IntBuffer intBuffer = allocateIntBuffer((int) numel(shape));
    intBuffer.put(data);
    return new Tensor_int32(intBuffer, shape, memoryFormat);
  }

  public static Tensor fromBlob(int[] data, long[] shape) {
    return fromBlob(data, shape, MemoryFormat.CONTIGUOUS);
  }

  /**
   * Creates a new Tensor instance with dtype torch.float32 with specified shape and data as array
   * of floats.
   *
   * @param data Tensor elements
   * @param shape Tensor shape
   */
  public static Tensor fromBlob(float[] data, long[] shape, MemoryFormat memoryFormat) {
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.length, shape);
    final FloatBuffer floatBuffer = allocateFloatBuffer((int) numel(shape));
    floatBuffer.put(data);
    return new Tensor_float32(floatBuffer, shape, memoryFormat);
  }

  public static Tensor fromBlob(float[] data, long[] shape) {
    return fromBlob(data, shape, MemoryFormat.CONTIGUOUS);
  }

  /**
   * Creates a new Tensor instance with dtype torch.int64 with specified shape and data as array of
   * longs.
   *
   * @param data Tensor elements
   * @param shape Tensor shape
   */
  public static Tensor fromBlob(long[] data, long[] shape, MemoryFormat memoryFormat) {
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.length, shape);
    final LongBuffer longBuffer = allocateLongBuffer((int) numel(shape));
    longBuffer.put(data);
    return new Tensor_int64(longBuffer, shape, memoryFormat);
  }

  public static Tensor fromBlob(long[] data, long[] shape) {
    return fromBlob(data, shape, MemoryFormat.CONTIGUOUS);
  }

  /**
   * Creates a new Tensor instance with dtype torch.float64 with specified shape and data as array
   * of doubles.
   *
   * @param shape Tensor shape
   * @param data Tensor elements
   */
  public static Tensor fromBlob(double[] data, long[] shape, MemoryFormat memoryFormat) {
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.length, shape);
    final DoubleBuffer doubleBuffer = allocateDoubleBuffer((int) numel(shape));
    doubleBuffer.put(data);
    return new Tensor_float64(doubleBuffer, shape, memoryFormat);
  }

  public static Tensor fromBlob(double[] data, long[] shape) {
    return fromBlob(data, shape, MemoryFormat.CONTIGUOUS);
  }

  /**
   * Creates a new Tensor instance with dtype torch.uint8 with specified shape and data.
   *
   * @param data Direct buffer with native byte order that contains {@code Tensor.numel(shape)}
   *     elements. The buffer is used directly without copying, and changes to its content will
   *     change the tensor.
   * @param shape Tensor shape
   */
  public static Tensor fromBlobUnsigned(ByteBuffer data, long[] shape, MemoryFormat memoryFormat) {
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.capacity(), shape);
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    checkArgument(
        (data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    return new Tensor_uint8(data, shape, memoryFormat);
  }

  public static Tensor fromBlobUnsigned(ByteBuffer data, long[] shape) {
    return fromBlobUnsigned(data, shape, MemoryFormat.CONTIGUOUS);
  }

  /**
   * Creates a new Tensor instance with dtype torch.int8 with specified shape and data.
   *
   * @param data Direct buffer with native byte order that contains {@code Tensor.numel(shape)}
   *     elements. The buffer is used directly without copying, and changes to its content will
   *     change the tensor.
   * @param shape Tensor shape
   */
  public static Tensor fromBlob(ByteBuffer data, long[] shape, MemoryFormat memoryFormat) {
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.capacity(), shape);
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    checkArgument(
        (data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    return new Tensor_int8(data, shape, memoryFormat);
  }

  public static Tensor fromBlob(ByteBuffer data, long[] shape) {
    return fromBlob(data, shape, MemoryFormat.CONTIGUOUS);
  }

  /**
   * Creates a new Tensor instance with dtype torch.int32 with specified shape and data.
   *
   * @param data Direct buffer with native byte order that contains {@code Tensor.numel(shape)}
   *     elements. The buffer is used directly without copying, and changes to its content will
   *     change the tensor.
   * @param shape Tensor shape
   */
  public static Tensor fromBlob(IntBuffer data, long[] shape, MemoryFormat memoryFormat) {
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.capacity(), shape);
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    checkArgument(
        (data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    return new Tensor_int32(data, shape, memoryFormat);
  }

  public static Tensor fromBlob(IntBuffer data, long[] shape) {
    return fromBlob(data, shape, MemoryFormat.CONTIGUOUS);
  }

  /**
   * Creates a new Tensor instance with dtype torch.float32 with specified shape and data.
   *
   * @param data Direct buffer with native byte order that contains {@code Tensor.numel(shape)}
   *     elements. The buffer is used directly without copying, and changes to its content will
   *     change the tensor.
   * @param shape Tensor shape
   */
  public static Tensor fromBlob(FloatBuffer data, long[] shape, MemoryFormat memoryFormat) {
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.capacity(), shape);
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    checkArgument(
        (data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    return new Tensor_float32(data, shape, memoryFormat);
  }

  public static Tensor fromBlob(FloatBuffer data, long[] shape) {
    return fromBlob(data, shape, MemoryFormat.CONTIGUOUS);
  }

  /**
   * Creates a new Tensor instance with dtype torch.int64 with specified shape and data.
   *
   * @param data Direct buffer with native byte order that contains {@code Tensor.numel(shape)}
   *     elements. The buffer is used directly without copying, and changes to its content will
   *     change the tensor.
   * @param shape Tensor shape
   */
  public static Tensor fromBlob(LongBuffer data, long[] shape, MemoryFormat memoryFormat) {
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.capacity(), shape);
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    checkArgument(
        (data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    return new Tensor_int64(data, shape, memoryFormat);
  }

  public static Tensor fromBlob(LongBuffer data, long[] shape) {
    return fromBlob(data, shape, MemoryFormat.CONTIGUOUS);
  }

  /**
   * Creates a new Tensor instance with dtype torch.float64 with specified shape and data.
   *
   * @param data Direct buffer with native byte order that contains {@code Tensor.numel(shape)}
   *     elements. The buffer is used directly without copying, and changes to its content will
   *     change the tensor.
   * @param shape Tensor shape
   */
  public static Tensor fromBlob(DoubleBuffer data, long[] shape, MemoryFormat memoryFormat) {
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.capacity(), shape);
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    checkArgument(
        (data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    return new Tensor_float64(data, shape, memoryFormat);
  }

  public static Tensor fromBlob(DoubleBuffer data, long[] shape) {
    return fromBlob(data, shape, MemoryFormat.CONTIGUOUS);
  }

  @DoNotStrip private HybridData mHybridData;

  private Tensor(long[] shape, MemoryFormat memoryFormat) {
    checkShape(shape);
    this.shape = Arrays.copyOf(shape, shape.length);
    this.memoryFormat = memoryFormat;
  }

  /** Returns the number of elements in this tensor. */
  public long numel() {
    return numel(this.shape);
  }

  /** Calculates the number of elements in a tensor with the specified shape. */
  public static long numel(long[] shape) {
    checkShape(shape);
    int result = 1;
    for (long s : shape) {
      result *= s;
    }
    return result;
  }

  /** Returns the shape of this tensor. (The array is a fresh copy.) */
  public long[] shape() {
    return Arrays.copyOf(shape, shape.length);
  }

  /** Returns the memory format of this tensor. */
  public MemoryFormat memoryFormat() {
    return memoryFormat;
  }

  /** @return data type of this tensor. */
  public abstract DType dtype();

  // Called from native
  @DoNotStrip
  int dtypeJniCode() {
    return dtype().jniCode;
  }

  // Called from native
  @DoNotStrip
  int memoryFormatJniCode() {
    return memoryFormat.jniCode;
  }

  /**
   * @return a Java byte array that contains the tensor data. This may be a copy or reference.
   * @throws IllegalStateException if it is called for a non-int8 tensor.
   */
  public byte[] getDataAsByteArray() {
    throw new IllegalStateException(
        "Tensor of type " + getClass().getSimpleName() + " cannot return data as byte array.");
  }

  /**
   * @return a Java byte array that contains the tensor data. This may be a copy or reference.
   * @throws IllegalStateException if it is called for a non-uint8 tensor.
   */
  public byte[] getDataAsUnsignedByteArray() {
    throw new IllegalStateException(
        "Tensor of type " + getClass().getSimpleName() + " cannot return data as byte array.");
  }

  /**
   * @return a Java int array that contains the tensor data. This may be a copy or reference.
   * @throws IllegalStateException if it is called for a non-int32 tensor.
   */
  public int[] getDataAsIntArray() {
    throw new IllegalStateException(
        "Tensor of type " + getClass().getSimpleName() + " cannot return data as int array.");
  }

  /**
   * @return a Java float array that contains the tensor data. This may be a copy or reference.
   * @throws IllegalStateException if it is called for a non-float32 tensor.
   */
  public float[] getDataAsFloatArray() {
    throw new IllegalStateException(
        "Tensor of type " + getClass().getSimpleName() + " cannot return data as float array.");
  }

  /**
   * @return a Java long array that contains the tensor data. This may be a copy or reference.
   * @throws IllegalStateException if it is called for a non-int64 tensor.
   */
  public long[] getDataAsLongArray() {
    throw new IllegalStateException(
        "Tensor of type " + getClass().getSimpleName() + " cannot return data as long array.");
  }

  /**
   * @return a Java double array that contains the tensor data. This may be a copy or reference.
   * @throws IllegalStateException if it is called for a non-float64 tensor.
   */
  public double[] getDataAsDoubleArray() {
    throw new IllegalStateException(
        "Tensor of type " + getClass().getSimpleName() + " cannot return data as double array.");
  }

  @DoNotStrip
  Buffer getRawDataBuffer() {
    throw new IllegalStateException(
        "Tensor of type " + getClass().getSimpleName() + " cannot " + "return raw data buffer.");
  }

  static class Tensor_uint8 extends Tensor {
    private final ByteBuffer data;

    private Tensor_uint8(ByteBuffer data, long[] shape, MemoryFormat memoryFormat) {
      super(shape, memoryFormat);
      this.data = data;
    }

    @Override
    public DType dtype() {
      return DType.UINT8;
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

    private Tensor_int8(ByteBuffer data, long[] shape, MemoryFormat memoryFormat) {
      super(shape, memoryFormat);
      this.data = data;
    }

    @Override
    public DType dtype() {
      return DType.INT8;
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

    private Tensor_int32(IntBuffer data, long[] shape, MemoryFormat memoryFormat) {
      super(shape, memoryFormat);
      this.data = data;
    }

    @Override
    public DType dtype() {
      return DType.INT32;
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

    Tensor_float32(FloatBuffer data, long[] shape, MemoryFormat memoryFormat) {
      super(shape, memoryFormat);
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
    public DType dtype() {
      return DType.FLOAT32;
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

    private Tensor_int64(LongBuffer data, long[] shape, MemoryFormat memoryFormat) {
      super(shape, memoryFormat);
      this.data = data;
    }

    @Override
    public DType dtype() {
      return DType.INT64;
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

    private Tensor_float64(DoubleBuffer data, long[] shape, MemoryFormat memoryFormat) {
      super(shape, memoryFormat);
      this.data = data;
    }

    @Override
    public DType dtype() {
      return DType.FLOAT64;
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
  @DoNotStrip
  private static Tensor nativeNewTensor(
      ByteBuffer data, long[] shape, int dtype, int memoryFormatCode, HybridData hybridData) {
    Tensor tensor = null;

    MemoryFormat memoryFormat = MemoryFormat.CONTIGUOUS;
    if (MemoryFormat.CHANNELS_LAST.jniCode == memoryFormatCode) {
      memoryFormat = MemoryFormat.CHANNELS_LAST;
    } else if (MemoryFormat.CHANNELS_LAST_3D.jniCode == memoryFormatCode) {
      memoryFormat = MemoryFormat.CHANNELS_LAST_3D;
    }

    if (DType.FLOAT32.jniCode == dtype) {
      tensor = new Tensor_float32(data.asFloatBuffer(), shape, memoryFormat);
    } else if (DType.INT32.jniCode == dtype) {
      tensor = new Tensor_int32(data.asIntBuffer(), shape, memoryFormat);
    } else if (DType.INT64.jniCode == dtype) {
      tensor = new Tensor_int64(data.asLongBuffer(), shape, memoryFormat);
    } else if (DType.FLOAT64.jniCode == dtype) {
      tensor = new Tensor_float64(data.asDoubleBuffer(), shape, memoryFormat);
    } else if (DType.UINT8.jniCode == dtype) {
      tensor = new Tensor_uint8(data, shape, memoryFormat);
    } else if (DType.INT8.jniCode == dtype) {
      tensor = new Tensor_int8(data, shape, memoryFormat);
    } else {
      new IllegalArgumentException("Unknown Tensor dtype");
    }
    tensor.mHybridData = hybridData;
    return tensor;
  }
}
