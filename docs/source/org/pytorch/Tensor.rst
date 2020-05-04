.. java:import:: java.nio Buffer

.. java:import:: java.nio ByteBuffer

.. java:import:: java.nio ByteOrder

.. java:import:: java.nio DoubleBuffer

.. java:import:: java.nio FloatBuffer

.. java:import:: java.nio IntBuffer

.. java:import:: java.nio LongBuffer

.. java:import:: java.util Arrays

.. java:import:: java.util Locale

Tensor
======

.. java:package:: org.pytorch
   :noindex:

.. java:type:: public abstract class Tensor

   Representation of a Tensor. Behavior is similar to PyTorch's tensor objects.

   Most tensors will be constructed as \ ``Tensor.fromBlob(data, shape)``\ , where \ ``data``\  can be an array or a direct \ :java:ref:`Buffer`\  (of the proper subclass). Helper methods are provided to allocate buffers properly.

   To access Tensor data, see \ :java:ref:`dtype()`\ , \ :java:ref:`shape()`\ , and various \ ``getDataAs*``\  methods.

   When constructing \ ``Tensor``\  objects with \ ``data``\  as an array, it is not specified whether this data is is copied or retained as a reference so it is recommended not to modify it after constructing. \ ``data``\  passed as a \ :java:ref:`Buffer`\  is not copied, so it can be modified between \ :java:ref:`Module`\  calls to avoid reallocation. Data retrieved from \ ``Tensor``\  objects may be copied or may be a reference to the \ ``Tensor``\ 's internal data buffer. \ ``shape``\  is always copied.

Methods
-------
allocateByteBuffer
^^^^^^^^^^^^^^^^^^

.. java:method:: public static ByteBuffer allocateByteBuffer(int numElements)
   :outertype: Tensor

   Allocates a new direct \ :java:ref:`java.nio.ByteBuffer`\  with native byte order with specified capacity that can be used in \ :java:ref:`Tensor.fromBlob(ByteBuffer,long[])`\ , \ :java:ref:`Tensor.fromBlobUnsigned(ByteBuffer,long[])`\ .

   :param numElements: capacity (number of elements) of result buffer.

allocateDoubleBuffer
^^^^^^^^^^^^^^^^^^^^

.. java:method:: public static DoubleBuffer allocateDoubleBuffer(int numElements)
   :outertype: Tensor

   Allocates a new direct \ :java:ref:`java.nio.DoubleBuffer`\  with native byte order with specified capacity that can be used in \ :java:ref:`Tensor.fromBlob(DoubleBuffer,long[])`\ .

   :param numElements: capacity (number of elements) of result buffer.

allocateFloatBuffer
^^^^^^^^^^^^^^^^^^^

.. java:method:: public static FloatBuffer allocateFloatBuffer(int numElements)
   :outertype: Tensor

   Allocates a new direct \ :java:ref:`java.nio.FloatBuffer`\  with native byte order with specified capacity that can be used in \ :java:ref:`Tensor.fromBlob(FloatBuffer,long[])`\ .

   :param numElements: capacity (number of elements) of result buffer.

allocateIntBuffer
^^^^^^^^^^^^^^^^^

.. java:method:: public static IntBuffer allocateIntBuffer(int numElements)
   :outertype: Tensor

   Allocates a new direct \ :java:ref:`java.nio.IntBuffer`\  with native byte order with specified capacity that can be used in \ :java:ref:`Tensor.fromBlob(IntBuffer,long[])`\ .

   :param numElements: capacity (number of elements) of result buffer.

allocateLongBuffer
^^^^^^^^^^^^^^^^^^

.. java:method:: public static LongBuffer allocateLongBuffer(int numElements)
   :outertype: Tensor

   Allocates a new direct \ :java:ref:`java.nio.LongBuffer`\  with native byte order with specified capacity that can be used in \ :java:ref:`Tensor.fromBlob(LongBuffer,long[])`\ .

   :param numElements: capacity (number of elements) of result buffer.

dtype
^^^^^

.. java:method:: public abstract DType dtype()
   :outertype: Tensor

   :return: data type of this tensor.

dtypeJniCode
^^^^^^^^^^^^

.. java:method::  int dtypeJniCode()
   :outertype: Tensor

fromBlob
^^^^^^^^

.. java:method:: public static Tensor fromBlob(byte[] data, long[] shape)
   :outertype: Tensor

   Creates a new Tensor instance with dtype torch.int8 with specified shape and data as array of bytes.

   :param data: Tensor elements
   :param shape: Tensor shape

fromBlob
^^^^^^^^

.. java:method:: public static Tensor fromBlob(int[] data, long[] shape)
   :outertype: Tensor

   Creates a new Tensor instance with dtype torch.int32 with specified shape and data as array of ints.

   :param data: Tensor elements
   :param shape: Tensor shape

fromBlob
^^^^^^^^

.. java:method:: public static Tensor fromBlob(float[] data, long[] shape)
   :outertype: Tensor

   Creates a new Tensor instance with dtype torch.float32 with specified shape and data as array of floats.

   :param data: Tensor elements
   :param shape: Tensor shape

fromBlob
^^^^^^^^

.. java:method:: public static Tensor fromBlob(long[] data, long[] shape)
   :outertype: Tensor

   Creates a new Tensor instance with dtype torch.int64 with specified shape and data as array of longs.

   :param data: Tensor elements
   :param shape: Tensor shape

fromBlob
^^^^^^^^

.. java:method:: public static Tensor fromBlob(long[] shape, double[] data)
   :outertype: Tensor

   Creates a new Tensor instance with dtype torch.float64 with specified shape and data as array of doubles.

   :param shape: Tensor shape
   :param data: Tensor elements

fromBlob
^^^^^^^^

.. java:method:: public static Tensor fromBlob(ByteBuffer data, long[] shape)
   :outertype: Tensor

   Creates a new Tensor instance with dtype torch.int8 with specified shape and data.

   :param data: Direct buffer with native byte order that contains \ ``Tensor.numel(shape)``\  elements. The buffer is used directly without copying, and changes to its content will change the tensor.
   :param shape: Tensor shape

fromBlob
^^^^^^^^

.. java:method:: public static Tensor fromBlob(IntBuffer data, long[] shape)
   :outertype: Tensor

   Creates a new Tensor instance with dtype torch.int32 with specified shape and data.

   :param data: Direct buffer with native byte order that contains \ ``Tensor.numel(shape)``\  elements. The buffer is used directly without copying, and changes to its content will change the tensor.
   :param shape: Tensor shape

fromBlob
^^^^^^^^

.. java:method:: public static Tensor fromBlob(FloatBuffer data, long[] shape)
   :outertype: Tensor

   Creates a new Tensor instance with dtype torch.float32 with specified shape and data.

   :param data: Direct buffer with native byte order that contains \ ``Tensor.numel(shape)``\  elements. The buffer is used directly without copying, and changes to its content will change the tensor.
   :param shape: Tensor shape

fromBlob
^^^^^^^^

.. java:method:: public static Tensor fromBlob(LongBuffer data, long[] shape)
   :outertype: Tensor

   Creates a new Tensor instance with dtype torch.int64 with specified shape and data.

   :param data: Direct buffer with native byte order that contains \ ``Tensor.numel(shape)``\  elements. The buffer is used directly without copying, and changes to its content will change the tensor.
   :param shape: Tensor shape

fromBlob
^^^^^^^^

.. java:method:: public static Tensor fromBlob(DoubleBuffer data, long[] shape)
   :outertype: Tensor

   Creates a new Tensor instance with dtype torch.float64 with specified shape and data.

   :param data: Direct buffer with native byte order that contains \ ``Tensor.numel(shape)``\  elements. The buffer is used directly without copying, and changes to its content will change the tensor.
   :param shape: Tensor shape

fromBlobUnsigned
^^^^^^^^^^^^^^^^

.. java:method:: public static Tensor fromBlobUnsigned(byte[] data, long[] shape)
   :outertype: Tensor

   Creates a new Tensor instance with dtype torch.uint8 with specified shape and data as array of bytes.

   :param data: Tensor elements
   :param shape: Tensor shape

fromBlobUnsigned
^^^^^^^^^^^^^^^^

.. java:method:: public static Tensor fromBlobUnsigned(ByteBuffer data, long[] shape)
   :outertype: Tensor

   Creates a new Tensor instance with dtype torch.uint8 with specified shape and data.

   :param data: Direct buffer with native byte order that contains \ ``Tensor.numel(shape)``\  elements. The buffer is used directly without copying, and changes to its content will change the tensor.
   :param shape: Tensor shape

getDataAsByteArray
^^^^^^^^^^^^^^^^^^

.. java:method:: public byte[] getDataAsByteArray()
   :outertype: Tensor

   :throws IllegalStateException: if it is called for a non-int8 tensor.
   :return: a Java byte array that contains the tensor data. This may be a copy or reference.

getDataAsDoubleArray
^^^^^^^^^^^^^^^^^^^^

.. java:method:: public double[] getDataAsDoubleArray()
   :outertype: Tensor

   :throws IllegalStateException: if it is called for a non-float64 tensor.
   :return: a Java double array that contains the tensor data. This may be a copy or reference.

getDataAsFloatArray
^^^^^^^^^^^^^^^^^^^

.. java:method:: public float[] getDataAsFloatArray()
   :outertype: Tensor

   :throws IllegalStateException: if it is called for a non-float32 tensor.
   :return: a Java float array that contains the tensor data. This may be a copy or reference.

getDataAsIntArray
^^^^^^^^^^^^^^^^^

.. java:method:: public int[] getDataAsIntArray()
   :outertype: Tensor

   :throws IllegalStateException: if it is called for a non-int32 tensor.
   :return: a Java int array that contains the tensor data. This may be a copy or reference.

getDataAsLongArray
^^^^^^^^^^^^^^^^^^

.. java:method:: public long[] getDataAsLongArray()
   :outertype: Tensor

   :throws IllegalStateException: if it is called for a non-int64 tensor.
   :return: a Java long array that contains the tensor data. This may be a copy or reference.

getDataAsUnsignedByteArray
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. java:method:: public byte[] getDataAsUnsignedByteArray()
   :outertype: Tensor

   :throws IllegalStateException: if it is called for a non-uint8 tensor.
   :return: a Java byte array that contains the tensor data. This may be a copy or reference.

getRawDataBuffer
^^^^^^^^^^^^^^^^

.. java:method::  Buffer getRawDataBuffer()
   :outertype: Tensor

numel
^^^^^

.. java:method:: public long numel()
   :outertype: Tensor

   Returns the number of elements in this tensor.

numel
^^^^^

.. java:method:: public static long numel(long[] shape)
   :outertype: Tensor

   Calculates the number of elements in a tensor with the specified shape.

shape
^^^^^

.. java:method:: public long[] shape()
   :outertype: Tensor

   Returns the shape of this tensor. (The array is a fresh copy.)
