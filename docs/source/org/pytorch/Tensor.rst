.. java:import:: java.nio Buffer

.. java:import:: java.nio ByteBuffer

.. java:import:: java.nio ByteOrder

.. java:import:: java.nio DoubleBuffer

.. java:import:: java.nio FloatBuffer

.. java:import:: java.nio IntBuffer

.. java:import:: java.nio LongBuffer

.. java:import:: java.util Arrays

.. java:import:: java.util Locale

org.pytorch.Tensor (Tensor)
===========================

Source Code
------------

Full code can be found in `Github <https://github.com/pytorch/pytorch/blob/master/android/pytorch_android/src/main/java/org/pytorch/Tensor.java>`_.

Overview
--------

Tensor supports dtypes ``uint8, int8, float32, int32, float64, int64``.
Tensor holds data in DirectByteBuffer of proper type with native bit order.

To create a Tensor user can use one of the factory methods:
::
    Tensor newUInt8Tensor(long[] shape, ByteBuffer data)
    Tensor newUInt8Tensor(long[] shape, byte[] data)

    Tensor newInt8Tensor(long[] shape, ByteBuffer data)
    Tensor newInt8Tensor(long[] shape, byte[] data)

    Tensor newFloat32Tensor(long[] shape, FloatBuffer data)
    Tensor newFloat32Tensor(long[] shape, float[] data)

    Tensor newInt32Tensor(long[] shape, IntBuffer data)
    Tensor newInt32Tensor(long[] shape, int[] data)

    Tensor newFloat64Tensor(long[] shape, DoubleBuffer data)
    Tensor newFloat64Tensor(long[] shape, double[] data)

    Tensor newInt64Tensor(long[] shape, LongBuffer data)
    Tensor newInt64Tensor(long[] shape, long[] data)

Where the first parameter ``long[] shape`` is shape of the Tensor as array of longs.

Content of the Tensor can be provided either as (a) java array  or (b) as java.nio.DirectByteBuffer of proper type with native bit order.

In case of (a) proper DirectByteBuffer will be created internally. (b) case has an advantage that user can keep the reference to DirectByteBuffer and change its content in future for the next run, avoiding allocation of DirectByteBuffer for repeated runs.

Java’s primitive type byte is signed and java does not have unsigned 8 bit type. For dtype=uint8 api uses byte that will be re-interpreted as uint8 on native side. On java side unsigned value of byte can be read as (byte & 0xFF).

Tensor content layout
^^^^^^^^^^^^^^^^^^^^^

Tensor content is represented as a one dimensional array (buffer),
where the first element has all zero indexes T\[0, ... 0\].

Lets assume tensor shape is {d\ :sub:`0`\, ... d\ :sub:`n-1`\ } and d\ :sub:`n-1`\ > 0.

The second element will be T\[0, ... 1\] and the last one T\[d\ :sub:`0`\ -1, ... d\ :sub:`n-1`\ - 1\]

Tensor has methods to check its dtype:
::
   int dtype()

That returns one of the dtype codes:
::

    Tensor.DTYPE_UINT8
    Tensor.DTYPE_INT8
    Tensor.DTYPE_INT32
    Tensor.DTYPE_FLOAT32
    Tensor.DTYPE_INT64
    Tensor.DTYPE_FLOAT64

The data of Tensor can be read as java array:
::

    byte[] getDataAsUnsignedByteArray()
    byte[] getDataAsByteArray()
    int[] getDataAsIntArray()
    long[] getDataAsLongArray()
    float[] getDataAsFloatArray()
    double[] getDataAsDoubleArray()

These methods throw IllegalStateException if called for inappropriate dtype.

Tensor API Details
------------------
.. java:package:: org.pytorch
   :noindex:

.. java:type:: public abstract class Tensor

   Representation of Tensor. Tensor shape is stored in \ :java:ref:`Tensor.shape`\ , elements are stored as \ :java:ref:`java.nio.DirectByteBuffer`\  of one of the supported types.

Fields
^^^^^^^
DTYPE_FLOAT32
~~~~~~~~~~~~~~

.. java:field:: public static final int DTYPE_FLOAT32
   :outertype: Tensor

   Code for dtype torch.float32. \ :java:ref:`Tensor.dtype()`\

DTYPE_FLOAT64
~~~~~~~~~~~~~~

.. java:field:: public static final int DTYPE_FLOAT64
   :outertype: Tensor

   Code for dtype torch.float64. \ :java:ref:`Tensor.dtype()`\

DTYPE_INT32
~~~~~~~~~~~~~~

.. java:field:: public static final int DTYPE_INT32
   :outertype: Tensor

   Code for dtype torch.int32. \ :java:ref:`Tensor.dtype()`\

DTYPE_INT64
~~~~~~~~~~~~~~

.. java:field:: public static final int DTYPE_INT64
   :outertype: Tensor

   Code for dtype torch.int64. \ :java:ref:`Tensor.dtype()`\

DTYPE_INT8
~~~~~~~~~~~~~~

.. java:field:: public static final int DTYPE_INT8
   :outertype: Tensor

   Code for dtype torch.int8. \ :java:ref:`Tensor.dtype()`\

DTYPE_UINT8
~~~~~~~~~~~~~~

.. java:field:: public static final int DTYPE_UINT8
   :outertype: Tensor

   Code for dtype torch.uint8. \ :java:ref:`Tensor.dtype()`\

shape
~~~~~~~~~~~~~~

.. java:field:: public final long[] shape
   :outertype: Tensor

   Shape of current tensor.

Methods
^^^^^^^^
allocateByteBuffer
~~~~~~~~~~~~~~

.. java:method:: public static ByteBuffer allocateByteBuffer(int numElements)
   :outertype: Tensor

   Allocates a new direct \ :java:ref:`java.nio.ByteBuffer`\  with native byte order with specified capacity that can be used in \ :java:ref:`Tensor.newInt8Tensor(long[],ByteBuffer)`\ , \ :java:ref:`Tensor.newUInt8Tensor(long[],ByteBuffer)`\ .

   :param numElements: capacity (number of elements) of result buffer.

allocateDoubleBuffer
~~~~~~~~~~~~~~

.. java:method:: public static DoubleBuffer allocateDoubleBuffer(int numElements)
   :outertype: Tensor

   Allocates a new direct \ :java:ref:`java.nio.DoubleBuffer`\  with native byte order with specified capacity that can be used in \ :java:ref:`Tensor.newFloat64Tensor(long[],DoubleBuffer)`\ .

   :param numElements: capacity (number of elements) of result buffer.

allocateFloatBuffer
~~~~~~~~~~~~~~

.. java:method:: public static FloatBuffer allocateFloatBuffer(int numElements)
   :outertype: Tensor

   Allocates a new direct \ :java:ref:`java.nio.FloatBuffer`\  with native byte order with specified capacity that can be used in \ :java:ref:`Tensor.newFloat32Tensor(long[],FloatBuffer)`\ .

   :param numElements: capacity (number of elements) of result buffer.

allocateIntBuffer
~~~~~~~~~~~~~~

.. java:method:: public static IntBuffer allocateIntBuffer(int numElements)
   :outertype: Tensor

allocateLongBuffer
~~~~~~~~~~~~~~

.. java:method:: public static LongBuffer allocateLongBuffer(int numElements)
   :outertype: Tensor

   Allocates a new direct \ :java:ref:`java.nio.LongBuffer`\  with native byte order with specified capacity that can be used in \ :java:ref:`Tensor.newInt64Tensor(long[],LongBuffer)`\ .

   :param numElements: capacity (number of elements) of result buffer.

dtype
~~~~~~~~~~~~~~

.. java:method:: public abstract int dtype()
   :outertype: Tensor

   Returns dtype of current tensor. Can be one of \ :java:ref:`Tensor.DTYPE_UINT8`\ , \ :java:ref:`Tensor.DTYPE_INT8`\ , \ :java:ref:`Tensor.DTYPE_INT32`\ ,\ :java:ref:`Tensor.DTYPE_FLOAT32`\ , \ :java:ref:`Tensor.DTYPE_INT64`\ , \ :java:ref:`Tensor.DTYPE_FLOAT64`\ .

getDataAsByteArray
~~~~~~~~~~~~~~

.. java:method:: public byte[] getDataAsByteArray()
   :outertype: Tensor

   Returns newly allocated java byte array that contains a copy of tensor data.

   :throws IllegalStateException: if it is called for a non-int8 tensor.

getDataAsDoubleArray
~~~~~~~~~~~~~~

.. java:method:: public double[] getDataAsDoubleArray()
   :outertype: Tensor

   Returns newly allocated java byte array that contains a copy of tensor data.

   :throws IllegalStateException: if it is called for a non-float64 tensor.

getDataAsFloatArray
~~~~~~~~~~~~~~

.. java:method:: public float[] getDataAsFloatArray()
   :outertype: Tensor

   Returns newly allocated java byte array that contains a copy of tensor data.

   :throws IllegalStateException: if it is called for a non-float32 tensor.

getDataAsIntArray
~~~~~~~~~~~~~~

.. java:method:: public int[] getDataAsIntArray()
   :outertype: Tensor

   Returns newly allocated java byte array that contains a copy of tensor data.

   :throws IllegalStateException: if it is called for a non-int32 tensor.

getDataAsLongArray
~~~~~~~~~~~~~~~~~~

.. java:method:: public long[] getDataAsLongArray()
   :outertype: Tensor

   Returns newly allocated java byte array that contains a copy of tensor data.

   :throws IllegalStateException: if it is called for a non-int64 tensor.

getDataAsUnsignedByteArray
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. java:method:: public byte[] getDataAsUnsignedByteArray()
   :outertype: Tensor

   Returns newly allocated java byte array that contains a copy of tensor data.

   :throws IllegalStateException: if it is called for a non-uint8 tensor.

getRawDataBuffer
~~~~~~~~~~~~~~

.. java:method::  Buffer getRawDataBuffer()
   :outertype: Tensor

newFloat32Tensor
~~~~~~~~~~~~~~

.. java:method:: public static Tensor newFloat32Tensor(long[] shape, float[] data)
   :outertype: Tensor

   Creates a new Tensor instance with dtype torch.float32 with specified shape and data as array of floats.

   :param shape: Tensor shape
   :param data: Tensor elements

newFloat32Tensor
~~~~~~~~~~~~~~

.. java:method:: public static Tensor newFloat32Tensor(long[] shape, FloatBuffer data)
   :outertype: Tensor

   Creates a new Tensor instance with dtype torch.float32 with specified shape and data.

   :param shape: Tensor shape
   :param data: Direct buffer with native byte order that contains \ ``Tensor#numel(shape)``\  elements. The buffer is used directly without copying, and changes to its content will change the tensor.

newFloat64Tensor
~~~~~~~~~~~~~~~~~

.. java:method:: public static Tensor newFloat64Tensor(long[] shape, double[] data)
   :outertype: Tensor

   Creates a new Tensor instance with dtype torch.float64 with specified shape and data as array of doubles.

   :param shape: Tensor shape
   :param data: Tensor elements

newFloat64Tensor
~~~~~~~~~~~~~~~~~

.. java:method:: public static Tensor newFloat64Tensor(long[] shape, DoubleBuffer data)
   :outertype: Tensor

   Creates a new Tensor instance with dtype torch.float64 with specified shape and data.

   :param shape: Tensor shape
   :param data: Direct buffer with native byte order that contains \ ``Tensor#numel(shape)``\  elements. The buffer is used directly without copying, and changes to its content will change the tensor.

newInt32Tensor
~~~~~~~~~~~~~~

.. java:method:: public static Tensor newInt32Tensor(long[] shape, int[] data)
   :outertype: Tensor

   Creates a new Tensor instance with dtype torch.int32 with specified shape and data as array of ints.

   :param shape: Tensor shape
   :param data: Tensor elements

newInt32Tensor
~~~~~~~~~~~~~~

.. java:method:: public static Tensor newInt32Tensor(long[] shape, IntBuffer data)
   :outertype: Tensor

   Creates a new Tensor instance with dtype torch.int32 with specified shape and data.

   :param shape: Tensor shape
   :param data: Direct buffer with native byte order that contains \ ``Tensor#numel(shape)``\  elements. The buffer is used directly without copying, and changes to its content will change the tensor.

newInt64Tensor
~~~~~~~~~~~~~~

.. java:method:: public static Tensor newInt64Tensor(long[] shape, long[] data)
   :outertype: Tensor

   Creates a new Tensor instance with dtype torch.int64 with specified shape and data as array of longs.

   :param shape: Tensor shape
   :param data: Tensor elements

newInt64Tensor
~~~~~~~~~~~~~~

.. java:method:: public static Tensor newInt64Tensor(long[] shape, LongBuffer data)
   :outertype: Tensor

   Creates a new Tensor instance with dtype torch.int64 with specified shape and data.

   :param shape: Tensor shape
   :param data: Direct buffer with native byte order that contains \ ``Tensor#numel(shape)``\  elements. The buffer is used directly without copying, and changes to its content will change the tensor.

newInt8Tensor
~~~~~~~~~~~~~~

.. java:method:: public static Tensor newInt8Tensor(long[] shape, byte[] data)
   :outertype: Tensor

   Creates a new Tensor instance with dtype torch.int8 with specified shape and data as array of bytes.

   :param shape: Tensor shape
   :param data: Tensor elements

newInt8Tensor
~~~~~~~~~~~~~~

.. java:method:: public static Tensor newInt8Tensor(long[] shape, ByteBuffer data)
   :outertype: Tensor

   Creates a new Tensor instance with dtype torch.int8 with specified shape and data.

   :param shape: Tensor shape
   :param data: Direct buffer with native byte order that contains \ ``Tensor#numel(shape)``\  elements. The buffer is used directly without copying, and changes to its content will change the tensor.

newUInt8Tensor
~~~~~~~~~~~~~~

.. java:method:: public static Tensor newUInt8Tensor(long[] shape, byte[] data)
   :outertype: Tensor

   Creates a new Tensor instance with dtype torch.uint8 with specified shape and data as array of bytes.

   :param shape: Tensor shape
   :param data: Tensor elements

newUInt8Tensor
~~~~~~~~~~~~~~

.. java:method:: public static Tensor newUInt8Tensor(long[] shape, ByteBuffer data)
   :outertype: Tensor

   Creates a new Tensor instance with dtype torch.uint8 with specified shape and data.

   :param shape: Tensor shape
   :param data: Direct buffer with native byte order that contains \ ``Tensor#numel(shape)``\  elements. The buffer is used directly without copying, and changes to its content will change the tensor.

numel
~~~~~~

.. java:method:: public long numel()
   :outertype: Tensor

   Calculates number of elements in current tensor instance.

numel
~~~~~~

.. java:method:: public static long numel(long[] shape)
   :outertype: Tensor

   Calculates number of elements in tensor with specified shape.





.. java:import:: android.graphics Bitmap

.. java:import:: android.graphics ImageFormat

.. java:import:: android.media Image

.. java:import:: org.pytorch Tensor

.. java:import:: java.nio ByteBuffer

.. java:import:: java.util Locale

TensorImageUtils
----------------

Overview
^^^^^^^^

.. java:package:: org.pytorch.torchvision
  :noindex:

.. java:type:: public final class TensorImageUtils

  Contains utility functions for \ :java:ref:`org.pytorch.Tensor`\  creation from \ :java:ref:`android.graphics.Bitmap`\  or \ :java:ref:`android.media.Image`\  source.

Fields
^^^^^^^^
TORCHVISION_NORM_MEAN_RGB
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. java:field:: public static float[] TORCHVISION_NORM_MEAN_RGB
  :outertype: TensorImageUtils

TORCHVISION_NORM_STD_RGB
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. java:field:: public static float[] TORCHVISION_NORM_STD_RGB
  :outertype: TensorImageUtils

Methods
^^^^^^^
bitmapToFloat32Tensor
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. java:method:: public static Tensor bitmapToFloat32Tensor(Bitmap bitmap, float[] normMeanRGB, float[] normStdRGB)
  :outertype: TensorImageUtils

  Creates new \ :java:ref:`org.pytorch.Tensor`\  from full \ :java:ref:`android.graphics.Bitmap`\ , normalized with specified in parameters mean and std.

  :param normMeanRGB: means for RGB channels normalization, length must equal 3, RGB order
  :param normStdRGB: standard deviation for RGB channels normalization, length must equal 3, RGB order

bitmapToFloat32Tensor
~~~~~~~~~~~~~~~~~~~~~~

.. java:method:: public static Tensor bitmapToFloat32Tensor(Bitmap bitmap, int x, int y, int width, int height, float[] normMeanRGB, float[] normStdRGB)
  :outertype: TensorImageUtils

  Creates new \ :java:ref:`org.pytorch.Tensor`\  from specified area of \ :java:ref:`android.graphics.Bitmap`\ , normalized with specified in parameters mean and std.

  :param bitmap: \ :java:ref:`android.graphics.Bitmap`\  as a source for Tensor data
  :param x: x coordinate of top left corner of bitmap's area
  :param y: y coordinate of top left corner of bitmap's area
  :param width: width of bitmap's area
  :param height: height of bitmap's area
  :param normMeanRGB: means for RGB channels normalization, length must equal 3, RGB order
  :param normStdRGB: standard deviation for RGB channels normalization, length must equal 3, RGB order

imageYUV420CenterCropToFloat32Tensor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. java:method:: public static Tensor imageYUV420CenterCropToFloat32Tensor(Image image, int rotateCWDegrees, int tensorWidth, int tensorHeight, float[] normMeanRGB, float[] normStdRGB)
  :outertype: TensorImageUtils

  Creates new \ :java:ref:`org.pytorch.Tensor`\  from specified area of \ :java:ref:`android.media.Image`\ , doing optional rotation, scaling (nearest) and center cropping.

  :param image: \ :java:ref:`android.media.Image`\  as a source for Tensor data
  :param rotateCWDegrees: Clockwise angle through which the input image needs to be rotated to be upright. Range of valid values: 0, 90, 180, 270
  :param tensorWidth: return tensor width, must be positive
  :param tensorHeight: return tensor height, must be positive
  :param normMeanRGB: means for RGB channels normalization, length must equal 3, RGB order
  :param normStdRGB: standard deviation for RGB channels normalization, length must equal 3, RGB order
