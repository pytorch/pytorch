.. java:import:: java.nio Buffer

.. java:import:: java.nio ByteBuffer

.. java:import:: java.nio ByteOrder

.. java:import:: java.nio DoubleBuffer

.. java:import:: java.nio FloatBuffer

.. java:import:: java.nio IntBuffer

.. java:import:: java.nio LongBuffer

.. java:import:: java.util Arrays

.. java:import:: java.util Locale

Tensor.Tensor_float32
=====================

.. java:package:: org.pytorch
   :noindex:

.. java:type:: static class Tensor_float32 extends Tensor
   :outertype: Tensor

Constructors
------------
Tensor_float32
^^^^^^^^^^^^^^

.. java:constructor::  Tensor_float32(FloatBuffer data, long[] shape)
   :outertype: Tensor.Tensor_float32

Methods
-------
dtype
^^^^^

.. java:method:: @Override public DType dtype()
   :outertype: Tensor.Tensor_float32

getDataAsFloatArray
^^^^^^^^^^^^^^^^^^^

.. java:method:: @Override public float[] getDataAsFloatArray()
   :outertype: Tensor.Tensor_float32

getRawDataBuffer
^^^^^^^^^^^^^^^^

.. java:method:: @Override  Buffer getRawDataBuffer()
   :outertype: Tensor.Tensor_float32

toString
^^^^^^^^

.. java:method:: @Override public String toString()
   :outertype: Tensor.Tensor_float32
