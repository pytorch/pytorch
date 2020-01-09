.. java:import:: java.util Locale

.. java:import:: java.util Map

IValue
======

.. java:package:: org.pytorch
   :noindex:

.. java:type:: public class IValue

   Java representation of a TorchScript value, which is implemented as tagged union that can be one of the supported types: https://pytorch.org/docs/stable/jit.html#types .

   Calling \ ``toX``\  methods for inappropriate types will throw \ :java:ref:`IllegalStateException`\ .

   \ ``IValue``\  objects are constructed with \ ``IValue.from(value)``\ , \ ``IValue.tupleFrom(value1, value2, ...)``\ , \ ``IValue.listFrom(value1, value2, ...)``\ , or one of the \ ``dict``\  methods, depending on the key type.

   Data is retrieved from \ ``IValue``\  objects with the \ ``toX()``\  methods. Note that \ ``str``\ -type IValues must be extracted with \ :java:ref:`toStr()`\ , rather than \ :java:ref:`toString()`\ .

   \ ``IValue``\  objects may retain references to objects passed into their constructors, and may return references to their internal state from \ ``toX()``\ .

Methods
-------
dictLongKeyFrom
^^^^^^^^^^^^^^^

.. java:method:: public static IValue dictLongKeyFrom(Map<Long, IValue> map)
   :outertype: IValue

   Creates a new \ ``IValue``\  of type \ ``Dict[int, V]``\ .

dictStringKeyFrom
^^^^^^^^^^^^^^^^^

.. java:method:: public static IValue dictStringKeyFrom(Map<String, IValue> map)
   :outertype: IValue

   Creates a new \ ``IValue``\  of type \ ``Dict[str, V]``\ .

from
^^^^

.. java:method:: public static IValue from(Tensor tensor)
   :outertype: IValue

   Creates a new \ ``IValue``\  of type \ ``Tensor``\ .

from
^^^^

.. java:method:: public static IValue from(boolean value)
   :outertype: IValue

   Creates a new \ ``IValue``\  of type \ ``bool``\ .

from
^^^^

.. java:method:: public static IValue from(long value)
   :outertype: IValue

   Creates a new \ ``IValue``\  of type \ ``int``\ .

from
^^^^

.. java:method:: public static IValue from(double value)
   :outertype: IValue

   Creates a new \ ``IValue``\  of type \ ``float``\ .

from
^^^^

.. java:method:: public static IValue from(String value)
   :outertype: IValue

   Creates a new \ ``IValue``\  of type \ ``str``\ .

isBool
^^^^^^

.. java:method:: public boolean isBool()
   :outertype: IValue

isBoolList
^^^^^^^^^^

.. java:method:: public boolean isBoolList()
   :outertype: IValue

isDictLongKey
^^^^^^^^^^^^^

.. java:method:: public boolean isDictLongKey()
   :outertype: IValue

isDictStringKey
^^^^^^^^^^^^^^^

.. java:method:: public boolean isDictStringKey()
   :outertype: IValue

isDouble
^^^^^^^^

.. java:method:: public boolean isDouble()
   :outertype: IValue

isDoubleList
^^^^^^^^^^^^

.. java:method:: public boolean isDoubleList()
   :outertype: IValue

isList
^^^^^^

.. java:method:: public boolean isList()
   :outertype: IValue

isLong
^^^^^^

.. java:method:: public boolean isLong()
   :outertype: IValue

isLongList
^^^^^^^^^^

.. java:method:: public boolean isLongList()
   :outertype: IValue

isNull
^^^^^^

.. java:method:: public boolean isNull()
   :outertype: IValue

isString
^^^^^^^^

.. java:method:: public boolean isString()
   :outertype: IValue

isTensor
^^^^^^^^

.. java:method:: public boolean isTensor()
   :outertype: IValue

isTensorList
^^^^^^^^^^^^

.. java:method:: public boolean isTensorList()
   :outertype: IValue

isTuple
^^^^^^^

.. java:method:: public boolean isTuple()
   :outertype: IValue

listFrom
^^^^^^^^

.. java:method:: public static IValue listFrom(boolean... list)
   :outertype: IValue

   Creates a new \ ``IValue``\  of type \ ``List[bool]``\ .

listFrom
^^^^^^^^

.. java:method:: public static IValue listFrom(long... list)
   :outertype: IValue

   Creates a new \ ``IValue``\  of type \ ``List[int]``\ .

listFrom
^^^^^^^^

.. java:method:: public static IValue listFrom(double... list)
   :outertype: IValue

   Creates a new \ ``IValue``\  of type \ ``List[float]``\ .

listFrom
^^^^^^^^

.. java:method:: public static IValue listFrom(Tensor... list)
   :outertype: IValue

   Creates a new \ ``IValue``\  of type \ ``List[Tensor]``\ .

listFrom
^^^^^^^^

.. java:method:: public static IValue listFrom(IValue... array)
   :outertype: IValue

   Creates a new \ ``IValue``\  of type \ ``List[T]``\ . All elements must have the same type.

optionalNull
^^^^^^^^^^^^

.. java:method:: public static IValue optionalNull()
   :outertype: IValue

   Creates a new \ ``IValue``\  of type \ ``Optional``\  that contains no value.

toBool
^^^^^^

.. java:method:: public boolean toBool()
   :outertype: IValue

toBoolList
^^^^^^^^^^

.. java:method:: public boolean[] toBoolList()
   :outertype: IValue

toDictLongKey
^^^^^^^^^^^^^

.. java:method:: public Map<Long, IValue> toDictLongKey()
   :outertype: IValue

toDictStringKey
^^^^^^^^^^^^^^^

.. java:method:: public Map<String, IValue> toDictStringKey()
   :outertype: IValue

toDouble
^^^^^^^^

.. java:method:: public double toDouble()
   :outertype: IValue

toDoubleList
^^^^^^^^^^^^

.. java:method:: public double[] toDoubleList()
   :outertype: IValue

toList
^^^^^^

.. java:method:: public IValue[] toList()
   :outertype: IValue

toLong
^^^^^^

.. java:method:: public long toLong()
   :outertype: IValue

toLongList
^^^^^^^^^^

.. java:method:: public long[] toLongList()
   :outertype: IValue

toStr
^^^^^

.. java:method:: public String toStr()
   :outertype: IValue

toTensor
^^^^^^^^

.. java:method:: public Tensor toTensor()
   :outertype: IValue

toTensorList
^^^^^^^^^^^^

.. java:method:: public Tensor[] toTensorList()
   :outertype: IValue

toTuple
^^^^^^^

.. java:method:: public IValue[] toTuple()
   :outertype: IValue

tupleFrom
^^^^^^^^^

.. java:method:: public static IValue tupleFrom(IValue... array)
   :outertype: IValue

   Creates a new \ ``IValue``\  of type \ ``Tuple[T0, T1, ...]``\ .
