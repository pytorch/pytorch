.. java:import:: java.util Locale

.. java:import:: java.util Map

org.pytorch.IValue (IValue)
===========================

Source Code
------------

Full code can be found in `Github <https://github.com/pytorch/pytorch/blob/master/android/pytorch_android/src/main/java/org/pytorch/IValue.java>`_.

Overview
--------

IValue represents a TorchScript variable that can be one of the supported
(by torchscript) `types <https://pytorch.org/docs/stable/jit.html#types>`_.
IValue is a tagged union. For every supported type it has a factory method, method
to check the type and a getter method to retrieve a value. Getters throw
IllegalStateException if called for inappropriate type.


IVaue API Details
------------------

.. java:package:: org.pytorch
   :noindex:

.. java:type:: public class IValue

   Java representation of a torchscript variable, which is implemented as tagged union that can be one of the supported types: https://pytorch.org/docs/stable/jit.html#types.

   Calling getters for inappropriate types will throw IllegalStateException.

Methods
^^^^^^^
bool
~~~~~~~~~~~~~~

.. java:method:: public static IValue bool(boolean value)
   :outertype: IValue

   Creates a new IValue instance of torchscript bool type.

boolList
~~~~~~~~~~~~~~

.. java:method:: public static IValue boolList(boolean... list)
   :outertype: IValue

   Creates a new IValue instance of torchscript List[bool] type.

dictLongKey
~~~~~~~~~~~~~~

.. java:method:: public static IValue dictLongKey(Map<Long, IValue> map)
   :outertype: IValue

   Creates a new IValue instance of torchscript Dict[int, V] type.

dictStringKey
~~~~~~~~~~~~~~

.. java:method:: public static IValue dictStringKey(Map<String, IValue> map)
   :outertype: IValue

   Creates a new IValue instance oftorchscript Dict[Str, V] type.

double64
~~~~~~~~~~~~~~

.. java:method:: public static IValue double64(double value)
   :outertype: IValue

   Creates a new IValue instance of torchscript float type.

doubleList
~~~~~~~~~~~~~~

.. java:method:: public static IValue doubleList(double... list)
   :outertype: IValue

   Creates a new IValue instance of torchscript List[float] type.

getBool
~~~~~~~~~~~~~~

.. java:method:: public boolean getBool()
   :outertype: IValue

getBoolList
~~~~~~~~~~~~~~

.. java:method:: public boolean[] getBoolList()
   :outertype: IValue

getDictLongKey
~~~~~~~~~~~~~~

.. java:method:: public Map<Long, IValue> getDictLongKey()
   :outertype: IValue

getDictStringKey
~~~~~~~~~~~~~~

.. java:method:: public Map<String, IValue> getDictStringKey()
   :outertype: IValue

getDouble
~~~~~~~~~~~~~~

.. java:method:: public double getDouble()
   :outertype: IValue

getDoubleList
~~~~~~~~~~~~~~

.. java:method:: public double[] getDoubleList()
   :outertype: IValue

getList
~~~~~~~~~~~~~~

.. java:method:: public IValue[] getList()
   :outertype: IValue

getLong
~~~~~~~~~~~~~~

.. java:method:: public long getLong()
   :outertype: IValue

getLongList
~~~~~~~~~~~~~~

.. java:method:: public long[] getLongList()
   :outertype: IValue

getString
~~~~~~~~~~~~~~

.. java:method:: public String getString()
   :outertype: IValue

getTensor
~~~~~~~~~~~~~~

.. java:method:: public Tensor getTensor()
   :outertype: IValue

getTensorList
~~~~~~~~~~~~~~

.. java:method:: public Tensor[] getTensorList()
   :outertype: IValue

getTuple
~~~~~~~~~~~~~~

.. java:method:: public IValue[] getTuple()
   :outertype: IValue

isBool
~~~~~~~~~~~~~~

.. java:method:: public boolean isBool()
   :outertype: IValue

isBoolList
~~~~~~~~~~~~~~

.. java:method:: public boolean isBoolList()
   :outertype: IValue

isDictLongKey
~~~~~~~~~~~~~~

.. java:method:: public boolean isDictLongKey()
   :outertype: IValue

isDictStringKey
~~~~~~~~~~~~~~

.. java:method:: public boolean isDictStringKey()
   :outertype: IValue

isDouble
~~~~~~~~~~~~~~

.. java:method:: public boolean isDouble()
   :outertype: IValue

isDoubleList
~~~~~~~~~~~~~~

.. java:method:: public boolean isDoubleList()
   :outertype: IValue

isList
~~~~~~~~~~~~~~

.. java:method:: public boolean isList()
   :outertype: IValue

isLong
~~~~~~~~~~~~~~

.. java:method:: public boolean isLong()
   :outertype: IValue

isLongList
~~~~~~~~~~~~~~

.. java:method:: public boolean isLongList()
   :outertype: IValue

isNull
~~~~~~~~~~~~~~

.. java:method:: public boolean isNull()
   :outertype: IValue

isString
~~~~~~~~~~~~~~

.. java:method:: public boolean isString()
   :outertype: IValue

isTensor
~~~~~~~~~~~~~~

.. java:method:: public boolean isTensor()
   :outertype: IValue

isTensorList
~~~~~~~~~~~~~~

.. java:method:: public boolean isTensorList()
   :outertype: IValue

isTuple
~~~~~~~~~~~~~~

.. java:method:: public boolean isTuple()
   :outertype: IValue

list
~~~~~~~~~~~~~~

.. java:method:: public static IValue list(IValue... array)
   :outertype: IValue

   Creates a new IValue instance of torchscript List[T] type. All elements must have the same type.

long64
~~~~~~~~~~~~~~

.. java:method:: public static IValue long64(long value)
   :outertype: IValue

   Creates a new IValue instance of torchscript int type.

longList
~~~~~~~~~~~~~~

.. java:method:: public static IValue longList(long... list)
   :outertype: IValue

   Creates a new IValue instance of torchscript List[int] type.

optionalNull
~~~~~~~~~~~~~~

.. java:method:: public static IValue optionalNull()
   :outertype: IValue

string
~~~~~~~~~~~~~~

.. java:method:: public static IValue string(String value)
   :outertype: IValue

   Creates new IValue instance of torchscript str type.

tensor
~~~~~~~~~~~~~~

.. java:method:: public static IValue tensor(Tensor tensor)
   :outertype: IValue

   Creates a new IValue instance of torchscript Tensor type.

tensorList
~~~~~~~~~~~~~~

.. java:method:: public static IValue tensorList(Tensor... list)
   :outertype: IValue

   Creates a new IValue instance of torchscript List[Tensor] type.

tuple
~~~~~~~~~~~~~~

.. java:method:: public static IValue tuple(IValue... array)
   :outertype: IValue

   Creates a new IValue instance of torchscript Tuple[T0, T1, ...] type.
