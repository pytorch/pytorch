.. contents::
    :local:
    :depth: 2


.. testsetup::

    # These are hidden from the docs, but these are necessary for `doctest`
    # since the `inspect` module doesn't play nicely with the execution
    # environment for `doctest`
    import torch

    original_script = torch.jit.script
    def script_wrapper(obj, *args, **kwargs):
        obj.__module__ = 'FakeMod'
        return original_script(obj, *args, **kwargs)
    torch.jit.script = script_wrapper

    original_trace = torch.jit.trace
    def trace_wrapper(obj, *args, **kwargs):
        obj.__module__ = 'FakeMod'
        return original_trace(obj, *args, **kwargs)
    torch.jit.trace = trace_wrapper


.. _type_system:


Type System (I): TorchScript Types
==================================

This section summarizes TorchScript type system.

Terminology
^^^^^^^^^^^

This document uses the following terminologies:

::

    " " represents real keywords and delimiters that are part of the syntax
    A | B indicates either A or B
    ( ) indicates grouping
    [ ] indicates optional
    A+ indicates a regular expression where term A is repeated at least once
    A* indicates a regular expression where term A is repeated zero or more times

TorchScript Types
^^^^^^^^^^^^^^^^^

The TorchScript type system consists of ``TSType`` and ``TSModuleType`` as defined below.

::

    TSAllType ::= TSType | TSModuleType
    TSType ::= TSMetaType | TSPrimitiveType | TSStructuralType | TSNominalType

``TSType`` represents the majority of TorchScript types that are composable and can be used in TorchScript type annotation.
``TSType`` can be further classified into:

* meta types, e.g., ``Any``
* primitive types, e.g., ``int``, ``float``, ``str``
* structural types, e.g., ``Optional[int]`` or ``List[MyClass]``
* nominal types (Python classes), e.g., ``MyClass`` (user-defined), ``torch.tensor`` (builtin)

``TSModuleType`` represents ``torch.nn.Module`` and its subclasses. It is treated differently from ``TSType`` because its type schema is inferred partly from the object instance and partly from the class definition.
As such, instances of a ``TSModuleType`` may not follow the same static type schema. ``TSModuleType`` cannot be used as a TorchScript type annotation or be composed with ``TSType`` for type safety considerations.

Meta Types
^^^^^^^^^^

Meta types are so abstract that they are more like type constraints than concrete types.
Currently TorchScript defines one meta-type, ``Any``, that represents any TorchScript type.

``Any`` Type
""""""""""""

The ``Any`` type literally represents any type. ``Any`` specifies no type constraints, thus there is no type checking on ``Any``.
As such it can be bound to any Python or TorchScript data types (e.g., int, TorchScript ``tuple``, or an arbitrary Python class that is not scripted).

::

    TSMetaType ::= "Any"

where

    * ``Any`` is the Python class name from the typing module, therefore usage of the ``Any`` type requires from ``typing import Any``
    * Since ``Any`` can represent any type, the set of operators allowed to operate on values of this type on Any is limited.

Operators supported for ``Any`` type
""""""""""""""""""""""""""""""""""""

* assignment to data of ``Any`` type
* binding to parameter or return of ``Any`` type
* ``x is``, ``x is not`` where ``x`` is of ``Any`` type
* ``isinstance(x, Type)`` where ``x`` is of ``Any`` type
* data of ``Any`` type is printable
* Data of ``List[Any]`` type may be sortable if the data is a list of values of the same type ``T`` and that ``T`` supports comparison operators

Compared to Python
""""""""""""""""""

``Any`` is the least constrained type in the TorchScript type system. In that sense, it is quite similar to
Object class in Python. However, ``Any`` only supports a subset of the operators and methods that are supported by Object.

Design notes
""""""""""""

When we script a PyTorch module, we may encounter data that are not involved in the execution of the script, nevertheless has to be described
by a type schema. It is not only cumbersome to describe static types for unused data (in the context of the script) but also may lead to unnecessary
scripting failures. ``Any`` is introduced to describe the type of the data where precise static types are not necessary for compilation.

Example:

This example illustrates how ``Any`` can be used to allow the second element of the tuple parameter to be of ``any`` type. This is possible,
because ``x[1]`` is not involved in any computation that requires knowing its precise type.

.. testcode::

    import torch

    from typing import Tuple
    from typing import Any

    @torch.jit.export
    def incFirstElement(x: Tuple[int, Any]):
    return (x[0]+1, x[1])

    m = torch.jit.script(incFirstElement)
    print(m((1,2.0)))
    print(m((1,(100,200))))

The example will generate the following output, where the second element of the tuple is of ``Any`` type
thus can bind to multiple types, e.g., (1, 2.0) binds a float type to ``Any`` as in ``Tuple[int, Any]``,
whereas ``(1, (100, 200))`` binds a tuple to ``Any`` in the second invocation.

.. testoutput::

    (2, 2.0)
    (2, (100, 200))

Example:

We can use ``isinstance`` to dynamically check the type of the data annotated as ``Any`` type.

.. testcode::

    import torch
    from typing import Any

    def f(a:Any):
        print(a)
        return (isinstance(a, torch.Tensor))

    ones = torch.ones([2])
    m = torch.jit.script(f)
    print(m(ones))

The above example produces the following output

.. testoutput::

    1
    1
    [ CPUFloatType{2} ]
    True

Primitive Types
^^^^^^^^^^^^^^^

Primitive TorchScript types represent types that represent a single type of value and go with a single pre-defined
type name.

::

    TSPrimitiveType ::= "int" | "float" | "double" | "complex" | "bool" | "str" | "None"

Structural Types
^^^^^^^^^^^^^^^^

Structural types are types that are structurally defined without a user-defined name (unlike nominal types),
such as ``Future[int]``. Structural types are composable with any ``TSType``.

::

    TSStructuralType ::=  TSTuple | TSNamedTuple | TSList | TSDict |
                        TSOptional | TSFuture | TSRRef

    TSTuple ::= "Tuple" "[" (TSType ",")* TSType "]"
    TSNamedTuple ::= "namedtuple" "(" (TSType ",")* TSType ")"
    TSList ::= "List" "[" TSType "]"
    TSOptional ::= "Optional" "[" TSType "]"
    TSFuture ::= "Future" "[" TSType "]"
    TSRRef ::= "RRef" "[" TSType "]"
    TSDict ::= "Dict" "[" KeyType "," TSType "]"
    KeyType ::= "str" | "int" | "float" | "bool" | TensorType | "Any"

where

* ``Tuple``, ``List``, ``Optional``, ``Union``, ``Future``, ``Dict`` represent Python type class names defined in module ``typing``. Therefore before using these type names, one must import them from ``typing`` (e.g., ``from typing import Tuple)``.
* ``namedtuple`` represents Python class  ``collections.namedtuple`` or ``typing.NamedTuple`` .
* ``Future`` and ``RRef`` represent Python classes  ``torch.futures``, ``torch.distributed.rpc``.

Compared to Python
""""""""""""""""""

* Apart from being composable with TorchScript types, these TorchScript structural types often support a common subset of
the operators and methods of their Python counterparts.

Example:

This example uses ``typing.NamedTuple`` syntax:

.. testcode::

    import torch
    from typing import NamedTuple
    from typing import Tuple

    class MyTuple(NamedTuple):
        first: int
        second: int

    def inc(x: MyTuple) -> Tuple[int, int]:
        return (x.first+1, x.second+1)

    t = MyTuple(first=1, second=2)
    scripted_inc = torch.jit.script(inc)
    print("TorchScript:", scripted_inc(t))

This example uses ``collections.namedtuple`` syntax:

.. testcode::

    import torch
    from typing import NamedTuple
    from typing import Tuple
    from collections import namedtuple

    _AnnotatedNamedTuple = NamedTuple('_NamedTupleAnnotated', [('first', int), ('second', int)])
    _UnannotatedNamedTuple = namedtuple('_NamedTupleAnnotated', ['first', 'second'])

    def inc(x: _AnnotatedNamedTuple) -> Tuple[int, int]:
        return (x.first+1, x.second+1)

    m = torch.jit.script(inc)
    print(inc(_UnannotatedNamedTuple(1,2)))

Example:

This example illustrates a common mistake of annotating structural types, i.e., not importing the composite type
classes from the ``typing`` module.

.. testcode::

    import torch

    # ERROR: Tuple not recognized because not imported from typing
    @torch.jit.export
    def inc(x: Tuple[int, int]):
    return (x[0]+1, x[1]+1)

    m = torch.jit.script(inc)
    print(inc((1,2)))

Running the above codes yields the following scripting error. The remedy is to add from ``typing import Tuple``.

.. testoutput::

    File "test-tuple.py", line 5, in <module>
        def inc(x: Tuple[int, int]):
    NameError: name 'Tuple' is not defined

Nominal Types
^^^^^^^^^^^^^

Nominal TorchScript types are Python classes. They are called nominal because these types are declared with a custom
name and are compared using class names. Nominal classes are further classified into the following categories:

::

    TSNominalType ::= TSBuiltinClasses | TSCustomClass | TSEnum

Among them, ``TSCustomClass`` and ``TSEnum`` must be compilable to TorchScript IR (as enforced by the type-checker).

Builtin Class
^^^^^^^^^^^^^

Builtin nominal types are Python classes whose semantics are built into the TorchScript system, such as tensor types.
TorchScript defines the semantics of these builtin nominal types, and often supports only a subset of the methods or
attributes of its Python class definition.

::

    TSBuiltinClass ::= TSTensor | "torch.device" | "torch.Stream" | "torch.dtype" |
                    "torch.nn.ModuleList" | "torch.nn.ModuleDict" | ...
    TSTensor ::= "torch.Tensor" | "common.SubTensor" | "common.SubWithTorchFunction" |
                "torch.nn.parameter.Parameter" | and subclasses of torch.Tensor


Special note on torch.nn.ModuleList and torch.nn.ModuleDict
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Although ``torch.nn.ModuleList`` and ``torch.nn.ModuleDict`` are defined as a list and dictionary in Python,
they behave more like tuples in TorchScript.

* In TorchScript, instances of ``torch.nn.ModuleList``  or ``torch.nn.ModuleDict`` are immutable.
* Code that iterates over ``torch.nn.ModuleList`` or ``torch.nn.ModuleDict`` is completely unrolled so that elements of ``torch.nn.ModuleList`` or keys of ``torch.nn.ModuleDict`` can be of different subclasses of ``torch.nn.Module``.

Example:

.. testcode::

    import torch

    @torch.jit.script
    class A():
        def __init__(self):
            self.x = torch.rand(3)

        def f(self, y: torch.device):
            return self.x.to(device=y)

    def g():
        a = A()
        return a.f(torch.device("cpu"))

    script_g = torch.jit.script(g)
    print(script_g.graph)

Custom Class
^^^^^^^^^^^^

Unlike built-in classes, semantics of custom classes are user-defined and the entire class definition must be compilable to TorchScript IR and subject to TorchScript type-checking rules.

::

    TSClassDef ::= [ "@torch.jit.script" ]
                "class" ClassName [ "(object)" ]  ":"
                        MethodDefinition |
                    [ "@torch.jit.ignore" ] | [ "@torch.jit.unused" ]
                        MethodDefinition

where

* Classes must be new-style classes (note that Python 3 supports only new-style classes, for Python 2.x new-style class is specified by subclassing from object)
* Instance data attributes are statically typed, and instance attributes must be declared by assignments inside the ``__init__()`` method
* Method overloading is not supported (i.e., cannot have multiple methods with the same method name)
* MethodDefinition must be compilable to TorchScript IR and adhere to TorchScript’s type-checking rules, (e.g., all methods must be valid TorchScript functions and class attribute definitions must be valid TorchScript statements)
* ``torch.jit.ignore``<placeholder for link to ``torch.jit.ignore``> and ``torch.jit.unused``<placeholder for link to ``torch.jit.unused``> can be used to ignore the method or function that is not fully torchscriptable or should be ignored by the compiler

Compared to Python
""""""""""""""""""

TorchScript custom classes are quite limited compared to their Python counterpart.

* do not support class attributes
* do not support subclassing except for subclassing an interface type or object
* do not support method overloading
* must initialize all its instance attributes in  ``__init__()``; this is because TorchScript constructs a static schema of the class by inferring attribute types in ``__init__()``
* must contain only methods that satisfy TorchScript type-checking rules and are compilable to TorchScript IRs

Example:

Python classes can be used in TorchScript if they are annotated with ``@torch.jit.script``, similar to how a TorchScript function would be declared:
.. testcode::

    @torch.jit.script
    class MyClass:
        def __init__(self, x: int):
            self.x = x

        def inc(self, val: int):
            self.x += val


Example:

A TorchScript custom class type must "declare" all its instance attributes by assignments in ``__init__()``. If an instance attribute is not defined in ``__init__()`` but accessed in other methods of the class, the class cannot be compiled as a TorchScript class, as shown in the following example:

.. testcode::

    @torch.jit.script
    class foo:
        def __init__(self):
            self.y = 1

        # ERROR: self.x is not defined in __init__
        def assign_x(self):
            self.x = torch.rand(2, 3)

 The above class will fail to be compiled and issue the following error:

.. testoutput::

    RuntimeError:
    Tried to set nonexistent attribute: x. Did you forget to initialize it in __init__()?:
    def assign_x(self):
        self.x = torch.rand(2, 3)
        ~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE

Example:

In this example, a TorchScript custom class defines a class variable name, which is not allowed.

.. testcode::

    import torch

    @torch.jit.script
    class MyClass(object):
        name = "MyClass"
        def __init__(self, x: int):
            self.x = x

    def fn(a: MyClass):
        return a.name

It leads to the following compile-time error:

.. testoutput::

    RuntimeError:
    Tried to access nonexistent attribute or method 'name' of type '__torch__.MyClass'. Did you forget to initialize an attribute in __init__()?:
        File "test-class2.py", line 10
    def fn(a: MyClass):
        return a.name
            ~~~~~~ <--- HERE

Enum Type
^^^^^^^^^^

Like custom classes, semantics of enum type are user-defined and the entire class definition must be compilable to TorchScript IR and adhere to TorchScript type-checking rules.

::

    TSEnumDef ::= "class" Identifier "(enum.Enum | TSEnumType)" ":"
                ( MemberIdentifier "=" Value )+
                ( MethodDefinition )*

where

* Value must be a TorchScript literal of type ``int``, ``float``, or ``str``, and must be of the same TorchScript type
* ``TSEnumType`` is the name of a TorchScript enumerated type. Similar to Python enum, TorchScript allows restricted ``Enum`` subclassing, that is, subclassing an enumerated is allowed only if it does not define any members.

Compared to Python
""""""""""""""""""

* TorchScript supports only ``enum.Enum``, but not other variations such as ``enum.IntEnum``, ``enum.Flag``, ``enum.IntFlag``, or  ``enum.auto``
* Values of TorchScript enum members must be of the same type and can only be of ``int``, ``float``, or ``str`` type, whereas Python enum members can be of any type
* Enums containing methods are ignored in TorchScript.

Example:

.. testcode::

    import torch
    from enum import Enum

    class Color(Enum):
        RED = 1
        GREEN = 2

    def enum_fn(x: Color, y: Color) -> bool:
        if x == Color.RED:
            return True
        return x == y

    m = torch.jit.script(enum_fn)

    print("Eager: ", enum_fn(Color.RED, Color.GREEN))
    print("TorchScript: ", m(Color.RED, Color.GREEN))

Example:

The following example shows the case of restricted enum subclassing, where ``BaseColor`` does not define any member, thus can be subclassed by ``Color``.

.. testcode::

    import torch
    from enum import Enum

    class BaseColor(Enum):
        def foo(self):
            pass

    class Color(BaseColor):
        RED = 1
        GREEN = 2

    def enum_fn(x: Color, y: Color) -> bool:
        if x == Color.RED:
            return True

        return x == y

    m = torch.jit.script(enum_fn)
    print("TorchScript: ", m(Color.RED, Color.GREEN))
    print("Eager: ", enum_fn(Color.RED, Color.GREEN))

TorchScript Module Class
^^^^^^^^^^^^^^^^^^^^^^^^

``TSModuleType`` is a special class type that is inferred from object instances created outside TorchScript. ``TSModuleType`` is named by the Python class of the object instance. The ``__init__()`` method of the Python class is not considered as a TorchScript method, so it does not have to comply with TorchScript’s type checking rules.

Since the type schema of module instance class is constructed directly from an instance object (created outside the scope of TorchScript), rather than inferred from ``__init__()`` like custom classes. It is possible that two objects of the same instance class type follow two different type schemas.

In this sense, ``TSModuleType`` is not really a static type. Therefore, for type safety considerations, ``TSModuleType`` cannot be used in a TorchScript type annotation or be composed with ``TSType``.

Module Instance Class
^^^^^^^^^^^^^^^^^^^^^

TorchScript module type represents type schema of a user-defined PyTorch module instance.  When scripting a PyTorch module, the module object is always created outside TorchScript (i.e., passed in as parameter to ``forward``), The Python module class is treated as a module instance class, so the ``__init__()`` method of the Python module class is not subject to the type checking rules of TorchScript.

::

    TSModuleType ::= "class" Identifier "(torch.nn.Module)" ":"
                        ClassBodyDefinition

where

* ``forward()`` and other methods decorated with ``@torch.jit.export`` must be compilable to TorchScript IR and subject to TorchScript’s type checking rules

Unlike custom classes, only the forward method and other methods decorated with ``@torch.jit.export``  of the module type need to be compilable. Most notably, ``__init__()`` is not considered a TorchScript method. Consequently, module type constructors cannot be invoked within the scope of TorchScript. Instead, TorchScript module objects are always constructed outside and passed into ``torch.jit.script(ModuleObj)``.

Example:

.. testcode::

    import torch

class TestModule(torch.nn.Module):
    def __init__(self, v):
        super().__init__()
        self.x = v

    def forward(self, inc: int):
        return self.x + inc

m = torch.jit.script(TestModule(1))
print(f"First instance: {m(3)}")

m = torch.jit.script(TestModule(torch.ones([5])))
print(f"Second instance: {m(3)}")

.. testoutput::

    First instance: 4
    Second instance: tensor([4., 4., 4., 4., 4.])

This example illustrates a few features of module types:

*  The ``TestModule`` instance is created outside the scope of TorchScript (i.e., before invoking ``torch.jit.script``).
* ``__init__()`` is not considered to be a TorchScript method, therefore it does not have to be annotated and can contain arbitrary Python code. In addition, the ``__init__()`` method of an instance class cannot be invoked in TorchScript code.* Because ``TestModule`` instances are instantiated in Python, in this example, ``TestModule(2.0)`` and ``TestModule(2)`` create two instances with different types for its data attributes. ``self.x is of type ``float`` for ``TestModule(2.0)``, whereas ``self.y`` is of type ``int`` for ``TestModule(2.0)``.
* TorchScript automatically compiles other methods (e.g., ``mul()``) invoked by methods annotated via ``@torch.jit.export`` or ``forward()`` methods
* Entry-points to a TorchScript program are either ``forward()`` of a module type or functions annotated as ``torch.jit.script`` or methods annotated as ``torch.jit.export``

Example:

The following shows an incorrect usage of module type. Specifically, this example invokes the constructor of ``TestModule`` inside the scope of TorchScript.

.. testcode::

    import torch

    class TestModule(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.x = v

        def forward(self, x: int):
            return self.x + x

    class MyModel:
        def __init__(self, v: int):
            self.val = v

        @torch.jit.export
        def doSomething(self, val: int) -> int:
            # error: should not invoke the constructor of module type
            myModel = TestModule(self.val)
            return myModel(val)

    m = torch.jit.script(MyModel(2))
    print(m.doSomething(3))

.. testoutput::

    RuntimeError: Could not get name of python class object

Appendix: TorchScript Type System Definition
============================================

::

    TSAllType ::= TSType | TSModuleType
    TSType ::= TSMetaType | TSPrimitiveType | TSStructuralType | TSNominalType

    TSMetaType ::= "Any"
    TSPrimitiveType ::= "int" | "float" | "double" | "complex" | "bool" | "str" | "None"

    TSStructualType ::=  TSTuple | TSNamedTuple | TSList | TSDict |
                        TSOptional | TSFuture | TSRRef
    TSTuple ::= "Tuple" "[" (TSType ",")* TSType "]"
    TSNamedTuple ::= "namedtuple" "(" (TSType ",")* TSType ")"
    TSList ::= "List" "[" TSType "]"
    TSOptional ::= "Optional" "[" TSType "]"
    TSUnion ::= "Union" "[" (TSType ",")* TSType "]"
    TSFuture ::= "Future" "[" TSType "]"
    TSRRef ::= "RRef" "[" TSType "]"
    TSDict ::= "Dict" "[" KeyType "," TSType "]"
    KeyType ::= "str" | "int" | "float" | "bool" | TensorType | "Any"

    TSNominalType ::= TSBuiltinClasses | TSCustomClass | TSEnum
    TSBuiltinClass ::= TSTensor | "torch.device" | "torch.stream"|
                    "torch.dtype" | "torch.nn.ModuleList" |
                    "torch.nn.ModuleDict" | ...
    TSTensor ::= "torch.tensor" and subclasses
