Custom Classes
==============

PyTorch allows registering custom C++ classes that can be used from Python
and TorchScript.

Header: ``torch/custom_class.h``

class\_ Template
----------------

.. doxygenclass:: torch::class_
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   #include <torch/custom_class.h>

   struct MyClass : torch::CustomClassHolder {
       int value;

       MyClass(int v) : value(v) {}

       int getValue() const { return value; }
       void setValue(int v) { value = v; }
   };

   TORCH_LIBRARY(my_classes, m) {
       m.class_<MyClass>("MyClass")
           .def(torch::init<int>())
           .def("getValue", &MyClass::getValue)
           .def("setValue", &MyClass::setValue)
           .def_readwrite("value", &MyClass::value);
   }

Registering Methods
-------------------

**Constructor:**

.. code-block:: cpp

   m.class_<MyClass>("MyClass")
       .def(torch::init<int>())  // Constructor taking int

**Methods:**

.. code-block:: cpp

   m.class_<MyClass>("MyClass")
       .def("getValue", &MyClass::getValue)
       .def("setValue", &MyClass::setValue)

**Properties:**

.. code-block:: cpp

   m.class_<MyClass>("MyClass")
       .def_readwrite("value", &MyClass::value)   // Read-write
       .def_readonly("const_value", &MyClass::const_value)  // Read-only

Using Custom Classes
--------------------

**From C++:**

.. code-block:: cpp

   auto my_obj = c10::make_intrusive<MyClass>(42);
   int val = my_obj->getValue();

**From Python:**

.. code-block:: python

   import torch
   torch.classes.load_library("path/to/library.so")
   obj = torch.classes.my_classes.MyClass(42)
   print(obj.getValue())

**In TorchScript:**

.. code-block:: python

   @torch.jit.script
   def use_my_class(x: torch.classes.my_classes.MyClass) -> int:
       return x.getValue()
