Type conversions
################

Apart from enabling cross-language function calls, a fundamental problem
that a binding tool like pybind11 must address is to provide access to
native Python types in C++ and vice versa. There are three fundamentally
different ways to do this—which approach is preferable for a particular type
depends on the situation at hand.

1. Use a native C++ type everywhere. In this case, the type must be wrapped
   using pybind11-generated bindings so that Python can interact with it.

2. Use a native Python type everywhere. It will need to be wrapped so that
   C++ functions can interact with it.

3. Use a native C++ type on the C++ side and a native Python type on the
   Python side. pybind11 refers to this as a *type conversion*.

   Type conversions are the most "natural" option in the sense that native
   (non-wrapped) types are used everywhere. The main downside is that a copy
   of the data must be made on every Python ↔ C++ transition: this is
   needed since the C++ and Python versions of the same type generally won't
   have the same memory layout.

   pybind11 can perform many kinds of conversions automatically. An overview
   is provided in the table ":ref:`conversion_table`".

The following subsections discuss the differences between these options in more
detail. The main focus in this section is on type conversions, which represent
the last case of the above list.

.. toctree::
   :maxdepth: 1

   overview
   strings
   stl
   functional
   chrono
   eigen
   custom

