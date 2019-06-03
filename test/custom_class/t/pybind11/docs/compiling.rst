.. _compiling:

Build systems
#############

Building with setuptools
========================

For projects on PyPI, building with setuptools is the way to go. Sylvain Corlay
has kindly provided an example project which shows how to set up everything,
including automatic generation of documentation using Sphinx. Please refer to
the [python_example]_ repository.

.. [python_example] https://github.com/pybind/python_example

Building with cppimport
========================

[cppimport]_ is a small Python import hook that determines whether there is a C++
source file whose name matches the requested module. If there is, the file is
compiled as a Python extension using pybind11 and placed in the same folder as
the C++ source file. Python is then able to find the module and load it.

.. [cppimport] https://github.com/tbenthompson/cppimport

.. _cmake:

Building with CMake
===================

For C++ codebases that have an existing CMake-based build system, a Python
extension module can be created with just a few lines of code:

.. code-block:: cmake

    cmake_minimum_required(VERSION 2.8.12)
    project(example)

    add_subdirectory(pybind11)
    pybind11_add_module(example example.cpp)

This assumes that the pybind11 repository is located in a subdirectory named
:file:`pybind11` and that the code is located in a file named :file:`example.cpp`.
The CMake command ``add_subdirectory`` will import the pybind11 project which
provides the ``pybind11_add_module`` function. It will take care of all the
details needed to build a Python extension module on any platform.

A working sample project, including a way to invoke CMake from :file:`setup.py` for
PyPI integration, can be found in the [cmake_example]_  repository.

.. [cmake_example] https://github.com/pybind/cmake_example

pybind11_add_module
-------------------

To ease the creation of Python extension modules, pybind11 provides a CMake
function with the following signature:

.. code-block:: cmake

    pybind11_add_module(<name> [MODULE | SHARED] [EXCLUDE_FROM_ALL]
                        [NO_EXTRAS] [SYSTEM] [THIN_LTO] source1 [source2 ...])

This function behaves very much like CMake's builtin ``add_library`` (in fact,
it's a wrapper function around that command). It will add a library target
called ``<name>`` to be built from the listed source files. In addition, it
will take care of all the Python-specific compiler and linker flags as well
as the OS- and Python-version-specific file extension. The produced target
``<name>`` can be further manipulated with regular CMake commands.

``MODULE`` or ``SHARED`` may be given to specify the type of library. If no
type is given, ``MODULE`` is used by default which ensures the creation of a
Python-exclusive module. Specifying ``SHARED`` will create a more traditional
dynamic library which can also be linked from elsewhere. ``EXCLUDE_FROM_ALL``
removes this target from the default build (see CMake docs for details).

Since pybind11 is a template library, ``pybind11_add_module`` adds compiler
flags to ensure high quality code generation without bloat arising from long
symbol names and duplication of code in different translation units. It
sets default visibility to *hidden*, which is required for some pybind11
features and functionality when attempting to load multiple pybind11 modules
compiled under different pybind11 versions.  It also adds additional flags
enabling LTO (Link Time Optimization) and strip unneeded symbols. See the
:ref:`FAQ entry <faq:symhidden>` for a more detailed explanation. These
latter optimizations are never applied in ``Debug`` mode.  If ``NO_EXTRAS`` is
given, they will always be disabled, even in ``Release`` mode. However, this
will result in code bloat and is generally not recommended.

By default, pybind11 and Python headers will be included with ``-I``. In order
to include pybind11 as system library, e.g. to avoid warnings in downstream
code with warn-levels outside of pybind11's scope, set the option ``SYSTEM``.

As stated above, LTO is enabled by default. Some newer compilers also support
different flavors of LTO such as `ThinLTO`_. Setting ``THIN_LTO`` will cause
the function to prefer this flavor if available. The function falls back to
regular LTO if ``-flto=thin`` is not available.

.. _ThinLTO: http://clang.llvm.org/docs/ThinLTO.html

Configuration variables
-----------------------

By default, pybind11 will compile modules with the C++14 standard, if available
on the target compiler, falling back to C++11 if C++14 support is not
available.  Note, however, that this default is subject to change: future
pybind11 releases are expected to migrate to newer C++ standards as they become
available.  To override this, the standard flag can be given explicitly in
``PYBIND11_CPP_STANDARD``:

.. code-block:: cmake

    # Use just one of these:
    # GCC/clang:
    set(PYBIND11_CPP_STANDARD -std=c++11)
    set(PYBIND11_CPP_STANDARD -std=c++14)
    set(PYBIND11_CPP_STANDARD -std=c++1z) # Experimental C++17 support
    # MSVC:
    set(PYBIND11_CPP_STANDARD /std:c++14)
    set(PYBIND11_CPP_STANDARD /std:c++latest) # Enables some MSVC C++17 features

    add_subdirectory(pybind11)  # or find_package(pybind11)

Note that this and all other configuration variables must be set **before** the
call to ``add_subdirectory`` or ``find_package``. The variables can also be set
when calling CMake from the command line using the ``-D<variable>=<value>`` flag.

The target Python version can be selected by setting ``PYBIND11_PYTHON_VERSION``
or an exact Python installation can be specified with ``PYTHON_EXECUTABLE``.
For example:

.. code-block:: bash

    cmake -DPYBIND11_PYTHON_VERSION=3.6 ..
    # or
    cmake -DPYTHON_EXECUTABLE=path/to/python ..

find_package vs. add_subdirectory
---------------------------------

For CMake-based projects that don't include the pybind11 repository internally,
an external installation can be detected through ``find_package(pybind11)``.
See the `Config file`_ docstring for details of relevant CMake variables.

.. code-block:: cmake

    cmake_minimum_required(VERSION 2.8.12)
    project(example)

    find_package(pybind11 REQUIRED)
    pybind11_add_module(example example.cpp)

Once detected, the aforementioned ``pybind11_add_module`` can be employed as
before. The function usage and configuration variables are identical no matter
if pybind11 is added as a subdirectory or found as an installed package. You
can refer to the same [cmake_example]_ repository for a full sample project
-- just swap out ``add_subdirectory`` for ``find_package``.

.. _Config file: https://github.com/pybind/pybind11/blob/master/tools/pybind11Config.cmake.in

Advanced: interface library target
----------------------------------

When using a version of CMake greater than 3.0, pybind11 can additionally
be used as a special *interface library* . The target ``pybind11::module``
is available with pybind11 headers, Python headers and libraries as needed,
and C++ compile definitions attached. This target is suitable for linking
to an independently constructed (through ``add_library``, not
``pybind11_add_module``) target in the consuming project.

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.0)
    project(example)

    find_package(pybind11 REQUIRED)  # or add_subdirectory(pybind11)

    add_library(example MODULE main.cpp)
    target_link_libraries(example PRIVATE pybind11::module)
    set_target_properties(example PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                             SUFFIX "${PYTHON_MODULE_EXTENSION}")

.. warning::

    Since pybind11 is a metatemplate library, it is crucial that certain
    compiler flags are provided to ensure high quality code generation. In
    contrast to the ``pybind11_add_module()`` command, the CMake interface
    library only provides the *minimal* set of parameters to ensure that the
    code using pybind11 compiles, but it does **not** pass these extra compiler
    flags (i.e. this is up to you).

    These include Link Time Optimization (``-flto`` on GCC/Clang/ICPC, ``/GL``
    and ``/LTCG`` on Visual Studio) and .OBJ files with many sections on Visual
    Studio (``/bigobj``).  The :ref:`FAQ <faq:symhidden>` contains an
    explanation on why these are needed.

Embedding the Python interpreter
--------------------------------

In addition to extension modules, pybind11 also supports embedding Python into
a C++ executable or library. In CMake, simply link with the ``pybind11::embed``
target. It provides everything needed to get the interpreter running. The Python
headers and libraries are attached to the target. Unlike ``pybind11::module``,
there is no need to manually set any additional properties here. For more
information about usage in C++, see :doc:`/advanced/embedding`.

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.0)
    project(example)

    find_package(pybind11 REQUIRED)  # or add_subdirectory(pybind11)

    add_executable(example main.cpp)
    target_link_libraries(example PRIVATE pybind11::embed)

.. _building_manually:

Building manually
=================

pybind11 is a header-only library, hence it is not necessary to link against
any special libraries and there are no intermediate (magic) translation steps.

On Linux, you can compile an example such as the one given in
:ref:`simple_example` using the following command:

.. code-block:: bash

    $ c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` example.cpp -o example`python3-config --extension-suffix`

The flags given here assume that you're using Python 3. For Python 2, just
change the executable appropriately (to ``python`` or ``python2``).

The ``python3 -m pybind11 --includes`` command fetches the include paths for
both pybind11 and Python headers. This assumes that pybind11 has been installed
using ``pip`` or ``conda``. If it hasn't, you can also manually specify
``-I <path-to-pybind11>/include`` together with the Python includes path
``python3-config --includes``.

Note that Python 2.7 modules don't use a special suffix, so you should simply
use ``example.so`` instead of ``example`python3-config --extension-suffix```.
Besides, the ``--extension-suffix`` option may or may not be available, depending
on the distribution; in the latter case, the module extension can be manually
set to ``.so``.

On Mac OS: the build command is almost the same but it also requires passing
the ``-undefined dynamic_lookup`` flag so as to ignore missing symbols when
building the module:

.. code-block:: bash

    $ c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` example.cpp -o example`python3-config --extension-suffix`

In general, it is advisable to include several additional build parameters
that can considerably reduce the size of the created binary. Refer to section
:ref:`cmake` for a detailed example of a suitable cross-platform CMake-based
build system that works on all platforms including Windows.

.. note::

    On Linux and macOS, it's better to (intentionally) not link against
    ``libpython``. The symbols will be resolved when the extension library
    is loaded into a Python binary. This is preferable because you might
    have several different installations of a given Python version (e.g. the
    system-provided Python, and one that ships with a piece of commercial
    software). In this way, the plugin will work with both versions, instead
    of possibly importing a second Python library into a process that already
    contains one (which will lead to a segfault).

Generating binding code automatically
=====================================

The ``Binder`` project is a tool for automatic generation of pybind11 binding
code by introspecting existing C++ codebases using LLVM/Clang. See the
[binder]_ documentation for details.

.. [binder] http://cppbinder.readthedocs.io/en/latest/about.html
