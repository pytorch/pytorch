MSVC_TOOLSET_VERSION
--------------------

.. versionadded:: 3.12

The toolset version of Microsoft Visual C/C++ being used if any.
If MSVC-like is being used, this variable is set based on the version
of the compiler as given by the :variable:`MSVC_VERSION` variable.

Known toolset version numbers are:

.. table::
  :align: left

  ===== ==============
  Value Version
  ===== ==============
  80    VS 2005 (8.0)
  90    VS 2008 (9.0)
  100   VS 2010 (10.0)
  110   VS 2012 (11.0)
  120   VS 2013 (12.0)
  140   VS 2015 (14.0)
  141   VS 2017 (15.0)
  142   VS 2019 (16.0)
  143   VS 2022 (17.0)
  145   VS 2026 (18.0)
  ===== ==============

Compiler versions newer than those known to CMake will be reported
as the latest known toolset version.

See also the :variable:`MSVC_VERSION` variable.
