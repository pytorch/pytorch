MSVC_VERSION
------------

The version of Microsoft Visual C/C++ being used if any.
If a compiler simulating Visual C++ is being used, this variable is set
to the toolset version simulated as given by the ``_MSC_VER``
preprocessor definition.

Known version numbers are:

.. table::
  :align: left

  ========= ==============
  Value     Version
  ========= ==============
  1200      VS  6.0
  1300      VS  7.0
  1310      VS  7.1
  1400      VS  8.0 (v80 toolset)
  1500      VS  9.0 (v90 toolset)
  1600      VS 10.0 (v100 toolset)
  1700      VS 11.0 (v110 toolset)
  1800      VS 12.0 (v120 toolset)
  1900      VS 14.0 (v140 toolset)
  1910-1919 VS 15.0 (v141 toolset)
  1920-1929 VS 16.0 (v142 toolset)
  1930-1949 VS 17.0 (v143 toolset)
  1950-1959 VS 18.0 (v145 toolset)
  ========= ==============

See also the  :variable:`CMAKE_<LANG>_COMPILER_VERSION` and
:variable:`MSVC_TOOLSET_VERSION` variable.
