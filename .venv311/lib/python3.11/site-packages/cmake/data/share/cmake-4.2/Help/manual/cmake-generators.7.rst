.. cmake-manual-description: CMake Generators Reference

cmake-generators(7)
*******************

.. only:: html

   .. contents::

Introduction
============

A *CMake Generator* is responsible for writing the input files for
a native build system.  Exactly one of the `CMake Generators`_ must be
selected for a build tree to determine what native build system is to
be used.  Optionally one of the `Extra Generators`_ may be selected
as a variant of some of the `Command-Line Build Tool Generators`_ to
produce project files for an auxiliary IDE.

CMake Generators are platform-specific so each may be available only
on certain platforms.  The :manual:`cmake(1)` command-line tool
:option:`--help <cmake --help>` output lists available generators on the
current platform.  Use its :option:`-G <cmake -G>` option to specify the
generator for a new build tree. The :manual:`cmake-gui(1)` offers
interactive selection of a generator when creating a new build tree.

CMake Generators
================

.. _`Command-Line Build Tool Generators`:

Command-Line Build Tool Generators
----------------------------------

These generators support command-line build tools.  In order to use them,
one must launch CMake from a command-line prompt whose environment is
already configured for the chosen compiler and build tool.

.. _`Makefile Generators`:

Makefile Generators
^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   /generator/Borland Makefiles
   /generator/MSYS Makefiles
   /generator/MinGW Makefiles
   /generator/NMake Makefiles
   /generator/NMake Makefiles JOM
   /generator/Unix Makefiles
   /generator/Watcom WMake

.. _`Ninja Generators`:

Ninja Generators
^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   /generator/Ninja
   /generator/Ninja Multi-Config

FASTBuild Generator
^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   /generator/FASTBuild

.. _`IDE Build Tool Generators`:

IDE Build Tool Generators
-------------------------

These generators support Integrated Development Environment (IDE)
project files.  Since the IDEs configure their own environment
one may launch CMake from any environment.

.. _`Visual Studio Generators`:

Visual Studio Generators
^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   /generator/Visual Studio 6
   /generator/Visual Studio 7
   /generator/Visual Studio 7 .NET 2003
   /generator/Visual Studio 8 2005
   /generator/Visual Studio 9 2008
   /generator/Visual Studio 10 2010
   /generator/Visual Studio 11 2012
   /generator/Visual Studio 12 2013
   /generator/Visual Studio 14 2015
   /generator/Visual Studio 15 2017
   /generator/Visual Studio 16 2019
   /generator/Visual Studio 17 2022
   /generator/Visual Studio 18 2026

Other Generators
^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   /generator/Green Hills MULTI
   /generator/Xcode

.. _`Extra Generators`:

Extra Generators
================

.. deprecated:: 3.27

  Support for "Extra Generators" is deprecated and will be removed from
  a future version of CMake.  IDEs may use the :manual:`cmake-file-api(7)`
  to view CMake-generated project build trees.

Some of the `CMake Generators`_ listed in the :manual:`cmake(1)`
command-line tool :option:`--help <cmake --help>` output may have
variants that specify an extra generator for an auxiliary IDE tool.
Such generator names have the form ``<extra-generator> - <main-generator>``.
The following extra generators are known to CMake.

.. toctree::
   :maxdepth: 1

   /generator/CodeBlocks
   /generator/CodeLite
   /generator/Eclipse CDT4
   /generator/Kate
   /generator/Sublime Text 2
