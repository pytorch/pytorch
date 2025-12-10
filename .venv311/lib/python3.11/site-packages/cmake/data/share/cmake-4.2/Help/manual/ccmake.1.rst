.. cmake-manual-description: CMake Curses Dialog Command-Line Reference

ccmake(1)
*********

Synopsis
========

.. parsed-literal::

 ccmake [<options>] -B <path-to-build> [-S <path-to-source>]
 ccmake [<options>] <path-to-source | path-to-existing-build>

Description
===========

The :program:`ccmake` executable is the CMake curses interface.  Project
configuration settings may be specified interactively through this
GUI.  Brief instructions are provided at the bottom of the terminal
when the program is running.

CMake is a cross-platform build system generator.  Projects specify
their build process with platform-independent CMake listfiles included
in each directory of a source tree with the name ``CMakeLists.txt``.
Users build a project by using CMake to generate a build system for a
native tool on their platform.

Options
=======

.. program:: ccmake

.. include:: include/OPTIONS_BUILD.rst

.. include:: include/OPTIONS_HELP.rst

See Also
========

.. include:: include/LINKS.rst
