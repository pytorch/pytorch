.. cmake-manual-description: CMake GUI Command-Line Reference

cmake-gui(1)
************

Synopsis
========

.. parsed-literal::

 cmake-gui [<options>]
 cmake-gui [<options>] -B <path-to-build> [-S <path-to-source>]
 cmake-gui [<options>] <path-to-source | path-to-existing-build>
 cmake-gui [<options>] --browse-manual [<filename>]

Description
===========

The :program:`cmake-gui` executable is the CMake GUI.  Project configuration
settings may be specified interactively.  Brief instructions are
provided at the bottom of the window when the program is running.

CMake is a cross-platform build system generator.  Projects specify
their build process with platform-independent CMake listfiles included
in each directory of a source tree with the name ``CMakeLists.txt``.
Users build a project by using CMake to generate a build system for a
native tool on their platform.

Options
=======

.. program:: cmake-gui

.. option:: -S <path-to-source>

 Path to root directory of the CMake project to build.

.. option:: -B <path-to-build>

 Path to directory which CMake will use as the root of build directory.

 If the directory doesn't already exist CMake will make it.

.. option:: --preset=<preset-name>

 Name of the preset to use from the project's
 :manual:`presets <cmake-presets(7)>` files, if it has them.

.. option:: --browse-manual [<filename>]

 Open the CMake reference manual in a browser and immediately exit. If
 ``<filename>`` is specified, open that file within the reference manual
 instead of ``index.html``.

.. include:: include/OPTIONS_HELP.rst

See Also
========

.. include:: include/LINKS.rst
