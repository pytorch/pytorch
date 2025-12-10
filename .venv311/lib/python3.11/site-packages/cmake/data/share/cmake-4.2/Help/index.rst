.. title:: CMake Reference Documentation

Introduction
############

CMake is a tool to manage building of source code.  Originally, CMake was
designed as a generator for various dialects of ``Makefile``, today
CMake generates modern buildsystems such as ``Ninja`` as well as project
files for IDEs such as Visual Studio and Xcode.

CMake is widely used for the C and C++ languages, but it may be used to
build source code of other languages too.

People encountering CMake for the first time may have different initial
goals.  To learn how to build a source code package downloaded from the
internet, start with the :guide:`User Interaction Guide`.
This will detail the steps needed to run the :manual:`cmake(1)` or
:manual:`cmake-gui(1)` executable and how to choose a generator, and
how to complete the build.

The :guide:`Using Dependencies Guide` is aimed at developers
wishing to get started using a third-party library.

For developers starting a project using CMake, the :guide:`CMake Tutorial`
is a suitable starting point.  The :manual:`cmake-buildsystem(7)`
manual is aimed at developers expanding their knowledge of maintaining
a buildsystem and becoming familiar with the build targets that
can be represented in CMake.  The :manual:`cmake-packages(7)` manual
explains how to create packages which can easily be consumed by
third-party CMake-based buildsystems.

Command-Line Tools
##################

.. toctree::
   :maxdepth: 1

   /manual/cmake.1
   /manual/ctest.1
   /manual/cpack.1

Interactive Dialogs
###################

.. toctree::
   :maxdepth: 1

   /manual/cmake-gui.1
   /manual/ccmake.1

Reference Manuals
#################

.. toctree::
   :maxdepth: 1

   /manual/cmake-buildsystem.7
   /manual/cmake-commands.7
   /manual/cmake-compile-features.7
   /manual/cmake-configure-log.7
   /manual/cmake-cxxmodules.7
   /manual/cmake-developer.7
   /manual/cmake-env-variables.7
   /manual/cmake-file-api.7
   /manual/cmake-generator-expressions.7
   /manual/cmake-generators.7
   /manual/cmake-instrumentation.7
   /manual/cmake-language.7
   /manual/cmake-modules.7
   /manual/cmake-packages.7
   /manual/cmake-policies.7
   /manual/cmake-presets.7
   /manual/cmake-properties.7
   /manual/cmake-qt.7
   /manual/cmake-server.7
   /manual/cmake-toolchains.7
   /manual/cmake-variables.7
   /manual/cpack-generators.7

.. only:: not man

 Guides
 ######

 .. toctree::
    :maxdepth: 1

    /guide/tutorial/index
    /guide/user-interaction/index
    /guide/using-dependencies/index
    /guide/importing-exporting/index
    /guide/ide-integration/index

.. only:: html or text

 Release Notes
 #############

 .. toctree::
    :maxdepth: 1

    /release/index

.. only:: html

 Index and Search
 ################

 * :ref:`genindex`
 * :ref:`search`
