ANDROID_STL_TYPE
----------------

.. versionadded:: 3.4

When :ref:`Cross Compiling for Android with NVIDIA Nsight Tegra Visual Studio
Edition`, this property specifies the type of STL support for the project.
This is a string property that could set to the one of the following values:

``none``
  No C++ Support
``system``
  Minimal C++ without STL
``gabi++_static``
  GAbi++ Static
``gabi++_shared``
  GAbi++ Shared
``gnustl_static``
  GNU libstdc++ Static
``gnustl_shared``
  GNU libstdc++ Shared
``stlport_static``
  STLport Static
``stlport_shared``
  STLport Shared

This property is initialized by the value of the
:variable:`CMAKE_ANDROID_STL_TYPE` variable if it is set when a target is
created.
