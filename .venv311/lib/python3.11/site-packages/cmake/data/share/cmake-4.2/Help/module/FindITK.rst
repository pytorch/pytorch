FindITK
-------

This module no longer exists.

This module existed in versions of CMake prior to 3.1, but became
only a thin wrapper around ``find_package(ITK NO_MODULE)`` to
provide compatibility for projects using long-outdated conventions.
Now ``find_package(ITK)`` will search for ``ITKConfig.cmake``
directly.
