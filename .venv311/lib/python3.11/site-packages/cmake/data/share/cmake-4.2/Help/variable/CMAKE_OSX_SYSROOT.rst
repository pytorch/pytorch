CMAKE_OSX_SYSROOT
-----------------

Specify the location or name of the macOS platform SDK to be used.

If not set explicitly, the value is initialized by the ``SDKROOT``
environment variable, if set.  Otherwise, the value defaults to empty,
and the compiler is expected to choose a default macOS SDK on its own.

.. versionchanged:: 4.0
  The default is now empty.  Previously a default was computed based on
  the :variable:`CMAKE_OSX_DEPLOYMENT_TARGET` or the host platform.

In order to pass an explicit macOS SDK via the compiler's ``-isysroot`` flag,
users may configure their build tree with ``-DCMAKE_OSX_SYSROOT=macosx``,
or ``export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"`` in their
environment.

Notes:

* macOS compilers in ``/usr/bin``, when not invoked with ``-isysroot``,
  search for headers in ``/usr/local/include`` before system SDK paths,
  matching the convention on many platforms.  Users on macOS-x86_64 hosts
  with Homebrew installed in ``/usr/local`` should pass an explicit SDK,
  as described above, when not building with Homebrew tools.

* Some Clang compilers have no default macOS SDK selection.  For these,
  if :variable:`CMAKE_OSX_SYSROOT` is empty, CMake will automatically pass
  ``-isysroot`` with the macOS SDK printed by ``xcrun --show-sdk-path``.

.. include:: include/CMAKE_OSX_VARIABLE.rst
