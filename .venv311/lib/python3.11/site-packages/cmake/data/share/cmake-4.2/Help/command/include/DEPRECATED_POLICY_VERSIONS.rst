.. versionchanged:: 4.0

  Compatibility with versions of CMake older than 3.5 is removed.
  Calls to :command:`cmake_minimum_required(VERSION)` or
  :command:`cmake_policy(VERSION)` that do not specify at least
  3.5 as their policy version (optionally via ``...<max>``)
  will produce an error in CMake 4.0 and above.

.. versionchanged:: 3.31

  Compatibility with versions of CMake older than 3.10 is deprecated.
  Calls to :command:`cmake_minimum_required(VERSION)` or
  :command:`cmake_policy(VERSION)` that do not specify at least
  3.10 as their policy version (optionally via ``...<max>``)
  will produce a deprecation warning in CMake 3.31 and above.

.. versionchanged:: 3.27

  Compatibility with versions of CMake older than 3.5 is deprecated.
  Calls to :command:`cmake_minimum_required(VERSION)` or
  :command:`cmake_policy(VERSION)` that do not specify at least
  3.5 as their policy version (optionally via ``...<max>``)
  will produce a deprecation warning in CMake 3.27 and above.

.. versionchanged:: 3.19

  Compatibility with versions of CMake older than 2.8.12 is deprecated.
  Calls to :command:`cmake_minimum_required(VERSION)` or
  :command:`cmake_policy(VERSION)` that do not specify at least
  2.8.12 as their policy version (optionally via ``...<max>``)
  will produce a deprecation warning in CMake 3.19 and above.
