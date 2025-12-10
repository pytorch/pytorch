VS_WINRT_COMPONENT
------------------

.. versionadded:: 3.1

Mark a target as a Windows Runtime component for the :ref:`Visual Studio Generators`.
Compile the target with ``C++/CX`` language extensions for Windows Runtime.
For ``SHARED`` and ``MODULE`` libraries, this also defines the
``_WINRT_DLL`` preprocessor macro.

.. note::
  Currently this is implemented only by :ref:`Visual Studio Generators`.
  Support may be added to other generators in the future.
