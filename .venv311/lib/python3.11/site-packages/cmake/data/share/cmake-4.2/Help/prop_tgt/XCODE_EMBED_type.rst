XCODE_EMBED_<type>
------------------

.. versionadded:: 3.20

Tell the :generator:`Xcode` generator to embed the specified list of items into
the target bundle.  ``<type>`` specifies the embed build phase to use.
See the Xcode documentation for the base location of each ``<type>``.

The supported values for ``<type>`` are:

``FRAMEWORKS``
  The specified items will be added to the ``Embed Frameworks`` build phase.
  The items can be CMake target names or paths to frameworks or libraries.

``APP_EXTENSIONS``
  .. versionadded:: 3.21

  The specified items will be added to the ``Embed App Extensions`` build
  phase, with ``Destination`` set to ``PlugIns and Foundation Extensions``
  They must be CMake target names.

``EXTENSIONKIT_EXTENSIONS``
  .. versionadded:: 3.26

  The specified items will be added to the ``Embed App Extensions`` build
  phase, with ``Destination`` set to ``ExtensionKit Extensions``
  They must be CMake target names, and should likely have the
  ``XCODE_PRODUCT_TYPE`` target property set to
  ``com.apple.product-type.extensionkit-extension``
  as well as the  ``XCODE_EXPLICIT_FILE_TYPE`` to
  ``wrapper.extensionkit-extension``

``PLUGINS``
  .. versionadded:: 3.23

  The specified items will be added to the ``Embed PlugIns`` build phase.
  They must be CMake target names.

``RESOURCES``
  .. versionadded:: 3.28

  The specified items will be added to the ``Embed Resources`` build phase.
  They must be CMake target names or folder paths.

``XPC_SERVICES``
  .. versionadded:: 3.29

  The specified items will be added to the ``Embed XPC Services`` build phase.
  They must be CMake target names.

When listing a target as any of the things to embed, Xcode must see that target
as part of the same Xcode project, or a sub-project of the one defining the
bundle.  In order to satisfy this constraint, the CMake project must ensure
at least one of the following:

* The :variable:`CMAKE_XCODE_GENERATE_TOP_LEVEL_PROJECT_ONLY` variable is set
  to true in the top level ``CMakeLists.txt`` file.  This is the simplest and
  most robust approach.
* Define the target-to-embed in a subdirectory of the one that defines the
  target being embedded into.
* If the target-to-embed and the target being embedded into are in separate,
  unrelated directories (i.e. they are siblings, not one a parent of the
  other), ensure they have a common :command:`project` call in a parent
  directory and no other :command:`project` calls between themselves and that
  common :command:`project` call.

See also :prop_tgt:`XCODE_EMBED_<type>_PATH`,
:prop_tgt:`XCODE_EMBED_<type>_REMOVE_HEADERS_ON_COPY` and
:prop_tgt:`XCODE_EMBED_<type>_CODE_SIGN_ON_COPY`.
