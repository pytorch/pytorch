XCODE_EMBED_<type>_PATH
-----------------------

.. versionadded:: 3.20

This property is used only by the :generator:`Xcode` generator.  When defined,
it specifies the relative path to use when embedding the items specified by
:prop_tgt:`XCODE_EMBED_<type>`.  The path is relative
to the base location of the ``Embed XXX`` build phase associated with
``<type>``.  See the Xcode documentation for the base location of each
``<type>``.

The supported values for ``<type>`` are:

``FRAMEWORKS``

``APP_EXTENSIONS``
  .. versionadded:: 3.21

``EXTENSIONKIT_EXTENSIONS``
  .. versionadded:: 3.26

``PLUGINS``
  .. versionadded:: 3.23

``RESOURCES``
  .. versionadded:: 3.28

``XPC_SERVICES``
  .. versionadded:: 3.29
