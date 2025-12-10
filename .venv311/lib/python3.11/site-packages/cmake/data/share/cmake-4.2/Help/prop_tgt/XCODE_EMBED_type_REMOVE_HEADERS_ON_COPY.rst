XCODE_EMBED_<type>_REMOVE_HEADERS_ON_COPY
-----------------------------------------

.. versionadded:: 3.20

Boolean property used only by the :generator:`Xcode` generator.  It specifies
whether to remove headers from all the frameworks that are embedded using the
:prop_tgt:`XCODE_EMBED_<type>` property.

The supported values for ``<type>`` are:

``FRAMEWORKS``
  If the ``XCODE_EMBED_FRAMEWORKS_REMOVE_HEADERS_ON_COPY`` property is not
  defined, headers will not be removed on copy by default.

``APP_EXTENSIONS``
  .. versionadded:: 3.21

  If the ``XCODE_EMBED_APP_EXTENSIONS_REMOVE_HEADERS_ON_COPY`` property is not
  defined, headers WILL be removed on copy by default.

``EXTENSIONKIT_EXTENSIONS``
  .. versionadded:: 3.26

  If the ``XCODE_EMBED_APP_EXTENSIONS_REMOVE_HEADERS_ON_COPY`` property is not
  defined, headers WILL be removed on copy by default.

``PLUGINS``
  .. versionadded:: 3.23
