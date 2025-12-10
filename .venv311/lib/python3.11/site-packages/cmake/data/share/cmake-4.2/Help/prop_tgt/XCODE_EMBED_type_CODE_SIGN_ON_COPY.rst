XCODE_EMBED_<type>_CODE_SIGN_ON_COPY
------------------------------------

.. versionadded:: 3.20

Boolean property used only by the :generator:`Xcode` generator.  It specifies
whether to perform code signing for the items that are embedded using the
:prop_tgt:`XCODE_EMBED_<type>` property.

The supported values for ``<type>`` are:

``FRAMEWORKS``

``APP_EXTENSIONS``
  .. versionadded:: 3.21

``EXTENSIONKIT_EXTENSIONS``
  .. versionadded:: 3.26

``PLUGINS``
  .. versionadded:: 3.23

If a ``XCODE_EMBED_<type>_CODE_SIGN_ON_COPY`` property is not defined on the
target, no code signing on copy will be performed for that ``<type>``.
