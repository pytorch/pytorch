CPACK_WIX_ACL
-------------

.. versionadded:: 3.1

Specifies access permissions for files or directories
installed by a WiX installer.

The property can contain multiple list entries,
each of which has to match the following format.

::

  <user>[@<domain>]=<permission>[,<permission>]

``<user>`` and ``<domain>`` specify the windows user and domain for which the
``<Permission>`` element should be generated.

``<permission>`` is any of the YesNoType attributes listed at:
https://docs.firegiant.com/wix3/xsd/wix/permission/

The property is currently only supported by the :cpack_gen:`CPack WIX Generator`.
