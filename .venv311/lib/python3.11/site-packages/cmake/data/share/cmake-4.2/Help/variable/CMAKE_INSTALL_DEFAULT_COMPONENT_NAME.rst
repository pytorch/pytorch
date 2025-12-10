CMAKE_INSTALL_DEFAULT_COMPONENT_NAME
------------------------------------

Default component used in :command:`install` commands.

If an :command:`install` command is used without the ``COMPONENT`` argument,
these files will be grouped into a default component.  The name of this
default install component will be taken from this variable.  It
defaults to ``Unspecified``.
