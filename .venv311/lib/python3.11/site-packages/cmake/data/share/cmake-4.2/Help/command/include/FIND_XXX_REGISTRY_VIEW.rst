Specify which registry views must be queried. This option is only meaningful
on ``Windows`` platforms and will be ignored on other ones. When not
specified, the |FIND_XXX_REGISTRY_VIEW_DEFAULT| view is used when the
:policy:`CMP0134` policy is ``NEW``. Refer to :policy:`CMP0134` for the
default view when the policy is ``OLD``.

``64``
  Query the 64-bit registry. On 32-bit Windows, it always returns the string
  ``/REGISTRY-NOTFOUND``.

``32``
  Query the 32-bit registry.

``64_32``
  Query both views (``64`` and ``32``) and generate a path for each.

``32_64``
  Query both views (``32`` and ``64``) and generate a path for each.

``HOST``
  Query the registry matching the architecture of the host: ``64`` on 64-bit
  Windows and ``32`` on 32-bit Windows.

``TARGET``
  Query the registry matching the architecture specified by the
  :variable:`CMAKE_SIZEOF_VOID_P` variable. If not defined, fall back to
  ``HOST`` view.

``BOTH``
  Query both views (``32`` and ``64``). The order depends on the following
  rules: If the :variable:`CMAKE_SIZEOF_VOID_P` variable is defined, use the
  following view depending on the content of this variable:

  * ``8``: ``64_32``
  * ``4``: ``32_64``

  If the :variable:`CMAKE_SIZEOF_VOID_P` variable is not defined, rely on the
  architecture of the host:

  * 64-bit: ``64_32``
  * 32-bit: ``32``
