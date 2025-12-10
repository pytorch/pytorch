RESOURCE_LOCK
-------------

Specify a list of resources that are locked by this test.

If multiple tests specify the same resource lock, they are guaranteed
not to run concurrently.

See also :prop_test:`FIXTURES_REQUIRED` if the resource requires any setup or
cleanup steps.

Both the :prop_test:`RESOURCE_GROUPS` and ``RESOURCE_LOCK`` properties serve
similar purposes, but they are distinct and orthogonal. Resources specified by
:prop_test:`RESOURCE_GROUPS` do not affect ``RESOURCE_LOCK``, and vice versa.
Whereas ``RESOURCE_LOCK`` is a simpler property that is used for locking one
global resource, :prop_test:`RESOURCE_GROUPS` is a more advanced property
that allows multiple tests to simultaneously use multiple resources of the
same type, specifying their requirements in a fine-grained manner.
