CTEST_CHANGE_ID
---------------

.. versionadded:: 3.4

Specify the CTest ``ChangeId`` setting
in a :manual:`ctest(1)` dashboard client script.

This setting allows CTest to pass arbitrary information about this
build up to CDash.  One use of this feature is to allow CDash to
post comments on your pull request if anything goes wrong with your build.
