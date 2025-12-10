CMAKE_SUBLIME_TEXT_2_ENV_SETTINGS
---------------------------------

.. versionadded:: 3.8

This variable contains a list of env vars as a list of tokens with the
syntax ``var=value``.

Example:

.. code-block:: cmake

  set(CMAKE_SUBLIME_TEXT_2_ENV_SETTINGS
     "FOO=FOO1\;FOO2\;FOON"
     "BAR=BAR1\;BAR2\;BARN"
     "BAZ=BAZ1\;BAZ2\;BAZN"
     "FOOBAR=FOOBAR1\;FOOBAR2\;FOOBARN"
     "VALID="
     )

In case of malformed variables CMake will fail:

.. code-block:: cmake

  set(CMAKE_SUBLIME_TEXT_2_ENV_SETTINGS
      "THIS_IS_NOT_VALID"
      )
