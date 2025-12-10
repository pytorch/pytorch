include_regular_expression
--------------------------

Set the regular expression used for dependency checking.

.. code-block:: cmake

  include_regular_expression(regex_match [regex_complain])

Sets the regular expressions used in dependency checking.  Only files
matching ``regex_match`` will be traced as dependencies.  Only files
matching ``regex_complain`` will generate warnings if they cannot be found
(standard header paths are not searched).

The default for ``regex_match`` is ``"^.*$"`` (match everything).
The default for ``regex_complain`` is ``"^$"`` (match empty string only).
