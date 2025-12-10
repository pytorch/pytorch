OBJECT_OUTPUTS
--------------

Additional outputs for a :generator:`Ninja` or :ref:`Makefile Generators` rule.

Additional outputs created by compilation of this source file.  If any
of these outputs is missing the object will be recompiled.  This is
supported only on the :generator:`Ninja` and :ref:`Makefile Generators`
and will be ignored on other generators.

This property supports
:manual:`generator expressions <cmake-generator-expressions(7)>`.
