Disclaimer: Most native build tools have poor support for escaping
certain values.  CMake has work-arounds for many cases but some values
may just not be possible to pass correctly.  If a value does not seem
to be escaped correctly, do not attempt to work-around the problem by
adding escape sequences to the value.  Your work-around may break in a
future version of CMake that has improved escape support.  Instead
consider defining the macro in a (configured) header file.  Then
report the limitation.  Known limitations include:

  ============= ========================
  ``#``         Broken almost everywhere.
  ``;``         Broken in VS IDE 7.0 and Borland Makefiles.
  ``,``         Broken in VS IDE.
  ``%``         Broken in some cases in NMake.
  ``& |``       Broken in some cases on MinGW.
  ``^ < > \ "`` Broken in most Make tools on Windows.
  ============= ========================

CMake does not reject these values outright because they do work in
some cases.  Use with caution.
