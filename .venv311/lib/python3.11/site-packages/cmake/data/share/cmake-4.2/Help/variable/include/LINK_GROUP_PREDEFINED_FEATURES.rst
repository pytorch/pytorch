``RESCAN``
  Some linkers are single-pass only.  For such linkers, circular references
  between libraries typically result in unresolved symbols.  This feature
  instructs the linker to search the specified static libraries repeatedly
  until no new undefined references are created.

  Normally, a static library is searched only once in the order that it is
  specified on the command line.  If a symbol in that library is needed to
  resolve an undefined symbol referred to by an object in a library that
  appears later on the command line, the linker would not be able to resolve
  that reference.  By grouping the static libraries with the ``RESCAN``
  feature, they will all be searched repeatedly until all possible references
  are resolved.  This will use linker options like ``--start-group`` and
  ``--end-group``, or on SunOS, ``-z rescan-start`` and ``-z rescan-end``.

  Using this feature has a significant performance cost. It is best to use it
  only when there are unavoidable circular references between two or more
  static libraries.

  This feature is available when using toolchains that target Linux, BSD, and
  SunOS.  It can also be used when targeting Windows platforms if the GNU
  toolchain is used.
