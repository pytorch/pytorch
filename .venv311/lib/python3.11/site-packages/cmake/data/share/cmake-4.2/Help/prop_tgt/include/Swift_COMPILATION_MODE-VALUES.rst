``incremental``
  Compiles each Swift source in the module separately, resulting in better
  parallelism in the build. The compiler emits additional information into
  the build directory improving rebuild performance when small changes are made
  to the source between rebuilds. This is the best option to use while
  iterating on changes in a project.

``wholemodule``
  Whole-module optimizations are slowest to compile, but results in the most
  optimized library. The entire context is loaded into once instance of the
  compiler, so there is no parallelism across source files in the module.

``singlefile``
  Compiles each source in a Swift modules separately, resulting in better
  parallelism. Unlike the ``incremental`` build mode, no additional information
  is emitted by the compiler during the build, so rebuilding after making small
  changes to the source file will not run faster. This option should be used
  sparingly, preferring ``incremental`` builds, unless working around a compiler
  bug.
