This directory contains contributions from various users.
They are provided "as is", without any support. Nevertheless,
most of them are subject to be included in Eigen in the future.

In order to use an unsupported module you have to do either:

 - add the path_to_eigen/unsupported directory to your include path and do:
   #include <Eigen/ModuleHeader>

 - or directly do:
   #include <unsupported/Eigen/ModuleHeader>


If you are interested in contributing to one of them, or have other stuff
you would like to share, feel free to contact us:
http://eigen.tuxfamily.org/index.php?title=Main_Page#Mailing_list

Any kind of contributions are much appreciated, even very preliminary ones.
However, it:
 - must rely on Eigen,
 - must be highly related to math,
 - should have some general purpose in the sense that it could
   potentially become an offical Eigen module (or be merged into another one).

In doubt feel free to contact us. For instance, if your addons is very too specific
but it shows an interesting way of using Eigen, then it could be a nice demo.


This directory is organized as follow:

unsupported/Eigen/ModuleHeader1
unsupported/Eigen/ModuleHeader2
unsupported/Eigen/...
unsupported/Eigen/src/Module1/SourceFile1.h
unsupported/Eigen/src/Module1/SourceFile2.h
unsupported/Eigen/src/Module1/...
unsupported/Eigen/src/Module2/SourceFile1.h
unsupported/Eigen/src/Module2/SourceFile2.h
unsupported/Eigen/src/Module2/...
unsupported/Eigen/src/...
unsupported/doc/snippets/.cpp   <- code snippets for the doc
unsupported/doc/examples/.cpp   <- examples for the doc
unsupported/doc/TutorialModule1.dox
unsupported/doc/TutorialModule2.dox
unsupported/doc/...
unsupported/test/.cpp           <- unit test files

The documentation is generated at the same time than the main Eigen documentation.
The .html files are generated in: build_dir/doc/html/unsupported/

