
PyTorch on Windows is alpha quality. In other words, don't be surprised if stuff doesn't work.

# Compile
BLAS and Lapack are required. They should be installed in pytorch/torch/lib/tmp_install/. Instructions for compiling OpenBLAS are below. Right now, only Python36 is tested. Open the Visual Studio command prompt, and cd to pytorch. Run:
```
python setup.py install
```

Note that setup.py runs the file torch/lib/build_all.bat, which compiles the core TH libraries. In the likely chance that you have problems compiling it, you can edit and run it manually.

## OpenBLAS
As far as I can tell, OpenBLAS is the fastest BLAS implementation on Windows (besides Intel MKL). Note that it includes Lapack. To compile it, you need Cygwin. Make sure you have x86_64-w64-mingw32-gcc and x86_64-w64-mingw32-gfortran installed. Open the file Makefile.rule, and edit it so that
```
CC = x86_64-w64-mingw32-gcc
FC = x86_64-w64-mingw32-gfortran
USE_THREAD = 0
```

Then, run the commands:
```
make -j3
make PREFIX=../pytorch/torch/lib/tmp_install install
cd ../pytorch/torch/lib/tmp_install/lib
cp libopenblas.dll.a libopenblas.lib
```

Since OpenBLAS is compiled with gcc, we require some other libraries from Cygwin for everything to work. More specifically, it depends on
```
libgcc_s_seh-1.dll
libgfortran-3.dll
libquadmath-0.dll
libwinpthread-1.dll
```
You'll find them scattered in C:\cygwin64\

# Known Problems
CUDA is only partially supported.
