CMAKE_<LANG>_COMPILER_ID
------------------------

Compiler identification string.

A short string unique to the compiler vendor.  Possible values
include:

=============================== ===============================================
Value                           Name
=============================== ===============================================
``Absoft``                      Absoft Fortran
``ADSP``                        Analog VisualDSP++
``AppleClang``                  Apple Clang
``ARMCC``                       ARM Compiler
``ARMClang``                    ARM Compiler based on Clang
``Bruce``                       Bruce C Compiler
``CCur``                        Concurrent Fortran
``Clang``                       `LLVM Clang`_
``Cray``                        Cray Compiler
``CrayClang``                   Cray Clang-based Compiler
``Diab``                        `Wind River Systems Diab Compiler`_
``Embarcadero``, ``Borland``    `Embarcadero`_
``Flang``                       `Classic Flang Fortran Compiler`_
``LLVMFlang``                   `LLVM Flang Fortran Compiler`_
``Fujitsu``                     Fujitsu HPC compiler (Trad mode)
``FujitsuClang``                Fujitsu HPC compiler (Clang mode)
``G95``                         `G95 Fortran`_
``GNU``                         `GNU Compiler Collection`_
``GHS``                         `Green Hills Software`_
``HP``                          Hewlett-Packard Compiler
``IAR``                         IAR Systems
``Intel``                       Intel Classic Compiler
``IntelLLVM``                   `Intel LLVM-Based Compiler`_
``LCC``                         MCST Elbrus C/C++/Fortran Compiler
``LFortran``                    LFortran Fortran Compiler
``MSVC``                        `Microsoft Visual Studio`_
``NVHPC``                       `NVIDIA HPC Compiler`_
``NVIDIA``                      `NVIDIA CUDA Compiler`_
``OrangeC``                     `OrangeC Compiler`_
``OpenWatcom``                  `Open Watcom`_
``PGI``                         The Portland Group
``PathScale``                   PathScale
``QCC``                         QNX C/C++ compiler
``Renesas``                     `Renesas Compiler`_
``SCO``                         SCO OpenServer/UnixWare C/C++ Compiler
``SDCC``                        `Small Device C Compiler`_
``SunPro``                      Oracle Developer Studio
``Tasking``                     `Tasking Compiler Toolsets`_
``TI``                          Texas Instruments
``TIClang``                     `Texas Instruments Clang-based Compilers`_
``TinyCC``                      `Tiny C Compiler`_
``XL``, ``VisualAge``, ``zOS``  IBM XL
``XLClang``                     IBM Clang-based XL
``IBMClang``                    IBM LLVM-based Compiler
=============================== ===============================================

This variable is not guaranteed to be defined for all compilers or
languages.

.. _LLVM Clang: https://clang.llvm.org
.. _Embarcadero: https://www.embarcadero.com
.. _Classic Flang Fortran Compiler: https://github.com/flang-compiler/flang
.. _LLVM Flang Fortran Compiler: https://github.com/llvm/llvm-project/tree/main/flang
.. _G95 Fortran: https://g95.sourceforge.net
.. _GNU Compiler Collection: https://gcc.gnu.org
.. _Green Hills Software: https://www.ghs.com/products/compiler.html
.. _Intel LLVM-Based Compiler:  https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html
.. _Microsoft Visual Studio: https://visualstudio.microsoft.com
.. _NVIDIA HPC Compiler: https://developer.nvidia.com/hpc-compilers
.. _NVIDIA CUDA Compiler: https://developer.nvidia.com/cuda-llvm-compiler
.. _Open Watcom: https://open-watcom.github.io
.. _OrangeC Compiler: https://github.com/LADSoft/OrangeC
.. _Renesas Compiler: https://www.renesas.com
.. _Small Device C Compiler: https://sdcc.sourceforge.net
.. _Tiny C Compiler: https://bellard.org/tcc
.. _Tasking Compiler Toolsets: https://www.tasking.com
.. _Texas Instruments Clang-based Compilers: https://www.ti.com/tool/download/ARM-CGT-CLANG
.. _Wind River Systems Diab Compiler: https://www.windriver.com/resource/wind-river-diab-compiler-product-overview
