if(__craylinux_crayprgenv_fortran)
  return()
endif()
set(__craylinux_crayprgenv_fortran 1)

include(Compiler/CrayPrgEnv)
__CrayPrgEnv_setup(Fortran)
