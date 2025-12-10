if(__craylinux_crayprgenv_c)
  return()
endif()
set(__craylinux_crayprgenv_c 1)

include(Compiler/CrayPrgEnv)
__CrayPrgEnv_setup(C)
