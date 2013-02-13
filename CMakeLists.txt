SET(src DiskFile.c File.c MemoryFile.c PipeFile.c Storage.c Tensor.c Timer.c utils.c init.c TensorOperator.c TensorMath.c random.c)
SET(luasrc init.lua File.lua Tensor.lua CmdLine.lua Tester.lua test/test.lua)

# Necessary do generate wrapper
ADD_TORCH_WRAP(tensormathwrap TensorMath.lua)
ADD_TORCH_WRAP(randomwrap random.lua)

ADD_TORCH_PACKAGE(torch "${src}" "${luasrc}")
ADD_TORCH_DOK(dok torch "Fundamentals" "Torch package" 1.1)

TARGET_LINK_LIBRARIES(torch luaT TH)

INSTALL(FILES generic/Storage.c generic/Tensor.c
  DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/torch/generic")
