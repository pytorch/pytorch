# This file is loaded when gcc/g++ is used for assembler files (the "ASM" cmake language)
include(Compiler/GNU)

set(CMAKE_ASM_SOURCE_FILE_EXTENSIONS s;S;asm)

__compiler_gnu(ASM)

if(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_MATCH STREQUAL "GNU assembler")
  set(CMAKE_DEPFILE_FLAGS_ASM${ASM_DIALECT} "--MD <DEP_FILE>")
  set(CMAKE_ASM${ASM_DIALECT}_LINK_EXECUTABLE
    "<CMAKE_LINKER> <FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
  set(CMAKE_ASM_DEFINE_FLAG "--defsym ")
endif()
