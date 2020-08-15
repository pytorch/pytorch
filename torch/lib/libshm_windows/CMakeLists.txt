if(NOT LIBSHM_INSTALL_LIB_SUBDIR)
  set(LIBSHM_INSTALL_BIN_SUBDIR "bin" CACHE PATH "libshm install binary directory")
  set(LIBSHM_INSTALL_LIB_SUBDIR "lib" CACHE PATH "libshm install library directory")
endif()

add_library(shm SHARED core.cpp)

target_compile_definitions(shm PRIVATE
  "_CRT_SECURE_NO_DEPRECATE=1"
  "SHM_EXPORTS"
)

target_include_directories(shm PRIVATE
  ${CMAKE_BINARY_DIR}/aten/src # provides "ATen/TypeExtendedInterface.h" to ATen.h
  ${TORCH_ROOT}/torch/lib # provides "libshm/libshm.h"
  ${CMAKE_BINARY_DIR}/caffe2/aten/src # provides <TH/THGeneral.h> to THC.h
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

target_link_libraries(shm torch c10)


install(TARGETS shm DESTINATION "${LIBSHM_INSTALL_LIB_SUBDIR}")
install(FILES libshm.h DESTINATION "include")

if(MSVC AND BUILD_SHARED_LIBS)
  install(FILES $<TARGET_PDB_FILE:shm> DESTINATION "${LIBSHM_INSTALL_LIB_SUBDIR}" OPTIONAL)
endif()
