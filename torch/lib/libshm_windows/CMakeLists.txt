IF(NOT LIBSHM_INSTALL_LIB_SUBDIR)
  SET(LIBSHM_INSTALL_BIN_SUBDIR "bin" CACHE PATH "libshm install binary directory")
  SET(LIBSHM_INSTALL_LIB_SUBDIR "lib" CACHE PATH "libshm install library directory")
ENDIF()

ADD_LIBRARY(shm SHARED core.cpp)

TARGET_COMPILE_DEFINITIONS(shm PRIVATE
  "_CRT_SECURE_NO_DEPRECATE=1"
  "SHM_EXPORTS"
)

target_include_directories(shm PRIVATE
  ${CMAKE_BINARY_DIR}/aten/src # provides "ATen/TypeExtendedInterface.h" to ATen.h
  ${TORCH_ROOT}/torch/lib # provides "libshm/libshm.h"
  ${CMAKE_BINARY_DIR}/caffe2/aten/src # provides <TH/THGeneral.h> to THC.h
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

TARGET_LINK_LIBRARIES(shm torch c10)


INSTALL(TARGETS shm DESTINATION "${LIBSHM_INSTALL_LIB_SUBDIR}")
INSTALL(FILES libshm.h DESTINATION "include")

if (MSVC AND BUILD_SHARED_LIBS)
  INSTALL(FILES $<TARGET_PDB_FILE:shm> DESTINATION "${LIBSHM_INSTALL_LIB_SUBDIR}" OPTIONAL)
endif()
