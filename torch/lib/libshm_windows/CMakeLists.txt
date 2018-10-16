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
  ${CMAKE_SOURCE_DIR}/aten/src/TH # provides "THAllocator.h" to THCGeneral.h
  ${CMAKE_BINARY_DIR}/aten/src # provides "ATen/TypeExtendedInterface.h" to ATen.h
  ${CMAKE_BINARY_DIR}/caffe2/aten/src # provides <TH/THGeneral.h> to THC.h
  ${CMAKE_BINARY_DIR}/caffe2/aten/src/THC # provides "THCGeneral.h" to THC.h
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

TARGET_LINK_LIBRARIES(shm caffe2 c10)


INSTALL(TARGETS shm
          RUNTIME DESTINATION "${LIBSHM_INSTALL_BIN_SUBDIR}"
          LIBRARY DESTINATION "${LIBSHM_INSTALL_LIB_SUBDIR}"
          ARCHIVE DESTINATION "${LIBSHM_INSTALL_LIB_SUBDIR}")
INSTALL(FILES libshm.h DESTINATION "include")
