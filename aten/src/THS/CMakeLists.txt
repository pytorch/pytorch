set(ATen_CPU_SRCS ${ATen_CPU_SRCS}
  ${CMAKE_CURRENT_SOURCE_DIR}/THSTensor.cpp
PARENT_SCOPE)

set(ATen_CPU_INCLUDE ${ATen_CPU_INCLUDE}
  "${CMAKE_CURRENT_BINARY_DIR}"
PARENT_SCOPE)

INSTALL(FILES
  THS.h
  THSGenerateAllTypes.h
  THSGenerateFloatTypes.h
  THSGenerateIntTypes.h
  THSTensor.h
  DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THS")

INSTALL(FILES
  generic/THSTensor.cpp
  generic/THSTensor.h
  generic/THSTensorMath.c
  generic/THSTensorMath.h
  DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THS/generic")
