set(ATen_CUDA_INCLUDE ${ATen_CUDA_INCLUDE}
  "${CMAKE_CURRENT_SOURCE_DIR}"
PARENT_SCOPE)

set(ATen_CUDA_SRCS ${ATen_CUDA_SRCS}
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensor.cpp

  ${CMAKE_CURRENT_SOURCE_DIR}/THCSleep.cu
  PARENT_SCOPE)

install(FILES
          THC.h
          THCGeneral.h
          THCGeneral.hpp
          THCSleep.h
          THCTensor.h
          THCAtomics.cuh
          THCGenerateByteType.h
          # See Note [TH abstraction violation]
          THCTensor.hpp
          DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THC")

install(FILES
          generic/THCTensor.cpp
          generic/THCTensor.h
          DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THC/generic")
