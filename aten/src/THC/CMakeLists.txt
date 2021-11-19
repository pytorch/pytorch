set(ATen_CUDA_INCLUDE ${ATen_CUDA_INCLUDE}
  "${CMAKE_CURRENT_SOURCE_DIR}"
PARENT_SCOPE)

set(ATen_CUDA_SRCS ${ATen_CUDA_SRCS}
  ${CMAKE_CURRENT_SOURCE_DIR}/THCStorage.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THCStorageCopy.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensor.cpp

  ${CMAKE_CURRENT_SOURCE_DIR}/THCSleep.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCStorage.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCStorageCopy.cu
  PARENT_SCOPE)

install(FILES
          THC.h
          THCGeneral.h
          THCGeneral.hpp
          THCSleep.h
          THCStorage.h
          THCStorageCopy.h
          THCTensor.h
          THCAtomics.cuh
          THCGenerateByteType.h
          # See Note [TH abstraction violation]
          THCTensor.hpp
          THCStorage.hpp
          DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THC")

install(FILES
          generic/THCStorage.cpp
          generic/THCStorage.cu
          generic/THCStorage.h
          generic/THCTensor.cpp
          generic/THCTensor.h
          generic/THCStorageCopy.cpp
          generic/THCStorageCopy.cu
          generic/THCStorageCopy.h
          DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THC/generic")
